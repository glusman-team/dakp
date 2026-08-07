"""Contraindication assertion aggregation — text-mined from DailyMed SPL.

Builds ``contraindication_assertions.tsv``: drug→condition contraindication assertions
**mined directly** from DailyMed SPL label text using DAKP's single composite NER backend
(:mod:`dakp_pipeline.ner.ner`). Contraindications are extracted from the label text itself,
which yields better coverage and DailyMed-grounded provenance than an externally-sourced list.

Two-pass mining rule (explicit and tested)
------------------------------------------
**Pass 1** — for every SPL set with a contraindication section (LOINC ``34070-3``) and ≥1
active ingredient, run the NER backend over the full section text to extract disease/phenotype
mentions. All text in a dedicated contraindication section is assumed relevant.

**Pass 2** — for every SPL set with an indications-and-usage section (LOINC ``34067-9``), filter
the section text to sentences containing contraindication-context keywords (``contraindicated``,
``should not be used``, ``not recommended``, …) and run the same NER extraction on the filtered
text. This catches contraindications embedded in the indication section — common in real
DailyMed labels where safety statements appear alongside indication text. The keyword filter
ensures indication-context diseases (``indicated for X``) are NOT mined as contraindications.

Mentions from both passes are paired with each active ingredient of the set to form a
``biolink:contraindicated_in`` assertion (ingredient = subject, mention = object). Rows are
aggregated by ``(subject_text, object_text)``: ``supporting_spl_sets`` and
``supporting_spl_documents`` are unioned, ``source_score`` takes the max NER span score.

Ontology mapping is Tablassert-only
-----------------------------------
The **object** is the mined disease MENTION TEXT (``object_text``); ``object_curie`` /
``object_name`` / ``object_category`` are left empty for Tablassert/fullmap to resolve at
``tablassert build-kg`` — DAKP does **not** resolve mentions to ontology CURIEs. The
**subject** (active ingredient) carries its text + UNII straight from the SPL source
(source-provided, not DAKP-mapped).

The NER backend
---------------
The shaper uses an injected ``params["ner"]`` :class:`~dakp_pipeline.ner.ner.DiseaseNER` when
present (tests / production wiring), else builds the deterministic **offline** backend from the
ontology fixture gazetteer (``<fixture_root>/ontology/disease_map.tsv``, read as term→type only
— CURIE columns ignored), falling back to the embedded gazetteer. There is no backend-name
selector. Constructing the backend is import-free, so module import + the test suite run with
no heavy NER deps imported.

Provenance: contraindications are text-mined from DailyMed, so
``primary_knowledge_source = infores:multiomics-drugapprovals``,
``upstream_resource_ids = infores:dailymed``, ``agent_type = text_mining_agent`` (the DAKP RIG
lists ``text_mining_agent`` for ``contraindicated_in``), ``knowledge_level = knowledge_assertion``.
"""

from __future__ import annotations

import importlib.machinery
import multiprocessing as mp
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dakp_pipeline.assertions import INFORES_DAILYMED, INFORES_DAKP, KL_ASSERTION, row_for
from dakp_pipeline.assertions.evidence import build_dailymed_evidence, sorted_pipe, write_assertion_table
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import logger, progress, stats, step
from dakp_pipeline.ner.dictionary import normalize_text
from dakp_pipeline.ner.ner import DiseaseNER, Mention, extract_contraindication_diseases

_TABLE = "contraindication_assertions"
_PREDICATE = "biolink:contraindicated_in"
_ONTOLOGY_FIXTURE = Path("ontology") / "disease_map.tsv"
#: One INFO progress line per this many mined contraindication sections (GLiNER is the slow step).
_MINING_PROGRESS_EVERY = 500

#: Agent type for text-mined contraindications (matches the DAKP RIG ``contraindicated_in``).
AT_TEXT_MINING = "text_mining_agent"

#: Regex matching contraindication-context sentence keywords. When Pass 2 filters indication
#: section text, only sentences matching this pattern are sent to the NER backend — this
#: prevents indication-context diseases ("indicated for X") from being mined as contraindications
#: while still catching embedded contraindication statements ("contraindicated in patients with Y").
#: Tuned empirically against ``urchade/gliner_small-v2.1`` on real DailyMed label patterns.
#: Configurable at runtime via ``ctx.params["contraindication_keywords"]`` (str or compiled Pattern).
DEFAULT_CONTRA_KEYWORDS: re.Pattern[str] = re.compile(
    r"\b(contraindicat\w*|should\s+not\s+be\s+used|must\s+not\s+(?:be\s+)?used|"
    r"do\s+not\s+use|not\s+recommended|avoid\s+(?:use\s+)?in|prohibit\w*|"
    r"must\s+avoid|use\s+is\s+contraindicat\w*|not\s+for\s+use\s+in)\b",
    re.IGNORECASE,
)

#: The 4x Tesla P100-PCIE-16GB GPUs on the DAKP build host (wenceslaus). Hardcoded - not
#: auto-detected — so the shaper always dispatches across all four when CUDA is available.
#: When CUDA is absent (CI, tests, non-GPU hosts) the shaper falls back to sequential.
CONTRAINDICATION_GPUS: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")


# --- sentence filtering for Pass 2 (indication-section contraindications) ------------


_SENTENCE_BOUNDARY = re.compile(r"[.;]\s+")


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentence-ish pieces at period/semicolon + whitespace boundaries.

    A simple, deterministic splitter sufficient for DailyMed label prose. Returns stripped
    non-empty sentences in text order. The last piece (after the final boundary) is always
    included so no text is lost.
    """
    sentences: list[str] = []
    pos = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        piece = text[pos : match.end()].strip()
        if piece:  # pragma: no branch - a boundary match always yields its punctuation char
            sentences.append(piece)
        pos = match.end()
    if pos < len(text):
        tail = text[pos:].strip()
        if tail:  # pragma: no branch - greedy boundary whitespace leaves a non-blank tail
            sentences.append(tail)
    return sentences


def _contraindication_sentences(text: str, keywords: re.Pattern[str]) -> str:
    """Return only the contraindication-context sentences from ``text``, space-joined.

    Splits ``text`` into sentences and keeps only those matching ``keywords``. The result is
    suitable for direct NER extraction — GLiNER sees only contraindication-relevant text so
    indication-context diseases are never mined. Returns ``""`` when no sentence matches.
    """
    filtered = [s for s in _split_sentences(text) if keywords.search(s)]
    return " ".join(filtered)


def _resolve_keywords(ctx: TaskContext) -> re.Pattern[str]:
    """The contraindication keyword pattern: ``ctx.params["contraindication_keywords"]`` or default.

    Accepts a raw string (compiled case-insensitively) or a pre-compiled ``re.Pattern``.
    Returns :data:`DEFAULT_CONTRA_KEYWORDS` when the param is absent.
    """
    raw = ctx.params.get("contraindication_keywords")
    if raw is None:
        return DEFAULT_CONTRA_KEYWORDS
    if isinstance(raw, re.Pattern):
        return raw
    return re.compile(raw, re.IGNORECASE)


def default_ner(fixture_root: Path | str | None) -> DiseaseNER:
    """The deterministic offline NER backend: gazetteer from the ontology fixture, else embedded.

    Reads the ontology fixture as a term→type gazetteer ONLY (``text`` + ``category`` columns;
    CURIE/name columns are ignored — DAKP does not map terms to ontology concepts). No heavy
    NER dep is imported.
    """
    if fixture_root is not None:
        ontology = Path(fixture_root) / _ONTOLOGY_FIXTURE
        if ontology.exists():
            return DiseaseNER.from_tsv(ontology, text_col="text", type_col="category")
    return DiseaseNER()


class ContraindicationsShaper:
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        with step(logger, "shape_contraindications"):
            ner_param = ctx.params.get("ner")
            ner = ner_param if isinstance(ner_param, DiseaseNER) else default_ner(ctx.fixture_root)
            devices = _resolve_devices(ner)
            keywords = _resolve_keywords(ctx)
            if devices:
                stats(logger, "shape_contraindications", dispatch_gpus=len(devices))
            rows = build_contraindication_rows(inputs, ner, devices=devices, keywords=keywords)
            return write_assertion_table(_TABLE, rows, inputs, ctx, operation="shape_contraindications")


def build_contraindication_rows(
    inputs: Iterable[ArtifactRef], ner: DiseaseNER, *, devices: Sequence[str] | None = None, keywords: re.Pattern[str] | None = None
) -> list[dict[str, str]]:
    """Mine DailyMed contraindication + indication sections into assertion rows (deterministic).

    **Pass 1:** For each SPL set with a contraindication section (LOINC ``34070-3``) and ≥1
    active ingredient, extract disease/phenotype mentions from the full section text.

    **Pass 2:** For each SPL set with an indication section (LOINC ``34067-9``), filter the text
    to contraindication-context sentences (via ``keywords`` or :data:`DEFAULT_CONTRA_KEYWORDS`)
    and extract mentions from the filtered text only. This catches contraindications embedded
    in the indication section while avoiding false positives from indication-context diseases.

    Mentions from both passes are paired with active ingredients and aggregated by
    ``(subject_text, object_text)``. When ``devices`` is provided (production multi-GPU), each
    pass is dispatched across half the GPUs concurrently (2+2 split). Output is byte-identical
    regardless of dispatch mode.
    """
    evidence = build_dailymed_evidence(inputs)
    kw = keywords or DEFAULT_CONTRA_KEYWORDS

    # Pass 1 work items: contraindication sections (all text is relevant).
    work_items_p1: list[tuple[str, str, str]] = []
    for set_id in sorted(evidence.contraindication_docs):
        if not evidence.active_ingredients_by_set.get(set_id):
            continue
        for doc_id, text in evidence.contraindication_docs[set_id]:
            work_items_p1.append((set_id, doc_id, text))

    # Pass 2 work items: indication sections filtered to contraindication-context sentences.
    work_items_p2: list[tuple[str, str, str]] = []
    for set_id in sorted(evidence.indication_docs):
        if not evidence.active_ingredients_by_set.get(set_id):
            continue
        for doc_id, text in evidence.indication_docs[set_id]:
            filtered = _contraindication_sentences(text, kw)
            if filtered.strip():
                work_items_p2.append((set_id, doc_id, filtered))

    all_work_items = work_items_p1 + work_items_p2
    stats(
        logger,
        "shape_contraindications",
        contraindication_sets=len(evidence.contraindication_docs),
        indication_sets=len(evidence.indication_docs),
        pass1_sections=len(work_items_p1),
        pass2_sections=len(work_items_p2),
        sections_to_mine=len(all_work_items),
    )

    # Extract mentions: two-pass multi-GPU when devices given + production NER + >1 item;
    # else sequential (with periodic progress narration — GLiNER mining is the slow step).
    if devices and len(all_work_items) > 1 and not ner._offline:
        if work_items_p2 and len(devices) >= 2:
            mined = _mine_two_passes_multi_gpu(work_items_p1, work_items_p2, ner, devices)
        else:
            # No Pass 2 work, or too few devices to split — all GPUs on combined work.
            mined = _mine_multi_gpu(all_work_items, ner, devices)
    else:
        mined = {}
        for done, (set_id, doc_id, text) in enumerate(all_work_items, start=1):
            mined[(set_id, doc_id)] = extract_contraindication_diseases(text, ner)
            progress(logger, "shape_contraindications", done, len(all_work_items), every=_MINING_PROGRESS_EVERY)

    # Aggregate mentions into assertion rows (unchanged logic).
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    mentions_mined = 0
    for set_id, doc_id, _text in all_work_items:
        ingredients = evidence.active_ingredients_by_set.get(set_id, [])
        for mention in mined.get((set_id, doc_id), []):
            mentions_mined += 1
            # Canonicalize the mined mention (lowercase / strip punctuation) so case variants
            # (asthma / Asthma / ASTHMA) aggregate to one object instead of fragmenting the rows.
            object_text = normalize_text(mention.text)
            if not object_text:
                continue
            for ingredient_name, ingredient_unii in ingredients:
                _accumulate(aggregated, set_id, doc_id, ingredient_name, ingredient_unii, object_text, mention)

    stats(logger, "shape_contraindications", mentions_mined=mentions_mined, assertions=len(aggregated))
    return [_finalize_row(agg) for _key, agg in sorted(aggregated.items())]


#: Module spawned workers re-import instead of the parent's ``__main__`` script (see
#: :func:`_spawn_safe_main`). Must import with zero side effects. A plain MODULE, not the
#: package itself — ``runpy.run_module`` cannot directly execute a package without a
#: ``__main__.py``.
_SPAWN_SAFE_MAIN_MODULE = "dakp_pipeline.logging_setup"


@contextmanager
def _spawn_safe_main() -> Iterator[None]:
    """Keep spawn children from re-executing the parent's ``__main__`` script.

    The spawn start method re-initializes ``__main__`` in every child: when
    ``__main__.__spec__`` exists it imports that module by name, otherwise it RE-EXECUTES
    ``__main__.__file__``. Under the Airflow task runtime ``__main__`` is the ``airflow`` CLI
    script (no ``__spec__``), so each mining worker re-executed the CLI and died initializing
    Airflow's DB settings there (no parseable ``sql_alchemy_conn`` in the child) — the pool
    broke before its first task (``BrokenProcessPool``). For the pool's lifetime, point spawn
    at a side-effect-free module; the worker callable itself is unpickled via its own
    module, and the original ``__main__`` spec is restored on exit.
    """
    main = sys.modules["__main__"]
    if getattr(main, "__spec__", None) is not None:
        yield  # module context (python -m / console scripts with a spec): spawn imports by name
        return
    main.__spec__ = importlib.machinery.ModuleSpec(_SPAWN_SAFE_MAIN_MODULE, loader=None)
    try:
        yield
    finally:
        main.__spec__ = None


# --- multi-GPU dispatch ---------------------------------------------------------


def _resolve_devices(ner: DiseaseNER) -> Sequence[str] | None:
    """The hardcoded GPU list capped to the VISIBLE device count; None when unusable.

    Only the production (GLiNER) backend benefits from multi-GPU dispatch — the offline
    gazetteer is CPU-only and deterministic. ``torch.cuda.is_available()`` guards against
    CI / test hosts with no CUDA (the lazy import never fires at module load). The list is
    capped at ``torch.cuda.device_count()`` because hosts vary: the build server has the full
    4x P100 set, laptops often expose a single GPU — dispatching a worker to a nonexistent
    ``cuda:N`` crashes the whole pool (torch refuses to deserialize onto a missing device).
    """
    if ner._offline:
        return None
    try:
        import torch  # lazy: no torch at module load
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    visible = torch.cuda.device_count()
    if visible <= 0:
        return None
    return CONTRAINDICATION_GPUS[:visible]


def _shard_by_text_length(items: list[tuple[str, str, str]], n: int) -> list[list[tuple[str, str, str]]]:
    """Distribute work items across ``n`` shards, balanced by text length (LPT scheduling).

    Sorts items by text length descending, then greedily assigns each to the shard with the
    least total text so far. Returns exactly ``n`` shards (some may be empty when ``n > len(items)``).
    """
    shards: list[list[tuple[str, str, str]]] = [[] for _ in range(n)]
    loads: list[int] = [0] * n
    for item in sorted(items, key=lambda x: len(x[2]), reverse=True):
        idx = loads.index(min(loads))
        shards[idx].append(item)
        loads[idx] += len(item[2])
    return shards


def _mine_shard(shard: list[tuple[str, str, str]], ner_config: dict[str, Any], device: str) -> list[tuple[str, str, list[Mention]]]:
    """ProcessPoolExecutor worker: load GLiNER on ``device``, mine each text, return mentions.

    Reconstructs a :class:`DiseaseNER` from the picklable ``ner_config`` pinned to ``device``,
    then runs extraction over every ``(set_id, doc_id, text)`` item in its shard. The model
    loads lazily on the first extract call, so each worker initializes its own CUDA context
    (safe under the ``spawn`` start method).
    """
    ner = DiseaseNER(device=device, **ner_config)
    return [(set_id, doc_id, ner.extract(text)) for set_id, doc_id, text in shard]


def _mine_multi_gpu(work_items: list[tuple[str, str, str]], ner: DiseaseNER, devices: Sequence[str]) -> dict[tuple[str, str], list[Mention]]:
    """Dispatch NER extraction across one worker per GPU and collect results.

    Shards ``work_items`` across ``len(devices)`` groups (LPT-balanced by text length), spawns
    one process per device via :class:`~concurrent.futures.ProcessPoolExecutor` (``spawn``
    start method — CUDA + ``fork`` is unsafe), and returns a ``{(set_id, doc_id): [mentions]}``
    map. The model cache on disk is shared read-only across workers.
    """
    n_workers = min(len(devices), len(work_items))
    shards = _shard_by_text_length(work_items, n_workers)
    ner_config = ner._config()
    ctx = mp.get_context("spawn")
    results: dict[tuple[str, str], list[Mention]] = {}
    with _spawn_safe_main(), ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = [pool.submit(_mine_shard, shard, ner_config, devices[i]) for i, shard in enumerate(shards)]
        for future in futures:
            for set_id, doc_id, mentions in future.result():
                results[(set_id, doc_id)] = mentions
    return results


def _mine_two_passes_multi_gpu(
    work_items_p1: list[tuple[str, str, str]], work_items_p2: list[tuple[str, str, str]], ner: DiseaseNER, devices: Sequence[str]
) -> dict[tuple[str, str], list[Mention]]:
    """Dispatch both extraction passes concurrently, splitting GPUs between them.

    Splits the device list in half: the first half mines Pass 1 (contraindication sections),
    the second half mines Pass 2 (filtered indication sections). Both passes are dispatched
    as futures in a single :class:`~concurrent.futures.ProcessPoolExecutor` so they run in
    parallel. When either pass has no work items, all GPUs fall back to the other pass via
    :func:`_mine_multi_gpu`. Results from both passes are merged into one
    ``{(set_id, doc_id): [mentions]}`` map.
    """
    if not work_items_p2:
        return _mine_multi_gpu(work_items_p1, ner, devices)
    if not work_items_p1:
        return _mine_multi_gpu(work_items_p2, ner, devices)

    mid = max(1, len(devices) // 2)
    devices_p1 = list(devices[:mid])
    devices_p2 = list(devices[mid:]) or list(devices[:1])  # defensive: never empty

    n_p1 = min(len(devices_p1), len(work_items_p1))
    n_p2 = min(len(devices_p2), len(work_items_p2))
    shards_p1 = _shard_by_text_length(work_items_p1, n_p1)
    shards_p2 = _shard_by_text_length(work_items_p2, n_p2)
    ner_config = ner._config()
    ctx = mp.get_context("spawn")

    results: dict[tuple[str, str], list[Mention]] = {}
    with _spawn_safe_main(), ProcessPoolExecutor(max_workers=n_p1 + n_p2, mp_context=ctx) as pool:
        futures = [
            *(pool.submit(_mine_shard, shard, ner_config, devices_p1[i]) for i, shard in enumerate(shards_p1)),
            *(pool.submit(_mine_shard, shard, ner_config, devices_p2[i]) for i, shard in enumerate(shards_p2)),
        ]
        for future in futures:
            for set_id, doc_id, mentions in future.result():
                results[(set_id, doc_id)] = mentions
    return results


def _accumulate(
    aggregated: dict[tuple[str, str], dict[str, Any]],
    set_id: str,
    doc_id: str,
    ingredient_name: str,
    ingredient_unii: str,
    object_text: str,
    mention: Mention,
) -> None:
    """Add one (ingredient, mention) observation to the ``(subject, object)`` aggregate.

    The subject carries the SPL-provided ingredient text + UNII; the object is the mined mention
    text with CURIE/name/category left empty for Tablassert/fullmap to resolve.
    """
    key = (ingredient_name, object_text)
    agg = aggregated.setdefault(
        key,
        {
            "subject_text": ingredient_name,
            "subject_curie": ingredient_unii,
            "subject_name": ingredient_name,
            "object_text": object_text,
            "object_curie": "",
            "object_name": "",
            "object_category": "",
            "sets": [],
            "docs": [],
            "scores": [],
        },
    )
    agg["sets"].append(set_id)
    agg["docs"].append(doc_id)
    agg["scores"].append(mention.score)


def _finalize_row(agg: dict[str, Any]) -> dict[str, str]:
    return row_for(
        _TABLE,
        subject_text=agg["subject_text"],
        subject_curie=agg["subject_curie"],
        subject_name=agg["subject_name"],
        subject_category="ChemicalEntity",
        predicate=_PREDICATE,
        object_text=agg["object_text"],
        object_curie=agg["object_curie"],
        object_name=agg["object_name"],
        object_category=agg["object_category"],
        supporting_spl_sets=sorted_pipe(agg["sets"]),
        supporting_spl_documents=sorted_pipe(agg["docs"]),
        source_score=_max_score(agg["scores"]),
        knowledge_level=KL_ASSERTION,
        agent_type=AT_TEXT_MINING,
        primary_knowledge_source=INFORES_DAKP,
        upstream_resource_ids=INFORES_DAILYMED,
    )


def _max_score(scores: list[float]) -> str:
    """Highest NER span score as a deterministic string ("" if none)."""
    if not scores:
        return ""
    return f"{max(scores):g}"


transform = ContraindicationsShaper().transform

__all__ = [
    "AT_TEXT_MINING",
    "CONTRAINDICATION_GPUS",
    "DEFAULT_CONTRA_KEYWORDS",
    "ContraindicationsShaper",
    "build_contraindication_rows",
    "default_ner",
    "transform",
]
