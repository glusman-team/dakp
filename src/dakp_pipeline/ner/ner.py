"""The single composite disease/phenotype NER backend — ONE backend, ONE entry point.

DAKP extracts **mentions only** (text spans + entity type). It never resolves terms to
ontology CURIEs — that is exclusively Tablassert's job (fullmap/BABEL at ``tablassert
build-kg``). There is no pluggable backend selector: this module is the one NER backend,
settled after benchmarking (see ``ner/BENCHMARK.md``).

Composite design (gazetteer-first, GLiNER-augmented)
----------------------------------------------------
* **Offline mode (default):** a curated disease/phenotype :class:`~dakp_pipeline.ner.dictionary.Gazetteer`
  + deterministic :class:`~dakp_pipeline.ner.lexical.LexicalMatcher`. Precision 1.000 / F1
  0.955 on the benchmark fixture, zero heavy dependencies, fully deterministic. Used by tests
  and offline runs.
* **Production mode (``offline=False``):** the same gazetteer anchors high-precision spans and
  GLiNER zero-shot (``gliner-community/gliner_large-v2.5``) fills out-of-gazetteer gaps.
  Non-overlapping GLiNER spans add recall; on overlap the **most specific span wins** — a model
  span that strictly contains a gazetteer span supersedes it (``pulmonary hypertension`` beats
  ``hypertension``), taking the model's boundary and the gazetteer's type. Equal spans, partial
  overlaps and spans covering several gazetteer terms (a conjunction) all go to the gazetteer.
  Model spans whose normalized surface is a population descriptor (``_POPULATION_PHRASES``, e.g.
  "women of childbearing potential") are dropped, leading hedge tokens (``recent``, ``a history
  of``) are trimmed, and spans a hard window split cuts across a phrase boundary are re-joined.
  Candidates are generated at ``DEFAULT_THRESHOLD`` and accepted at ``DEFAULT_ACCEPT_THRESHOLD``
  — the same 0.35 floor by default (the lowest score at which GLiNER is still accurate), so
  nothing generated is abstained; raise ``accept_threshold`` to decide narrower than you
  generate. Below the floor the backend **abstains** rather than asserting a low-confidence
  mention or falling back to a less specific one. GLiNER is natively
  multi-entity: one ``predict_entities`` call scores every requested label (disease + phenotype
  here) and returns any number of spans per label. GLiNER silently truncates inputs past
  ``config.max_len`` word tokens (768 on the shipped v2.5 checkpoint), so long sections are
  predicted in exact-substring windows (:func:`_windows`) whose spans are remapped back into
  full-text offsets before the merge. ``gliner`` is a core
  DAKP dependency but is imported lazily on first use (no torch at module load), raising
  :class:`~dakp_pipeline.ner.model_cache.NERDependencyError` ("reinstall with `uv sync`") if it is
  somehow not importable.

The offline/production toggle is a mode on this ONE backend, not a backend-name enum.

Entry points: :func:`extract_disease_mentions` (general) and :func:`extract_contraindication_diseases`
(contraindication sections). All offsets are half-open: ``mention.text == text[mention.start:mention.end]``.
Output is sorted deterministically by ``(start, end, type, text)``.
"""

from __future__ import annotations

import fcntl
import itertools
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dakp_pipeline.logging_setup import logger, stats
from dakp_pipeline.ner.dictionary import CONTRAINDICATION_DISEASE_TYPES, TYPE_DISEASE, TYPE_PHENOTYPE, Gazetteer, canonical_type, normalize_text
from dakp_pipeline.ner.lexical import LexicalMatcher, Mention
from dakp_pipeline.ner.model_cache import NERDependencyError, default_model_cache_dir, ensure_model

# GLiNER v2.5 large (deberta-v3-large encoder, max_len 768 word tokens, multi-entity: up to
# ``max_types`` labels per call). Override for a smaller / biomedical-tuned checkpoint.
DEFAULT_MODEL = "gliner-community/gliner_large-v2.5"
#: Candidate-**generation** threshold, passed straight to ``predict_entities``. 0.35 is the
#: lowest score at which GLiNER is still accurate, so generation never goes below it. Keeping it
#: at (not above) the acceptance floor matters: a specific span often scores lower than its
#: generic head (``drug hypersensitivity`` 0.35 vs ``hypersensitivity``), and generating at the
#: old 0.5 hid exactly the spans the specificity merge exists to prefer.
DEFAULT_THRESHOLD = 0.35
#: DAKP-side **acceptance** floor — the confidence a model span must reach to be emitted at all.
#: Spans in the ``[DEFAULT_THRESHOLD, DEFAULT_ACCEPT_THRESHOLD)`` band are visible to the merge
#: (so they can still win a boundary contest) but are abstained on rather than asserted.
#: Default: the floor sits at the generation threshold — spans at the model's accuracy floor are
#: still correct, so nothing generated is abstained. Raise ``accept_threshold`` to tighten
#: deliberately; do not lower ``threshold`` below 0.35 (the model's accuracy floor).
DEFAULT_ACCEPT_THRESHOLD = 0.35

# GLiNER counts input in word tokens from its whitespace splitter and silently truncates anything
# past ``config.max_len`` tokens (only a UserWarning). Mirror that exact token pattern (gliner's
# ``WhitespaceTokenSplitter``: every punctuation glyph is its own token) so windows never exceed
# the model's budget.
_GLINER_TOKEN = re.compile(r"\w+(?:[-_]\w+)*|\S")

# Sentence-ish piece for window packing: a run of non-terminal characters, trailing terminal
# punctuation, trailing whitespace. Matches tile the text when the tiling check in
# :func:`_sentence_piece_spans` holds; otherwise the whole text is one piece.
_SENTENCE_PIECE = re.compile(r"[^.!?;]+[.!?;]*\s*")

#: Window-budget fallback (GLiNER word tokens) when a model exposes no ``config.max_len``; the
#: shipped ``gliner-community/gliner_large-v2.5`` checkpoint sets ``max_len: 768``.
_DEFAULT_WORD_BUDGET = 384

# Curated high-precision disease/phenotype gazetteer — the offline mode's embedded vocabulary
# (the same terms benchmarked in ner/BENCHMARK.md). Not exhaustive by design: production mode
# adds GLiNER recall for out-of-gazetteer mentions. term -> canonical type.
EMBEDDED_GAZETTEER: dict[str, str] = {
    # disease (named disorders / syndromes)
    "asthma": TYPE_DISEASE,
    "liver disease": TYPE_DISEASE,
    "active liver disease": TYPE_DISEASE,
    "heart failure": TYPE_DISEASE,
    "severe heart failure": TYPE_DISEASE,
    "myocardial infarction": TYPE_DISEASE,
    "peptic ulcer": TYPE_DISEASE,
    "peptic ulcer disease": TYPE_DISEASE,
    "renal impairment": TYPE_DISEASE,
    "severe renal impairment": TYPE_DISEASE,
    "renal disease": TYPE_DISEASE,
    "end stage renal disease": TYPE_DISEASE,
    "hypertension": TYPE_DISEASE,
    "uncontrolled hypertension": TYPE_DISEASE,
    "arrhythmia": TYPE_DISEASE,
    "ventricular arrhythmias": TYPE_DISEASE,
    "stroke": TYPE_DISEASE,
    "hemorrhagic stroke": TYPE_DISEASE,
    "hepatic impairment": TYPE_DISEASE,
    "severe hepatic impairment": TYPE_DISEASE,
    "epilepsy": TYPE_DISEASE,
    "glaucoma": TYPE_DISEASE,
    "narrow angle glaucoma": TYPE_DISEASE,
    "hypercholesterolemia": TYPE_DISEASE,
    "diabetes mellitus": TYPE_DISEASE,
    "type 2 diabetes mellitus": TYPE_DISEASE,
    "chronic obstructive pulmonary disease": TYPE_DISEASE,
    "atrial fibrillation": TYPE_DISEASE,
    "arthritis": TYPE_DISEASE,
    "rheumatoid arthritis": TYPE_DISEASE,
    "depression": TYPE_DISEASE,
    "migraine": TYPE_DISEASE,
    # High-frequency contraindication diseases observed in hand-checked label/FAERS snippets
    # (see ner/BENCHMARK.md "Small-example checks"): gazetteer growth keeps the maximal-span
    # policy ("congestive heart failure", not just "heart failure") and speeds offline recall.
    "congestive heart failure": TYPE_DISEASE,
    "cirrhosis": TYPE_DISEASE,
    "hepatic encephalopathy": TYPE_DISEASE,
    "acute kidney injury": TYPE_DISEASE,
    "hepatocellular carcinoma": TYPE_DISEASE,
    # phenotype (clinical findings / symptoms / condition-states)
    "hypersensitivity": TYPE_PHENOTYPE,
    "transaminase elevations": TYPE_PHENOTYPE,
    "bleeding": TYPE_PHENOTYPE,
    "active bleeding": TYPE_PHENOTYPE,
    "gastrointestinal bleeding": TYPE_PHENOTYPE,
    "pregnancy": TYPE_PHENOTYPE,
    "qt prolongation": TYPE_PHENOTYPE,
    "seizure": TYPE_PHENOTYPE,
    "seizures": TYPE_PHENOTYPE,
    "headache": TYPE_PHENOTYPE,
    "pain": TYPE_PHENOTYPE,
    "back pain": TYPE_PHENOTYPE,
    "nausea": TYPE_PHENOTYPE,
    "vomiting": TYPE_PHENOTYPE,
    "fatigue": TYPE_PHENOTYPE,
}

# Population/demographic descriptors GLiNER likes to tag as phenotypes in contraindication text
# ("Contraindicated in women of childbearing potential." — observed false positive, score ~0.5-0.6).
# These are subject populations, not disease/phenotype mentions, so model spans whose normalized
# surface equals one are dropped. The curated gazetteer never matches them, so offline mode is
# unaffected. Normalized exact match only — deterministic and precision-safe.
_POPULATION_PHRASES: frozenset[str] = frozenset(
    (
        "women",
        "men",
        "children",
        "patients",
        "individuals",
        "subjects",
        "women of childbearing potential",
        "childbearing potential",
        "women of childbearing age",
        "pregnant women",
    )
)

# Non-clinical tokens trimmed off the LEFT edge of a model span before the specificity merge.
# This is what makes "the more specific span always wins" safe: it removes the modifier
# over-extension (``recent myocardial infarction``, ``a history of peptic ulcer disease``) that
# ``ner/BENCHMARK.md`` cites as the reason overlap-extension was originally skipped, while leaving
# real qualifiers alone. The list is a CLOSED class (determiners, prepositions, temporal/
# evidential hedges, population heads) and the polarity matters: anything NOT listed counts as a
# clinical qualifier and is kept, so ``severe`` / ``active`` / ``congestive`` / ``pulmonary``
# survive — the gazetteer itself ships ``severe heart failure`` and ``active liver disease``.
# Left edge only: trimming the right edge would turn the population descriptor "pregnant women"
# into the emittable "pregnant", defeating _POPULATION_PHRASES.
_HEDGE_TOKENS: frozenset[str] = frozenset(
    (
        # determiners / quantifiers
        "a",
        "an",
        "the",
        "any",
        "all",
        "some",
        "other",
        "certain",
        "this",
        "these",
        "those",
        # prepositions / conjunctions
        "of",
        "with",
        "in",
        "to",
        "and",
        "or",
        "for",
        "on",
        "at",
        # temporal / evidential hedges
        "recent",
        "recently",
        "prior",
        "previous",
        "previously",
        "history",
        "known",
        "suspected",
        "possible",
        "potential",
        "current",
        "currently",
        "ongoing",
        "documented",
        "existing",
        "preexisting",
        "underlying",
        # population heads (the subject, not the condition)
        "patient",
        "patients",
        "women",
        "men",
        "children",
        "individuals",
        "subjects",
    )
)


def _install_message(module: str) -> str:
    return f"NER production mode requires the '{module}' package (a core DAKP dependency) but it is not importable. Install all dependencies with: uv sync"


def _sort_key(mention: Mention) -> tuple[int, int, str, str]:
    return (mention.start, mention.end, mention.type, mention.text)


def _overlaps(start: int, end: int, other: tuple[int, int]) -> bool:
    return start < other[1] and other[0] < end


def _overlaps_any(start: int, end: int, covered: list[tuple[int, int]]) -> bool:
    return any(_overlaps(start, end, span) for span in covered)


def _contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    """True when ``outer`` covers ``inner`` (equal spans included — callers exclude equality)."""
    return outer[0] <= inner[0] and inner[1] <= outer[1]


@dataclass(frozen=True)
class _ModelSpan:
    """A GLiNER span candidate (full-text offsets) awaiting the gazetteer merge."""

    start: int
    end: int
    type: str
    score: float


@dataclass(frozen=True)
class _Candidate:
    """A trimmed model span that survived the gazetteer contest, plus what it supersedes.

    ``anchor`` is the index of the single gazetteer mention this span strictly contains — the
    more-specific-boundary case (``pulmonary hypertension`` ⊃ ``hypertension``) — or ``-1`` when
    the span stands alone with no gazetteer interaction.
    """

    span: _ModelSpan
    anchor: int


def _trim_hedges(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Trim leading :data:`_HEDGE_TOKENS` off a model span; ``None`` if nothing survives.

    Offsets move only to :data:`_GLINER_TOKEN` match boundaries, so the half-open invariant
    ``mention.text == text[start:end]`` still holds. A span with no leading hedge is returned
    untouched (leading whitespace/punctuation included), so spans the model already got right
    are byte-identical to the pre-trim behavior.
    """
    tokens = list(_GLINER_TOKEN.finditer(text, start, end))
    if not tokens:
        return None
    for position, token in enumerate(tokens):
        if normalize_text(token.group()) not in _HEDGE_TOKENS:
            return (token.start(), end) if position else (start, end)
    return None  # every token was a hedge ("patients with") — not a mention


def _candidates_vs_gazetteer(model_spans: list[_ModelSpan], gazetteer_spans: list[tuple[int, int]]) -> list[_Candidate]:
    """Resolve each model span against the gazetteer; drop the ones the gazetteer wins.

    * strictly contains exactly one gazetteer span -> candidate that **supersedes** it (the model
      found the more specific boundary);
    * equal span, or partial overlap in either direction -> gazetteer wins (its span AND its type
      are the high-precision anchor), model span dropped;
    * contains **several** gazetteer spans -> a conjunction (``asthma or hypertension``), not a
      qualifier: the gazetteer spans stand and the model span is dropped;
    * no overlap at all -> free-standing candidate (``anchor=-1``), the OOV-recall case.
    """
    candidates: list[_Candidate] = []
    for span in model_spans:
        bounds = (span.start, span.end)
        contained: list[int] = []
        blocked = False
        for index, gazetteer_span in enumerate(gazetteer_spans):
            if not _overlaps(span.start, span.end, gazetteer_span):
                continue
            if gazetteer_span != bounds and _contains(bounds, gazetteer_span):
                contained.append(index)
            else:
                blocked = True
                break
        if blocked or len(contained) > 1:
            continue
        candidates.append(_Candidate(span=span, anchor=contained[0] if contained else -1))
    return candidates


def _select_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    """De-overlap model candidates, most specific first: longest span wins.

    Ties break on score, then ``(start, end, type)`` so the result is deterministic regardless of
    the order GLiNER returned its entities in.
    """
    ordered = sorted(candidates, key=lambda c: (-(c.span.end - c.span.start), -c.span.score, c.span.start, c.span.end, c.span.type))
    kept: list[_Candidate] = []
    for candidate in ordered:
        if not _overlaps_any(candidate.span.start, candidate.span.end, [(k.span.start, k.span.end) for k in kept]):
            kept.append(candidate)
    return kept


def _merge_straddling_spans(windows: list[tuple[int, str]], spans_by_window: list[list[_ModelSpan]]) -> None:
    """Rejoin model spans a hard window split cut across a phrase boundary.

    Sentence-piece windows tile the text exactly (gap-free), so a mention cannot straddle those
    boundaries — only budget hard splits drop the whitespace between one window's last token and
    the next window's first token, which can cut a multiword mention into two partial spans
    (``myasthenia | gravis`` → ``myasthenia`` + ``gravis``, observed on a run-on sentence). When
    one window's last span ends flush at its end and the next window's first span begins flush at
    its start, the pair is re-unified into a single span; type and score come from the
    higher-scoring side (ties go left, deterministically). A lone edge span is kept as-is. Merged
    spans re-enter the next window pair, so a mention cut across two boundaries chains too.
    """
    for window_pair, span_pair in zip(itertools.pairwise(windows), itertools.pairwise(spans_by_window), strict=True):
        (prev_start, prev_window), (curr_start, _curr_window) = window_pair
        prev_spans, curr_spans = span_pair
        prev_end = prev_start + len(prev_window)
        if curr_start == prev_end:
            continue  # contiguous (sentence-piece) boundary: mentions cannot span punctuation
        left = next((span for span in reversed(prev_spans) if span.end == prev_end), None)
        right = next((span for span in curr_spans if span.start == curr_start), None)
        if left is None or right is None:
            continue
        anchor = left if left.score >= right.score else right
        prev_spans.remove(left)
        curr_spans.remove(right)
        curr_spans.append(_ModelSpan(start=left.start, end=right.end, type=anchor.type, score=max(left.score, right.score)))


def _sentence_piece_spans(text: str) -> list[tuple[int, int]]:
    """Char spans of sentence-ish pieces tiling ``text`` exactly (gap-free, in order).

    Falls back to a single whole-text piece when the piece regex cannot tile the text (leading
    punctuation, dangling terminals), so callers always get contiguous coverage.
    """
    spans = [(match.start(), match.end()) for match in _SENTENCE_PIECE.finditer(text)]
    if (
        not spans
        or spans[0][0] != 0
        or spans[-1][1] != len(text)
        or any(prev_end != next_start for (_, prev_end), (next_start, _) in itertools.pairwise(spans))
    ):
        return [(0, len(text))]
    return spans


def _hard_split_spans(text: str, start: int, end: int, budget: int) -> list[tuple[int, int, int]]:
    """Split one over-budget ``text[start:end]`` slice into budget-sized ``(start, end, tokens)``
    windows at GLiNER token-match boundaries (each window stays an exact substring)."""
    matches = list(_GLINER_TOKEN.finditer(text, start, end))
    return [(window[0].start(), window[-1].end(), len(window)) for window in (matches[i : i + budget] for i in range(0, len(matches), budget))]


def _windows(text: str, budget: int) -> list[tuple[int, str]]:
    """Slice ``text`` into exact-substring windows of at most ``budget`` GLiNER word tokens.

    Returns ``(start, window)`` pairs with ``window == text[start:start + len(window)]``; the
    windows follow each other in text order, so entity char offsets predicted on a window remap to
    full-text coordinates by adding ``start``. Sentence-ish pieces are packed greedily; a single
    piece longer than the budget is hard-split at token boundaries. Blank text yields no windows.
    """
    if not text.strip():
        return []
    pieces: list[tuple[int, int, int]] = []
    for piece_start, piece_end in _sentence_piece_spans(text):
        tokens = len(_GLINER_TOKEN.findall(text[piece_start:piece_end]))
        if tokens > budget:
            pieces.extend(_hard_split_spans(text, piece_start, piece_end, budget))
        else:
            pieces.append((piece_start, piece_end, tokens))
    windows: list[tuple[int, str]] = []
    window_start = window_end = token_total = 0
    open_window = False
    for piece_start, piece_end, tokens in pieces:
        if open_window and token_total + tokens > budget:
            windows.append((window_start, text[window_start:window_end]))
            open_window = False
            token_total = 0
        if not open_window:
            window_start = piece_start
            open_window = True
        window_end = piece_end
        token_total += tokens
    if open_window:  # pragma: no cover - defensive: pieces is never empty (non-blank text
        # always yields >=1 piece and the last iteration always opens a window)
        windows.append((window_start, text[window_start:window_end]))
    return windows


def _token_budget(model: Any, override: int | None) -> int:
    """Window budget in GLiNER word tokens: explicit ``override``, else the model's
    ``config.max_len``, else :data:`_DEFAULT_WORD_BUDGET` — never below 1."""
    if override is not None:
        return max(1, int(override))
    max_len = getattr(getattr(model, "config", None), "max_len", None)
    if isinstance(max_len, int) and max_len >= 1:
        return max_len
    return _DEFAULT_WORD_BUDGET


def _cuda_device_supported(torch_mod: Any, device_index: int) -> bool:
    """True when the installed torch has kernels compiled for ``cuda:device_index``.

    ``torch.cuda.is_available()`` only checks driver + runtime — it lies when torch was built
    without kernels for the present GPU arch (torch >=2.8 cu128 wheels have no sm_60 code, so
    the P100s pass ``is_available()`` but raise on first use). A device is usable iff its
    compute capability appears in the arch list torch was compiled for.
    """
    try:
        major, minor = torch_mod.cuda.get_device_capability(device_index)
    except Exception:  # defensive: driver/runtime errors surface as unusable, not a crash
        return False
    return f"sm_{major}{minor}" in torch_mod.cuda.get_arch_list()


def _model_device() -> str:
    """Device the GLiNER model runs on: CUDA when available AND supported by the installed
    torch build (orders of magnitude faster than the CPU fallback, which saturates every
    core), else CPU.

    The :func:`_cuda_device_supported` gate covers hosts whose GPUs predate the torch build's
    compiled architectures (e.g. the P100/sm_60 build server against a cu128 wheel line):
    there, ``is_available()`` is True but the first CUDA call raises, so mining silently
    downgrades to CPU instead (see ``plans/fix-p100-torch-sm60.md``).
    """
    try:
        import torch  # lazy: no torch at module load
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    if not _cuda_device_supported(torch, 0):
        logger.warning("ner_device_unsupported: cuda:0 arch not in torch build arch list = {}; falling back to CPU", torch.cuda.get_arch_list())
        return "cpu"
    return "cuda"


# --- one GLiNER model per GPU -------------------------------------------------

#: Env var overriding the GPU-lock directory (tests point it at a tmp dir).
_GPU_LOCK_DIR_ENV = "DAKP_GPU_LOCK_DIR"


def _cuda_index(device: str) -> int:
    """Numeric index from a CUDA device string: ``"cuda"`` -> 0, ``"cuda:2"`` -> 2."""
    return int(device.partition(":")[2] or 0)


def _gpu_lock_dir(cache_dir: Path | str | None = None, workdir: Path | str | None = None) -> Path:
    """Where per-device lock files live, resolved like the model cache dir.

    ``$DAKP_GPU_LOCK_DIR`` wins (tests); then ``<workdir>/cache/gpu-locks``; then, with no
    workdir, a sibling of the model cache root (``<cache>/../gpu-locks``, i.e.
    ``~/.cache/dakp/gpu-locks`` for the default user cache).
    """
    override = os.environ.get(_GPU_LOCK_DIR_ENV, "").strip()
    if override:
        return Path(override)
    if workdir is not None:
        return Path(workdir) / "cache" / "gpu-locks"
    cache = Path(cache_dir) if cache_dir is not None else default_model_cache_dir()
    return cache.parent / "gpu-locks"


def _acquire_gpu_lock(device: str, lock_dir: Path) -> int:
    """Acquire the BLOCKING exclusive flock for a CUDA device; return the open fd.

    One GLiNER model per GPU is a hard cap (two models OOM a 16 GB P100), and the Airflow
    ``ner_mining`` pool is only a scheduler hint — a second DAG run, a manual task trigger,
    or a stray CLI invocation bypasses it. The kernel flock is the correctness guarantee:
    every production CUDA load parks here until the previous holder's fd closes. The caller
    keeps the returned fd open for the life of the loaded model; there is no explicit
    release path — closing the fd (or process exit, which is the whole lifecycle of the
    spawned per-GPU mining workers) releases the lock.
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"cuda-{_cuda_index(device)}.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stats(logger, "ner_gpu_lock", level="DEBUG", device=device, path=str(path), waited=False)
        except BlockingIOError:
            stats(logger, "ner_gpu_lock", device=device, path=str(path), waited=True)
            fcntl.flock(fd, fcntl.LOCK_EX)
    except BaseException:
        os.close(fd)
        raise
    return fd


class DiseaseNER:
    """The single composite disease/phenotype mention extractor.

    Constructing a ``DiseaseNER`` never imports heavy deps — even in production mode ``gliner``
    is imported only on the first :meth:`extract` (no torch at module load). ``gliner`` is a core
    DAKP dependency (installed by ``uv sync``); the lazy import keeps ``import dakp_pipeline.ner.ner``
    — and the whole test suite — fast and light.

    Args:
        offline: ``True`` (default) = deterministic gazetteer only; ``False`` = gazetteer +
            GLiNER zero-shot recall.
        gazetteer: a :class:`Gazetteer`, a ``{surface: type}`` mapping, or ``None`` to use the
            curated :data:`EMBEDDED_GAZETTEER`.
        model_id: GLiNER checkpoint (production mode).
        threshold: GLiNER candidate-**generation** threshold (production mode). Generate wide:
            this only decides what the merge gets to look at.
        accept_threshold: DAKP-side **acceptance** floor (production mode). A model span below it
            is abstained on — emitted as nothing, never downgraded to the generic gazetteer term
            it was competing with. Keep ``threshold <= accept_threshold``; an ``accept_threshold``
            below ``threshold`` is simply a no-op (GLiNER already filtered).
        chunk_words: window budget in GLiNER word tokens for long texts (production mode);
            ``None`` = the loaded model's ``config.max_len``, fallback 384. GLiNER silently
            truncates anything longer, so longer texts are predicted window by window.
        cache_dir / workdir: model-cache location (see :func:`dakp_pipeline.ner.model_cache.ensure_model`).
        device: explicit CUDA device for multi-GPU dispatch (e.g. ``"cuda:2"``). ``None``
            (default) auto-detects via :func:`_model_device` (CUDA when available, else CPU).
            Set by multi-GPU workers to pin the GLiNER model to a specific GPU.
    """

    def __init__(
        self,
        *,
        offline: bool = True,
        gazetteer: Gazetteer | Mapping[str, str] | None = None,
        model_id: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
        chunk_words: int | None = None,
        cache_dir: Path | str | None = None,
        workdir: Path | str | None = None,
        device: str | None = None,
    ) -> None:
        if isinstance(gazetteer, Gazetteer):
            resolved = gazetteer
        elif gazetteer is None:
            resolved = Gazetteer(EMBEDDED_GAZETTEER)
        else:
            resolved = Gazetteer(gazetteer)
        self._gazetteer = resolved
        self._matcher = LexicalMatcher(resolved)
        self._offline = offline
        self._model_id = model_id
        self._threshold = threshold
        self._accept = accept_threshold
        self._chunk_words = chunk_words
        self._cache_dir = cache_dir
        self._workdir = workdir
        self._device = device
        self._model: Any = None
        self._gpu_lock_fd: int | None = None

    # -- builders --------------------------------------------------------------
    @classmethod
    def from_tsv(cls, path: Path, *, offline: bool = True, text_col: str = "text", type_col: str = "type", **kwargs: Any) -> DiseaseNER:
        """Build an offline backend from a term/type TSV (see :meth:`Gazetteer.from_tsv`)."""
        return cls(offline=offline, gazetteer=Gazetteer.from_tsv(path, text_col=text_col, type_col=type_col), **kwargs)

    # -- extraction ------------------------------------------------------------
    def extract(self, text: str) -> list[Mention]:
        """Extract disease/phenotype mentions, deterministically ordered.

        Offline: gazetteer spans only. Production: gazetteer spans merged with GLiNER spans,
        most specific span winning on containment (see :meth:`_merge_model_spans`). Empty/blank
        text yields no mentions — as does text whose only candidates fall below the acceptance
        floor, so callers must always handle an empty list.
        """
        if not text or not text.strip():
            return []
        mentions = list(self._matcher.match(text))
        if not self._offline:
            mentions = self._merge_model_spans(text, mentions)
        return sorted(mentions, key=_sort_key)

    # -- production model (lazy) -----------------------------------------------
    def _load_model(self) -> Any:
        if self._model is None:
            stats(logger, "ner_model_load", model_id=self._model_id)
            try:
                from gliner import GLiNER  # lazy: no torch at module load  # type: ignore[import-not-found]
            except ImportError as exc:
                raise NERDependencyError(_install_message("gliner")) from exc
            ref = ensure_model(self._model_id, cache_dir=self._cache_dir, workdir=self._workdir)
            device = self._device or _model_device()
            if device.startswith("cuda"):
                # One GLiNER per GPU is a hard cap (16 GB cards OOM with two models). The
                # Airflow ``ner_mining`` pool serializes shape tasks at the scheduler level,
                # but this per-device flock is the correctness guarantee for any concurrent
                # loader (second DAG run, manual trigger, CLI). The fd is held on the instance
                # for the life of the model; process exit releases it, which is the whole
                # lifecycle of a spawned GPU worker. CPU and offline loads never lock.
                self._gpu_lock_fd = _acquire_gpu_lock(device, _gpu_lock_dir(cache_dir=self._cache_dir, workdir=self._workdir))
            started = time.monotonic()
            self._model = GLiNER.from_pretrained(str(ref.path), map_location=device)
            stats(logger, "ner_model_load", model_id=self._model_id, device=device, b3=ref.b3, elapsed_s=round(time.monotonic() - started, 3))
        return self._model

    def _config(self) -> dict[str, Any]:
        """Serializable construction kwargs (for multi-process GPU worker reconstruction).

        Returns the ``DiseaseNER`` init kwargs as plain JSON-picklable values so a
        :class:`~concurrent.futures.ProcessPoolExecutor` worker can reconstruct an
        equivalent backend pinned to a specific device via ``DiseaseNER(device=..., **config)``.
        The ``device`` kwarg itself is deliberately excluded — the caller sets it per-worker.
        """
        return {
            "offline": self._offline,
            "gazetteer": self._gazetteer,
            "model_id": self._model_id,
            "threshold": self._threshold,
            "accept_threshold": self._accept,
            "chunk_words": self._chunk_words,
            "cache_dir": self._cache_dir,
            "workdir": self._workdir,
        }

    def _merge_model_spans(self, text: str, gazetteer_mentions: list[Mention]) -> list[Mention]:
        """Merge GLiNER spans with the gazetteer, preferring the **most specific** span.

        GLiNER silently truncates inputs past ``config.max_len`` word tokens, so long text is
        predicted in exact-substring windows (:func:`_windows`) and each span's offsets are
        shifted back into full-text coordinates first. Model spans whose normalized surface is a
        population descriptor (:data:`_POPULATION_PHRASES`) are dropped, and spans a hard split
        cut across a phrase boundary are re-joined (:func:`_merge_straddling_spans`) — the rejoin
        runs before trimming so window-flush offsets still line up.

        Surviving spans are then hedge-trimmed (:func:`_trim_hedges`), contested against the
        gazetteer (:func:`_candidates_vs_gazetteer`), de-overlapped longest-first
        (:func:`_select_candidates`) and finally gated on :data:`DEFAULT_ACCEPT_THRESHOLD` — see
        :meth:`_emit`.
        """
        model = self._load_model()
        budget = _token_budget(model, self._chunk_words)
        labels = list(CONTRAINDICATION_DISEASE_TYPES)
        windows = _windows(text, budget)
        spans_by_window: list[list[_ModelSpan]] = []
        for window_start, window in windows:
            raw = model.predict_entities(window, labels, threshold=self._threshold)
            spans: list[_ModelSpan] = []
            for entity in raw:
                etype = canonical_type(str(entity["label"]))
                if etype not in CONTRAINDICATION_DISEASE_TYPES:
                    continue
                start, end = window_start + int(entity["start"]), window_start + int(entity["end"])
                if normalize_text(text[start:end]) in _POPULATION_PHRASES:
                    continue
                spans.append(_ModelSpan(start=start, end=end, type=etype, score=float(entity["score"])))
            spans_by_window.append(spans)
        _merge_straddling_spans(windows, spans_by_window)
        trimmed = self._trimmed_spans(text, [span for spans in spans_by_window for span in spans])
        gazetteer_spans = [(mention.start, mention.end) for mention in gazetteer_mentions]
        candidates = _select_candidates(_candidates_vs_gazetteer(trimmed, gazetteer_spans))
        return self._emit(text, gazetteer_mentions, candidates)

    def _trimmed_spans(self, text: str, spans: list[_ModelSpan]) -> list[_ModelSpan]:
        """Hedge-trim every model span, re-applying the population filter to the trimmed surface.

        The population re-check matters because trimming can *reveal* a descriptor: ``in pregnant
        women`` trims to ``pregnant women``, which is a subject population, not a phenotype.
        """
        trimmed: list[_ModelSpan] = []
        for span in spans:
            bounds = _trim_hedges(text, span.start, span.end)
            if bounds is None:
                continue
            start, end = bounds
            if normalize_text(text[start:end]) in _POPULATION_PHRASES:
                continue
            trimmed.append(_ModelSpan(start=start, end=end, type=span.type, score=span.score))
        return trimmed

    def _emit(self, text: str, gazetteer_mentions: list[Mention], candidates: list[_Candidate]) -> list[Mention]:
        """Apply the acceptance floor and build the final mention list.

        A candidate that supersedes a gazetteer span takes the model's **boundary** but keeps the
        **gazetteer's type**: ``ner/BENCHMARK.md`` records GLiNER mistyping exactly these terms
        (``gastrointestinal bleeding``, ``QT prolongation``, ``seizures``), and type confusion is
        orthogonal to boundary precision — the gazetteer is still the better authority on type.

        Below the floor the backend **abstains**: a superseded gazetteer mention is not
        resurrected, because emitting ``hypertension`` for text that reads ``pulmonary
        hypertension`` asserts a broader contraindication than the label supports. Returning
        nothing is the honest answer, and ``extract`` already contracts an empty list.
        """
        superseded: set[int] = set()
        accepted: list[Mention] = []
        for candidate in candidates:
            span = candidate.span
            surface = text[span.start : span.end]
            supersedes = candidate.anchor >= 0
            if supersedes:
                superseded.add(candidate.anchor)
            if span.score < self._accept:
                self._abstain(surface, span.score, "superseded_unresolved" if supersedes else "below_accept_floor")
                continue
            etype = gazetteer_mentions[candidate.anchor].type if supersedes else span.type
            accepted.append(
                Mention(
                    text=surface,
                    start=span.start,
                    end=span.end,
                    type=etype,
                    score=span.score,
                    normalized=normalize_text(surface),
                    notes="gliner:extends" if supersedes else "gliner",
                )
            )
        kept = [mention for index, mention in enumerate(gazetteer_mentions) if index not in superseded]
        return kept + accepted

    def _abstain(self, surface: str, score: float, reason: str) -> None:
        """Record a mention the backend declined to assert.

        DEBUG level on purpose: per-span detail is what you need to retune ``accept_threshold``,
        but a production run mines tens of thousands of sections and this must not flood INFO.
        """
        stats(logger, "ner_abstain", level="DEBUG", reason=reason, surface=surface, score=round(score, 4), accept_threshold=self._accept)


# --- module-level entry points -------------------------------------------------


def extract_disease_mentions(text: str, ner: DiseaseNER | None = None) -> list[Mention]:
    """Extract disease/phenotype mentions from ``text`` (one unified entry point).

    Uses ``ner`` or a default offline :class:`DiseaseNER` (deterministic embedded gazetteer).
    """
    return (ner or DiseaseNER()).extract(text)


def extract_contraindication_diseases(text: str, ner: DiseaseNER | None = None) -> list[Mention]:
    """Extract disease/phenotype mentions from a DailyMed contraindication section (LOINC 34070-3).

    The backend is specialized to disease/phenotype mentions, so this is
    :func:`extract_disease_mentions` under a contraindication-specific name; the assertion
    builder feeds these mention spans (text + type) into contraindication rows.
    """
    return extract_disease_mentions(text, ner)


__all__ = [
    "CONTRAINDICATION_DISEASE_TYPES",
    "DEFAULT_ACCEPT_THRESHOLD",
    "DEFAULT_MODEL",
    "DEFAULT_THRESHOLD",
    "EMBEDDED_GAZETTEER",
    "TYPE_DISEASE",
    "TYPE_PHENOTYPE",
    "DiseaseNER",
    "Mention",
    "NERDependencyError",
    "extract_contraindication_diseases",
    "extract_disease_mentions",
]
