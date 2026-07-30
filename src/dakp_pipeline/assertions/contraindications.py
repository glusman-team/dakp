"""Contraindication assertion aggregation — text-mined from DailyMed SPL.

Builds ``contraindication_assertions.tsv``: drug→condition contraindication assertions
**mined directly** from DailyMed SPL "Contraindications" sections (LOINC ``34070-3``) using
DAKP's single composite NER backend (:mod:`dakp_pipeline.ner.ner`). Contraindications are
extracted from the label text itself, which yields better coverage and DailyMed-grounded
provenance than an externally-sourced list.

Mining rule (explicit and tested)
---------------------------------
For every SPL set that carries a contraindication section **and** at least one active
ingredient, run the NER backend over the section text
(:func:`~dakp_pipeline.ner.ner.extract_contraindication_diseases`) to extract disease/phenotype
mentions; pair each mention with each active ingredient of that set to form a
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
selector. Constructing the backend is import-free, so the base install + test suite run with
the ``[ner]`` extra NOT installed.

Provenance: contraindications are text-mined from DailyMed, so
``primary_knowledge_source = infores:multiomics-drugapprovals``,
``upstream_resource_ids = infores:dailymed``, ``agent_type = text_mining_agent`` (the DAKP RIG
lists ``text_mining_agent`` for ``contraindicated_in``), ``knowledge_level = knowledge_assertion``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dakp_pipeline.assertions import INFORES_DAILYMED, INFORES_DAKP, KL_ASSERTION, row_for
from dakp_pipeline.assertions.evidence import build_dailymed_evidence, find_table, sorted_pipe, write_assertion_table
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.ner.ner import DiseaseNER, Mention, extract_contraindication_diseases

_TABLE = "contraindication_assertions"
_PREDICATE = "biolink:contraindicated_in"
_INGREDIENTS_FILE = "spl_ingredients.parquet"
_ONTOLOGY_FIXTURE = Path("ontology") / "disease_map.tsv"

#: Agent type for text-mined contraindications (matches the DAKP RIG ``contraindicated_in``).
AT_TEXT_MINING = "text_mining_agent"


def default_ner(fixture_root: Path | str | None) -> DiseaseNER:
    """The deterministic offline NER backend: gazetteer from the ontology fixture, else embedded.

    Reads the ontology fixture as a term→type gazetteer ONLY (``text`` + ``category`` columns;
    CURIE/name columns are ignored — DAKP does not map terms to ontology concepts). No heavy
    ``[ner]`` dep is imported.
    """
    if fixture_root is not None:
        ontology = Path(fixture_root) / _ONTOLOGY_FIXTURE
        if ontology.exists():
            return DiseaseNER.from_tsv(ontology, text_col="text", type_col="category")
    return DiseaseNER()


class ContraindicationsShaper:
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        ner_param = ctx.params.get("ner")
        ner = ner_param if isinstance(ner_param, DiseaseNER) else default_ner(ctx.fixture_root)
        rows = build_contraindication_rows(inputs, ner)
        return write_assertion_table(_TABLE, rows, inputs, ctx, operation="shape_contraindications")


def build_contraindication_rows(inputs: Iterable[ArtifactRef], ner: DiseaseNER) -> list[dict[str, str]]:
    """Mine DailyMed contraindication sections into assertion rows (deterministic).

    For each SPL set with a contraindication section and >=1 active ingredient, extract
    disease/phenotype mentions from the section text and pair each with each active ingredient.
    Aggregated by ``(subject_text, object_text)`` and sorted for stable output. The object is
    the mined mention TEXT; object CURIE/name/category are left empty for Tablassert.
    """
    evidence = build_dailymed_evidence(inputs)
    ingredients_by_set = _active_ingredients_by_set(inputs)

    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for set_id in sorted(evidence.contraindication_docs):
        ingredients = ingredients_by_set.get(set_id, [])
        if not ingredients:
            continue
        for doc_id, text in evidence.contraindication_docs[set_id]:
            for mention in extract_contraindication_diseases(text, ner):
                object_text = mention.text.strip()
                if not object_text:
                    continue
                for ingredient_name, ingredient_unii in ingredients:
                    _accumulate(aggregated, set_id, doc_id, ingredient_name, ingredient_unii, object_text, mention)

    return [_finalize_row(agg) for _key, agg in sorted(aggregated.items())]


def _active_ingredients_by_set(inputs: Iterable[ArtifactRef]) -> dict[str, list[tuple[str, str]]]:
    """Map ``spl_set_id`` -> sorted ``[(ingredient_name, ingredient_unii)]`` for active ingredients.

    Reads ``spl_ingredients.parquet`` directly (rather than the single-ingredient-per-set
    evidence index) so a combination product pairs a mention with *every* active moiety.
    Deterministic: de-duplicated and sorted per set.
    """
    ingredients = find_table(list(inputs), _INGREDIENTS_FILE)
    by_set: dict[str, list[tuple[str, str]]] = {}
    if ingredients is None:
        return by_set
    seen: set[tuple[str, str, str]] = set()
    for rec in ingredients.iter_rows(named=True):
        if str(rec.get("role") or "").strip().lower() != "active":
            continue
        set_id = str(rec.get("spl_set_id") or "").strip()
        name = str(rec.get("ingredient_name") or "").strip()
        unii = str(rec.get("ingredient_unii") or "").strip()
        if not set_id or not name:
            continue
        key = (set_id, name.lower(), unii)
        if key in seen:
            continue
        seen.add(key)
        by_set.setdefault(set_id, []).append((name, unii))
    return {set_id: sorted(pairs) for set_id, pairs in by_set.items()}


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

__all__ = ["AT_TEXT_MINING", "ContraindicationsShaper", "build_contraindication_rows", "default_ner", "transform"]
