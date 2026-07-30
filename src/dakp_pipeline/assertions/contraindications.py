"""Contraindication assertion aggregation — text-mined from DailyMed SPL.

Builds ``contraindication_assertions.tsv``: drug→condition contraindication assertions
**mined directly** from DailyMed SPL "Contraindications" sections (LOINC ``34070-3``) using
a configurable NER backend (PLAN.md "Phase 4: NER / entity resolution strategy"). This
replaces the former externally-sourced contraindication list: contraindications are now
extracted from the label text itself, which yields better coverage and DailyMed-grounded
provenance.

Mining rule (explicit and tested)
---------------------------------
For every SPL set that carries a contraindication section **and** at least one active
ingredient, run the NER backend over the section text
(:func:`~dakp_pipeline.ner.backends.extract_contraindication_diseases`) to extract
disease/phenotype mentions; pair each mention with each active ingredient of that set to
form a ``biolink:contraindicated_in`` assertion (ingredient = subject, mention = object).
Rows are aggregated by ``(subject_text, object_text)``: ``supporting_spl_sets`` and
``supporting_spl_documents`` are unioned, ``source_score`` takes the max NER span score.
Object CURIE/name/category are resolved from the lexical disease baseline (``disease_map``)
where the mined mention is known (text-first; CURIEs populated only where available).

Backend selection (:func:`resolve_ner_backend`)
-----------------------------------------------
``params["ner_backend"]`` (an :class:`~dakp_pipeline.ner.backends.NERBackend` instance) wins;
else ``params["ner_backend_name"]`` selects one via
:func:`~dakp_pipeline.ner.backends.get_backend` (``mock|dictionary|gliner|scispacy`` — the
real GLiNER/SciSpacy backends need the ``[ner]`` extra); else the offline default is the
deterministic :class:`~dakp_pipeline.ner.backends.DictionaryNERBackend` built from the
ontology fixture (``<fixture_root>/ontology/disease_map.tsv``), falling back to an empty
:class:`~dakp_pipeline.ner.backends.MockNERBackend`. Constructing any backend is import-free,
so the base install + test suite run with the ``[ner]`` extra NOT installed.

Provenance: contraindications are text-mined from DailyMed, so
``primary_knowledge_source = infores:multiomics-drugapprovals``,
``upstream_resource_ids = infores:dailymed``, ``agent_type = text_mining_agent`` (the DAKP
RIG lists ``text_mining_agent`` for ``contraindicated_in``), ``knowledge_level =
knowledge_assertion``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from dakp_pipeline.assertions import INFORES_DAILYMED, INFORES_DAKP, KL_ASSERTION, match_diseases, row_for
from dakp_pipeline.assertions.evidence import build_dailymed_evidence, find_table, sorted_pipe, write_assertion_table
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.ner.backends import (
    TYPE_PHENOTYPE,
    DictionaryNERBackend,
    EntitySpan,
    MockNERBackend,
    NERBackend,
    canonical_type,
    extract_contraindication_diseases,
    get_backend,
)

_TABLE = "contraindication_assertions"
_PREDICATE = "biolink:contraindicated_in"
_INGREDIENTS_FILE = "spl_ingredients.parquet"
_ONTOLOGY_FIXTURE = Path("ontology") / "disease_map.tsv"

#: Agent type for text-mined contraindications (matches the DAKP RIG ``contraindicated_in``).
AT_TEXT_MINING = "text_mining_agent"


def resolve_ner_backend(fixture_root: Path | str | None, params: Mapping[str, Any]) -> NERBackend:
    """Resolve the contraindication-mining NER backend (configurable; offline by default).

    Priority: an injected ``params["ner_backend"]`` instance -> ``params["ner_backend_name"]``
    via :func:`get_backend` (``mock|dictionary|gliner|scispacy``) -> the deterministic
    dictionary baseline over the ontology fixture -> an empty mock backend. Never imports a
    heavy ``[ner]`` dep at construction time.
    """
    backend = params.get("ner_backend")
    if backend is not None:
        return backend  # type: ignore[no-any-return]
    name = str(params.get("ner_backend_name") or "").strip().lower()
    if name:
        return get_backend(name)
    if fixture_root is not None:
        ontology = Path(fixture_root) / _ONTOLOGY_FIXTURE
        if ontology.exists():
            return DictionaryNERBackend.from_tsv(ontology)
    return MockNERBackend()


class ContraindicationsShaper:
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        disease_map: dict[str, dict[str, str]] = ctx.params.get("disease_map", {})  # type: ignore[assignment]
        backend = resolve_ner_backend(ctx.fixture_root, ctx.params)
        rows = build_contraindication_rows(inputs, backend, disease_map)
        return write_assertion_table(_TABLE, rows, inputs, ctx, operation="shape_contraindications")


def build_contraindication_rows(
    inputs: Iterable[ArtifactRef], backend: NERBackend, disease_map: Mapping[str, Mapping[str, str]]
) -> list[dict[str, str]]:
    """Mine DailyMed contraindication sections into assertion rows (deterministic).

    For each SPL set with a contraindication section and >=1 active ingredient, extract
    disease/phenotype mentions from the section text and pair each with each active
    ingredient. Aggregated by ``(subject_text, object_text)`` and sorted for stable output.
    """
    evidence = build_dailymed_evidence(inputs)
    ingredients_by_set = _active_ingredients_by_set(inputs)

    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for set_id in sorted(evidence.contraindication_docs):
        ingredients = ingredients_by_set.get(set_id, [])
        if not ingredients:
            continue
        for doc_id, text in evidence.contraindication_docs[set_id]:
            for span in extract_contraindication_diseases(text, backend):
                object_text = span.text.strip()
                if not object_text:
                    continue
                for ingredient_name, ingredient_unii in ingredients:
                    _accumulate(aggregated, set_id, doc_id, ingredient_name, ingredient_unii, object_text, span, disease_map)

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
    span: EntitySpan,
    disease_map: Mapping[str, Mapping[str, str]],
) -> None:
    """Add one (ingredient, mention) observation to the ``(subject, object)`` aggregate."""
    key = (ingredient_name, object_text)
    agg = aggregated.setdefault(
        key,
        {
            "subject_text": ingredient_name,
            "subject_curie": ingredient_unii,
            "subject_name": ingredient_name,
            "object_text": object_text,
            "object_curie": "",
            "object_name": object_text,
            "object_category": _category_for_type(span.type),
            "sets": [],
            "docs": [],
            "scores": [],
        },
    )
    agg["sets"].append(set_id)
    agg["docs"].append(doc_id)
    agg["scores"].append(span.score)
    # Resolve the mined mention to a canonical disease/phenotype where the baseline knows it.
    if not agg["object_curie"]:
        matches = match_diseases(object_text, disease_map)
        if matches:
            agg["object_curie"] = matches[0]["curie"]
            agg["object_name"] = matches[0]["name"] or object_text
            agg["object_category"] = matches[0]["category"] or agg["object_category"]


def _category_for_type(span_type: str) -> str:
    """Biolink object category for a canonical NER span type (phenotype vs disease)."""
    return "PhenotypicFeature" if canonical_type(span_type) == TYPE_PHENOTYPE else "Disease"


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

__all__ = ["AT_TEXT_MINING", "ContraindicationsShaper", "build_contraindication_rows", "resolve_ner_backend", "transform"]
