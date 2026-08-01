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
  GLiNER zero-shot (``urchade/gliner_small-v2.1``, laptop-safe) fills out-of-gazetteer gaps.
  Gazetteer spans win on overlap; non-overlapping GLiNER spans add recall. ``gliner`` is a core
  DAKP dependency but is imported lazily on first use (no torch at module load), raising
  :class:`~dakp_pipeline.ner.model_cache.NERDependencyError` ("reinstall with `uv sync`") if it is
  somehow not importable.

The offline/production toggle is a mode on this ONE backend, not a backend-name enum.

Entry points: :func:`extract_disease_mentions` (general) and :func:`extract_contraindication_diseases`
(contraindication sections). All offsets are half-open: ``mention.text == text[mention.start:mention.end]``.
Output is sorted deterministically by ``(start, end, type, text)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dakp_pipeline.ner.dictionary import CONTRAINDICATION_DISEASE_TYPES, TYPE_DISEASE, TYPE_PHENOTYPE, Gazetteer, canonical_type, normalize_text
from dakp_pipeline.ner.lexical import LexicalMatcher, Mention
from dakp_pipeline.ner.model_cache import NERDependencyError, ensure_model

# Laptop-safe small zero-shot model; override for a larger / biomedical-tuned checkpoint.
DEFAULT_MODEL = "urchade/gliner_small-v2.1"
DEFAULT_THRESHOLD = 0.5

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


def _install_message(module: str) -> str:
    return f"NER production mode requires the '{module}' package (a core DAKP dependency) but it is not importable. Install all dependencies with: uv sync"


def _sort_key(mention: Mention) -> tuple[int, int, str, str]:
    return (mention.start, mention.end, mention.type, mention.text)


def _overlaps_any(start: int, end: int, covered: list[tuple[int, int]]) -> bool:
    return any(start < cov_end and cov_start < end for cov_start, cov_end in covered)


def _model_device() -> str:
    """Device the GLiNER model runs on: CUDA when available (orders of magnitude faster than the
    CPU fallback, which saturates every core), else CPU."""
    try:
        import torch  # lazy: no torch at module load
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


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
        threshold: GLiNER confidence threshold (production mode).
        cache_dir / workdir: model-cache location (see :func:`dakp_pipeline.ner.model_cache.ensure_model`).
    """

    def __init__(
        self,
        *,
        offline: bool = True,
        gazetteer: Gazetteer | Mapping[str, str] | None = None,
        model_id: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        cache_dir: Path | str | None = None,
        workdir: Path | str | None = None,
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
        self._cache_dir = cache_dir
        self._workdir = workdir
        self._model: Any = None

    # -- builders --------------------------------------------------------------
    @classmethod
    def from_tsv(cls, path: Path, *, offline: bool = True, text_col: str = "text", type_col: str = "type", **kwargs: Any) -> DiseaseNER:
        """Build an offline backend from a term/type TSV (see :meth:`Gazetteer.from_tsv`)."""
        return cls(offline=offline, gazetteer=Gazetteer.from_tsv(path, text_col=text_col, type_col=type_col), **kwargs)

    # -- extraction ------------------------------------------------------------
    def extract(self, text: str) -> list[Mention]:
        """Extract disease/phenotype mentions, deterministically ordered.

        Offline: gazetteer spans only. Production: gazetteer spans plus non-overlapping GLiNER
        spans (gazetteer wins on overlap). Empty/blank text yields no mentions.
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
            try:
                from gliner import GLiNER  # lazy: no torch at module load  # type: ignore[import-not-found]
            except ImportError as exc:
                raise NERDependencyError(_install_message("gliner")) from exc
            ref = ensure_model(self._model_id, cache_dir=self._cache_dir, workdir=self._workdir)
            self._model = GLiNER.from_pretrained(str(ref.path), map_location=_model_device())
        return self._model

    def _merge_model_spans(self, text: str, gazetteer_mentions: list[Mention]) -> list[Mention]:
        """Add non-overlapping GLiNER disease/phenotype spans; gazetteer spans win on overlap."""
        model = self._load_model()
        raw = model.predict_entities(text, list(CONTRAINDICATION_DISEASE_TYPES), threshold=self._threshold)
        covered: list[tuple[int, int]] = [(mention.start, mention.end) for mention in gazetteer_mentions]
        merged = list(gazetteer_mentions)
        for entity in raw:
            etype = canonical_type(str(entity["label"]))
            if etype not in CONTRAINDICATION_DISEASE_TYPES:
                continue
            start, end = int(entity["start"]), int(entity["end"])
            if _overlaps_any(start, end, covered):
                continue
            covered.append((start, end))
            surface = text[start:end]
            merged.append(
                Mention(
                    text=surface, start=start, end=end, type=etype, score=float(entity["score"]), normalized=normalize_text(surface), notes="gliner"
                )
            )
        return merged


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
