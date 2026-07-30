"""NER benchmark harness — precision/recall/F1 of candidate disease/phenotype extractors.

Evaluates candidate mention-extraction approaches against the hand-labeled gold fixture
(``ner_gold.json``: DailyMed contraindication sections (LOINC 34070-3) + FAERS indication
strings with gold disease/phenotype spans). This is the evidence base for settling on ONE
composite NER backend (see ``ner/README.md``).

Candidates benchmarked:
  * ``gazetteer``  — the deterministic curated-gazetteer + lexical matcher (offline baseline).
  * ``gliner``     — GLiNER zero-shot biomedical NER (``urchade/gliner_small-v2.1``).
  * ``scispacy``   — SciSpacy BC5CDR (``en_ner_bc5cdr_md``) — skipped if the model is absent.

Scoring is span-level, micro-averaged: a prediction is a true positive only if its
``(start, end, type)`` exactly matches a gold span. Run with::

    uv run python tests/eval/benchmark_ner.py            # all runnable candidates
    uv run python tests/eval/benchmark_ner.py --json out.json

This script is an evaluation artifact: it is NOT collected by pytest (filename is not
``test_*.py``) and is not part of the coverage-gated package. It imports ``ner.backends``
(which lazy-loads heavy deps) and never imports ``gliner``/``spacy`` directly, so it
type-checks with or without the ``[ner]`` extra installed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLD_PATH = HERE / "ner_gold.json"

# --- the curated disease/phenotype gazetteer (the offline baseline under test) -----
# term -> canonical type. This is the high-precision anchor of the settled composite
# backend; it is intentionally NOT exhaustive — the three out-of-gazetteer gold cases
# (porphyria / myasthenia gravis / pheochromocytoma) measure model generalization that a
# gazetteer alone cannot provide.
GAZETTEER: dict[str, str] = {
    # disease (named disorders / syndromes)
    "asthma": "disease",
    "liver disease": "disease",
    "active liver disease": "disease",
    "heart failure": "disease",
    "severe heart failure": "disease",
    "myocardial infarction": "disease",
    "peptic ulcer": "disease",
    "peptic ulcer disease": "disease",
    "renal impairment": "disease",
    "severe renal impairment": "disease",
    "renal disease": "disease",
    "end stage renal disease": "disease",
    "hypertension": "disease",
    "uncontrolled hypertension": "disease",
    "arrhythmia": "disease",
    "ventricular arrhythmias": "disease",
    "stroke": "disease",
    "hemorrhagic stroke": "disease",
    "hepatic impairment": "disease",
    "severe hepatic impairment": "disease",
    "epilepsy": "disease",
    "glaucoma": "disease",
    "narrow angle glaucoma": "disease",
    "hypercholesterolemia": "disease",
    "diabetes mellitus": "disease",
    "type 2 diabetes mellitus": "disease",
    "chronic obstructive pulmonary disease": "disease",
    "atrial fibrillation": "disease",
    "arthritis": "disease",
    "rheumatoid arthritis": "disease",
    "depression": "disease",
    "migraine": "disease",
    # phenotype (clinical findings / symptoms / condition-states)
    "hypersensitivity": "phenotype",
    "transaminase elevations": "phenotype",
    "bleeding": "phenotype",
    "active bleeding": "phenotype",
    "gastrointestinal bleeding": "phenotype",
    "pregnancy": "phenotype",
    "qt prolongation": "phenotype",
    "seizure": "phenotype",
    "seizures": "phenotype",
    "headache": "phenotype",
    "pain": "phenotype",
    "back pain": "phenotype",
    "nausea": "phenotype",
    "vomiting": "phenotype",
    "fatigue": "phenotype",
}


@dataclass(frozen=True)
class Gold:
    start: int
    end: int
    type: str
    surface: str


@dataclass(frozen=True)
class Case:
    case_id: str
    source: str
    text: str
    gold: tuple[Gold, ...]


@dataclass(frozen=True)
class Pred:
    start: int
    end: int
    type: str
    text: str


Predictor = Callable[[str], list[Pred]]


# --- gold loading --------------------------------------------------------------


def load_cases(path: Path = GOLD_PATH) -> list[Case]:
    """Load the gold fixture, resolving each surface to unique ``(start, end)`` offsets."""
    data = json.loads(path.read_text("utf-8"))
    cases: list[Case] = []
    for entry in data["cases"]:
        text = entry["text"]
        gold: list[Gold] = []
        for mention in entry["mentions"]:
            surface = mention["surface"]
            if text.count(surface) != 1:
                msg = f"gold surface {surface!r} not unique in case {entry['id']!r}"
                raise ValueError(msg)
            start = text.index(surface)
            gold.append(Gold(start=start, end=start + len(surface), type=mention["type"], surface=surface))
        cases.append(Case(case_id=entry["id"], source=entry["source"], text=text, gold=tuple(gold)))
    return cases


# --- candidate predictors (wrap ner.backends; heavy deps load lazily) ----------


def gazetteer_predictor() -> Predictor:
    """Deterministic curated-gazetteer + lexical matcher (the offline baseline)."""
    from dakp_pipeline.ner.backends import CONTRAINDICATION_DISEASE_TYPES, DictionaryNERBackend
    from dakp_pipeline.ner.dictionary import DictionaryEntry, DictionaryIndex, normalize_text

    category = {"disease": "Disease", "phenotype": "PhenotypicFeature"}
    entries = [
        DictionaryEntry(normalized=normalize_text(term), curie="", name=term, category=category[etype], source="gazetteer", original=term)
        for term, etype in GAZETTEER.items()
    ]
    backend = DictionaryNERBackend(DictionaryIndex(entries))

    def predict(text: str) -> list[Pred]:
        return [Pred(s.start, s.end, s.type, s.text) for s in backend.extract(text, CONTRAINDICATION_DISEASE_TYPES)]

    return predict


def gliner_predictor(threshold: float = 0.5) -> Predictor:
    """GLiNER zero-shot biomedical NER."""
    from dakp_pipeline.ner.backends import CONTRAINDICATION_DISEASE_TYPES, GLiNERBackend

    backend = GLiNERBackend(threshold=threshold)

    def predict(text: str) -> list[Pred]:
        return [Pred(s.start, s.end, s.type, s.text) for s in backend.extract(text, CONTRAINDICATION_DISEASE_TYPES)]

    return predict


def scispacy_predictor() -> Predictor:
    """SciSpacy BC5CDR biomedical NER (DISEASE/CHEMICAL only — no phenotype label)."""
    from dakp_pipeline.ner.backends import CONTRAINDICATION_DISEASE_TYPES, SciSpacyBackend

    backend = SciSpacyBackend()

    def predict(text: str) -> list[Pred]:
        return [Pred(s.start, s.end, s.type, s.text) for s in backend.extract(text, CONTRAINDICATION_DISEASE_TYPES)]

    return predict


# --- scoring -------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _prf(tp: int, fp: int, fn: int) -> Metrics:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return Metrics(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def score(predict: Predictor, cases: Sequence[Case], *, type_aware: bool = True) -> Metrics:
    """Span-level micro P/R/F1. Strict: a TP requires matching ``(start, end)`` and type."""
    tp = fp = fn = 0
    for case in cases:
        gold = {(g.start, g.end, g.type) for g in case.gold} if type_aware else {(g.start, g.end, "") for g in case.gold}
        preds = predict(case.text)
        pred = {(p.start, p.end, p.type) for p in preds} if type_aware else {(p.start, p.end, "") for p in preds}
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return _prf(tp, fp, fn)


# --- driver --------------------------------------------------------------------


def _candidates() -> dict[str, Predictor]:
    """Runnable candidate predictors; a backend that cannot load is skipped with a note."""
    candidates: dict[str, Predictor] = {"gazetteer": gazetteer_predictor()}
    try:
        candidates["gliner"] = gliner_predictor()
    except Exception as exc:
        print(f"[skip] gliner: {exc}")
    try:
        candidates["scispacy"] = scispacy_predictor()
    except Exception as exc:
        print(f"[skip] scispacy: {exc}")
    return candidates


def run(json_out: Path | None = None) -> dict[str, dict[str, float]]:
    cases = load_cases()
    total_gold = sum(len(case.gold) for case in cases)
    print(
        f"NER benchmark: {len(cases)} cases, {total_gold} gold mentions "
        f"({sum(1 for c in cases if c.source == 'dailymed')} DailyMed, "
        f"{sum(1 for c in cases if c.source == 'faers')} FAERS)\n"
    )

    results: dict[str, dict[str, float]] = {}
    header = f"{'candidate':<12} {'P':>7} {'R':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}"
    print(header)
    print("-" * len(header))
    for name, predict in _candidates().items():
        strict = score(predict, cases)
        lenient = score(predict, cases, type_aware=False)
        results[name] = {"precision": strict.precision, "recall": strict.recall, "f1": strict.f1, "lenient_f1": lenient.f1}
        print(f"{name:<12} {strict.precision:>7.3f} {strict.recall:>7.3f} {strict.f1:>7.3f} {strict.tp:>5} {strict.fp:>5} {strict.fn:>5}")
    print("\n(strict = exact (start,end,type) match; lenient_f1 = offset match ignoring type)")

    if json_out is not None:
        json_out.write_text(
            json.dumps({"schema_version": "dakp.ner.benchmark.v1", "cases": len(cases), "gold_mentions": total_gold, "results": results}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {json_out}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark candidate NER backends on the gold fixture.")
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write the results JSON.")
    args = parser.parse_args()
    run(json_out=args.json)


if __name__ == "__main__":
    main()
