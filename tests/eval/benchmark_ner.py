"""NER benchmark harness — precision/recall/F1 of disease/phenotype mention extraction.

Evaluates mention-extraction approaches against the hand-labeled gold fixture
(``ner_gold.json``: DailyMed contraindication sections (LOINC 34070-3) + FAERS indication
strings with gold disease/phenotype spans). This is the evidence base for the settled single
composite NER backend (see ``ner/BENCHMARK.md``).

Approaches benchmarked (all via the ONE ``DiseaseNER`` backend in different modes):
  * ``gazetteer``  — offline mode: curated gazetteer + lexical matcher (deterministic).
  * ``gliner``     — production mode with an EMPTY gazetteer: GLiNER zero-shot only.
  * ``composite``  — production mode with the curated gazetteer: the settled backend
                     (gazetteer anchors high-precision spans; GLiNER fills OOV recall).

Scoring is span-level, micro-averaged: a prediction is a true positive only if its
``(start, end, type)`` exactly matches a gold span. Run with::

    uv run python tests/eval/benchmark_ner.py            # all runnable approaches
    uv run python tests/eval/benchmark_ner.py --json out.json

This script is an evaluation artifact: it is NOT collected by pytest (filename is not
``test_*.py``) and is not part of the coverage-gated package. It imports ``ner.ner`` (which
lazy-loads ``gliner``) and never imports ``gliner`` directly, so it type-checks with or without
the NER dependencies installed (the model approaches simply skip when it is absent).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from dakp_pipeline.ner.ner import DEFAULT_THRESHOLD, EMBEDDED_GAZETTEER

HERE = Path(__file__).resolve().parent
GOLD_PATH = HERE / "ner_gold.json"

# --- the curated disease/phenotype gazetteer (the offline baseline under test) -----
# The shipped composite backend's embedded gazetteer itself — imported, not duplicated, so the
# benchmark always measures the terms the backend ships. It is intentionally NOT exhaustive:
# the three out-of-gazetteer gold cases (porphyria / myasthenia gravis / pheochromocytoma)
# measure model generalization that a gazetteer alone cannot provide.
GAZETTEER: dict[str, str] = EMBEDDED_GAZETTEER


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


# --- candidate predictors (all via the single DiseaseNER backend) --------------


def gazetteer_predictor() -> Predictor:
    """Offline mode: curated gazetteer + lexical matcher (deterministic, no heavy deps)."""
    from dakp_pipeline.ner.ner import DiseaseNER

    ner = DiseaseNER(offline=True, gazetteer=GAZETTEER)
    return lambda text: [Pred(m.start, m.end, m.type, m.text) for m in ner.extract(text)]


def gliner_predictor(threshold: float = DEFAULT_THRESHOLD) -> Predictor:
    """Production mode with an EMPTY gazetteer: GLiNER zero-shot only (isolates the model)."""
    from dakp_pipeline.ner.ner import DiseaseNER

    ner = DiseaseNER(offline=False, gazetteer={}, threshold=threshold)
    return lambda text: [Pred(m.start, m.end, m.type, m.text) for m in ner.extract(text)]


def composite_predictor(threshold: float = DEFAULT_THRESHOLD) -> Predictor:
    """Production mode with the curated gazetteer: the settled backend (gazetteer + GLiNER)."""
    from dakp_pipeline.ner.ner import DiseaseNER

    ner = DiseaseNER(offline=False, gazetteer=GAZETTEER, threshold=threshold)
    return lambda text: [Pred(m.start, m.end, m.type, m.text) for m in ner.extract(text)]


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
    """Runnable predictors; a model approach that cannot load is skipped with a note."""
    candidates: dict[str, Predictor] = {"gazetteer": gazetteer_predictor()}
    try:
        candidates["gliner"] = gliner_predictor()
        candidates["composite"] = composite_predictor()
    except Exception as exc:  # report any model/load failure and continue with the gazetteer
        print(f"[skip] gliner/composite (model unavailable): {exc}")
    return candidates


def run(json_out: Path | None = None) -> dict[str, dict[str, float]]:
    cases = load_cases()
    total_gold = sum(len(case.gold) for case in cases)
    dailymed = sum(1 for c in cases if c.source == "dailymed")
    faers = sum(1 for c in cases if c.source == "faers")
    print(f"NER benchmark: {len(cases)} cases, {total_gold} gold mentions ({dailymed} DailyMed, {faers} FAERS)\n")

    results: dict[str, dict[str, float]] = {}
    header = f"{'approach':<12} {'P':>7} {'R':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}"
    print(header)
    print("-" * len(header))
    for name, predict in _candidates().items():
        strict = score(predict, cases)
        lenient = score(predict, cases, type_aware=False)
        results[name] = {"precision": strict.precision, "recall": strict.recall, "f1": strict.f1, "lenient_f1": lenient.f1}
        print(f"{name:<12} {strict.precision:>7.3f} {strict.recall:>7.3f} {strict.f1:>7.3f} {strict.tp:>5} {strict.fp:>5} {strict.fn:>5}")
    print("\n(strict = exact (start,end,type) match; lenient_f1 = offset match ignoring type)")

    if json_out is not None:
        payload = {"schema_version": "dakp.ner.benchmark.v1", "cases": len(cases), "gold_mentions": total_gold, "results": results}
        json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {json_out}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark disease/phenotype NER approaches on the gold fixture.")
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write the results JSON.")
    args = parser.parse_args()
    run(json_out=args.json)


if __name__ == "__main__":
    main()
