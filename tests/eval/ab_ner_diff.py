"""A/B mention-diff harness — real-text parity gate for inference-config changes.

Samples real SPL section texts from an interim ``spl_sections.parquet``, extracts mentions with
the PRODUCTION backend (v1.15.0-shaped defaults, served from the persistent two-tier cache that
the just-finished build populated), then re-extracts with an EXPERIMENT backend carrying the
inference-config change under test (``chunk_words`` / ``inference_batch_size`` / ``compute_dtype``),
and diffs the mention multisets per text.

A config change is adoptable only when the diff is CLEAN: every sampled text produces the exact
same ``(text, start, end, type)`` multiset under both configs. The baseline side is cache-served
(zero GPU); only the experiment side mines.

Run on the GPU host, when no DAG run is active (the per-device GPU flocks would deadlock the
harness against a mining shaper)::

    uv run python tests/eval/ab_ner_diff.py --tabular tmp/tabular --workdir tmp \
        --chunk-words 1024 --batch-size 32 --dtype fp16 --out ab.json --gate

This is an evaluation artifact: NOT collected by pytest, not part of the coverage-gated package
(same convention as ``benchmark_ner.py``).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from dakp_pipeline.ner.mention_cache import MentionCache
from dakp_pipeline.ner.ner import DiseaseNER

# The LOINC passes the contraindication shaper actually mines (mirrors evidence.py's routing).
_MINED_LOINCS = {
    "34070-3",  # contraindications (pass 1)
    "34067-9",  # indications (pass 2)
    "34066-1",  # boxed warning (pass 3)
    "43685-7",  # warnings/precautions (pass 3)
    "34071-1",  # warnings/precautions (pass 3)
    "42232-9",  # warnings/precautions (pass 3)
}


def _load_texts(tabular: Path, sample: int, seed: int) -> list[str]:
    """Sample real section texts exactly as the shaper sees them (clean_text with raw fallback)."""
    path = tabular / "spl_sections.parquet"
    frame = pl.read_parquet(path)
    mined = frame.filter(pl.col("loinc_code").is_in(sorted(_MINED_LOINCS)))
    texts = (
        mined.select(pl.coalesce(pl.col("clean_text"), pl.col("raw_text")).alias("text"))
        .filter(pl.col("text").str.strip_chars().str.len_chars() > 0)
        .get_column("text")
        .to_list()
    )
    if sample < len(texts):
        rng = random.Random(seed)
        texts = rng.sample(texts, sample)
    return [str(text).strip() for text in texts]


def _mention_multiset(mentions: list[dict[str, Any]]) -> Counter[tuple[str, int, int, str]]:
    return Counter((m["text"], int(m["start"]), int(m["end"]), m["type"]) for m in mentions)


def _extract(ner: DiseaseNER, items: Sequence[Any]) -> dict[tuple[str, str], Any]:
    """Mine the given items in one batched call, keyed by the items' own ids.

    Keys MUST come from the items (``mine_with_cache`` calls this closure per chunk, so a
    local ``doc{i}`` counter would collide across chunks and silently keep only the last
    chunk's mentions - the off-by-one the first sweep reported as 89% "differing").
    """
    mined = ner.extract_batch([item[2] for item in items])
    # strict=True: `extract_batch` returns exactly one mention list per input item, so a length
    # mismatch means the batching broke. Silent truncation here would re-key mentions onto the
    # wrong (set_id, doc_id) pairs, which is the failure the docstring above exists to prevent.
    return {(item[0], item[1]): mentions for item, mentions in zip(items, mined, strict=True)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tabular", type=Path, required=True, help="Interim tabular dir holding spl_sections.parquet")
    parser.add_argument("--workdir", type=Path, required=True, help="Production workdir (owns the mention cache + model cache)")
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--chunk-words", type=int, default=None, help="Experiment window budget (default: production)")
    parser.add_argument("--batch-size", type=int, default=None, help="Experiment inference batch (default: production)")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default=None, help="Experiment compute dtype (default: production)")
    parser.add_argument(
        "--device", type=str, default=None, help="Device for BOTH sides' mines (e.g. cuda:2); one config per GPU lets a sweep run in parallel"
    )
    parser.add_argument("--out", type=Path, default=None, help="Write the diff report JSON here")
    parser.add_argument("--gate", action="store_true", help="Exit nonzero when any text differs (adoptability gate)")
    args = parser.parse_args()

    texts = _load_texts(args.tabular, args.sample, args.seed)
    if not texts:
        print(f"error: no minable section texts under {args.tabular}")
        return 2
    print(f"sampled {len(texts)} real section texts (seed={args.seed})")

    device_kwargs: dict[str, Any] = {"device": args.device} if args.device else {}
    baseline = DiseaseNER.for_contraindications(offline=False, workdir=args.workdir, **device_kwargs)
    experiment_kwargs: dict[str, Any] = dict(device_kwargs)
    if args.chunk_words is not None:
        experiment_kwargs["chunk_words"] = args.chunk_words
    if args.batch_size is not None:
        experiment_kwargs["inference_batch_size"] = args.batch_size
    if args.dtype is not None:
        experiment_kwargs["compute_dtype"] = args.dtype
    experiment = DiseaseNER.for_contraindications(offline=False, workdir=args.workdir, **experiment_kwargs)

    differing = 0
    baseline_missing = 0
    changed_texts: list[dict[str, Any]] = []
    with MentionCache(args.workdir) as cache:
        from dakp_pipeline.assertions.ner_dispatch import mine_with_cache

        items = [("ab", f"doc{i}", text) for i, text in enumerate(texts)]
        started = time.monotonic()
        baseline_mentions = mine_with_cache(items, baseline, lambda it: _extract(baseline, it), cache)
        baseline_seconds = time.monotonic() - started
        started = time.monotonic()
        experiment_mentions = mine_with_cache(
            items,
            experiment,
            lambda it: _extract(experiment, it),
            None,  # the experiment config must not write sweep entries into the production cache
        )
        experiment_seconds = time.monotonic() - started
        for i, text in enumerate(texts):
            base = _mention_multiset([m.to_dict() for m in baseline_mentions.get(("ab", f"doc{i}"), [])])
            exp = _mention_multiset([m.to_dict() for m in experiment_mentions.get(("ab", f"doc{i}"), [])])
            if not base:
                baseline_missing += 1
            if base != exp:
                differing += 1
                if len(changed_texts) < 25:
                    changed_texts.append(
                        {
                            "index": i,
                            "preview": text[:120],
                            "only_baseline": [list(m) for m in (base - exp).elements()][:10],
                            "only_experiment": [list(m) for m in (exp - base).elements()][:10],
                        }
                    )

    report = {
        "sample": len(texts),
        "seed": args.seed,
        "experiment": {"chunk_words": args.chunk_words, "batch_size": args.batch_size, "dtype": args.dtype, "device": args.device},
        # Wall time of each side. The baseline side is cache-served (CPU re-merge only), so the
        # experiment side's seconds are the GPU cost of the config under test; compare configs
        # against a control run (no experiment flags) on the same sample, not against baseline.
        "baseline_seconds": round(baseline_seconds, 1),
        "experiment_seconds": round(experiment_seconds, 1),
        "experiment_texts_per_s": round(len(texts) / experiment_seconds, 2) if experiment_seconds else None,
        "baseline_cache_served": len(texts) - baseline_missing,
        "texts_with_no_baseline_entry": baseline_missing,
        "differing": differing,
        "diff_clean": differing == 0,
        "examples": changed_texts,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    if args.gate:
        return 0 if differing == 0 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
