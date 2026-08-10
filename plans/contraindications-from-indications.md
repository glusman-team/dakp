# Plan: Mine contraindications from indication sections via GLiNER

## Context

GLiNER currently only mines disease/phenotype mentions from **Contraindication** sections
(LOINC `34070-3`). The **Indications and Usage** sections (LOINC `34067-9`) are processed
exclusively by the lexical dictionary matcher (`match_diseases`) in the approved-treats shaper.

Real DailyMed labels sometimes embed contraindication statements inside the indications section
(e.g., *"Drug X is indicated for hypertension. It is contraindicated in patients with severe
hepatic impairment."*). These are currently missed entirely.

**Goal:** Run GLiNER over indication section text too — but only the contraindication-relevant
parts — so embedded contraindications are captured. Use a two-pass design with parallel GPU
dispatch (2 GPUs per pass).

## Empirical label evaluation (run on this machine)

Tested 9 candidate GLiNER label phrasings on 6 example sentences (pure contraindications,
pure indications, embedded contraindications, warning-style, combined section text):

| Approach | False positives | Recall |
|----------|----------------|--------|
| `["disease", "phenotype"]` (current) | **High** — catches all indications too | Good |
| `["contraindicated condition"]` | Zero | **Very low** — misses most contraindications |
| `["contraindication"]` | Zero | **Zero** — finds nothing |
| `["condition contraindicated for"]` | High | Low-medium |
| All other contextual labels | Zero or low FP | **Very low** |

**Conclusion:** The small model (`urchade/gliner_small-v2.1`) cannot do semantic role
detection — contraindication-specific labels have near-zero recall. The model is a span
extractor, not a relation classifier.

### Winning approach: sentence-level keyword filter + existing labels

Instead of changing GLiNER labels, **filter sentences** before extraction:

1. Split combined text into sentences
2. Keep only sentences containing contraindication keywords (`contraindicated`, `should not
   be used`, `not recommended`, `avoid`, etc.)
3. Run GLiNER with existing `["disease", "phenotype"]` labels on the filtered text

**Results (threshold=0.5):**

| Test case | Contraindications found | False positives |
|-----------|------------------------|-----------------|
| Pure contraindication | `active liver disease` ✓ | 0 |
| Pure indication | (nothing) ✓ | 0 |
| Embedded in indication | `severe hepatic impairment` ✓ | 0 |
| "Should not be used" | `severe renal impairment`, `end stage renal disease` ✓ | 0 |
| Combined section text | `asthma` ✓ | 0 |
| Warning-style | `narrow angle glaucoma`, `heart failure` ✓ | 0 |

**Known limitation:** Double negatives (*"not contraindicated in patients with X"*) are
matched by the keyword filter — extremely rare in real DailyMed labels, acceptable trade-off
for recall.

## Approach

### 1. Sentence-level keyword filter (`contraindications.py`)

Add a configurable keyword regex + sentence splitter:

```python
#: Regex matching contraindication-context sentence keywords. Configurable via
#: ``ctx.params["contraindication_keywords"]`` to tune recall/precision per deployment.
DEFAULT_CONTRA_KEYWORDS = re.compile(
    r"\b(contraindicat\w*|should\s+not\s+be\s+used|must\s+not\s+(?:be\s+)?used|"
    r"do\s+not\s+use|not\s+recommended|avoid\s+(?:use\s+)?in|prohibit\w*|"
    r"must\s+avoid|use\s+is\s+contraindicat\w*|not\s+for\s+use\s+in)\b",
    re.IGNORECASE,
)


def _contraindication_sentences(text: str, keywords: re.Pattern) -> str:
    """Return only the contraindication-context sentences from ``text`` (joined)."""
    sentences = _split_sentences(text)
    filtered = [s for s in sentences if keywords.search(s)]
    return " ".join(filtered)
```

### 2. Two-pass extraction with parallel GPU dispatch

**Pass 1 (unchanged):** Contraindication section text (LOINC 34070-3) → existing
`ner.extract()` (gazetteer + GLiNER with disease/phenotype labels). All text is assumed
contraindication-relevant. **No behavior change.**

**Pass 2 (new):** Indication section text (LOINC 34067-9) → keyword filter → existing
`ner.extract()` on contraindication-context sentences only. Only sets whose indication text
contains ≥1 contraindication-context sentence produce Pass 2 work items (pre-filtered in the
main process to avoid dispatching no-op work to GPUs).

**Merge:** Union of Pass 1 + Pass 2 mentions, deduplicated by `(subject, object_text)` via
the existing `_accumulate` aggregation.

### 3. Parallel GPU dispatch (2+2 split)

Current: all 4 GPUs (`cuda:0`–`cuda:3`) on Pass 1 sequentially.

New: split the GPU list in half — first 2 GPUs on Pass 1, last 2 on Pass 2, dispatched
**concurrently** via a single `ProcessPoolExecutor`:

```python
def _mine_two_passes_multi_gpu(work_items_p1, work_items_p2, ner, devices):
    mid = len(devices) // 2
    dev_p1, dev_p2 = devices[:mid], devices[mid:]
    shards_p1 = _shard_by_text_length(work_items_p1, min(len(dev_p1), len(work_items_p1)))
    shards_p2 = _shard_by_text_length(work_items_p2, min(len(dev_p2), len(work_items_p2) or 1))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(shards_p1) + len(shards_p2), mp_context=ctx) as pool:
        futures = [pool.submit(_mine_shard, s, ner._config(), dev_p1[i]) for i, s in enumerate(shards_p1)] + [
            pool.submit(_mine_shard, s, ner._config(), dev_p2[i]) for i, s in enumerate(shards_p2)
        ]
        results = {}
        for fut in futures:
            for set_id, doc_id, mentions in fut.result():
                results.setdefault((set_id, doc_id), []).extend(mentions)
        return results
```

If Pass 2 has no work items (no indication sections with contraindication keywords), all 4
GPUs fall back to Pass 1 (current behavior).

### 4. Work item structure

| Pass | Source | Work item shape | Pre-filter |
|------|--------|----------------|------------|
| 1 | `evidence.contraindication_docs` | `(set_id, doc_id, text)` — unchanged | ≥1 active ingredient |
| 2 | `evidence.indication_docs` | `(set_id, doc_id, filtered_text)` | ≥1 active ingredient AND keyword filter yields non-empty text |

The filtered text in Pass 2 replaces the full indication section text — GLiNER only sees
contraindication-context sentences. Provenance (`supporting_spl_documents`) records the
indication-section doc_id (`SETID#34067-9`) so contraindications found in indication text are
traceable to their source section.

### 5. Configurable keywords

The keyword regex is a module-level constant (`DEFAULT_CONTRA_KEYWORDS`) overridable via
`ctx.params["contraindication_keywords"]` (accepts a compiled pattern or a raw string). This
allows empirical tuning per deployment without code changes. The `_resolve_keywords(ctx)` helper
reads the param or falls back to the default.

## Files to modify

| File | Change |
|------|--------|
| `src/dakp_pipeline/assertions/contraindications.py` | Keyword filter, Pass 2 work items, two-pass parallel dispatch, configurable keywords |
| `tests/unit/test_assertions_contraindications.py` | New fixture with embedded contraindication in indication section; test that it's mined |
| `tests/unit/test_assertions_contraindications_edge.py` | Monkeypatched fake GLiNER mock for Pass 2; parallel dispatch tests; no-regression tests |

**No changes needed to:** `ner.py` (existing `ner.extract()` + gazetteer reused as-is), `evidence.py`
(`indication_docs` already indexed).

## Reuse

- `DiseaseNER.extract()` (`ner.py`) — gazetteer + GLiNER, unchanged; called on filtered text
- `_shard_by_text_length` / `_mine_shard` / `_mine_multi_gpu` (`contraindications.py`) — extended for 2-pass dispatch
- `_accumulate` / `_finalize_row` (`contraindications.py`) — aggregation unchanged (handles more work items)
- `build_dailymed_evidence` (`evidence.py`) — already indexes `indication_docs` and `contraindication_docs`
- `_install_fake_gliner` pattern (`test_ner_edge.py`) — template for the new monkeypatched mock

## Steps

- [ ] Add `DEFAULT_CONTRA_KEYWORDS` regex + `_contraindication_sentences()` helper to `contraindications.py`
- [ ] Add `_split_sentences()` — simple sentence boundary splitter (period/semicolon + whitespace)
- [ ] Add `_resolve_keywords(ctx)` — read `ctx.params["contraindication_keywords"]` or use default
- [ ] Update `build_contraindication_rows` to collect Pass 2 work items from `evidence.indication_docs` (pre-filtered)
- [ ] Add `_mine_two_passes_multi_gpu()` — split devices 2+2, dispatch both passes concurrently
- [ ] Update `ContraindicationsShaper.transform` to pass keywords to `build_contraindication_rows`
- [ ] Extend `_mine_shard` work item handling for the new `(set_id, doc_id, filtered_text)` shape
- [ ] Update fallback: when no Pass 2 work items, all GPUs → Pass 1 (current behavior)
- [ ] Add test fixture: DailyMed XML with an indication section containing embedded contraindication text
- [ ] Add monkeypatched fake GLiNER mock that returns disease entities (test that sentence filter prevents false positives)
- [ ] Add test: contraindication found in indication section → assertion with `SETID#34067-9` provenance
- [ ] Add test: indication-only diseases (no contraindication keywords) → NOT mined as contraindications
- [ ] Add test: no-regression — sets with only contraindication sections produce identical output
- [ ] Add test: parallel dispatch (2+2 split) produces same results as sequential

## Verification

1. **Unit tests:** `uv run pytest tests/unit/test_assertions_contraindications.py tests/unit/test_assertions_contraindications_edge.py tests/unit/test_ner.py tests/unit/test_ner_edge.py -v`
2. **No regression:** Sets with only a contraindication section (no indication section, or indication with no contraindication keywords) produce identical output to current behavior.
3. **New coverage:** A set whose indication section contains *"contraindicated in patients with X"* yields a contraindication assertion for X, with `supporting_spl_documents` = `SETID#34067-9`.
4. **No false positives:** A set whose indication section lists only indications (*"indicated for Y"*) does NOT produce contraindication assertions for Y.
5. **Determinism:** Run twice, confirm identical `contraindication_assertions.tsv`.
6. **Production GPU run:** On the 4×P100 host, confirm all 4 GPUs active (2 on Pass 1, 2 on Pass 2) and that Pass 2 yields additional contraindications beyond Pass 1 alone.
