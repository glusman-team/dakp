# Fix: GLiNER silently truncates long contraindication sections

## Context

`DiseaseNER._merge_model_spans` (`src/dakp_pipeline/ner/ner.py:196`) feeds the **entire**
DailyMed contraindication section to `model.predict_entities(text, ...)`. GLiNER's processor
hard-truncates the input at `config.max_len` **word tokens** (whitespace-split words), emitting
only a `UserWarning`:

- Verified in installed `gliner==0.2.28`: `data_processing/processor.py:517-520` / `834-837` —
  `tokens = tokens[:max_len]` + `warnings.warn(...)`.
- The shipped checkpoint (`urchade/gliner_small-v2.1`, cached config) sets `max_len: 384`.
- Real section lengths (359 sections sampled from `tmp/airflow-run` raw SPL): median 48 words,
  p95 229, max 415 — and sections up to **~3000 words** exist in the full corpus. Everything
  past token 384 is silently cut: those disease mentions never reach the
  `contraindication_assertions` table.
- `predict_entities` in 0.2.28 has **no** built-in long-text chunking.
- Offsets are safe to remap: GLiNER computes entity `start`/`end` as char offsets **into the exact
  string passed** (`model.py:2173` — `"text": valid_texts[i][start:end]`), so predicting on exact
  substrings of the original text and shifting by the chunk start preserves the
  `mention.text == text[start:end]` invariant.
- **Token-count gotcha:** the budget is *not* `len(text.split())`. GLiNER's whitespace splitter
  (`data_processing/tokenizer.py` `WhitespaceTokenSplitter`) uses
  `re.compile(r"\w+(?:[-_]\w+)*|\S")` — every punctuation glyph is its own token, so GLiNER token
  count ≥ whitespace word count. Window sizing must count with that exact pattern or a window can
  still exceed `max_len` and get truncated.

## Approach

Chunk inside `_merge_model_spans` — the one place GLiNER is called. No caller/API changes.

1. **Token budget**: `chunk_words` constructor kwarg (default `None` = read
   `model.config.max_len` at predict time, fallback `384`). Reading the model config keeps the
   backend correct if the checkpoint ever changes; the kwarg makes tests deterministic.
2. **Sentence-aware greedy packing** (new module-level helper, e.g. `_windows(text, budget)`):
   - Split `text` into sentence-ish pieces with a regex (`(?<=[.!?;])\s+` and `\n+`) using
     `finditer` so pieces + gaps tile the text exactly — every window is an **exact substring**
     `text[a:b]` (offsets remap by just adding `a`).
   - Count window size in **GLiNER tokens** with the splitter's exact pattern
     `_GLINER_TOKEN = re.compile(r"\w+(?:[-_]\w+)*|\S")` (comment points at gliner's
     `WhitespaceTokenSplitter`), not `str.split()`.
   - Greedily pack pieces into windows of ≤ budget tokens.
   - A single piece longer than the budget (one giant sentence) is hard-split at token-match
     boundaries into budget-sized windows (still exact substrings).
   - Result: `list[(start, chunk_text)]`; short texts yield exactly one window == the full text,
     so existing behavior/tests are unchanged for everything ≤ budget. A 3000-word section yields
     ~8-10 windows — linear scaling, no special-casing.
3. **Per-window predict + remap**: call `model.predict_entities(chunk, ...)` per window; shift
   each entity's `start`/`end` by the window start before the existing type-filter /
   gazetteer-overlap merge (which already works in full-text coordinates). No overlap between
   windows → no duplicate-span dedup needed. (Future speedup if runtime ever matters: pass all
   windows of a section as one batched `model.inference([...])` call — not in this change.)
4. Docs: note chunking in the `DiseaseNER` docstring + one line in `ner/README.md`.

## Files to modify

- `src/dakp_pipeline/ner/ner.py` — `_windows` helper, `_merge_model_spans` rewrite, `chunk_words`
  kwarg, docstrings.
- `tests/unit/test_ner_edge.py` — new tests (reuse the `_FakeGLiNER` pattern already there).
- `src/dakp_pipeline/ner/README.md` — one-line behavior note.

## Reuse

- Existing fake-GLiNER test harness in `tests/unit/test_ner_edge.py` (`_FakeGLiNER`,
  `_install_fake_gliner`) — extend `_FakeGLiNERModel.calls` assertions for per-chunk calls.
- Existing merge logic (`_overlaps_any`, `canonical_type`, `Mention`) is reused unchanged.
- No new dependencies (regex-based sentence split; the repo has no splitter utility to reuse).

## Steps

- [x] Add `_windows(text, budget)` helper + unit-level coverage.
- [x] Add `chunk_words: int | None = None` to `DiseaseNER.__init__`; resolve budget from
      `model.config.max_len` when unset.
- [x] Rewrite `_merge_model_spans` to predict per window and remap offsets.
- [x] Tests: long text (~3000-word scale simulated with a small `chunk_words` budget) → multiple
      calls, every window ≤ budget GLiNER tokens, windows tile the text exactly, remapped offsets
      satisfy `text[m.start:m.end] == m.text`; punctuation-heavy text counted with the GLiNER
      pattern (not `str.split()`); short text → single call with full text (existing tests keep
      passing); oversize single sentence hard-split; gazetteer-wins still holds across windows.
- [x] Doc updates (`ner.py` docstring, `ner/README.md`).

## Verification

- `uv run pytest tests/unit/test_ner.py tests/unit/test_ner_edge.py -q` — all green, no heavy
  deps imported.
- `uv run pytest tests/unit -q` — no regressions in assertion shapers.
- Fake-model coverage is the agreed verification scope (no real-weight smoke run).
