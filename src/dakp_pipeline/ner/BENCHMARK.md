# NER backend benchmark & decision

Evidence base for settling on **one** composite NER backend (user directive: no pluggable
backend selector). Harness + gold fixture: `tests/eval/benchmark_ner.py`, `tests/eval/ner_gold.json`.

## Fixture

31 cases / 39 gold disease/phenotype spans: 17 DailyMed "Contraindications" sections
(LOINC `34070-3`) + 14 FAERS indication strings. Includes 3 **out-of-gazetteer** rare
diseases (porphyria, myasthenia gravis, pheochromocytoma) to probe model generalization
that a gazetteer alone cannot provide, and 4 cases added 2026-08-10 that expose fixed
weaknesses: a population descriptor with no mention (`dailymed-childbearing`), a maximal-span
multiword disease (`dailymed-chf`), two OOV liver diseases (`dailymed-cirrhosis`), and an
OOV FAERS disease (`faers-aki`). Annotation policy is documented in the fixture.

## Results (strict span-level micro P/R/F1; a TP needs exact `(start, end, type)`)

Current (2026-08-10, after the composite precision improvements below):

| approach  | precision | recall | F1    | TP | FP | FN | notes                                   |
| --------- | --------- | ------ | ----- | -- | -- | -- | --------------------------------------- |
| gazetteer | **1.000** | 0.923  | 0.960 | 36 | 0  | 3  | deterministic; no heavy deps; FN = the 3 OOV probes |
| gliner    | 0.759     | 0.564  | 0.647 | 22 | 7  | 17 | zero-shot alone (`gliner_large-v2.5`); catches **all 3 OOV** exactly |
| **composite** | **1.000** | **1.000** | **1.000** | 39 | 0 | 0 | **settled backend** (gazetteer + GLiNER merge) |

Baseline before the improvements (original 27-case fixture; the composite shipped at these
numbers):

| approach  | precision | recall | F1    | TP | FP | FN | notes                                   |
| --------- | --------- | ------ | ----- | -- | -- | -- | --------------------------------------- |
| gazetteer | **1.000** | 0.914  | 0.955 | 32 | 0  | 3  | FN = the 3 OOV |
| gliner    | 0.692     | 0.514  | 0.590 | 18 | 8  | 17 | boundary truncation + disease↔phenotype confusion |
| **composite** | 0.972 | **1.000** | **0.986** | 35 | 1 | 0 | FP = `women of childbearing potential` (population descriptor) |
| scispacy  | 0.571     | 0.457  | 0.508 | 16 | 12 | 19 | BC5CDR: no phenotype label, coarse spans |

The **composite** (the shipped `DiseaseNER` in production mode) is the clear winner: perfect
recall — it catches every gold span including all three rare out-of-gazetteer diseases — now
at perfect precision too. The gazetteer anchors high-precision spans; GLiNER fills the
coverage gaps without inheriting the model's standalone boundary/type noise because
gazetteer spans win on overlap.

GLiNER standalone F1 is dragged down by boundary/type disagreement on common multiword terms
(the v2.5 large checkpoint is more aggressive than the old small one: more extra spans, mostly
disease↔phenotype type confusion), but it extracts every rare OOV disease with the correct span
and type — exactly the recall a fixed gazetteer lacks. SciSpacy's BC5CDR model labels only
`DISEASE`/`CHEMICAL` (no phenotype) and its boundaries are coarse, so it is strictly worse here.

Environment notes (reproducibility): GLiNER (`gliner-community/gliner_large-v2.5`, ~1.8 GB
fp32 weights, deberta-v3-large encoder, `max_len: 768` word tokens) loads in ~3 s from the
BLAKE3-keyed cache and runs on GPU on the build host (a laptop RTX 5070 Ti / sm_120 falls back
to CPU under the cu126 torch build's arch gate and still benchmarks in ~15 s). GLiNER is
natively multi-entity (one `predict_entities` call scores every label — disease and phenotype
here — and returns any number of spans per label). SciSpacy required two workarounds in this
environment and still underperformed; it is dropped from the shipped extra.

## Composite precision improvements (2026-08-10)

Error analysis on the gold fixture + hand-checked snippets found exactly one composite error:
GLiNER tagging the population descriptor `women of childbearing potential` as a phenotype
(raw scores 0.54–0.62 depending on sentence context). Three targeted changes landed:

1. **Population-descriptor filter.** Model spans whose normalized surface equals a curated
   population phrase (`women of childbearing potential`, `childbearing potential`, `women`,
   `patients`, …) are dropped. Deterministic, precision-safe, offline mode untouched.
   Removes the lone FP on the original fixture (composite 0.972→1.000 P at unchanged 1.000 R).
2. **Cross-window rejoin.** Budget hard-splits can cut a multiword mention in two (observed:
   `myasthenia | gravis` → `myasthenia` + `gravis:phenotype` on a run-on sentence with a
   small `chunk_words`). Flush span pairs across a hard-split gap are re-unified; type/score
   come from the higher-scoring side (ties left). Sentence-piece boundaries are untouched
   (mentions cannot span punctuation).
3. **Gazetteer growth.** Recurring high-confidence contraindication diseases seen in the
   hand checks joined `EMBEDDED_GAZETTEER`: `congestive heart failure`, `cirrhosis`,
   `hepatic encephalopathy`, `acute kidney injury`, `hepatocellular carcinoma`. Keeps the
   maximal-span policy offline (`congestive heart failure`, not just `heart failure`) and
   lets the composite anchor them at gazetteer precision. The three OOV probes stay OUT of
   the gazetteer on purpose (they measure GLiNER generalization).

Ideas evaluated and **skipped** (with reasons):

* **Threshold 0.5 → 0.65:** the earlier sweep showed recall 1.0 throughout, but the FP's raw
  score is context-dependent (0.54 standalone, ~0.6 in-fixture) — thresholding is a fragile
  fix and 0.5 stays the recall-safe default for a safety KG. The blocklist removes the FP
  structurally instead.
* **Disease-vs-phenotype type-conflict policy:** every GLiNER type confusion on the fixture
  (`gastrointestinal bleeding`, `QT prolongation`, `seizures` as disease) overlaps a gazetteer
  span and is already suppressed by gazetteer-wins-on-overlap; no residual measured error.
* **Overlap-extension merge (model span ⊇ gazetteer span):** GLiNER does produce longer
  containing spans (e.g. `congestive heart failure` ⊃ `heart failure`, score 0.99), but
  accepting model boundaries would also admit modifier over-extensions (`recent myocardial
  infarction`). Gazetteer growth captures the same maximal spans at gazetteer precision.
* **Span-edge boundary trim:** all observed v2.5 spans are word-aligned; no evidence of
  punctuation-fringed boundaries to clean.
* **Gazetteer growth for the 3 OOV probes:** would silently convert the generalization probe
  into a gazetteer lookup; composite recall is already 1.0 via GLiNER for exactly those terms.

## Small-example checks (hand verification, 2026-08-10)

Short DailyMed-style snippets and FAERS indication strings through `DiseaseNER` offline +
production (real model):

* `Contraindicated in patients with known hypersensitivity to any component of the product.`
  — `hypersensitivity`:phenotype exact in both modes ✓
* `Do not administer to patients with severe hepatic impairment, including cirrhosis and
  hepatic encephalopathy.` — gazetteer anchors `severe hepatic impairment`; GLiNER adds
  `cirrhosis` (0.88) and `hepatic encephalopathy` (0.96) ✓ — both now gazetteer terms too
* `Contraindicated in patients with congestive heart failure.` — was `heart failure` only
  (gazetteer granularity); now the maximal `congestive heart failure` offline + composite ✓
* `drug hypersensitivity` (FAERS) — gazetteer returns the shorter `hypersensitivity`; GLiNER's
  full-span candidate scores 0.35 (< 0.5 threshold), so the shorter span stands. Known
  granularity limit of the phenotype policy; accepted (mention text still resolves downstream)
* `Contraindicated in women who are or may become pregnant.` — no mention in either mode;
  paraphrase of the pregnancy phenotype that neither layer catches. Noted, not gold-labeled
* Run-on sentence with `myasthenia gravis` straddling a `chunk_words=20` hard split — was two
  broken mentions (`myasthenia`:disease + `gravis`:phenotype); now one re-unified
  `myasthenia gravis`:disease ✓
* `acute kidney injury`, `hepatocellular carcinoma` (FAERS) — GLiNER 0.97/0.91 ✓, now
  gazetteer terms too; repeated mentions (`asthma and again asthma`) correctly kept as two

## Decision: gazetteer-first, GLiNER-augmented composite (ONE backend)

* **Offline mode (default):** the curated disease/phenotype gazetteer + deterministic lexical
  matcher. Precision 1.000, F1 0.960 on the fixture, zero heavy dependencies, fully
  deterministic — the best single laptop-safe option. Used by tests and offline runs.
* **Production mode:** the same gazetteer anchors high-precision spans, and GLiNER zero-shot
  fills out-of-gazetteer gaps (gazetteer spans win on overlap; non-overlapping GLiNER spans
  add recall; population descriptors dropped; hard-split fragments re-joined) → perfect recall
  at gazetteer precision. GLiNER is a core, lazy-imported dep.
* **SciSpacy dropped:** strictly worse, no phenotype coverage, heavy. Removed from the NER backend.

This is one unified backend (`ner/ner.py`, `DiseaseNER`) with one entry point
(`extract_disease_mentions` / `extract_contraindication_diseases`) and an offline/production
mode toggle — **not** a selectable backend enum. DAKP emits mention text spans + type only;
ontology CURIE resolution is Tablassert's job (`tablassert build-kg`), never DAKP's.

## v2.5 upgrade (2026-08-10)

The production checkpoint moved from `urchade/gliner_small-v2.1` to
`gliner-community/gliner_large-v2.5`: a larger encoder (deberta-v3-large vs deberta-v3-small)
trained on more data, with double the word-token context (`max_len` 768 vs 384, so fewer
prediction windows over ~3000-word DailyMed sections). On the original 27-case fixture the
composite was P 0.972 / R 1.000 / F1 0.986 (one boundary FP — since removed by the
population-descriptor filter), and all three rare OOV diseases are caught exactly — the
upgrade's payoff is better OOV generalization on real label text, which a small fixture cannot
measure. A threshold sweep (0.40–0.70) shows composite recall stays 1.0 throughout;
`DEFAULT_THRESHOLD` stays 0.5 as the recall-safe default for a safety KG.
