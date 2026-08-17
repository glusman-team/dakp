# NER backend benchmark & decision

Evidence base for settling on **one** composite NER backend (user directive: no pluggable
backend selector). Harness + gold fixture: `tests/eval/benchmark_ner.py`, `tests/eval/ner_gold.json`.

## Fixture

34 cases / 42 gold disease/phenotype spans: 20 DailyMed "Contraindications" sections
(LOINC `34070-3`) + 14 FAERS indication strings. Includes 3 **out-of-gazetteer** rare
diseases (porphyria, myasthenia gravis, pheochromocytoma) to probe model generalization
that a gazetteer alone cannot provide, 4 cases added 2026-08-10 that expose fixed
weaknesses: a population descriptor with no mention (`dailymed-childbearing`), a maximal-span
multiword disease (`dailymed-chf`), two OOV liver diseases (`dailymed-cirrhosis`), and an
OOV FAERS disease (`faers-aki`), and 3 cases added 2026-08-14 for the specificity merge: two
qualified diseases whose generic head is a gazetteer term (`dailymed-pulm-htn`,
`dailymed-portal-htn`) and one hedge-prefixed mention (`dailymed-recent-mi`). Annotation policy
is documented in the fixture — it now states explicitly that concept-changing qualifiers are
part of the span while temporal/evidential hedges are not.

## Results (strict span-level micro P/R/F1; a TP needs exact `(start, end, type)`)

Current (2026-08-14, after the specificity merge + abstention below):

| approach  | precision | recall | F1    | TP | FP | FN | notes                                   |
| --------- | --------- | ------ | ----- | -- | -- | -- | --------------------------------------- |
| gazetteer | 0.949     | 0.881  | 0.914 | 37 | 2  | 5  | deterministic; no heavy deps; FN = 3 OOV probes + the 2 qualified-disease cases |
| gliner    | 0.781     | 0.595  | 0.676 | 25 | 7  | 17 | zero-shot alone (`gliner_large-v2.5`); catches **all 3 OOV** exactly |
| **composite** | **1.000** | **1.000** | **1.000** | 42 | 0 | 0 | **settled backend** (gazetteer + GLiNER merge) |

The composite holds perfect precision *and* recall across the widened fixture: the two qualified
diseases and the hedge-prefixed mention all come out exactly right.

The **gazetteer** row moved down (was 1.000 / 0.923 / 0.960 on 31 cases) and that is expected,
not a regression. Offline mode was deliberately left unchanged (see "Specificity merge" below),
so on `pulmonary hypertension` it still emits its generic head `hypertension` — wrong offsets
against the new gold, hence one FP + one FN per qualified case. Offline remains the deterministic
zero-dependency baseline; specificity is a production-mode capability.

Previous (2026-08-10, 31 cases / 39 gold spans, before the specificity merge):

| approach  | precision | recall | F1    | TP | FP | FN |
| --------- | --------- | ------ | ----- | -- | -- | -- |
| gazetteer | 1.000     | 0.923  | 0.960 | 36 | 0  | 3  |
| gliner    | 0.759     | 0.564  | 0.647 | 22 | 7  | 17 |
| composite | 1.000     | 1.000  | 1.000 | 39 | 0  | 0  |

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
  — **SUPERSEDED 2026-08-14**, see "Specificity merge" below. Gazetteer growth does not
  generalize: it only ever covers the qualified terms someone thought to enumerate, and the
  cross product (pulmonary/portal/ocular/intracranial/systemic × hypertension, …) is unbounded.
  The over-extension objection was real but narrower than it looked — the offending prefixes are
  a closed class of hedges, so trimming them makes containment safe to accept.
* **Span-edge boundary trim:** all observed v2.5 spans are word-aligned; no evidence of
  punctuation-fringed boundaries to clean.
* **Gazetteer growth for the 3 OOV probes:** would silently convert the generalization probe
  into a gazetteer lookup; composite recall is already 1.0 via GLiNER for exactly those terms.

## Specificity merge + abstention (2026-08-14)

**Problem.** The backend prioritized the generic head over the specific term: on `severe
pulmonary hypertension` it emitted `hypertension`, because `hypertension` is a gazetteer term and
the merge dropped *any* GLiNER span overlapping a gazetteer span. `pulmonary hypertension`
(MONDO:0005149) is not `hypertension` (MONDO:0005044), so the row asserted a broader
contraindication than the label supports. Same shape for `portal hypertension`, `ocular
hypertension`, and every other qualified form of a gazetteer head.

**Fix — most specific span wins.** Gazetteer-wins-on-overlap is now decided by span structure:

1. **Hedge trim.** Leading tokens from a closed class (`_HEDGE_TOKENS`: determiners,
   prepositions, temporal/evidential hedges, population heads) come off the model span first.
   The polarity is deliberate — anything *not* listed counts as a clinical qualifier and is kept,
   so `severe` / `active` / `congestive` / `pulmonary` survive while `recent myocardial
   infarction` and `a history of peptic ulcer disease` collapse onto the gazetteer term. This is
   what defuses the over-extension objection that got the merge skipped in the first place. Left
   edge only: trimming the right edge would turn the population descriptor `pregnant women` into
   an emittable `pregnant`.
2. **Containment.** A trimmed model span that strictly contains exactly one gazetteer span
   supersedes it, taking the model's **boundary** and the gazetteer's **type** — GLiNER's
   documented weakness is disease↔phenotype confusion, not boundaries, so the two authorities are
   split along the axis each is actually good at. Equal spans and partial overlaps still go to the
   gazetteer. A span containing *several* gazetteer terms is a conjunction (`asthma or
   hypertension`), not a qualifier, and is discarded.
3. **Model-vs-model** overlaps resolve longest-first, ties on score then offsets (deterministic).

**Fix — abstention.** Thresholds are split: candidates are **generated** at
`DEFAULT_THRESHOLD = 0.35` and **accepted** at `DEFAULT_ACCEPT_THRESHOLD = 0.5`. Generating below
the acceptance floor is what makes the merge work at all — a specific span often scores lower
than its generic head, so the old generate-at-0.5 hid exactly the spans worth preferring. Spans
in the band between are visible to the merge but abstained on rather than asserted, logged at
DEBUG as `ner_abstain` with surface, score and reason. Crucially a superseded gazetteer span is
**not** resurrected when its challenger falls short: the backend returns nothing rather than the
term it now has reason to believe is too general. `extract` already contracted an empty list, so
no caller changed.

**Measured (real `gliner_large-v2.5`, CPU).** Composite 1.000 / 1.000 / 1.000 on the widened
34-case fixture. Hand checks:

* `Contraindicated in patients with severe pulmonary hypertension.` — `pulmonary
  hypertension`:disease 0.965 `gliner:extends` ✓ (was `hypertension`)
* `Contraindicated in patients with portal hypertension and cirrhosis.` — `portal
  hypertension`:disease 0.986 `gliner:extends` + `cirrhosis` at gazetteer precision ✓
* `Contraindicated in patients with a recent myocardial infarction.` — `myocardial
  infarction`:disease, hedge trimmed, gazetteer span intact ✓
* `Do not use in patients with a history of peptic ulcer disease.` — `peptic ulcer disease` ✓
* `Contraindicated in patients with asthma or hypertension.` — both gazetteer spans kept, the
  conjunction span discarded ✓
* `Contraindicated in patients with idiopathic pulmonary fibrosis.` — OOV, `gliner` 0.995 ✓
* `Contraindicated in women of childbearing potential.` — still no mention ✓
* Forcing `accept_threshold=0.99` on the 0.965 `pulmonary hypertension` span returns `[]` and
  logs `ner_abstain reason=superseded_unresolved` — it does **not** fall back to `hypertension` ✓

**Known boundary case.** `drug hypersensitivity` scores exactly 0.35 — right at the generation
threshold, so GLiNER filters it out and the gazetteer's `hypersensitivity` stands (unchanged from
before). Had it been generated it would have superseded and then abstained, since 0.35 is below
the acceptance floor. Noted rather than tuned around: moving the generation threshold to chase one
knife-edge span is the same fragility this file warns about for the 0.5→0.65 sweep.

**Offline mode deliberately unchanged.** It has no model, so it still returns the generic
gazetteer head. Specificity is a production-mode capability; offline stays the deterministic,
zero-dependency baseline that tests and offline runs rely on. The cost is visible and accepted in
the gazetteer row of the results table above.

## Acceptance floor lowered to the generation floor (2026-08-17)

Production mining showed the `[0.35, 0.5)` abstention band was dropping spans GLiNER had gotten
right: **0.35 is the lowest score at which the model is still accurate**. The acceptance floor
came down to meet the generation threshold — `DEFAULT_ACCEPT_THRESHOLD` 0.5 → 0.35, while
`DEFAULT_THRESHOLD` stays at 0.35 (never generate below the accuracy floor). By default the band
is now empty: every span GLiNER is willing to generate is asserted. The two-knob mechanism is
unchanged — raising `accept_threshold` re-opens the band and abstention (including
`superseded_unresolved`) exactly as before. Two recorded notes flip with it:

* The `drug hypersensitivity` knife-edge (score exactly 0.35, above) is still filtered by
  GLiNER's own generation filter, so that case is unchanged — but if it is ever generated it is
  now **accepted** rather than superseded-and-abstained, resolving that granularity limit toward
  recall.
* The "0.5 recall-safe default" in the v2.5-upgrade note is superseded: the recall-safe floor is
  0.35, still enforced at acceptance (now equal to generation).

**Measured (real `gliner_large-v2.5`, CPU, regenerated `benchmark_results.json`).** Composite
holds 1.000 / 1.000 / 1.000 on the 34-case fixture — the emptied band adds no FPs. GLiNER
standalone *improves* to P 0.794 / R 0.643 / F1 0.711 (was 0.781 / 0.595 / 0.676): +2 TP at
unchanged FP count, i.e. the formerly-abstained band spans that reach gold are correct.

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
  matcher. P 0.949 / F1 0.914 on the current fixture, zero heavy dependencies, fully
  deterministic — the best single laptop-safe option. Used by tests and offline runs. It returns
  the generic head for qualified diseases (`hypertension` for `pulmonary hypertension`); that
  granularity limit is inherent to a fixed vocabulary and is why production mode exists.
* **Production mode:** the same gazetteer anchors high-precision spans, and GLiNER zero-shot
  fills out-of-gazetteer gaps (non-overlapping GLiNER spans add recall; on overlap the most
  specific span wins — a containing model span supersedes the gazetteer's boundary but not its
  type; population descriptors dropped; hedge prefixes trimmed; hard-split fragments re-joined;
  sub-floor spans abstained on) → perfect recall at perfect precision. GLiNER is a core,
  lazy-imported dep.
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
`DEFAULT_THRESHOLD` stayed 0.5 as the recall-safe default for a safety KG. *(Superseded
2026-08-14: 0.5 is now the* acceptance *floor `DEFAULT_ACCEPT_THRESHOLD`, and generation dropped
to 0.35 so the specificity merge can see candidates it is allowed to prefer. The recall-safe
operating point is unchanged — only which stage enforces it.)*
