# NER backend benchmark & decision

Evidence base for settling on **one** composite NER backend (user directive: no pluggable
backend selector). Harness + gold fixture: `tests/eval/benchmark_ner.py`, `tests/eval/ner_gold.json`.

## Fixture

27 cases / 35 gold disease/phenotype spans: 14 DailyMed "Contraindications" sections
(LOINC `34070-3`) + 13 FAERS indication strings. Includes 3 **out-of-gazetteer** rare
diseases (porphyria, myasthenia gravis, pheochromocytoma) to probe model generalization
that a gazetteer alone cannot provide. Annotation policy is documented in the fixture.

## Results (strict span-level micro P/R/F1; a TP needs exact `(start, end, type)`)

| approach  | precision | recall | F1    | TP | FP | FN | notes                                   |
| --------- | --------- | ------ | ----- | -- | -- | -- | --------------------------------------- |
| gazetteer | **1.000** | 0.914  | 0.955 | 32 | 0  | 3  | deterministic; no heavy deps; FN = the 3 OOV |
| gliner    | 0.864     | 0.543  | 0.667 | 19 | 3  | 16 | zero-shot alone; catches **all 3 OOV** exactly |
| **composite** | 0.972 | **1.000** | **0.986** | 35 | 1 | 0 | **settled backend** (gazetteer + GLiNER merge) |
| scispacy  | 0.571     | 0.457  | 0.508 | 16 | 12 | 19 | BC5CDR: no phenotype label, coarse spans |

The **composite** (the shipped `DiseaseNER` in production mode) is the clear winner: perfect
recall — it catches every gold span including all three rare out-of-gazetteer diseases — at
near-perfect precision (one false positive). The gazetteer anchors high-precision spans; GLiNER
fills the coverage gaps without inheriting the model's standalone boundary/type noise because
gazetteer spans win on overlap.

GLiNER standalone F1 is dragged down by boundary/type disagreement on common multiword
terms, but it extracts every rare OOV disease with the correct span and type — exactly the
recall a fixed gazetteer lacks. SciSpacy's BC5CDR model labels only `DISEASE`/`CHEMICAL`
(no phenotype) and its boundaries are coarse, so it is strictly worse here.

Environment notes (reproducibility): GLiNER (`urchade/gliner_small-v2.1`) loads in ~21s on
CPU and is fully laptop-safe. SciSpacy required two workarounds in this environment — the
resolved `typer` no longer pulls `click` (which `spacy` hard-imports) and `spacy download`
has no 3.8-compatible `en_ner_bc5cdr_md` (installed the v0.5.4 S3 wheel directly) — and still
underperformed; it is dropped from the shipped extra.

## Decision: gazetteer-first, GLiNER-augmented composite (ONE backend)

* **Offline mode (default):** the curated disease/phenotype gazetteer + deterministic lexical
  matcher. Precision 1.000, F1 0.955 on the fixture, zero heavy dependencies, fully
  deterministic — the best single laptop-safe option. Used by tests and the mock pipeline.
* **Production mode:** the same gazetteer anchors high-precision spans, and GLiNER zero-shot
  fills out-of-gazetteer gaps (gazetteer spans win on overlap; non-overlapping GLiNER spans
  add recall) → near-perfect recall at gazetteer precision. Needs the `[ner]` extra.
* **SciSpacy dropped:** strictly worse, no phenotype coverage, heavy. Removed from `[ner]`.

This is one unified backend (`ner/ner.py`, `DiseaseNER`) with one entry point
(`extract_disease_mentions` / `extract_contraindication_diseases`) and an offline/production
mode toggle — **not** a selectable backend enum. DAKP emits mention text spans + type only;
ontology CURIE resolution is Tablassert's job (`tablassert build-kg`), never DAKP's.
