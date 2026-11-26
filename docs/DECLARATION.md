<!-- Team edits this. build_submission.py puts it at the top of the notebook and REFUSES to build while "TODO" remains.
     Rulebook s.6: every external dataset, pretrained model and API must be listed, or it is a violation. -->
## Declaration

### 2. External datasets, pretrained models and APIs
| Resource | Type (dataset / model / API) | Source | Licence / provider | Used for |
|---|---|---|---|---|
| intfloat/multilingual-e5-small (if `sbert` feature used) | pretrained model | Hugging Face | MIT | sentence embeddings |
| sentence-transformers, scikit-learn, pandas, numpy, scipy, torch | libraries | PyPI | Apache-2.0 / BSD | pipeline |
| MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (if `nli` feature used) | pretrained model | Hugging Face | MIT | zero-shot NLI scores vs labelling-guide hypotheses |
| avro.py | library | PyPI | MIT OR Apache-2.0 | Bangla <-> Banglish transliteration of real training rows |
| Hosted LLM via OpenAI-compatible API (if `llm` feature used): provider + model = TODO | API | TODO | TODO | zero-shot label/techniques/action judgements as features; key in .env; usage logged to llm_usage.csv |
| TODO: any other model / API you add (e.g. bge-m3: MIT; check the licence on the model card of community models) | TODO | TODO | TODO | TODO |

(Write "None" if nothing external is used. Include LLM APIs used for labelling, generation, augmentation, features or inference.)

### 3. Method and validation strategy
TODO: describe the method in a few sentences, and how validation is done (group-aware folds, why, metric).

### 4. Reproduction notes
TODO: run order, approximate runtime, LLM temperature / non-determinism notes. Random seeds are printed in the config section below.
