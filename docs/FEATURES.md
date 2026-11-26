# Feature engineering (Member B): research, design, measured effect

## Constraints that shape everything
- 86 labelled rows, 500,000 test rows; hidden subsets for domain shift, novel scenarios and corrupted text (Data tab).
- **No use of test text for training or fitting** (Kaggle Rules tab s.5) -> every feature is fit on training rows only,
  and the runner fits features **inside each CV fold** (supervised features cannot leak validation labels).
- Inference must be cheap: everything below runs on 500k rows in minutes.

## Feature groups (`src/features/`)
| Group | File | What / why |
|---|---|---|
| De-obfuscation | `src/data/normalize.py` | NFKC + strip all format chars (zero-width, bidi, soft hyphen) + Cyrillic/Greek homoglyph table + repeat squashing. Published defences show each step repairs its own attack type and normalisation restores clean scores. ~11 s / 500k rows. |
| Char TF-IDF | `tfidf.py` | `char_wb` 2-5-grams: robust to typos, Banglish spelling variation, mixed scripts. |
| Word TF-IDF | `tfidf_word.py` | word unigrams for content words. |
| Hand-crafted tactics | `handcrafted.py` | 66 features from the labelling guide: credential / prize / job / parcel / promo / urgency / payment / link / install / contact / organisation lexicons (EN, Banglish, Bangla), despaced matching for long keywords, URL / short-code / phone / amount patterns, conversation structure (turns, speaker markers, requests that appear only in later turns), noise-only and gibberish detection, hard-legitimate interactions (OTP + do-not-share warning, messages *about* scams). ~1.6 min / 500k rows. |
| Sentence embeddings | `embeddings.py` | multilingual encoder, fp16, length-sorted batches, disk cache per 50k test chunk. Default `intfloat/multilingual-e5-small` (best quality/speed for Bangla in benchmarks). Swap in `kazalbrur/bangla-embed-e5-small-banglish` (Banglish-aware) or `BAAI/bge-m3` (strongest, ~5x slower). |
| SetFit fine-tuning | `embeddings.py` (`finetune_steps`) | contrastive pair fine-tuning of the encoder per fold (SetFit body), then the classifier head. |
| ~~Synthetic data~~ | removed | ~~ seeded templates written from the labelling guide: every class x EN/Banglish/Bangla, invented brands/identities, priority-order cases, hard negatives, and register wrappers (multi-turn chats with late reveals, group chats, call-style markers, formal emails, notices). |
| Augmentation | `augment.py` | the corruption families listed on the Data tab (insert/delete/substitute/repeat, spacing, caps, emoji, homoglyphs, zero-width, sensitive-word misspellings, noise strings) + noise-only class-0 rows. Training rows only; a corrupted copy of validation rows gives the local `robust` score. |

## Measured (grouped CV on 50 labelled sample rows; local composite = renormalised A/B/C/robust/calibration)
| Setup | composite | A | B | C | robust |
|---|---|---|---|---|---|
| char TF-IDF only | 0.420 | 0.451 | 0.290 | 0.371 | 0.379 |
| + word TF-IDF + hand-crafted | 0.810 | 0.893 | 0.731 | 0.565 | 0.800 |
| + synthetic templates | 0.892 | 0.918 | 0.899 | 0.806 | 0.837 |
| + synthetic + augmentation (`preli_nogpu`) | **0.905** | **0.935** | 0.861 | **0.844** | **0.889** |
| trained on synthetic ONLY, scored on real rows | - | 0.874 | 0.870 | 0.710 | - |
The last row is the best evidence for the hidden "novel scenario" subset: templates alone reach 0.87 macro-F1 on real
messages they never saw. Embedding numbers need the real encoder (run `preli_full` on Colab GPU) - not measurable here.

## Experiments to run next (in order)
1. `preli_full` vs `preli_nogpu` on the real sample (does the encoder help?).
2. Encoder swap: `features.3.params.model=kazalbrur/bangla-embed-e5-small-banglish`, then `BAAI/bge-m3`.
3. SetFit: `features.3.params.finetune_steps=40` (set `cv.n_repeats=1` while exploring).
4. `synthetic.n_per_class` 100 / 200 / 400; `augment.copies` 0 / 1 / 2.
5. Optional, costs money: LLM-generated training data via an API from the notebook (declare it, key in `.env`).

## References
- Bangla encoders benchmark (18 encoders; e5-small best quality/efficiency): https://dspace.bracu.ac.bd/bitstreams/aef74f93-ebd2-4880-8b64-dc180bbad433/download
- Bangla-Embed-E5-Banglish: https://huggingface.co/kazalbrur/bangla-embed-e5-small-banglish
- BanglaBERT: https://arxiv.org/abs/2101.00204 ; BanglaTLit (romanized Bangla): https://aclanthology.org/2024.findings-emnlp.859
- Transliteration robustness in Bangla (BLP-2025): https://aclanthology.org/2025.banglalp-1.27
- SetFit: https://huggingface.co/docs/setfit/conceptual_guides/setfit
- Adversarial text normalisation: https://arxiv.org/abs/2206.04137 ; homoglyph/zero-width sanitisation: https://arxiv.org/pdf/2610.18151
- Smishing for Bengali speakers (LLM synthetic + translated data): https://researchportal.murdoch.edu.au/esploro/outputs/graduate/A-Multilingual-Fine-Tuned-LLM-Framework-for/991005909771807891
- EmbeddingGemma: https://arxiv.org/abs/2509.20354

## Update (27 Nov, after public LB 0.739): no synthetic data
Team decision: no generated messages. Templates removed. Added instead:
- `zeroshot.py` `nli`: multilingual NLI (mDeBERTa-v3 xnli, 26 languages incl. Bengali) entailment scores for 16 hypotheses
  from the labelling guide - needs no training rows, targets novel scenarios.
- `embeddings.py` `labeldesc`: cosine similarity to 13 guide descriptions (class / technique / action).
- `llm.py` `llm`: optional zero-shot LLM judgements (OpenAI-compatible API, key in .env, cached, usage logged).
- augmentation of REAL rows only: corruptions, Bangla<->Banglish transliteration (avro.py), conversation views.
- validation: shift holdouts (Bangla-script / Banglish / conversations held out) reported as `worst`; temperature
  scaling for Task D fitted on out-of-fold predictions.
- normaliser fix: repeated digits are no longer squashed ("50000" stayed "500" before).

Local (50 sample rows, CPU features only): real rows only 0.805; + corruptions 0.801; + translit/views 0.793 - all
within noise, so the GPU zero-shot features are what must move the score. Shift `worst` 0.56-0.58.
