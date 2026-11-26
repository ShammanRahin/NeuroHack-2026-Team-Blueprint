# Rulebook v1.3: what matters for us
Full text: `docs/NeuraHack_2026_Rulebook_v1.3.pdf`. This page is a working summary, not a replacement.

> **CONFLICT, read this first.** The PDF rulebook (v1.3) allows pseudo-labelling on `test.csv`, but the competition's own
> **Rules tab on Kaggle prohibits using test text for training in any form** (pseudo-labelling, self-training, label
> propagation, clustering-based labelling, domain adaptation). The Kaggle Rules tab is the authority here. **Never train or
> fit anything on test text**, including TF-IDF vocabularies, scalers and clustering. Fit on `sample.csv` only; transform test.
> Also: the Rules tab names the ZIP `NeuraHack_Preli_<Team_Name>.zip` while the PDF says `NeuraHack2026_<Team_Name>_<Team_Code>.zip`.
> Check the organizers' email and use what it says.

## Dates (BST, UTC+6)
| When | What |
|---|---|
| **26 Nov 6:00 PM** | Prelims start (Kaggle community competition, 24 h) |
| 27 Nov 6:00 PM | Prelims end |
| **27 Nov 10:00 PM** | Final ZIP due |
| 28 Nov 2:00 PM | Results; top 35 of top 60 advance after notebook review |
| 4 Dec | Final hackathon |

## What we must hand in
One ZIP named `NeuraHack2026_<TeamName>_<TeamCode>.zip` with **exactly four items, nothing else**:
1. `submission.csv`: must match one of our 2 selected Kaggle submissions
2. one `.ipynb` that regenerates the predictions
3. `.env`: every API key the notebook uses (dedicated key, spending limit, valid and funded until verification ends)
4. `README.md`: run order, approx runtime, API usage notes

The notebook **must start with a declaration section** (markdown): libraries + versions; every external dataset, model
and API with source/licence; method and validation strategy; seeds and reproduction steps.
`tools/build_submission.py` builds all of this from `src/` (see below).

## Allowed
Any AI tool and coding assistant; open-source pretrained models (XLM-R, mBERT, BanglaBERT, multilingual
embeddings); hosted LLM APIs for training, labelling, generation, augmentation, features and inference **if the calls
are made by code in the notebook and declared**; public external datasets if declared; (~~pseudo-labelling / domain adaptation on `test.csv`~~:
**prohibited by the Kaggle Rules tab, see the conflict note above**).

## Prohibited (disqualification)
- Recovering / looking up test labels any way except model predictions.
- **Hand labelling or hand editing test predictions, including pasting answers from a chat window into a file.**
  Every prediction and every piece of generated training data must come from code in the notebook.
- **Any training, fitting or labelling that uses test text** (Kaggle Rules tab, section 5).
- Hardcoded predictions or a manually created label file.
- Sharing code/data/predictions/keys privately with other teams.
- More than one account per person, or more than one team per person.
- **Any undeclared external dataset, model or API.** Declare everything in `docs/DECLARATION.md`.
- Hardcoded API keys in the notebook (read from the environment, never paste into cells).

## Scoring and selection
- Macro F1, 10 submissions per team per day, select **2 finals** before the deadline.
- Public LB = 30% of test; **private (70%) decides**. If we don't select, Kaggle picks the best 2 *public* scores.
  So **choose finals by our own validation, not the public score.**
- The test set is shifted (new phrasings, brands/operators, different class mix). Organizers may also run our
  notebook on extra unseen data for the top 60, so the pipeline must work on new text with no manual changes.
- Verification: clean Kaggle/Colab re-run with internet using our `.env`; reproduced private score within 0.01 of
  ours is accepted. LLM non-determinism: use temperature 0 and fixed seeds.
- Notebooks are inspected for undeclared resources, hardcoded predictions, hand-made label files, hardcoded keys.

## What this means for each role
- **A (Mahdi), data:** build **group-aware CV**: group near-duplicate/similar messages so they never straddle train and
  validation (e.g. cluster by normalised text or char-ngram similarity, then `StratifiedGroupKFold`). Random folds
  over-estimate the score on shifted data. Also make `validate_submission` strict: ids exactly once, `label` an integer 0-5.
- **B (Shanto), features:** multilingual embeddings / LLM-derived features are allowed, but every external model or API
  goes in `docs/DECLARATION.md`. Keep features deterministic; cache LLM calls.
- **C (Adibur), models:** prefer approaches that survive shift (ensembles, calibrated probabilities, class-prior
  handling for macro-F1). Keep `predict` usable on brand-new text. Fix seeds and temperature 0 for any LLM call.
- **Everyone:** never paste LLM output into a data or label file by hand: only code in the pipeline may call APIs.
  Keep `.env` out of git (it is in `.gitignore`).

## Final package (see `tools/build_submission.py`)
```
python tools/build_submission.py --exp <final_config> --team-name <Name> --team-code <NHxxx> \
       --submission <path to the selected Kaggle submission.csv> --env <path to .env>
```
It writes `dist/NeuraHack2026_<Name>_<Code>.zip` and refuses to finish if the ZIP isn't exactly the four items, the
notebook contains a key from `.env`, or the CSV is malformed.
