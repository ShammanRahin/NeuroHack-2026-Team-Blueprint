# NeuroHack 2026 Team Blueprint

Three people, three folders, one command. Nobody edits anybody else's files.

| Who | Owns | Contract |
|---|---|---|
| A: data validation (@Berserker268, Mahdi) | `src/data/` | `load_clean(cfg) -> Bundle`, `folds()`, `validate_submission()` |
| B: feature engineering (@ShammanRahin, Shanto) | `src/features/` | `@feature("name")` returns a matrix, **train rows first, then test rows** |
| C: model architecture (@adiburrahman2004, Adibur) | `src/models/` | `@model("name")` class with `fit(X, y)` / `predict_proba(X)` |
| everyone (by PR) | `src/common/`, `configs/` | `contracts.py` documents the interfaces |

## Daily loop (any machine)
```
python tests/smoke_test.py                 # 5 s, must pass before every push
python -m src.run exp=baseline debug=true  # seconds, real data, tiny subsample
python -m src.run exp=baseline             # full run: CV score + validated submission + log row
python -m src.run exp=baseline model.params.C=3   # override any YAML value from the CLI
```
Full setup per role: [docs/SETUP.md](docs/SETUP.md).

**Shared Drive folder (cache + outputs, edit access, delete after the contest):** https://drive.google.com/drive/folders/1hB8kMuau_CtBpbR3c6CQuM_iA5o9vCcx?usp=sharing
After opening it: Organize → Add shortcut → My Drive, or Colab can't see it.

Add a feature: new file in `src/features/` + one line under `features:` in a YAML. Add a model: new file in
`src/models/` + `model:` in a YAML. No registry to edit, so no merge conflicts.

## Why it stays conflict-free
- Own folder only (`CODEOWNERS` enforces review by the owner). Shared files change by PR with all three OK.
- Every experiment is a YAML in `configs/`. Copy one, give it a new `name:`, edit.
- Same folds for everyone (fixed seed), same metric (`src/common/metrics.py`), so scores are comparable.
- `results/log.csv` is append-only with `merge=union`: two people logging at once never conflict.
- Cache key includes the feature's **source code**, so editing a feature can never serve stale results.
- Each run saves `oof.npy` + `test_proba.npy` in `outputs/<name>/`: blend members' models without retraining.
- Notebooks hold no logic (just setup + `python -m src.run`), so no notebook merge conflicts.
- Branch per person, PR into `main`, CI runs the smoke test. `main` always runs.

## Colab
Open `notebooks/colab.ipynb`, edit the 4 settings, run. It clones your branch, installs, downloads the
competition data with the Kaggle API, and keeps cache on Drive. Push your branch from the last cell.

## Kaggle notebook
- Internet **on**: `!git clone https://github.com/ShammanRahin/NeuroHack-2026-Team-Blueprint` then `!python -m src.run exp=...`.
- Internet **off** (check the rules!): upload the repo as a Kaggle Dataset (`kaggle datasets create/version`),
  attach it to the notebook, `%cd /kaggle/input/<dataset>` and run the same command.
  Pretrained weights must be attached too (Kaggle Models/Datasets). Prepare these BEFORE the contest.

## Contest-day checklist
1. Pull `main`, run smoke test. 2. Put the problem statement's details into a new YAML (`data:` section, metric
in `metrics.py`). 3. A adapts `validate.py` checks, B/C add groups/models. 4. Submit only files that
`validate_submission` passed. Disclose external pretrained models in the notebook if rules require it.
