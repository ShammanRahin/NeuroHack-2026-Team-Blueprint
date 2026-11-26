# Team setup guide

Read **Part 0** (team lead, once), **Part 1** (everyone), then only **your own role**.

---
## Part 0: Team lead, once (about 15 min)

1. **GitHub:** create a *private* repo. Push the scaffold to `main`.
2. **Invite** both teammates (Settings → Collaborators, Write access).
3. **Fix `CODEOWNERS`:** replace the three `@member-...` placeholders with real GitHub usernames; commit.
4. **Protect `main`** (Settings → Branches → Add rule for `main`): require a pull request, require the
   `smoke` check to pass. Nobody pushes straight to `main`.
5. **Kaggle:** everyone joins the competition with the email used for registration; the leader creates the
   team and invites the others (see the competition's Overview page).
6. Send teammates the repo name. Each teammate does Part 1.

---
## Part 1: Everyone (about 10 min, once)

### 1.1 Accounts and keys
| Need | How | Where it goes |
|---|---|---|
| GitHub token | GitHub → Settings → Developer settings → Fine-grained tokens → *Generate*. Repository access: **only this repo**. Permission: **Contents → Read and write**. | Colab secret `GH_TOKEN` |
| Kaggle key | kaggle.com → Settings → API → *Create New Token* (gives your username + key) | Colab secrets `KAGGLE_USERNAME`, `KAGGLE_KEY` |

Never paste these into notebook cells or commit them.

### 1.2 Colab
1. Open `notebooks/colab.ipynb` from the repo on GitHub → *Open in Colab* (or upload it). Save a copy to your Drive.
2. Left bar → key icon (Secrets) → add the three secrets above and switch **Notebook access ON** for each.
3. Runtime → Change runtime type → GPU (only needed when your work uses one; sklearn baselines don't).
4. In the first cell edit: `REPO` (owner/repo), `BRANCH` (your branch, see 1.3), `COMP`, `EXP`, `EXP_USER` (your name).
5. Run cell 1 (clones, installs, downloads data; approve the Drive prompt) → cell 2 (smoke test + tiny run).
   Both must succeed before you change anything. If one fails, tell the team lead; do not edit shared files to "fix" it.

### 1.3 Your branch and the loop
```
git checkout -b <role>/<what>        # e.g. features/word-ngrams   (from latest main)
# ...edit ONLY your own folder + your own configs/<your_exp>.yaml...
python tests/smoke_test.py           # must pass
python -m src.run exp=<your_exp>     # CV score + validated submission + a row in results/log.csv
git add -A && git commit -m "features: add word tf-idf (cv +0.02)" && git push origin <your-branch>
```
Then open a Pull Request into `main`. Your folder owner (that's you) plus one teammate approve; CI must be green.
Pull `main` into your branch regularly: `git fetch origin && git merge origin/main`.

**Rules of the road**
- Edit only your folder. Need a change in someone else's folder or in `src/common/`? Ask them / open a PR they approve.
- One experiment = one YAML in `configs/` with a **unique `name:`** (prefix with your role letter).
- Put the *why* in the commit message and the `note:` field of your YAML; it ends up in the log.

### 1.4 Optional: shared Drive folder
Create one Drive folder, share it with the team (Editor), and set `WORK_DIR` in cell 1 to
`/content/drive/MyDrive/<shared folder>/work`. Then everyone's cached features and `outputs/<exp>/` (oof and
test predictions) are visible to everyone, so C can blend B's and their own results without recomputing.
Experiment names must be unique, because the name is the output folder.

---
## Part 2A: Member A, data validation (owns `src/data/`)

You are the gatekeeper. Nothing bad passes you, and nothing invalid is ever uploaded.

**Setup check:** run `python -m src.run exp=mock_prelims debug=true` in Colab. You should see a CV score and no errors.

**What you own** (`src/data/validate.py`)
- `load_clean(cfg)`: reads the files, cleans text (Unicode NFKC, whitespace), then checks. It already stops on:
  missing columns, duplicate ids, null labels, identical text with conflicting labels, test ids not matching
  `sample_submission`. It warns on text shared between train and test.
- `folds()`: fixed-seed repeated stratified folds. Everyone's CV is comparable because of this.
- `validate_submission()`: runs automatically before a submission file is written. Wrong columns, missing or
  duplicated ids, invalid labels all raise.

**Your first-day tasks**
1. Run `exp=mock_prelims` and read the warnings. Look at the class counts and text lengths in the real data.
2. Add checks specific to this data (empty text after cleaning, mixed scripts, suspiciously long rows). Raise
   `ValueError` for anything that makes results wrong; just `print("[data] WARN ...")` for anything odd.
3. With 60 labelled rows, check whether `cv.n_splits: 5` is sane (needs at least `n_splits` rows per class).
4. Write down in the PR what the data looks like (classes, languages, duplicates) so B and C can use it.

**On contest day:** the problem statement arrives 30 minutes before start. Within those 30 minutes:
copy `configs/mock_prelims.yaml` to a new file and set `data:` (file names, `id_col`, `target_col`, `text_col`),
adapt `load_clean` for the new files, tell B and C the schema, and run the smoke test.
If the metric differs from macro-F1, change `src/common/metrics.py` (shared file: say so in the team chat).

---
## Part 2B: Member B, feature engineering (owns `src/features/`)

You turn cleaned data into a matrix. **Contract:** return one matrix with **train rows first, then test rows**
(`n_train + n_test` rows), dense or scipy-sparse.

**Setup check:** `python -m src.run exp=mock_prelims debug=true` works in Colab.

**Add a feature group in 3 steps**
1. New file `src/features/tfidf_word.py` (an example already exists):
```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.common.registry import feature

@feature("tfidf_word")
def tfidf_word(bundle, max_features=20000):
    text = np.concatenate([bundle.train[bundle.text_col], bundle.test[bundle.text_col]])  # train rows first
    return TfidfVectorizer(max_features=max_features, sublinear_tf=True).fit_transform(text)
```
2. New config `configs/b_word_plus_char.yaml` (a list **replaces** the base list, so repeat what you keep):
```yaml
base: baseline
name: b_word_plus_char
note: add word tf-idf next to char n-grams
features:
  - {name: tfidf_char, params: {ngram: [2, 5]}}
  - {name: tfidf_word, params: {max_features: 20000}}
```
3. `python -m src.run exp=b_word_plus_char`: compare `cv_mean` in `results/log.csv` with the baseline row.

**Things to know**
- Features are cached on disk. The cache key includes your function's source code, so editing it
  recomputes automatically. Heavy work (embeddings from a pretrained model) is computed once per change.
- With only 60 labelled rows, CV scores are noisy: compare `cv_mean` **and** `cv_std`. A +0.01 change inside
  the std is noise. Keep what helps clearly.
- Pretrained models: if Kaggle notebooks have no internet, the weights must be attached as a Kaggle Model/Dataset.
  Prepare and test this before the contest, and note every external model you use (the rules ask for disclosure).
- Don't use test labels or hand-label test rows (rules). Using unlabelled test text for unsupervised features
  such as TF-IDF is what the baseline already does; check the rules before doing anything fancier.

---
## Part 2C: Member C, model architecture (owns `src/models/`)

You turn a matrix into probabilities. **Contract:** a class with `__init__(seed, **params)`, `fit(X, y)`,
`predict_proba(X)` returning `(n, n_classes)` with columns in sorted class-label order.

**Setup check:** `python -m src.run exp=mock_prelims debug=true` works in Colab.

**Add a model in 3 steps**
1. New file `src/models/nb.py` (an example already exists):
```python
from sklearn.naive_bayes import ComplementNB
from src.common.registry import model

@model("nb")
class NB:
    def __init__(self, seed=0, **params):   # seed is passed to every model, even if unused
        self.m = ComplementNB(**params)
    def fit(self, X, y):
        self.m.fit(X, y); return self
    def predict_proba(self, X):
        return self.m.predict_proba(X)
```
2. New config `configs/c_nb.yaml` (switching `model.name` replaces the base model's params):
```yaml
base: baseline
name: c_nb
model: {name: nb, params: {alpha: 0.5}}
```
3. `python -m src.run exp=c_nb`, then compare in `results/log.csv`.

**Blending (your edge):** every run saves `oof.npy` (out-of-fold probabilities for the labelled rows) and
`test_proba.npy` into `outputs/<exp>/` under `WORK_DIR`. To combine two runs:
```python
import numpy as np
a, b = np.load("outputs/c_nb/oof.npy"), np.load("outputs/baseline/oof.npy")
blend = (a + b) / 2        # score it with src.common.metrics.score on the labelled y; tune weights only if the gain is clear
```
Do the same with `test_proba.npy` for the final submission. Keep the blending code as its own model or script in your folder.

**Things to know**
- GPU models (fine-tuning a transformer) are fine: `fit` can use `torch`; just keep the same interface and make
  sure `fit` on ~50 rows per fold finishes in minutes, because CV trains it 15 times (`cv.n_splits × n_repeats`).
  Lower `cv.n_repeats` in your YAML while experimenting (`cv.n_repeats=1`).
- Macro-F1 punishes ignoring small classes. `class_weight="balanced"` (baseline) is a good default; the
  test class distribution is not uniform, so also look at per-class F1 before trusting a score.

---
## Troubleshooting
| Symptom | Fix |
|---|---|
| `403` / `Repository not found` on clone | Token lacks access to *this* repo or Contents read/write; recreate it |
| Kaggle `403` / `401` on download | Account hasn't joined the competition / accepted rules, or wrong key |
| `FileNotFoundError` on data | Cell 1's download didn't finish; delete `/content/data` and rerun cell 1 |
| Smoke test fails after `git merge origin/main` | Someone merged a break; tell the team lead; don't patch other folders |
| Score changes but code didn't | Different `cv.n_repeats`/seed override: compare `cfg_hash` column in the log |
| Colab restarted, features recomputed | `WORK_DIR` isn't on Drive (check cell 1 ran the Drive mount) |
