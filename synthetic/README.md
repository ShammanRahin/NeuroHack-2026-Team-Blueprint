# ScamShield BD synthetic training set: 1,000,000 unique labelled rows

Four gzipped CSV parts of 250,000 rows each (about 18 MB per part, under GitHub's file limit).

```python
import pandas as pd, glob
df = pd.concat(pd.read_csv(f) for f in sorted(glob.glob("synthetic/ScamShieldBD_synth_1000k_part*.csv.gz")))
```

Columns:

| column | meaning |
|---|---|
| `id` | 1..1,000,000 |
| `text` | message or conversation (`|||` between turns, like the real data) |
| `label` | Task A class 0-5 |
| `techniques` | Task B, `T1|T2...` or `NONE` |
| `action` | Task C, 0-5 |
| `template` | generator id (`<class>-<n>` = hand-written template, `c-<hook>-<request>` = composer, `p/l-compose` = promo/legit composer, `noise`) |
| `register` | sms / conversation / email / group_chat / call |
| `lang` | english / banglish / bangla / mixed (Bangla script + Latin) |
| `corrupted` | 1 if adversarial noise was applied (homoglyphs, zero-width chars, typos, noise strings, misspelt sensitive words) |

## How it was made

`python tools/make_synthetic_dataset.py --n 1000000 --out synthetic` (seeded, about 3 min on CPU). The text comes from `src/data/synth_data.py`, in two ways:
- Hand-written templates based on the official labelling guide: about 400 templates in English, Banglish and Bangla.
- A composer that combines a lure (account, prize, job/investment, parcel) with a request (credential, payment, link, join/install, call, personal details), plus optional sender, context and urgency sentences.

It also includes hard negatives: legit OTPs, receipts, delivery updates, fee-free job offers, and warnings that quote a scam (all label 0).

Nothing is derived from `test.csv`. The organisers' Data tab explicitly allows synthetic data. It is declared in `docs/DECLARATION.md`.

## Label conventions (from the labelling guide)

- **Priority when more than one class fits:** 1 > 2 > 4 > 3 > 5 > 0. For example, a prize or job message that asks for an OTP is class 1. A "gift" parcel demanding customs payment is class 2.
- **Classes 0 and 5:** techniques `NONE`, action 0. **Every scam:** at least one technique and action > 0. Both rules are checked when the set is built.
- **Techniques:**
  - T1 organisation or role claimed as sender
  - T2 credential request
  - T3 payment demand
  - T4 reward lure
  - T5 urgency, threat or pressure
  - T6 unrealistic earnings
- **Actions:**
  - 1 hand over credentials or personal details
  - 2 pay or send money
  - 3 open a link
  - 4 install an app, join a group, or do tasks
  - 5 call or WhatsApp a number

## Quality controls

- 1,000,000 unique texts. At most 2 rows share a skeleton (the lower-cased text with every digit run masked), so rows that differ only in an amount or a code can't pad the set.
- 1% noise-only rows (label 0, as the Data tab specifies). 30% of rows are corrupted in place.
- Distribution (full numbers in `stats.json`):
  - **Label:** 0 20% · 1 20% · 2 15% · 3 15% · 4 15% · 5 16%
  - **Register:** SMS 57% · conversation 20% · email 13% · group chat 6% · call 4%
  - **Language:** English 54% · mixed script 29% · Banglish 17%
  - **Length:** median 156 characters, p95 403, max 683
- **Sanity check:** a char+word TF-IDF + logistic regression trained on 120k rows of part 1 gets label macro-F1 1.00 and action macro-F1 0.98 on the 86 real `sample.csv` rows.
  - Read this as an upper bound. The templates were written by people who had read those 86 rows.
- **On the public leaderboard,** the same generator family scored 0.739 → 0.808 → 0.841 → 0.858 → 0.862 as variety grew.
