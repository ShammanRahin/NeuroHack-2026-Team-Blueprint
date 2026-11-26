# Member A validation report

## Pipeline behavior

The data loader keeps the existing Bundle and runner interfaces. It normalizes text
with Unicode NFKC and collapses whitespace, then rejects missing or duplicate IDs,
empty or invisible-only text, invalid training labels, exact cleaned text with
conflicting labels, and inconsistent sample submission IDs. Labels must have integer
dtype and be in the inclusive range 0 to 5. The loader warns about mixed writing
scripts, cleaned messages longer than 2,000 characters, and exact cleaned text
shared by train and test. Its aggregate output includes class counts, row counts,
missing text, duplicate rows, script combinations, and length quantiles.

The CV grouping key case-folds text and normalizes separators. Equal keys and
pairs with cosine similarity of at least 0.90 over binary character 3- to 5-grams
are joined transitively. Groups are built from training text, then aligned with
the retained rows in debug mode. The loader passes group IDs through run-local
config state, leaving the shared runner and Bundle contract unchanged.

Repeated StratifiedGroupKFold uses seeds derived from the configured seed. It
requires every observed class in both partitions of each fold. The requested
fold count is capped by the number of independent groups available to the
rarest class. The splitter tries seeded layouts and reduces the fold count with
a warning if needed. It raises if it cannot find a valid layout with at least
two folds.

Submission validation enforces the sample's column order when present and the
configured ID-then-label order otherwise. It requires each test ID exactly once
and integer labels in 0 through 5.

## Verification

| Check | Result |
| --- | --- |
| Baseline smoke test before and after the changes | Passed. Both full and debug runs reported macro-F1 1.0000 over 15 folds. |
| Mock prelims with the local competition files in debug mode | Passed. Macro-F1 0.5321 +/- 0.1601 over 15 folds using 74 labeled rows and the first 50 test rows. |
| Mock prelims with temporary smoke fixtures | Passed in debug mode with macro-F1 1.0000 over 15 folds. |
| Focused synthetic checks | Passed for exact and near-duplicate grouping, fixed-seed reproducibility, group separation, validation coverage, fold reduction from five to two, mixed-script and long-text warnings, and rejection of invalid text, labels, IDs, and submission columns. |
| Full labeled dataset split check | Passed. All six classes occurred in both partitions of every fold. Groups were disjoint, splits reproduced with seed 42, and each row appeared in validation three times. |
| Competition template schema check | Passed on a small template slice. Missing auxiliary columns and reordered columns were rejected. |

The smoke fixture had 60 training records across three classes (20 each) and
30 test records. All texts used Latin letters and were 19 characters after
cleaning, with no missing or exact duplicate texts. Training had 59 independent
groups, with 20, 20, and 19 groups per class, so five folds per repeat were
feasible on that fixture.

## Competition data

Both configurations resolve to the repository's `data/` folder. The
`mock_prelims` config points to `sample.csv`, `test.csv`, and
`sample_submission.csv`. The `baseline` config expects `train.csv`, which
is absent. The labeled file contains 86 rows and the test and template files
each contain 500,000. No CSV had missing values, duplicate full rows, or
duplicate IDs. The template IDs match the test IDs in the same order.

| Statistic | Labeled sample | Test |
| --- | ---: | ---: |
| Exact duplicate rows by cleaned text | 0 | 0 |
| Duplicate rows by case-folded, separator-normalized key | 0 | 31,058 |
| Cleaned characters (min / median / p95 / max) | 10 / 83 / 149.25 / 245 | 4 / 96 / 255 / 616 |
| Latin only | 77 | 404,215 |
| Bengali only | 6 | 2,950 |
| Bengali and Latin | 0 | 50,297 |
| Cyrillic and Latin | 3 | 40,948 |
| Bengali, Cyrillic, and Latin | 0 | 1,590 |

There were zero exact cleaned-text matches across train and test and zero
messages above the 2,000-character warning threshold. The labeled sample
contains English, Bangla in Bengali script, and romanized Bangla. Some Latin
words contain Cyrillic lookalikes. Script counts describe letters, so they
cannot distinguish these substitutions from a separate language or count
English and romanized Bangla separately.

Labels 0 through 5 have 32, 12, 10, 11, 9, and 12 rows respectively. All 86
rows formed separate groups at the current 0.90 similarity cutoff. Five folds
were feasible across all three repeats, with 17 to 18 validation rows per fold.
The smallest class has nine independent groups. The cutoff is a heuristic for
lexical similarity.

## Remaining integration

The competition template requires `id,label,techniques,action,confidence`.
The validator accepts this template schema and enforces its column order, test
IDs, and label range. The shared runner produces only `id,label`, so its
non-debug output cannot pass the competition template check. Producing the
other fields requires their official definitions and a shared pipeline change.
The changes here are confined to `src/data/`.

The baseline's existing
TF-IDF vocabulary is fitted across train and test before CV, so the group split
isolates training messages at model fit time while TF-IDF preprocessing remains
transductive. Reproducing exact split assignments across machines should use
the same scikit-learn version.
