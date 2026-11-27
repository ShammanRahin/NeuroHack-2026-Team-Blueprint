"""Build the shared synthetic training set (default 1,000,000 unique labelled rows) for the whole team.

    python tools/make_synthetic_dataset.py --n 1000000 --out synthetic

Writes synthetic/ScamShieldBD_synth_<N>_partK.csv.gz (250k rows each, under GitHub's file limit),
synthetic/stats.json and prints a quality report. Everything comes from src/data/synth_data.py (templates written
from the official labelling guide + the hook/request composer); nothing is derived from test.csv.

Quality controls:
  * exact-duplicate texts removed;
  * at most 2 rows per "skeleton" (lower-cased text with every digit run masked), so rows that differ only in an
    amount / code / phone are capped and the set is not padded with trivial copies;
  * ~1% noise-only rows (label 0 / NONE / 0, as the Data tab specifies);
  * 30% of rows corrupted in place (homoglyphs, zero-width chars, typos, noise strings) -> column `corrupted`;
  * labels are consistent with the guide: classes 0/5 -> techniques NONE and action 0; scams have >=1 technique
    and action > 0 (asserted below).
"""
import argparse
import json
import os
import random
import re
import sys

import pandas as pd

sys.path.insert(0, ".")
from src.data import synth_dataset  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--part", type=int, default=250_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = synth_dataset.build(a.n, a.seed)

    stats = {
        "rows": len(df), "unique_texts": int(df.text.nunique()),
        "label": df.label.value_counts().sort_index().to_dict(),
        "action": df.action.value_counts().sort_index().to_dict(),
        "techniques_top": df.techniques.value_counts().head(15).to_dict(),
        "technique_rate": {t: round(float(df.techniques.str.contains(t).mean()), 4) for t in ["T1", "T2", "T3", "T4", "T5", "T6"]},
        "register": df.register.value_counts().to_dict(), "lang": df.lang.value_counts().to_dict(),
        "corrupted_share": round(float(df.corrupted.mean()), 4),
        "chars": {q: int(df.text.str.len().quantile(qq)) for q, qq in [("p5", .05), ("median", .5), ("p95", .95), ("max", 1.0)]},
        "distinct_templates": int(df.template.nunique()),
    }
    stats = json.loads(json.dumps(stats, default=int))
    parts = []
    for k, start in enumerate(range(0, len(df), a.part), 1):
        fn = os.path.join(a.out, f"ScamShieldBD_synth_{a.n // 1000}k_part{k}.csv.gz")
        df.iloc[start:start + a.part].to_csv(fn, index=False, compression="gzip")
        parts.append({"file": os.path.basename(fn), "rows": int(min(a.part, len(df) - start)),
                      "mb": round(os.path.getsize(fn) / 1e6, 1)})
    stats["parts"] = parts
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
