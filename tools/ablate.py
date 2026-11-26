"""Ablation: run every variant through the SAME repeated grouped CV (+ shift holdouts, robustness, calibration), save a
table + chart, and name the peak.   python tools/ablate.py configs/ablation.yaml [--data-dir DIR] [--only name,name]
Peak rule: highest local composite; variants within one fold-std of the best count as tied and the tie is broken by
the worst shift holdout (the hidden test is shifted). No test inference (cv_only), so it is fast."""
import argparse, pathlib, sys, time, traceback
import pandas as pd, yaml
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import run
from src.common import env

a = argparse.ArgumentParser()
a.add_argument("spec")
a.add_argument("--data-dir", default=None)
a.add_argument("--only", default=None)
a.add_argument("--out", default=None)
a = a.parse_args()
spec = yaml.safe_load(open(a.spec))
only = set(a.only.split(",")) if a.only else None
out = pathlib.Path(a.out or env.work_dir() / "ablation")
out.mkdir(parents=True, exist_ok=True)
rows = []
for v in spec["variants"]:
    if only and v["name"] not in only:
        continue
    argv = [f"exp={v.get('exp', spec.get('exp', 'preli_nogpu'))}", "cv_only=true", f"name=ab_{v['name']}"]
    argv += [f"data.dir={a.data_dir}"] if a.data_dir else []
    argv += list(spec.get("common", [])) + list(v.get("set", []))
    t0 = time.time()
    try:
        r = run.main(argv)
        r = {k: v2 for k, v2 in r.items() if k != "exp"}
        rows.append({"variant": v["name"], "minutes": round((time.time() - t0) / 60, 2), **r})
    except Exception as e:
        traceback.print_exc()
        rows.append({"variant": v["name"], "error": str(e)[:200]})
    pd.DataFrame(rows).to_csv(out / "ablation.csv", index=False)

df = pd.DataFrame(rows)
ok = df[df.get("composite").notna()] if "composite" in df else df.iloc[0:0]
if len(ok):
    best = ok["composite"].max()
    tol = ok.loc[ok["composite"].idxmax(), "composite_std"]
    tied = ok[ok["composite"] >= best - tol]
    peak = tied.sort_values(["shift_worst", "composite"], ascending=False).iloc[0]["variant"]
    cols = [c for c in ["variant", "composite", "composite_std", "A", "B", "C", "robust", "calib", "shift_worst", "minutes"] if c in ok]
    print("\nABLATION TABLE\n" + ok[cols].round(4).to_csv(index=False))
    print(f"PEAK: {peak}  (best composite {best:.4f}; tie tolerance {tol:.4f} = 1 fold-std; tie-break: worst shift holdout)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(ok) + 1.5))
    y = range(len(ok))
    ax.barh(y, ok["composite"], xerr=ok["composite_std"], color=["#c2410c" if v == peak else "#94a3b8" for v in ok["variant"]])
    ax.scatter(ok["shift_worst"], y, marker="D", color="#0f172a", zorder=3, label="worst shift holdout (Task A)")
    ax.set_yticks(list(y), ok["variant"]); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("local composite (CV, 86 labelled rows) - error bar = fold std"); ax.legend(loc="lower right")
    ax.set_title(f"Ablation - peak: {peak}")
    fig.tight_layout(); fig.savefig(out / "ablation.png", dpi=130)
    print(f"saved {out / 'ablation.csv'} and {out / 'ablation.png'}")
