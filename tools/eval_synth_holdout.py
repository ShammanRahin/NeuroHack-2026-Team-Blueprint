"""Stricter check of the synthetic data: hold out whole TEMPLATES.

Train on synthetic rows from half of the templates (+ noise rows, + augmentation), then score
  (a) synthetic rows from the OTHER half of the templates (unseen phrasings/scenarios), and
  (b) the real labelled rows.
Usage: python tools/eval_synth_holdout.py exp=preli_synth_cpu data.dir=... [n=20000]
"""
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src import run  # noqa: E402
from src.common import config, metrics, registry  # noqa: E402
from src.data import normalize, synth_data, validate  # noqa: E402


def main(argv):
    n = int(next((a.split("=")[1] for a in argv if a.startswith("n=")), 20000))
    cfg = config.load([a for a in argv if not a.startswith("n=")])
    cfg.setdefault("tasks", ["label"])
    registry.load_all()
    b = validate.load_clean(cfg)
    lab = run._labeled_frame(b, cfg)
    g = synth_data.generate(n, seed=cfg["data"].get("synthetic", {}).get("seed", 7))
    d = cfg["data"]

    def frame(x):
        f = pd.DataFrame({b.text_col: [normalize.robust(t) for t in x["text"]], b.target_col: x["label"].to_numpy()})
        f[b.text_col + "_raw"] = x["text"].to_numpy()
        f[d["action_col"]] = x["action"].to_numpy()
        f[d["techniques_col"]] = x["techniques"].to_numpy()
        return f[lab.columns]

    half = g["tpl"].map(lambda t: zlib.crc32(t.encode()) % 2 == 0)
    tr, te = frame(g[half]), frame(g[~half].sample(min(4000, (~half).sum()), random_state=0))
    M = registry.MODELS[cfg["model"]["name"]]
    make = lambda: M(seed=cfg["seed"], **cfg["model"].get("params", {}))
    fit = run._training_frame(cfg, b, lab.iloc[:0], tr, seed=1)
    transforms = run._fit_features(cfg, b, fit)
    ft = run._targets(fit, cfg, b)
    heads = run._fit_heads(cfg, make, run._transform(transforms, fit), ft)
    pri = run._priors(ft)
    for name, ev in (("unseen-templates", te), ("real-86", lab)):
        y = run._targets(ev, cfg, b)
        dec = run._decode(run._predict_heads(heads, run._transform(transforms, ev)), cfg, pri, ev[b.text_col].tolist())
        sc = run._score(dec, y, cfg["tasks"])
        print(f"[holdout] train {len(tr)} rows from {tr.shape[0] and half.sum()} | {name}: n={len(ev)} "
              + " ".join(f"{k}={v:.4f}" for k, v in sc.items())
              + f" composite~{metrics.local_composite(dict(sc, robust=sc['A'])):.4f}")
        if name == "unseen-templates":
            from sklearn.metrics import confusion_matrix
            print(confusion_matrix(y["label"], dec["label"]))


if __name__ == "__main__":
    main(sys.argv[1:])
