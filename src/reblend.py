"""Re-blend saved test probabilities of a finished `preli_final` run with another encoder weight, no retraining.

    python -m src.reblend exp=preli_final encoder.blend_weight=0.7 [reblend.dir=/kaggle/input/...]

Uses outputs/<exp>/test_proba.npz (linear member) and test_proba_encoder.npz (encoder member) written by src/run.py,
recomputes the decoder's class priors from the same seeded training frame, and decodes exactly like the encoder stage.
Running `python -m src.run exp=preli_final encoder.blend_weight=<w>` end to end gives the same submission (up to GPU
non-determinism), so this is only a shortcut that avoids re-training. Test text is only read to route noise-only rows.
"""
import glob
import pathlib
import sys

import numpy as np
import pandas as pd

from src import run
from src.common import config, env, registry
from src.data import validate


def _find(name, root):
    hits = sorted(glob.glob(f"{root}/**/{name}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def main(argv=None):
    cfg = config.load(sys.argv[1:] if argv is None else argv)
    registry.load_all()
    w = float(cfg["encoder"]["blend_weight"])
    root = (cfg.get("reblend") or {}).get("dir") or ("/kaggle/input" if pathlib.Path("/kaggle/input").exists()
                                                     else str(env.work_dir() / "outputs" / cfg["name"]))
    P_lin = dict(np.load(_find("test_proba.npz", root)))
    P_enc = dict(np.load(_find("test_proba_encoder.npz", root)))
    b = validate.load_clean(cfg)
    assert len(P_lin["label"]) == len(P_enc["label"]) == len(b.test), "saved probabilities do not match test.csv"
    lab = run._labeled_frame(b, cfg)
    extra = run._extra_rows(cfg, b)
    fit = run._training_frame(cfg, b, lab, extra, seed=cfg["seed"] + 999)  # same seeded frame as the full fit
    pri = run._priors(run._targets(fit, cfg, b))
    P = {k: w * P_enc[k] + (1 - w) * P_lin[k] for k in P_lin if k in P_enc}
    dec = run._decode(P, cfg, pri, b.test[b.text_col].tolist())
    sub = pd.DataFrame({b.id_col: b.test[b.id_col].to_numpy(), b.target_col: dec["label"],
                        "techniques": run._techniques_str(dec["techniques"]), "action": dec["action"],
                        "confidence": dec["confidence"].round(6)})
    if b.sample_sub is not None:
        sub = sub[[c for c in b.sample_sub.columns if c in sub.columns]]
    validate.validate_submission(sub, b)
    out = env.work_dir() / "outputs" / f"{cfg['name']}_w{int(round(100 * w))}"
    out.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out / "submission.csv", index=False)
    agree = float(np.mean(P_lin["label"].argmax(1) == dec["label"]))
    print(f"[reblend] encoder weight {w}: label distribution {np.bincount(dec['label'], minlength=6).tolist()}, "
          f"agreement with linear-only {agree:.3f} -> {out / 'submission.csv'}")
    return sub


if __name__ == "__main__":
    main()
