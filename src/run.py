"""python -m src.run exp=preli_full [debug=true] [model.params.C=3]
Identical on Kaggle, Colab and laptop.

Tasks (cfg `tasks`): label (Task A), action (Task C), techniques (Task B); confidence (Task D) = max label probability.
Leakage / rules safety:
  * features are FIT INSIDE EVERY CV FOLD on that fold's training rows (+ synthetic + augmented copies of them only);
  * test text is never used to fit anything (Kaggle Rules tab s.5); the 500k test rows are transformed in chunks;
  * validation rows stay clean; a seeded corrupted copy of them estimates the hidden robustness score."""
import dataclasses
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common import config, env, log, metrics, registry
from src.data import validate

N_CLS, TECH, CHUNK = 6, validate.TECHNIQUES, 50_000


# ------------------------------------------------------------------------------------------------ training frames
def _labeled_frame(b, cfg):
    """Original labelled rows as one frame: text, label (+ action, techniques string)."""
    d = cfg["data"]
    cols = [b.text_col, b.target_col] + [c for c in (d.get("action_col"), d.get("techniques_col")) if c]
    return b.train[cols].reset_index(drop=True)


def _extra_rows(cfg, b):
    """Synthetic template rows + noise-only rows (fold-independent, seeded)."""
    d, parts = cfg["data"], []
    s = cfg.get("synthetic") or {}
    if s.get("n_per_class", 0) > 0:
        from src.features import synth
        g = synth.generate(n_per_class=s["n_per_class"], seed=cfg["seed"])
        from src.data.normalize import robust
        g["text"] = g["text"].map(robust)
        parts.append(g.rename(columns={"text": b.text_col, "label": b.target_col,
                                       "action": d.get("action_col") or "action",
                                       "techniques": d.get("techniques_col") or "techniques"}))
    a = cfg.get("augment") or {}
    if a.get("noise_rows", 0) > 0:
        from src.features import augment
        nr = augment.noise_rows(a["noise_rows"], b.text_col, [b.text_col, b.target_col], seed=cfg["seed"])
        if d.get("action_col"):
            nr[d["action_col"]] = 0
        if d.get("techniques_col"):
            nr[d["techniques_col"]] = "NONE"
        parts.append(nr)
    cols = _labeled_frame(b, cfg).columns
    return pd.concat([p[cols] for p in parts], ignore_index=True) if parts else pd.DataFrame(columns=cols)


def _training_frame(cfg, b, rows, extra, seed):
    """Fold training rows + extra rows + corrupted copies of both (augmentation only ever sees training rows)."""
    base = pd.concat([rows, extra], ignore_index=True)
    a = cfg.get("augment") or {}
    if a.get("copies", 0) > 0:
        from src.features import augment
        base = pd.concat([base, augment.augment(base, b.text_col, a["copies"], a.get("strength", 0.06), seed)],
                         ignore_index=True)
    return base


def _targets(frame, cfg, b):
    d = cfg["data"]
    t = {"label": frame[b.target_col].to_numpy(dtype=int)}
    if d.get("action_col"):
        t["action"] = frame[d["action_col"]].to_numpy(dtype=int)
    if d.get("techniques_col"):
        t["techniques"] = validate._parse_techniques(frame[d["techniques_col"]].reset_index(drop=True)).to_numpy()
    return t


# ------------------------------------------------------------------------------------------------ features & heads
def _stack(mats):
    return sp.hstack(mats).tocsr() if any(sp.issparse(m) for m in mats) else np.hstack(mats)


def _fit_features(cfg, b, frame):
    fb = dataclasses.replace(b, train=frame, aux={})
    return [registry.FEATURES[f["name"]](fb, **f.get("params", {})) for f in cfg["features"]]


def _transform(transforms, df):
    return _stack([t(df) for t in transforms])


def _fit_one(make, X, y):
    cls = np.unique(y)
    return ("const", int(cls[0])) if len(cls) == 1 else ("model", make().fit(X, y), cls)


def _proba_one(head, X, n_out):
    out = np.zeros((X.shape[0], n_out))
    if head[0] == "const":
        out[:, head[1]] = 1.0
    else:
        out[:, head[2]] = head[1].predict_proba(X)
    return out


def _fit_heads(cfg, make, X, targets):
    heads = {}
    for t in cfg["tasks"]:
        y = targets[t]
        heads[t] = [_fit_one(make, X, y[:, j]) for j in range(len(TECH))] if t == "techniques" else _fit_one(make, X, y)
    return heads


def _predict_heads(heads, X):
    return {t: np.stack([_proba_one(hj, X, 2)[:, 1] for hj in h], axis=1) if t == "techniques" else _proba_one(h, X, N_CLS)
            for t, h in heads.items()}


def _decode(P):
    """Probabilities -> predictions, applying the labelling-guide conventions."""
    label = P["label"].argmax(1)
    d = {"label": label, "confidence": np.clip(P["label"].max(1), 0.0, 1.0)}
    benign = np.isin(label, (0, 5))
    if "action" in P:
        act = P["action"].argmax(1)
        fraud0 = ~benign & (act == 0)                       # a scam always asks the victim to do something
        act[fraud0] = 1 + P["action"][fraud0][:, 1:].argmax(1)
        act[benign] = 0                                     # classes 0 and 5: action 0
        d["action"] = act
    if "techniques" in P:
        T = P["techniques"] >= 0.5
        T[benign] = False                                   # classes 0 and 5: techniques NONE
        none_fraud = ~benign & ~T.any(1)                    # a scam uses at least one technique
        T[none_fraud, P["techniques"][none_fraud].argmax(1)] = True
        d["techniques"] = T
    return d


def _score(dec, y, tasks):
    parts = {"A": metrics.macro_f1(y["label"], dec["label"]),
             "calib": metrics.calibration(dec["confidence"], dec["label"] == y["label"])}
    if "techniques" in tasks:
        parts["B"] = metrics.techniques_f1(y["techniques"], dec["techniques"])
    if "action" in tasks:
        parts["C"] = metrics.macro_f1(y["action"], dec["action"])
    return parts


def _techniques_str(T):
    return ["|".join(TECH[j] for j in np.flatnonzero(r)) or "NONE" for r in T]


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    env.load_dotenv()
    cfg = config.load(sys.argv[1:] if argv is None else argv)
    cfg.setdefault("tasks", ["label"])
    registry.load_all()
    b = validate.load_clean(cfg)
    lab = _labeled_frame(b, cfg)
    y_all = _targets(lab, cfg, b)
    for t in cfg["tasks"]:
        if t not in y_all:
            raise ValueError(f"task {t!r} needs its column in data (data.action_col / data.techniques_col)")
    multi = len(cfg["tasks"]) > 1
    extra = _extra_rows(cfg, b)
    M = registry.MODELS[cfg["model"]["name"]]
    make = lambda: M(seed=cfg["seed"], **cfg["model"].get("params", {}))
    robust_eval = (cfg.get("augment") or {}).get("robust_eval", True)
    print(f"[{cfg['name']}] {len(lab)} labelled + {len(extra)} synthetic/noise rows; "
          f"augment copies {(cfg.get('augment') or {}).get('copies', 0)}")

    rows = []
    for i, (tr, va) in enumerate(validate.folds(cfg, y_all["label"])):
        fit = _training_frame(cfg, b, lab.iloc[tr], extra, seed=cfg["seed"] + i)
        transforms = _fit_features(cfg, b, fit)
        heads = _fit_heads(cfg, make, _transform(transforms, fit), _targets(fit, cfg, b))
        yv = {k: v[va] for k, v in y_all.items()}
        parts = _score(_decode(_predict_heads(heads, _transform(transforms, lab.iloc[va]))), yv, cfg["tasks"])
        if robust_eval:
            from src.features import augment
            noisy = augment.augment(lab.iloc[va], b.text_col, copies=1, seed=10_000 + i)
            parts["robust"] = metrics.macro_f1(yv["label"], _decode(_predict_heads(heads, _transform(transforms, noisy)))["label"])
        rows.append(parts)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    comps = [metrics.local_composite(r) for r in rows]
    shown = " ".join(f"{k}={v:.4f}" for k, v in mean.items())
    print(f"[{cfg['name']}] CV local composite {np.mean(comps):.4f} +/- {np.std(comps):.4f} | {shown} ({len(rows)} folds)")

    fit = _training_frame(cfg, b, lab, extra, seed=cfg["seed"] + 999)
    transforms = _fit_features(cfg, b, fit)
    heads = _fit_heads(cfg, make, _transform(transforms, fit), _targets(fit, cfg, b))
    t0, chunks = time.time(), []
    for s in range(0, len(b.test), CHUNK):
        chunks.append(_predict_heads(heads, _transform(transforms, b.test.iloc[s:s + CHUNK])))
    P = {t: np.concatenate([c[t] for c in chunks]) for t in chunks[0]}
    dt = time.time() - t0
    print(f"[{cfg['name']}] inference {len(b.test)} rows in {dt:.1f}s ({len(b.test) / max(dt, 1e-9):,.0f} rows/s, "
          f"{1000 * dt / len(b.test):.3f} ms/row)")
    dec = _decode(P)

    out = env.work_dir() / "outputs" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "test_proba.npz", **P)
    sub = pd.DataFrame({b.id_col: b.test[b.id_col].to_numpy(), b.target_col: dec["label"]})
    if "techniques" in dec:
        sub["techniques"] = _techniques_str(dec["techniques"])
    if "action" in dec:
        sub["action"] = dec["action"]
    if multi or (b.sample_sub is not None and "confidence" in b.sample_sub.columns):
        sub["confidence"] = dec["confidence"].round(6)
    if b.sample_sub is not None:
        sub = sub[[c for c in b.sample_sub.columns if c in sub.columns]]
    if not cfg.get("debug"):
        validate.validate_submission(sub, b)
        sub.to_csv(out / "submission.csv", index=False)
        print(f"[{cfg['name']}] wrote {out / 'submission.csv'} ({len(sub)} rows, validated)")
        log.append(dict(platform=env.platform(), exp=cfg["name"], cfg_hash=cfg["_hash"],
                        cv_mean=round(float(np.mean(comps)), 4), cv_std=round(float(np.std(comps)), 4),
                        score_a=round(mean["A"], 4), score_b=round(mean["B"], 4) if "B" in mean else "",
                        score_c=round(mean["C"], 4) if "C" in mean else "", calib=round(mean["calib"], 4),
                        n_folds=len(rows), note=cfg.get("note", "")))
    return float(np.mean(comps))


if __name__ == "__main__":
    main()
