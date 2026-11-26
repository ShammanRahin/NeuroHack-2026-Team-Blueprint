"""python -m src.run exp=preli_full [debug=true] [model.params.C=3]
Identical on Kaggle, Colab and laptop.

Tasks (cfg `tasks`): label (Task A), action (Task C), techniques (Task B); confidence (Task D) = max label probability.
Leakage / rules safety:
  * features are FIT INSIDE EVERY CV FOLD on that fold's training rows (+ augmented copies of those real rows only);
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
    if b.text_col + "_raw" in b.train.columns:
        cols.append(b.text_col + "_raw")
    return b.train[cols].reset_index(drop=True)


def _extra_rows(cfg, b):
    """Noise-only rows (fold-independent, seeded). No synthetic message generation."""
    d, parts = cfg["data"], []
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
    for p in parts:
        if b.text_col + "_raw" in cols and b.text_col + "_raw" not in p.columns:
            p[b.text_col + "_raw"] = p[b.text_col]
    return pd.concat([p[cols] for p in parts], ignore_index=True) if parts else pd.DataFrame(columns=cols)


def _training_frame(cfg, b, rows, extra, seed):
    """Fold training rows + extra rows + corrupted copies of both (augmentation only ever sees training rows)."""
    a = cfg.get("augment") or {}
    base = pd.concat([rows, extra], ignore_index=True)
    if a.get("translit") or a.get("views"):
        from src.features import augment
        base = pd.concat([base, augment.real_row_views(rows, b.text_col, a.get("translit", False), a.get("views", False))],
                         ignore_index=True)
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


PRIORITY = [1, 2, 4, 3, 5, 0]  # labelling guide: first matching class wins


def _priors(fit_targets):
    """Class-conditional technique / action frequencies from the TRAINING rows (Laplace-smoothed)."""
    y = fit_targets["label"]
    pri = {}
    if "techniques" in fit_targets:
        T = fit_targets["techniques"]
        pri["tech"] = np.stack([(T[y == c].sum(0) + 0.5) / ((y == c).sum() + 1.0) for c in range(N_CLS)])
    if "action" in fit_targets:
        A = fit_targets["action"]
        pri["act"] = np.stack([(np.bincount(A[y == c], minlength=N_CLS) + 0.5) / ((y == c).sum() + 3.0)
                               for c in range(N_CLS)])
        pri["act"] /= pri["act"].sum(1, keepdims=True)
    return pri


def _decode(P, cfg=None, pri=None, texts=None, acc_oof=None):
    """Probabilities -> predictions with the labelling-guide structure.
    cfg['decode']: tech_prior_weight, action_prior_weight, priority_tau, tech_threshold,
                   conf_post (confidence of the SUBMITTED label), conf_shrink (a in a*c+(1-a)*acc), noise_router."""
    d_cfg = ((cfg or {}).get("decode") or {})
    pri = pri or {}
    PL = P["label"]
    label = PL.argmax(1)
    tau = d_cfg.get("priority_tau")
    if tau is not None and tau < 1.0:  # among classes close to the best, the guide's priority order decides
        close = PL >= tau * PL.max(1, keepdims=True)
        rank = np.array([PRIORITY.index(c) for c in range(N_CLS)])
        label = np.where(close, rank[None, :], 99).argmin(1)
    routed = np.zeros(len(label), bool)
    if d_cfg.get("noise_router") and texts is not None:
        from src.features.handcrafted import noise_only
        routed = np.array([noise_only(t) for t in texts])
        label = np.where(routed, 0, label)
    conf = PL[np.arange(len(label)), label] if d_cfg.get("conf_post", False) else PL.max(1)
    a = d_cfg.get("conf_shrink", 1.0)
    if a < 1.0 and acc_oof is not None:
        conf = a * conf + (1 - a) * acc_oof
    conf = np.where(routed, d_cfg.get("router_conf", 0.8), conf)
    d = {"label": label, "confidence": np.clip(conf, 0.0, 1.0)}
    benign = np.isin(label, (0, 5))
    if "action" in P:
        Pa = P["action"]
        w = d_cfg.get("action_prior_weight", 0.0)
        if w > 0 and "act" in pri:
            Pa = (1 - w) * Pa + w * (PL @ pri["act"])
        act = Pa.argmax(1)
        fraud0 = ~benign & (act == 0)                       # a scam always asks the victim to do something
        act[fraud0] = 1 + Pa[fraud0][:, 1:].argmax(1)
        act[benign] = 0                                     # classes 0 and 5: action 0
        d["action"] = act
    if "techniques" in P:
        Pt = P["techniques"]
        w = d_cfg.get("tech_prior_weight", 0.0)
        if w > 0 and "tech" in pri:
            onehot = np.eye(N_CLS)[label]
            Pt = (1 - w) * Pt + w * (onehot @ pri["tech"])
        T = Pt >= d_cfg.get("tech_threshold", 0.5)
        T[benign] = False                                   # classes 0 and 5: techniques NONE
        none_fraud = ~benign & ~T.any(1)                    # a scam uses at least one technique
        T[none_fraud, Pt[none_fraud].argmax(1)] = True
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


def _temper(P, T):
    """Temperature-scale probabilities: p^(1/T), renormalised (argmax unchanged)."""
    if T == 1.0:
        return P
    L = np.log(np.clip(P, 1e-9, 1.0)) / T
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def _fit_temperature(oof, y):
    correct = oof.argmax(1) == y
    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 41))
    return float(min(grid, key=lambda T: np.mean((_temper(oof, T).max(1) - correct) ** 2)))


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
    print(f"[{cfg['name']}] {len(lab)} labelled + {len(extra)} noise-only rows; "
          f"augment copies {(cfg.get('augment') or {}).get('copies', 0)}")

    rows, oof_all, oof_pri, cnt = [], {}, [], np.zeros(len(lab))
    for i, (tr, va) in enumerate(validate.folds(cfg, y_all["label"])):
        fit = _training_frame(cfg, b, lab.iloc[tr], extra, seed=cfg["seed"] + i)
        transforms = _fit_features(cfg, b, fit)
        ft = _targets(fit, cfg, b)
        pri = _priors(ft)
        heads = _fit_heads(cfg, make, _transform(transforms, fit), ft)
        yv = {k: v[va] for k, v in y_all.items()}
        Pv = _predict_heads(heads, _transform(transforms, lab.iloc[va]))
        for k in Pv:
            oof_all.setdefault(k, np.zeros((len(lab), Pv[k].shape[1])))[va] += Pv[k]
        oof_pri.append(pri)
        cnt[va] += 1
        parts = _score(_decode(Pv, cfg, pri, lab[b.text_col].iloc[va].tolist()), yv, cfg["tasks"])
        if robust_eval:
            from src.features import augment
            noisy = augment.augment(lab.iloc[va], b.text_col, copies=1, seed=10_000 + i)
            parts["robust"] = metrics.macro_f1(yv["label"], _decode(_predict_heads(heads, _transform(transforms, noisy)),
                                                                    cfg, pri, noisy[b.text_col].tolist())["label"])
        rows.append(parts)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    comps = [metrics.local_composite(r) for r in rows]
    shown = " ".join(f"{k}={v:.4f}" for k, v in mean.items())
    print(f"[{cfg['name']}] CV local composite {np.mean(comps):.4f} +/- {np.std(comps):.4f} | {shown} ({len(rows)} folds)")

    # Task D: temperature fitted on out-of-fold label probabilities; then the full decoder (priority order, router,
    # confidence of the submitted label, shrink toward OOF accuracy) is scored on OOF -> calib_final
    oof = {k: v / np.maximum(cnt, 1)[:, None] for k, v in oof_all.items()}
    T = _fit_temperature(oof["label"], y_all["label"]) if cfg.get("calibrate", True) else 1.0
    oofT = dict(oof, label=_temper(oof["label"], T))
    pri_avg = {k: np.mean([p[k] for p in oof_pri], 0) for k in oof_pri[0]} if oof_pri else {}
    texts_lab = lab[b.text_col].tolist()
    dec0 = _decode(oofT, cfg, pri_avg, texts_lab)
    acc_oof = float(np.mean(dec0["label"] == y_all["label"]))
    decf = _decode(oofT, cfg, pri_avg, texts_lab, acc_oof=acc_oof)
    calib_final = metrics.calibration(decf["confidence"], decf["label"] == y_all["label"])
    print(f"[{cfg['name']}] temperature T={T:.2f}, OOF acc {acc_oof:.3f}, final-decoder OOF calibration {calib_final:.4f}")
    mean_final = dict(mean, calib=calib_final)
    composite = metrics.local_composite(mean_final)

    # distribution-shift holdouts (language / register): train on the rest, score Task A on the held-out slice
    shift = {}
    for name, (tr, va) in validate.shift_splits(lab[b.text_col]).items():
        fit = _training_frame(cfg, b, lab.iloc[tr], extra, seed=cfg["seed"] + 500)
        transforms = _fit_features(cfg, b, fit)
        ft = _targets(fit, cfg, b)
        heads = _fit_heads(cfg, make, _transform(transforms, fit), ft)
        pred = _decode(_predict_heads(heads, _transform(transforms, lab.iloc[va])), cfg, _priors(ft),
                       lab[b.text_col].iloc[va].tolist())["label"]
        shift[name] = metrics.macro_f1(y_all["label"][va], pred)
    if shift:
        print(f"[{cfg['name']}] shift holdouts (Task A macro-F1): "
              + " ".join(f"{k}={v:.3f}" for k, v in shift.items()) + f" | worst={min(shift.values()):.3f}")

    result = {"exp": cfg["name"], "composite": composite, "composite_std": float(np.std(comps)),
              "shift_worst": min(shift.values()) if shift else float("nan"), "T": T, **mean_final,
              **{f"shift_{k}": v for k, v in shift.items()}}
    print(f"[{cfg['name']}] RESULT " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                for k, v in result.items()))
    if cfg.get("cv_only"):
        return result

    fit = _training_frame(cfg, b, lab, extra, seed=cfg["seed"] + 999)
    transforms = _fit_features(cfg, b, fit)
    ft = _targets(fit, cfg, b)
    pri_full = _priors(ft)
    heads = _fit_heads(cfg, make, _transform(transforms, fit), ft)
    t0, chunks = time.time(), []
    for s in range(0, len(b.test), CHUNK):
        chunks.append(_predict_heads(heads, _transform(transforms, b.test.iloc[s:s + CHUNK])))
    P = {t: np.concatenate([c[t] for c in chunks]) for t in chunks[0]}
    dt = time.time() - t0
    print(f"[{cfg['name']}] inference {len(b.test)} rows in {dt:.1f}s ({len(b.test) / max(dt, 1e-9):,.0f} rows/s, "
          f"{1000 * dt / len(b.test):.3f} ms/row)")
    P["label"] = _temper(P["label"], T)
    dec = _decode(P, cfg, pri_full, b.test[b.text_col].tolist(), acc_oof=acc_oof)

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
                        cv_mean=round(composite, 4), cv_std=round(float(np.std(comps)), 4),
                        score_a=round(mean["A"], 4), score_b=round(mean["B"], 4) if "B" in mean else "",
                        score_c=round(mean["C"], 4) if "C" in mean else "", calib=round(calib_final, 4),
                        shift_min=round(min(shift.values()), 4) if shift else "", temperature=round(T, 3),
                        n_folds=len(rows), note=cfg.get("note", "")))
    return result


if __name__ == "__main__":
    main()
