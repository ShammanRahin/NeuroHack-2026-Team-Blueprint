"""python -m src.run exp=baseline [debug=true] [model.params.C=3]
Identical on Kaggle, Colab and laptop."""
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from src.common import cache, config, env, log, metrics, registry
from src.data import validate


def build_features(cfg, b):
    mats = []
    for f in cfg["features"]:
        fn, params = registry.FEATURES[f["name"]], f.get("params", {})
        k = cache.key(fn, f["name"], params, b)
        mats.append(cache.get_or_compute(k, lambda: fn(b, **params)))
    X = sp.hstack(mats).tocsr() if any(sp.issparse(m) for m in mats) else np.hstack(mats)
    assert X.shape[0] == len(b.train) + len(b.test), "feature rows must be train rows then test rows"
    return X


def main(argv=None):
    cfg = config.load(sys.argv[1:] if argv is None else argv)
    registry.load_all()
    b = validate.load_clean(cfg)
    n = len(b.train)
    X = build_features(cfg, b)
    Xtr, Xte = X[:n], X[n:]
    y = b.train[b.target_col].to_numpy()
    classes = np.unique(y)
    M = registry.MODELS[cfg["model"]["name"]]
    mk = lambda: M(seed=cfg["seed"], **cfg["model"].get("params", {}))

    fl = validate.folds(cfg, y)
    oof, cnt, scores = np.zeros((n, len(classes))), np.zeros(n), []
    for tr, va in fl:
        p = mk().fit(Xtr[tr], y[tr]).predict_proba(Xtr[va])
        oof[va] += p
        cnt[va] += 1
        scores.append(metrics.score(y[va], classes[p.argmax(1)]))
    oof /= cnt[:, None]
    print(f"[{cfg['name']}] CV macro-F1 {np.mean(scores):.4f} +/- {np.std(scores):.4f}  ({len(scores)} folds)")

    test_p = mk().fit(Xtr, y).predict_proba(Xte)
    out = env.work_dir() / "outputs" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof.npy", oof)            # for blending across members' models
    np.save(out / "test_proba.npy", test_p)
    sub = pd.DataFrame({b.id_col: b.test[b.id_col], b.target_col: classes[test_p.argmax(1)]})
    if not cfg.get("debug"):
        validate.validate_submission(sub, b)
        sub.to_csv(out / "submission.csv", index=False)
        print(f"[{cfg['name']}] wrote {out / 'submission.csv'} (validated)")
        log.append(dict(platform=env.platform(), exp=cfg["name"], cfg_hash=cfg["_hash"],
                        cv_mean=round(float(np.mean(scores)), 4), cv_std=round(float(np.std(scores)), 4),
                        n_folds=len(scores), note=cfg.get("note", "")))
    return float(np.mean(scores))


if __name__ == "__main__":
    main()
