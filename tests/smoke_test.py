"""Whole pipeline on synthetic data. CI + every member runs this before a PR."""
import os, pathlib, sys, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import run


def make(d, multi=False):
    rng = np.random.default_rng(0)
    words = {k: [f"w{k}{i}" for i in range(8)] for k in range(3)}
    mk = lambda ids, ys: pd.DataFrame({"id": ids, "text": [" ".join(rng.choice(words[y], 5)) for y in ys]})
    ytr = np.repeat(np.arange(3), 20)
    tr = mk(range(60), ytr).assign(label=ytr)
    te = mk(range(100, 130), rng.integers(0, 3, 30))
    sub = pd.DataFrame({"id": te["id"], "label": 0})
    if multi:  # competition format: techniques + action + confidence
        tr["techniques"] = np.array(["NONE", "T1|T2", "T4"])[ytr]
        tr["action"] = np.array([0, 1, 3])[ytr]
        sub = sub.assign(techniques="NONE", action=0, confidence=0.5)
    tr.to_csv(d / "train.csv", index=False)
    te.to_csv(d / "test.csv", index=False)
    sub.to_csv(d / "sample_submission.csv", index=False)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        make(t)
        os.environ.update(WORK_DIR=str(t / "work"), RESULTS_LOG=str(t / "log.csv"), EXP_USER="ci")
        s = run.main(["exp=baseline", f"data.dir={t}"])
        assert s["composite"] > 0.9, f"baseline should solve the toy task, got {s}"
        sub = pd.read_csv(t / "work/outputs/baseline/submission.csv")
        assert list(sub.columns) == ["id", "label"] and len(sub) == 30
        assert (t / "log.csv").exists()
        run.main(["exp=baseline", f"data.dir={t}", "debug=true"])
    with tempfile.TemporaryDirectory() as t:  # 4-task competition format + augmentation + handcrafted
        t = pathlib.Path(t)
        make(t, multi=True)
        os.environ.update(WORK_DIR=str(t / "work"), RESULTS_LOG=str(t / "log.csv"), EXP_USER="ci")
        s = run.main(["exp=smoke_multi", f"data.dir={t}"])
        sub = pd.read_csv(t / "work/outputs/smoke_multi/submission.csv")
        assert list(sub.columns) == ["id", "label", "techniques", "action", "confidence"] and len(sub) == 30
        assert sub["techniques"].notna().all() and sub["confidence"].between(0, 1).all()
        benign = sub["label"].isin([0, 5])
        assert (sub.loc[benign, "techniques"] == "NONE").all() and (sub.loc[benign, "action"] == 0).all()
        assert (sub.loc[~benign, "techniques"] != "NONE").all() and (sub.loc[~benign, "action"] > 0).all()
    print("SMOKE OK")
