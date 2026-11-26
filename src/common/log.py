"""Shared experiment log. results/log.csv is committed; .gitattributes merge=union keeps appends conflict-free."""
import csv, os, pathlib, subprocess
from datetime import datetime, timezone
from .config import ROOT

FIELDS = ["time", "user", "platform", "exp", "cfg_hash", "git", "cv_mean", "cv_std", "score_a", "score_b", "score_c", "calib", "shift_min", "temperature", "n_folds", "note"]


def _sh(*a):
    try:
        return subprocess.check_output(a, cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "?"


def append(row):
    p = pathlib.Path(os.environ.get("RESULTS_LOG", ROOT / "results" / "log.csv"))
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists() or p.stat().st_size == 0
    row = {"time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
           "user": os.environ.get("EXP_USER") or _sh("git", "config", "user.name"),
           "git": _sh("git", "rev-parse", "--short", "HEAD") + ("*" if _sh("git", "status", "--porcelain") else ""),
           **row}
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    return p
