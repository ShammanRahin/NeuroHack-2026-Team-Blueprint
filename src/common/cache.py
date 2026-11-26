"""Cache keyed by (params, data fingerprint, SOURCE CODE of the feature) so edits never serve stale results."""
import hashlib, inspect, json, pathlib
import joblib
import pandas as pd
from . import env


def key(fn, name, params, bundle):
    fp = int(pd.util.hash_pandas_object(pd.concat([bundle.train, bundle.test]), index=False).sum())
    src = pathlib.Path(inspect.getsourcefile(fn)).read_bytes()
    blob = json.dumps([name, params, fp], sort_keys=True).encode() + src
    return hashlib.sha1(blob).hexdigest()[:16]


def get_or_compute(k, compute):
    p = env.work_dir() / "cache" / f"{k}.joblib"
    if p.exists():
        return joblib.load(p)
    out = compute()
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, p)
    return out
