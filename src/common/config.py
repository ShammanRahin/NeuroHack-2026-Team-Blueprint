"""YAML configs with `base:` inheritance and CLI overrides:
python -m src.run exp=baseline model.params.C=3 debug=true"""
import copy, hashlib, json, pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _merge(a, b):
    a = copy.deepcopy(a)
    for k, v in b.items():
        # same-kind dicts merge; a block that switches `name` (another model) replaces, so old params don't leak
        same = isinstance(v, dict) and isinstance(a.get(k), dict) and v.get("name", a[k].get("name")) == a[k].get("name")
        a[k] = _merge(a[k], v) if same else v
    return a


def _load(name):
    c = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    base = c.pop("base", None)
    return _merge(_load(base), c) if base else c


def load(argv):
    kv = dict(a.split("=", 1) for a in argv)
    cfg = _load(kv.pop("exp"))
    for k, v in kv.items():
        *path, last = k.split(".")
        d = cfg
        for p in path:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(v)
    cfg["_hash"] = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    return cfg
