"""Drop a new file in src/features or src/models and it is auto-registered. No central list to edit."""
import importlib, pkgutil

FEATURES, MODELS = {}, {}


def feature(name):
    def deco(fn):
        FEATURES[name] = fn
        return fn
    return deco


def model(name):
    def deco(cls):
        MODELS[name] = cls
        return cls
    return deco


def load_all():
    for pkg in ("src.features", "src.models"):
        m = importlib.import_module(pkg)
        for _, n, _ in pkgutil.iter_modules(m.__path__):
            importlib.import_module(f"{pkg}.{n}")
