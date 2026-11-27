"""The shared 1M-row synthetic training set as a function (used by tools/make_synthetic_dataset.py and by the
encoder stage in src/run.py, so the Kaggle notebook rebuilds exactly the dataset in synthetic/, seed 2026).
Nothing here reads test.csv."""
import random
import re

import pandas as pd

from src.data import synth_data
from src.features import augment

REG = [("call", r"^Call transcript:"), ("email", r"^Subject:"), ("group_chat", r"\[Group\]"), ("conversation", r"\|\|\|")]


def register(t):
    for name, pat in REG:
        if re.search(pat, t):
            return name
    return "sms"


def lang(t):
    bn = bool(re.search(r"[\u0980-\u09FF]", t))
    lat = bool(re.search(r"[A-Za-z]{3,}", t))
    if bn and lat:
        return "mixed"
    return "bangla" if bn else synth_data._lang(t).replace("bl", "banglish").replace("en", "english")


def build(n=1_000_000, seed=2026, verbose=True, meta=True):
    """Unique rows, <=2 per digit-masked skeleton, 1% noise-only, 30% corrupted in place; guide-consistent."""
    import types
    a = types.SimpleNamespace(n=n, seed=seed)
    target, chunks, i = int(a.n * 1.0), [], 0
    pool = pd.DataFrame(columns=["text", "label", "techniques", "action", "tpl"])
    while True:
        chunks.append(synth_data.generate(300_000, seed=a.seed + 97 * i))
        i += 1
        pool = pd.concat(chunks, ignore_index=True).drop_duplicates("text")
        sk = pool.text.str.lower().str.replace(r"\d+", "#", regex=True).str.replace(r"\s+", " ", regex=True)
        kept = pool[pool.groupby(sk).cumcount() < 2]
        verbose and print(f"[synth] batches={i} unique={len(pool):,} after skeleton cap={len(kept):,}", flush=True)
        if len(kept) >= target * 0.99 + 20_000 or i >= 12:
            break
    rng = random.Random(a.seed)
    noise_n = int(a.n * 0.01)
    nz = augment.noise_rows(noise_n, "text", ["text", "label"], seed=a.seed)
    noise = pd.DataFrame({"text": nz["text"], "label": 0, "techniques": "NONE", "action": 0, "tpl": "noise"})
    df = pd.concat([kept.sample(frac=1.0, random_state=a.seed).head(a.n - noise_n), noise], ignore_index=True)
    df = df.drop_duplicates("text").sample(frac=1.0, random_state=a.seed + 1).reset_index(drop=True)

    df["corrupted"] = 0
    idx = [k for k in range(len(df)) if df.tpl[k] != "noise" and rng.random() < 0.30]
    df.loc[idx, "text"] = [augment.corrupt(df.text[k], rng, 0.06) for k in idx]
    df.loc[idx, "corrupted"] = 1
    df = df.drop_duplicates("text").reset_index(drop=True)
    if meta:
        df["register"] = df.text.map(register)
        df["lang"] = df.text.map(lang)
    df.insert(0, "id", range(1, len(df) + 1))
    df = df.rename(columns={"tpl": "template"})
    # guide consistency checks
    benign = df.label.isin([0, 5])
    assert (df.loc[benign, "techniques"] == "NONE").all() and (df.loc[benign, "action"] == 0).all()
    assert (df.loc[~benign, "techniques"] != "NONE").all() and (df.loc[~benign, "action"] > 0).all()

    return df
