"""MEMBER A. Data loading, cleaning, checks, folds, and the submission validator.
Fail loudly here so bad data never reaches features/models."""
import unicodedata
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from src.common import env
from src.common.contracts import Bundle


def _clean_text(s):
    return " ".join(unicodedata.normalize("NFKC", str(s)).split())


def load_clean(cfg) -> Bundle:
    d, c = env.data_dir(cfg), cfg["data"]
    train, test = pd.read_csv(d / c["train"]), pd.read_csv(d / c["test"])
    sub = pd.read_csv(d / c["sample_submission"]) if c.get("sample_submission") and (d / c["sample_submission"]).exists() else None
    idc, tc, xc = c["id_col"], c["target_col"], c.get("text_col")

    for name, df, need in (("train", train, [idc, tc]), ("test", test, [idc])):
        miss = [x for x in need + ([xc] if xc else []) if x not in df.columns]
        if miss:
            raise ValueError(f"{name}: missing columns {miss}")
        if df[idc].duplicated().any():
            raise ValueError(f"{name}: duplicate ids")
    if train[tc].isna().any():
        raise ValueError("train: null labels")
    if xc:
        for df in (train, test):
            df[xc] = df[xc].fillna("").map(_clean_text)
        dup = train[train.duplicated(xc, keep=False)].groupby(xc)[tc].nunique()
        if (dup > 1).any():
            raise ValueError(f"train: identical text with conflicting labels: {list(dup[dup > 1].index[:3])}")
        leak = set(train[xc]) & set(test[xc])
        if leak:
            print(f"[data] WARN {len(leak)} texts appear in both train and test")
    if sub is not None and set(sub[idc]) != set(test[idc]):
        raise ValueError("test ids do not match sample_submission ids")

    if cfg.get("debug"):
        train = train.groupby(tc, group_keys=False).head(20).reset_index(drop=True)
        test = test.head(50).reset_index(drop=True)
        sub = None
    return Bundle(train, test, idc, tc, xc, sub)


def folds(cfg, y):
    """Fixed seed => identical folds for every member, on every machine."""
    cv = cfg["cv"]
    return list(RepeatedStratifiedKFold(n_splits=cv["n_splits"], n_repeats=cv["n_repeats"],
                                        random_state=cfg["seed"]).split(y, y))


def validate_submission(sub, b: Bundle):
    """Run before ANY upload."""
    if b.sample_sub is None:
        return
    if list(sub.columns) != list(b.sample_sub.columns):
        raise ValueError(f"columns {list(sub.columns)} != {list(b.sample_sub.columns)}")
    if len(sub) != len(b.sample_sub) or set(sub[b.id_col]) != set(b.sample_sub[b.id_col]) or sub[b.id_col].duplicated().any():
        raise ValueError("submission ids missing/extra/duplicated")
    if sub[b.target_col].isna().any() or not pd.api.types.is_integer_dtype(sub[b.target_col]):
        raise ValueError("labels must be integers (no blanks, no 2.0, no strings)")
    if not set(sub[b.target_col]) <= set(b.train[b.target_col]):
        raise ValueError("submission has labels not seen in training data")
