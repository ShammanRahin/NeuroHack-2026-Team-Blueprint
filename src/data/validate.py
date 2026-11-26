"""Member A's data loading, validation, and cross-validation."""
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

from src.common import env
from src.common.contracts import Bundle
from src.data import normalize


_CV_STATE = "_data_cv_state"
TECHNIQUES = [f"T{i}" for i in range(1, 7)]
_NEAR_DUPLICATE_SIMILARITY = 0.90
_LONG_TEXT_CHARS = 2000
_MAX_SPLIT_ATTEMPTS = 64


def _clean_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _has_visible_content(text):
    return any(unicodedata.category(char)[0] in "LNPS" for char in text)


def _writing_scripts(text):
    scripts = set()
    for char in text:
        if unicodedata.category(char).startswith("L"):
            name = unicodedata.name(char, "")
            if name:
                scripts.add(name.split()[0])
    return scripts


def _check_text(name, df, text_col):
    raw = df[text_col]
    cleaned = raw.fillna("").astype(str).map(_clean_text)
    empty = ~cleaned.map(_has_visible_content)
    if empty.any():
        raise ValueError(
            f"{name}: {empty.sum()} empty texts after cleaning ({raw.isna().sum()} missing)"
        )
    scripts = cleaned.map(_writing_scripts)
    mixed = scripts.map(len).gt(1)
    if mixed.any():
        print(f"[data] WARN {name}: {mixed.sum()} texts contain letters from multiple scripts")
    lengths = cleaned.str.len()
    long = lengths.gt(_LONG_TEXT_CHARS)
    if long.any():
        print(f"[data] WARN {name}: {long.sum()} texts exceed {_LONG_TEXT_CHARS} characters")
    script_counts = scripts.map(lambda values: "+".join(sorted(values)) or "no letters").value_counts()
    print(
        f"[data] {name}: {raw.isna().sum()} missing texts, "
        f"{cleaned.duplicated().sum()} duplicate rows by cleaned text, "
        f"scripts {script_counts.to_dict()}, cleaned characters "
        f"min/median/p95/max {lengths.min()}/{lengths.median():g}/{lengths.quantile(.95):g}/{lengths.max()}"
    )
    df[text_col] = cleaned.map(normalize.robust)  # stats above are taken BEFORE de-obfuscation


def _check_ids(ids, name):
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name}: missing ids")
    if ids.duplicated().any():
        raise ValueError(f"{name}: duplicate ids")


def _check_labels(labels, name):
    if (
        labels.isna().any()
        or not pd.api.types.is_integer_dtype(labels)
        or not labels.between(0, 5).all()
    ):
        raise ValueError(f"{name}: labels must be integers from 0 to 5")


def _duplicate_key(text):
    normalized = " ".join(
        "".join(char if unicodedata.category(char)[0] in "LMN" else " " for char in text.casefold()).split()
    )
    # Preserve distinctions between messages made entirely of symbols or punctuation.
    return normalized or text.casefold()


def _duplicate_groups(texts):
    keys, row_to_key = np.unique([_duplicate_key(text) for text in texts], return_inverse=True)
    parents = np.arange(len(keys))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    # Exact normalized keys already share an index. Link near neighbors transitively.
    eligible = np.flatnonzero([len(key) >= 3 for key in keys])
    if len(eligible) > 1:
        vectors = CountVectorizer(analyzer="char", ngram_range=(3, 5), binary=True).fit_transform(keys[eligible])
        neighbors = NearestNeighbors(
            radius=1 - _NEAR_DUPLICATE_SIMILARITY + 1e-12, metric="cosine", algorithm="brute"
        ).fit(vectors)
        for start in range(0, len(eligible), 128):
            batch = neighbors.radius_neighbors(vectors[start:start + 128], return_distance=False)
            for offset, matches in enumerate(batch, start):
                for match in matches[matches > offset]:
                    first, second = root(eligible[offset]), root(eligible[match])
                    parents[max(first, second)] = min(first, second)

    return np.array([root(index) for index in row_to_key], dtype=int)


def _parse_techniques(series):
    """'T1|T2' / 'NONE' -> DataFrame of 0/1 columns T1..T6."""
    out = pd.DataFrame(0, index=series.index, columns=TECHNIQUES, dtype=int)
    for idx, value in series.items():
        value = str(value).strip()
        if value == "NONE":
            continue
        codes = value.split("|")
        bad = [c for c in codes if c not in TECHNIQUES]
        if bad:
            raise ValueError(f"train: invalid technique codes {bad} (row {idx})")
        out.loc[idx, codes] = 1
    return out


def _aux_targets(train, data):
    aux = {}
    if data.get("action_col"):
        action = train[data["action_col"]]
        if action.isna().any() or not pd.api.types.is_integer_dtype(action) or not action.between(0, 5).all():
            raise ValueError("train: action must be integers 0..5")
        aux["action"] = action.to_numpy()
    if data.get("techniques_col"):
        aux["techniques"] = _parse_techniques(train[data["techniques_col"]]).to_numpy()
    return aux


def _check_sample(sample, test, id_col, target_col):
    if not sample.columns.is_unique or not {id_col, target_col}.issubset(sample.columns):
        raise ValueError("sample_submission: expected unique columns including the configured id and label")
    _check_ids(sample[id_col], "sample_submission")
    if set(sample[id_col]) != set(test[id_col]):
        raise ValueError("test ids do not match sample_submission ids")


def load_clean(cfg) -> Bundle:
    cfg.pop(_CV_STATE, None)
    directory, data = env.data_dir(cfg), cfg["data"]
    train = pd.read_csv(directory / data["train"])
    test = pd.read_csv(directory / data["test"])
    sample_path = directory / data["sample_submission"] if data.get("sample_submission") else None
    sample = pd.read_csv(sample_path) if sample_path and sample_path.exists() else None
    id_col, target_col, text_col = data["id_col"], data["target_col"], data.get("text_col")

    for name, frame, columns in (("train", train, [id_col, target_col]), ("test", test, [id_col])):
        required = columns + ([text_col] if text_col else [])
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")
        if frame.empty:
            raise ValueError(f"{name}: no records")
        _check_ids(frame[id_col], name)

    _check_labels(train[target_col], "train")
    print(f"[data] train: {len(train)} rows, label counts {train[target_col].value_counts().sort_index().to_dict()}")
    print(f"[data] test: {len(test)} rows")
    if text_col:
        _check_text("train", train, text_col)
        _check_text("test", test, text_col)
        duplicate_labels = train.groupby(text_col)[target_col].nunique()
        conflicts = duplicate_labels.gt(1).sum()
        if conflicts:
            raise ValueError(f"train: {conflicts} identical texts with conflicting labels")
        overlap = len(set(train[text_col]) & set(test[text_col]))
        if overlap:
            print(f"[data] WARN {overlap} cleaned texts appear in both train and test")
    if sample is not None:
        _check_sample(sample, test, id_col, target_col)

    # Keep duplicate components intact even if debug mode removes bridging rows.
    groups = _duplicate_groups(train[text_col]) if text_col else np.arange(len(train))
    if cfg.get("debug"):
        train = train.groupby(target_col, group_keys=False).head(20)
        groups = groups[train.index.to_numpy()]
        train = train.reset_index(drop=True)
        test = test.head(50).reset_index(drop=True)
        sample = None
    cfg[_CV_STATE] = (train[target_col].to_numpy(copy=True), groups)
    return Bundle(train, test, id_col, target_col, text_col, sample, aux=_aux_targets(train, data))


def folds(cfg, y):
    """Return deterministic stratified splits with a disjoint set of groups in each fold."""
    y = np.asarray(y)
    labels, groups = cfg.get(_CV_STATE, (y, np.arange(len(y))))
    if not np.array_equal(y, labels):
        raise ValueError("CV labels do not match the row order from load_clean")
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("CV requires at least two observed classes")

    cv = cfg["cv"]
    requested, repeats = cv["n_splits"], cv["n_repeats"]
    if type(requested) is not int or requested < 2 or type(repeats) is not int or repeats < 1:
        raise ValueError("CV requires integer n_splits >= 2 and n_repeats >= 1")
    group_counts = {int(label): len(np.unique(groups[y == label])) for label in classes}
    upper = min(requested, *group_counts.values())
    print(f"[data] CV: {len(np.unique(groups))} distinct groups, groups per class {group_counts}")

    rng = np.random.RandomState(cfg["seed"])
    seeds = rng.randint(0, np.iinfo(np.int32).max, size=_MAX_SPLIT_ATTEMPTS * repeats)
    for n_splits in range(upper, 1, -1):
        splits = []
        for seed in seeds:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            candidate = list(splitter.split(y, y, groups))
            if all(
                len(np.unique(y[train])) == len(classes) and len(np.unique(y[valid])) == len(classes)
                for train, valid in candidate
            ):
                splits.extend(candidate)
                if len(splits) == n_splits * repeats:
                    if n_splits < requested:
                        print(f"[data] WARN CV reduced from {requested} to {n_splits} folds per repeat")
                    return splits
    raise ValueError(f"CV could not form group-disjoint folds covering every class: {group_counts}")


def validate_submission(sub, b: Bundle):
    """Validate the output against the sample schema when present and the actual test IDs."""
    columns = [b.id_col, b.target_col] + [c for c in ("techniques", "action", "confidence") if c in sub.columns]
    if b.sample_sub is not None:
        _check_sample(b.sample_sub, b.test, b.id_col, b.target_col)
        columns = list(b.sample_sub.columns)
    if list(sub.columns) != columns:
        raise ValueError(f"submission columns {list(sub.columns)} != {columns}")
    _check_ids(sub[b.id_col], "submission")
    if set(sub[b.id_col]) != set(b.test[b.id_col]):
        raise ValueError("submission ids do not match test ids")
    _check_labels(sub[b.target_col], "submission")
    if "techniques" in sub.columns:
        if sub["techniques"].isna().any() or sub["techniques"].astype(str).str.strip().eq("").any():
            raise ValueError("submission: techniques must never be blank (use NONE)")
        for value in sub["techniques"].astype(str).unique():
            codes = value.split("|")
            if value != "NONE" and (not set(codes) <= set(TECHNIQUES) or "NONE" in codes):
                raise ValueError(f"submission: invalid techniques value {value!r}")
    if "action" in sub.columns:
        _check_labels(sub["action"], "submission action")
    if "confidence" in sub.columns:
        conf = pd.to_numeric(sub["confidence"], errors="coerce")
        if conf.isna().any() or not conf.between(0, 1).all():
            raise ValueError("submission: confidence must be a number in [0, 1]")
