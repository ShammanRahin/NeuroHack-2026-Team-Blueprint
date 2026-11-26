"""THE interfaces between the three roles. Change only by PR that all 3 approve.

DATA     (Member A)  src/data/validate.py : load_clean(cfg) -> Bundle
FEATURES (Member B)  src/features/*.py    : @feature("name") def f(bundle, **params) -> matrix
                       rows = train rows FIRST, then test rows (n_train + n_test), dense or scipy sparse
MODELS   (Member C)  src/models/*.py      : @model("name") class M: __init__(seed, **params); fit(X, y);
                       predict_proba(X) -> (n, n_classes), columns ordered by sorted class labels
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Bundle:
    train: pd.DataFrame
    test: pd.DataFrame
    id_col: str
    target_col: str
    text_col: Optional[str]
    sample_sub: Optional[pd.DataFrame]
