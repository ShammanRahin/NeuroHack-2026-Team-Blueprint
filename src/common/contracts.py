"""THE interfaces between the three roles. Change only by PR that all 3 approve.

DATA     (Member A)  src/data/validate.py : load_clean(cfg) -> Bundle
FEATURES (Member B)  src/features/*.py    : @feature("name") def f(bundle, **params) -> transform
                       FIT ON bundle.train ONLY (the Kaggle rules prohibit any use of test text for training/fitting).
                       Return transform(df) -> matrix (dense or scipy sparse, one row per row of df). The runner calls it
                       on the train frame and then on test chunks (the test set has 500,000 rows: stay fast and low-memory).
                       The runner fits features INSIDE EVERY CV FOLD, so supervised features are safe.
MODELS   (Member C)  src/models/*.py      : @model("name") class M: __init__(seed, **params); fit(X, y);
                       predict_proba(X) -> (n, n_classes), columns ordered by sorted class labels
"""
from dataclasses import dataclass, field
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
    aux: dict = field(default_factory=dict)  # extra train targets: 'action' (n,), 'techniques' (n, 6) 0/1
