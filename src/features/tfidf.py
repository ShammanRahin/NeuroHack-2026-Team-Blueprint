"""MEMBER B. One file per feature group. Add a file + a YAML entry; touch nothing else.
Contract: FIT ON bundle.train ONLY, return transform(df) -> matrix (test text must never be used to fit anything)."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.common.registry import feature


@feature("tfidf_char")
def tfidf_char(bundle, ngram=(2, 5), max_features=50000, fit_rows=80000):
    """fit_rows: the vocabulary/idf is fitted on at most this many training rows (seeded sample); counting every
    char n-gram of a few hundred thousand long synthetic texts before pruning runs out of RAM on Kaggle."""
    txt = bundle.train[bundle.text_col]
    if fit_rows and len(txt) > fit_rows:
        txt = txt.sample(fit_rows, random_state=0)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(ngram), max_features=max_features,
                          sublinear_tf=True, dtype=np.float32).fit(txt)
    return lambda df: vec.transform(df[bundle.text_col])
