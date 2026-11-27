import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.common.registry import feature


@feature("tfidf_word")
def tfidf_word(bundle, max_features=20000, fit_rows=120000):
    txt = bundle.train[bundle.text_col]
    if fit_rows and len(txt) > fit_rows:
        txt = txt.sample(fit_rows, random_state=0)
    vec = TfidfVectorizer(max_features=max_features, sublinear_tf=True, dtype=np.float32).fit(txt)
    return lambda df: vec.transform(df[bundle.text_col])
