import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.common.registry import feature


@feature("tfidf_word")
def tfidf_word(bundle, max_features=20000):
    vec = TfidfVectorizer(max_features=max_features, sublinear_tf=True, dtype=np.float32).fit(bundle.train[bundle.text_col])
    return lambda df: vec.transform(df[bundle.text_col])
