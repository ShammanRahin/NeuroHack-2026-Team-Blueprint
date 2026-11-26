import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.common.registry import feature


@feature("tfidf_word")
def tfidf_word(bundle, max_features=20000):
    text = np.concatenate([bundle.train[bundle.text_col], bundle.test[bundle.text_col]])  # train rows first
    return TfidfVectorizer(max_features=max_features, sublinear_tf=True).fit_transform(text)
