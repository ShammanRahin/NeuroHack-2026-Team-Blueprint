"""MEMBER B. One file per feature group. Add a file + a YAML entry; touch nothing else."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.common.registry import feature


@feature("tfidf_char")
def tfidf_char(bundle, ngram=(2, 5), max_features=50000):
    text = np.concatenate([bundle.train[bundle.text_col], bundle.test[bundle.text_col]])  # train rows first
    return TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(ngram), max_features=max_features,
                           sublinear_tf=True).fit_transform(text)
