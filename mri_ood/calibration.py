import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_platt(src_scores, src_labels):
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(np.asarray(src_scores).reshape(-1, 1), src_labels)
    return lr


def apply_platt(lr, scores):
    return lr.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]
