import numpy as np


def fixed_threshold(src_scores, src_labels, target_fpr=0.05):
    # threshold set once on source validation normals so that FPR = target_fpr
    return float(np.quantile(src_scores[src_labels == 0], 1 - target_fpr))


def robust_stats(x):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)) * 1.4826 + 1e-8)
    return med, mad


def fit_adaptive_k(src_scores, src_labels, target_fpr=0.05):
    # adaptive rule: thr_domain = median + k * MAD of that domain's own (unlabeled) scores.
    # k is calibrated once on the source domain so it reproduces the fixed threshold there.
    thr = fixed_threshold(src_scores, src_labels, target_fpr)
    med, mad = robust_stats(src_scores)
    return (thr - med) / mad


def adaptive_threshold(target_scores, k):
    med, mad = robust_stats(target_scores)
    return med + k * mad
# NOTE: swap adaptive_threshold for your SAAD thresholding rule to link this repo to that paper.
