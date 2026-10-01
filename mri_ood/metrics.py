import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .utils import ece_binary


def rates(scores, labels, thr):
    pred = scores > thr
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return tpr, fpr, f1


def domain_metrics(scores, labels, thr_fixed, thr_adapt, probs):
    tf, ff, f1f = rates(scores, labels, thr_fixed)
    ta, fa, f1a = rates(scores, labels, thr_adapt)
    return dict(
        auroc=float(roc_auc_score(labels, scores)), auprc=float(average_precision_score(labels, scores)),
        ece=ece_binary(probs, labels), brier=float(brier_score_loss(labels, probs)),
        tpr_fixed=tf, fpr_fixed=ff, f1_fixed=f1f, tpr_adapt=ta, fpr_adapt=fa, f1_adapt=f1a)
