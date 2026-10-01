# Interface for the anomaly detectors you already built (ViT-based and f-AnoGAN).
# Wire your existing code in here, then run extract_scores to write scores/<domain>.csv.
import numpy as np


class AnomalyScorer:
    name = "base"

    def score(self, slices):
        # slices: array (N, H, W) of preprocessed MRI slices -> anomaly scores (N,), higher = more anomalous
        raise NotImplementedError


class ViTScorer(AnomalyScorer):
    name = "vit"

    def __init__(self, checkpoint):
        raise NotImplementedError("load your trained ViT anomaly model from `checkpoint` and implement score()")


class FAnoGANScorer(AnomalyScorer):
    name = "fanogan"

    def __init__(self, checkpoint):
        raise NotImplementedError("load your trained f-AnoGAN (generator+encoder) and implement score()")
