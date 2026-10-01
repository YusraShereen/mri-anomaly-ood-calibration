import os

import numpy as np
import pandas as pd


def make_synthetic_scores(out_dir, seed=42, n=1500):
    # anomaly scores for a source validation set and shifted target domains. Smoke test only.
    rng = np.random.default_rng(seed)
    domains = {  # name: (normal_mean, normal_std, anomaly_mean, anomaly_std)
        "source_val": (0.0, 1.0, 2.5, 1.2),
        "scanner_B": (0.8, 1.2, 3.2, 1.4),
        "scanner_C": (1.5, 1.5, 3.5, 1.8),
    }
    os.makedirs(out_dir, exist_ok=True)
    for name, (mn, sn, ma, sa) in domains.items():
        y = (rng.random(n) < 0.3).astype(int)
        s = np.where(y == 1, rng.normal(ma, sa, n), rng.normal(mn, sn, n))
        pd.DataFrame({"score": s, "label": y}).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
