import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .calibration import apply_platt, fit_platt  # noqa: E402
from .metrics import domain_metrics  # noqa: E402
from .synthetic import make_synthetic_scores  # noqa: E402
from .thresholds import adaptive_threshold, fit_adaptive_k, fixed_threshold  # noqa: E402
from .utils import md_table, plot_reliability, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    set_seed(cfg.get("seed", 42))
    sdir, out = cfg["scores_dir"], cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    if cfg.get("synthetic"):
        make_synthetic_scores(sdir, cfg.get("seed", 42))

    src = pd.read_csv(os.path.join(sdir, cfg["source_domain"] + ".csv"))
    s_src, y_src = src.score.values, src.label.values
    fpr = cfg.get("target_fpr", 0.05)
    thr_fixed = fixed_threshold(s_src, y_src, fpr)
    k = fit_adaptive_k(s_src, y_src, fpr)
    platt = fit_platt(s_src, y_src)

    rows, curves = [], {}
    for f in sorted(os.listdir(sdir)):
        if not f.endswith(".csv"):
            continue
        name = f[:-4]
        d = pd.read_csv(os.path.join(sdir, f))
        s, y = d.score.values, d.label.values
        probs = apply_platt(platt, s)
        thr_a = adaptive_threshold(s, k)
        m = domain_metrics(s, y, thr_fixed, thr_a, probs)
        m.update(domain=name, n=len(d), thr_fixed=thr_fixed, thr_adapt=thr_a)
        rows.append(m)
        curves[name] = (probs, y)
    res = pd.DataFrame(rows).set_index("domain").reset_index()
    res.to_csv(os.path.join(out, "ood_results.csv"), index=False)
    with open(os.path.join(out, "results_table.md"), "w") as fh:
        fh.write(md_table(res[["domain", "n", "auroc", "auprc", "ece", "brier", "tpr_fixed", "fpr_fixed",
                               "tpr_adapt", "fpr_adapt"]]))

    plot_reliability(curves, os.path.join(out, "reliability_by_domain.png"), "Calibration under shift (Platt fit on source)")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(res))
    ax[0].bar(x, res.auroc)
    ax[0].set_xticks(x, res.domain, rotation=20)
    ax[0].set_ylabel("AUROC")
    ax[0].set_title("Discrimination under shift")
    ax[1].bar(x - 0.2, res.fpr_fixed, 0.4, label="fixed threshold")
    ax[1].bar(x + 0.2, res.fpr_adapt, 0.4, label="adaptive threshold")
    ax[1].axhline(fpr, color="k", ls="--", lw=1, label=f"target FPR={fpr}")
    ax[1].set_xticks(x, res.domain, rotation=20)
    ax[1].set_ylabel("false-positive rate")
    ax[1].set_title("Threshold stability under shift")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "shift_main.png"), dpi=200)
    print(md_table(res[["domain", "auroc", "ece", "fpr_fixed", "fpr_adapt"]]))


if __name__ == "__main__":
    main()
