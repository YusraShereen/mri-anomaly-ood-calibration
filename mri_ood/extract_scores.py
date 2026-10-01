# Template: run a trained scorer over each dataset/domain and write scores/<domain>.csv (columns: score,label).
# Domains to build (healthy = label 0, pathological = label 1):
#   source_val : held-out split of the training distribution (e.g. IXI healthy + a BraTS-style anomaly set)
#   <domain>   : each target domain, e.g. a different scanner/site, field strength, or dataset
# Fill in the dataset readers for your preprocessing, then:  python -m mri_ood.extract_scores --model vit --ckpt path
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["vit", "fanogan"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.parse_args()
    raise NotImplementedError("implement dataset readers for IXI / BraTS / fastMRI and call scorer.score()")


if __name__ == "__main__":
    main()
