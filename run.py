#!/usr/bin/env python3
"""
Decode Neuro -- closed-loop tVNS speech-rehab EEG decoder.
End-to-end runner: load -> evaluate (leakage-free) -> fit -> save -> latency.

Trains on real recorded EEG only. There is no synthetic backend: a decoder that
gates stimulation must be fit on real recordings, and its accuracy must mean
something.

Two sources:

  --calibration   your own same-session recordings from calibrate.py. Use this
                  for --task go. Attempt and rest are interleaved in one
                  recording, so the score reflects speech attempt.
  --data          the ds003626 (BIDS) root. Fine for --task word, but its only
                  rest is a separate baseline block, which confounds --task go
                  (0.80 vs 0.54 at a matched window; see README).

    pip install -r requirements.txt

    python calibrate.py --port <serial> --subject 1 --channel-names ...
    python run.py --calibration 'calib/*.npz' --task go

    openneuro-py download --dataset ds003626 --target-dir ds003626
    python run.py --data ./ds003626 --task word --condition inner

Outputs (in ./outputs), suffixed per task: model_<task>.joblib,
metrics_<task>.json, and plots.

The training itself lives in eeg_tvns/training.py, which the dashboard's Train tab
also calls -- so a model trained here and one trained there are identical.
"""
from __future__ import annotations

import argparse
import logging

from eeg_tvns.config import Config
from eeg_tvns.training import train


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", default=None,
                    help="path to the ds003626 root (BIDS) of real recorded EEG.")
    src.add_argument("--calibration", default=None,
                    help="glob of same-session recordings from calibrate.py, e.g. "
                         "'calib/*.npz'. Preferred for --task go: ds003626's rest "
                         "is a separate baseline block and confounds that score.")
    p.add_argument("--task", choices=["go", "word"], default="go",
                   help="'go' = binary speech-attempt vs rest (drives tVNS); 'word' = 4-class.")
    p.add_argument("--condition", choices=["inner", "pronounced", "visualized", "overt_scaffold"],
                   default="inner", help="speech condition to decode (word task).")
    p.add_argument("--classifier", choices=["lda", "logreg", "mdm", "svm"], default="lda")
    p.add_argument("--no-align", action="store_true", help="disable Riemannian Alignment (ablation).")
    p.add_argument("--filter-bank", action="store_true", help="use mu/beta/low-gamma filter-bank covariances.")
    p.add_argument("--all-channels", action="store_true", help="use all channels (default: low-density montage).")
    p.add_argument("--subjects", type=int, nargs="*", default=None, help="subset of subject ids.")
    p.add_argument("--permutations", type=int, default=50,
                   help="label-shuffle permutations for empirical chance (0 to skip; this is the slow step). "
                        "Raise to 200+ for a publication-grade p-value.")
    p.add_argument("--out", default="outputs", help="output directory.")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    cfg = Config(
        data_root=a.data,
        calibration_glob=a.calibration,
        subjects=a.subjects,
        task=a.task,
        condition=a.condition,
        classifier=a.classifier,
        align=not a.no_align,
        filter_bank=a.filter_bank,
        use_low_density=not a.all_channels,
        n_permutations=a.permutations,
        out_dir=a.out,
        make_plots=not a.no_plots,
        random_state=a.seed,
    )
    cfg.validate()
    return cfg


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = parse_args()
    res = train(cfg)

    for report in res.reports:
        print(report)
    perm = res.metrics.get("permutation") or {}
    if perm.get("n_permutations"):
        print(f"[permutation] observed={perm['observed']:.3f}  "
              f"null={perm['null_mean']:.3f}+/-{perm['null_std']:.3f}  "
              f"p={perm['p_value']:.4f}")
    lat = res.metrics.get("latency") or {}
    if lat:
        print(f"[latency] median={lat['median_ms']:.2f} ms  p95={lat['p95_ms']:.2f} ms  "
              f"within {cfg.latency_budget_ms:g} ms: "
              f"{lat['fraction_within_budget']*100:.1f}%")

    print("\n" + "=" * 64)
    print(f"Saved model  -> {res.model_path}")
    print(f"Saved metrics-> {res.metrics_path}")
    for pth in res.plots:
        print(f"Saved plot   -> {pth}")
    if res.verdict:
        print(f"Gating       -> {res.verdict['label']}: {res.verdict['detail']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
