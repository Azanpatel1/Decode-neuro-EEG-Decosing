#!/usr/bin/env python3
"""
Decode Neuro -- closed-loop tVNS speech-rehab EEG decoder.
End-to-end runner: load -> evaluate (leakage-free) -> fit -> save -> latency.

Quick start (no data needed -- runs on synthetic EEG and self-tests):

    pip install -r requirements.txt
    python run.py --synthetic --task go
    python run.py --synthetic --task word

On the real dataset (Nieto "Thinking out loud", OpenNeuro ds003626):

    # get the data once, e.g.:
    #   pip install openneuro-py && openneuro-py download --dataset ds003626 --target-dir ds003626
    python run.py --data ./ds003626 --task go   --condition overt_scaffold
    python run.py --data ./ds003626 --task word --condition inner

Outputs (in ./outputs): trained model (.joblib), metrics.json, and plots.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

from eeg_tvns.config import Config
from eeg_tvns.data_loader import load_dataset
from eeg_tvns.pipeline import build_pipeline
from eeg_tvns.evaluate import (
    evaluate_cross_subject,
    evaluate_within_subject,
    permutation_chance,
)
from eeg_tvns.realtime import benchmark_latency


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--data", default=None, help="path to ds003626 root (BIDS). Omit for synthetic.")
    src.add_argument("--synthetic", action="store_true", help="use synthetic EEG (default if no --data).")

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


def maybe_plots(cfg: Config, xsub, latency) -> list:
    if not cfg.make_plots:
        return []
    paths = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        if xsub.y_true is not None:
            labels = sorted(xsub.labels)
            cm = confusion_matrix(xsub.y_true, xsub.y_pred, labels=labels, normalize="true")
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
            names = [xsub.labels[l] for l in labels]
            ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.set_title("Cross-subject confusion (normalised)")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                            color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
            fig.colorbar(im, fraction=0.046)
            fig.tight_layout()
            cpath = os.path.join(cfg.out_dir, "confusion_matrix.png")
            fig.savefig(cpath, dpi=140); plt.close(fig); paths.append(cpath)

        lat = latency["_latencies"]
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.hist(lat, bins=30, color="#2E6E8E", alpha=0.85)
        ax.axvline(cfg.latency_budget_ms, color="#9E2B25", ls="--",
                   label=f"budget {cfg.latency_budget_ms:g} ms")
        ax.set_xlabel("decode latency (ms)"); ax.set_ylabel("count")
        ax.set_title("Closed-loop decision latency"); ax.legend()
        fig.tight_layout()
        lpath = os.path.join(cfg.out_dir, "latency_hist.png")
        fig.savefig(lpath, dpi=140); plt.close(fig); paths.append(lpath)
    except Exception as exc:  # plotting is optional
        logging.getLogger("eeg_tvns").warning("Plotting skipped: %s", exc)
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("eeg_tvns")
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1) data --------------------------------------------------------------
    ep = load_dataset(cfg)

    # 2) leakage-free evaluation ------------------------------------------
    log.info("=== Cross-subject (leave-one-subject-out) ===")
    xsub = evaluate_cross_subject(ep, cfg)
    print(xsub.report())

    log.info("=== Within-subject (per-patient calibration) ===")
    wsub = evaluate_within_subject(ep, cfg)
    print(wsub.report())

    perm = None
    if cfg.n_permutations > 0:
        log.info("=== Empirical chance (label permutation) ===")
        perm = permutation_chance(ep, cfg, n_perm=cfg.n_permutations)
        print(f"[permutation] observed={perm['observed']:.3f}  "
              f"null={perm['null_mean']:.3f}+/-{perm['null_std']:.3f}  "
              f"p={perm['p_value']:.4f}")

    # 3) fit the deployable model on ALL data -----------------------------
    log.info("=== Fitting final model on all data ===")
    model = build_pipeline(cfg)
    model.fit(ep.X, ep.y)

    # 4) closed-loop latency ----------------------------------------------
    log.info("=== Real-time latency benchmark ===")
    latency = benchmark_latency(model, ep, cfg)
    print(f"[latency] median={latency['median_ms']:.2f} ms  p95={latency['p95_ms']:.2f} ms  "
          f"within {cfg.latency_budget_ms:g} ms: {latency['fraction_within_budget']*100:.1f}%")

    # 5) persist -----------------------------------------------------------
    import joblib
    model_path = os.path.join(cfg.out_dir, f"model_{cfg.task}.joblib")
    joblib.dump({
        "model": model,
        "config": cfg,
        "label_names": ep.label_names,
        "ch_names": ep.ch_names,   # training channel order -> live channel remap
        "sfreq": cfg.sfreq,        # training rate -> live resample target
    }, model_path)

    metrics = {
        "task": cfg.task,
        "condition": cfg.condition,
        "classifier": cfg.classifier,
        "align": cfg.align,
        "n_channels": int(ep.X.shape[1]),
        "n_trials": int(ep.X.shape[0]),
        "n_subjects": int(len(np.unique(ep.subjects))),
        "cross_subject_bacc_mean": xsub.mean,
        "cross_subject_bacc_std": xsub.std,
        "within_subject_bacc_mean": wsub.mean,
        "within_subject_bacc_std": wsub.std,
        "permutation": {k: v for k, v in (perm or {}).items()},
        "latency": {k: v for k, v in latency.items() if not k.startswith("_")},
    }
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)

    plots = maybe_plots(cfg, xsub, latency)

    print("\n" + "=" * 64)
    print(f"Saved model  -> {model_path}")
    print(f"Saved metrics-> {os.path.join(cfg.out_dir, 'metrics.json')}")
    for pth in plots:
        print(f"Saved plot   -> {pth}")
    print("=" * 64)


if __name__ == "__main__":
    main()
