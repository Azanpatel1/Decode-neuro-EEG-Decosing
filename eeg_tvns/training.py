"""The one training path: load -> evaluate -> fit -> save -> benchmark.

This used to live in `run.py`'s `main()`. It moved here so the dashboard's Train
tab and the CLI run *the same code*: two implementations would eventually disagree
about preprocessing, folds, or what goes into the bundle, and the model that gates
stimulation would then differ from the one whose score you read. `run.py` is now a
thin argument parser over `train()`.

`progress` and `should_stop` are optional, so nothing about the CLI's behaviour
changes; the dashboard passes them to drive its progress bar and Cancel button.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import Config
from .data_loader import load_dataset
from .evaluate import (
    Cancelled,
    evaluate_cross_subject,
    evaluate_within_subject,
    permutation_chance,
)
from .models import gating_verdict
from .pipeline import build_pipeline
from .realtime import benchmark_latency

log = logging.getLogger("eeg_tvns.training")

# (fraction 0-1, stage label) -> None
Progress = Callable[[float, str], None]
ShouldStop = Callable[[], bool]

# Rough share of wall-clock per stage, used only to make one honest overall bar
# instead of five that each restart at zero. The permutation test dominates.
_STAGES = {
    "load": (0.00, 0.10),
    "loso": (0.10, 0.30),
    "within": (0.30, 0.42),
    "permutation": (0.42, 0.88),
    "fit": (0.88, 0.96),
    "save": (0.96, 1.00),
}


@dataclass
class TrainResult:
    model_path: str = ""
    metrics_path: str = ""
    plots: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    reports: List[str] = field(default_factory=list)
    verdict: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "model_path": self.model_path, "metrics_path": self.metrics_path,
            # Basenames: the UI fetches these under /outputs/.
            "plots": [os.path.basename(p) for p in self.plots],
            "metrics": self.metrics, "reports": self.reports,
            "verdict": self.verdict,
        }


class _Reporter:
    """Maps per-stage progress into one monotonic 0-1 fraction."""

    def __init__(self, progress: Optional[Progress]):
        self._p = progress
        self.stage = "load"

    def enter(self, stage: str, label: str = "") -> None:
        self.stage = stage
        self.emit(0.0, label or stage)

    def emit(self, frac: float, label: str) -> None:
        if self._p is None:
            return
        lo, hi = _STAGES.get(self.stage, (0.0, 1.0))
        try:
            self._p(lo + (hi - lo) * max(0.0, min(1.0, frac)), label)
        except Exception:
            log.exception("progress callback failed; continuing")

    def counter(self):
        """An evaluate.Progress callback bound to the current stage."""
        def cb(done: int, total: int, label: str) -> None:
            self.emit(done / max(1, total), label)
        return cb


def source_of(cfg: Config) -> Optional[str]:
    """Where the training data came from -- recorded in the bundle for provenance.

    `models.gating_verdict` keys off this: same-session calibration data can gate
    stimulation, ds003626's baseline-block rest cannot.
    """
    if cfg.calibration_glob:
        return "calibration"
    if cfg.data_root:
        return "ds003626"
    return None


def make_plots(cfg: Config, xsub, latency) -> List[str]:
    if not cfg.make_plots:
        return []
    paths: List[str] = []
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
            cpath = os.path.join(cfg.out_dir, f"confusion_matrix_{cfg.task}.png")
            fig.savefig(cpath, dpi=140); plt.close(fig); paths.append(cpath)

        lat = latency["_latencies"]
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.hist(lat, bins=30, color="#2E6E8E", alpha=0.85)
        ax.axvline(cfg.latency_budget_ms, color="#9E2B25", ls="--",
                   label=f"budget {cfg.latency_budget_ms:g} ms")
        ax.set_xlabel("decode latency (ms)"); ax.set_ylabel("count")
        ax.set_title("Closed-loop decision latency"); ax.legend()
        fig.tight_layout()
        lpath = os.path.join(cfg.out_dir, f"latency_hist_{cfg.task}.png")
        fig.savefig(lpath, dpi=140); plt.close(fig); paths.append(lpath)
    except Exception as exc:  # plotting is optional
        log.warning("Plotting skipped: %s", exc)
    return paths


def train(cfg: Config, progress: Optional[Progress] = None,
          should_stop: Optional[ShouldStop] = None) -> TrainResult:
    """Run the full pipeline and write the model, metrics, and plots."""
    cfg.validate()
    os.makedirs(cfg.out_dir, exist_ok=True)
    rep = _Reporter(progress)
    res = TrainResult()

    def check() -> None:
        if should_stop is not None and should_stop():
            raise Cancelled("cancelled")

    # 1) data ---------------------------------------------------------------
    rep.enter("load", "loading recordings")
    ep = load_dataset(cfg)
    check()

    # 2) leakage-free evaluation -------------------------------------------
    log.info("=== Cross-subject (leave-one-subject-out) ===")
    rep.enter("loso")
    xsub = evaluate_cross_subject(ep, cfg, progress=rep.counter())
    res.reports.append(xsub.report())
    check()

    log.info("=== Within-subject (per-patient calibration) ===")
    rep.enter("within")
    wsub = evaluate_within_subject(ep, cfg, progress=rep.counter())
    res.reports.append(wsub.report())
    check()

    perm = None
    if cfg.n_permutations > 0:
        log.info("=== Empirical chance (label permutation) ===")
        rep.enter("permutation")
        perm = permutation_chance(ep, cfg, n_perm=cfg.n_permutations,
                                  progress=rep.counter(), should_stop=should_stop)
    else:
        log.warning("Permutations disabled: this run will report an accuracy with "
                    "no empirical chance level beside it.")

    # 3) fit the deployable model on ALL data ------------------------------
    log.info("=== Fitting final model on all data ===")
    rep.enter("fit", "fitting final model")
    model = build_pipeline(cfg)
    model.fit(ep.X, ep.y)
    check()

    # 4) closed-loop latency ----------------------------------------------
    log.info("=== Real-time latency benchmark ===")
    rep.emit(0.6, "benchmarking decode latency")
    latency = benchmark_latency(model, ep, cfg)

    # 5) persist -----------------------------------------------------------
    rep.enter("save", "writing model and metrics")
    import joblib

    source = source_of(cfg)
    res.model_path = os.path.join(cfg.out_dir, f"model_{cfg.task}.joblib")
    joblib.dump({
        "model": model,
        "config": cfg,
        "label_names": ep.label_names,
        "ch_names": ep.ch_names,   # training channel order -> live channel remap
        "sfreq": cfg.sfreq,        # training rate -> live resample target
        # Provenance, so a bundle can be judged for gating without its metrics
        # file beside it (see models.gating_verdict).
        "source": source,
        "loso": xsub.mean,
        "trained_at": time.time(),
    }, res.model_path)

    res.metrics = {
        "task": cfg.task,
        "condition": cfg.condition,
        "classifier": cfg.classifier,
        "align": cfg.align,
        "source": source,
        "n_channels": int(ep.X.shape[1]),
        "n_trials": int(ep.X.shape[0]),
        "n_subjects": int(len(np.unique(ep.subjects))),
        "n_classes": int(len(ep.label_names)),
        "cross_subject_bacc_mean": xsub.mean,
        "cross_subject_bacc_std": xsub.std,
        "within_subject_bacc_mean": wsub.mean,
        "within_subject_bacc_std": wsub.std,
        "permutation": {k: v for k, v in (perm or {}).items()},
        "latency": {k: v for k, v in latency.items() if not k.startswith("_")},
    }
    # Per-task filename: the go and word runs would otherwise clobber each other's
    # results, silently leaving metrics that describe a different decoder.
    res.metrics_path = os.path.join(cfg.out_dir, f"metrics_{cfg.task}.json")
    with open(res.metrics_path, "w") as fh:
        json.dump(res.metrics, fh, indent=2, default=float)

    res.plots = make_plots(cfg, xsub, latency)
    res.verdict = gating_verdict(cfg.task, source, xsub.mean).to_dict()
    rep.emit(1.0, "done")
    return res
