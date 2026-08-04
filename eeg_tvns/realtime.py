"""Closed-loop real-time decoding: a thin wrapper that turns a fitted pipeline
into a GO/no-GO gate for the tVNS stimulator, plus a latency benchmark that
proves the compute fits inside the VNS pairing window.

This is the online counterpart of the offline pipeline. It adds:
  * a sliding-window `decode()` returning (go, probability, latency_ms)
  * online Riemannian recentering (`update_reference`) to track session drift
  * `benchmark_latency` to measure the decode-to-decision compute time
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from pyriemann.estimation import Covariances
from pyriemann.utils.geodesic import geodesic
from pyriemann.utils.base import invsqrtm

from .config import Config

log = logging.getLogger("eeg_tvns.realtime")


@dataclass
class Decision:
    go: bool
    probability: float
    latency_ms: float
    predicted_label: int
    # Full posterior over the pipeline's classes, aligned with `classes`.
    # Populated for readouts that need more than the GO probability (e.g. the
    # 4-class word decoder). Never used to gate stimulation.
    probabilities: Optional[Tuple[float, ...]] = None
    classes: Optional[Tuple[int, ...]] = None


class RealTimeDecoder:
    """Wrap a fitted scikit-learn pipeline for online, single-window decoding.

    Parameters
    ----------
    pipeline : fitted Pipeline from build_pipeline(...)
    cfg      : Config (uses go_threshold, metric, latency_budget_ms)
    positive_label : which class means "fire tVNS" (default 1 = attempt / GO)
    """

    def __init__(self, pipeline, cfg: Config, positive_label: int = 1):
        self.pipe = pipeline
        self.cfg = cfg
        self.positive_label = positive_label
        self._classes = list(getattr(pipeline, "classes_", [0, 1]))
        self._align = pipeline.named_steps.get("align", None)
        self._cov = Covariances(estimator=cfg.cov_estimator)
        self.adapt_alpha = 0.05  # step size for online recentering

    # -- inference ----------------------------------------------------------
    def decode(self, window: np.ndarray) -> Decision:
        """window: (n_channels, n_samples). Returns a timed GO decision."""
        t0 = time.perf_counter()
        X = window[None].astype(np.float64)
        proba = self.pipe.predict_proba(X)[0]
        classes = list(self.pipe.classes_)
        p_go = float(proba[classes.index(self.positive_label)]) if self.positive_label in classes else float(proba.max())
        pred = int(self.pipe.classes_[int(np.argmax(proba))])
        latency_ms = (time.perf_counter() - t0) * 1e3
        return Decision(
            go=p_go >= self.cfg.go_threshold,
            probability=p_go,
            latency_ms=latency_ms,
            predicted_label=pred,
            probabilities=tuple(float(p) for p in proba),
            classes=tuple(int(c) for c in classes),
        )

    # -- online drift tracking ---------------------------------------------
    def update_reference(self, window: np.ndarray) -> None:
        """Nudge the alignment reference toward the incoming data (label-free).
        Keeps the decoder aligned as the session drifts."""
        if self._align is None:
            return
        cov = self._cov.fit_transform(window[None].astype(np.float64))[0]
        self._align.reference_ = geodesic(
            self._align.reference_, cov, self.adapt_alpha, metric=self.cfg.metric
        )
        self._align.iM_ = invsqrtm(self._align.reference_)


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------
def _windows_from_trial(trial: np.ndarray, cfg: Config) -> np.ndarray:
    """Yield sliding windows (n_win, ch, win_samp) across one trial's time axis."""
    win = int(round(cfg.window_s * cfg.sfreq))
    hop = int(round(cfg.hop_s * cfg.sfreq))
    n = trial.shape[1]
    if win >= n:
        return trial[None, :, :]
    starts = range(0, n - win + 1, max(1, hop))
    return np.stack([trial[:, s:s + win] for s in starts])


def benchmark_latency(
    pipeline, ep, cfg: Config, max_trials: int = 60
) -> Dict:
    """Measure decode-to-decision compute latency by replaying recorded epochs.

    Windows are slid over real trials from `ep` -- no generated signal is involved;
    this times the decode path, it does not fabricate input.

    Returns median / p95 / max latency (ms) and the fraction of decisions that
    land within cfg.latency_budget_ms -- i.e. inside the VNS pairing window.
    """
    rt = RealTimeDecoder(pipeline, cfg)
    rng = np.random.default_rng(cfg.random_state)
    idx = rng.permutation(len(ep.y))[:max_trials]
    lats = []
    for i in idx:
        for w in _windows_from_trial(ep.X[i], cfg):
            lats.append(rt.decode(w).latency_ms)
    lats = np.asarray(lats)
    within = float(np.mean(lats <= cfg.latency_budget_ms))
    out = {
        "n_decisions": int(lats.size),
        "median_ms": float(np.median(lats)),
        "p95_ms": float(np.percentile(lats, 95)),
        "max_ms": float(lats.max()),
        "budget_ms": cfg.latency_budget_ms,
        "fraction_within_budget": within,
        "_latencies": lats,
    }
    log.info(
        "Latency: median=%.2f ms  p95=%.2f ms  max=%.2f ms  within %g ms: %.1f%%",
        out["median_ms"], out["p95_ms"], out["max_ms"], cfg.latency_budget_ms, within * 100,
    )
    return out
