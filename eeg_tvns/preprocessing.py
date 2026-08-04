"""Shared temporal preprocessing — used by BOTH the offline training path
(`data_loader`) and the live acquisition path (`acquisition`).

Having a single implementation is what guarantees train/inference parity: the
covariance features the model sees online are produced by the exact same
band-pass + notch design it was trained on. (Mismatched filter families — e.g.
FIR offline vs. IIR online — silently shift the covariance distribution and
degrade a model that benchmarks fine offline.)

Zero-phase SciPy IIR (Butterworth band-pass + optional notch, applied with
`sosfiltfilt`). IIR is the right choice here because it stays stable and
edge-effect-light on the short (~1-2 s) windows the closed loop operates on,
where a long FIR would not fit.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def design_filters(
    sfreq: float,
    l_freq: float,
    h_freq: float,
    notch: Optional[float] = None,
    order: int = 4,
):
    """Return (bandpass_sos, notch_sos|None) designed for `sfreq`."""
    from scipy.signal import butter, iirnotch, tf2sos

    ny = sfreq / 2.0
    hi = min(h_freq, ny * 0.99)
    bp_sos = butter(order, [l_freq / ny, hi / ny], btype="band", output="sos")
    notch_sos = None
    if notch and notch < ny:
        b, a = iirnotch(w0=notch / ny, Q=30.0)
        notch_sos = tf2sos(b, a)
    return bp_sos, notch_sos


def apply_filters(X: np.ndarray, bp_sos, notch_sos=None) -> np.ndarray:
    """Apply the designed filters along the last axis; returns a float64 copy.

    Padding is clamped so the call is safe on short real-time windows.
    """
    from scipy.signal import sosfiltfilt

    x = np.asarray(X, dtype=np.float64)
    n = x.shape[-1]

    def _pad(sos) -> int:
        default = 3 * (2 * sos.shape[0] + 1)
        return int(min(default, max(0, n - 1)))

    x = sosfiltfilt(bp_sos, x, axis=-1, padlen=_pad(bp_sos))
    if notch_sos is not None:
        x = sosfiltfilt(notch_sos, x, axis=-1, padlen=_pad(notch_sos))
    return x


def preprocess(X: np.ndarray, sfreq: float, cfg) -> np.ndarray:
    """Band-pass (+ notch) an array using the cfg's frequency settings."""
    bp_sos, notch_sos = design_filters(sfreq, cfg.l_freq, cfg.h_freq, cfg.notch)
    return apply_filters(X, bp_sos, notch_sos)


def resample_window(x: np.ndarray, sfreq_in: float, sfreq_out: float) -> np.ndarray:
    """Resample (n_channels, n_samples) from sfreq_in to sfreq_out along time.

    Used online to bring a board window onto the model's training rate so the
    covariance is computed on identically-sampled data.
    """
    if abs(sfreq_in - sfreq_out) < 1e-6:
        return np.asarray(x, dtype=np.float64)
    from fractions import Fraction
    from scipy.signal import resample_poly

    frac = Fraction(float(sfreq_out) / float(sfreq_in)).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    if up == 0:
        up = 1
    return resample_poly(np.asarray(x, dtype=np.float64), up, down, axis=-1)


def channel_order_from_names(board_names, train_names) -> Tuple[list, list]:
    """Map physical board channels onto the model's trained channel order.

    Returns (order, missing): `order` indexes `board_names` so that
    board[order] follows `train_names`; `missing` lists train channels absent
    from the board (a real montage-mismatch warning, not a silent identity map).
    """
    board_names = [str(c).strip() for c in board_names]
    lut = {c.lower(): i for i, c in enumerate(board_names)}
    order, missing = [], []
    for name in train_names:
        i = lut.get(str(name).strip().lower())
        if i is None:
            missing.append(name)
        else:
            order.append(i)
    return order, missing
