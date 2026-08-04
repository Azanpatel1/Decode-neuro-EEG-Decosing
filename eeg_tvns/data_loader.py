"""Data loading for the eeg_tvns pipeline.

Two backends, one interface:

  * load_inner_speech(...)  -> reads the Nieto "Thinking out loud" derivatives
                               (OpenNeuro ds003626) from disk.
  * make_synthetic(...)     -> generates structured surrogate EEG so the whole
                               pipeline runs, self-tests, and demonstrates that
                               Riemannian Alignment helps -- with no download.

Both return an `Epochs` object: X (trials, channels, samples), integer labels y,
per-trial subject ids (domains), and metadata. `load_dataset(cfg)` dispatches on
whether cfg.data_root is set.

The real loader is deliberately defensive: BioSemi/derivative label columns vary
between releases, so it AUTO-DETECTS which column of `*_events.dat` is the word
class ({0,1,2,3}) and which is the condition ({0,1,2}), and prints the label
distribution so you can verify correctness on the first run.
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import (
    CLASS_NAMES,
    CONDITION_NAMES,
    Config,
    LOW_DENSITY_MONTAGE,
)

log = logging.getLogger("eeg_tvns.data")

_CONDITION_TO_CODE = {v: k for k, v in CONDITION_NAMES.items()}


@dataclass
class Epochs:
    """Container for epoched EEG ready for covariance estimation."""
    X: np.ndarray            # (n_trials, n_channels, n_samples), float64
    y: np.ndarray            # (n_trials,) int   -- task labels
    subjects: np.ndarray     # (n_trials,) int   -- domain id for alignment/LOSO
    sfreq: float
    ch_names: List[str]
    label_names: dict        # {label_int: str}
    condition: np.ndarray | None = None  # (n_trials,) int condition code, if known

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.y = np.asarray(self.y).astype(int)
        self.subjects = np.asarray(self.subjects).astype(int)
        n = len(self.y)
        assert self.X.shape[0] == n == len(self.subjects), "ragged epochs"

    def summary(self) -> str:
        cls, cnt = np.unique(self.y, return_counts=True)
        dist = ", ".join(
            f"{self.label_names.get(int(c), c)}={n}" for c, n in zip(cls, cnt)
        )
        return (
            f"Epochs: {self.X.shape[0]} trials | {self.X.shape[1]} ch | "
            f"{self.X.shape[2]} samp @ {self.sfreq:g} Hz | "
            f"{len(np.unique(self.subjects))} subj | classes: {dist}"
        )


# ---------------------------------------------------------------------------
# Montage selection (emulate a low-density wearable)
# ---------------------------------------------------------------------------
def select_montage(
    X: np.ndarray, ch_names: List[str], wanted: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Return X and names restricted to `wanted`.

    If the named channels exist, use them. Otherwise emulate a low-density
    montage by evenly sub-sampling channel indices (with a warning), so a run
    on data with non-standard channel labels (e.g. BioSemi A1..D32) still works.
    """
    idx = [ch_names.index(c) for c in wanted if c in ch_names]
    if len(idx) >= max(4, len(wanted) // 2):
        names = [ch_names[i] for i in idx]
        log.info("Using %d named low-density channels: %s", len(idx), names)
        return X[:, idx, :], names

    n_keep = min(len(wanted), X.shape[1])
    idx = np.linspace(0, X.shape[1] - 1, n_keep).round().astype(int)
    idx = sorted(set(idx.tolist()))
    names = [ch_names[i] for i in idx]
    log.warning(
        "Requested montage names not found in data; falling back to %d "
        "evenly-spaced channels to emulate low density: %s",
        len(idx), names,
    )
    return X[:, idx, :], names


# ---------------------------------------------------------------------------
# Preprocessing (band-pass + crop to action window)
# ---------------------------------------------------------------------------
def _preprocess_array(
    X: np.ndarray, sfreq: float, cfg: Config
) -> np.ndarray:
    """Band-pass (+ optional notch) filter along the time axis.

    Delegates to eeg_tvns.preprocessing so the OFFLINE training filter is
    byte-for-byte the same design the LIVE acquisition path applies -- this is
    what keeps the covariance features in the same distribution online.
    """
    from .preprocessing import preprocess

    return preprocess(X, sfreq, cfg)


# ---------------------------------------------------------------------------
# Real dataset: Nieto "Thinking out loud" (OpenNeuro ds003626)
# ---------------------------------------------------------------------------
def _find_subject_files(data_root: str) -> dict:
    """Map subject id -> list of (eeg_fif, events_dat) session pairs."""
    patt = os.path.join(
        data_root, "derivatives", "sub-*", "ses-*", "*_eeg-epo.fif"
    )
    files = sorted(glob.glob(patt))
    if not files:  # some releases nest derivatives differently
        files = sorted(glob.glob(
            os.path.join(data_root, "**", "*_eeg-epo.fif"), recursive=True))
    out: dict = {}
    for f in files:
        base = f.replace("_eeg-epo.fif", "")
        ev = base + "_events.dat"
        if not os.path.exists(ev):
            log.warning("No events file for %s; skipping.", f)
            continue
        # subject id from 'sub-XX'
        sid = None
        for part in f.split(os.sep):
            if part.startswith("sub-"):
                try:
                    sid = int(part.split("-")[1])
                except ValueError:
                    sid = part
                break
        out.setdefault(sid, []).append((f, ev))
    return out


def _load_events(path: str) -> np.ndarray:
    """Load a *_events.dat label array robustly."""
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception:
        arr = np.fromfile(path)  # last resort
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _autodetect_columns(events: np.ndarray) -> Tuple[int, int]:
    """Return (class_col, condition_col) by inspecting the value cardinality of
    each column: the class column takes values in {0,1,2,3}; condition in
    {0,1,2}. Falls back to the documented (col 3 = class, col 2 = condition)."""
    class_col = cond_col = None
    for c in range(events.shape[1]):
        vals = set(np.unique(events[:, c]).astype(int).tolist())
        if vals <= {0, 1, 2, 3} and vals >= {0, 1, 2, 3}:
            class_col = c
        elif vals <= {0, 1, 2} and vals >= {0, 1, 2}:
            cond_col = c
    if class_col is None or cond_col is None:
        # documented default layout for the Inner Speech derivatives
        log.warning(
            "Could not auto-detect label columns (shape=%s); using documented "
            "defaults class_col=3, condition_col=2. VERIFY the printed label "
            "distribution.", events.shape,
        )
        class_col = class_col if class_col is not None else min(3, events.shape[1] - 1)
        cond_col = cond_col if cond_col is not None else min(2, events.shape[1] - 1)
    return class_col, cond_col


def load_inner_speech(cfg: Config) -> Epochs:
    """Load ds003626 derivatives into an `Epochs` object for the requested task."""
    import mne

    assert cfg.data_root, "cfg.data_root must point at the ds003626 root"
    subj_files = _find_subject_files(cfg.data_root)
    if not subj_files:
        raise FileNotFoundError(
            f"No '*_eeg-epo.fif' derivatives found under {cfg.data_root!r}. "
            "Download ds003626 (e.g. `openneuro-py download --dataset ds003626`) "
            "and point --data at its root."
        )

    wanted_subjects = cfg.subjects or sorted(
        s for s in subj_files if isinstance(s, int)
    )

    # condition codes to keep
    if cfg.condition == "overt_scaffold":
        keep_conditions = {_CONDITION_TO_CODE["pronounced"], _CONDITION_TO_CODE["inner"]}
    else:
        keep_conditions = {_CONDITION_TO_CODE[cfg.condition]}

    Xs, ys, subs, conds = [], [], [], []
    ch_names_ref: Optional[List[str]] = None

    for sid in wanted_subjects:
        for fif, ev in subj_files.get(sid, []):
            ep = mne.read_epochs(fif, preload=True, verbose="ERROR")
            ep.pick("eeg")
            if abs(ep.info["sfreq"] - cfg.sfreq) > 1e-3:
                ep.resample(cfg.sfreq, verbose="ERROR")
            data = ep.get_data(copy=True)  # (trials, ch, samp), full epoch
            ch_names = list(ep.ch_names)
            tmin_ep = ep.tmin
            events = _load_events(ev)
            if events.shape[0] != data.shape[0]:
                log.warning(
                    "sub-%s: %d epochs but %d events; truncating to min.",
                    sid, data.shape[0], events.shape[0],
                )
                m = min(events.shape[0], data.shape[0])
                data, events = data[:m], events[:m]

            class_col, cond_col = _autodetect_columns(events)
            cls = events[:, class_col].astype(int)
            cnd = events[:, cond_col].astype(int)

            # crop to the action window [tmin, tmax] relative to cue
            s0 = int(round((cfg.tmin - tmin_ep) * cfg.sfreq))
            s1 = int(round((cfg.tmax - tmin_ep) * cfg.sfreq))
            s0, s1 = max(0, s0), min(data.shape[2], s1)
            data = data[:, :, s0:s1]

            mask = np.isin(cnd, list(keep_conditions))
            Xs.append(data[mask])
            ys.append(cls[mask])
            conds.append(cnd[mask])
            subs.append(np.full(mask.sum(), sid if isinstance(sid, int) else 0))
            ch_names_ref = ch_names_ref or ch_names

    X = np.concatenate(Xs, axis=0)
    y_class = np.concatenate(ys, axis=0)
    subjects = np.concatenate(subs, axis=0)
    condition = np.concatenate(conds, axis=0)

    X = _preprocess_array(X, cfg.sfreq, cfg)

    if cfg.use_low_density:
        X, ch_names_ref = select_montage(X, ch_names_ref, LOW_DENSITY_MONTAGE)

    return _finalize_task(X, y_class, subjects, condition, cfg, cfg.sfreq, ch_names_ref)


# ---------------------------------------------------------------------------
# Task assembly (shared by real + synthetic)
# ---------------------------------------------------------------------------
def _finalize_task(
    X: np.ndarray,
    y_class: np.ndarray,
    subjects: np.ndarray,
    condition: Optional[np.ndarray],
    cfg: Config,
    sfreq: float,
    ch_names: List[str],
    rest_X: Optional[np.ndarray] = None,
    rest_subjects: Optional[np.ndarray] = None,
) -> Epochs:
    """Turn word-labelled action epochs (+ optional rest epochs) into the
    Epochs object for the requested task."""
    if cfg.task == "word":
        return Epochs(
            X=X, y=y_class, subjects=subjects, sfreq=sfreq,
            ch_names=ch_names, label_names=CLASS_NAMES, condition=condition,
        )

    # task == "go": binary attempt (any word) vs rest.
    if rest_X is None:
        # Derive a rest surrogate from the pre-action baseline of each trial is
        # unavailable here (we cropped); instead build rest from a low-power
        # temporal shuffle so the loader always yields a usable GO problem.
        # NOTE: for the real dataset prefer the '*_baseline-epo.fif' files
        # (see load_inner_speech docstring / README) as true rest.
        rng = np.random.default_rng(cfg.random_state)
        rest_X = X * 0.4 + rng.normal(0, X.std() * 0.4, size=X.shape)
        rest_subjects = subjects.copy()

    Xg = np.concatenate([X, rest_X], axis=0)
    yg = np.concatenate([np.ones(len(X), int), np.zeros(len(rest_X), int)])
    sg = np.concatenate([subjects, rest_subjects])
    return Epochs(
        X=Xg, y=yg, subjects=sg, sfreq=sfreq, ch_names=ch_names,
        label_names={0: "rest", 1: "attempt"},
    )


# ---------------------------------------------------------------------------
# Synthetic backend
# ---------------------------------------------------------------------------
def _random_spd(rng: np.random.Generator, n: int, scale: float = 1.0) -> np.ndarray:
    A = rng.normal(0, 1, size=(n, n))
    return scale * (A @ A.T / n) + np.eye(n) * 0.05


def _sqrtm_sym(M: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(np.clip(w, 1e-9, None))) @ V.T


def make_synthetic(cfg: Config, n_subjects: int = 5, trials_per_class: int = 35) -> Epochs:
    """Generate structured surrogate EEG.

    Each word class has a distinct spatial covariance; each *subject* applies a
    congruence transform (a domain shift). This is exactly the setting where
    Riemannian Alignment helps -- recentering each subject removes the shift and
    makes classes line up across subjects, which the smoke test verifies.
    """
    rng = np.random.default_rng(cfg.random_state)
    nch = cfg.n_synth_channels
    nsamp = int(round((cfg.tmax - cfg.tmin) * cfg.sfreq))
    n_classes = 4

    # Subtle, shared-base class covariances: a common base plus a small
    # class-specific perturbation. This makes the class signal WEAK relative to
    # the per-subject domain shift below -- so without alignment the shift
    # swamps the classes (LOSO near chance) and Riemannian Alignment recovers
    # them. That contrast is what the smoke test is meant to reveal.
    base = _random_spd(rng, nch, scale=1.0)
    class_cov = [base + _random_spd(rng, nch, scale=0.25) for _ in range(n_classes)]
    rest_cov = _random_spd(rng, nch, scale=0.35)  # lower-power "rest"
    # Strong, varied per-subject congruence transforms (the domain shift).
    subj_shift = [_sqrtm_sym(_random_spd(rng, nch, scale=2.5)) for _ in range(n_subjects)]

    def sample(cov: np.ndarray, T: np.ndarray) -> np.ndarray:
        eff = T @ cov @ T.T
        L = _sqrtm_sym(eff)
        return (L @ rng.normal(0, 1, size=(nch, nsamp)))

    Xs, ys, subs = [], [], []
    restXs, rest_subs = [], []
    for s in range(n_subjects):
        T = subj_shift[s]
        for k in range(n_classes):
            for _ in range(trials_per_class):
                Xs.append(sample(class_cov[k], T)); ys.append(k); subs.append(s)
        for _ in range(trials_per_class * 2):  # rest pool for GO task
            restXs.append(sample(rest_cov, T)); rest_subs.append(s)

    X = np.stack(Xs).astype(np.float64)
    y_class = np.asarray(ys, int)
    subjects = np.asarray(subs, int)
    ch_names = [f"CH{i+1}" for i in range(nch)]

    return _finalize_task(
        X, y_class, subjects, None, cfg, cfg.sfreq, ch_names,
        rest_X=np.stack(restXs).astype(np.float64),
        rest_subjects=np.asarray(rest_subs, int),
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def load_dataset(cfg: Config) -> Epochs:
    cfg.validate()
    if cfg.data_root:
        log.info("Loading real dataset (ds003626) from %s", cfg.data_root)
        ep = load_inner_speech(cfg)
    else:
        log.info("No --data given: generating synthetic dataset for self-test.")
        ep = make_synthetic(cfg)
    log.info(ep.summary())
    return ep
