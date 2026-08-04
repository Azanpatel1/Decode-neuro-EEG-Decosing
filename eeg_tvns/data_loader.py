"""Data loading for the eeg_tvns pipeline.

Real recorded EEG only:

  * load_inner_speech(...)  -> reads the Nieto "Thinking out loud" derivatives
                               (OpenNeuro ds003626) from disk.
  * load_dataset(cfg)       -> entry point; requires cfg.data_root and raises if
                               it is missing. There is no surrogate fallback --
                               a decoder that gates stimulation must be fit on
                               real recordings.

Returns an `Epochs` object: X (trials, channels, samples), integer labels y,
per-trial subject ids (domains), and metadata.

The loader is deliberately defensive: BioSemi/derivative label columns vary
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
    """Map subject id -> list of (eeg_fif, events_dat, baseline_fif|None) sessions.

    `baseline_fif` is the session's `*_baseline-epo.fif` recording, used as the
    true rest class for the GO task. It is None when the release does not ship it.
    """
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
        baseline = base + "_baseline-epo.fif"
        if not os.path.exists(baseline):
            baseline = None
        # subject id from 'sub-XX'
        sid = None
        for part in f.split(os.sep):
            if part.startswith("sub-"):
                try:
                    sid = int(part.split("-")[1])
                except ValueError:
                    sid = part
                break
        out.setdefault(sid, []).append((f, ev, baseline))
    return out


def _slice_into_windows(epoch_data: np.ndarray, n_samp: int) -> List[np.ndarray]:
    """Cut a (channels, samples) recording into consecutive `n_samp` windows.

    Used to turn each long baseline recording into rest trials the same length as
    the action epochs. Any remainder shorter than one window is dropped rather
    than padded -- padding would invent samples.
    """
    n_win = epoch_data.shape[1] // n_samp
    return [epoch_data[:, w * n_samp:(w + 1) * n_samp] for w in range(n_win)]


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
    # Raw baseline recordings per session, kept unsliced until the action-window
    # length is known. Only collected for the GO task, which needs a rest class.
    baseline_raw: List[Tuple[object, np.ndarray]] = []
    sessions_missing_baseline: List[str] = []

    for sid in wanted_subjects:
        for fif, ev, baseline_fif in subj_files.get(sid, []):
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

            if cfg.task == "go":
                if baseline_fif is None:
                    sessions_missing_baseline.append(fif)
                else:
                    bep = mne.read_epochs(baseline_fif, preload=True, verbose="ERROR")
                    bep.pick("eeg")
                    if abs(bep.info["sfreq"] - cfg.sfreq) > 1e-3:
                        bep.resample(cfg.sfreq, verbose="ERROR")
                    baseline_raw.append((sid, bep.get_data(copy=True)))

    X = np.concatenate(Xs, axis=0)
    y_class = np.concatenate(ys, axis=0)
    subjects = np.concatenate(subs, axis=0)
    condition = np.concatenate(conds, axis=0)

    rest_X = rest_subjects = None
    if cfg.task == "go":
        rest_X, rest_subjects = _build_rest_from_baseline(
            baseline_raw, sessions_missing_baseline,
            n_samp=X.shape[2], n_channels=X.shape[1],
            attempt_subjects=subjects, cfg=cfg,
        )
        rest_X = _preprocess_array(rest_X, cfg.sfreq, cfg)

    X = _preprocess_array(X, cfg.sfreq, cfg)

    if cfg.use_low_density:
        full_ch_names = list(ch_names_ref or [])
        X, ch_names_ref = select_montage(X, full_ch_names, LOW_DENSITY_MONTAGE)
        if rest_X is not None:
            rest_X, _ = select_montage(rest_X, full_ch_names, LOW_DENSITY_MONTAGE)

    return _finalize_task(
        X, y_class, subjects, condition, cfg, cfg.sfreq, ch_names_ref,
        rest_X=rest_X, rest_subjects=rest_subjects,
    )


def _build_rest_from_baseline(
    baseline_raw: List[Tuple[object, np.ndarray]],
    sessions_missing_baseline: List[str],
    n_samp: int,
    n_channels: int,
    attempt_subjects: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the GO task's rest class from real `*_baseline-epo.fif` recordings.

    Each baseline recording is cut into windows the same length as the action
    epochs, then subsampled per subject to roughly match that subject's attempt
    count so the classifier is not fed a lopsided problem. Subsampling only ever
    *drops* real windows.

    Raises if no baseline data is available: the GO decoder gates stimulation, so
    a missing rest class must stop the run rather than be filled in.
    """
    if sessions_missing_baseline:
        log.warning(
            "%d session(s) have no '*_baseline-epo.fif'; their rest trials are "
            "absent (first: %s).",
            len(sessions_missing_baseline), sessions_missing_baseline[0],
        )
    if not baseline_raw:
        raise FileNotFoundError(
            "The GO task needs real rest epochs, but no '*_baseline-epo.fif' files "
            "were found in the dataset. Re-download ds003626 including its "
            "derivatives, or run the word task instead (--task word). Rest will "
            "not be synthesized."
        )

    rng = np.random.default_rng(cfg.random_state)
    windows: List[np.ndarray] = []
    subs: List[int] = []
    for sid, bdata in baseline_raw:
        sid_int = sid if isinstance(sid, int) else 0
        subj_windows: List[np.ndarray] = []
        for epoch_data in bdata:  # (channels, samples)
            subj_windows.extend(_slice_into_windows(epoch_data, n_samp))
        if not subj_windows:
            log.warning("sub-%s: baseline shorter than one %d-sample window; "
                        "no rest trials from it.", sid, n_samp)
            continue
        # Cap at this subject's attempt count to keep the classes comparable.
        n_attempts = int(np.sum(attempt_subjects == sid_int))
        if n_attempts and len(subj_windows) > n_attempts:
            pick = rng.choice(len(subj_windows), size=n_attempts, replace=False)
            subj_windows = [subj_windows[i] for i in sorted(pick)]
        windows.extend(subj_windows)
        subs.extend([sid_int] * len(subj_windows))

    if not windows:
        raise ValueError(
            f"No baseline window is as long as the {n_samp}-sample action window. "
            "Shorten the action window (Config.tmin/tmax) so real rest epochs fit."
        )

    rest_X = np.stack(windows).astype(np.float64)
    if rest_X.shape[1] != n_channels:
        raise ValueError(
            f"Baseline recordings have {rest_X.shape[1]} channels but action epochs "
            f"have {n_channels}. Refusing to guess a mapping."
        )
    log.info("GO rest class: %d real baseline windows from %d session(s).",
             len(rest_X), len(baseline_raw))
    return rest_X, np.asarray(subs, int)


# ---------------------------------------------------------------------------
# Task assembly
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
    # Rest must be real recorded baseline (see _build_rest_from_baseline). There is
    # no surrogate fallback: this decoder gates stimulation, and a rest class made
    # of scaled-down attempt epochs would make the GO score meaningless.
    if rest_X is None or rest_subjects is None:
        raise ValueError(
            "The GO task requires real rest epochs, but none were supplied. "
            "Rest is never synthesized -- see _build_rest_from_baseline."
        )

    Xg = np.concatenate([X, rest_X], axis=0)
    yg = np.concatenate([np.ones(len(X), int), np.zeros(len(rest_X), int)])
    sg = np.concatenate([subjects, rest_subjects])
    log.info("GO task: %d attempt vs %d rest trials.", len(X), len(rest_X))
    return Epochs(
        X=Xg, y=yg, subjects=sg, sfreq=sfreq, ch_names=ch_names,
        label_names={0: "rest", 1: "attempt"},
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def load_dataset(cfg: Config) -> Epochs:
    """Load real recorded EEG. There is no surrogate/synthetic fallback.

    A decoder that gates stimulation must be fit on real recordings, so a missing
    dataset is a hard error rather than something quietly filled in with
    generated data.
    """
    cfg.validate()
    if not cfg.data_root:
        raise ValueError(
            "No dataset given. This pipeline only runs on real recorded EEG.\n"
            "Pass --data /path/to/ds003626 (Nieto 'Thinking out loud').\n"
            "To fetch it:\n"
            "    pip install openneuro-py\n"
            "    openneuro-py download --dataset ds003626 --target-dir ds003626"
        )
    if not os.path.isdir(cfg.data_root):
        raise FileNotFoundError(
            f"Dataset directory not found: {cfg.data_root!r}. "
            "Point --data at the ds003626 root."
        )
    log.info("Loading real dataset (ds003626) from %s", cfg.data_root)
    ep = load_inner_speech(cfg)
    log.info(ep.summary())
    return ep
