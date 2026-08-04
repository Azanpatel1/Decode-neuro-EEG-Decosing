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
def _nearest_by_position(
    ch_names: List[str], wanted: List[str], source_montage: str = "biosemi128",
    warn_dist_m: float = 0.025,
) -> Optional[Tuple[List[int], List[str]]]:
    """Map high-density electrode labels onto `wanted` 10-20 sites by 3D position.

    ds003626 is a 128-channel BioSemi cap labelled A1..D32, so the wearable 10-20
    names we want mostly do not appear -- and matching by *name* is actively
    dangerous: BioSemi 'C3'/'C4' exist but sit nowhere near 10-20 C3/C4 over
    sensorimotor cortex. Instead we look up both label sets in their standard
    montages and pick, for each wanted site, the closest unused source electrode.

    Returns (indices into ch_names, wanted-site names) or None if the positions
    are unavailable. The returned names are the 10-20 site labels, so the model
    bundle records where the channels actually are -- which is what lets the live
    board be remapped onto the trained montage (Invariant D).
    """
    try:
        import mne

        src_pos = mne.channels.make_standard_montage(source_montage).get_positions()["ch_pos"]
        tgt_pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    except Exception:
        log.warning("Could not load standard montages for position mapping.", exc_info=True)
        return None

    available = [(i, n) for i, n in enumerate(ch_names) if n in src_pos]
    if len(available) < len(wanted):
        return None

    idx, names, used, report = [], [], set(), []
    for site in wanted:
        if site not in tgt_pos:
            log.warning("10-20 site %s not in standard_1020; skipping.", site)
            continue
        target = tgt_pos[site]
        best, best_d = None, np.inf
        for i, n in available:
            if i in used:
                continue
            d = float(np.linalg.norm(src_pos[n] - target))
            if d < best_d:
                best, best_d = (i, n), d
        if best is None:
            continue
        used.add(best[0])
        idx.append(best[0])
        names.append(site)
        report.append(f"{site}<-{best[1]}({best_d * 1000:.0f}mm)")
        if best_d > warn_dist_m:
            log.warning("Nearest electrode to %s is %s, %.0f mm away -- coarse match.",
                        site, best[1], best_d * 1000)

    if len(idx) < max(4, len(wanted) // 2):
        return None
    order = np.argsort(idx)
    idx = [idx[i] for i in order]
    names = [names[i] for i in order]
    log.info("Mapped %d high-density electrodes onto 10-20 sites by position: %s",
             len(idx), ", ".join(report))
    return idx, names


def resolve_montage_indices(
    ch_names: List[str], wanted: List[str]
) -> Tuple[List[int], List[str]]:
    """Resolve `wanted` low-density sites to indices into `ch_names`.

    Tries exact names, then nearest-position mapping for high-density caps, then
    falls back to evenly-spaced channels. Returned separately from the data so a
    long recording can be reduced per session instead of after concatenation.
    """
    exact = [ch_names.index(c) for c in wanted if c in ch_names]
    if len(exact) == len(wanted):
        log.info("Using %d named low-density channels.", len(exact))
        return exact, [ch_names[i] for i in exact]

    mapped = _nearest_by_position(ch_names, wanted)
    if mapped is not None:
        return mapped

    n_keep = min(len(wanted), len(ch_names))
    idx = sorted(set(np.linspace(0, len(ch_names) - 1, n_keep).round().astype(int).tolist()))
    names = [ch_names[i] for i in idx]
    log.warning(
        "Could not resolve montage by name or position; falling back to %d "
        "evenly-spaced channels: %s. These labels do NOT carry 10-20 positions, "
        "so a live board cannot be reliably remapped onto them.", len(idx), names,
    )
    return idx, names


def select_montage(
    X: np.ndarray, ch_names: List[str], wanted: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Return X and names restricted to the resolved low-density montage."""
    idx, names = resolve_montage_indices(ch_names, wanted)
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


def _slice_into_windows(
    epoch_data: np.ndarray, n_samp: int, overlap: float = 0.0
) -> List[np.ndarray]:
    """Cut a (channels, samples) recording into `n_samp` windows.

    Used to turn each baseline recording into rest trials the same length as the
    action epochs. `overlap` (0..1) is the fraction shared between consecutive
    windows. Any remainder shorter than one window is dropped rather than padded --
    padding would invent samples.
    """
    step = max(1, int(round(n_samp * (1.0 - min(max(overlap, 0.0), 0.95)))))
    out = []
    start = 0
    while start + n_samp <= epoch_data.shape[1]:
        out.append(epoch_data[:, start:start + n_samp])
        start += step
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
    # Raw baseline recordings per session, kept unsliced until the action-window
    # length is known. Only collected for the GO task, which needs a rest class.
    baseline_raw: List[Tuple[object, np.ndarray]] = []
    sessions_missing_baseline: List[str] = []
    # Channel reduction is resolved once and applied per session: the full 128-ch
    # epochs are ~236 MB each, so reducing after concatenating all 30 sessions
    # would need tens of GB of RAM.
    sel_idx: Optional[List[int]] = None
    sel_names: Optional[List[str]] = None

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

            # Reduce channels here, per session: the full 128-ch epochs are
            # ~236 MB each, so reducing only after concatenating all 30 sessions
            # would need tens of GB. Filtering is per-channel, so reducing before
            # the band-pass gives identical results to reducing after.
            if cfg.use_low_density and sel_idx is None:
                sel_idx, sel_names = resolve_montage_indices(ch_names, LOW_DENSITY_MONTAGE)
            if sel_idx is not None:
                data = data[:, sel_idx, :]

            mask = np.isin(cnd, list(keep_conditions))
            Xs.append(data[mask])
            ys.append(cls[mask])
            conds.append(cnd[mask])
            subs.append(np.full(mask.sum(), sid if isinstance(sid, int) else 0))
            ch_names_ref = ch_names_ref or (sel_names if sel_idx is not None else ch_names)

            if cfg.task == "go":
                if baseline_fif is None:
                    sessions_missing_baseline.append(fif)
                else:
                    bep = mne.read_epochs(baseline_fif, preload=True, verbose="ERROR")
                    # Select by NAME, not by type: the baseline files type the 8
                    # external channels as EEG while the action epochs do not, so
                    # pick("eeg") yields 136 vs 128. Matching names (and ordering
                    # to them) keeps rest and attempt on the same montage.
                    missing = [c for c in ch_names if c not in bep.ch_names]
                    if missing:
                        raise ValueError(
                            f"{os.path.basename(baseline_fif)} is missing "
                            f"{len(missing)} channel(s) present in the action "
                            f"epochs (e.g. {missing[:5]}). Refusing to guess a mapping."
                        )
                    bep.pick(ch_names)
                    bep.reorder_channels(ch_names)
                    if abs(bep.info["sfreq"] - cfg.sfreq) > 1e-3:
                        bep.resample(cfg.sfreq, verbose="ERROR")
                    bdata = bep.get_data(copy=True)
                    if sel_idx is not None:
                        bdata = bdata[:, sel_idx, :]
                    baseline_raw.append((sid, bdata))

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

    return _finalize_task(
        X, y_class, subjects, condition, cfg, cfg.sfreq, ch_names_ref,
        rest_X=rest_X, rest_subjects=rest_subjects,
    )


def _balance_go_classes(
    X: np.ndarray,
    subjects: np.ndarray,
    rest_X: np.ndarray,
    rest_subjects: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trim the majority class per subject so attempt/rest counts are comparable.

    ds003626 ships ~15 s of baseline per session against hundreds of action
    epochs, so attempts outnumber rest heavily. Left alone, a classifier can
    score well by leaning on the prior; balancing per subject keeps each domain's
    recentering and the LOSO folds interpretable. This only ever *drops* real
    trials -- nothing is duplicated, weighted, or synthesized.
    """
    rng = np.random.default_rng(cfg.random_state)
    keep_att, keep_rest = [], []
    for sid in np.unique(np.concatenate([subjects, rest_subjects])):
        a_idx = np.flatnonzero(subjects == sid)
        r_idx = np.flatnonzero(rest_subjects == sid)
        n = min(len(a_idx), len(r_idx))
        if n == 0:
            log.warning("sub-%s: has %d attempt and %d rest trials; dropping it "
                        "from the GO problem.", sid, len(a_idx), len(r_idx))
            continue
        if len(a_idx) > n:
            a_idx = np.sort(rng.choice(a_idx, size=n, replace=False))
        if len(r_idx) > n:
            r_idx = np.sort(rng.choice(r_idx, size=n, replace=False))
        keep_att.append(a_idx)
        keep_rest.append(r_idx)

    if not keep_att:
        raise ValueError(
            "No subject has both attempt and rest trials; cannot build the GO "
            "problem. Check that '*_baseline-epo.fif' files loaded correctly."
        )
    ka = np.concatenate(keep_att)
    kr = np.concatenate(keep_rest)
    log.info("Balanced GO classes: kept %d/%d attempt and %d/%d rest trials.",
             len(ka), len(X), len(kr), len(rest_X))
    return X[ka], subjects[ka], rest_X[kr], rest_subjects[kr]


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
    # Sessions of the same subject are pooled, so accumulate per subject first.
    per_subject: dict = {}
    for sid, bdata in baseline_raw:
        sid_int = sid if isinstance(sid, int) else 0
        for epoch_data in bdata:  # (channels, samples)
            per_subject.setdefault(sid_int, []).extend(
                _slice_into_windows(epoch_data, n_samp, cfg.baseline_overlap)
            )

    for sid_int, subj_windows in sorted(per_subject.items()):
        if not subj_windows:
            log.warning("sub-%s: baseline shorter than one %d-sample window; "
                        "no rest trials from it.", sid_int, n_samp)
            continue
        n_attempts = int(np.sum(attempt_subjects == sid_int))
        if cfg.balance_go_classes and n_attempts and len(subj_windows) > n_attempts:
            pick = rng.choice(len(subj_windows), size=n_attempts, replace=False)
            subj_windows = [subj_windows[i] for i in sorted(pick)]
        log.info("sub-%s: %d attempt vs %d rest windows.",
                 sid_int, n_attempts, len(subj_windows))
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
    log.warning(
        "GO rest comes from the separate '*_baseline-epo.fif' block, which differs "
        "from the task blocks in more than speech attempt (~1.6x broadband "
        "amplitude, eyes-closed/arousal and drift differences). Measured on "
        "ds003626 at a matched 0.5 s window: baseline-block rest scores 0.80 LOSO "
        "balanced accuracy, but same-block rest (pre-cue interval of the same "
        "trials) scores only 0.54. Treat this task's score as an UPPER BOUND that "
        "mostly reflects block identity, and do NOT use a model trained this way "
        "to gate stimulation -- online, rest is same-block rest. For a deployable "
        "gate, record calibration data with rest interleaved in the same session."
    )
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

    if cfg.balance_go_classes:
        X, subjects, rest_X, rest_subjects = _balance_go_classes(
            X, subjects, rest_X, rest_subjects, cfg
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
def load_calibration(cfg: Config) -> Epochs:
    """Load same-session calibration recordings written by `calibrate.py`.

    This is the trustworthy source for the GO task: attempt and rest trials are
    randomly interleaved inside one recording, so they share block, impedance and
    arousal context. Contrast with ds003626, whose only rest is a separate
    baseline block that confounds the GO score.

    Recordings are stored raw at board rate; filtering happens here through the
    shared `preprocessing` module, so calibration data and the live path get the
    same filter design (Invariant A). `cfg.sfreq` is adopted from the recordings
    so the model trains at the board's native rate and the online path needs no
    resampling.
    """
    paths = sorted(glob.glob(cfg.calibration_glob or ""))
    if not paths:
        raise FileNotFoundError(
            f"No calibration recordings matched {cfg.calibration_glob!r}. "
            "Record some first:\n"
            "    python calibrate.py --port <serial> --subject 1 --channel-names ..."
        )

    Xs, y_go, y_word, subs = [], [], [], []
    sfreq = ch_names = action_s = None
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            names = [str(c) for c in z["ch_names"]]
            fs = float(z["sfreq"])
            if sfreq is None:
                sfreq, ch_names, action_s = fs, names, float(z["action_s"])
            if fs != sfreq:
                raise ValueError(
                    f"{os.path.basename(p)} was recorded at {fs} Hz but earlier "
                    f"files at {sfreq} Hz. Refusing to mix sampling rates."
                )
            if names != ch_names:
                raise ValueError(
                    f"{os.path.basename(p)} has a different montage "
                    f"({names}) than {ch_names}. Refusing to guess a mapping."
                )
            X = np.asarray(z["X"], dtype=np.float64)
            if X.shape[2] != int(round(action_s * sfreq)):
                log.warning("%s window is %d samples, expected %d.",
                            os.path.basename(p), X.shape[2],
                            int(round(action_s * sfreq)))
            Xs.append(X)
            y_go.append(np.asarray(z["y_go"], int))
            y_word.append(np.asarray(z["y_word"], int))
            subs.append(np.full(len(X), int(z["subject"])))

    X = np.concatenate(Xs)
    y_go = np.concatenate(y_go)
    y_word = np.concatenate(y_word)
    subjects = np.concatenate(subs)

    # Train at the board's native rate: the recordings define the rate, so the
    # online path can run with --no-resample and match exactly.
    if abs(cfg.sfreq - sfreq) > 1e-6:
        log.info("Adopting calibration sampling rate %.0f Hz (was %.0f).", sfreq, cfg.sfreq)
        cfg.sfreq = sfreq

    X = _preprocess_array(X, sfreq, cfg)
    log.info("Calibration: %d recording(s), %d subject(s), %d trials, %d ch @ %.0f Hz.",
             len(paths), len(np.unique(subjects)), len(X), X.shape[1], sfreq)

    if cfg.task == "word":
        keep = y_word >= 0
        if not keep.any():
            raise ValueError(
                "No attempt trials with word labels in the calibration data; "
                "the word task needs them."
            )
        return Epochs(
            X=X[keep], y=y_word[keep], subjects=subjects[keep], sfreq=sfreq,
            ch_names=ch_names, label_names=CLASS_NAMES,
        )

    rest = y_go == 0
    log.info("GO task from same-session calibration: %d attempt vs %d rest trials.",
             int((~rest).sum()), int(rest.sum()))
    return _finalize_task(
        X[~rest], np.zeros((~rest).sum(), int), subjects[~rest], None, cfg, sfreq,
        ch_names, rest_X=X[rest], rest_subjects=subjects[rest],
    )


def load_dataset(cfg: Config) -> Epochs:
    """Load real recorded EEG. There is no surrogate/synthetic fallback.

    A decoder that gates stimulation must be fit on real recordings, so a missing
    dataset is a hard error rather than something quietly filled in with
    generated data.
    """
    cfg.validate()
    if cfg.calibration_glob:
        ep = load_calibration(cfg)
        log.info(ep.summary())
        return ep
    if not cfg.data_root:
        raise ValueError(
            "No data given. This pipeline only runs on real recorded EEG.\n"
            "Either pass your own same-session recordings (preferred for the GO "
            "task, which is confounded on ds003626):\n"
            "    python calibrate.py --port <serial> --subject 1 --channel-names ...\n"
            "    python run.py --calibration 'calib/*.npz' --task go\n"
            "or the public dataset:\n"
            "    pip install openneuro-py\n"
            "    openneuro-py download --dataset ds003626 --target-dir ds003626\n"
            "    python run.py --data ./ds003626 --task word"
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
