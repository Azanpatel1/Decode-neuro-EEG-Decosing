"""Same-session calibration recorder for the GO decoder.

Why this exists
---------------
ds003626's only rest data is a separate 15 s baseline block, and that block
differs from the task blocks in far more than speech attempt (~1.6x broadband
amplitude, plus eyes-closed/arousal and drift differences). Measured on that
dataset at a matched 0.5 s window, holding the action epochs fixed and changing
only where rest comes from:

    rest = separate baseline block     -> 0.798 LOSO balanced accuracy
    rest = pre-cue interval, same trial -> 0.540

So a GO decoder trained against a separate baseline block mostly learns *which
block a window came from*, and a permutation test does not catch it. Online,
rest is same-block rest -- the 0.54 case.

This module records the data that fixes that: cued **overt** speech attempts and
cued rest, randomly interleaved inside a single recording, on the operator's own
electrodes. Both classes then share block, impedance, arousal and drift context,
so a score on this data reflects speech attempt rather than recording context.

Design notes
------------
* Trials are **randomly interleaved**, never blocked. Blocking would reintroduce
  exactly the confound this recorder exists to remove.
* Raw board-rate samples are stored **unfiltered**. Filtering happens at load
  time through `eeg_tvns.preprocessing`, so calibration data goes through the
  same filter design as the live path (Invariant A).
* Channel names are **required**, so the recording carries an explicit montage
  and can never be silently identity-mapped (Invariant D).
* Recording at the board's native rate also removes the online resample step
  (train at 125 Hz, run at 125 Hz).

Run:
    python calibrate.py --port /dev/cu.usbserial-XXXX --subject 1 \
        --channel-names "F7,F8,FC5,FC6,FT7,FT8,T7,T8,C3,C4,CP5,CP6,P7,P8,Cz,Pz"
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import CLASS_NAMES, Config

log = logging.getLogger("eeg_tvns.calibrate")

REST = 0
ATTEMPT = 1


@dataclass
class Trial:
    """One cued trial. `word` is None for rest trials."""

    kind: int                      # REST or ATTEMPT
    word: Optional[int] = None     # 0..3 index into CLASS_NAMES, attempts only
    onset_unix: Optional[float] = None   # wall-clock time the action window began


@dataclass
class CueSchedule:
    """A balanced, randomly interleaved sequence of attempt and rest trials."""

    trials: List[Trial] = field(default_factory=list)
    action_s: float = 2.5
    prepare_s: float = 1.0
    iti_range_s: Tuple[float, float] = (1.5, 2.5)

    @property
    def n_attempt(self) -> int:
        return sum(1 for t in self.trials if t.kind == ATTEMPT)

    @property
    def n_rest(self) -> int:
        return sum(1 for t in self.trials if t.kind == REST)

    def estimated_duration_s(self) -> float:
        per = self.prepare_s + self.action_s + float(np.mean(self.iti_range_s))
        return per * len(self.trials)


def build_schedule(
    n_per_class: int = 40,
    action_s: float = 2.5,
    prepare_s: float = 1.0,
    iti_range_s: Tuple[float, float] = (1.5, 2.5),
    seed: int = 0,
    max_run: int = 3,
) -> CueSchedule:
    """Balanced attempt/rest trials in random order, with words cycled evenly.

    `max_run` caps how many identical cues may appear consecutively. Long runs of
    one class let slow drift line up with class identity, which is the confound
    this recorder exists to avoid; capping the run length keeps both classes
    spread across the whole session.
    """
    if n_per_class < 2:
        raise ValueError("n_per_class must be >= 2 to give both classes trials.")

    rng = np.random.default_rng(seed)
    words = np.tile(np.arange(len(CLASS_NAMES)), int(np.ceil(n_per_class / len(CLASS_NAMES))))
    words = words[:n_per_class]
    rng.shuffle(words)

    kinds = [ATTEMPT] * n_per_class + [REST] * n_per_class
    for _ in range(10_000):
        rng.shuffle(kinds)
        if _max_run_length(kinds) <= max_run:
            break
    else:  # pragma: no cover - astronomically unlikely
        log.warning("Could not satisfy max_run=%d; using the last shuffle.", max_run)

    wi = iter(words)
    trials = [
        Trial(kind=k, word=(int(next(wi)) if k == ATTEMPT else None))
        for k in kinds
    ]
    sched = CueSchedule(trials=trials, action_s=action_s,
                        prepare_s=prepare_s, iti_range_s=iti_range_s)
    log.info("Schedule: %d attempt + %d rest trials, ~%.1f min.",
             sched.n_attempt, sched.n_rest, sched.estimated_duration_s() / 60.0)
    return sched


def _max_run_length(seq: List[int]) -> int:
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


class BoardRecorder:
    """Continuously drains an OpenBCI board into memory, with timestamps.

    Uses draining reads so no sample is lost or double-counted, and keeps
    BrainFlow's timestamp row so cue onsets (wall clock) can be mapped to exact
    sample indices afterwards rather than relying on polling granularity.
    """

    def __init__(self, serial_port: str, poll_s: float = 0.2):
        self.serial_port = serial_port
        self.poll_s = poll_s
        self._chunks: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.sfreq: float = 0.0
        self.eeg_rows: List[int] = []
        self.ts_row: int = -1

    def __enter__(self) -> "BoardRecorder":
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

        self.board_id = BoardIds.CYTON_DAISY_BOARD
        params = BrainFlowInputParams()
        params.serial_port = self.serial_port
        self.board = BoardShim(self.board_id, params)
        self.sfreq = float(BoardShim.get_sampling_rate(self.board_id))
        self.eeg_rows = list(BoardShim.get_eeg_channels(self.board_id))
        self.ts_row = int(BoardShim.get_timestamp_channel(self.board_id))
        self.board.prepare_session()
        self.board.start_stream()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()
        log.info("Recording from %s at %.0f Hz, %d EEG channels.",
                 self.serial_port, self.sfreq, len(self.eeg_rows))
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.board.stop_stream()
        finally:
            self.board.release_session()

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                data = self.board.get_board_data()  # draining
            except Exception:
                log.exception("board read failed; stopping recorder thread")
                return
            if data is not None and data.size and data.shape[1] > 0:
                with self._lock:
                    self._chunks.append(data)
            self._stop.wait(self.poll_s)

    def n_samples(self) -> int:
        with self._lock:
            return int(sum(c.shape[1] for c in self._chunks))

    def collected(self) -> np.ndarray:
        """All rows recorded so far, concatenated in arrival order."""
        with self._lock:
            if not self._chunks:
                return np.empty((0, 0))
            return np.concatenate(self._chunks, axis=1)


def epoch_by_onsets(
    raw: np.ndarray,
    timestamps: np.ndarray,
    trials: List[Trial],
    action_s: float,
    sfreq: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut `action_s` windows starting at each trial's recorded onset.

    Onsets are matched to the nearest board timestamp, so window placement does
    not depend on how often the recorder thread happened to poll. Trials whose
    window runs past the end of the recording are dropped with a warning rather
    than zero-padded -- padding would invent samples.
    """
    n_samp = int(round(action_s * sfreq))
    Xs, y_go, y_word = [], [], []
    dropped = 0
    for t in trials:
        if t.onset_unix is None:
            dropped += 1
            continue
        i0 = int(np.searchsorted(timestamps, t.onset_unix))
        if i0 + n_samp > raw.shape[1]:
            dropped += 1
            continue
        Xs.append(raw[:, i0:i0 + n_samp])
        y_go.append(t.kind)
        y_word.append(-1 if t.word is None else int(t.word))
    if dropped:
        log.warning("Dropped %d trial(s) that ran past the end of the recording.", dropped)
    if not Xs:
        raise ValueError("No complete trials were recorded.")
    return np.stack(Xs), np.asarray(y_go, int), np.asarray(y_word, int)


def save_calibration(
    path: str,
    X: np.ndarray,
    y_go: np.ndarray,
    y_word: np.ndarray,
    ch_names: List[str],
    sfreq: float,
    subject: int,
    session: int,
    action_s: float,
    paradigm: str = "overt",
) -> None:
    """Write one calibration recording (raw, unfiltered, board-rate)."""
    if X.shape[1] != len(ch_names):
        raise ValueError(
            f"{X.shape[1]} channels recorded but {len(ch_names)} names given. "
            "Channel names must describe the board in board order."
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        X=X.astype(np.float64),
        y_go=y_go,
        y_word=y_word,
        ch_names=np.asarray(ch_names, dtype=object),
        sfreq=float(sfreq),
        subject=int(subject),
        session=int(session),
        action_s=float(action_s),
        paradigm=paradigm,
        filtered=False,
        created_unix=time.time(),
    )
    log.info("Saved %s: %d trials (%d attempt / %d rest), %d ch @ %.0f Hz.",
             path, len(X), int((y_go == ATTEMPT).sum()), int((y_go == REST).sum()),
             X.shape[1], sfreq)


def check_signal(
    serial_port: str,
    ch_names: List[str],
    seconds: float = 4.0,
    line_freq: Optional[float] = 60.0,
    impedance: bool = True,
    impedance_input: str = "n",
) -> List:
    """Report contact quality per electrode before a session.

    Collects a plain EEG window for the continuous metrics (amplitude, mains
    noise, railing), then optionally runs the ADS1299 lead-off impedance check.
    Impedance injects current and so cannot run at the same time as the EEG
    measurement -- hence the two phases.
    """
    from .signal_quality import live_quality, measure_impedance, merge_quality

    with BoardRecorder(serial_port) as rec:
        n_board_ch = len(rec.eeg_rows)
        if n_board_ch != len(ch_names):
            raise ValueError(
                f"Board exposes {n_board_ch} EEG channels but {len(ch_names)} "
                "names were given. Pass one label per board channel, in board order."
            )
        log.info("Collecting %.1f s of EEG for signal metrics…", seconds)
        time.sleep(seconds)
        data = rec.collected()
        if data.size == 0:
            raise RuntimeError(
                "No data arrived from the board. Check the dongle, the Cyton's "
                "power switch, and that the OpenBCI GUI is closed."
            )
        window = data[rec.eeg_rows, :]
        live = live_quality(window, rec.sfreq, ch_names, line_freq=line_freq)

        if impedance:
            log.info("Measuring impedance (injects 6 nA @ 31.2 Hz, one channel at "
                     "a time; not EEG during this phase)…")
            imp = measure_impedance(rec.board, rec.eeg_rows, rec.sfreq, ch_names,
                                    input_side=impedance_input)
            live = merge_quality(live, imp)
    return live


def run_calibration(
    serial_port: str,
    schedule: CueSchedule,
    ch_names: List[str],
    present: Callable[[str, str, float], None],
    settle_s: float = 3.0,
) -> Dict:
    """Record the schedule from the board, returning epoched raw data.

    `present(kind, text, seconds)` displays a cue; it is injected so the recorder
    stays free of I/O and the CLI owns presentation.
    """
    with BoardRecorder(serial_port) as rec:
        if rec.sfreq <= 0:
            raise RuntimeError("Board reported a non-positive sampling rate.")
        n_board_ch = len(rec.eeg_rows)
        if n_board_ch != len(ch_names):
            raise ValueError(
                f"Board exposes {n_board_ch} EEG channels but {len(ch_names)} "
                "channel names were given. Pass one label per board channel, in "
                "board order."
            )

        present("settle", "Sit still — letting the amplifier settle", settle_s)
        time.sleep(settle_s)

        rng = np.random.default_rng(0)
        for i, t in enumerate(schedule.trials, 1):
            label = "REST" if t.kind == REST else f"SPEAK ALOUD: {CLASS_NAMES[t.word]}"
            present("prepare", f"[{i}/{len(schedule.trials)}] get ready", schedule.prepare_s)
            time.sleep(schedule.prepare_s)

            t.onset_unix = time.time()
            present("action", label, schedule.action_s)
            time.sleep(schedule.action_s)

            iti = float(rng.uniform(*schedule.iti_range_s))
            present("rest", "relax", iti)
            time.sleep(iti)

        # Let the tail of the last window arrive before draining.
        time.sleep(0.5)
        data = rec.collected()
        if data.size == 0:
            raise RuntimeError(
                "No data arrived from the board. Check the dongle, the Cyton's "
                "power switch, and that the OpenBCI GUI is closed."
            )
        raw = data[rec.eeg_rows, :]
        ts = data[rec.ts_row, :]
        sfreq = rec.sfreq

    X, y_go, y_word = epoch_by_onsets(raw, ts, schedule.trials, schedule.action_s, sfreq)
    return {"X": X, "y_go": y_go, "y_word": y_word, "sfreq": sfreq,
            "ch_names": list(ch_names), "action_s": schedule.action_s}
