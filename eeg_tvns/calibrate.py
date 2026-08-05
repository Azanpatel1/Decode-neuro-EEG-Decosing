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

from .boards import DEFAULT_BOARD, resolve_board_id
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
    # Pre-scheduled runs only: when the cue was *planned* to appear, how late it
    # actually appeared, and whether that lag was too large to trust. Recorded so
    # cue/EEG misalignment is measured rather than assumed (see plan_timeline).
    planned_unix: Optional[float] = None
    display_lag_s: Optional[float] = None
    lag_flagged: bool = False

    def label(self) -> str:
        return "REST" if self.kind == REST else f"SPEAK ALOUD: {CLASS_NAMES[self.word]}"


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

    def __init__(self, serial_port: str, poll_s: float = 0.2,
                 board: str = DEFAULT_BOARD):
        self.serial_port = serial_port
        self.poll_s = poll_s
        self.board_name = board
        self._chunks: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.sfreq: float = 0.0
        self.eeg_rows: List[int] = []
        self.ts_row: int = -1

    def __enter__(self) -> "BoardRecorder":
        from brainflow.board_shim import BoardShim, BrainFlowInputParams

        self.board_id = resolve_board_id(self.board_name)
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

    Returns (X, y_go, y_word, kept_indices); `kept_indices` indexes into `trials`
    so per-trial metadata can be aligned with the epochs that survived.
    """
    n_samp = int(round(action_s * sfreq))
    Xs, y_go, y_word, kept = [], [], [], []
    dropped = 0
    for i, t in enumerate(trials):
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
        kept.append(i)
    if dropped:
        log.warning("Dropped %d trial(s) that ran past the end of the recording.", dropped)
    if not Xs:
        raise ValueError("No complete trials were recorded.")
    return (np.stack(Xs), np.asarray(y_go, int), np.asarray(y_word, int),
            np.asarray(kept, int))


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
    cue_source: str = "terminal",
    display_lag_s: Optional[np.ndarray] = None,
    lag_flagged: Optional[np.ndarray] = None,
) -> None:
    """Write one calibration recording (raw, unfiltered, board-rate).

    `display_lag_s` is per trial and only present for browser-cued runs: how much
    later than planned each cue actually appeared. It is stored rather than
    averaged away so anyone re-analysing the file can see how well cue and EEG
    were aligned instead of trusting that they were.
    """
    if X.shape[1] != len(ch_names):
        raise ValueError(
            f"{X.shape[1]} channels recorded but {len(ch_names)} names given. "
            "Channel names must describe the board in board order."
        )
    n = len(X)
    lag = (np.full(n, np.nan) if display_lag_s is None
           else np.asarray(display_lag_s, dtype=float))
    flag = (np.zeros(n, dtype=bool) if lag_flagged is None
            else np.asarray(lag_flagged, dtype=bool))
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
        cue_source=cue_source,
        display_lag_s=lag,
        lag_flagged=flag,
    )
    n_flag = int(flag.sum())
    if n_flag:
        log.warning("%d trial(s) had an untrustworthy cue-display report and kept "
                    "their planned onset.", n_flag)
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
    board: str = DEFAULT_BOARD,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> List:
    """Report contact quality per electrode before a session.

    Collects a plain EEG window for the continuous metrics (amplitude, mains
    noise, railing), then optionally runs the ADS1299 lead-off impedance check.
    Impedance injects current and so cannot run at the same time as the EEG
    measurement -- hence the two phases.
    """
    from .signal_quality import live_quality, measure_impedance, merge_quality

    n_ch = len(ch_names)
    total = n_ch + 1 if impedance else 1

    def tick(done: int, stage: str) -> None:
        if on_progress is not None:
            try:
                on_progress(done, total, stage)
            except Exception:
                log.exception("signal-check progress callback failed; continuing")

    with BoardRecorder(serial_port, board=board) as rec:
        n_board_ch = len(rec.eeg_rows)
        if n_board_ch != n_ch:
            raise ValueError(
                f"Board exposes {n_board_ch} EEG channels but {n_ch} "
                "names were given. Pass one label per board channel, in board order."
            )
        log.info("Collecting %.1f s of EEG for signal metrics…", seconds)
        tick(0, f"collecting {seconds:.0f} s of EEG")
        time.sleep(seconds)
        data = rec.collected()
        if data.size == 0:
            raise RuntimeError(
                "No data arrived from the board. Check the dongle, the Cyton's "
                "power switch, and that the OpenBCI GUI is closed."
            )
        window = data[rec.eeg_rows, :]
        live = live_quality(window, rec.sfreq, ch_names, line_freq=line_freq)

        tick(1, "measuring impedance" if impedance else "done")
        if impedance:
            log.info("Measuring impedance (injects 6 nA @ 31.2 Hz, one channel at "
                     "a time; not EEG during this phase)…")

            def imp_tick(i: int, name: str) -> None:
                tick(1 + i, f"impedance on {name} ({i + 1}/{n_ch})")

            imp = measure_impedance(rec.board, rec.eeg_rows, rec.sfreq, ch_names,
                                    input_side=impedance_input, on_channel=imp_tick)
            live = merge_quality(live, imp)
        tick(total, "done")
    return live


def _record(
    serial_port: str,
    schedule: CueSchedule,
    ch_names: List[str],
    drive: Callable[[BoardRecorder], None],
    board: str = DEFAULT_BOARD,
    tail_s: float = 0.5,
) -> Dict:
    """Open the board, let `drive` present the run, then epoch what was recorded.

    Shared by both cue drivers (terminal and pre-scheduled browser) so the part
    that touches hardware and cuts epochs exists exactly once. `drive` is
    responsible only for presenting cues and stamping each trial's `onset_unix`.
    """
    with BoardRecorder(serial_port, board=board) as rec:
        if rec.sfreq <= 0:
            raise RuntimeError("Board reported a non-positive sampling rate.")
        n_board_ch = len(rec.eeg_rows)
        if n_board_ch != len(ch_names):
            raise ValueError(
                f"Board exposes {n_board_ch} EEG channels but {len(ch_names)} "
                "channel names were given. Pass one label per board channel, in "
                "board order."
            )

        drive(rec)

        # Let the tail of the last window arrive before draining.
        time.sleep(tail_s)
        data = rec.collected()
        if data.size == 0:
            raise RuntimeError(
                "No data arrived from the board. Check the dongle, the Cyton's "
                "power switch, and that the OpenBCI GUI is closed."
            )
        raw = data[rec.eeg_rows, :]
        ts = data[rec.ts_row, :]
        sfreq = rec.sfreq

    X, y_go, y_word, kept = epoch_by_onsets(
        raw, ts, schedule.trials, schedule.action_s, sfreq)
    lags = np.array([
        np.nan if schedule.trials[i].display_lag_s is None
        else schedule.trials[i].display_lag_s for i in kept], dtype=float)
    flagged = np.array([schedule.trials[i].lag_flagged for i in kept], dtype=bool)
    return {"X": X, "y_go": y_go, "y_word": y_word, "sfreq": sfreq,
            "ch_names": list(ch_names), "action_s": schedule.action_s,
            "display_lag_s": lags, "lag_flagged": flagged}


def run_calibration(
    serial_port: str,
    schedule: CueSchedule,
    ch_names: List[str],
    present: Callable[[str, str, float], None],
    settle_s: float = 3.0,
    board: str = DEFAULT_BOARD,
) -> Dict:
    """Record the schedule from the board, returning epoched raw data.

    `present(kind, text, seconds)` displays a cue; it is injected so the recorder
    stays free of I/O and the CLI owns presentation. Each onset is stamped when
    the cue is issued, which is right for a terminal where printing is immediate.
    """
    def drive(rec: BoardRecorder) -> None:
        present("settle", "Sit still — letting the amplifier settle", settle_s)
        time.sleep(settle_s)

        rng = np.random.default_rng(0)
        for i, t in enumerate(schedule.trials, 1):
            present("prepare", f"[{i}/{len(schedule.trials)}] get ready", schedule.prepare_s)
            time.sleep(schedule.prepare_s)

            t.onset_unix = time.time()
            present("action", t.label(), schedule.action_s)
            time.sleep(schedule.action_s)

            iti = float(rng.uniform(*schedule.iti_range_s))
            present("rest", "relax", iti)
            time.sleep(iti)

    return _record(serial_port, schedule, ch_names, drive, board=board)


# ---------------------------------------------------------------------------
# Pre-scheduled cues, for a browser front end
# ---------------------------------------------------------------------------
# A browser cannot be cued per trial over the network without inheriting the
# network's jitter into the epoch boundaries. So the server plans the whole run as
# absolute wall-clock times up front, the browser syncs its clock once and
# schedules every cue locally, and then reports back when each cue was actually
# painted. The onset used for epoching is the *reported paint time* when it is
# credible, and the planned time otherwise -- with the difference recorded per
# trial, so cue/EEG alignment is a measured number in the file rather than an
# assumption.
DISPLAY_TOLERANCE_S = 0.5


@dataclass
class Timeline:
    """Absolute times for a whole run, handed to the browser as the cue script."""

    id: str
    start_at: float          # settle cue appears
    settle_until: float      # first trial's prepare begins
    end_at: float
    rows: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"id": self.id, "start_at": self.start_at,
                "settle_until": self.settle_until, "end_at": self.end_at,
                "trials": self.rows}


def plan_timeline(schedule: CueSchedule, start_at: float, settle_s: float = 3.0,
                  seed: int = 0) -> Timeline:
    """Lay the schedule out on the wall clock, ITIs included.

    Also seeds each trial's `onset_unix` with its planned time, so a run whose
    paint reports never arrive still epochs at sensible boundaries instead of
    dropping every trial.
    """
    rng = np.random.default_rng(seed)
    t = start_at + settle_s
    rows: List[Dict] = []
    for i, tr in enumerate(schedule.trials):
        prepare_at = t
        action_at = prepare_at + schedule.prepare_s
        iti_at = action_at + schedule.action_s
        t = iti_at + float(rng.uniform(*schedule.iti_range_s))
        tr.planned_unix = action_at
        tr.onset_unix = action_at
        tr.display_lag_s = None
        tr.lag_flagged = False
        rows.append({"index": i, "kind": int(tr.kind), "word": tr.word,
                     "label": tr.label(), "prepare_at": prepare_at,
                     "action_at": action_at, "iti_at": iti_at})
    return Timeline(id=f"{start_at:.3f}-{len(schedule.trials)}", start_at=start_at,
                    settle_until=start_at + settle_s, end_at=t, rows=rows)


class ScheduledCueRun:
    """Accepts paint reports for a planned timeline and tracks the display lag."""

    def __init__(self, schedule: CueSchedule, timeline: Timeline,
                 tolerance_s: float = DISPLAY_TOLERANCE_S):
        self.schedule = schedule
        self.timeline = timeline
        self.tolerance_s = tolerance_s
        self._lock = threading.Lock()
        self.n_reported = 0
        self.n_flagged = 0

    def report_paint(self, index: int, shown_at: float) -> None:
        """Record when the browser actually painted trial `index`'s action cue."""
        if not (0 <= index < len(self.schedule.trials)):
            return
        tr = self.schedule.trials[index]
        if tr.planned_unix is None:
            return
        lag = shown_at - tr.planned_unix
        with self._lock:
            tr.display_lag_s = lag
            # A cue cannot genuinely appear before it was scheduled, and a very
            # late one means the tab was throttled or the clocks disagree. Either
            # way the report is not evidence, so keep the planned onset and flag it.
            if -0.05 <= lag <= self.tolerance_s:
                tr.onset_unix = shown_at
                tr.lag_flagged = False
            else:
                tr.onset_unix = tr.planned_unix
                tr.lag_flagged = True
                self.n_flagged += 1
            self.n_reported += 1

    def lag_stats(self) -> Dict:
        with self._lock:
            lags = [t.display_lag_s for t in self.schedule.trials
                    if t.display_lag_s is not None]
            flagged = self.n_flagged
        if not lags:
            return {"n": 0, "flagged": flagged}
        arr = np.asarray(lags, dtype=float) * 1000.0
        return {"n": len(lags), "median_ms": float(np.median(arr)),
                "max_ms": float(np.max(np.abs(arr))), "flagged": int(flagged)}


def run_calibration_scheduled(
    serial_port: str,
    schedule: CueSchedule,
    ch_names: List[str],
    timeline: Timeline,
    board: str = DEFAULT_BOARD,
    on_trial: Optional[Callable[[int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict:
    """Record a pre-planned timeline while the browser presents the cues.

    This side does no presentation at all: it opens the board, keeps it streaming
    until the planned end, and reports which trial the clock is on so the server
    can show progress.
    """
    def drive(rec: BoardRecorder) -> None:
        n = len(schedule.trials)
        last = -1
        while True:
            now = time.time()
            if now >= timeline.end_at:
                break
            if should_stop is not None and should_stop():
                raise KeyboardInterrupt("calibration aborted")
            idx = sum(1 for r in timeline.rows if r["action_at"] <= now)
            if idx != last and on_trial is not None:
                last = idx
                on_trial(min(idx, n))
            time.sleep(0.1)
        if on_trial is not None:
            on_trial(n)

    return _record(serial_port, schedule, ch_names, drive, board=board)
