"""Live-decode plumbing: display buffer, word readout, and the acquisition thread.

This is the layer between `run_loop` and anything that wants to watch it. It was
part of `dashboard.py` when the dashboard was the only consumer; the session
manager now starts and stops decode sessions too, so it lives here and both import
it rather than importing each other.

Nothing here is on the decision path. `run_loop` makes and acts on its GO decision
first and only then calls the observer, so display work, contact metrics, and the
word readout cannot delay or influence stimulation (Invariants C and E).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Callable, Deque, List, Optional

import numpy as np

from .acquisition import fire_tvns, load_decoder, run_loop
from .boards import explain_board_error
from .config import CLASS_NAMES
from .realtime import Decision, RealTimeDecoder
from .signal_quality import live_quality, scalp_positions

log = logging.getLogger("eeg_tvns.live")


# ---------------------------------------------------------------------------
# Thread-safe frame hub
# ---------------------------------------------------------------------------
class FrameHub:
    """Rolling display buffer + latest decode state.

    The acquisition thread calls `publish(...)` after each decode. The UI reads
    the current snapshot from `snapshot()`.

    Also retains the exact window that produced the most recent tVNS fire, so the
    UI can show the evidence behind a stimulation event rather than only the
    continuously scrolling trace.
    """

    def __init__(
        self,
        n_channels: int,
        ch_names: List[str],
        display_sfreq: float,
        buffer_s: float = 6.0,
        history_n: int = 600,
        source: str = "openbci",
        model_path: str = "",
        word_names: Optional[dict] = None,
        word_model_path: str = "",
        go_threshold: float = 0.5,
    ):
        self.n_channels = n_channels
        self.ch_names = ch_names
        self.display_sfreq = float(display_sfreq)
        self.buffer_samples = int(round(buffer_s * self.display_sfreq))
        self.source = source
        self.model_path = model_path
        self.word_names = word_names or {}
        self.word_model_path = word_model_path
        self.go_threshold = float(go_threshold)

        self._lock = threading.Lock()
        self._eeg = np.zeros((n_channels, self.buffer_samples), dtype=np.float32)
        self._t0 = time.perf_counter()

        self._p_history: Deque[float] = deque(maxlen=history_n)
        self._go_history: Deque[int] = deque(maxlen=history_n)
        self._lat_history: Deque[float] = deque(maxlen=history_n)
        self._t_history: Deque[float] = deque(maxlen=history_n)
        # A GO event is a threshold crossing that passed the refractory check; a
        # stimulation is one that actually reached the device. They differ whenever
        # the session is disarmed, and conflating them would let the display claim
        # stimulation that never happened.
        self._go_events: Deque[float] = deque(maxlen=history_n)
        self._stim_times: Deque[float] = deque(maxlen=history_n)

        self._word: dict = {"label": None, "name": None, "probs": {}, "latency_ms": 0.0}
        self._trigger: Optional[dict] = None
        self._status: dict = {"state": "starting", "detail": ""}
        self._quality: List[dict] = []
        self._impedance: dict = {}
        # Top-down scalp coordinates for the trained montage, so the head map can
        # draw each electrode where it actually sits. Sites we cannot place are
        # reported rather than drawn somewhere invented.
        self.scalp = scalp_positions(self.ch_names)
        self.unplaced = [c for c in self.ch_names if c not in self.scalp]
        if self.unplaced:
            log.warning("No 10-20 position for %s; omitted from the head map.",
                        ", ".join(self.unplaced))

        self._latest: dict = {
            "t": 0.0,
            "p_go": 0.0,
            "go": False,
            "fired": False,
            "latency_ms": 0.0,
            "predicted_label": -1,
            "n_decisions": 0,
            "n_go_events": 0,
            "n_stimulations": 0,
        }
        self._frame_id = 0

    def set_quality(self, qualities: List[dict]) -> None:
        with self._lock:
            self._quality = qualities

    def quality(self) -> List[dict]:
        with self._lock:
            return list(self._quality)

    def set_impedance(self, data: dict) -> None:
        """Attach impedance results from a prior contact check.

        Impedance needs injected current, so it cannot be measured while decoding;
        the UI shows these with their age to make clear they are not live.
        """
        with self._lock:
            self._impedance = data or {}

    def set_status(self, state: str, detail: str = "") -> None:
        """Report acquisition health so the UI can distinguish 'no signal' from
        'stream never started' (e.g. board powered off)."""
        with self._lock:
            self._status = {"state": state, "detail": detail}

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def _downsample(self, window: np.ndarray) -> np.ndarray:
        """Stride-decimate a window for display only; never fed back to decode()."""
        stride = max(1, int(round(window.shape[1] / max(1, int(self.display_sfreq * 2)))))
        return window[: self.n_channels, ::stride].astype(np.float32, copy=False)

    # -- called from acquisition thread ------------------------------------
    def publish(
        self,
        window: np.ndarray,
        decision: Decision,
        fired: bool,
        word: Optional[dict] = None,
        stimulated: bool = True,
    ) -> None:
        """`fired` = a GO event passed the refractory check; `stimulated` = the
        stimulator was actually triggered (false while disarmed)."""
        now = time.perf_counter() - self._t0
        display = self._downsample(window)
        # Freeze the window behind the GO event before taking the lock.
        trigger = None
        if fired:
            trigger = {
                "t": now,
                "p_go": float(decision.probability),
                "threshold": self.go_threshold,
                "latency_ms": float(decision.latency_ms),
                "window_s": window.shape[1] / self.display_sfreq,
                "eeg": display.tolist(),
                "word": dict(word) if word else None,
                "stimulated": bool(stimulated),
            }
        with self._lock:
            n_new = display.shape[1]
            if n_new >= self.buffer_samples:
                self._eeg[: display.shape[0], :] = display[:, -self.buffer_samples :]
            else:
                self._eeg = np.roll(self._eeg, -n_new, axis=1)
                self._eeg[: display.shape[0], -n_new:] = display
            self._p_history.append(float(decision.probability))
            self._go_history.append(1 if decision.go else 0)
            self._lat_history.append(float(decision.latency_ms))
            self._t_history.append(now)
            if fired:
                self._go_events.append(now)
                if stimulated:
                    self._stim_times.append(now)
                self._trigger = trigger
            if word is not None:
                self._word = word
            self._latest = {
                "t": now,
                "p_go": float(decision.probability),
                "go": bool(decision.go),
                "fired": bool(fired),
                "stimulated": bool(fired and stimulated),
                "latency_ms": float(decision.latency_ms),
                "predicted_label": int(decision.predicted_label),
                "n_decisions": len(self._p_history),
                "n_go_events": len(self._go_events),
                "n_stimulations": len(self._stim_times),
            }
            self._frame_id += 1

    # -- called from UI thread ---------------------------------------------
    def snapshot(self, include_eeg: bool = True) -> dict:
        """Current state for the UI.

        `include_eeg` is False when no view is drawing traces: the buffer is by far
        the largest part of the payload and there is no reason to serialise it for
        a tab that will not draw it.
        """
        with self._lock:
            lat_arr = np.asarray(self._lat_history) if self._lat_history else np.array([0.0])
            snap = {
                "frame_id": self._frame_id,
                "source": self.source,
                "model_path": self.model_path,
                "word_model_path": self.word_model_path,
                "word_names": self.word_names,
                "go_threshold": self.go_threshold,
                "ch_names": self.ch_names,
                "display_sfreq": self.display_sfreq,
                "buffer_s": self.buffer_samples / self.display_sfreq,
                "eeg": self._eeg.tolist() if include_eeg else [],
                "p_history": list(self._p_history),
                "go_history": list(self._go_history),
                "t_history": list(self._t_history),
                "go_events": list(self._go_events),
                "stim_times": list(self._stim_times),
                "latest": dict(self._latest),
                "word": dict(self._word),
                "trigger": dict(self._trigger) if self._trigger else None,
                "status": dict(self._status),
                "quality": list(self._quality),
                "impedance": dict(self._impedance),
                "scalp": self.scalp,
                "unplaced": list(self.unplaced),
                "latency_stats": {
                    "median_ms": float(np.median(lat_arr)),
                    "p95_ms": float(np.percentile(lat_arr, 95)),
                    "last_ms": float(lat_arr[-1]) if lat_arr.size else 0.0,
                },
            }
        if not include_eeg and snap["trigger"]:
            snap["trigger"] = {k: v for k, v in snap["trigger"].items() if k != "eeg"}
        return snap


def explain_stream_error(exc: BaseException) -> str:
    """Turn a streamer failure into something actionable in the UI."""
    return explain_board_error(exc)


class WordReadout:
    """Display-only 4-class word decoder (up / down / right / left).

    Invariant C: the word decoder must never gate stimulation. That is enforced
    structurally here -- this runs inside the dashboard's `on_frame` *observer*,
    which is called after `run_loop` has already made and acted on its GO
    decision, and has no path back into the fire logic.

    Only decoded on GO frames: identity is meaningless during rest, and it keeps
    the extra work off the majority of loop iterations.
    """

    def __init__(self, decoder: RealTimeDecoder, label_names: dict):
        self.decoder = decoder
        self.label_names = label_names

    def decode(self, window: np.ndarray) -> dict:
        d = self.decoder.decode(window)
        probs = {}
        if d.probabilities and d.classes:
            for cls, p in zip(d.classes, d.probabilities):
                probs[self.label_names.get(int(cls), str(cls))] = float(p)
        return {
            "label": int(d.predicted_label),
            "name": self.label_names.get(int(d.predicted_label), str(d.predicted_label)),
            "probs": probs,
            "latency_ms": float(d.latency_ms),
        }


def load_word_readout(path: Optional[str],
                      n_go_channels: Optional[int]) -> Optional[WordReadout]:
    """Load the 4-class word decoder for display. Returns None if unavailable.

    A missing or mismatched word model degrades the readout only -- the GO gate
    and stimulation path are unaffected.
    """
    if not path or path.lower() == "none":
        return None
    if not os.path.exists(path):
        log.warning("word model %s not found -- word readout disabled. "
                    "Train one with: python run.py --data ./ds003626 --task word", path)
        return None
    try:
        w_bundle, _w_cfg, w_decoder, _, w_n_ch = load_decoder(path)
    except Exception:
        log.exception("could not load word model %s -- word readout disabled", path)
        return None
    if n_go_channels and w_n_ch and w_n_ch != n_go_channels:
        log.warning("word model has %d channels but GO model has %d -- word readout "
                    "disabled to avoid feeding it mismatched windows", w_n_ch, n_go_channels)
        return None
    label_names = w_bundle.get("label_names") or CLASS_NAMES
    label_names = {int(k): str(v) for k, v in label_names.items()}
    log.info("word readout (display only): %s | classes: %s",
             path, ", ".join(label_names[k] for k in sorted(label_names)))
    return WordReadout(w_decoder, label_names)


class AcquisitionThread(threading.Thread):
    def __init__(self, streamer, decoder: RealTimeDecoder, hub: FrameHub,
                 hop_s: float, refractory_s: float,
                 word_readout: Optional[WordReadout] = None,
                 line_freq: float = 60.0, quality_period_s: float = 0.5,
                 fire: Optional[Callable[[], None]] = None,
                 stimulated: Optional[Callable[[], bool]] = None,
                 on_exit: Optional[Callable[[Optional[BaseException]], None]] = None):
        super().__init__(daemon=True, name="eeg-tvns-acq")
        self.streamer = streamer
        self.decoder = decoder
        self.hub = hub
        self.hop_s = hop_s
        self.refractory_s = refractory_s
        self.word_readout = word_readout
        self.line_freq = line_freq
        self.quality_period_s = quality_period_s
        # Injected so stimulation can be gated behind an ARM switch without the
        # loop knowing anything about it. Defaults to the real trigger, which is
        # what the CLI path has always done.
        self.fire = fire if fire is not None else fire_tvns
        # Reports whether the most recent `fire` call actually reached the device.
        # `run_loop` calls fire() then on_frame() on this same thread, so reading it
        # in the observer is deterministic. Defaults to True: the CLI's fire_tvns is
        # unconditional, so there every GO event is a stimulation.
        self.stimulated = stimulated if stimulated is not None else (lambda: True)
        self.on_exit = on_exit
        self._last_quality_t = 0.0
        self._stop = threading.Event()
        self.error: Optional[BaseException] = None

    def stop(self) -> None:
        self._stop.set()

    def _update_quality(self) -> None:
        """Contact metrics from the raw board window, throttled.

        Runs in the observer (after the GO decision has been made and acted on) and
        only a couple of times a second, so the FFTs never sit on the decision path.
        Uses the *raw* window: the processed one has mains notched out and DC
        removed, which is exactly what these metrics look for.
        """
        now = time.perf_counter()
        if now - self._last_quality_t < self.quality_period_s:
            return
        raw = getattr(self.streamer, "last_raw", None)
        if raw is None:
            return
        self._last_quality_t = now
        order = getattr(self.streamer, "channel_order", None)
        eeg = raw[order, :] if order is not None else raw
        names = self.hub.ch_names
        if eeg.shape[0] != len(names):
            return
        qs = live_quality(eeg, self.streamer.board_fs, names, line_freq=self.line_freq)
        self.hub.set_quality([q.to_dict() for q in qs])

    def _on_frame(self, window: np.ndarray, decision: Decision, fired: bool) -> None:
        word = None
        if self.word_readout is not None and decision.go:
            word = self.word_readout.decode(window)
        stim = bool(fired and self.stimulated())
        self.hub.publish(window, decision, fired, word=word, stimulated=stim)
        try:
            self._update_quality()
        except Exception:
            log.exception("quality update failed; continuing")

    def run(self) -> None:
        err: Optional[BaseException] = None
        try:
            with self.streamer:
                self.hub.set_status("streaming")
                run_loop(
                    self.streamer,
                    self.decoder,
                    self.fire,
                    hop_s=self.hop_s,
                    refractory_s=self.refractory_s,
                    on_frame=self._on_frame,
                    stop_flag=self._stop.is_set,
                )
            self.hub.set_status("stopped")
        except BaseException as e:
            err = self.error = e
            self.hub.set_status("error", explain_stream_error(e))
            log.exception("acquisition thread crashed")
        finally:
            if self.on_exit is not None:
                try:
                    self.on_exit(err)
                except Exception:
                    log.exception("acquisition on_exit hook raised")
