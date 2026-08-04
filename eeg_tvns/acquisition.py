"""Live hardware acquisition -> eeg_tvns closed-loop decoder.

Bridges a real-time EEG stream into the trained decoder and gates tVNS on GO:

    board window (n_ch, n_samp @ board_rate)
      -> channel reorder to the model's trained montage
      -> band-pass + notch   (SAME shared filter as training)
      -> resample to the model's training rate
      -> RealTimeDecoder.decode(window)  -> GO gate -> fire_tvns()

Two streamers share one interface:
  * OpenBCIStreamer   - real OpenBCI Cyton+Daisy (16 ch) via BrainFlow.
  * SimulatedStreamer - pure NumPy, no hardware: lets you exercise the whole
                        closed loop (and this integrates cleanly into tests).

Run live:
    python -m eeg_tvns.acquisition --port /dev/ttyUSB0 --model outputs/model_go.joblib
Dry-run with no hardware:
    python -m eeg_tvns.acquisition --simulate --model outputs/model_go.joblib --duration 8

Install for real hardware (same env as eeg_tvns):
    pip install brainflow
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Callable, List, Optional

import joblib
import numpy as np

from .config import Config
from .preprocessing import (
    apply_filters,
    channel_order_from_names,
    design_filters,
    resample_window,
)
from .realtime import Decision, RealTimeDecoder

log = logging.getLogger("eeg_tvns.acquire")


# ---------------------------------------------------------------------------
# Streamer interface
# ---------------------------------------------------------------------------
class BaseStreamer:
    """Owns a data source and yields decode-ready windows.

    Subclasses set: self.board_fs, self.n_source_channels, and implement
    _read_raw(win_samples) -> (n_source_channels, win_samples) | None.
    The shared process_window() does reorder -> filter -> resample so both
    streamers feed the decoder identically-prepared data.
    """

    def __init__(
        self,
        cfg: Config,
        model_sfreq: float,
        channel_order: Optional[List[int]] = None,
        resample: bool = True,
    ):
        self.cfg = cfg
        self.model_sfreq = model_sfreq
        self.channel_order = channel_order
        self.resample = resample
        self.board_fs: float = model_sfreq          # subclasses override
        self.n_source_channels: int = 0             # subclasses override
        self._bp = self._notch = None

    def _init_filters(self) -> None:
        self._bp, self._notch = design_filters(
            self.board_fs, self.cfg.l_freq, self.cfg.h_freq, self.cfg.notch
        )
        self.win_samples = int(round(self.cfg.window_s * self.board_fs))

    # -- context management -------------------------------------------------
    def __enter__(self):
        self._open()
        self._init_filters()
        log.info(
            "%s @ %.0f Hz | %d source ch | window=%d samp | resample->%.0f Hz: %s",
            type(self).__name__, self.board_fs, self.n_source_channels,
            self.win_samples, self.model_sfreq, self.resample,
        )
        return self

    def __exit__(self, *exc):
        self._close()

    # -- to override --------------------------------------------------------
    def _open(self) -> None: ...
    def _close(self) -> None: ...
    def _read_raw(self, win_samples: int) -> Optional[np.ndarray]: ...

    # -- shared window preparation -----------------------------------------
    def process_window(self, raw: np.ndarray) -> np.ndarray:
        """raw: (n_source_channels, win_samples) -> (n_model_channels, n_samp)."""
        eeg = raw
        if self.channel_order is not None:
            eeg = eeg[self.channel_order, :]
        eeg = apply_filters(eeg, self._bp, self._notch)
        if self.resample:
            eeg = resample_window(eeg, self.board_fs, self.model_sfreq)
        return eeg

    def latest_window(self) -> Optional[np.ndarray]:
        raw = self._read_raw(self.win_samples)
        if raw is None or raw.shape[1] < self.win_samples:
            return None
        return self.process_window(raw)


# ---------------------------------------------------------------------------
# Real hardware: OpenBCI Cyton+Daisy via BrainFlow
# ---------------------------------------------------------------------------
class OpenBCIStreamer(BaseStreamer):
    def __init__(self, serial_port: str, cfg: Config, model_sfreq: float,
                 channel_order=None, resample: bool = True):
        super().__init__(cfg, model_sfreq, channel_order, resample)
        self.serial_port = serial_port

    def _open(self):
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

        self.board_id = BoardIds.CYTON_DAISY_BOARD           # 16 ch, id=2
        params = BrainFlowInputParams()
        params.serial_port = self.serial_port
        self.board = BoardShim(self.board_id, params)
        self.board_fs = float(BoardShim.get_sampling_rate(self.board_id))   # 125 Hz
        self.eeg_rows = BoardShim.get_eeg_channels(self.board_id)           # 16 rows
        self.n_source_channels = len(self.eeg_rows)
        self.board.prepare_session()
        self.board.start_stream()

    def _close(self):
        try:
            self.board.stop_stream()
        finally:
            self.board.release_session()

    def _read_raw(self, win_samples: int) -> Optional[np.ndarray]:
        data = self.board.get_current_board_data(win_samples)  # non-draining
        if data.shape[1] < win_samples:
            return None
        return data[self.eeg_rows, :]


# ---------------------------------------------------------------------------
# No-hardware simulator (for dry-runs and tests)
# ---------------------------------------------------------------------------
class SimulatedStreamer(BaseStreamer):
    """Generate windows with alternating 'attempt-like' (high-power) and
    'rest-like' (low-power) segments so the closed loop actually toggles GO."""

    def __init__(self, cfg: Config, model_sfreq: float, n_channels: int,
                 board_fs: float = 125.0, seed: int = 0):
        super().__init__(cfg, model_sfreq, channel_order=None, resample=True)
        self.board_fs = float(board_fs)
        self.n_source_channels = n_channels
        self._rng = np.random.default_rng(seed)
        self._t0 = None

    def _open(self):
        self._t0 = time.perf_counter()

    def _read_raw(self, win_samples: int) -> Optional[np.ndarray]:
        # ~2 s attempt / 2 s rest cycle drives the GO gate up and down.
        phase = (time.perf_counter() - self._t0) % 4.0
        amp = 1.4 if phase < 2.0 else 0.4
        x = self._rng.normal(0, amp, size=(self.n_source_channels, win_samples))
        # add shared low-freq structure so band-pass leaves real content
        t = np.arange(win_samples) / self.board_fs
        x += 0.5 * amp * np.sin(2 * np.pi * 10 * t)[None, :]
        return x


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------
def run_loop(
    streamer: BaseStreamer,
    decoder: RealTimeDecoder,
    fire_tvns: Callable[[], None],
    hop_s: float,
    refractory_s: float = 1.0,
    duration_s: Optional[float] = None,
    on_frame: Optional[Callable[[np.ndarray, Decision, bool], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> dict:
    """Slide over the live stream, gate tVNS on GO, recenter after each fire.

    Returns a small run summary (decisions, fires, latency stats) -- handy for
    logging and for tests. Runs until Ctrl-C or `duration_s` elapses.

    Optional hooks (used by the dashboard; behavior unchanged when omitted):
      * on_frame(window, decision, fired) -- observer called after each decode.
      * stop_flag()                        -- callable; when True the loop exits.
    """
    last_fire = -1e9
    t_start = time.perf_counter()
    n_dec = n_fire = 0
    lats: List[float] = []
    try:
        while True:
            if duration_s is not None and (time.perf_counter() - t_start) >= duration_s:
                break
            if stop_flag is not None and stop_flag():
                break
            time.sleep(hop_s)
            win = streamer.latest_window()
            if win is None:
                continue
            d = decoder.decode(win)
            n_dec += 1
            lats.append(d.latency_ms)
            now = time.perf_counter()
            fired = False
            if d.go and (now - last_fire) >= refractory_s:
                fire_tvns()
                last_fire = now
                n_fire += 1
                fired = True
                decoder.update_reference(win)       # online recentering after event
            log.info("p_go=%.3f  latency=%.2f ms  GO=%s%s",
                     d.probability, d.latency_ms, d.go, "  <FIRE>" if fired else "")
            if on_frame is not None:
                try:
                    on_frame(win, d, fired)
                except Exception:  # observer must never break the loop
                    log.exception("on_frame observer raised; continuing")
    except KeyboardInterrupt:
        log.info("stopped by user")
    lats_arr = np.asarray(lats) if lats else np.array([0.0])
    summary = {
        "decisions": n_dec,
        "fires": n_fire,
        "median_latency_ms": float(np.median(lats_arr)),
        "p95_latency_ms": float(np.percentile(lats_arr, 95)),
    }
    log.info("run summary: %s", summary)
    return summary


def fire_tvns():
    """Stub stimulator trigger. Replace with your device's serial/GPIO/TTL call."""
    log.info(">>> tVNS FIRE")


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------
def _resolve_channel_order(bundle, board_names: Optional[List[str]]) -> Optional[List[int]]:
    """Build a board->training channel remap from names if provided."""
    train_names = bundle.get("ch_names")
    if not board_names or not train_names:
        return None
    order, missing = channel_order_from_names(board_names, train_names)
    if missing:
        log.warning(
            "Montage mismatch: %d trained channel(s) not found on the board: %s. "
            "Decoding will proceed on the matched channels only -- verify wiring.",
            len(missing), missing,
        )
    return order or None


def load_decoder(model_path: str):
    bundle = joblib.load(model_path)
    cfg: Config = bundle["config"]
    model_sfreq = float(bundle.get("sfreq", cfg.sfreq))
    decoder = RealTimeDecoder(bundle["model"], cfg, positive_label=1)
    n_model_ch = None
    if bundle.get("ch_names"):
        n_model_ch = len(bundle["ch_names"])
    return bundle, cfg, decoder, model_sfreq, n_model_ch


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="eeg_tvns live acquisition + closed loop")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="OpenBCI dongle serial port, e.g. /dev/ttyUSB0 or COM3")
    src.add_argument("--simulate", action="store_true", help="no hardware: run the loop on a numpy stream")

    ap.add_argument("--model", default="outputs/model_go.joblib")
    ap.add_argument("--hop", type=float, default=None, help="seconds between decisions")
    ap.add_argument("--refractory", type=float, default=1.0, help="min seconds between tVNS fires")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds (default: until Ctrl-C)")
    ap.add_argument("--threshold", type=float, default=None, help="override GO probability threshold")
    ap.add_argument("--no-resample", action="store_true", help="do not resample board windows to the model rate")
    ap.add_argument("--channel-names", default=None,
                    help="comma-separated physical board channel labels, in board order, "
                         "to remap onto the model's trained montage (e.g. 'F7,F8,FC5,...').")
    args = ap.parse_args(argv)

    bundle, cfg, decoder, model_sfreq, n_model_ch = load_decoder(args.model)
    if args.threshold is not None:
        cfg.go_threshold = args.threshold
    hop = args.hop if args.hop is not None else cfg.hop_s
    board_names = [c.strip() for c in args.channel_names.split(",")] if args.channel_names else None
    channel_order = _resolve_channel_order(bundle, board_names)

    if args.simulate:
        n_ch = n_model_ch or cfg.n_synth_channels
        streamer = SimulatedStreamer(cfg, model_sfreq, n_channels=n_ch)
    else:
        streamer = OpenBCIStreamer(
            args.port, cfg, model_sfreq,
            channel_order=channel_order, resample=not args.no_resample,
        )

    with streamer:
        run_loop(streamer, decoder, fire_tvns, hop_s=hop,
                 refractory_s=args.refractory, duration_s=args.duration)


if __name__ == "__main__":
    main()
