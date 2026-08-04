"""Live EEG + decode dashboard.

Runs the same closed-loop acquisition as `eeg_tvns.acquisition`, but publishes
each frame (EEG window + Decision) to a browser dashboard over a WebSocket.

    python -m eeg_tvns.dashboard --port /dev/cu.usbserial-XXXX \
        --model outputs/model_go.joblib

Then open http://127.0.0.1:8765

Everything shown is real recorded EEG from the board. There is no simulated
source: on a strip chart, generated traces look exactly like brain activity, so
the platform will not display anything it did not measure.

The decode path is untouched: this only observes windows already produced by
the streamer and downsamples them for display.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from .acquisition import (
    OpenBCIStreamer,
    _resolve_channel_order,
    fire_tvns,
    load_decoder,
    run_loop,
)
from .config import CLASS_NAMES, Config
from .realtime import Decision, RealTimeDecoder
from .signal_quality import live_quality, scalp_positions

log = logging.getLogger("eeg_tvns.dashboard")


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
        self._fire_times: Deque[float] = deque(maxlen=history_n)

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
            "n_fires": 0,
        }
        self._frame_id = 0

    def set_quality(self, qualities: List[dict]) -> None:
        with self._lock:
            self._quality = qualities

    def set_impedance(self, data: dict) -> None:
        """Attach impedance results from a prior `calibrate.py --check-signal` run.

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
    ) -> None:
        now = time.perf_counter() - self._t0
        display = self._downsample(window)
        # Freeze the window that triggered stimulation before taking the lock.
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
                self._fire_times.append(now)
                self._trigger = trigger
            if word is not None:
                self._word = word
            self._latest = {
                "t": now,
                "p_go": float(decision.probability),
                "go": bool(decision.go),
                "fired": bool(fired),
                "latency_ms": float(decision.latency_ms),
                "predicted_label": int(decision.predicted_label),
                "n_decisions": len(self._p_history),
                "n_fires": len(self._fire_times),
            }
            self._frame_id += 1

    # -- called from UI thread ---------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            lat_arr = np.asarray(self._lat_history) if self._lat_history else np.array([0.0])
            return {
                "frame_id": self._frame_id,
                "source": self.source,
                "model_path": self.model_path,
                "word_model_path": self.word_model_path,
                "word_names": self.word_names,
                "go_threshold": self.go_threshold,
                "ch_names": self.ch_names,
                "display_sfreq": self.display_sfreq,
                "buffer_s": self.buffer_samples / self.display_sfreq,
                "eeg": self._eeg.tolist(),
                "p_history": list(self._p_history),
                "go_history": list(self._go_history),
                "t_history": list(self._t_history),
                "fire_times": list(self._fire_times),
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


# ---------------------------------------------------------------------------
# Acquisition thread
# ---------------------------------------------------------------------------
def _explain_stream_error(exc: BaseException) -> str:
    """Turn a streamer failure into something actionable in the UI."""
    text = str(exc)
    if "BOARD_NOT_READY" in text or "BOARD_NOT_READY_ERROR" in text:
        return ("Board not ready. The dongle is detected but the Cyton is not "
                "responding -- check the board's power switch, then restart.")
    if "ANOTHER_BOARD_IS_CREATED" in text or "PORT_ALREADY_OPEN" in text:
        return "Serial port is busy. Quit the OpenBCI GUI, then restart."
    if "UNABLE_TO_OPEN_PORT" in text:
        return "Could not open the serial port. Check --port and that the dongle is plugged in."
    return f"{type(exc).__name__}: {text}"


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


class AcquisitionThread(threading.Thread):
    def __init__(self, streamer, decoder: RealTimeDecoder, hub: FrameHub,
                 hop_s: float, refractory_s: float,
                 word_readout: Optional[WordReadout] = None,
                 line_freq: float = 60.0, quality_period_s: float = 0.5):
        super().__init__(daemon=True, name="eeg-tvns-acq")
        self.streamer = streamer
        self.decoder = decoder
        self.hub = hub
        self.hop_s = hop_s
        self.refractory_s = refractory_s
        self.word_readout = word_readout
        self.line_freq = line_freq
        self.quality_period_s = quality_period_s
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
        self.hub.publish(window, decision, fired, word=word)
        try:
            self._update_quality()
        except Exception:
            log.exception("quality update failed; continuing")

    def run(self) -> None:
        try:
            with self.streamer:
                self.hub.set_status("streaming")
                run_loop(
                    self.streamer,
                    self.decoder,
                    fire_tvns,
                    hop_s=self.hop_s,
                    refractory_s=self.refractory_s,
                    on_frame=self._on_frame,
                    stop_flag=self._stop.is_set,
                )
            self.hub.set_status("stopped")
        except BaseException as e:
            self.error = e
            self.hub.set_status("error", _explain_stream_error(e))
            log.exception("acquisition thread crashed")


# ---------------------------------------------------------------------------
# HTML UI (single page, dark instrument-panel look)
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>eeg_tvns live monitor</title>
<style>
  :root {
    --bg: #0a0e14;
    --panel: #10151d;
    --line: #1c2430;
    --text: #d8e0ea;
    --muted: #7a8699;
    --accent: #59d1a0;
    --danger: #ff5a5a;
    --go: #ffb454;
    --trace: #9dd6ff;
    --word: #b48cff;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
    font-family: "JetBrains Mono", "SF Mono", "Menlo", monospace; }
  header { display: flex; align-items: baseline; gap: 24px; padding: 14px 22px;
    border-bottom: 1px solid var(--line); }
  header h1 { margin: 0; font-size: 15px; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--text); font-weight: 600; }
  header .meta { color: var(--muted); font-size: 12px; }
  header .meta span { color: var(--text); margin-right: 12px; }
  main { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 14px;
    padding: 14px; height: calc(100% - 72px); }
  .leftCol { display: grid; grid-template-rows: 1.35fr 1fr; gap: 14px; min-height: 0; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
    padding: 12px; display: flex; flex-direction: column; min-height: 0; }
  .panel h2 { margin: 0 0 8px 0; font-size: 11px; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--muted); font-weight: 500;
    display: flex; justify-content: space-between; align-items: baseline; }
  .panel h2 .note { letter-spacing: 0; text-transform: none; font-size: 11px;
    color: var(--muted); }
  .panel h2 .note b { color: var(--text); font-weight: 500; }
  canvas { width: 100%; height: 100%; display: block; min-height: 0; }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 4px; }
  .stat { background: #0c1119; border: 1px solid var(--line); border-radius: 4px;
    padding: 8px 10px; }
  .stat .k { color: var(--muted); font-size: 10px; letter-spacing: 0.14em;
    text-transform: uppercase; }
  .stat .v { color: var(--text); font-size: 18px; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .go-indicator { display: flex; align-items: center; justify-content: center;
    font-size: 28px; font-weight: 700; letter-spacing: 0.2em;
    padding: 22px 0; border-radius: 6px; border: 1px solid var(--line);
    background: #0c1119; color: var(--muted); transition: background 120ms, color 120ms; }
  .go-indicator.on { background: rgba(255, 180, 84, 0.14); color: var(--go);
    border-color: rgba(255, 180, 84, 0.4); }
  .fire-flash { position: absolute; inset: 0; background: rgba(255, 90, 90, 0.22);
    opacity: 0; pointer-events: none; border-radius: 6px; transition: opacity 200ms; }
  .go-wrap { position: relative; }
  .rightCol { display: flex; flex-direction: column; gap: 14px; min-height: 0;
    overflow-y: auto; }
  .rightCol .panel { flex: 0 0 auto; }
  #pgoPanel { height: 150px; }
  #latPanel { height: 130px; }

  /* word readout */
  .word-hero { font-size: 30px; font-weight: 700; letter-spacing: 0.16em;
    text-align: center; padding: 10px 0 12px; color: var(--word);
    text-transform: uppercase; }
  .word-hero.idle { color: var(--muted); font-size: 15px; letter-spacing: 0.12em; }
  .bars { display: flex; flex-direction: column; gap: 7px; }
  .bar { display: grid; grid-template-columns: 46px 1fr 44px; align-items: center;
    gap: 8px; font-size: 11px; }
  .bar .lbl { color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
  .bar .track { height: 8px; background: #0c1119; border: 1px solid var(--line);
    border-radius: 4px; overflow: hidden; }
  .bar .fill { height: 100%; width: 0%; background: var(--word); opacity: 0.45;
    transition: width 120ms linear; }
  .bar.top .fill { opacity: 1; }
  .bar.top .lbl { color: var(--text); }
  .bar .pct { text-align: right; color: var(--text); font-variant-numeric: tabular-nums; }
  .disclaimer { margin-top: 10px; font-size: 10px; line-height: 1.5; color: var(--muted); }

  footer { padding: 6px 22px; color: var(--muted); font-size: 11px;
    border-top: 1px solid var(--line); display: flex; justify-content: space-between; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--muted); margin-right: 6px; vertical-align: middle; }
  .dot.live { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
  #banner { display: none; padding: 9px 22px; font-size: 12px;
    background: rgba(255, 90, 90, 0.14); color: #ffb0b0;
    border-bottom: 1px solid rgba(255, 90, 90, 0.35); }
  #banner.show { display: block; }
  #banner b { color: #fff; }
  /* Tall enough that the head circle is limited by the column width, not by
     height -- at 208px the markers for close pairs (FT7/FC5) collided. */
  #head { width: 100%; height: 296px; display: block; }
  .legend { display: flex; gap: 12px; font-size: 10px; color: var(--muted);
    margin-top: 4px; justify-content: center; }
  .legend i.sw { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 4px; vertical-align: middle; }
  .sw.good { background: #35d07f; } .sw.ok { background: #e8c34a; }
  .sw.bad { background: #ff5a5a; } .sw.unknown { background: #3a4452; }
  #headTip { min-height: 28px; }
  #headTip b { color: var(--text); }
</style>
</head>
<body>
<header>
  <h1>eeg_tvns live monitor</h1>
  <div class="meta">
    <span id="source">-</span>
    <span id="model">-</span>
    <span id="wordModel">-</span>
    <span id="channels">-</span>
    <span><span class="dot" id="connDot"></span><span id="conn">connecting</span></span>
  </div>
</header>
<div id="banner"></div>
<main>
  <div class="leftCol">
    <section id="eegPanel" class="panel">
      <h2>Live EEG traces <span class="note">rolling <b id="bufLen">6</b> s &middot; red line = tVNS fire</span></h2>
      <canvas id="eeg"></canvas>
    </section>
    <section id="trigPanel" class="panel">
      <h2>
        Decoded window that fired tVNS
        <span class="note" id="trigMeta">awaiting first fire</span>
      </h2>
      <canvas id="trig"></canvas>
    </section>
  </div>
  <div class="rightCol">
    <section class="panel go-wrap">
      <h2>GO decoder <span class="note">gates stimulation</span></h2>
      <div id="goInd" class="go-indicator">IDLE</div>
      <div id="fireFlash" class="fire-flash"></div>
      <div class="stats">
        <div class="stat"><div class="k">p(go)</div><div class="v" id="pgo">0.000</div></div>
        <div class="stat"><div class="k">Threshold</div><div class="v" id="thresh">0.50</div></div>
        <div class="stat"><div class="k">Fires</div><div class="v" id="fires">0</div></div>
        <div class="stat"><div class="k">Decisions</div><div class="v" id="decs">0</div></div>
      </div>
    </section>
    <section class="panel">
      <h2>Word decode <span class="note">display only</span></h2>
      <div id="wordHero" class="word-hero idle">no word model</div>
      <div class="bars" id="wordBars"></div>
      <div class="disclaimer" id="wordNote">
        4-class attempt identity, decoded only while GO is active.
        Never gates stimulation.
      </div>
    </section>
    <section id="headPanel" class="panel">
      <h2>Electrode contact <span class="note" id="headMeta">live</span></h2>
      <canvas id="head"></canvas>
      <div class="legend">
        <span><i class="sw good"></i>good</span>
        <span><i class="sw ok"></i>marginal</span>
        <span><i class="sw bad"></i>bad</span>
        <span><i class="sw unknown"></i>no data</span>
      </div>
      <div id="headTip" class="disclaimer">
        Hover an electrode for its measured values.
      </div>
      <div id="impNote" class="disclaimer"></div>
    </section>
    <section id="pgoPanel" class="panel">
      <h2>p(go) history</h2>
      <canvas id="pgo_chart"></canvas>
    </section>
    <section id="latPanel" class="panel">
      <h2>Latency vs 300 ms budget</h2>
      <canvas id="lat_chart"></canvas>
    </section>
  </div>
</main>
<footer>
  <div id="lat">latency last / median / p95: -</div>
  <div>hop-driven UI; decoder runs at its native rate</div>
</footer>
<script>
const $ = (id) => document.getElementById(id);
const state = { data: null, ws: null, lastFireT: -1, headHits: [] };

function fitCanvas(c) {
  const r = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.floor(r.width * dpr) || c.height !== Math.floor(r.height * dpr)) {
    c.width = Math.floor(r.width * dpr);
    c.height = Math.floor(r.height * dpr);
  }
  return dpr;
}

// Stacked multi-channel trace renderer, shared by the live strip chart and the
// frozen trigger window. Each channel is auto-scaled to its own row.
function drawTraces(canvasId, eeg, chNames, opts) {
  opts = opts || {};
  const c = $(canvasId); const ctx = c.getContext('2d'); const dpr = fitCanvas(c);
  ctx.clearRect(0, 0, c.width, c.height);
  if (!eeg || !eeg.length) return;
  const nCh = eeg.length, nSamp = eeg[0].length;
  if (nSamp < 2) return;
  const w = c.width, h = c.height;
  const rowH = h / nCh;
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1 * dpr;
  for (let i = 1; i < nCh; i++) {
    ctx.beginPath(); ctx.moveTo(0, i * rowH); ctx.lineTo(w, i * rowH); ctx.stroke();
  }
  const stroke = opts.color || 'rgba(157,214,255,0.85)';
  ctx.font = `${11 * dpr}px "JetBrains Mono", monospace`;
  for (let ch = 0; ch < nCh; ch++) {
    const row = eeg[ch];
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < row.length; i++) { const v = row[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
    const span = Math.max(1e-6, mx - mn);
    const mid = (mn + mx) / 2;
    const yMid = (ch + 0.5) * rowH;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.1 * dpr;
    ctx.beginPath();
    for (let i = 0; i < nSamp; i++) {
      const x = (i / (nSamp - 1)) * w;
      const y = yMid - ((row[i] - mid) / span) * (rowH * 0.9);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = 'rgba(216,224,234,0.55)';
    ctx.fillText((chNames && chNames[ch]) ? chNames[ch] : `CH${ch + 1}`,
                 6 * dpr, ch * rowH + 14 * dpr);
  }
  // Vertical event markers, positioned as fractions of the x axis.
  for (const frac of (opts.markers || [])) {
    ctx.strokeStyle = 'rgba(255,90,90,0.9)';
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath(); ctx.moveTo(frac * w, 0); ctx.lineTo(frac * w, h); ctx.stroke();
  }
}

function drawEEG() {
  const d = state.data; if (!d) return;
  const tNow = d.latest.t;
  const bufS = d.buffer_s || 6;
  const markers = (d.fire_times || [])
    .map((t) => tNow - t)
    .filter((dt) => dt >= 0 && dt <= bufS)
    .map((dt) => 1 - dt / bufS);
  drawTraces('eeg', d.eeg, d.ch_names, { markers });
}

function drawTrigger() {
  const d = state.data; if (!d) return;
  const trig = d.trigger;
  if (!trig) {
    const c = $('trig'); const ctx = c.getContext('2d'); fitCanvas(c);
    ctx.clearRect(0, 0, c.width, c.height);
    return;
  }
  // The fire happens at the end of the window that triggered it.
  drawTraces('trig', trig.eeg, d.ch_names,
             { color: 'rgba(255,140,140,0.85)', markers: [1.0] });
}

function renderWord() {
  const d = state.data; if (!d) return;
  const hero = $('wordHero'), bars = $('wordBars'), note = $('wordNote');
  if (!d.word_model_path) {
    hero.className = 'word-hero idle';
    hero.textContent = 'no word model';
    bars.innerHTML = '';
    note.textContent = 'Train one with: python run.py --data ./ds003626 --task word';
    return;
  }
  const names = Object.keys(d.word_names || {})
    .sort((a, b) => Number(a) - Number(b))
    .map((k) => d.word_names[k]);
  const probs = (d.word && d.word.probs) ? d.word.probs : {};
  const hasProbs = Object.keys(probs).length > 0;

  if (hasProbs) {
    hero.className = 'word-hero';
    hero.textContent = d.word.name || '-';
  } else {
    hero.className = 'word-hero idle';
    hero.textContent = 'awaiting GO';
  }

  let topName = null, topVal = -1;
  for (const n of names) {
    const v = probs[n] || 0;
    if (v > topVal) { topVal = v; topName = n; }
  }
  bars.innerHTML = names.map((n) => {
    const v = probs[n] || 0;
    const cls = (hasProbs && n === topName) ? 'bar top' : 'bar';
    return `<div class="${cls}">
      <div class="lbl">${n}</div>
      <div class="track"><div class="fill" style="width:${(v * 100).toFixed(1)}%"></div></div>
      <div class="pct">${(v * 100).toFixed(0)}%</div>
    </div>`;
  }).join('');

  note.textContent = hasProbs
    ? `decoded in ${(d.word.latency_ms || 0).toFixed(2)} ms \u2014 display only, never gates tVNS`
    : 'decoded only while GO is active \u2014 display only, never gates tVNS';
}

function drawSeries(canvasId, values, opts) {
  const c = $(canvasId); const ctx = c.getContext('2d'); const dpr = fitCanvas(c);
  ctx.clearRect(0, 0, c.width, c.height);
  const w = c.width, h = c.height;
  const pad = 24 * dpr;
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath(); ctx.moveTo(pad, h - pad); ctx.lineTo(w, h - pad); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, h - pad); ctx.stroke();
  const yMin = opts.yMin, yMax = opts.yMax;
  ctx.fillStyle = 'rgba(216,224,234,0.55)';
  ctx.font = `${10 * dpr}px "JetBrains Mono", monospace`;
  ctx.fillText(String(yMax), 4 * dpr, pad + 3 * dpr);
  ctx.fillText(String(yMin), 4 * dpr, h - pad + 3 * dpr);
  if (opts.threshold !== undefined) {
    const y = h - pad - ((opts.threshold - yMin) / (yMax - yMin)) * (h - 2 * pad);
    ctx.strokeStyle = 'rgba(255,180,84,0.5)';
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w, y); ctx.stroke();
    ctx.setLineDash([]);
  }
  const n = values.length;
  if (n < 2) return;
  ctx.strokeStyle = opts.color || 'rgba(89,209,160,0.9)';
  ctx.lineWidth = 1.4 * dpr;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = pad + (i / (n - 1)) * (w - pad);
    const y = h - pad - ((values[i] - yMin) / (yMax - yMin)) * (h - 2 * pad);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

const STATUS_FILL = { good: '#35d07f', ok: '#e8c34a', bad: '#ff5a5a', unknown: '#3a4452' };

// Top-down scalp map. Positions come from the server (standard_1020, azimuthal
// projection) and colours from measured values -- nothing here is decorative.
function drawHead() {
  const d = state.data, c = $('head');
  if (!c) return;
  const dpr = fitCanvas(c), ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const scalp = d.scalp || {}, names = Object.keys(scalp);
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 16 * dpr;

  ctx.strokeStyle = '#2a3340';
  ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
  // nose (up) and ears, so left/right is unambiguous
  ctx.beginPath();
  ctx.moveTo(cx - 7 * dpr, cy - R); ctx.lineTo(cx, cy - R - 10 * dpr);
  ctx.lineTo(cx + 7 * dpr, cy - R); ctx.stroke();
  [-1, 1].forEach(s => {
    ctx.beginPath();
    ctx.ellipse(cx + s * R, cy, 4 * dpr, 11 * dpr, 0, 0, Math.PI * 2);
    ctx.stroke();
  });

  // The closest 10-20 pair on this layout (FT8/FC6) sits 0.192 head-radii apart,
  // so cap the marker at half that to guarantee they never overlap at any size.
  const r = Math.max(4 * dpr, Math.min(12.5 * dpr, R * 0.192 * 0.5));

  const byName = {};
  (d.quality || []).forEach(q => { byName[q.name] = q; });
  const imp = (d.impedance && d.impedance.channels) || {};
  state.headHits = [];

  names.forEach(n => {
    const p = scalp[n];
    // Projection is nose-up (+y); canvas y grows downward, so flip it.
    const x = cx + p[0] * R, y = cy - p[1] * R;
    const q = byName[n];
    const zk = imp[n];
    let status = q ? q.status : 'unknown';
    if (!q && zk != null) {
      status = zk <= 5 ? 'good' : (zk <= 20 ? 'ok' : 'bad');
    }
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = STATUS_FILL[status] || STATUS_FILL.unknown;
    ctx.globalAlpha = status === 'unknown' ? 0.5 : 0.85;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = 'rgba(0,0,0,0.45)'; ctx.lineWidth = 1 * dpr; ctx.stroke();
    ctx.fillStyle = status === 'unknown' ? '#8b97a8' : '#0b0e13';
    ctx.font = `600 ${Math.max(7 * dpr, r * 0.8)}px ui-monospace, monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(n, x, y);
    state.headHits.push({ name: n, x: x / dpr, y: y / dpr, r: r / dpr });
  });

  const qs = d.quality || [];
  const nBad = qs.filter(q => q.status === 'bad').length;
  const nOk = qs.filter(q => q.status === 'ok').length;
  $('headMeta').innerHTML = !qs.length ? 'awaiting data'
    : nBad ? `<b>${nBad}</b> bad` + (nOk ? `, ${nOk} marginal` : '')
    : nOk ? `<b>${nOk}</b> marginal, rest good`
    : 'all channels good';

  const note = $('impNote');
  if (d.impedance && d.impedance.timestamp) {
    const age = (Date.now() / 1000 - d.impedance.timestamp) / 60;
    note.innerHTML = `Impedance from a check <b>${age.toFixed(0)} min</b> ago `
      + `(not live \u2014 measuring it injects current and suspends EEG).`;
  } else {
    note.textContent = 'Colour is from live amplitude and mains noise. For true '
      + 'impedance in k\u03A9, run: calibrate.py --check-signal';
  }
  if ((d.unplaced || []).length) {
    note.innerHTML += `<br>No 10-20 position for: ${d.unplaced.join(', ')} `
      + `\u2014 not shown on the map.`;
  }
}

function headTipFor(name) {
  const d = state.data;
  const q = (d.quality || []).find(x => x.name === name);
  const zk = ((d.impedance && d.impedance.channels) || {})[name];
  if (!q && zk == null) return `<b>${name}</b>: no measurement yet.`;
  const bits = [];
  if (zk != null) bits.push(`impedance <b>${zk.toFixed(1)} k\u03A9</b>`);
  if (q) {
    bits.push(`RMS <b>${q.rms_uv.toFixed(1)} \u00B5V</b>`);
    bits.push(`mains <b>${(q.line_ratio * 100).toFixed(0)}%</b> of power`);
  }
  let s = `<b>${name}</b> \u2014 ` + bits.join(' \u00B7 ');
  if (q && q.reason) s += `<br>${q.reason}`;
  return s;
}

(function bindHeadHover() {
  const c = $('head');
  if (!c) return;
  c.addEventListener('mousemove', (e) => {
    const rect = c.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const hit = (state.headHits || []).find(
      h => (mx - h.x) ** 2 + (my - h.y) ** 2 <= (h.r + 3) ** 2);
    $('headTip').innerHTML = hit
      ? headTipFor(hit.name)
      : 'Hover an electrode for its measured values.';
  });
  c.addEventListener('mouseleave', () => {
    $('headTip').innerHTML = 'Hover an electrode for its measured values.';
  });
})();

function render() {
  const d = state.data;
  if (!d) return;
  $('source').textContent = `src: ${d.source}`;
  $('model').textContent = `go: ${d.model_path.split('/').slice(-1)[0]}`;
  $('wordModel').textContent = d.word_model_path
    ? `word: ${d.word_model_path.split('/').slice(-1)[0]}` : 'word: none';
  $('channels').textContent = `${d.ch_names.length} ch @ ${d.display_sfreq.toFixed(0)} Hz disp`;
  $('bufLen').textContent = (d.buffer_s || 6).toFixed(0);
  $('pgo').textContent = d.latest.p_go.toFixed(3);
  $('thresh').textContent = (d.go_threshold ?? 0.5).toFixed(2);
  $('fires').textContent = d.latest.n_fires;
  $('decs').textContent = d.latest.n_decisions;
  const go = $('goInd');
  if (d.latest.go) { go.classList.add('on'); go.textContent = 'GO'; }
  else { go.classList.remove('on'); go.textContent = 'IDLE'; }
  const nFires = (d.fire_times || []).length;
  if (nFires > state.lastFireT) {
    state.lastFireT = nFires;
    const f = $('fireFlash'); f.style.opacity = '1';
    setTimeout(() => (f.style.opacity = '0'), 200);
  }
  const ls = d.latency_stats || {};
  $('lat').textContent = `latency last / median / p95: ${ls.last_ms?.toFixed(2)} / ${ls.median_ms?.toFixed(2)} / ${ls.p95_ms?.toFixed(2)} ms`;

  const st = d.status || {};
  const banner = $('banner');
  if (st.state === 'error') {
    banner.classList.add('show');
    banner.innerHTML = `<b>Acquisition stopped.</b> ${st.detail || ''}`;
  } else {
    banner.classList.remove('show');
  }

  const trig = d.trigger;
  $('trigMeta').innerHTML = trig
    ? `${(trig.window_s || 0).toFixed(1)} s window &middot; p(go)=<b>${trig.p_go.toFixed(3)}</b> \u2265 ${trig.threshold.toFixed(2)}`
      + (trig.word && trig.word.name ? ` &middot; word=<b>${trig.word.name}</b>` : '')
      + ` &middot; ${(d.latest.t - trig.t).toFixed(1)} s ago`
    : 'awaiting first fire';

  drawEEG();
  drawTrigger();
  renderWord();
  drawHead();
  drawSeries('pgo_chart', d.p_history || [], {
    yMin: 0, yMax: 1, threshold: d.go_threshold ?? 0.5, color: 'rgba(89,209,160,0.95)' });
}

function drawLatFromSnap() {
  const arr = window.__latSeries || [];
  drawSeries('lat_chart', arr, { yMin: 0, yMax: 30, color: 'rgba(157,214,255,0.9)' });
}

window.addEventListener('resize', () => { if (state.data) { render(); drawLatFromSnap(); } });

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;
  ws.onopen = () => { $('connDot').classList.add('live'); $('conn').textContent = 'live'; };
  ws.onclose = () => {
    $('connDot').classList.remove('live'); $('conn').textContent = 'disconnected';
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => {
    try {
      const d = JSON.parse(ev.data);
      state.data = d;
      // maintain latency history for the lat chart
      window.__latSeries = window.__latSeries || [];
      window.__latSeries.push(d.latency_stats.last_ms);
      if (window.__latSeries.length > 300) window.__latSeries.shift();
      render();
      drawLatFromSnap();
    } catch (e) { console.error(e); }
  };
}
connect();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# FastAPI server (defined lazily so import doesn't require fastapi)
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]


def build_app(hub: FrameHub, publish_hz: float = 10.0):
    if FastAPI is None:  # pragma: no cover
        raise SystemExit(
            "Dashboard requires fastapi + uvicorn. Install with:\n"
            "    pip install fastapi uvicorn websockets"
        )

    import asyncio

    app = FastAPI(title="eeg_tvns live monitor")
    period = 1.0 / max(1.0, publish_hz)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    @app.get("/snapshot")
    async def snapshot() -> dict:
        return hub.snapshot()

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        try:
            while True:
                snap = hub.snapshot()
                await sock.send_text(json.dumps(snap))
                await asyncio.sleep(period)
        except WebSocketDisconnect:
            return
        except Exception:
            log.exception("websocket error")
            try:
                await sock.close()
            except Exception:
                pass

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_word_readout(path: Optional[str], n_go_channels: Optional[int]) -> Optional[WordReadout]:
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
        w_bundle, w_cfg, w_decoder, _, w_n_ch = load_decoder(path)
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


def _make_streamer(args, cfg: Config, model_sfreq: float, channel_order):
    return (
        OpenBCIStreamer(
            args.port, cfg, model_sfreq,
            channel_order=channel_order, resample=not args.no_resample,
        ),
        f"openbci:{args.port}",
    )


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="eeg_tvns live dashboard")
    ap.add_argument("--port", required=True,
                    help="OpenBCI dongle serial port, e.g. /dev/cu.usbserial-XXXX")
    ap.add_argument("--model", default="outputs/model_go.joblib",
                    help="GO decoder -- the only model that gates tVNS")
    ap.add_argument("--word-model", default="outputs/model_word.joblib",
                    help="4-class word decoder, DISPLAY ONLY (never gates tVNS). "
                         "Pass 'none' to disable.")
    ap.add_argument("--hop", type=float, default=None)
    ap.add_argument("--refractory", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--no-resample", action="store_true")
    ap.add_argument("--channel-names", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--web-port", type=int, default=8765)
    ap.add_argument("--publish-hz", type=float, default=10.0,
                    help="UI refresh rate (decode loop rate is unchanged)")
    ap.add_argument("--line-freq", type=float, default=60.0,
                    help="mains frequency for the contact-quality metric (60 US, 50 EU)")
    ap.add_argument("--impedance-json", default="outputs/impedance.json",
                    help="impedances from 'calibrate.py --check-signal', shown on the "
                         "head map with their age. Impedance cannot be measured live.")
    args = ap.parse_args(argv)

    bundle, cfg, decoder, model_sfreq, n_model_ch = load_decoder(args.model)
    if args.threshold is not None:
        cfg.go_threshold = args.threshold
    hop = args.hop if args.hop is not None else cfg.hop_s
    board_names = [c.strip() for c in args.channel_names.split(",")] if args.channel_names else None
    channel_order = _resolve_channel_order(bundle, board_names)

    streamer, source = _make_streamer(args, cfg, model_sfreq, channel_order)

    word_readout = _load_word_readout(args.word_model, n_model_ch)

    ch_names = bundle.get("ch_names")
    if not ch_names:
        raise SystemExit(
            f"Model {args.model!r} has no channel names. Retrain it so the live "
            "board can be remapped onto the trained montage."
        )
    hub = FrameHub(
        n_channels=len(ch_names),
        ch_names=ch_names,
        display_sfreq=model_sfreq,
        source=source,
        model_path=args.model,
        word_names=word_readout.label_names if word_readout else {},
        word_model_path=args.word_model if word_readout else "",
        go_threshold=cfg.go_threshold,
    )

    if args.impedance_json and os.path.exists(args.impedance_json):
        try:
            with open(args.impedance_json) as fh:
                hub.set_impedance(json.load(fh))
            log.info("Loaded impedance check from %s", args.impedance_json)
        except Exception:
            log.warning("Could not read %s; head map will show live metrics only.",
                        args.impedance_json, exc_info=True)

    acq = AcquisitionThread(streamer, decoder, hub, hop_s=hop,
                            refractory_s=args.refractory, word_readout=word_readout,
                            line_freq=args.line_freq)
    acq.start()

    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            "Dashboard requires fastapi + uvicorn. Install with:\n"
            "    pip install fastapi uvicorn"
        ) from e

    app = build_app(hub, publish_hz=args.publish_hz)
    log.info("dashboard: http://%s:%d  (source=%s, model=%s)",
             args.host, args.web_port, source, args.model)
    try:
        uvicorn.run(app, host=args.host, port=args.web_port, log_level="info")
    finally:
        acq.stop()
        acq.join(timeout=3.0)


if __name__ == "__main__":
    main()
