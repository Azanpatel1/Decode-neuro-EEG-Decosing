"""Live EEG dashboard and control plane.

    python dashboard.py                       # boots empty; connect from the UI
    python dashboard.py --port /dev/cu.usbserial-XXXX --model outputs/model_go.joblib

Then open http://127.0.0.1:8765

The server starts with nothing connected and acquires hardware on request, so the
whole workflow -- check the board, measure electrode contact, record calibration,
train a decoder, assign it, run the closed loop -- happens in the browser. `--port`
and `--model` are only a convenience that pre-fills those choices.

Everything shown is real recorded EEG from the board. There is no simulated source:
on a strip chart, generated traces look exactly like brain activity, so the platform
will not display anything it did not measure. With no board connected the monitor is
empty, and that is the correct display.

Two safety properties are structural rather than advisory:

* The word decoder cannot gate stimulation. It runs in the `on_frame` observer,
  after `run_loop` has already acted on its GO decision, and the API refuses to put
  a word bundle in the GO slot at all (Invariant C).
* tVNS fires only while explicitly ARMED, which requires a running loop, a
  loopback binding, the GO model's filename typed back, and an acknowledgement when
  that model is not validated for gating. See `eeg_tvns.session`.

The decode path itself is untouched: this only observes windows already produced by
the streamer and downsamples them for display (Invariant E).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Optional

from .boards import DEFAULT_BOARD, board_choices
from .jobs import DONE, FAILED
# Re-exported: AGENTS.md refers to `dashboard.WordReadout` when describing how
# Invariant C is upheld, and external scripts may import these from here.
from .live import (  # noqa: F401
    AcquisitionThread,
    FrameHub,
    WordReadout,
    explain_stream_error,
    load_word_readout,
)
from .session import Busy, SessionManager

log = logging.getLogger("eeg_tvns.dashboard")

# The UI lives in eeg_tvns/web/ as real files rather than an inline string: it is a
# five-view application now, which is unmaintainable as a Python literal.
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


# ---------------------------------------------------------------------------
# FastAPI server (imported at module level: a local import of FastAPI inside
# build_app produced a WebSocket handshake that Starlette rejected with 403)
# ---------------------------------------------------------------------------
try:
    from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]


def build_app(session: SessionManager, publish_hz: float = 10.0,
              state_hz: float = 2.0):
    """HTTP + WebSocket surface over a SessionManager.

    Control routes are plain `def` handlers so Starlette runs them in a worker
    thread: opening a board or probing it blocks for a second or more, and doing
    that on the event loop would stall every other client.
    """
    if FastAPI is None:  # pragma: no cover
        raise SystemExit(
            "Dashboard requires fastapi + uvicorn. Install with:\n"
            "    pip install fastapi uvicorn websockets"
        )

    app = FastAPI(title="eeg_tvns control plane")
    frame_period = 1.0 / max(1.0, publish_hz)
    state_period = 1.0 / max(0.2, state_hz)

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    # Plots and metrics written by a training run, shown on the Train tab.
    os.makedirs(session.out_dir, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=session.out_dir), name="outputs")

    def guard(fn, *args, **kwargs):
        """Map domain errors onto status codes the UI can explain to the operator."""
        try:
            return fn(*args, **kwargs)
        except Busy as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # -- pages ------------------------------------------------------------
    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    @app.get("/api/time")
    def server_time():
        """Clock reference for the browser's cue scheduling.

        Deliberately does nothing else: the browser measures round-trip time
        against this and keeps the fastest sample, so any work here would show up
        as clock error.
        """
        return {"server_time": time.time()}

    @app.get("/api/state")
    def state():
        return session.control_state()

    @app.get("/snapshot")
    def snapshot():
        hub = session.hub
        return hub.snapshot() if hub is not None else {}

    # -- hardware ---------------------------------------------------------
    @app.get("/api/ports")
    def ports():
        return {"ports": session.rescan_ports(), "boards": board_choices()}

    @app.post("/api/ports/rescan")
    def rescan():
        return {"ports": session.rescan_ports()}

    @app.post("/api/board/probe")
    def probe(body: dict = Body(default={})):
        return guard(session.probe, body.get("port"), body.get("board"))

    @app.post("/api/session/decode/start")
    def decode_start(body: dict = Body(default={})):
        return guard(session.start_decode, body.get("port"), body.get("board"),
                     body.get("channel_names"))

    @app.post("/api/session/stop")
    def session_stop():
        return guard(session.stop)

    @app.post("/api/arm")
    def arm(body: dict = Body(default={})):
        if body.get("armed"):
            return guard(session.arm, body.get("confirm", ""),
                         bool(body.get("acknowledge_unvalidated")))
        return guard(session.disarm, "operator")

    @app.post("/api/signal_check/start")
    def signal_check(body: dict = Body(default={})):
        return guard(
            session.start_signal_check,
            body.get("port"), body.get("board"), body.get("channel_names"),
            bool(body.get("impedance", True)), body.get("impedance_input", "n"),
            body.get("line_freq"),
        )

    # -- calibration ------------------------------------------------------
    @app.post("/api/calibration/start")
    def calibration_start(body: dict = Body(default={})):
        return guard(
            session.start_calibration,
            int(body.get("subject", 1)), int(body.get("session", 1)),
            body.get("port"), body.get("board"), body.get("channel_names"),
            int(body.get("trials_per_class", 40)),
            float(body.get("action_s", 2.5)), int(body.get("seed", 0)),
        )

    @app.post("/api/calibration/display")
    def calibration_display(body: dict = Body(default={})):
        return guard(session.report_paints, body.get("paints") or [])

    # -- models -----------------------------------------------------------
    @app.get("/api/models")
    def models():
        return {"models": session.model_list(), "slots": session.control_state()["slots"]}

    @app.post("/api/models/assign")
    def assign(body: dict = Body(default={})):
        return guard(session.assign_model, body.get("slot", ""), body.get("path", ""))

    @app.put("/api/models/import")
    async def import_model(request: Request, name: str = "model.joblib"):
        # Raw body rather than multipart: one binary file does not justify a
        # python-multipart dependency.
        raw = await request.body()
        return guard(session.import_model, raw, name)

    # -- training ---------------------------------------------------------
    @app.post("/api/train/start")
    def train_start(body: dict = Body(default={})):
        return guard(
            session.start_training,
            body.get("source", "calibration"), body.get("path", ""),
            body.get("task", "go"), body.get("condition", "inner"),
            body.get("classifier", "lda"), int(body.get("permutations", 50)),
            bool(body.get("align", True)),
        )

    @app.post("/api/train/cancel")
    def train_cancel():
        return guard(session.cancel_training)

    # -- streaming --------------------------------------------------------
    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        """One socket, two message kinds.

        `frame` carries the EEG buffer at `publish_hz`; `state` carries the control
        plane at a much lower rate because it changes on operator actions, not per
        decode. The client asks for frames only while a view will draw them.
        """
        await sock.accept()
        want_eeg = True

        async def read_prefs() -> None:
            nonlocal want_eeg
            try:
                while True:
                    msg = await sock.receive_text()
                    try:
                        want_eeg = bool(json.loads(msg).get("eeg", True))
                    except Exception:
                        pass
            except Exception:
                return

        reader = asyncio.create_task(read_prefs())
        next_state = 0.0
        try:
            while True:
                now = time.perf_counter()
                if now >= next_state:
                    next_state = now + state_period
                    await sock.send_text(json.dumps(
                        {"kind": "state", "data": session.control_state()},
                        default=str))
                hub = session.hub
                if hub is not None:
                    await sock.send_text(json.dumps(
                        {"kind": "frame", "data": hub.snapshot(include_eeg=want_eeg)}))
                await asyncio.sleep(frame_period)
        except WebSocketDisconnect:
            return
        except RuntimeError as exc:
            # A client that navigates away mid-send surfaces as an ASGI complaint
            # about writing to a closed socket rather than WebSocketDisconnect.
            # That is an ordinary disconnect, not something to dump a traceback for.
            if "websocket.close" in str(exc) or "response already completed" in str(exc):
                return
            log.exception("websocket error")
        except Exception:
            log.exception("websocket error")
        finally:
            reader.cancel()
            try:
                await sock.close()
            except Exception:
                pass

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="eeg_tvns dashboard + control plane")
    ap.add_argument("--port", default=None,
                    help="OpenBCI dongle serial port to pre-select, e.g. "
                         "/dev/cu.usbserial-XXXX. Optional: you can pick one in the UI.")
    ap.add_argument("--board", default=DEFAULT_BOARD,
                    choices=[b["id"] for b in board_choices()])
    ap.add_argument("--model", default="outputs/model_go.joblib",
                    help="GO decoder to pre-assign -- the only model that gates tVNS")
    ap.add_argument("--word-model", default="outputs/model_word.joblib",
                    help="4-class word decoder, DISPLAY ONLY (never gates tVNS). "
                         "Pass 'none' to skip.")
    ap.add_argument("--channel-names", default=None,
                    help="comma-separated board channel labels, in board order")
    ap.add_argument("--hop", type=float, default=None)
    ap.add_argument("--refractory", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--no-resample", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--web-port", type=int, default=8765)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--publish-hz", type=float, default=10.0,
                    help="UI refresh rate (decode loop rate is unchanged)")
    ap.add_argument("--line-freq", type=float, default=60.0,
                    help="mains frequency for the contact-quality metric (60 US, 50 EU)")
    ap.add_argument("--allow-remote-arm", action="store_true",
                    help="permit arming stimulation when not bound to loopback. Off "
                         "by default: a nerve stimulator should not be armable from "
                         "another machine by accident.")
    ap.add_argument("--start", action="store_true",
                    help="start the closed loop immediately (needs --port and a GO "
                         "model). Still starts DISARMED.")
    args = ap.parse_args(argv)

    names = ([c.strip() for c in args.channel_names.split(",") if c.strip()]
             if args.channel_names else None)
    word = None if (args.word_model or "").lower() == "none" else args.word_model
    session = SessionManager(
        out_dir=args.out_dir, calib_dir=args.calib_dir, host=args.host,
        port=args.port, board=args.board, go_model=args.model, word_model=word,
        channel_names=names, line_freq=args.line_freq, hop_s=args.hop,
        refractory_s=args.refractory, threshold=args.threshold,
        resample=not args.no_resample, allow_remote_arm=args.allow_remote_arm,
    )

    if args.start:
        try:
            session.start_decode()
        except Exception as exc:
            log.error("Could not start the closed loop: %s", exc)
            log.error("The dashboard is still up -- fix it on the Hardware tab.")

    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            "Dashboard requires fastapi + uvicorn. Install with:\n"
            "    pip install fastapi uvicorn"
        ) from e

    app = build_app(session, publish_hz=args.publish_hz)
    log.info("dashboard: http://%s:%d", args.host, args.web_port)
    if not session.slots.get("go"):
        log.warning("No GO decoder assigned. Train or import one from the browser; "
                    "the closed loop cannot start without it.")
    try:
        uvicorn.run(app, host=args.host, port=args.web_port, log_level="info")
    finally:
        session.stop()


if __name__ == "__main__":
    main()
