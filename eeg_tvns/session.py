"""Session manager: one owner for the board, the ARM switch, and the audit log.

Why a state machine
-------------------
The serial port is exclusive. Probing it, checking contact impedance, recording
calibration, and running the closed loop all need sole control of the same board,
and two of them at once does not fail cleanly — it fails as garbage data or a
half-open port. So every hardware activity goes through one mode:

    idle -> probing | signal_check | calibrating | decoding -> idle

A request that arrives while another mode holds the board is refused (`Busy`, which
the API turns into HTTP 409) rather than queued or forced. Refusing is the honest
answer: the operator asked for something the hardware cannot do right now.

Why ARM is separate from running
--------------------------------
Starting the closed loop and firing a nerve stimulator are different decisions, so
they are different actions here. The loop always starts disarmed and decodes,
displays, and logs without triggering anything. Arming requires that the loop is
already running, that the request echoes the GO model's filename, that the server
is bound to loopback, and — when the GO model is not validated for gating — an
explicit acknowledgement. It then auto-disarms on stop, on a GO-model change, and
after `max_armed_s`.

Everything that changes mode, arms, disarms, or fires is appended to
`outputs/session_log.jsonl`, so a session can be reconstructed afterwards.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional

from . import models as model_store
from .acquisition import OpenBCIStreamer, _resolve_channel_order, fire_tvns, load_decoder
from .boards import DEFAULT_BOARD, board_choices, explain_board_error, list_serial_ports, probe_board
from .calibrate import (
    ATTEMPT,
    REST,
    ScheduledCueRun,
    build_schedule,
    check_signal,
    plan_timeline,
    run_calibration_scheduled,
    save_calibration,
)
from .config import Config
from .jobs import JobRunner
from .live import AcquisitionThread, FrameHub, load_word_readout
from .signal_quality import scalp_positions
from .training import train

log = logging.getLogger("eeg_tvns.session")

IDLE = "idle"
PROBING = "probing"
SIGNAL_CHECK = "signal_check"
CALIBRATING = "calibrating"
DECODING = "decoding"

MAX_ARMED_S = 30 * 60.0


class Busy(RuntimeError):
    """The board is in use by another activity."""


class SessionManager:
    def __init__(
        self,
        out_dir: str = "outputs",
        calib_dir: str = "calib",
        host: str = "127.0.0.1",
        port: Optional[str] = None,
        board: str = DEFAULT_BOARD,
        go_model: Optional[str] = None,
        word_model: Optional[str] = None,
        channel_names: Optional[List[str]] = None,
        line_freq: float = 60.0,
        hop_s: Optional[float] = None,
        refractory_s: float = 1.0,
        threshold: Optional[float] = None,
        resample: bool = True,
        allow_remote_arm: bool = False,
        max_armed_s: float = MAX_ARMED_S,
    ):
        self.out_dir = out_dir
        self.calib_dir = calib_dir
        self.host = host
        self.line_freq = line_freq
        self.hop_s = hop_s
        self.refractory_s = refractory_s
        self.threshold = threshold
        self.resample = resample
        self.allow_remote_arm = allow_remote_arm
        self.max_armed_s = max_armed_s

        self.port = port
        self.board = board
        self.channel_names: List[str] = list(channel_names or [])

        self._lock = threading.RLock()
        self._state = IDLE
        self._detail = ""
        self._since = time.time()

        self.hub: Optional[FrameHub] = None
        self._acq: Optional[AcquisitionThread] = None
        self._worker: Optional[threading.Thread] = None
        self._abort = threading.Event()

        self._armed = False
        self._armed_at = 0.0
        self._arm_path = ""
        self._arm_mtime = 0.0
        self._arm_timer: Optional[threading.Timer] = None
        self.n_fired = 0
        self.n_suppressed = 0
        self._last_stimulated = False

        self.slots: Dict[str, Optional[str]] = {"go": None, "word": None}
        self.jobs = JobRunner()

        self._ports = list_serial_ports()
        self._signal: Dict = {"running": False, "done": 0, "total": 0, "stage": "",
                              "results": [], "impedance": {}, "error": ""}
        self._calib: Dict = {"running": False, "trial": 0, "total": 0, "timeline": None,
                             "message": "", "error": False, "lag": {}}
        self._cue_run: Optional[ScheduledCueRun] = None

        self._audit_path = os.path.join(out_dir, "session_log.jsonl")
        self._audit_lock = threading.Lock()

        for slot, path in (("go", go_model), ("word", word_model)):
            if path and os.path.exists(path):
                try:
                    self.assign_model(slot, path, audit=False)
                except Exception as exc:
                    log.warning("Could not use %s as the %s model: %s", path, slot, exc)
        self._load_impedance_file()
        self.audit("startup", host=host, port=port, board=board)

    # ------------------------------------------------------------------ audit
    def audit(self, event: str, **fields) -> None:
        rec = {"t": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "event": event, "state": self._state, "armed": self._armed}
        rec.update(fields)
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            with self._audit_lock, open(self._audit_path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            log.warning("Could not append to the session log", exc_info=True)

    # ------------------------------------------------------------ transitions
    def _require_idle(self, what: str) -> None:
        if self._state != IDLE:
            raise Busy(f"Cannot {what}: the board is busy with "
                       f"{self._state.replace('_', ' ')}. Stop that first.")

    def _enter(self, state: str, detail: str = "") -> None:
        self._state, self._detail, self._since = state, detail, time.time()
        log.info("session -> %s%s", state, f" ({detail})" if detail else "")
        self.audit("state", to=state, detail=detail)

    def _to_idle(self, detail: str = "") -> None:
        # Both stop() and the acquisition thread's exit hook land here, whichever
        # happens first; only the transition is worth recording.
        changed = self._state != IDLE
        self._state, self._detail, self._since = IDLE, detail, time.time()
        if changed:
            self.audit("state", to=IDLE, detail=detail)

    @property
    def state_name(self) -> str:
        return self._state

    # -------------------------------------------------------------- hardware
    def rescan_ports(self) -> List[Dict]:
        self._ports = list_serial_ports()
        return self._ports

    def _resolve_target(self, port: Optional[str], board: Optional[str],
                        channel_names: Optional[List[str]]) -> None:
        """Adopt the requested port/board/montage, remembering them for next time."""
        if port:
            self.port = port
        if board:
            self.board = board
        if channel_names:
            self.channel_names = [c for c in channel_names if c]
        if not self.port:
            raise ValueError("No serial port selected. Pick one on the Hardware tab.")

    def probe(self, port: Optional[str] = None, board: Optional[str] = None) -> Dict:
        with self._lock:
            self._require_idle("probe the board")
            self._resolve_target(port, board, None)
            self._enter(PROBING, self.port or "")
        try:
            info = probe_board(self.port, self.board)
            self.audit("probe", ok=True, **info)
            return info
        except Exception as exc:
            self.audit("probe", ok=False, error=str(exc))
            raise RuntimeError(explain_board_error(exc)) from exc
        finally:
            with self._lock:
                self._to_idle()

    # ------------------------------------------------------------ decode loop
    def start_decode(self, port: Optional[str] = None, board: Optional[str] = None,
                     channel_names: Optional[List[str]] = None) -> Dict:
        with self._lock:
            self._require_idle("start the closed loop")
            self._resolve_target(port, board, channel_names)
            go_path = self.slots.get("go")
            if not go_path:
                raise ValueError(
                    "No GO decoder is assigned. Assign one on the Models tab, or "
                    "train one on the Train tab. The GO decoder is what gates tVNS.")

            bundle, cfg, decoder, model_sfreq, n_model_ch = load_decoder(go_path)
            if self.threshold is not None:
                cfg.go_threshold = self.threshold
            ch_names = bundle.get("ch_names")
            if not ch_names:
                raise ValueError(
                    f"{os.path.basename(go_path)} has no channel names, so the "
                    "board cannot be remapped onto the trained montage. Retrain it.")
            if self.channel_names and len(self.channel_names) != len(ch_names):
                log.warning("%d board labels given but the model has %d channels; "
                            "only matching names will be used.",
                            len(self.channel_names), len(ch_names))
            channel_order = _resolve_channel_order(bundle, self.channel_names or None)

            word_readout = load_word_readout(self.slots.get("word"), n_model_ch)
            hop = self.hop_s if self.hop_s is not None else cfg.hop_s
            streamer = OpenBCIStreamer(
                self.port, cfg, model_sfreq, channel_order=channel_order,
                resample=self.resample, board=self.board,
            )
            hub = FrameHub(
                n_channels=len(ch_names), ch_names=list(ch_names),
                display_sfreq=model_sfreq, source=f"openbci:{self.port}",
                model_path=go_path,
                word_names=word_readout.label_names if word_readout else {},
                word_model_path=self.slots.get("word") or "" if word_readout else "",
                go_threshold=cfg.go_threshold,
            )
            hub.set_impedance(self._signal.get("impedance") or {})
            self.hub = hub
            self._acq = AcquisitionThread(
                streamer, decoder, hub, hop_s=hop, refractory_s=self.refractory_s,
                word_readout=word_readout, line_freq=self.line_freq,
                fire=self._gated_fire, stimulated=self._did_stimulate,
                on_exit=self._on_acq_exit,
            )
            self.n_fired = self.n_suppressed = 0
            self._enter(DECODING, f"{os.path.basename(go_path)} @ {self.port}")
            self._acq.start()
        return {"state": self._state, "model": go_path,
                "channels": len(self.hub.ch_names) if self.hub else 0}

    def _on_acq_exit(self, err: Optional[BaseException]) -> None:
        """Return to idle when the loop ends, however it ended."""
        with self._lock:
            self.disarm("acquisition ended")
            if self._state == DECODING:
                self._to_idle(explain_board_error(err) if err else "")
        if err is not None:
            self.audit("decode_failed", error=str(err))

    def stop(self) -> Dict:
        """Stop whatever is holding the board and return to idle."""
        with self._lock:
            self.disarm("session stopped")
            self._abort.set()
            acq, worker = self._acq, self._worker
            was = self._state
        if acq is not None:
            acq.stop()
            acq.join(timeout=4.0)
        if worker is not None:
            worker.join(timeout=6.0)
        with self._lock:
            self._acq = None
            self._worker = None
            self._abort.clear()
            self._signal["running"] = False
            self._calib["running"] = False
            self._calib["timeline"] = None
            self._cue_run = None
            self._to_idle()
        log.info("stopped (was %s)", was)
        return {"state": IDLE, "was": was}

    # ------------------------------------------------------------------- ARM
    def _is_loopback(self) -> bool:
        try:
            return ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            return self.host in ("localhost", "")

    def arm_blockers(self) -> List[str]:
        """Everything currently standing between the operator and a live trigger."""
        out = []
        if self._state != DECODING:
            out.append("Start the closed loop first; arming only means anything "
                       "while the GO decoder is running.")
        if not self._is_loopback():
            out.append(f"The server is bound to {self.host}, not loopback. "
                       "Stimulation cannot be armed over a network without "
                       "--allow-remote-arm.")
        if not self.slots.get("go"):
            out.append("No GO decoder is assigned.")
        return out

    def go_verdict(self) -> Dict:
        path = self.slots.get("go")
        if not path:
            return {}
        return model_store.inspect_model(path).verdict.to_dict()

    def arm(self, confirm: str = "", acknowledge_unvalidated: bool = False) -> Dict:
        with self._lock:
            blockers = self.arm_blockers()
            if self.allow_remote_arm:
                blockers = [b for b in blockers if "loopback" not in b]
            if blockers:
                raise PermissionError(" ".join(blockers))

            path = self.slots["go"]
            expect = os.path.basename(path)
            if (confirm or "").strip() != expect:
                raise PermissionError(
                    f"To arm, confirm the GO model's filename exactly: {expect}")

            verdict = self.go_verdict()
            if verdict.get("level") != "ok" and not acknowledge_unvalidated:
                raise PermissionError(
                    f"{expect} is {verdict.get('label', 'unverified')}. "
                    f"{verdict.get('detail', '')} Arming it requires an explicit "
                    "acknowledgement.")

            self._armed = True
            self._armed_at = time.time()
            self._arm_path = path
            self._arm_mtime = os.path.getmtime(path)
            self._start_arm_timer()
            log.warning("ARMED: tVNS will fire on GO using %s", expect)
            self.audit("arm", model=path, verdict=verdict,
                       acknowledged=bool(acknowledge_unvalidated))
        return self.arm_state()

    def disarm(self, reason: str = "operator") -> Dict:
        with self._lock:
            was = self._armed
            self._armed = False
            if self._arm_timer is not None:
                self._arm_timer.cancel()
                self._arm_timer = None
            if was:
                log.warning("DISARMED (%s)", reason)
                self.audit("disarm", reason=reason, fired=self.n_fired,
                           suppressed=self.n_suppressed)
        return self.arm_state()

    def _start_arm_timer(self) -> None:
        if self._arm_timer is not None:
            self._arm_timer.cancel()
        self._arm_timer = threading.Timer(
            self.max_armed_s, lambda: self.disarm("armed-duration limit reached"))
        self._arm_timer.daemon = True
        self._arm_timer.start()

    def _arm_still_valid(self) -> bool:
        """A GO model swapped or retrained under an armed session must not fire."""
        if not self._armed:
            return False
        if self.slots.get("go") != self._arm_path:
            self.disarm("GO model changed")
            return False
        try:
            if os.path.getmtime(self._arm_path) != self._arm_mtime:
                self.disarm("GO model file was rewritten")
                return False
        except OSError:
            self.disarm("GO model file disappeared")
            return False
        return True

    def _gated_fire(self) -> None:
        """The only path to the stimulator. Runs on the decision path: keep cheap.

        `refractory_s` bounds this to about one call per second, so the audit write
        cannot accumulate into the latency budget.
        """
        if not self._arm_still_valid():
            self._last_stimulated = False
            self.n_suppressed += 1
            log.info("GO event suppressed (disarmed); %d so far", self.n_suppressed)
            return
        fire_tvns()
        self._last_stimulated = True
        self.n_fired += 1
        self.audit("fire", model=os.path.basename(self._arm_path), n=self.n_fired)

    def _did_stimulate(self) -> bool:
        """Whether the last `_gated_fire` reached the device.

        The display reads this so a suppressed GO event is shown as suppressed. A
        monitor that counted blocked crossings as stimulations would be claiming
        something happened to the patient that did not.
        """
        return self._last_stimulated

    def arm_state(self) -> Dict:
        blockers = self.arm_blockers()
        remaining = None
        if self._armed:
            remaining = max(0.0, self.max_armed_s - (time.time() - self._armed_at))
        return {
            "armed": self._armed,
            "can_arm": not blockers,
            "reason": " ".join(blockers),
            "expires_in_s": remaining,
            "n_fired": self.n_fired,
            "n_suppressed": self.n_suppressed,
            "loopback": self._is_loopback(),
        }

    # ---------------------------------------------------------- signal check
    def start_signal_check(self, port: Optional[str] = None, board: Optional[str] = None,
                           channel_names: Optional[List[str]] = None,
                           impedance: bool = True, impedance_input: str = "n",
                           line_freq: Optional[float] = None,
                           seconds: float = 4.0) -> Dict:
        with self._lock:
            self._require_idle("run a contact check")
            self._resolve_target(port, board, channel_names)
            if not self.channel_names:
                raise ValueError(
                    "Enter one channel label per board channel first, so each "
                    "measurement is attributed to a known electrode.")
            if line_freq:
                self.line_freq = line_freq
            names = list(self.channel_names)
            self._signal.update({"running": True, "done": 0,
                                 "total": (len(names) + 1) if impedance else 1,
                                 "stage": "opening the board", "results": [],
                                 "error": ""})
            self._abort.clear()
            self._enter(SIGNAL_CHECK, self.port or "")
            self._worker = threading.Thread(
                target=self._signal_worker, daemon=True, name="eeg-tvns-signal",
                args=(names, impedance, impedance_input, self.line_freq, seconds))
            self._worker.start()
        return {"state": self._state, "total": self._signal["total"]}

    def _signal_worker(self, names: List[str], impedance: bool, side: str,
                       line_freq: float, seconds: float) -> None:
        def on_progress(done: int, total: int, stage: str) -> None:
            self._signal.update({"done": done, "total": total, "stage": stage})

        try:
            qs = check_signal(self.port, names, seconds=seconds, line_freq=line_freq,
                              impedance=impedance, impedance_input=side,
                              board=self.board, on_progress=on_progress)
            results = [q.to_dict() for q in qs]
            measured = {q.name: q.impedance_kohm for q in qs
                        if q.impedance_kohm is not None}
            with self._lock:
                self._signal["results"] = results
                if measured:
                    imp = {"timestamp": time.time(), "channels": measured,
                           "input_side": side}
                    self._signal["impedance"] = imp
                    self._write_impedance_file(imp)
                    if self.hub is not None:
                        self.hub.set_impedance(imp)
            n_bad = sum(1 for q in results if q["status"] == "bad")
            self.audit("signal_check", channels=len(results), bad=n_bad,
                       impedance=bool(measured))
        except Exception as exc:
            log.exception("contact check failed")
            self._signal["error"] = explain_board_error(exc)
            self.audit("signal_check_failed", error=str(exc))
        finally:
            with self._lock:
                self._signal["running"] = False
                self._signal["stage"] = ""
                if self._state == SIGNAL_CHECK:
                    self._to_idle()

    def _impedance_file(self) -> str:
        return os.path.join(self.out_dir, "impedance.json")

    def _write_impedance_file(self, imp: Dict) -> None:
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            with open(self._impedance_file(), "w") as fh:
                json.dump(imp, fh, indent=2)
        except Exception:
            log.warning("Could not write %s", self._impedance_file(), exc_info=True)

    def _load_impedance_file(self) -> None:
        path = self._impedance_file()
        if not os.path.exists(path):
            return
        try:
            with open(path) as fh:
                self._signal["impedance"] = json.load(fh)
            log.info("Loaded impedance from a previous check: %s", path)
        except Exception:
            log.warning("Could not read %s", path, exc_info=True)

    # ----------------------------------------------------------- calibration
    def start_calibration(self, subject: int, session: int = 1,
                          port: Optional[str] = None, board: Optional[str] = None,
                          channel_names: Optional[List[str]] = None,
                          trials_per_class: int = 40, action_s: float = 2.5,
                          seed: int = 0, lead_in_s: float = 2.0,
                          settle_s: float = 3.0) -> Dict:
        """Plan a run, open the board, and hand the browser the cue timeline."""
        with self._lock:
            self._require_idle("record calibration")
            self._resolve_target(port, board, channel_names)
            if not self.channel_names:
                raise ValueError(
                    "Enter one channel label per board channel first: a recording "
                    "without an explicit montage cannot be mapped to electrodes.")

            schedule = build_schedule(n_per_class=trials_per_class,
                                      action_s=action_s, seed=seed)
            # lead_in gives the browser time to receive the timeline and schedule
            # its first cue; settle covers amplifier settling before trial one.
            timeline = plan_timeline(schedule, start_at=time.time() + lead_in_s,
                                    settle_s=settle_s, seed=seed)
            self._cue_run = ScheduledCueRun(schedule, timeline)
            self._calib.update({
                "running": True, "trial": 0, "total": len(schedule.trials),
                "timeline": timeline.to_dict(), "message": "", "error": False,
                "lag": {}, "subject": subject, "session": session,
            })
            self._abort.clear()
            self._enter(CALIBRATING, f"sub-{subject:02d} ses-{session:02d}")
            self._worker = threading.Thread(
                target=self._calib_worker, daemon=True, name="eeg-tvns-calib",
                args=(schedule, timeline, subject, session, list(self.channel_names)))
            self._worker.start()
        return {"state": self._state, "timeline": timeline.to_dict(),
                "server_time": time.time()}

    def _calib_worker(self, schedule, timeline, subject: int, session: int,
                      names: List[str]) -> None:
        def on_trial(i: int) -> None:
            self._calib["trial"] = i
            if self._cue_run is not None:
                self._calib["lag"] = self._cue_run.lag_stats()

        try:
            rec = run_calibration_scheduled(
                self.port, schedule, names, timeline, board=self.board,
                on_trial=on_trial, should_stop=self._abort.is_set)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(
                self.calib_dir,
                f"sub-{subject:02d}_ses-{session:02d}_{stamp}_calib.npz")
            save_calibration(
                path, rec["X"], rec["y_go"], rec["y_word"], rec["ch_names"],
                rec["sfreq"], subject=subject, session=session,
                action_s=rec["action_s"], paradigm="overt", cue_source="browser",
                display_lag_s=rec.get("display_lag_s"),
                lag_flagged=rec.get("lag_flagged"))
            n_att = int((rec["y_go"] == ATTEMPT).sum())
            n_rest = int((rec["y_go"] == REST).sum())
            lag = self._cue_run.lag_stats() if self._cue_run else {}
            self._calib.update({
                "message": f"Saved {os.path.basename(path)} — {n_att} attempt / "
                           f"{n_rest} rest at {rec['sfreq']:.0f} Hz. Train a GO "
                           f"decoder from it on the Train tab.",
                "error": False, "lag": lag})
            self.audit("calibration_saved", path=path, attempt=n_att, rest=n_rest,
                       sfreq=rec["sfreq"], lag=lag)
        except KeyboardInterrupt:
            self._calib.update({"message": "Aborted; nothing was saved.",
                                "error": True})
            self.audit("calibration_aborted")
        except Exception as exc:
            log.exception("calibration failed")
            self._calib.update({"message": explain_board_error(exc), "error": True})
            self.audit("calibration_failed", error=str(exc))
        finally:
            with self._lock:
                self._calib["running"] = False
                self._calib["timeline"] = None
                if self._state == CALIBRATING:
                    self._to_idle()

    def report_paints(self, paints: List[Dict]) -> Dict:
        """Accept the browser's cue paint times for the active run."""
        run = self._cue_run
        if run is None:
            raise Busy("No calibration run is active.")
        for p in paints or []:
            try:
                run.report_paint(int(p["index"]), float(p["shown_at"]))
            except (KeyError, TypeError, ValueError):
                log.warning("ignoring malformed paint report: %r", p)
        stats = run.lag_stats()
        self._calib["lag"] = stats
        return stats

    # ---------------------------------------------------------------- models
    def assign_model(self, slot: str, path: str, audit: bool = True) -> Dict:
        if slot not in ("go", "word"):
            raise ValueError("Slot must be 'go' or 'word'.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist.")
        info = model_store.inspect_model(path)
        if info.error:
            raise ValueError(f"{info.name} is not a usable bundle: {info.error}")
        # Invariant C, enforced at the boundary rather than by convention: a
        # 4-class word decoder can never be the thing that gates stimulation.
        if slot == "go" and info.task == "word":
            raise ValueError(
                f"{info.name} is a word decoder and cannot gate stimulation. "
                "Multi-class attempt identity is unreliable in patients, so the "
                "GO slot only accepts a binary GO decoder.")
        if slot == "word" and info.task == "go":
            raise ValueError(f"{info.name} is a GO decoder, not a 4-class word "
                             "decoder; it would have nothing to display.")
        with self._lock:
            self.slots[slot] = path
            if slot == "go" and self._armed:
                self.disarm("GO model reassigned")
            if not self.channel_names and info.ch_names:
                # A visible, editable starting point for the montage -- not a
                # silent identity mapping: the field is shown in the UI and
                # _resolve_channel_order still warns on any mismatch.
                self.channel_names = list(info.ch_names)
                log.info("Pre-filled the channel list from %s; check it matches "
                         "how your electrodes are actually plugged in.", info.name)
        if audit:
            self.audit("assign_model", slot=slot, path=path,
                       verdict=info.verdict.to_dict())
        if slot == "word" and self._state == DECODING:
            log.warning("Word model assigned while decoding; it takes effect on the "
                        "next start (the running loop keeps its current readout).")
        return info.to_dict()

    def import_model(self, raw: bytes, name: str) -> Dict:
        info = model_store.import_model(raw, name, self.out_dir)
        self.audit("import_model", path=info.path, task=info.task)
        note = ""
        if info.task == "go" and info.verdict.level != "ok":
            note = f"Note: {info.verdict.label}."
        return {**info.to_dict(), "note": note}

    def model_list(self) -> List[Dict]:
        # Also scan wherever the assigned models live, so a model handed in on the
        # command line from another directory still appears in the table (and can
        # be seen to be assigned) instead of looking like it does not exist.
        dirs = [self.out_dir]
        for path in self.slots.values():
            if path:
                d = os.path.dirname(os.path.abspath(path))
                if d not in dirs:
                    dirs.append(d)
        return [m.to_dict() for m in model_store.discover_models(dirs)]

    # -------------------------------------------------------------- training
    def start_training(self, source: str, path: str, task: str = "go",
                       condition: str = "inner", classifier: str = "lda",
                       permutations: int = 50, align: bool = True,
                       all_channels: bool = False, seed: int = 42) -> Dict:
        with self._lock:
            if self._state == DECODING:
                raise Busy(
                    "Training is refused while the closed loop is running: a "
                    "multi-minute fit would compete for CPU with the decoder that "
                    "gates stimulation. Stop the loop first.")
            if not path:
                raise ValueError("Give a path or glob for the training data.")
            if source == "calibration":
                cfg = Config(calibration_glob=path)
            elif source == "data":
                cfg = Config(data_root=path)
            else:
                raise ValueError("Source must be 'calibration' or 'data'.")
            cfg.task = task
            cfg.condition = condition
            cfg.classifier = classifier
            cfg.n_permutations = max(0, int(permutations))
            cfg.align = bool(align)
            cfg.use_low_density = not all_channels
            cfg.out_dir = self.out_dir
            cfg.random_state = int(seed)
            cfg.validate()

        def run(progress, should_stop):
            return train(cfg, progress=progress, should_stop=should_stop).to_dict()

        job = self.jobs.submit(f"train {task} from {source}", run, kind="train")
        self.audit("train_start", source=source, path=path, task=task,
                   permutations=cfg.n_permutations, align=cfg.align)
        return {"state": job.state, "name": job.name}

    def cancel_training(self) -> Dict:
        self.jobs.cancel()
        self.audit("train_cancel")
        return {"state": "cancelling"}

    # ------------------------------------------------------------------ state
    def _scalp_for(self, names: List[str]) -> Dict:
        """Cached scalp layout: the state is pushed a couple of times a second and
        building it loads an MNE montage, which is far too slow to redo each time."""
        key = tuple(names)
        if getattr(self, "_scalp_key", None) != key:
            self._scalp_key = key
            self._scalp_cache = scalp_positions(names) if names else {}
        return self._scalp_cache

    def control_state(self) -> Dict:
        """Everything the UI needs that is not a live EEG frame."""
        if self._armed:
            self._arm_still_valid()
        scalp = self._scalp_for(self.channel_names)
        slots = {}
        for slot, path in self.slots.items():
            if not path:
                slots[slot] = None
                continue
            info = model_store.inspect_model(path)
            slots[slot] = {"path": path, "name": info.name, "task": info.task,
                           "n_channels": info.n_channels, "sfreq": info.sfreq,
                           "verdict": info.verdict.to_dict()}
        return {
            "session": {"state": self._state, "detail": self._detail,
                        "port": self.port, "board": self.board,
                        "since": self._since,
                        "error": bool(self._state == IDLE and self._detail)},
            "arm": self.arm_state(),
            "slots": slots,
            "models": self.model_list(),
            "ports": self._ports,
            "boards": board_choices(),
            "channel_names": list(self.channel_names),
            "scalp": scalp,
            "unplaced": [c for c in self.channel_names if c not in scalp],
            "signal_check": self._signal,
            "calibration": {**self._calib,
                            "recordings": model_store.list_recordings(
                                os.path.join(self.calib_dir, "*.npz"))},
            "training": self.jobs.to_dict(),
            "host": self.host,
            "line_freq": self.line_freq,
            "server_time": time.time(),
        }
