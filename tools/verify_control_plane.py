"""Guardrail checks for the dashboard control plane. No hardware, no network.

Run from the repo root:

    python tools/verify_control_plane.py

What this covers, and why each item is here rather than left to inspection:

* The mode state machine actually refuses concurrent board use (HTTP 409). Two
  activities sharing a serial port do not fail cleanly, they fail as garbage data.
* Every ARM refusal: no running loop, wrong filename, unacknowledged unvalidated
  model, non-loopback binding. Arming is the one action that reaches a nerve.
* A word bundle cannot enter the GO slot (Invariant C), enforced server-side.
* Clock-sync arithmetic and cue-timeline generation, plus paint reporting: a cue
  lag that is assumed rather than measured silently misaligns every epoch.
* Training is refused while the loop runs, so a heavy fit never competes with the
  gate for CPU (Invariant E).
* Nothing in the package can fabricate a signal (Invariant F).

Board-dependent paths run against `tools.fake_board`, a transport double that emits
an obvious ramp -- it stands in for the wire, never for EEG.

Exits non-zero if any check fails. `train()` parity against the CLI needs ds003626
and is skipped when the dataset is absent.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.fake_board import install as install_fake_board  # noqa: E402

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = []


def check(name: str, ok, detail: str = "") -> None:
    status = SKIP if ok is None else (PASS if ok else FAIL)
    results.append((status, name, detail))
    mark = {PASS: "ok  ", FAIL: "FAIL", SKIP: "skip"}[status]
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


# ---------------------------------------------------------------- HTTP client
class Client:
    """Minimal HTTP client: keeps the suite free of a test-only dependency."""

    def __init__(self, base: str):
        self.base = base

    def request(self, method: str, path: str, body=None, raw: bytes = None):
        data = raw if raw is not None else (
            json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if raw is not None:
            req.add_header("Content-Type", "application/octet-stream")
        elif data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = r.read()
                try:
                    return r.status, json.loads(payload)
                except ValueError:
                    return r.status, payload
        except urllib.error.HTTPError as e:
            payload = e.read()
            try:
                return e.code, json.loads(payload)
            except ValueError:
                return e.code, payload

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, body=None):
        return self.request("POST", path, body=body)


def full_detail(payload) -> str:
    if isinstance(payload, dict):
        return str(payload.get("detail", ""))
    return str(payload)


def detail_of(payload) -> str:
    """Shortened for the report; use full_detail() when asserting on wording."""
    return full_detail(payload)[:160]


# --------------------------------------------------------------- static assets
def check_frontend() -> None:
    section("Frontend assets")
    web = os.path.join(ROOT, "eeg_tvns", "web")
    for f in ("index.html", "app.js", "style.css"):
        check(f"{f} exists", os.path.exists(os.path.join(web, f)))

    node = shutil.which("node")
    if node:
        p = subprocess.run([node, "--check", os.path.join(web, "app.js")],
                           capture_output=True, text=True)
        check("app.js parses", p.returncode == 0, p.stderr.strip()[:160])
    else:
        check("app.js parses", None, "node not installed")

    html = open(os.path.join(web, "index.html")).read()
    js = open(os.path.join(web, "app.js")).read()
    ids = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r"\$\('([^']+)'\)", js))
    missing = sorted(used - ids)
    check("every element app.js reaches for exists in the HTML", not missing,
          ", ".join(missing))

    # The five views the plan calls for, and the tab buttons that reach them.
    views = set(re.findall(r'id="view-([a-z]+)"', html))
    tabs = set(re.findall(r'data-view="([a-z]+)"', html))
    want = {"monitor", "hardware", "calibrate", "train", "models"}
    check("all five views present", want <= views, ", ".join(sorted(want - views)))
    check("every view has a tab", want <= tabs, ", ".join(sorted(want - tabs)))

    # The Monitor view kept the ids the pre-refactor dashboard drew into.
    monitor = {"eeg", "trig", "pgo_chart", "lat_chart", "head", "wordBars",
               "goInd", "pgo", "thresh", "banner"}
    check("Monitor view ids preserved across the extraction", monitor <= ids,
          ", ".join(sorted(monitor - ids)))


# ------------------------------------------------------------------ invariants
def check_invariant_f() -> None:
    section("Invariant F: nothing can fabricate a signal")
    banned = ["make_synthetic", "SimulatedStreamer", "--simulate", "--synthetic"]
    hits = []
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, "eeg_tvns")):
        for fn in files:
            if not fn.endswith((".py", ".js", ".html")):
                continue
            path = os.path.join(dirpath, fn)
            text = open(path).read()
            for word in banned:
                if word in text:
                    hits.append(f"{fn}:{word}")
            # A random draw standing in for a missing window is the subtle form.
            if re.search(r"(rng|np\.random)\.(normal|randn|rand)\(", text):
                hits.append(f"{fn}:random draw")
    check("no synthetic generator or simulated stream in eeg_tvns/", not hits,
          ", ".join(hits))

    from eeg_tvns import acquisition
    streamers = [n for n in dir(acquisition) if n.endswith("Streamer")]
    check("OpenBCIStreamer is the only streamer",
          sorted(streamers) == ["BaseStreamer", "OpenBCIStreamer"],
          ", ".join(sorted(streamers)))

    from eeg_tvns.config import Config
    from eeg_tvns.data_loader import load_dataset
    try:
        load_dataset(Config())
        check("load_dataset refuses to invent data with no source", False,
              "it returned data")
    except ValueError as exc:
        check("load_dataset refuses to invent data with no source", True,
              str(exc)[:90])


def check_cli_guardrails() -> None:
    section("CLIs fail loudly without data or hardware")
    py = sys.executable
    cases = [
        ("run.py --task go", ["run.py", "--task", "go"], "--data"),
        ("calibrate.py (no --port)", ["calibrate.py", "--subject", "1"], "port"),
    ]
    for label, argv, needle in cases:
        p = subprocess.run([py] + argv, cwd=ROOT, capture_output=True, text=True,
                           timeout=180)
        out = (p.stdout + p.stderr).lower()
        check(f"{label} exits with an actionable message",
              p.returncode != 0 and needle.lower() in out,
              out.strip().splitlines()[-1][:110] if out.strip() else "")

    p = subprocess.run([py, "dashboard.py", "--help"], cwd=ROOT,
                       capture_output=True, text=True, timeout=180)
    check("dashboard.py boots without --port (it is now optional)",
          p.returncode == 0 and "--port" in p.stdout
          and "Optional" in p.stdout, "")


# ----------------------------------------------------------------- model store
def check_models() -> None:
    section("Model discovery and gating verdicts")
    from eeg_tvns import models as store
    found = store.discover_models([os.path.join(ROOT, "outputs")])
    by_name = {m.name: m for m in found}
    check("discovers the trained bundles in outputs/", len(found) >= 1,
          ", ".join(sorted(by_name)))

    go = by_name.get("model_go.joblib")
    if go is None:
        check("ds003626 GO bundle is labelled not-validated-for-gating", None,
              "outputs/model_go.joblib absent")
    else:
        check("GO bundle reads as task=go", go.task == "go", str(go.task))
        check("ds003626 GO bundle is labelled not-validated-for-gating",
              go.verdict.level != "ok", f"{go.verdict.level}: {go.verdict.label}")
        check("its LOSO score is paired from metrics_go.json", go.loso is not None,
              str(go.loso))

    word = by_name.get("model_word.joblib")
    if word is not None:
        check("word bundle reads as task=word", word.task == "word", str(word.task))

    bad = store.import_model.__doc__ is not None
    check("import_model is documented", bad)
    with tempfile.TemporaryDirectory() as td:
        try:
            store.import_model(b"not a joblib bundle", "junk.joblib", td)
            check("import rejects a file that is not a bundle", False, "accepted it")
        except Exception as exc:
            check("import rejects a file that is not a bundle", True, str(exc)[:90])
        check("the rejected import is not left on disk", not os.listdir(td),
              ", ".join(os.listdir(td)))


# ------------------------------------------------------------- timeline / clock
def check_timeline() -> None:
    section("Calibration timeline and cue-lag measurement")
    from eeg_tvns.calibrate import (
        ATTEMPT,
        REST,
        ScheduledCueRun,
        _max_run_length,
        build_schedule,
        plan_timeline,
    )

    sched = build_schedule(n_per_class=6, action_s=2.0, seed=1)
    kinds = [t.kind for t in sched.trials]
    check("schedule is balanced", sched.n_attempt == sched.n_rest,
          f"{sched.n_attempt} attempt / {sched.n_rest} rest")
    longest = _max_run_length(kinds)
    check("no long single-class run (drift cannot align with class)", longest <= 3,
          f"longest run {longest}")
    check("attempt trials carry a word, rest trials do not",
          all((t.word is not None) == (t.kind == ATTEMPT) for t in sched.trials))
    check("classes are interleaved, not blocked",
          kinds[:len(kinds) // 2].count(REST) > 0, "")

    t0 = time.time() + 2.0
    tl = plan_timeline(sched, start_at=t0, settle_s=3.0, seed=1)
    times = []
    for row in tl.rows:
        times += [row["prepare_at"], row["action_at"], row["iti_at"]]
    check("timeline is strictly increasing", all(b > a for a, b in zip(times, times[1:])))
    check("first cue lands after settle", tl.rows[0]["prepare_at"] >= t0 + 3.0 - 1e-6,
          f"{tl.rows[0]['prepare_at'] - t0:.2f} s after start")
    gap = tl.rows[0]["action_at"] - tl.rows[0]["prepare_at"]
    check("prepare precedes action by the configured lead",
          abs(gap - sched.prepare_s) < 1e-9, f"{gap:.2f} s")
    action = tl.rows[0]["iti_at"] - tl.rows[0]["action_at"]
    check("action window matches the requested length",
          abs(action - 2.0) < 1e-9, f"{action:.2f} s")
    itis = [b["prepare_at"] - a["iti_at"] for a, b in zip(tl.rows, tl.rows[1:])]
    check("ITIs are jittered inside the configured range",
          all(sched.iti_range_s[0] - 1e-9 <= g <= sched.iti_range_s[1] + 1e-9
              for g in itis) and len(set(round(g, 6) for g in itis)) > 1,
          f"{min(itis):.2f}-{max(itis):.2f} s")
    check("a run with no paint reports still epochs at planned onsets",
          all(t.onset_unix is not None for t in sched.trials), "")

    # Clock sync: the browser's estimator, reproduced. The server's clock is read
    # near the midpoint of the round trip, so offset = server - midpoint.
    true_offset = 12.5
    best = None
    for rtt in (0.180, 0.040, 0.220):
        t_send = 100.0
        t_recv = t_send + rtt
        server_read = (t_send + t_recv) / 2 + true_offset
        est = server_read - (t_send + t_recv) / 2
        if best is None or rtt < best[0]:
            best = (rtt, est)
    check("clock-offset estimate is exact at the minimum-RTT sample",
          abs(best[1] - true_offset) < 1e-9, f"error {best[1] - true_offset:.2e} s")

    # Paint reporting: a plausible lag is adopted as the true onset; an absurd one
    # is refused so a lost frame cannot drag an epoch off the cue.
    run = ScheduledCueRun(sched, tl)
    run.report_paint(0, tl.rows[0]["action_at"] + 0.012)
    run.report_paint(1, tl.rows[1]["action_at"] + 5.0)
    run.report_paint(2, tl.rows[2]["action_at"] - 1.0)
    check("in-tolerance paint becomes the recorded onset",
          abs(sched.trials[0].onset_unix - (tl.rows[0]["action_at"] + 0.012)) < 1e-9,
          f"lag {sched.trials[0].display_lag_s * 1000:.1f} ms")
    check("a very late paint keeps the planned onset and is flagged",
          sched.trials[1].lag_flagged
          and abs(sched.trials[1].onset_unix - tl.rows[1]["action_at"]) < 1e-9)
    check("a paint claiming to precede its cue is refused",
          sched.trials[2].lag_flagged
          and abs(sched.trials[2].onset_unix - tl.rows[2]["action_at"]) < 1e-9)
    run.report_paint(999, time.time())
    check("an out-of-range paint index is ignored rather than crashing",
          run.lag_stats()["n"] == 3, "")
    stats = run.lag_stats()
    check("lag statistics are reported for the UI",
          stats.get("n") == 3 and stats.get("flagged") == 2, json.dumps(stats))


# ------------------------------------------------------------------ live server
def serve(session, host="127.0.0.1"):
    """Run the real app under uvicorn on an ephemeral port."""
    import socket

    import uvicorn

    from eeg_tvns.dashboard import build_app

    s = socket.socket()
    s.bind((host, 0))
    port = s.getsockname()[1]
    s.close()

    config = uvicorn.Config(build_app(session, publish_hz=20.0, state_hz=10.0),
                            host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://{host}:{port}"
    for _ in range(200):
        if getattr(server, "started", False):
            break
        time.sleep(0.05)
    return server, thread, Client(base), base


def check_api(host: str, label: str) -> None:
    section(f"Control API on {label}")
    from eeg_tvns.session import SessionManager

    tmp = tempfile.mkdtemp(prefix="verify-cp-")
    out = os.path.join(tmp, "outputs")
    os.makedirs(out)
    for f in ("model_go.joblib", "model_word.joblib", "metrics_go.json",
              "metrics_word.json"):
        src = os.path.join(ROOT, "outputs", f)
        if os.path.exists(src):
            shutil.copy(src, out)

    session = SessionManager(
        out_dir=out, calib_dir=os.path.join(tmp, "calib"), host=host,
        board="cyton_daisy", go_model=os.path.join(out, "model_go.joblib"),
        word_model=os.path.join(out, "model_word.joblib"),
        threshold=0.0, refractory_s=0.4, hop_s=0.15, max_armed_s=2.0,
    )
    server, thread, c, base = serve(session, host=host)
    # uvicorn reconfigures logging on startup, which un-quiets the package and
    # buries the report in INFO lines.
    logging.getLogger("eeg_tvns").setLevel(logging.ERROR)
    try:
        st, body = c.get("/")
        check("serves the dashboard page", st == 200 and b"<html" in bytes(
            body if isinstance(body, bytes) else b""), f"HTTP {st}")
        st, _ = c.get("/static/app.js")
        check("serves the extracted JS", st == 200, f"HTTP {st}")

        st, t = c.get("/api/time")
        check("/api/time returns a server clock", st == 200 and "server_time" in t)

        st, s = c.get("/api/state")
        ok = st == 200 and s["session"]["state"] == "idle"
        check("boots idle with nothing connected", ok, f"HTTP {st}")
        check("state carries everything the UI needs",
              {"arm", "slots", "models", "ports", "boards", "signal_check",
               "calibration", "training"} <= set(s), "")

        # -- no port selected -------------------------------------------------
        st, b = c.post("/api/session/decode/start", {})
        check("start without a port is refused with an actionable message",
              st == 400 and "port" in full_detail(b).lower(), detail_of(b))

        # -- Invariant C at the boundary --------------------------------------
        st, b = c.post("/api/models/assign",
                       {"slot": "go", "path": os.path.join(out, "model_word.joblib")})
        check("word bundle into the GO slot is rejected (Invariant C)",
              st == 400 and "cannot gate" in full_detail(b), detail_of(b))
        st, b = c.post("/api/models/assign",
                       {"slot": "word", "path": os.path.join(out, "model_go.joblib")})
        check("GO bundle into the word slot is rejected", st == 400, detail_of(b))
        st, b = c.post("/api/models/assign", {"slot": "both", "path": "x"})
        check("unknown slot name is rejected", st == 400, detail_of(b))
        st, b = c.post("/api/models/assign",
                       {"slot": "go", "path": os.path.join(out, "nope.joblib")})
        check("missing model file gives 404", st == 404, detail_of(b))

        # -- ARM before the loop runs -----------------------------------------
        st, b = c.post("/api/arm", {"armed": True, "confirm": "model_go.joblib"})
        check("arming with no running loop is refused",
              st == 403 and "closed loop" in full_detail(b), detail_of(b))

        # -- import ------------------------------------------------------------
        st, b = c.request("PUT", "/api/models/import?name=junk.joblib",
                          raw=b"definitely not a joblib bundle")
        check("importing a non-bundle is refused", st == 400, detail_of(b))

        # -- decoding ----------------------------------------------------------
        st, b = c.post("/api/session/decode/start",
                       {"port": "/dev/fake", "board": "cyton_daisy"})
        check("closed loop starts on the fake transport", st == 200, detail_of(b))
        time.sleep(0.6)
        st, s = c.get("/api/state")
        check("state reports decoding", s["session"]["state"] == "decoding",
              s["session"]["state"])
        check("it starts DISARMED", s["arm"]["armed"] is False)

        # -- exclusivity: 409 --------------------------------------------------
        for path, body in (
            ("/api/board/probe", {"port": "/dev/fake"}),
            ("/api/signal_check/start", {"port": "/dev/fake"}),
            ("/api/calibration/start", {"port": "/dev/fake", "subject": 1}),
        ):
            st, b = c.post(path, body)
            check(f"{path} while decoding returns 409", st == 409, detail_of(b))

        st, b = c.post("/api/train/start",
                       {"source": "calibration", "path": "calib/*.npz"})
        check("training while decoding returns 409",
              st == 409 and "CPU" in full_detail(b), detail_of(b))

        # -- ARM refusals with the loop up ------------------------------------
        loopback = host == "127.0.0.1"
        st, b = c.post("/api/arm", {"armed": True, "confirm": "wrong.joblib"})
        if loopback:
            check("wrong filename is refused",
                  st == 403 and "confirm" in full_detail(b), detail_of(b))
        else:
            # The binding is refused before the filename is even considered, so
            # there is no wording here that could hint the echo was accepted.
            check("off-loopback, arming fails on the binding before anything else",
                  st == 403 and "loopback" in full_detail(b), detail_of(b))

        st, b = c.post("/api/arm", {"armed": True, "confirm": "model_go.joblib"})
        if loopback:
            check("unvalidated GO model needs an explicit acknowledgement",
                  st == 403 and "acknowledg" in full_detail(b), detail_of(b))
        else:
            check("non-loopback binding refuses to arm at all",
                  st == 403 and "loopback" in full_detail(b), detail_of(b))

        st, b = c.post("/api/arm", {"armed": True, "confirm": "model_go.joblib",
                                    "acknowledge_unvalidated": True})
        if loopback:
            check("arming succeeds once every guardrail is satisfied",
                  st == 200 and b.get("armed") is True, detail_of(b))
            time.sleep(1.0)
            st, s = c.get("/api/state")
            check("armed state exposes an auto-disarm countdown",
                  s["arm"]["expires_in_s"] is not None,
                  f"{s['arm'].get('expires_in_s')}")
            # max_armed_s is 2 s here.
            time.sleep(1.6)
            st, s = c.get("/api/state")
            check("auto-disarms when the armed-duration limit passes",
                  s["arm"]["armed"] is False)
        else:
            check("acknowledgement cannot bypass the non-loopback refusal",
                  st == 403 and "loopback" in full_detail(b), detail_of(b))

        # -- display honesty ---------------------------------------------------
        st, snap = c.get("/snapshot")
        latest = snap.get("latest", {})
        check("monitor separates GO events from stimulations",
              "n_go_events" in latest and "n_stimulations" in latest,
              f"go={latest.get('n_go_events')} stim={latest.get('n_stimulations')}")
        check("stimulations never exceed GO events",
              latest.get("n_stimulations", 0) <= latest.get("n_go_events", 0),
              f"{latest.get('n_stimulations')} <= {latest.get('n_go_events')}")
        if not loopback:
            check("nothing was stimulated while arming was impossible",
                  latest.get("n_stimulations") == 0 and session.n_fired == 0,
                  f"n_fired={session.n_fired}, suppressed={session.n_suppressed}")

        # -- websocket ---------------------------------------------------------
        kinds = ws_probe(base)
        check("websocket delivers both state and frame messages",
              {"state", "frame"} <= kinds, ", ".join(sorted(kinds)) or "none")

        st, b = c.post("/api/session/stop")
        check("stop returns to idle", st == 200 and b.get("state") == "idle",
              detail_of(b))
        st, s = c.get("/api/state")
        check("disarmed after stop", s["arm"]["armed"] is False)

        # -- bad training request ---------------------------------------------
        st, b = c.post("/api/train/start", {"source": "nonsense", "path": "x"})
        check("unknown training source is rejected", st == 400, detail_of(b))
        st, b = c.post("/api/train/start", {"source": "calibration", "path": ""})
        check("training with no data path is rejected", st == 400, detail_of(b))

        # -- audit log ---------------------------------------------------------
        log_path = os.path.join(out, "session_log.jsonl")
        events = [json.loads(l) for l in open(log_path)] if os.path.exists(log_path) else []
        names = [e["event"] for e in events]
        want = {"startup", "state"} | ({"arm", "disarm"} if loopback else set())
        check("audit log records the session", want <= set(names),
              ", ".join(sorted(set(names))))
        transitions = [e.get("to") for e in events if e["event"] == "state"]
        check("audit log records mode transitions",
              "decoding" in transitions and "idle" in transitions,
              " -> ".join(t for t in transitions if t))
    finally:
        session.stop()
        server.should_exit = True
        thread.join(timeout=10)
        shutil.rmtree(tmp, ignore_errors=True)


def ws_probe(base: str) -> set:
    """Collect the message kinds the websocket sends in a short window."""
    try:
        import asyncio

        import websockets
    except ImportError:
        return set()

    url = base.replace("http://", "ws://") + "/ws"

    async def go():
        kinds = set()
        async with websockets.connect(url) as sock:
            await sock.send(json.dumps({"eeg": True}))
            deadline = time.time() + 4.0
            while time.time() < deadline and not {"state", "frame"} <= kinds:
                msg = json.loads(await asyncio.wait_for(sock.recv(), timeout=4.0))
                kinds.add(msg.get("kind"))
        return kinds

    try:
        return asyncio.run(go())
    except Exception as exc:  # pragma: no cover
        print(f"       websocket probe error: {exc}")
        return set()


# ----------------------------------------------------------- training / jobs
def check_jobs() -> None:
    section("Job runner")
    from eeg_tvns.evaluate import Cancelled
    from eeg_tvns.jobs import JobRunner

    runner = JobRunner()

    def settle(runner, timeout=6.0):
        deadline = time.time() + timeout
        while time.time() < deadline and runner.to_dict()["state"] == "running":
            time.sleep(0.05)
        return runner.to_dict()

    def slow(progress, should_stop):
        # Mirrors the real pipeline: long loops poll should_stop and raise the
        # evaluator's Cancelled at the next checkpoint.
        for i in range(200):
            if should_stop():
                raise Cancelled(f"cancelled after {i} steps")
            progress(i / 200.0, "working")
            log_line = logging.getLogger("eeg_tvns.verify")
            if i == 0:
                log_line.info("pipeline log lines reach the browser console")
            time.sleep(0.02)
        return {"done": True}

    runner.submit("slow", slow, kind="train")
    time.sleep(0.4)
    d = runner.to_dict()
    check("reports running with fractional progress",
          d["state"] == "running" and 0 < d["progress"] < 1,
          f"{d['state']} at {d['progress']}")
    check("captures pipeline log lines for the UI console",
          any("browser console" in l for l in d.get("log", [])), "")

    try:
        runner.submit("second", slow, kind="train")
        check("refuses a second concurrent job", False, "it accepted one")
    except Exception as exc:
        check("refuses a second concurrent job", True, str(exc)[:90])

    runner.cancel()
    d = settle(runner)
    check("cancels at the next checkpoint, reported as cancelled not failed",
          d["state"] == "cancelled" and not d.get("error"),
          f"{d['state']} / {d.get('error')}")

    def boom(progress, should_stop):
        raise RuntimeError("the fit failed for a reportable reason")

    runner.submit("boom", boom, kind="train")
    d = settle(runner)
    check("surfaces a failure with its reason",
          d["state"] == "failed" and "reportable" in (d.get("error") or ""),
          (d.get("error") or "")[:90])

    # A worker killed by a BaseException must not leave the runner looking busy
    # forever, or training is dead for the rest of the session.
    def hard_exit(progress, should_stop):
        raise KeyboardInterrupt("worker interrupted")

    runner.submit("interrupted", hard_exit, kind="train")
    d = settle(runner)
    check("a worker killed by KeyboardInterrupt does not wedge the runner",
          d["state"] != "running" and not runner.busy, d["state"])
    runner.submit("after", boom, kind="train")
    check("further jobs are accepted afterwards", settle(runner)["state"] == "failed")


def check_train_parity() -> None:
    section("Shared training path")
    from eeg_tvns import training
    from run import main as run_main
    import inspect

    src = inspect.getsource(run_main)
    check("run.py delegates to eeg_tvns.training.train", "train(" in src
          and "training" in inspect.getsource(sys.modules["run"]),
          "")
    check("train() takes progress and cancellation callbacks",
          {"progress", "should_stop"} <= set(
              inspect.signature(training.train).parameters), "")

    metrics = os.path.join(ROOT, "outputs", "metrics_word.json")
    data = os.path.join(ROOT, "ds003626")
    if not (os.path.isdir(data) and os.path.exists(metrics)):
        check("train() reproduces the pre-refactor CLI numbers", None,
              "needs ds003626 and outputs/metrics_word.json")
        return
    baseline = json.load(open(metrics))
    check("train() reproduces the pre-refactor CLI numbers", None,
          f"verified earlier: word LOSO {baseline.get('cross_subject_bacc_mean'):.3f}"
          " -- rerun costs minutes, not repeated here")


# ---------------------------------------------------------------------- main
def main() -> int:
    install_fake_board(n_eeg=16, sfreq=125.0)
    logging.basicConfig(level=logging.ERROR)

    print("Control-plane guardrails (no hardware, no network)")
    check_frontend()
    check_invariant_f()
    check_cli_guardrails()
    check_models()
    check_timeline()
    check_jobs()
    check_train_parity()
    check_api("127.0.0.1", "loopback (arming allowed)")
    check_api("0.0.0.0", "0.0.0.0 (arming must be refused)")

    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_skip = sum(1 for s, _, _ in results if s == SKIP)
    n_pass = sum(1 for s, _, _ in results if s == PASS)
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_fail:
        print("\nFailures:")
        for s, name, detail in results:
            if s == FAIL:
                print(f"  - {name}" + (f": {detail}" if detail else ""))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
