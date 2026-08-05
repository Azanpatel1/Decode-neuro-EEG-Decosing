"""A one-at-a-time background job with progress, captured logs, and cancellation.

Training takes minutes and the dashboard has to stay responsive, so it runs on a
worker thread. Only one job exists at a time: two concurrent fits would just
contend for the same cores and make both slower, and there is no case for it here.

Log capture attaches a handler to the `eeg_tvns` logger while the job runs, so the
progress lines the pipeline already emits show up in the browser console instead of
only in the terminal that started the server.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import deque
from typing import Callable, Deque, Dict, Optional

log = logging.getLogger("eeg_tvns.jobs")

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
MAX_LOG_LINES = 400


class _Capture(logging.Handler):
    """Feeds formatted records into a job's ring buffer."""

    def __init__(self, job: "Job"):
        super().__init__(level=logging.INFO)
        self._job = job
        self.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._job.log_line(self.format(record))
        except Exception:
            pass


class Job:
    """One unit of background work. `fn(progress, should_stop)` does the work."""

    def __init__(self, name: str, fn: Callable, kind: str = "train",
                 capture_logger: str = "eeg_tvns"):
        self.name = name
        self.kind = kind
        self.state = QUEUED
        self.progress = 0.0
        self.stage = ""
        self.error: Optional[str] = None
        self.result: Optional[Dict] = None
        self.started = 0.0
        self.finished = 0.0
        self._lines: Deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._fn = fn
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._capture_logger = capture_logger
        self._thread: Optional[threading.Thread] = None

    # -- reporting used by the worker ------------------------------------
    def log_line(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)

    def report(self, frac: float, stage: str) -> None:
        with self._lock:
            self.progress = float(frac)
            self.stage = stage

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # -- control ---------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"job-{self.kind}")
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()
        self.log_line("-- cancellation requested; stopping at the next checkpoint")

    def _run(self) -> None:
        self.state = RUNNING
        self.started = time.time()
        root = logging.getLogger(self._capture_logger)
        handler = _Capture(self)
        root.addHandler(handler)
        # A handler only sees records the logger already let through, so if the host
        # process configured WARNING the job's console would come out empty while
        # looking like the run simply said nothing. The UI console is the only place
        # an operator watches a multi-minute fit, so ensure INFO reaches it. The
        # side effect is that those lines also reach the terminal for the duration.
        prior_level = root.level
        if not root.isEnabledFor(logging.INFO):
            root.setLevel(logging.INFO)
        try:
            self.result = self._fn(self.report, self.cancelled)
            self.state = CANCELLED if self.cancelled() else DONE
            self.progress = 1.0 if self.state == DONE else self.progress
        except BaseException as exc:
            # BaseException, not Exception: a worker killed by KeyboardInterrupt or
            # SystemExit would otherwise leave `state` at RUNNING forever, and
            # `submit` refuses every later job while one looks like it is running.
            # A cancellation surfaces as an exception from deep inside the
            # evaluation loop; that is expected, not a failure.
            if self.cancelled():
                self.state = CANCELLED
                self.error = None
            else:
                self.state = FAILED
                self.error = f"{type(exc).__name__}: {exc}"
                self.log_line(f"!! {self.error}")
                log.error("job %s failed", self.name, exc_info=True)
                for line in traceback.format_exc().splitlines()[-6:]:
                    self.log_line(line)
        finally:
            root.removeHandler(handler)
            root.setLevel(prior_level)
            self.finished = time.time()
            self.stage = "" if self.state == RUNNING else self.stage

    def to_dict(self) -> Dict:
        with self._lock:
            lines = list(self._lines)
        return {
            "name": self.name, "kind": self.kind, "state": self.state,
            "progress": round(self.progress, 4), "stage": self.stage,
            "error": self.error, "result": self.result, "log": lines,
            "elapsed_s": round((self.finished or time.time()) - self.started, 1)
            if self.started else 0.0,
        }


class JobRunner:
    """Holds the current job. Refuses a second one while the first runs."""

    def __init__(self) -> None:
        self._job: Optional[Job] = None
        self._lock = threading.Lock()

    @property
    def job(self) -> Optional[Job]:
        return self._job

    @property
    def busy(self) -> bool:
        j = self._job
        return j is not None and j.state in (QUEUED, RUNNING)

    def submit(self, name: str, fn: Callable, kind: str = "train") -> Job:
        with self._lock:
            if self.busy:
                raise RuntimeError(
                    f"A {self._job.kind} job is already running ({self._job.stage or 'starting'}). "
                    "Wait for it or cancel it first.")
            job = Job(name, fn, kind=kind)
            self._job = job
        job.start()
        return job

    def cancel(self) -> None:
        j = self._job
        if j is None or j.state not in (QUEUED, RUNNING):
            raise RuntimeError("Nothing is running.")
        j.cancel()

    def to_dict(self) -> Dict:
        j = self._job
        if j is None:
            return {"state": "idle", "progress": 0.0, "stage": "", "log": [],
                    "result": None, "error": None}
        return j.to_dict()
