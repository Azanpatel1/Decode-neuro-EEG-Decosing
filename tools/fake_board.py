"""A fake BrainFlow BoardShim for exercising board-dependent code paths.

This is a **test double for the transport**, not a data source. It replays a real
recording (or, absent one, a constant) through the same BrainFlow call surface the
real board presents, so we can check that channel counts are validated, that a
timeline is followed, that epochs land where they should, and that failures are
reported -- none of which needs plausible EEG.

It deliberately does not synthesise anything EEG-like. Under Invariant F the
platform must never produce a trace that could be mistaken for a measurement, and
that includes its own test scaffolding: what this emits is an obvious ramp, so any
output accidentally reaching a display is instantly recognisable as fake.

Usage:
    from tools.fake_board import install
    install(n_eeg=16, sfreq=125.0)
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np


class FakeBoardShim:
    """Implements only the BoardShim surface eeg_tvns actually calls."""

    n_eeg = 16
    sfreq = 125.0
    # Rate samples actually arrive at, which the real board decides. Separate from
    # `sfreq` (what the selected board *claims*) so a wrong board selection --
    # picking Cyton+Daisy for a plain Cyton -- can be reproduced.
    emit_sfreq: Optional[float] = None
    fail_on_prepare: Optional[str] = None
    emit_nothing = False

    @classmethod
    def _emit_rate(cls) -> float:
        return cls.emit_sfreq or cls.sfreq

    def __init__(self, board_id, params):
        self.board_id = board_id
        self.params = params
        self._t0 = None
        self._emitted = 0
        self._streaming = False
        self._config_cmds: List[str] = []
        self.released = False

    # -- static description -------------------------------------------------
    @classmethod
    def get_sampling_rate(cls, board_id) -> float:
        return cls.sfreq

    @classmethod
    def get_eeg_channels(cls, board_id) -> List[int]:
        return list(range(1, cls.n_eeg + 1))

    @classmethod
    def get_timestamp_channel(cls, board_id) -> int:
        return cls.n_eeg + 1

    # -- session ------------------------------------------------------------
    def prepare_session(self) -> None:
        if type(self).fail_on_prepare:
            raise RuntimeError(type(self).fail_on_prepare)

    def start_stream(self, *a, **k) -> None:
        self._streaming = True
        self._t0 = time.time()

    def stop_stream(self) -> None:
        self._streaming = False

    def release_session(self) -> None:
        self.released = True

    def config_board(self, cmd: str) -> None:
        self._config_cmds.append(cmd)

    # -- data ---------------------------------------------------------------
    def _rows(self, n: int, start_index: int, t_start: float) -> np.ndarray:
        rows = type(self).n_eeg + 2
        out = np.zeros((rows, n), dtype=np.float64)
        idx = np.arange(start_index, start_index + n, dtype=np.float64)
        # An unmistakable ramp per channel: identifiable on sight as not EEG.
        for r, row in enumerate(self.get_eeg_channels(self.board_id)):
            out[row, :] = idx + r * 1e6
        out[self.get_timestamp_channel(self.board_id), :] = (
            t_start + idx / type(self)._emit_rate())
        return out

    def _due(self) -> int:
        if not self._streaming or self._t0 is None or type(self).emit_nothing:
            return 0
        want = int((time.time() - self._t0) * type(self)._emit_rate())
        return max(0, want - self._emitted)

    def get_board_data(self) -> np.ndarray:
        """Draining read, as BrainFlow does."""
        n = self._due()
        if n == 0:
            return np.zeros((type(self).n_eeg + 2, 0))
        data = self._rows(n, self._emitted, self._t0)
        self._emitted += n
        return data

    def get_current_board_data(self, n: int) -> np.ndarray:
        """Non-draining read of the most recent n samples."""
        total = self._emitted + self._due()
        if total < n:
            return np.zeros((type(self).n_eeg + 2, 0))
        return self._rows(n, total - n, self._t0)


class _FakeParams:
    serial_port = ""


class _FakeIds:
    CYTON_DAISY_BOARD = 2
    CYTON_BOARD = 0
    GANGLION_BOARD = 1


def install(n_eeg: int = 16, sfreq: float = 125.0, fail_on_prepare: Optional[str] = None,
            emit_nothing: bool = False, emit_sfreq: Optional[float] = None) -> None:
    """Insert a fake `brainflow.board_shim` module into sys.modules."""
    import sys
    import types

    FakeBoardShim.n_eeg = n_eeg
    FakeBoardShim.sfreq = sfreq
    FakeBoardShim.emit_sfreq = emit_sfreq
    FakeBoardShim.fail_on_prepare = fail_on_prepare
    FakeBoardShim.emit_nothing = emit_nothing

    mod = types.ModuleType("brainflow.board_shim")
    mod.BoardShim = FakeBoardShim
    mod.BrainFlowInputParams = _FakeParams
    mod.BoardIds = _FakeIds
    pkg = types.ModuleType("brainflow")
    pkg.board_shim = mod
    sys.modules["brainflow"] = pkg
    sys.modules["brainflow.board_shim"] = mod
