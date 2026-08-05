"""Board registry, serial-port enumeration, and a live probe.

The dashboard has to let you choose the board rather than assume one, because
picking wrong is silently wrong: Cyton exposes 8 EEG rows and Cyton+Daisy 16, so
a mismatch shifts every channel index and the montage you carefully labelled maps
onto the wrong electrodes.

`probe_board` is a real connection test, not a guess — it opens the board, streams
briefly, and reports whether samples actually arrived. That distinguishes "dongle
present but Cyton switched off" (the most common failure) from a working board,
which no amount of port enumeration can tell you.
"""
from __future__ import annotations

import glob
import logging
import time
from typing import Dict, List, Optional

log = logging.getLogger("eeg_tvns.boards")

# id -> (BoardIds attribute name, human label, expected EEG channels)
BOARDS: Dict[str, Dict] = {
    "cyton_daisy": {"attr": "CYTON_DAISY_BOARD", "label": "OpenBCI Cyton+Daisy (16 ch, 125 Hz)",
                    "n_eeg": 16},
    "cyton": {"attr": "CYTON_BOARD", "label": "OpenBCI Cyton (8 ch, 250 Hz)", "n_eeg": 8},
    "ganglion": {"attr": "GANGLION_BOARD", "label": "OpenBCI Ganglion (4 ch, 200 Hz)",
                 "n_eeg": 4},
}
DEFAULT_BOARD = "cyton_daisy"


def board_choices() -> List[Dict]:
    return [{"id": k, "label": v["label"], "n_eeg": v["n_eeg"]} for k, v in BOARDS.items()]


def resolve_board_id(board: Optional[str]):
    """Map our board name onto a BrainFlow BoardIds value."""
    from brainflow.board_shim import BoardIds

    key = (board or DEFAULT_BOARD).strip().lower()
    if key not in BOARDS:
        raise ValueError(
            f"Unknown board '{board}'. Choose one of: {', '.join(BOARDS)}.")
    return getattr(BoardIds, BOARDS[key]["attr"])


def board_label(board: Optional[str]) -> str:
    return BOARDS.get((board or DEFAULT_BOARD).lower(), {}).get("label", str(board))


def list_serial_ports() -> List[Dict]:
    """Candidate serial ports. Uses pyserial when available, else globs.

    pyserial is optional: without it we can still list the device paths a USB
    dongle creates, just without the human-readable description.
    """
    try:
        from serial.tools import list_ports  # type: ignore

        out = [{"device": p.device, "description": (p.description or "").strip()}
               for p in list_ports.comports()]
        if out:
            out.sort(key=lambda p: ("usbserial" not in p["device"], p["device"]))
            return out
    except Exception:
        log.debug("pyserial unavailable; falling back to globbing device paths.")

    devices: List[str] = []
    for pat in ("/dev/cu.usbserial*", "/dev/tty.usbserial*", "/dev/ttyUSB*",
                "/dev/ttyACM*", "/dev/cu.usbmodem*"):
        devices.extend(glob.glob(pat))
    return [{"device": d, "description": ""} for d in sorted(set(devices))]


def explain_board_error(exc: BaseException) -> str:
    """Turn BrainFlow's error codes into something actionable.

    Kept here so every entry point (probe, signal check, calibration, decode)
    reports the same diagnosis for the same failure.
    """
    msg = str(exc)
    low = msg.lower()
    if "board_not_ready" in low or "board not ready" in low:
        return ("The dongle is there but the board did not answer. Switch the "
                "Cyton on (and the Daisy, if fitted), keep the dongle's GPIO-6 "
                "switch on the correct side, and make sure the OpenBCI GUI or "
                "another script is not already streaming.")
    if "unable_to_open_port" in low or "port is busy" in low or "unable to open port" in low:
        return ("That serial port is already in use. Quit the OpenBCI GUI or any "
                "other program holding it, then try again.")
    if "invalid_argument" in low and "serial" in low:
        return "That serial port does not exist. Rescan and pick one from the list."
    if "no module named 'brainflow'" in low:
        return ("BrainFlow is not installed in this interpreter. Install it with "
                "`pip install brainflow` into the same environment running this "
                "server.")
    if "anonymous" in low or "timeout" in low:
        return (f"{msg} — usually a powered-off board or a dongle on the wrong "
                "GPIO-6 setting.")
    return msg


def probe_board(port: str, board: str = DEFAULT_BOARD, seconds: float = 1.5) -> Dict:
    """Open the board, stream briefly, and report what actually arrived."""
    from brainflow.board_shim import BoardShim, BrainFlowInputParams

    if not port:
        raise ValueError("No serial port selected.")
    board_id = resolve_board_id(board)
    params = BrainFlowInputParams()
    params.serial_port = port
    sfreq = float(BoardShim.get_sampling_rate(board_id))
    eeg_rows = list(BoardShim.get_eeg_channels(board_id))

    shim = BoardShim(board_id, params)
    shim.prepare_session()
    try:
        shim.start_stream()
        time.sleep(max(0.3, seconds))
        data = shim.get_board_data()
    finally:
        try:
            shim.stop_stream()
        finally:
            shim.release_session()

    n = 0 if data is None or not data.size else int(data.shape[1])
    if n == 0:
        raise RuntimeError(
            "Connected to the dongle but no samples arrived. The board is "
            "probably switched off, or another program is streaming from it.")
    observed = n / max(0.3, seconds)
    note = ""
    # A sample rate far off nominal means the wrong board is selected -- Cyton
    # alone streams 250 Hz, so choosing Cyton+Daisy for it looks like a 2x rate.
    if observed < sfreq * 0.5 or observed > sfreq * 1.8:
        note = (f"Warning: observed about {observed:.0f} samples/s but this board "
                f"is nominally {sfreq:.0f} Hz. Check the board selection.")
    return {
        "board": board, "board_label": board_label(board), "port": port,
        "sfreq": sfreq, "n_eeg": len(eeg_rows), "n_samples": n,
        "observed_rate": round(observed, 1), "note": note,
    }
