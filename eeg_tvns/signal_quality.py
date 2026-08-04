"""Electrode contact quality: real impedance measurement + live signal metrics.

Two different things, measured two different ways, because they cannot both be
done at once:

**Impedance (discrete check).** The ADS1299 in the Cyton injects a 6 nA AC current
at 31.2 Hz into a channel; the resulting voltage gives electrode-to-scalp
impedance. While that current is flowing the channel is *not* reading EEG, so this
is a check you run before a session, never during one. Enabled per channel with
the Cyton command ``z CHANNEL PCHAN NCHAN Z`` and converted with

    Z = (sqrt(2) * std_uV * 1e-6) / 6e-9 - 2200

where 2200 Ohm is Cyton's per-channel series resistor. Sources: OpenBCI Cyton SDK
(LeadOff Impedance Commands) and the OpenBCI GUI's DataProcessing implementation.

**Live quality (continuous).** From the ordinary EEG stream we can always compute
amplitude, mains-noise dominance, and railing/flatlining. These do not need
current injection, so they run during acquisition and are what a live display
should colour itself by.

A deliberate safeguard on the impedance path: which input the test current should
drive (P or N) depends on how the reference is wired, and driving the wrong one
yields confident but meaningless kOhm values. So we measure the 31.2 Hz band with
injection off and again with it on, and only report an impedance if the tone
actually appeared (`INJECTION_RISE_MIN`). Otherwise the channel is reported as
unmeasured rather than given a plausible-looking number.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("eeg_tvns.quality")

# --- ADS1299 / Cyton constants (see module docstring for sources) -------------
LEADOFF_CURRENT_A = 6.0e-9        # 6 nA injected test current
LEADOFF_FREQ_HZ = 31.2           # ADS1299 FLEAD_OFF = AC 31.2 Hz
SERIES_RESISTOR_OHM = 2200.0     # Cyton series resistor per channel
IMPEDANCE_BAND_HZ = (28.0, 35.0)  # narrow band around the injected tone
INJECTION_RISE_MIN = 3.0         # band power must rise >=3x to trust a reading

# --- quality thresholds ------------------------------------------------------
# Impedance: clinical gel-electrode practice targets <5 kOhm; BCI setups commonly
# accept up to ~20 kOhm. Above that, contact is poor enough to dominate the data.
IMPEDANCE_GOOD_KOHM = 5.0
IMPEDANCE_OK_KOHM = 20.0
# Amplitude: EEG sits at single-digit-to-tens of uV. Well below that means the
# electrode is not connected; far above means motion/EMG or a floating lead.
RMS_FLAT_UV = 0.5
RMS_GOOD_MAX_UV = 50.0
RMS_OK_MAX_UV = 150.0
# Cyton full scale at gain 24: +-4.5 V / 24 = +-187.5 mV.
RAILED_UV = 187_500.0 * 0.9
# Mains noise: fraction of 1-40 Hz power sitting in the line-noise band.
LINE_GOOD_RATIO = 0.2
LINE_OK_RATIO = 0.5

GOOD, OK, BAD, UNKNOWN = "good", "ok", "bad", "unknown"


@dataclass
class ChannelQuality:
    """Contact quality for one electrode. `impedance_kohm` is None if unmeasured."""

    index: int
    name: str
    rms_uv: float = 0.0
    line_ratio: float = 0.0
    railed_frac: float = 0.0
    impedance_kohm: Optional[float] = None
    impedance_note: str = ""
    peak_hz: Optional[float] = None
    # True once EEG-derived metrics have been filled in. Without it an
    # impedance-only reading would look flat (rms 0) and be scored bad.
    has_live: bool = False

    @property
    def status(self) -> str:
        """Worst of the *measured* indicators -- contact problems should not hide.

        Only indicators that were actually measured get a vote, so a channel with
        impedance but no live window is judged on impedance alone (and vice versa).
        """
        votes = []
        if self.has_live:
            votes += [self._amplitude_status(), self._line_status()]
        if self.impedance_kohm is not None:
            votes.append(self._impedance_status())
        for level in (BAD, OK, GOOD):
            if level in votes:
                return level
        return UNKNOWN

    def _impedance_status(self) -> str:
        z = self.impedance_kohm
        if z is None:
            return UNKNOWN
        if z <= IMPEDANCE_GOOD_KOHM:
            return GOOD
        return OK if z <= IMPEDANCE_OK_KOHM else BAD

    def _amplitude_status(self) -> str:
        if self.railed_frac > 0.01 or self.rms_uv < RMS_FLAT_UV:
            return BAD
        if self.rms_uv <= RMS_GOOD_MAX_UV:
            return GOOD
        return OK if self.rms_uv <= RMS_OK_MAX_UV else BAD

    def _line_status(self) -> str:
        if self.line_ratio <= LINE_GOOD_RATIO:
            return GOOD
        return OK if self.line_ratio <= LINE_OK_RATIO else BAD

    def reason(self) -> str:
        """Short human-readable explanation of a non-good status."""
        bits = []
        if not self.has_live:
            return self.impedance_note or (
                "" if self.impedance_kohm is None
                or self.impedance_kohm <= IMPEDANCE_OK_KOHM
                else f"impedance {self.impedance_kohm:.0f} kOhm - re-gel/reseat")
        if self.railed_frac > 0.01:
            # A saturated channel also has near-zero variance; report the cause,
            # not the symptom, or "railed" and "flat" contradict each other.
            bits.append(f"railed {self.railed_frac * 100:.0f}% of samples "
                        "- electrode floating or amp saturated")
        elif self.rms_uv < RMS_FLAT_UV:
            bits.append(f"flat ({self.rms_uv:.2f} uV) - electrode not connected?")
        elif self.rms_uv > RMS_OK_MAX_UV:
            bits.append(f"very high amplitude ({self.rms_uv:.0f} uV) - motion/EMG?")
        if self.line_ratio > LINE_OK_RATIO:
            bits.append(f"mains noise dominates ({self.line_ratio * 100:.0f}% of power)")
        if self.impedance_kohm is not None and self.impedance_kohm > IMPEDANCE_OK_KOHM:
            bits.append(f"impedance {self.impedance_kohm:.0f} kOhm - re-gel/reseat")
        if self.impedance_note:
            bits.append(self.impedance_note)
        return "; ".join(bits)

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "name": self.name,
            "rms_uv": round(float(self.rms_uv), 3),
            "line_ratio": round(float(self.line_ratio), 4),
            "railed_frac": round(float(self.railed_frac), 4),
            "impedance_kohm": (None if self.impedance_kohm is None
                               else round(float(self.impedance_kohm), 1)),
            "impedance_note": self.impedance_note,
            "status": self.status,
            "reason": self.reason(),
        }


def _band_power(x: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    """Mean power in [lo, hi] Hz per channel, via rFFT."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    n = x.shape[-1]
    if n < 8:
        return np.zeros(x.shape[0])
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    mag = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    sel = (freqs >= lo) & (freqs <= hi)
    if not sel.any():
        return np.zeros(x.shape[0])
    return mag[:, sel].mean(axis=-1)


def _bandpass_std_uv(x: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    """Std of each channel after a narrow band-pass, in the input's units."""
    from scipy.signal import butter, sosfiltfilt

    ny = sfreq / 2.0
    hi = min(hi, ny * 0.98)
    if lo >= hi:
        return np.zeros(x.shape[0])
    sos = butter(4, [lo / ny, hi / ny], btype="band", output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1).std(axis=-1)


def impedance_from_std_uv(std_uv: float) -> float:
    """Convert the 31.2 Hz band std (uV) to electrode impedance in Ohms.

    Z = (sqrt(2) * std_uV * 1e-6) / 6e-9 - 2200, clamped at 0. Matches the
    OpenBCI GUI computation so values are comparable with the official tool.
    """
    z = (np.sqrt(2.0) * float(std_uv) * 1e-6) / LEADOFF_CURRENT_A - SERIES_RESISTOR_OHM
    return max(0.0, float(z))


def live_quality(
    window: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    line_freq: Optional[float] = 60.0,
) -> List[ChannelQuality]:
    """Per-channel quality from an ordinary EEG window (no current injection).

    `window` is (channels, samples) in microvolts. Safe to call during streaming;
    this is what a live display should use.
    """
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2:
        raise ValueError(f"expected (channels, samples), got {window.shape}")
    if len(ch_names) != window.shape[0]:
        raise ValueError(
            f"{window.shape[0]} channels but {len(ch_names)} names given."
        )

    centred = window - window.mean(axis=1, keepdims=True)
    rms = np.sqrt((centred ** 2).mean(axis=1))
    railed = (np.abs(window) >= RAILED_UV).mean(axis=1)

    broad = _band_power(window, sfreq, 1.0, 40.0)
    if line_freq and line_freq < sfreq / 2.0:
        line = _band_power(window, sfreq, line_freq - 2.0, line_freq + 2.0)
    else:
        line = np.zeros_like(broad)
    ratio = np.divide(line, broad + line, out=np.zeros_like(broad),
                      where=(broad + line) > 0)

    return [
        ChannelQuality(index=i, name=str(ch_names[i]), rms_uv=float(rms[i]),
                       line_ratio=float(ratio[i]), railed_frac=float(railed[i]),
                       has_live=True)
        for i in range(window.shape[0])
    ]


def measure_impedance(
    board,
    eeg_rows: Sequence[int],
    sfreq: float,
    ch_names: Sequence[str],
    input_side: str = "n",
    seconds: float = 2.0,
    settle_s: float = 0.5,
) -> List[ChannelQuality]:
    """Measure per-channel impedance by injecting the ADS1299 lead-off current.

    Requires a prepared, streaming BrainFlow ``BoardShim``. Channels are measured
    one at a time: a baseline 31.2 Hz reading is taken with injection off, then
    injection is enabled for that channel only and the reading repeated. An
    impedance is reported only if the tone actually rose, so a wrong `input_side`
    surfaces as "no test signal detected" instead of a bogus value.

    `input_side` is "n" or "p" -- which ADS input the current drives. Which one is
    correct depends on how your electrodes and reference are wired; if every
    channel reports no test signal, try the other side.
    """
    side = input_side.lower()
    if side not in ("n", "p"):
        raise ValueError("input_side must be 'n' or 'p'")

    def _cmd(ch_1based: int, on: bool) -> str:
        applied = "1" if on else "0"
        p = applied if side == "p" else "0"
        n = applied if side == "n" else "0"
        return f"z{ch_1based}{p}{n}Z"

    lo, hi = IMPEDANCE_BAND_HZ
    out: List[ChannelQuality] = []

    for i, row in enumerate(eeg_rows):
        name = str(ch_names[i]) if i < len(ch_names) else f"CH{i + 1}"
        q = ChannelQuality(index=i, name=name)

        try:
            board.get_board_data()          # drain stale samples
            time.sleep(seconds / 2.0)
            base = board.get_board_data()
            base_p = (_band_power(base[[row], :], sfreq, lo, hi)[0]
                      if base is not None and base.size and base.shape[1] > 8 else 0.0)

            board.config_board(_cmd(i + 1, True))
            time.sleep(settle_s)
            board.get_board_data()          # discard the settling transient
            time.sleep(seconds)
            data = board.get_board_data()
        except Exception as exc:
            q.impedance_note = f"board error: {exc}"
            out.append(q)
            continue
        finally:
            try:
                board.config_board(_cmd(i + 1, False))
            except Exception:
                log.exception("could not disable injection on channel %d", i + 1)

        if data is None or not data.size or data.shape[1] < int(0.5 * sfreq):
            q.impedance_note = "not enough samples returned"
            out.append(q)
            continue

        chan = data[[row], :]
        inj_p = _band_power(chan, sfreq, lo, hi)[0]
        if base_p > 0 and inj_p < INJECTION_RISE_MIN * base_p:
            q.impedance_note = (
                f"no test signal detected (31.2 Hz power rose {inj_p / base_p:.1f}x, "
                f"need {INJECTION_RISE_MIN:.0f}x) - try --impedance-input "
                f"{'p' if side == 'n' else 'n'}"
            )
            out.append(q)
            continue

        std_uv = float(_bandpass_std_uv(chan, sfreq, lo, hi)[0])
        q.impedance_kohm = impedance_from_std_uv(std_uv) / 1000.0
        # Report where the tone actually landed, so a wrong assumption is visible.
        freqs = np.fft.rfftfreq(chan.shape[1], d=1.0 / sfreq)
        mag = np.abs(np.fft.rfft(chan[0] - chan[0].mean())) ** 2
        band = (freqs >= 20.0) & (freqs <= min(45.0, sfreq / 2.0 - 1))
        if band.any():
            q.peak_hz = float(freqs[band][int(np.argmax(mag[band]))])
        out.append(q)
        log.info("ch %2d %-4s impedance %.1f kOhm (tone at %.1f Hz)",
                 i + 1, name, q.impedance_kohm, q.peak_hz or float("nan"))

    return out


def merge_quality(
    live: Sequence[ChannelQuality], imp: Sequence[ChannelQuality]
) -> List[ChannelQuality]:
    """Attach impedance readings to live metrics, matched by channel index."""
    by_index = {q.index: q for q in imp}
    for q in live:
        src = by_index.get(q.index)
        if src is not None:
            q.impedance_kohm = src.impedance_kohm
            q.impedance_note = src.impedance_note
            q.peak_hz = src.peak_hz
    return list(live)


def scalp_positions(ch_names: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    """2D top-down scalp coordinates in [-1, 1] for standard 10-20 labels.

    Uses the standard_1020 montage's 3D positions with an azimuthal-equidistant
    projection (the usual EEG topographic layout: nose up, left on the left).
    Unknown labels are omitted so the caller can report them rather than draw an
    electrode in a made-up place.
    """
    try:
        import mne

        pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    except Exception:
        log.warning("standard_1020 montage unavailable; no scalp layout.", exc_info=True)
        return {}

    lut = {k.lower(): v for k, v in pos.items()}
    found = {n: lut[str(n).strip().lower()] for n in ch_names
             if str(n).strip().lower() in lut}
    if not found:
        return {}

    out: Dict[str, Tuple[float, float]] = {}
    radii = []
    for name, xyz in found.items():
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        r_xy = float(np.hypot(x, y))
        # Azimuthal equidistant: radius from the vertex grows with polar angle.
        theta = np.arctan2(r_xy, max(z, 1e-9))
        if r_xy < 1e-9:
            out[name] = (0.0, 0.0)
            radii.append(0.0)
            continue
        r = theta / (np.pi / 2.0)
        out[name] = (r * x / r_xy, r * y / r_xy)
        radii.append(r)

    # Scale so the outermost electrode sits just inside the drawn head outline.
    scale = 0.95 / max(max(radii), 1e-9) if max(radii) > 1.0 else 1.0
    return {n: (round(x * scale, 4), round(y * scale, 4)) for n, (x, y) in out.items()}
