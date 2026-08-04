#!/usr/bin/env python3
"""Record same-session calibration data for the GO decoder.

Cues overt speech attempts and rest, randomly interleaved in ONE recording, so
both classes share block, impedance and arousal context. This is what makes a GO
score mean "speech attempt vs rest" rather than "which recording block" -- see
the confound measurements in eeg_tvns/calibrate.py and the README.

    python calibrate.py --port /dev/cu.usbserial-XXXX --subject 1 \
        --channel-names "F7,F8,FC5,FC6,FT7,FT8,T7,T8,C3,C4,CP5,CP6,P7,P8,Cz,Pz"

Then train on it:

    python run.py --calibration "calib/*.npz" --task go

Requires real hardware (pip install brainflow). Nothing here simulates a board.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from eeg_tvns.calibrate import (
    ATTEMPT,
    REST,
    build_schedule,
    check_signal,
    run_calibration,
    save_calibration,
)

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, YELLOW, CYAN, RED = "\033[32m", "\033[33m", "\033[36m", "\033[31m"
STATUS_COLOUR = {"good": GREEN, "ok": YELLOW, "bad": RED, "unknown": DIM}
STATUS_MARK = {"good": "GOOD", "ok": "OK", "bad": "BAD", "unknown": "?"}


def print_quality(qualities) -> int:
    """Print the per-electrode quality table. Returns the count of bad channels."""
    print(f"\n{BOLD}Electrode contact quality{RESET}")
    print(f"  {DIM}impedance: <5 kOhm good, 5-20 ok, >20 bad · "
          f"amplitude and mains noise measured from live EEG{RESET}")
    print(f"\n  {'Ch':>2}  {'Site':<5} {'Impedance':>10}  {'RMS':>9}  "
          f"{'Mains':>6}  Status")
    print("  " + "-" * 66)
    n_bad = 0
    for q in qualities:
        col = STATUS_COLOUR[q.status]
        z = "—" if q.impedance_kohm is None else f"{q.impedance_kohm:.1f} kΩ"
        print(f"  {q.index + 1:>2}  {q.name:<5} {z:>10}  {q.rms_uv:>7.1f} µV  "
              f"{q.line_ratio * 100:>5.0f}%  {col}{STATUS_MARK[q.status]}{RESET}")
        if q.status == "bad":
            n_bad += 1
        if q.reason():
            print(f"      {DIM}{q.reason()}{RESET}")
    return n_bad


def present(kind: str, text: str, seconds: float) -> None:
    """Show a cue on one rewritten terminal line."""
    colour = {"action": GREEN, "prepare": YELLOW, "settle": CYAN}.get(kind, DIM)
    if kind == "action":
        line = f"{colour}{BOLD}>>> {text} <<<{RESET}"
    elif kind == "prepare":
        line = f"{colour}{text}…{RESET}"
    else:
        line = f"{colour}{text}{RESET}"
    sys.stdout.write("\r\033[K" + line)
    sys.stdout.flush()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="OpenBCI dongle serial port")
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--channel-names", required=True,
                    help="comma-separated electrode labels, one per board channel, "
                         "in board order (required so the montage is explicit)")
    ap.add_argument("--trials-per-class", type=int, default=40,
                    help="attempt trials and rest trials each (default 40)")
    ap.add_argument("--action-s", type=float, default=2.5,
                    help="length of the analysed window per trial")
    ap.add_argument("--out-dir", default="calib")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-signal", action="store_true",
                    help="report per-electrode contact quality and exit (no recording)")
    ap.add_argument("--no-impedance", action="store_true",
                    help="with --check-signal, skip the current-injection impedance "
                         "phase and report live metrics only")
    ap.add_argument("--impedance-input", choices=["n", "p"], default="n",
                    help="which ADS input the test current drives; depends on how the "
                         "reference is wired. If every channel reports no test "
                         "signal, try the other one (default: n)")
    ap.add_argument("--line-freq", type=float, default=60.0,
                    help="mains frequency for the noise metric (60 US, 50 EU)")
    ap.add_argument("--impedance-out", default="outputs/impedance.json",
                    help="where --check-signal writes measured impedances for the "
                         "dashboard head map")
    args = ap.parse_args(argv)

    ch_names = [c.strip() for c in args.channel_names.split(",") if c.strip()]

    if args.check_signal:
        try:
            q = check_signal(args.port, ch_names, line_freq=args.line_freq,
                             impedance=not args.no_impedance,
                             impedance_input=args.impedance_input)
        except Exception as exc:
            print(f"{YELLOW}Signal check failed:{RESET} {exc}")
            return 1
        n_bad = print_quality(q)
        measured = {c.name: c.impedance_kohm for c in q if c.impedance_kohm is not None}
        if measured:
            os.makedirs(os.path.dirname(args.impedance_out) or ".", exist_ok=True)
            with open(args.impedance_out, "w") as fh:
                json.dump({"timestamp": time.time(), "channels": measured,
                           "input_side": args.impedance_input}, fh, indent=2)
            print(f"\n{DIM}Impedance written to {args.impedance_out}; the dashboard "
                  f"displays it with its age.{RESET}")
        if n_bad:
            print(f"\n{YELLOW}{n_bad} channel(s) need attention{RESET} — reseat or "
                  f"re-gel those electrodes, then re-run this check.")
        else:
            print(f"\n{GREEN}All channels look usable.{RESET} Ready to record.")
        return 1 if n_bad else 0
    sched = build_schedule(n_per_class=args.trials_per_class, action_s=args.action_s,
                           seed=args.seed)

    print(f"\n{BOLD}Overt-speech calibration{RESET}")
    print(f"  {sched.n_attempt} attempt + {sched.n_rest} rest trials, randomly interleaved")
    print(f"  ~{sched.estimated_duration_s() / 60.0:.1f} min")
    print(f"  On {GREEN}SPEAK ALOUD: <word>{RESET} say the word out loud, once.")
    print(f"  On {DIM}REST{RESET} sit still and stay silent.")
    print("  Keep still otherwise — jaw and neck movement dominates EEG.\n")
    try:
        input("Press Enter when the electrodes are seated and impedances look good… ")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return 1

    try:
        rec = run_calibration(args.port, sched, ch_names, present)
    except KeyboardInterrupt:
        print("\nAborted; nothing saved.")
        return 1
    except Exception as exc:
        print(f"\n{YELLOW}Recording failed:{RESET} {exc}")
        return 1
    finally:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(
        args.out_dir, f"sub-{args.subject:02d}_ses-{args.session:02d}_{stamp}_calib.npz"
    )
    save_calibration(
        path, rec["X"], rec["y_go"], rec["y_word"], rec["ch_names"], rec["sfreq"],
        subject=args.subject, session=args.session, action_s=rec["action_s"],
        paradigm="overt",
    )
    n_att = int((rec["y_go"] == ATTEMPT).sum())
    n_rest = int((rec["y_go"] == REST).sum())
    print(f"\n{BOLD}Saved{RESET} {path}")
    print(f"  {n_att} attempt / {n_rest} rest trials at {rec['sfreq']:.0f} Hz")
    print(f"\nTrain the GO decoder on it:\n"
          f"  python run.py --calibration '{args.out_dir}/*.npz' --task go\n"
          f"Cross-subject LOSO needs several subjects; with one recording you will\n"
          f"get the within-subject number only.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
