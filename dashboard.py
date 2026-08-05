#!/usr/bin/env python3
"""Entry point for the EEG dashboard and control plane.

    python dashboard.py
    python dashboard.py --port /dev/cu.usbserial-XXXX --model outputs/model_go.joblib

Then open http://127.0.0.1:8765

Boots with nothing connected: connect the board, check electrode contact, record
calibration, train and assign decoders, and run the closed loop from the browser.
The arguments above only pre-fill those choices.

Displays real recorded EEG only -- there is no simulated stream source, so an empty
monitor with no board attached is the correct display. tVNS fires only while
explicitly armed, and never from the word decoder.

The implementation lives in eeg_tvns/dashboard.py.
"""
from eeg_tvns.dashboard import main

if __name__ == "__main__":
    main()
