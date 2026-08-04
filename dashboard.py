#!/usr/bin/env python3
"""Entry point for the live EEG + decode dashboard.

    python dashboard.py --simulate --model outputs/model_go.joblib
    python dashboard.py --port /dev/cu.usbserial-XXXX --model outputs/model_go.joblib

Then open http://127.0.0.1:8765
The implementation lives in eeg_tvns/dashboard.py.
"""
from eeg_tvns.dashboard import main

if __name__ == "__main__":
    main()
