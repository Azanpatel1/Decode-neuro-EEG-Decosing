#!/usr/bin/env python3
"""Entry point for live acquisition (kept at repo root for convenience).

    python acquisition.py --port /dev/cu.usbserial-XXXX --model outputs/model_go.joblib

Requires real hardware -- there is no simulated stream source.
The implementation lives in eeg_tvns/acquisition.py (importable + testable).
"""
from eeg_tvns.acquisition import main

if __name__ == "__main__":
    main()
