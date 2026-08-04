#!/usr/bin/env python3
"""Entry point for live acquisition (kept at repo root for convenience).

    python acquisition.py --port /dev/ttyUSB0 --model outputs/model_go.joblib
    python acquisition.py --simulate --model outputs/model_go.joblib --duration 8

The implementation lives in eeg_tvns/acquisition.py (importable + testable).
"""
from eeg_tvns.acquisition import main

if __name__ == "__main__":
    main()
