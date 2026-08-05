"""
eeg_tvns — Riemannian EEG decoding pipeline for closed-loop tVNS speech rehabilitation.

Operates on real recorded EEG only -- there is no synthetic data backend and no
simulated stream source.

Public API:
    from eeg_tvns import (
        load_dataset, LOW_DENSITY_MONTAGE,
        build_pipeline, RiemannianAlignment,
        evaluate_cross_subject, evaluate_within_subject, permutation_chance,
        RealTimeDecoder, benchmark_latency,
        train,                      # the one training path (CLI and dashboard)
        SessionManager,             # owns the board, the modes, and the ARM switch
    )
"""
from .config import Config, LOW_DENSITY_MONTAGE
from .data_loader import load_dataset, load_inner_speech, Epochs
from .pipeline import build_pipeline, RiemannianAlignment
from .evaluate import (
    evaluate_cross_subject,
    evaluate_within_subject,
    permutation_chance,
)
from .realtime import RealTimeDecoder, benchmark_latency
from .preprocessing import preprocess, design_filters, apply_filters, resample_window
from .acquisition import (
    BaseStreamer,
    OpenBCIStreamer,
    run_loop,
    load_decoder,
)
from .signal_quality import (
    ChannelQuality,
    impedance_from_std_uv,
    live_quality,
    measure_impedance,
    scalp_positions,
)
from .boards import DEFAULT_BOARD, board_choices, list_serial_ports, probe_board
from .models import discover_models, gating_verdict, inspect_model
from .training import train
from .jobs import Job, JobRunner
from .session import Busy, SessionManager

__version__ = "1.2.0"

__all__ = [
    "Config",
    "LOW_DENSITY_MONTAGE",
    "load_dataset",
    "load_inner_speech",
    "Epochs",
    "build_pipeline",
    "RiemannianAlignment",
    "evaluate_cross_subject",
    "evaluate_within_subject",
    "permutation_chance",
    "RealTimeDecoder",
    "benchmark_latency",
    "preprocess",
    "design_filters",
    "apply_filters",
    "resample_window",
    "BaseStreamer",
    "OpenBCIStreamer",
    "run_loop",
    "load_decoder",
    "ChannelQuality",
    "live_quality",
    "measure_impedance",
    "impedance_from_std_uv",
    "scalp_positions",
    # Board registry and control plane
    "DEFAULT_BOARD",
    "board_choices",
    "list_serial_ports",
    "probe_board",
    "discover_models",
    "gating_verdict",
    "inspect_model",
    "train",
    "Job",
    "JobRunner",
    "Busy",
    "SessionManager",
]
