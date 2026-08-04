"""
eeg_tvns — Riemannian EEG decoding pipeline for closed-loop tVNS speech rehabilitation.

Public API:
    from eeg_tvns import (
        load_dataset, make_synthetic, LOW_DENSITY_MONTAGE,
        build_pipeline, RiemannianAlignment,
        evaluate_cross_subject, evaluate_within_subject, permutation_chance,
        RealTimeDecoder, benchmark_latency,
    )
"""
from .config import Config, LOW_DENSITY_MONTAGE
from .data_loader import load_dataset, load_inner_speech, make_synthetic, Epochs
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
    SimulatedStreamer,
    run_loop,
    load_decoder,
)

__version__ = "1.1.0"

__all__ = [
    "Config",
    "LOW_DENSITY_MONTAGE",
    "load_dataset",
    "load_inner_speech",
    "make_synthetic",
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
    "SimulatedStreamer",
    "run_loop",
    "load_decoder",
]
