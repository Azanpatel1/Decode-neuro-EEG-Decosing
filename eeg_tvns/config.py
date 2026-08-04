"""Central configuration for the eeg_tvns pipeline.

Everything the pipeline needs to be reproducible lives here so that a single
Config object fully describes a run. All defaults are tuned for the
Decode Neuro closed-loop tVNS speech-rehab context:

  * post-stroke aphasia
  * low-density (8-16 channel) wearable EEG
  * closed-loop decision must land inside the ~300-400 ms VNS pairing window
  * publicly accessible tooling only (pyRiemann / MNE / scikit-learn)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Word intentions (classes) in the Nieto "Thinking out loud" dataset (ds003626)
# ---------------------------------------------------------------------------
CLASS_NAMES = {0: "up", 1: "down", 2: "right", 3: "left"}          # arriba/abajo/derecha/izquierda
CONDITION_NAMES = {0: "pronounced", 1: "inner", 2: "visualized"}   # overt / imagined / visual

# A curated low-density montage that emulates an 8-16 channel wearable and
# concentrates electrodes over speech-motor and fronto-temporal cortex.
# If these names are not present in the loaded data, the loader falls back to
# an evenly-spaced index subset (with a warning) so the run still completes.
LOW_DENSITY_MONTAGE: List[str] = [
    "F7", "F8", "FC5", "FC6", "FT7", "FT8", "T7", "T8",
    "C3", "C4", "CP5", "CP6", "P7", "P8", "Cz", "Pz",
]


@dataclass
class Config:
    # --- data source -------------------------------------------------------
    # Path to the ds003626 (BIDS) root. Required: there is no synthetic backend,
    # so load_dataset() raises when this is unset.
    data_root: Optional[str] = None
    subjects: Optional[List[int]] = None  # e.g. [1,2,3]; None -> all found
    sfreq: float = 256.0                  # target sampling rate (Hz)

    # --- task --------------------------------------------------------------
    # "go"   -> binary speech-attempt (any word) vs. rest/baseline  (drives tVNS)
    # "word" -> 4-class direction decode                            (tracking)
    task: str = "go"
    # which speech condition to use for the word task / attempt epochs
    # "inner", "pronounced", "visualized", or "overt_scaffold" (pron + inner)
    condition: str = "inner"

    # --- epoching / preprocessing -----------------------------------------
    l_freq: float = 1.0                   # band-pass low edge (Hz)
    h_freq: float = 40.0                  # band-pass high edge (Hz) -> keeps mu/beta/low-gamma
    notch: Optional[float] = 50.0         # line-noise notch (Hz); set None to disable
    tmin: float = 1.0                     # action-window start, s from cue (imagery onset)
    tmax: float = 3.5                     # action-window end,   s from cue
    filter_bank: bool = False             # if True, stack band-limited covariances

    # --- montage -----------------------------------------------------------
    use_low_density: bool = True          # emulate wearable montage (recommended)

    # --- model -------------------------------------------------------------
    cov_estimator: str = "oas"            # regularised covariance (good for short/low-ch)
    metric: str = "riemann"               # Riemannian metric for mean / tangent space
    align: bool = True                    # Riemannian Alignment (per-domain recentering)
    classifier: str = "lda"               # "lda" | "logreg" | "mdm" | "svm"

    # --- evaluation --------------------------------------------------------
    inner_cv_folds: int = 5               # within-subject StratifiedKFold
    n_permutations: int = 200             # label-shuffle permutations for empirical chance
    random_state: int = 42

    # --- closed-loop -------------------------------------------------------
    window_s: float = 2.0                 # sliding-window length for online decoding (s)
    hop_s: float = 0.1                    # hop between decisions (s)
    go_threshold: float = 0.5             # GO probability threshold to fire tVNS
    latency_budget_ms: float = 300.0      # compute budget within the VNS pairing window

    # --- output ------------------------------------------------------------
    out_dir: str = "outputs"
    make_plots: bool = True

    freq_bands: List = field(default_factory=lambda: [(8, 13), (13, 30), (30, 40)])

    def validate(self) -> None:
        assert self.task in ("go", "word"), f"unknown task {self.task!r}"
        assert self.condition in (
            "inner", "pronounced", "visualized", "overt_scaffold",
        ), f"unknown condition {self.condition!r}"
        assert self.classifier in ("lda", "logreg", "mdm", "svm")
        assert self.tmax > self.tmin
        assert self.h_freq > self.l_freq
