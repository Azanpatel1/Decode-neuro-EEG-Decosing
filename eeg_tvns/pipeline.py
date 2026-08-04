"""The decoding pipeline: covariance -> Riemannian Alignment -> tangent space -> classifier.

This is the core recommended by the Decode Neuro design doc. Every component is
open source (pyRiemann / scikit-learn). The pipeline is a plain scikit-learn
`Pipeline`, so it slots directly into cross-validation and into the real-time
decoder.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import invsqrtm

from .config import Config

log = logging.getLogger("eeg_tvns.pipeline")


class RiemannianAlignment(BaseEstimator, TransformerMixin):
    """Recenter SPD covariance matrices to the identity (a.k.a. Riemannian
    Alignment / recentering, Zanini et al. 2018).

    ``fit`` estimates the geometric mean M of the covariances it is shown and
    ``transform`` maps each C -> M^{-1/2} C M^{-1/2}. When fit on a single
    subject/session's data (unsupervised -- no labels used) this removes that
    domain's covariance offset, which is the dominant source of cross-session
    and cross-subject non-stationarity in a longitudinal rehab BCI.

    Functionally equivalent to ``pyriemann.transfer.TLCenter`` but usable as a
    stand-alone, label-free step inside an ordinary scikit-learn Pipeline.
    """

    def __init__(self, metric: str = "riemann"):
        self.metric = metric

    def fit(self, X: np.ndarray, y=None):
        self.reference_ = mean_covariance(X, metric=self.metric)
        self.iM_ = invsqrtm(self.reference_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # iM_ is (c, c); X is (n, c, c). matmul broadcasts iM_ across trials.
        return self.iM_ @ X @ self.iM_


class TraceNormalize(BaseEstimator, TransformerMixin):
    """Scale each covariance matrix to unit trace, removing global signal power.

    Why this matters here: ds003626's rest class comes from a separate 15 s
    baseline block whose broadband amplitude is ~1.6x the action epochs', so an
    unnormalised covariance pipeline can separate rest from attempt on overall
    power alone -- a block/drift artifact, not a speech-attempt signature. A
    decoder built on that cue would fire on movement and impedance drift online.
    Normalising the trace forces the classifier onto the spatial *correlation
    structure* instead.

    The cost is that genuine power changes (ERD/ERS) are discarded too, so this
    trades some real signal for a cue that transfers to live use. It is a Pipeline
    step, so training and the online decoder get it identically (Invariant A).
    """

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        tr = np.trace(X, axis1=-2, axis2=-1)[..., None, None]
        return X / np.clip(tr, 1e-12, None) * X.shape[-1]


class FilterBankCovariances(BaseEstimator, TransformerMixin):
    """Optional light filter bank: covariance per frequency band, block-diagonally
    stacked. Captures mu / beta / low-gamma structure the design doc highlights.

    Input X is raw epoch data (n, ch, samp); output is (n, B*ch, B*ch) block-diag.
    """

    def __init__(self, sfreq: float, bands: List[Tuple[float, float]], estimator="oas"):
        self.sfreq = sfreq
        self.bands = bands
        self.estimator = estimator

    def _filt(self, X: np.ndarray, lo: float, hi: float) -> np.ndarray:
        from scipy.signal import butter, sosfiltfilt

        ny = self.sfreq / 2.0
        sos = butter(4, [lo / ny, min(hi, ny * 0.99) / ny], btype="band", output="sos")
        return sosfiltfilt(sos, X, axis=-1)

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        cov = Covariances(estimator=self.estimator)
        blocks = [cov.fit_transform(self._filt(X, lo, hi)) for lo, hi in self.bands]
        n, c, _ = blocks[0].shape
        B = len(blocks)
        out = np.zeros((n, B * c, B * c))
        for bi, blk in enumerate(blocks):
            out[:, bi * c:(bi + 1) * c, bi * c:(bi + 1) * c] = blk
        # tiny ridge to keep the block-diagonal SPD
        out += np.eye(B * c) * 1e-6
        return out


def _make_classifier(cfg: Config):
    if cfg.classifier == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if cfg.classifier == "logreg":
        return LogisticRegression(max_iter=1000, C=1.0)
    if cfg.classifier == "svm":
        return SVC(kernel="rbf", probability=True, C=1.0)
    if cfg.classifier == "mdm":
        return MDM(metric=cfg.metric)  # operates on covariances directly
    raise ValueError(cfg.classifier)


def build_pipeline(cfg: Config) -> Pipeline:
    """Assemble the scikit-learn Pipeline described in the design document.

    covariance -> (Riemannian Alignment) -> tangent space -> classifier
    (MDM is a special case: it classifies covariances directly, no tangent step.)
    """
    steps: list = []
    if cfg.filter_bank:
        steps.append(("cov", FilterBankCovariances(cfg.sfreq, cfg.freq_bands, cfg.cov_estimator)))
    else:
        steps.append(("cov", Covariances(estimator=cfg.cov_estimator)))

    if cfg.trace_normalize:
        steps.append(("norm", TraceNormalize()))

    if cfg.align:
        steps.append(("align", RiemannianAlignment(metric=cfg.metric)))

    if cfg.classifier == "mdm":
        steps.append(("clf", _make_classifier(cfg)))
    else:
        steps.append(("ts", TangentSpace(metric=cfg.metric)))
        steps.append(("clf", _make_classifier(cfg)))

    pipe = Pipeline(steps)
    log.info("Pipeline: %s", " -> ".join(name for name, _ in steps))
    return pipe
