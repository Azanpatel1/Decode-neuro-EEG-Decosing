"""Leakage-free evaluation: cross-subject (LOSO), within-subject, and an
empirical-chance permutation test.

Design-doc rule baked in here: fit every transform (covariance recentering,
tangent-space reference, classifier) INSIDE the training fold only, report
cross-subject and within-subject separately, and always compare against the
empirical chance level -- never against zero.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import invsqrtm

from .config import Config
from .data_loader import Epochs
from .pipeline import (
    FilterBankCovariances,
    TraceNormalize,
    _make_classifier,
    build_pipeline,
)

log = logging.getLogger("eeg_tvns.eval")

# (done, total, label) -> None. Optional throughout, so the CLI path is unchanged
# and the dashboard can show progress on the slow loops without either knowing
# about the other.
Progress = Callable[[int, int, str], None]
ShouldStop = Callable[[], bool]


def _tick(progress: Optional[Progress], done: int, total: int, label: str) -> None:
    if progress is None:
        return
    try:
        progress(done, total, label)
    except Exception:
        log.exception("progress callback failed; continuing")


class Cancelled(RuntimeError):
    """Raised when a caller's should_stop() asks a long evaluation to give up."""


@dataclass
class EvalResult:
    kind: str
    fold_scores: List[float] = field(default_factory=list)
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    y_score: Optional[np.ndarray] = None      # prob of positive class (binary)
    labels: Optional[Dict[int, str]] = None
    extra: Dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_scores)) if self.fold_scores else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.fold_scores)) if self.fold_scores else float("nan")

    def report(self) -> str:
        lines = [
            f"[{self.kind}] balanced accuracy = "
            f"{self.mean:.3f} +/- {self.std:.3f}  (n_folds={len(self.fold_scores)})"
        ]
        if self.y_true is not None and self.labels is not None:
            n_cls = len(np.unique(self.y_true))
            lines.append(f"    chance (theoretical) = {1.0 / n_cls:.3f}")
            if n_cls == 2 and self.y_score is not None:
                try:
                    auc = roc_auc_score(self.y_true, self.y_score)
                    tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
                    sens = tp / (tp + fn) if (tp + fn) else float("nan")
                    spec = tn / (tn + fp) if (tn + fp) else float("nan")
                    lines.append(
                        f"    AUC = {auc:.3f} | sensitivity = {sens:.3f} | "
                        f"specificity = {spec:.3f}"
                    )
                except Exception:
                    pass
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Covariance helpers (shared)
# ---------------------------------------------------------------------------
def _covariances(X: np.ndarray, cfg: Config) -> np.ndarray:
    """Covariance features for the evaluation paths.

    Must stay in step with `pipeline.build_pipeline`'s cov (+ normalise) stages:
    LOSO and the permutation test build features here rather than through the
    Pipeline, so any feature-stage change made there has to be mirrored here or
    the headline score would describe different features than the saved model.
    """
    if cfg.filter_bank:
        covs = FilterBankCovariances(cfg.sfreq, cfg.freq_bands, cfg.cov_estimator).fit_transform(X)
    else:
        covs = Covariances(estimator=cfg.cov_estimator).fit_transform(X)
    if cfg.trace_normalize:
        covs = TraceNormalize().fit_transform(covs)
    return covs


def _align_per_domain(covs: np.ndarray, domains: np.ndarray, metric: str) -> np.ndarray:
    """Recenter each domain (subject/session) to the identity using ONLY that
    domain's own covariances -- label-free, hence leakage-free even for the
    held-out subject."""
    out = np.empty_like(covs)
    for d in np.unique(domains):
        m = domains == d
        ref = mean_covariance(covs[m], metric=metric)
        iM = invsqrtm(ref)
        out[m] = iM @ covs[m] @ iM
    return out


def _cov_classifier(cfg: Config):
    """A classifier that consumes covariance matrices directly."""
    if cfg.classifier == "mdm":
        return MDM(metric=cfg.metric)
    return Pipeline([("ts", TangentSpace(metric=cfg.metric)), ("clf", _make_classifier(cfg))])


# ---------------------------------------------------------------------------
# Cross-subject (leave-one-subject-out) -- the headline generalisation number
# ---------------------------------------------------------------------------
def evaluate_cross_subject(ep: Epochs, cfg: Config,
                           progress: Optional[Progress] = None) -> EvalResult:
    covs = _covariances(ep.X, cfg)
    subjects, y = ep.subjects, ep.y
    uniq = np.unique(subjects)
    if len(uniq) < 2:
        log.warning("Only one subject present; cross-subject LOSO is not meaningful.")

    res = EvalResult(kind="cross-subject LOSO", labels=ep.label_names)
    yt, yp, ys = [], [], []
    for done, test_s in enumerate(uniq):
        _tick(progress, done, len(uniq), f"LOSO fold {done + 1}/{len(uniq)}")
        tr = subjects != test_s
        te = ~tr
        if te.sum() == 0 or tr.sum() == 0:
            continue
        Xtr, Xte = covs[tr], covs[te]
        if cfg.align:
            Xtr = _align_per_domain(Xtr, subjects[tr], cfg.metric)
            Xte = _align_per_domain(Xte, subjects[te], cfg.metric)
        model = _cov_classifier(cfg)
        model.fit(Xtr, y[tr])
        pred = model.predict(Xte)
        score = balanced_accuracy_score(y[te], pred)
        res.fold_scores.append(score)
        yt.append(y[te]); yp.append(pred)
        if len(ep.label_names) == 2:
            ys.append(_positive_scores(model, Xte))
        log.info("  LOSO subject %s: balanced acc = %.3f", test_s, score)

    res.y_true = np.concatenate(yt) if yt else None
    res.y_pred = np.concatenate(yp) if yp else None
    res.y_score = np.concatenate(ys) if ys else None
    return res


def _positive_scores(model, Xte) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xte)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(Xte)
    return model.predict(Xte).astype(float)


# ---------------------------------------------------------------------------
# Within-subject (per-patient calibration ceiling)
# ---------------------------------------------------------------------------
def evaluate_within_subject(ep: Epochs, cfg: Config,
                            progress: Optional[Progress] = None) -> EvalResult:
    res = EvalResult(kind="within-subject CV", labels=ep.label_names)
    uniq = np.unique(ep.subjects)
    for done, s in enumerate(uniq):
        _tick(progress, done, len(uniq), f"within-subject {done + 1}/{len(uniq)}")
        m = ep.subjects == s
        Xs, ys_ = ep.X[m], ep.y[m]
        if len(np.unique(ys_)) < 2 or np.min(np.bincount(ys_)) < cfg.inner_cv_folds:
            continue
        skf = StratifiedKFold(cfg.inner_cv_folds, shuffle=True, random_state=cfg.random_state)
        subj_scores = []
        for tr, te in skf.split(Xs, ys_):
            pipe = build_pipeline(cfg)
            pipe.fit(Xs[tr], ys_[tr])
            subj_scores.append(balanced_accuracy_score(ys_[te], pipe.predict(Xs[te])))
        res.fold_scores.append(float(np.mean(subj_scores)))
        log.info("  within-subject %s: balanced acc = %.3f", s, np.mean(subj_scores))
    return res


# ---------------------------------------------------------------------------
# Empirical chance via label permutation
# ---------------------------------------------------------------------------
def permutation_chance(ep: Epochs, cfg: Config, n_perm: Optional[int] = None,
                       progress: Optional[Progress] = None,
                       should_stop: Optional[ShouldStop] = None) -> Dict:
    """Pooled k-fold score with true labels vs. a null distribution built by
    shuffling labels. Returns observed score, null mean, and a p-value.

    This is the dominant cost of a training run (`n_perm` full refits), so it is
    the one loop that reports progress and honours cancellation.
    """
    n_perm = n_perm if n_perm is not None else cfg.n_permutations
    covs = _covariances(ep.X, cfg)
    y = ep.y
    rng = np.random.default_rng(cfg.random_state)
    # 3-fold keeps the permutation loop (the slowest step) tractable.
    skf = StratifiedKFold(3, shuffle=True, random_state=cfg.random_state)

    def pooled_score(labels: np.ndarray) -> float:
        scores = []
        for tr, te in skf.split(covs, labels):
            model = _cov_classifier(cfg)
            model.fit(covs[tr], labels[tr])
            scores.append(balanced_accuracy_score(labels[te], model.predict(covs[te])))
        return float(np.mean(scores))

    _tick(progress, 0, n_perm + 1, "permutation: observed score")
    observed = pooled_score(y)
    if n_perm <= 0:
        # No null distribution means no empirical chance level, so say so rather
        # than emitting a p-value computed from nothing.
        log.warning("Permutation test skipped (n_permutations=0): the observed "
                    "%.3f has no chance baseline to be compared against.", observed)
        return {"observed": observed, "null_mean": None, "null_std": None,
                "p_value": None, "n_permutations": 0}
    null_list = []
    for i in range(n_perm):
        if should_stop is not None and should_stop():
            raise Cancelled(f"cancelled after {i} of {n_perm} permutations")
        null_list.append(pooled_score(rng.permutation(y)))
        _tick(progress, i + 1, n_perm + 1, f"permutation {i + 1}/{n_perm}")
    null = np.array(null_list)
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    log.info(
        "Permutation: observed=%.3f  null=%.3f+/-%.3f  p=%.4f (n=%d)",
        observed, null.mean(), null.std(), p, n_perm,
    )
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_value": float(p),
        "n_permutations": n_perm,
    }
