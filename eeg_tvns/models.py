"""Decoder discovery, inspection, and gating verdicts.

The dashboard lets you pick which decoder gates stimulation, so it has to answer
a question the CLI never had to: *is this model trustworthy enough to fire tVNS
from?* That is not a property of the file, it is a property of how the model was
trained, and the bundle already records it — `config.data_root` versus
`config.calibration_glob` tells us whether the GO decoder learned "speech attempt
vs rest" or merely "which recording block this window came from".

Per AGENTS.md T2b, ds003626's only rest is a separate baseline block. Holding the
action epochs fixed and changing only the rest source moves LOSO balanced accuracy
from 0.798 (baseline-block rest) to 0.540 (same-block rest). So a GO model trained
on ds003626 scores high for the wrong reason and must not gate stimulation, and a
model trained on same-session calibration data only earns trust if it beats that
0.540 control.

Verdict levels: "ok" (usable), "unknown" (unscored / cannot tell), "bad" (known
to be confounded or at chance), "n/a" (word decoder, never gates by design).
"""
from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("eeg_tvns.models")

# The honest same-block GO control from AGENTS.md; a deployable GO decoder has to
# beat this, not the confounded 0.92.
SAME_BLOCK_CONTROL = 0.540
REQUIRED_KEYS = ("model", "config", "ch_names")


@dataclass
class Verdict:
    level: str
    label: str
    detail: str = ""

    def to_dict(self) -> Dict:
        return {"level": self.level, "label": self.label, "detail": self.detail}


@dataclass
class ModelInfo:
    path: str
    name: str
    task: Optional[str] = None
    condition: Optional[str] = None
    n_channels: Optional[int] = None
    sfreq: Optional[float] = None
    ch_names: List[str] = field(default_factory=list)
    label_names: Dict[int, str] = field(default_factory=dict)
    loso: Optional[float] = None
    source: Optional[str] = None
    mtime: float = 0.0
    error: Optional[str] = None
    verdict: Verdict = field(default_factory=lambda: Verdict("unknown", "unscored"))

    def to_dict(self) -> Dict:
        return {
            "path": self.path, "name": self.name, "task": self.task,
            "condition": self.condition, "n_channels": self.n_channels,
            "sfreq": self.sfreq, "ch_names": self.ch_names,
            "label_names": {str(k): v for k, v in self.label_names.items()},
            "loso": self.loso, "source": self.source, "mtime": self.mtime,
            "error": self.error, "verdict": self.verdict.to_dict(),
        }


def gating_verdict(task: Optional[str], source: Optional[str],
                   loso: Optional[float]) -> Verdict:
    """Judge whether a decoder may be trusted to gate stimulation."""
    if task == "word":
        return Verdict("n/a", "display only",
                       "The word decoder never gates stimulation (Invariant C): "
                       "multi-class attempt identity is unreliable in patients.")
    if task != "go":
        return Verdict("unknown", "unknown task",
                       "Could not read the task from this bundle's Config.")

    if source == "ds003626":
        return Verdict(
            "bad", "not validated for gating",
            "Trained against ds003626, whose only rest is a separate baseline "
            "block. At a matched window that scores 0.798 LOSO while same-block "
            "rest scores 0.540, so the accuracy reflects block identity rather "
            "than speech attempt. Fine for offline research, not for firing tVNS.")

    if source == "calibration":
        if loso is None:
            return Verdict("unknown", "same-session, unscored",
                           "Recorded in one session with rest interleaved, which "
                           "removes the block confound, but no score was found. "
                           f"It still has to beat the {SAME_BLOCK_CONTROL:.3f} "
                           "same-block control to mean anything.")
        if loso > SAME_BLOCK_CONTROL + 0.02:
            return Verdict("ok", f"same-session, {loso:.3f}",
                           f"Beats the {SAME_BLOCK_CONTROL:.3f} same-block control "
                           "on interleaved same-session data.")
        return Verdict("bad", f"at control ({loso:.3f})",
                       f"Does not beat the {SAME_BLOCK_CONTROL:.3f} same-block "
                       "control, so it has not been shown to detect speech "
                       "attempts at all.")

    return Verdict("unknown", "unknown provenance",
                   "Could not tell what this model was trained on.")


def _metrics_for(path: str, task: Optional[str]) -> Optional[float]:
    """Cross-subject LOSO score from the sibling metrics_<task>.json, if present."""
    if not task:
        return None
    mpath = os.path.join(os.path.dirname(path) or ".", f"metrics_{task}.json")
    if not os.path.exists(mpath):
        return None
    try:
        with open(mpath) as fh:
            m = json.load(fh)
    except Exception:
        return None
    # Only trust the metrics file if it actually describes this task.
    if m.get("task") != task:
        return None
    v = m.get("cross_subject_bacc_mean")
    return None if v is None else float(v)


_cache: Dict[str, Tuple[float, ModelInfo]] = {}


def inspect_model(path: str) -> ModelInfo:
    """Read a joblib bundle's metadata. Cached on (path, mtime)."""
    name = os.path.basename(path)
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        return ModelInfo(path=path, name=name, error=str(exc),
                         verdict=Verdict("unknown", "unreadable"))

    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]

    info = ModelInfo(path=path, name=name, mtime=mtime)
    try:
        import joblib

        bundle = joblib.load(path)
        missing = [k for k in REQUIRED_KEYS if k not in bundle]
        if missing:
            raise ValueError(f"bundle is missing {', '.join(missing)}")
        cfg = bundle["config"]
        info.task = getattr(cfg, "task", None)
        info.condition = getattr(cfg, "condition", None)
        info.ch_names = [str(c) for c in (bundle.get("ch_names") or [])]
        info.n_channels = len(info.ch_names) or None
        info.sfreq = float(bundle.get("sfreq", getattr(cfg, "sfreq", 0.0)) or 0.0) or None
        info.label_names = {int(k): str(v) for k, v in (bundle.get("label_names") or {}).items()}
        # Provenance: newer bundles record it explicitly; older ones are inferred
        # from the Config that trained them.
        info.source = bundle.get("source") or (
            "calibration" if getattr(cfg, "calibration_glob", None)
            else "ds003626" if getattr(cfg, "data_root", None) else None
        )
        info.loso = bundle.get("loso")
        if info.loso is None:
            info.loso = _metrics_for(path, info.task)
    except Exception as exc:
        info.error = f"{type(exc).__name__}: {exc}"
        info.verdict = Verdict("unknown", "unreadable", str(exc))
        _cache[path] = (mtime, info)
        return info

    info.verdict = gating_verdict(info.task, info.source, info.loso)
    _cache[path] = (mtime, info)
    return info


def discover_models(dirs: List[str]) -> List[ModelInfo]:
    """All *.joblib under the given directories, newest first."""
    seen, out = set(), []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.joblib"))):
            real = os.path.abspath(path)
            if real in seen:
                continue
            seen.add(real)
            out.append(inspect_model(path))
    out.sort(key=lambda m: m.mtime, reverse=True)
    return out


def import_model(raw: bytes, name: str, dest_dir: str) -> ModelInfo:
    """Write an uploaded bundle to `dest_dir` after checking it is usable.

    Validation is not paranoia about the upload; it is that a bundle without
    `ch_names` or `sfreq` cannot be mapped onto a live board at all, so accepting
    one would only fail later with a confusing error.
    """
    base = os.path.basename(name or "").strip() or "imported.joblib"
    if os.sep in base or (os.altsep and os.altsep in base) or base.startswith("."):
        raise ValueError("Invalid filename.")
    if not base.endswith(".joblib"):
        base += ".joblib"
    if not raw:
        raise ValueError("Uploaded file was empty.")

    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, base)
    stem = base[: -len(".joblib")]
    n = 1
    while os.path.exists(dest):
        dest = os.path.join(dest_dir, f"{stem}-{n}.joblib")
        n += 1

    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(raw)
    try:
        info = inspect_model(tmp)
        if info.error:
            raise ValueError(f"Not a usable model bundle: {info.error}")
        if not info.ch_names:
            raise ValueError("Bundle has no ch_names, so a live board cannot be "
                             "remapped onto it.")
        if not info.sfreq:
            raise ValueError("Bundle has no sfreq, so the online path cannot "
                             "resample to the training rate.")
    except Exception:
        os.remove(tmp)
        raise
    os.replace(tmp, dest)
    _cache.pop(tmp, None)
    log.info("Imported model %s (task=%s, %d ch @ %.0f Hz)",
             dest, info.task, info.n_channels or 0, info.sfreq or 0)
    return inspect_model(dest)


_rec_cache: Dict[str, Tuple[float, Dict]] = {}


def list_recordings(pattern: str = "calib/*.npz") -> List[Dict]:
    """Summarise calibration recordings for the UI, newest first.

    Cached on mtime: this is called on every control-state push, and opening every
    npz twice a second would be pointless I/O.
    """
    import numpy as np

    out = []
    for path in glob.glob(pattern):
        try:
            mtime = os.path.getmtime(path)
            hit = _rec_cache.get(path)
            if hit and hit[0] == mtime:
                out.append(hit[1])
                continue
            with np.load(path, allow_pickle=True) as z:
                y_go = z["y_go"]
                lag = z["display_lag_s"] if "display_lag_s" in z else None
                median_lag = None
                if lag is not None and np.isfinite(lag).any():
                    median_lag = round(float(np.nanmedian(lag)) * 1000.0, 1)
                rec = {
                    "path": path,
                    "name": os.path.basename(path),
                    "subject": int(z["subject"]),
                    "session": int(z["session"]),
                    "n_attempt": int((y_go == 1).sum()),
                    "n_rest": int((y_go == 0).sum()),
                    "sfreq": float(z["sfreq"]),
                    "n_channels": int(z["X"].shape[1]),
                    "created": float(z["created_unix"]) if "created_unix" in z else 0.0,
                    "cue_source": str(z["cue_source"]) if "cue_source" in z else "terminal",
                    "median_lag_ms": median_lag,
                }
            _rec_cache[path] = (mtime, rec)
            out.append(rec)
        except Exception as exc:
            log.warning("Could not read recording %s: %s", path, exc)
    out.sort(key=lambda r: r["created"], reverse=True)
    return out
