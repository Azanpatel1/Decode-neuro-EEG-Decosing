# AGENTS.md — eeg_tvns

> Briefing for an AI coding agent (Cursor) working in this repo. Read this fully
> before editing. It explains what the project is, the invariants you must not
> break, how to run and verify things, and the concrete tasks that are open.
> Place this file at the repo root; Cursor loads `AGENTS.md` automatically.

---

## 1. What this project is

`eeg_tvns` is a **closed-loop EEG decoder for a tVNS speech-rehabilitation
experiment** (post-stroke aphasia, low-density 8–16 channel wearable EEG). It
decodes a patient's speech *attempt* from EEG in real time and fires
transcutaneous vagus nerve stimulation (tVNS) inside the neural "pairing window"
to drive rehabilitative plasticity.

The decoding approach is deliberately **not** the highest-accuracy deep net in
the literature. Those numbers are inflated (data leakage / intra-subject
evaluation) or proprietary. This project uses a **reproducible, low-latency,
open-source Riemannian pipeline** that survives leakage-free cross-subject
validation:

```
covariance  ->  Riemannian Alignment  ->  tangent space  ->  shallow classifier (LDA)
```

Two decoders:
- **GO decoder** (binary: speech-attempt vs. rest) — *this* is what gates tVNS.
- **Word decoder** (4-class: up / down / left / right) — progress tracking only.

Everything is open source: **pyRiemann · MNE · scikit-learn · BrainFlow**.

**Real data only.** There is no synthetic dataset and no simulated stream — see
Golden Constraint 5 and Invariant F. Training needs ds003626 on disk; the live
loop needs an OpenBCI board. Both fail loudly when those are absent.

---

## 2. Golden constraints (the "why" behind decisions — respect these)

1. **Publicly accessible tooling only.** No proprietary/closed models or paid
   toolboxes. If you add a dependency, it must be open-source and on PyPI.
2. **Reproducibility over headline accuracy.** Never report accuracy without
   comparing to the empirical chance level (permutation test). A "90%" with no
   chance baseline is meaningless here.
3. **Low latency.** The online decode must fit inside the VNS pairing window
   (`Config.latency_budget_ms`, default 300 ms). Do not add heavy models
   (transformers, foundation models, GPUs) to the *online* path.
4. **Train/inference parity.** The features the model sees live must match those
   it trained on (see Invariant A below).
5. **Real recorded data only.** The platform has no synthetic dataset generator
   and no simulated stream, and must not regain one. Rationale: on a strip chart
   generated traces are indistinguishable from EEG, and a model fit on surrogate
   data emits confident probabilities that mean nothing about a real brain — both
   are ways for a stimulation-gating system to look like it works when it does
   not. If you need a fixture for tests, use a short real recording committed as
   test data, never a generator (see Invariant F).

---

## 3. Repo map

```
eeg_tvns_pipeline/
├── AGENTS.md              ← this file
├── README.md             human-facing usage guide
├── requirements.txt      core deps (brainflow/openneuro-py optional, commented)
├── run.py                CLI: load → evaluate → fit → save model → latency + plots
├── acquisition.py        thin root entry point for the live loop
├── dashboard.py          thin root entry point for the live browser dashboard
└── eeg_tvns/
    ├── __init__.py       public API + version
    ├── config.py         Config dataclass — the single source of run settings
    ├── preprocessing.py  SHARED band-pass/notch/resample (train + live use this)
    ├── data_loader.py    ds003626 loader + montage select (real data only)
    ├── pipeline.py       Covariances → RiemannianAlignment → TangentSpace → clf
    ├── evaluate.py       leakage-free LOSO + within-subject + permutation chance
    ├── realtime.py       RealTimeDecoder (GO gate + online recentering) + latency
    ├── acquisition.py    live OpenBCI streamer + closed loop (no simulator)
    └── dashboard.py      FastAPI/WebSocket live monitor (observes run_loop only)
```

Data flow, offline → online:

```
OFFLINE:  load_dataset → preprocess → Covariances → RiemannianAlignment
          → TangentSpace → LDA → evaluate (LOSO) → joblib bundle

ONLINE:   board window → reorder channels → SAME preprocess → resample to model
          rate → RealTimeDecoder.decode → GO? → fire_tvns() → update_reference
```

---

## 4. Invariants — do NOT break these

**A. Preprocessing parity.** Offline (`data_loader._preprocess_array`) and online
(`acquisition.BaseStreamer.process_window`) MUST both filter via
`eeg_tvns/preprocessing.py`. If you change the band-pass, notch, or filter
family, it changes for both paths automatically — never fork one path onto a
different filter design. This parity is the whole reason `preprocessing.py`
exists.

**B. No data leakage in evaluation.** In `evaluate.py`, every transform
(covariance recentering, tangent-space reference, classifier) is fit **inside the
training fold only**. Riemannian Alignment recenters each domain using **only
that domain's own covariances** (label-free), which is why aligning the held-out
subject is not leakage. Do not "helpfully" fit anything on the full dataset
before splitting.

**C. The GO decoder gates stimulation, not the word decoder.** Multi-class
imagined-word identity is unreliable in patients. Stimulation timing must come
from the binary GO decoder. Keep that separation. The dashboard's word readout
(`dashboard.WordReadout`) upholds this structurally: it runs inside the `on_frame`
*observer*, which `run_loop` calls only after it has already made and acted on its
GO decision, so the word decoder has no path into the fire logic. Do not move it
into `run_loop` or let it influence `fire_tvns`.

**D. Channel order must be explicit.** The live board's channels must be remapped
to the model's trained montage (`bundle["ch_names"]`). Never assume identity
order silently. `--channel-names` drives `channel_order_from_names`, which warns
on mismatch. Preserve that warning.

**E. Latency budget.** Keep `RealTimeDecoder.decode` cheap. If you add anything to
the online path, re-run the latency benchmark and confirm it stays well under
`Config.latency_budget_ms`.

**F. No fabricated signals, anywhere.** `load_dataset` raises when `cfg.data_root`
is unset, and `OpenBCIStreamer` is the only streamer. Do not reintroduce a
`make_synthetic`, a `SimulatedStreamer`, a `--synthetic`/`--simulate` flag, or an
inline `rng.normal(...)` fallback for a missing window — not even "just for the
demo" or "just for tests". If a code path cannot run without data or hardware, it
must fail loudly with an actionable message instead of producing numbers. The
corollary: an empty dashboard is a *correct* dashboard when nothing is connected.

---

## 5. Environment setup (important for Cursor)

This is a plain Python ≥3.9 project. The single most common failure is a
**Python-interpreter mismatch**: Cursor runs the file with one interpreter while
deps were installed into another → `ModuleNotFoundError: No module named
'pyriemann'`.

Do this once:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then in Cursor, select the `.venv` interpreter (Command Palette → "Python: Select
Interpreter" → pick `./.venv`). Install and run must use the same interpreter.

Optional extras (only when needed):
- `pip install brainflow`      → live OpenBCI hardware
- `pip install openneuro-py`   → download the real dataset (ds003626)

---

## 6. How to run & verify

Everything requires real data; the offline half needs no hardware. **Get the
dataset first** — without it nothing runs, by design (Invariant F):

```bash
pip install openneuro-py
openneuro-py download --dataset ds003626 --target-dir ds003626
```

```bash
# 1. Train the GO decoder (this is what gates tVNS)
python run.py --data ./ds003626 --task go --condition overt_scaffold
#    -> writes outputs/model_go.joblib, metrics.json, confusion_matrix.png, latency_hist.png

# 2. 4-class word decoder (display/tracking only)
python run.py --data ./ds003626 --task word --condition inner

# 3. Ablation: alignment matters (cross-subject score should drop with --no-align)
python run.py --data ./ds003626 --task word --no-align

# 4. Live closed loop -- requires the Cyton powered on and the OpenBCI GUI closed
python acquisition.py --port /dev/cu.usbserial-XXXX --model outputs/model_go.joblib --duration 8
#    -> logs p_go, latency, GO, and ">>> tVNS FIRE" events

# 5. Live browser dashboard (traces + p_go + word readout + trigger window)
pip install "fastapi>=0.110,<0.120" "uvicorn>=0.27,<0.35" "websockets>=12"
python dashboard.py --port /dev/cu.usbserial-XXXX --model outputs/model_go.joblib
#    -> open http://127.0.0.1:8765
```

**Expected sanity signals:** permutation `observed` sits well above `null_mean`
with a low `p_value` (report nothing without this comparison); online latency is
single-digit milliseconds, far under the 300 ms budget; on the live loop, `p_go`
tracks actual speech attempts and tVNS fires with the refractory respected.

**Verifying without data or hardware** — you can still check that the code
imports, the CLIs parse, and the guardrails hold. These must all fail loudly
rather than invent data:

```bash
python -c "import eeg_tvns; print(eeg_tvns.__version__)"
python run.py --task go            # must exit: --data is required
python dashboard.py                # must exit: --port is required
```

There is no formal test suite yet (see task T5). Until then, the commands above
are the acceptance smoke test — **run what your change touches** and confirm the
signals above still hold. Do not add a synthetic mode to make the smoke test
easier to run.

---

## 7. Open tasks (backlog with acceptance criteria)

**T1 — Implement `fire_tvns()` for a real stimulator.**
Location: `eeg_tvns/acquisition.py` (`fire_tvns` stub). Replace the log stub with
the device trigger (serial command / GPIO / TTL pulse / vendor SDK). Keep it a
fast, non-blocking call — it runs inside the loop and must not blow the latency
budget. *Accept when:* a real or mocked trigger fires on GO and the loop's median
latency stays < `Config.latency_budget_ms`.

**T2 — DONE. True "rest" epochs for the GO decoder.**
`data_loader._build_rest_from_baseline` builds the rest class from the dataset's
`*_baseline-epo.fif` recordings, sliced into action-length windows and subsampled
per subject to match attempt counts; `_finalize_task` raises if rest is absent.
The old low-power surrogate is gone and must not come back (Invariant F). Still
worth verifying against real data: confirm the printed attempt/rest counts are
balanced and that GO LOSO beats its permutation baseline.

**T2b — Record same-session calibration data for a deployable GO gate.**
The ds003626 GO problem is confounded: its only rest is a separate baseline block,
and at a matched 0.5 s window baseline-block rest scores 0.80 LOSO balanced
accuracy while same-block rest (pre-cue interval of the same trials) scores 0.54.
The high GO numbers therefore reflect block identity, not speech attempt, and a
model trained that way must not gate stimulation. What's needed: a calibration
recording on the target hardware that interleaves cued attempt and cued rest
within one session, so both classes share block, impedance and arousal context.
*Accept when:* `--task go` can be built from a same-session calibration recording,
and its LOSO/within-session score is reported next to the same-block control above.

**T3 — Add stimulation control arms.**
Add `paired` (current behavior), `unpaired/delayed`, and `sham` modes to
`run_loop` (config flag), so the experiment can show the effect is timing-
specific. *Accept when:* a `--mode {paired,delayed,sham}` flag changes when/if
`fire_tvns` is called, and each mode is logged in the run summary.

**T4 — Optional: train at the board's native rate.**
125 Hz board vs. 256 Hz model currently reconciled by online resampling. Add a
path to calibrate/train a model at `Config.sfreq = 125` for the cleanest match.
*Accept when:* a 125 Hz model runs live with `--no-resample` and matching window
sizes.

**T5 — Add a pytest suite.**
Cover: preprocessing parity (offline vs. online filter output identical for the
same input), latency under budget, `load_dataset` raising without `--data`, and
both CLIs rejecting a missing `--port`/`--data`. Fixtures must be either a small
**real** recording committed as test data or a recorded window replayed from disk
— never a generator (Invariant F). Accuracy/alignment tests need real epochs, so
scope them to a checked-in subject subset or mark them as requiring ds003626.
*Accept when:* `pytest` passes locally with no network or hardware.

When you pick up a task, keep changes scoped to it and re-run the section-6 smoke
test before declaring done.

---

## 8. Conventions

- Style: standard PEP 8, type hints on public functions, module-level
  `logging.getLogger("eeg_tvns.<module>")` — no `print` in library code (the CLIs
  print results; libraries log).
- `Config` (in `config.py`) is the single source of truth for run settings. Add
  new knobs there with a sensible default and document them; don't scatter magic
  numbers.
- New heavy or optional dependencies go in `requirements.txt` **commented** unless
  they're needed for the core offline training path.
- Keep the online path (`realtime.py`, `acquisition.process_window`) allocation-
  light and free of heavyweight models.

---

## 9. What NOT to do

- Don't add closed-source or paid dependencies (violates Golden Constraint 1).
- Don't move preprocessing off `preprocessing.py` for either path (Invariant A).
- Don't fit scalers/feature-selectors/means on the full dataset before CV
  splitting (Invariant B).
- Don't let the word (4-class) decoder trigger stimulation (Invariant C).
- Don't put transformers/RNNs/foundation models/GPU inference in the online path
  (Golden Constraint 3). They're fine as *offline* research comparisons.
- Don't claim an accuracy number without its permutation chance baseline. Note
  that a permutation test does NOT catch a block confound: the ds003626 GO task
  passes at p=0.005 while mostly decoding which recording block a window came
  from. Compare against a same-block control too (see T2b).
- Don't reintroduce synthetic data or a simulated stream in any form — no
  `make_synthetic`, no `SimulatedStreamer`, no `--synthetic`/`--simulate`, no
  random-noise fallback for a missing window or a missing dataset (Invariant F /
  Golden Constraint 5). Prefer a loud failure over a plausible-looking trace.

---

## 10. Glossary

- **tVNS** — transcutaneous vagus nerve stimulation (auricular). Paired with a
  correctly-timed speech attempt to drive plasticity.
- **GO decoder** — binary "speech-attempt vs rest" classifier; gates tVNS.
- **Riemannian Alignment (recentering)** — recenters each session/subject's
  covariance matrices to the identity to remove domain shift; the key to
  cross-session robustness. See `pipeline.RiemannianAlignment` (≡
  `pyriemann.transfer.TLCenter`).
- **Tangent space** — maps SPD covariance matrices to a flat Euclidean space so
  ordinary classifiers (LDA) can operate on them.
- **LOSO** — leave-one-subject-out cross-validation (the honest generalization
  metric).
- **ds003626** — Nieto "Thinking out loud" inner-speech EEG dataset (OpenNeuro);
  the real data source. Includes overt (pronounced) and imagined conditions.
- **pairing window** — the short (~300–400 ms) interval after a speech attempt in
  which VNS must fire to reinforce it.
