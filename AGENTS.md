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

The operator surface is `dashboard.py`: a browser control plane that owns the board
and runs the whole workflow — probe hardware, check electrode contact, record
calibration, train and assign decoders, run the closed loop, watch it. It boots with
nothing connected. Firing the stimulator sits behind an explicit ARM switch that
defaults off (Invariant G).

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
├── run.py                thin CLI over eeg_tvns.training.train
├── calibrate.py          thin root entry point for the calibration recorder
├── acquisition.py        thin root entry point for the live loop
├── dashboard.py          thin root entry point for the dashboard / control plane
├── tools/
│   ├── fake_board.py     BoardShim test double (transport only; emits a ramp)
│   └── verify_control_plane.py  hardware-free guardrail suite
└── eeg_tvns/
    ├── __init__.py       public API + version
    ├── config.py         Config dataclass — the single source of run settings
    ├── preprocessing.py  SHARED band-pass/notch/resample (train + live use this)
    ├── data_loader.py    ds003626 + calibration loaders, montage select (real only)
    ├── calibrate.py      same-session cued attempt/rest recorder (GO training)
    ├── signal_quality.py ADS1299 impedance + live contact metrics + scalp layout
    ├── pipeline.py       Covariances → RiemannianAlignment → TangentSpace → clf
    ├── evaluate.py       leakage-free LOSO + within-subject + permutation chance
    ├── training.py       train(cfg, progress, should_stop) — the ONLY training path
    ├── realtime.py       RealTimeDecoder (GO gate + online recentering) + latency
    ├── acquisition.py    live OpenBCI streamer + closed loop (no simulator)
    ├── boards.py         board registry, port enumeration, probing, error text
    ├── models.py         bundle discovery/inspection + gating verdicts
    ├── jobs.py           one-at-a-time background job with progress and logs
    ├── live.py           FrameHub, AcquisitionThread, WordReadout (display side)
    ├── session.py        SessionManager: board modes, ARM switch, audit log
    ├── dashboard.py      FastAPI routes + WebSocket over a SessionManager
    └── web/              index.html, app.js, style.css (the five-tab UI)
```

Data flow, offline → online:

```
OFFLINE:  load_dataset → preprocess → Covariances → RiemannianAlignment
          → TangentSpace → LDA → evaluate (LOSO) → joblib bundle

ONLINE:   board window → reorder channels → SAME preprocess → resample to model
          rate → RealTimeDecoder.decode → GO? → ARMED? → fire_tvns()
          → update_reference
```

The dashboard is a control plane, not just a view. `SessionManager` owns the board
through one mode at a time (`idle | probing | signal_check | calibrating |
decoding`), because the serial port is exclusive and two activities sharing it fail
as garbage data rather than cleanly. A request arriving while another mode holds the
board is refused with `Busy` → HTTP 409; do not make it queue or force the port.

`fire_tvns` is reachable only through `SessionManager._gated_fire`, which requires
an armed session (see section 4G). Control-plane work — probing, contact checks,
training — stays off the decision path (Invariant E); training is refused outright
while decoding.

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
from the binary GO decoder. Keep that separation. The word readout
(`live.WordReadout`, re-exported as `dashboard.WordReadout`) upholds this
structurally: it runs inside the `on_frame` *observer*, which `run_loop` calls only
after it has already made and acted on its GO decision, so the word decoder has no
path into the fire logic. Do not move it into `run_loop` or let it influence
`fire_tvns`. `SessionManager.assign_model` additionally **rejects** a `task ==
"word"` bundle for the GO slot, so this holds at the API boundary and not only by
convention. Keep that rejection.

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

This extends to the test scaffolding. `tools/fake_board.py` stands in for the
BrainFlow *transport* and emits an obvious per-channel ramp; it must never be made
to emit anything EEG-like, or it becomes the simulator this invariant forbids.

**G. Stimulation requires an explicit ARM, and the display must not overstate it.**
`fire_tvns` has exactly one caller in the dashboard path,
`SessionManager._gated_fire`, and it fires only when `arm()` has succeeded. Arming
requires a running loop, a loopback binding (unless `--allow-remote-arm`), the GO
model's filename echoed back, and an acknowledgement when
`models.gating_verdict` says that model is not validated for gating. It auto-disarms
on stop, on any GO-model change or rewrite, and after `max_armed_s`. Do not add a
second path to the stimulator, do not default ARM to on, and do not relax the
loopback rule.

The monitor counts **GO events** (threshold crossings past the refractory window)
separately from **stimulations** (crossings that actually reached the device), via
the `stimulated` flag `AcquisitionThread` passes to `FrameHub.publish`. Never
collapse them back into one counter: a display that counted a suppressed crossing
as a stimulation would be claiming something happened to the patient that did not.

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
# 1. GO decoder. Prefer your own same-session recording -- the ds003626 GO task is
#    block-confounded (see T2b) and must not gate stimulation.
python calibrate.py --port /dev/cu.usbserial-XXXX --subject 1 --channel-names "F7,F8,..."
python run.py --calibration 'calib/*.npz' --task go
#    -> writes outputs/model_go.joblib, metrics_go.json, and plots
#    ds003626 variant, offline research only:
python run.py --data ./ds003626 --task go --condition overt_scaffold

# 2. 4-class word decoder (display/tracking only; measured AT CHANCE on real data)
python run.py --data ./ds003626 --task word --condition inner

# 3. Ablation: alignment matters (cross-subject score should drop with --no-align)
python run.py --data ./ds003626 --task word --no-align

# 4. Live closed loop -- requires the Cyton powered on and the OpenBCI GUI closed
python acquisition.py --port /dev/cu.usbserial-XXXX --model outputs/model_go.joblib --duration 8
#    -> logs p_go, latency, GO, and ">>> tVNS FIRE" events

# 5. Dashboard / control plane. Boots with nothing connected: --port and --model
#    only pre-fill the form. Everything in steps 1-4 can be driven from the tabs.
pip install "fastapi>=0.110,<0.120" "uvicorn>=0.27,<0.35" "websockets>=12"
python dashboard.py
#    -> open http://127.0.0.1:8765; the loop always starts DISARMED
```

**Expected sanity signals:** online latency is single-digit milliseconds, far under
the 300 ms budget; on the live loop, `p_go` tracks actual speech attempts and tVNS
fires with the refractory respected. Permutation `observed` should sit above
`null_mean` with a low `p_value` — but treat that as necessary, not sufficient: it
only shows *some* learnable structure exists, and on the ds003626 GO task that
structure is which recording block a window came from (p=0.005 while decoding
block identity). Always pair it with a control that shares the confound.

**Measured baselines on real data (leakage-free LOSO, 10 subjects, 16 ch)** — do
not "improve" these by loosening the evaluation:

| Problem | LOSO bacc | Chance | Note |
|---|---|---|---|
| Word, 4-class inner | 0.266 | 0.250 | at chance, p=0.50 |
| GO, baseline-block rest | 0.920 | 0.500 | confounded; block identity |
| GO, same-block rest (0.5 s control) | 0.540 | 0.500 | the honest estimate |

**Verifying without data or hardware** — run the guardrail suite. It exercises the
session state machine and its 409s, every ARM refusal, Invariant C at the API, the
cue timeline and display-lag arithmetic, the job runner, model gating verdicts, and
the absence of any way to fabricate a signal. Board-dependent paths run against
`tools/fake_board.py`:

```bash
python tools/verify_control_plane.py   # ~15 s, no network, no hardware
```

It must end `0 failed`. **Run it after any change to `session.py`, `dashboard.py`,
`live.py`, `jobs.py`, `models.py`, `boards.py`, or the files in `web/`,** and extend
it when you add a guardrail. Individual smoke checks, still true:

```bash
python -c "import eeg_tvns; print(eeg_tvns.__version__)"
python run.py --task go            # must exit: --data or --calibration is required
python dashboard.py --help         # --port is optional now; the UI picks one
```

There is no pytest suite yet (see task T5); the suite above plus the commands in
this section are the acceptance smoke test — **run what your change touches** and
confirm the signals above still hold. Do not add a synthetic mode to make any of
this easier to run.

---

## 7. Open tasks (backlog with acceptance criteria)

**T1 — Implement `fire_tvns()` for a real stimulator.**
Location: `eeg_tvns/acquisition.py` (`fire_tvns` stub). Replace the log stub with
the device trigger (serial command / GPIO / TTL pulse / vendor SDK). Keep it a
fast, non-blocking call — it runs inside the loop and must not blow the latency
budget. Keep it reachable only through `SessionManager._gated_fire` on the
dashboard path (Invariant G); do not add a second call site. *Accept when:* a real
or mocked trigger fires on GO **while armed**, nothing fires while disarmed, and the
loop's median latency stays < `Config.latency_budget_ms`.

**T2 — DONE. True "rest" epochs for the GO decoder.**
`data_loader._build_rest_from_baseline` builds the rest class from the dataset's
`*_baseline-epo.fif` recordings, sliced into action-length windows and subsampled
per subject to match attempt counts; `_finalize_task` raises if rest is absent.
The old low-power surrogate is gone and must not come back (Invariant F). Still
worth verifying against real data: confirm the printed attempt/rest counts are
balanced and that GO LOSO beats its permutation baseline.

**T2b — PARTLY DONE. Same-session calibration for a deployable GO gate.**
Built: `eeg_tvns/calibrate.py` + `calibrate.py` record cued overt attempt vs rest
randomly interleaved in one recording (runs capped at 3 so drift cannot align with
class), stored raw at board rate; `data_loader.load_calibration` + `run.py
--calibration` train on it at the board's native rate. The dashboard's Calibrate
tab drives the same recorder with browser-presented cues and stores the measured
per-trial `display_lag_s`. **Still open:** nobody has recorded real calibration data
yet, so the GO decoder remains unvalidated. Until that exists,
`outputs/model_go.joblib` is a ds003626 baseline-block model and must not gate
stimulation — `models.gating_verdict` marks it as such, and the dashboard requires
an explicit acknowledgement to arm it. *Accept when:* a real recording exists, its
GO score is reported next to the 0.54 same-block control, and it beats that control.

Background — why this task exists: ds003626's only rest is a separate baseline
block. At a matched 0.5 s window, holding the action epochs fixed and changing
only the rest source, baseline-block rest scores 0.798 LOSO balanced accuracy
while same-block rest (pre-cue interval of the same trials) scores 0.540. The high
GO numbers reflect block identity, not speech attempt.

**T3 — Add stimulation control arms.**
Add `paired` (current behavior), `unpaired/delayed`, and `sham` modes to
`run_loop` (config flag), so the experiment can show the effect is timing-
specific. Route them through `SessionManager._gated_fire` so ARM still governs, and
expose the mode on the Hardware tab and in the audit log. *Accept when:* a
`--mode {paired,delayed,sham}` flag changes when/if `fire_tvns` is called, each mode
is logged in the run summary, and the monitor's stimulation count still reflects only
real device triggers (Invariant G).

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
— never a generator (Invariant F); `tools/fake_board.py` may stand in for the
transport. Fold in `tools/verify_control_plane.py`, which already covers the
control-plane guardrails and is the model to follow. Accuracy/alignment tests need
real epochs, so scope them to a checked-in subject subset or mark them as requiring
ds003626. *Accept when:* `pytest` passes locally with no network or hardware.

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
  Golden Constraint 5). Prefer a loud failure over a plausible-looking trace. That
  includes making `tools/fake_board.py` emit anything that resembles EEG.
- Don't weaken the ARM path: no ARM-on-by-default, no second caller of `fire_tvns`,
  no dropping the loopback rule, the typed confirmation, or the unvalidated-model
  acknowledgement (Invariant G).
- Don't report a suppressed GO event as a stimulation anywhere in the UI, and don't
  merge the two counters.
- Don't duplicate the training pipeline for the dashboard. `run.py` and the Train
  tab both go through `eeg_tvns.training.train`; two copies would drift into
  producing different models from the same inputs.
- Don't run heavy work on a request handler or inside `run_loop`. Long jobs belong
  in `jobs.JobRunner`, and hardware activities in a `SessionManager` mode.

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
- **ARM** — the explicit, revocable permission for the closed loop to trigger the
  stimulator. Off by default; see Invariant G.
- **GO event vs stimulation** — a GO event is a threshold crossing past the
  refractory window; a stimulation is one that actually reached the device. They
  differ whenever the session is disarmed.
- **gating verdict** — `models.gating_verdict`'s judgement of whether a bundle is fit
  to gate stimulation (e.g. a ds003626 GO model is not, because its rest class is a
  separate recording block).
- **mode** — which activity currently owns the board: `idle`, `probing`,
  `signal_check`, `calibrating`, `decoding`. Exclusive, because the serial port is.
