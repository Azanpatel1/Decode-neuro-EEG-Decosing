# eeg_tvns — Riemannian EEG decoder for closed-loop tVNS speech rehab

A complete, runnable signal-processing + decoding pipeline for the Decode Neuro
closed-loop tVNS speech-rehabilitation experiment (post-stroke aphasia,
low-density 8–16 channel wearable EEG). It implements the design recommended in
the project's algorithm report:

> **covariance → Riemannian Alignment → tangent space → shallow classifier (LDA)**,
> with a **binary GO decoder** (speech-attempt vs. rest) that gates stimulation
> and a **4-class word decoder** (up / down / left / right) for tracking.

Everything is open source (pyRiemann · MNE · scikit-learn).

**Real data only.** This platform has no synthetic dataset and no simulated
stream. Both were removed deliberately: on a strip chart, generated traces are
indistinguishable from EEG, and a model fit on surrogate data will report
confident probabilities on real brain signals that mean nothing. Training
requires the real **Nieto "Thinking out loud"** dataset (OpenNeuro **ds003626**);
the live loop requires real hardware.

---

## 1. Install

```bash
pip install -r requirements.txt
```

Core deps: numpy, scipy, scikit-learn, pyriemann, mne, joblib, matplotlib.

## 2. Get the dataset (required)

```bash
pip install openneuro-py
openneuro-py download --dataset ds003626 --target-dir ds003626
```

Nothing in the pipeline runs without it — `run.py` requires `--data`, and
`load_dataset` raises rather than falling back to generated data.

## 3. Train the decoders

```bash
# GO decoder, overt-speech-scaffolded (best for aphasia) -- this gates tVNS:
python run.py --data ./ds003626 --task go   --condition overt_scaffold

# 4-class imagined-word decoder (tracking / display only):
python run.py --data ./ds003626 --task word --condition inner
```

Ablation showing why Riemannian Alignment matters (cross-subject score should
drop):

```bash
python run.py --data ./ds003626 --task word --no-align
```

The loader reads the derivatives (`*_eeg-epo.fif` + `*_events.dat`),
**auto-detects** which label column is the word class ({0,1,2,3}) and which is
the condition ({0,1,2}), band-passes 1–40 Hz, crops to the action window
(1.0–3.5 s), and selects a low-density wearable-like montage. On the first run,
**check the printed class distribution** to confirm the labels decoded correctly.

## 4. What each run produces (in `./outputs`)

| File | Contents |
|---|---|
| `model_<task>.joblib` | the fitted, deployable pipeline + config + label names |
| `metrics.json` | cross-subject / within-subject balanced accuracy, permutation p-value, latency |
| `confusion_matrix.png` | normalised cross-subject confusion |
| `latency_hist.png` | closed-loop decision latency vs. the VNS pairing budget |

## 5. Key options

```
--task {go,word}            go = binary attempt/rest (default); word = 4-class
--condition {inner,pronounced,visualized,overt_scaffold}
--classifier {lda,logreg,mdm,svm}     default lda (shrinkage)
--no-align                  ablate Riemannian Alignment
--filter-bank               mu/beta/low-gamma filter-bank covariances
--all-channels              use all channels (default: low-density montage)
--subjects 1 2 3            restrict to a subject subset
--permutations N            empirical-chance permutations (0 to skip; slowest step)
```

## 6. How the pieces map to the design doc

| Stage | Module | Notes |
|---|---|---|
| Preprocess (band-pass, crop) | `data_loader.py` | MNE with SciPy fallback |
| Low-density montage | `data_loader.select_montage` | emulates the 8–16 ch wearable |
| Covariance | `pipeline.Covariances` / `FilterBankCovariances` | OAS-regularised |
| **Riemannian Alignment** | `pipeline.RiemannianAlignment` | per-domain recentering; ≡ `pyriemann.transfer.TLCenter` |
| Tangent space | `pipeline.TangentSpace` | |
| Classifier | `pipeline._make_classifier` | LDA (default), logreg, SVM, MDM |
| Leakage-free eval | `evaluate.py` | LOSO + within-subject + permutation chance |
| Closed-loop GO gate | `realtime.RealTimeDecoder` | GO threshold + online recentering |
| Latency benchmark | `realtime.benchmark_latency` | proves compute fits the pairing window |

## 7. Live hardware acquisition (OpenBCI Cyton+Daisy)

`acquisition.py` bridges a live BrainFlow stream into the trained decoder and
gates tVNS on GO. It reuses the **same** shared preprocessing as training
(`eeg_tvns/preprocessing.py`), resamples each board window to the model's rate,
and remaps board channels onto the trained montage — so the covariance features
online match what the model saw offline.

Live on an OpenBCI Cyton+Daisy (16 ch @ 125 Hz). Real hardware is the only
stream source — `--port` is required and there is no dry-run mode:

```bash
pip install brainflow
python acquisition.py --port /dev/cu.usbserial-XXXX --model outputs/model_go.joblib \
    --channel-names "F7,F8,FC5,FC6,FT7,FT8,T7,T8,C3,C4,CP5,CP6,P7,P8,Cz,Pz"
```

`--channel-names` gives the physical electrode label of each board channel **in
board order**; the adapter reorders them to the model's trained montage and warns
on any mismatch (never a silent identity mapping). Wire your stimulator into the
`fire_tvns()` stub in `eeg_tvns/acquisition.py` (serial / GPIO / TTL). Options:
`--hop`, `--refractory` (min seconds between fires), `--threshold`,
`--no-resample`, `--duration`.

The board runs at 125 Hz while the model trains at 256 Hz; by default each window
is resampled to the model rate so features line up. For the cleanest match,
calibrate/train a model at the board's rate (`Config.sfreq = 125`).

## 7b. Live browser dashboard

For a real-time monitor of the EEG traces and the decoder's output, use
`dashboard.py`. It runs the same closed loop as `acquisition.py` and streams
each decode to a browser UI over a WebSocket.

```bash
pip install "fastapi>=0.110,<0.120" "uvicorn>=0.27,<0.35" "websockets>=12"

python dashboard.py --port /dev/cu.usbserial-XXXX --model outputs/model_go.joblib
```

Every trace it draws is measured from the board. Because there is no simulated
source, an empty or flatlined chart always means a real acquisition problem
(board off, port busy, electrodes not seated) — never synthetic filler.

Open http://127.0.0.1:8765. Panels:

| Panel | Shows |
|---|---|
| Live EEG traces | 16-channel rolling strip chart (last 6 s), red line at each tVNS fire |
| **Decoded window that fired tVNS** | the *frozen* window that actually crossed threshold, with its `p(go)`, threshold, and age -- the evidence behind the stimulation |
| GO decoder | GO/IDLE state, `p(go)`, active threshold, fire and decision counts |
| **Word decode** | 4-class attempt identity (up / down / right / left) with per-class probability bars |
| p(go) history | `p(go)` over time with the threshold line |
| Latency | decision latency against the 300 ms pairing budget |

The **word decode is display only**. It loads `outputs/model_word.joblib` (train
with `python run.py --data ./ds003626 --task word`) and runs inside the dashboard's
observer callback -- *after* `run_loop` has already made its GO decision -- so it
has no path back into the stimulation trigger (Invariant C). It is decoded only
on GO frames, since word identity during rest is meaningless. Disable with
`--word-model none`; a missing model just greys the panel out.

If acquisition can't start (e.g. Cyton powered off, or the OpenBCI GUI holding
the port), a red banner explains what to fix instead of showing silent flatlines.

Extra flags: `--word-model`, `--host`, `--web-port`, `--publish-hz` (UI refresh
rate; the decoder always runs at its native `hop_s`). All acquisition flags are
supported.

## 8. Using the trained model directly in your own loop

```python
import joblib, numpy as np
from eeg_tvns.realtime import RealTimeDecoder

bundle = joblib.load("outputs/model_go.joblib")
decoder = RealTimeDecoder(bundle["model"], bundle["config"], positive_label=1)

# window: (n_channels, n_samples) pulled from your acquisition stream
d = decoder.decode(window)
if d.go:                      # p(attempt) >= go_threshold
    fire_tvns()               # <-- trigger your stimulator here
decoder.update_reference(window)   # online drift tracking between trials
print(d.probability, d.latency_ms)
```

## 9. Honest caveats (read these)

- **No synthetic anywhere.** There is no surrogate dataset and no simulated
  stream, so every number and every trace comes from a real recording. The cost
  is that you cannot exercise the pipeline without ds003626 and hardware; that
  tradeoff is intentional.
- **Imagined speech is hard.** Real imagined-speech accuracy is modest and, per
  the literature, multi-class imagined-word ID may sit near chance in patients —
  which is exactly why the closed loop leans on the **binary GO decoder** plus
  overt scaffolding, not word ID, to time stimulation.
- **Always read `metrics.json` against the permutation `null_mean`**, not against
  zero. A number above chance with a low p-value is the only thing that counts.
- **GO "rest" is real baseline — and its score is a block confound, not a
  decoder.** Rest comes from the dataset's `*_baseline-epo.fif` recordings, sliced
  into action-length windows and subsampled per subject to match attempt counts
  (nothing is synthesized; missing baselines raise). But that block differs from
  the task blocks in far more than speech attempt: ~1.6× the broadband amplitude,
  plus eyes-closed/arousal and drift differences. Measured on ds003626 at a
  matched 0.5 s window, holding the action epochs fixed and changing only where
  rest comes from:

  | Rest source | LOSO balanced acc | AUC |
  |---|---|---|
  | Separate baseline block | 0.798 | 0.867 |
  | Pre-cue interval, same trials | 0.540 | 0.561 |

  So the headline GO number (0.92 at a 2.5 s window) mostly reflects *which block
  a window came from*. Online, rest is same-block rest, where this pipeline sits
  near chance. **Do not use a baseline-block GO model to gate stimulation.** For a
  deployable gate, record calibration data on your own hardware with rest
  interleaved into the same session as the attempts.
- **Label columns:** auto-detection is robust but verify the printed class
  distribution on your first real run; override in `_autodetect_columns` if needed.
