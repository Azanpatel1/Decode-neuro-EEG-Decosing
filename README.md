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

## 2. Pick a data source

Two sources, and **which one you use matters for the GO decoder**:

| Source | Flag | Use for |
|---|---|---|
| Your own same-session recording | `--calibration` | **the GO decoder** (gates tVNS) |
| ds003626 (public) | `--data` | the word decoder, and offline research |

The GO task on ds003626 is confounded — its only rest is a separate baseline
block (see section 9). For a GO decoder you intend to actually gate stimulation
with, record your own calibration data where rest is interleaved with attempts in
one session.

### 2a. Record your own calibration data (for the GO decoder)

```bash
pip install brainflow
python calibrate.py --port /dev/cu.usbserial-XXXX --subject 1 \
    --channel-names "F7,F8,FC5,FC6,FT7,FT8,T7,T8,C3,C4,CP5,CP6,P7,P8,Cz,Pz"
```

Cues ~40 overt speech attempts and ~40 rest trials, **randomly interleaved** (runs
of the same cue are capped at 3) so slow drift cannot line up with class identity.
Takes about 7 minutes. On `SPEAK ALOUD: <word>` say the word once; on `REST` sit
still and silent. Samples are stored raw at the board's native rate, so the shared
preprocessing applies at load time and the model trains at 125 Hz — which also
means the online path needs no resampling (`--no-resample`).

```bash
python run.py --calibration 'calib/*.npz' --task go
```

One recording gives you the within-subject number; cross-subject LOSO needs
several subjects.

You can also do all of this from the dashboard's **Calibrate** tab (section 7b),
which presents the cues in the browser and records how late each one actually
appeared, then train from it on the **Train** tab.

### 2b. Get ds003626 (for the word decoder)

```bash
pip install openneuro-py
openneuro-py download --dataset ds003626 --target-dir ds003626
```

Only the `derivatives/` tree is needed (~7 GB). Nothing runs without a real
source: `run.py` requires `--data` or `--calibration`, and `load_dataset` raises
rather than falling back to generated data.

## 3. Train the decoders

```bash
# 4-class imagined-word decoder (tracking / display only):
python run.py --data ./ds003626 --task word --condition inner

# GO decoder from ds003626 -- offline research only, see the caveat in section 9:
python run.py --data ./ds003626 --task go --condition overt_scaffold
```

Ablation showing why Riemannian Alignment matters (cross-subject score should
drop):

```bash
python run.py --data ./ds003626 --task word --no-align
```

### What the real data actually gives you

Measured here on ds003626, leakage-free (LOSO, 10 subjects, 16-channel
low-density montage), with permutation baselines:

| Decoder | LOSO balanced acc | Chance | Permutation | Verdict |
|---|---|---|---|---|
| Word (4-class, inner) | 0.266 | 0.250 | p = 0.50 | **at chance** |
| GO (baseline-block rest) | 0.920 | 0.500 | p = 0.005 | **confounded, see §9** |

Neither is currently a working decoder. The word result matches the literature on
imagined-word identity and is exactly why the word decoder never gates
stimulation (Invariant C). The GO result is a block artifact, not a decoder.

The loader reads the derivatives (`*_eeg-epo.fif` + `*_events.dat`),
**auto-detects** which label column is the word class ({0,1,2,3}) and which is
the condition ({0,1,2}), band-passes 1–40 Hz, crops to the action window
(1.0–3.5 s), and selects a low-density wearable-like montage. On the first run,
**check the printed class distribution** to confirm the labels decoded correctly.

## 4. What each run produces (in `./outputs`)

| File | Contents |
|---|---|
| `model_<task>.joblib` | the fitted, deployable pipeline + config + label names |
| `metrics_<task>.json` | cross-subject / within-subject balanced accuracy, permutation p-value, latency |
| `confusion_matrix_<task>.png` | normalised cross-subject confusion |
| `latency_hist_<task>.png` | closed-loop decision latency vs. the VNS pairing budget |
| `session_log.jsonl` | dashboard audit trail: mode changes, arm/disarm, fires |
| `impedance.json` | the most recent contact check, reused by the head map |

Filenames carry the task so training a GO decoder cannot overwrite the word
decoder's metrics.

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
| Preprocess (band-pass, crop) | `preprocessing.py` | shared by training and the live path |
| Low-density montage | `data_loader.resolve_montage_indices` | picks 10-20 sites by 3D position, not by label |
| Covariance | `pipeline.Covariances` / `FilterBankCovariances` | OAS-regularised |
| **Riemannian Alignment** | `pipeline.RiemannianAlignment` | per-domain recentering; ≡ `pyriemann.transfer.TLCenter` |
| Tangent space | `pipeline.TangentSpace` | |
| Classifier | `pipeline._make_classifier` | LDA (default), logreg, SVM, MDM |
| Leakage-free eval | `evaluate.py` | LOSO + within-subject + permutation chance |
| Closed-loop GO gate | `realtime.RealTimeDecoder` | GO threshold + online recentering |
| Latency benchmark | `realtime.benchmark_latency` | proves compute fits the pairing window |
| Training entry point | `training.train` | one path shared by `run.py` and the dashboard |
| Hardware ownership + ARM | `session.SessionManager` | exclusive board modes, audit log |
| Background jobs | `jobs.JobRunner` | training off the request thread, with progress |
| Model registry + verdicts | `models.py` | is this bundle fit to gate stimulation? |
| Board registry / probing | `boards.py` | port enumeration, board IDs, error explanations |
| Live display plumbing | `live.py` | `FrameHub`, acquisition thread, word readout |

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

## 7b. The dashboard (full control plane)

The dashboard is the whole workflow in a browser: find the board, check electrode
contact, record calibration, train a decoder, assign it, run the closed loop, and
watch it. It boots with **nothing connected** and acquires hardware only when you
ask it to, so `--port` and `--model` are just conveniences that pre-fill the form.

```bash
pip install "fastapi>=0.110,<0.120" "uvicorn>=0.27,<0.35" "websockets>=12"
pip install pyserial   # optional: nicer port names; otherwise ports are globbed

python dashboard.py                      # then do everything in the browser
```

Open http://127.0.0.1:8765. Five tabs:

| Tab | What you can do |
|---|---|
| **Monitor** | live traces, GO state and `p(go)`, the decoded window behind the last GO event, word readout, `p(go)` and latency history, electrode head map |
| **Hardware** | enumerate serial ports, probe a board (reports real EEG channel count and sample rate), set the channel labels, start/stop the closed loop, run a contact check with real ADS1299 impedance, **ARM / DISARM** stimulation |
| **Calibrate** | record same-session interleaved attempt/rest with cues presented in the browser; shows measured cue display lag and lists past recordings |
| **Train** | run the same training pipeline as `run.py` as a background job, with live progress, the pipeline's own log lines, metrics and plots |
| **Models** | list every bundle in `outputs/` with its task, montage, rate, LOSO score and **gating verdict**; import a `.joblib`; assign it to the GO or word slot |

The serial port is exclusive, so one activity owns the board at a time: probing,
a contact check, calibration and decoding cannot overlap. A request that arrives
while another holds the board is refused (HTTP 409) rather than queued, and the
tab tells you what to stop first. Every mode change, arm, disarm and fire is
appended to `outputs/session_log.jsonl`.

Every trace it draws is measured from the board. Because there is no simulated
source, an empty or flatlined chart always means a real acquisition problem
(board off, port busy, electrodes not seated) — never synthetic filler. If
acquisition can't start, a red banner explains what to fix.

### Stimulation is behind an ARM switch

Running the loop and firing a nerve stimulator are separate decisions. The loop
always starts **DISARMED**: it decodes, displays, and logs GO events without
triggering anything. Arming requires all of:

- the closed loop already running,
- the server bound to loopback (`--allow-remote-arm` to override; the default host
  is `127.0.0.1`),
- the GO model's filename typed back exactly, so it cannot be a stray click,
- an explicit acknowledgement when that model is **not validated for gating**.

It auto-disarms on stop, on any GO-model change or rewrite, and after 30 minutes.

The Monitor tab therefore counts **GO events** and **stimulations** separately: a
threshold crossing while disarmed is a GO event and is drawn amber, while a real
stimulation is drawn red. The display never implies the patient was stimulated
when they were not.

`models.py` flags a bundle trained from ds003626 with a GO task as *not validated
for gating*, because its rest class is a separate baseline block (section 9). That
is exactly the `outputs/model_go.joblib` you get from section 3, so arming it
takes a deliberate acknowledgement.

### The word decoder cannot gate, structurally

It runs in the `on_frame` observer, *after* `run_loop` has already acted on its GO
decision, so it has no path into the trigger. The API additionally refuses to put
a `task == "word"` bundle in the GO slot at all (Invariant C enforced server-side,
not by convention). It is decoded only on GO frames, since word identity during
rest is meaningless.

### Calibration cue timing is measured, not assumed

The server plans the entire cue timeline as absolute times and sends it to the
browser; the browser syncs clocks against `/api/time` over several round trips
(keeping the minimum-RTT sample), pre-schedules every cue locally so no per-cue
network jitter exists, and reports back when each cue was actually painted. An
onset within tolerance of plan replaces the planned onset; one that is absurd
(before its cue, or very late) keeps the planned onset and is flagged. Per-trial
`display_lag_s` goes into the `.npz` and the tab shows the median and worst lag.

### Training goes through one code path

`run.py` and the Train tab both call `eeg_tvns.training.train()`, so a model
trained in the browser is the same model the CLI would produce. Training is
refused while the loop is decoding — a multi-minute fit must never compete for CPU
with the decoder that gates stimulation.

Extra flags: `--word-model`, `--host`, `--web-port`, `--publish-hz` (UI refresh
rate; the decoder always runs at its native `hop_s`), `--allow-remote-arm`,
`--start` (start the loop immediately; still disarmed). All acquisition flags are
supported.

### Checking it without hardware

```bash
python tools/verify_control_plane.py
```

Exercises the mode state machine and its 409s, every ARM refusal, Invariant C at
the API, the cue timeline and lag arithmetic, the job runner, model verdicts, and
the guarantee that nothing in the package can fabricate a signal. Board-dependent
paths run against a transport double that emits an obvious ramp, never anything
EEG-like. No network, no hardware.

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
  near chance. **Do not use a baseline-block GO model to gate stimulation.** Use
  `calibrate.py` (section 2a) instead, which interleaves rest with attempts in one
  session so both classes share block, impedance and arousal context.

  Note that the **permutation test does not catch this**: it shuffles labels, so
  it detects "is there any learnable structure" — and block identity *is*
  learnable structure. Passing at p=0.005 says nothing about *what* was learned.
  A same-block control is the check that matters.
- **The dashboard can arm a stimulator.** That is the point of the ARM switch, the
  loopback restriction, the typed confirmation and the gating verdict (section 7b):
  the default state is that nothing fires. Before arming anything on a person,
  have a GO model trained from your own calibration data and reported against the
  0.54 same-block control, and keep `fire_tvns()` wired to a device you can cut
  power to.
- **Label columns:** auto-detection is robust but verify the printed class
  distribution on your first real run; override in `_autodetect_columns` if needed.
