# Sensor-Spoofing Detection for Drones — ANN Lab

Implementation of the model and the Flask application from the project proposal
(Muhammad Ibram ul Haq, Taha Asad — BS(AI)-5A, SZABIST-ISB).

## Run it

```bash
pip install -r requirements.txt
python app.py            # opens on http://127.0.0.1:5000  (trained models are included)
```

To retrain everything from scratch (about 2 minutes on a laptop CPU):

```bash
python train.py          # writes artifacts/ (models, thresholds, results.json, training_curves.png)
python train.py --quick  # 30-second smoke test
```

## What is in the folder

| File | Purpose |
|---|---|
| `sim.py` | Flight simulator + GPS attacks (stand-in for the dataset, see "Important" below) |
| `core.py` | Features, windowing, LSTM/GRU/autoencoder/classifier, CUSUM, metrics |
| `train.py` | Split by flight, train all five models, calibrate k and h, evaluate on held-out flights |
| `app.py`, `templates/index.html` | Flask demo |
| `artifacts/` | Trained weights, scalers, thresholds, `results.json`, training curves, `train_log.txt` |

## How it follows the proposal

* **Detector (Section 5.1).** An LSTM or GRU reads a 2-second window (20 steps at 10 Hz) of orientation, angular velocity,
  linear acceleration, magnetic field and barometric altitude. **GPS is never an input.** It predicts the next GPS
  displacement; the normalised gap (residual) goes into a CUSUM accumulator; an alarm is raised when it crosses `h`.
* **Training on attack-free flights only.** LSTM, GRU, autoencoder and Isolation Forest never see an attack.
  Only the supervised LSTM classifier sees labels (it is the "what would labels buy" reference).
* **Leakage control.** Split by whole flight (60/20/20, balanced across plans); scalers fitted on training flights only;
  `k` and `h` set from attack-free **validation** flights, never from test data.
* **Five models, same test flights, same metrics.** Precision, recall, F1, ROC-AUC, PR-AUC, detection delay,
  false alarms per hour, inference time per step.
* **Unseen attacks.** Besides the dataset-style stealthy drift, step offset, fast ramp and freeze attacks are injected
  into attack-free test flights.

### Two small refinements to the proposal's formula (be ready to explain these in the viva)

1. Displacements are expressed in the drone's **heading frame** (along-track, cross-track) using yaw from the orientation
   sensor. A perpendicular drift then has a constant sign, which is what makes accumulation work, and the target is easier
   to predict.
2. The residual is signed, so the CUSUM is **two-sided** (`S+ = max(0, S+ + z − k)` and `S− = max(0, S− − z − k)`),
   checked on both channels. The proposal's one-sided form assumes a non-negative score.

Also: the prediction target is the 2-D displacement (along-track, cross-track); GPS velocity was left out to keep scope small.

## Results (held-out test flights, simulated data)

12 test flights (4 per plan). Full tables, including per-plan results, are in `artifacts/results.json` and in the app.

| Model | Drift F1 | Detected | Mean delay | False alarms / h | Unseen attacks detected (step / ramp / freeze) |
|---|---|---|---|---|---|
| LSTM predictor | 0.98 | 100% | 2.4 s | 2.2 | 100% / 100% / 100% |
| GRU predictor | 0.98 | 100% | 3.4 s | 2.2 | 100% / 100% / 100% |
| Isolation Forest | 0.03 | 33% | 33.6 s | 52.6 | 25% / 33% / 42% |
| Dense autoencoder | 0.03 | 33% | 16.8 s | 181.2 | 100% / 92% / 42% |
| Supervised LSTM | 1.00 | 100% | 0.7 s | 11.2 | 100% / 100% / **0%** |

What the numbers do and do not say:

* The supervised classifier is the best on the attack it was trained on, but it completely misses the freeze attack, and
  its F1 on the step attack is 0.03 (it fires at the jump, then goes quiet). The CUSUM predictors detect all three unseen attacks. This is the
  proposal's central argument, and it holds on this data.
* Step offset: the CUSUM detectors catch the jump instantly but the alarm decays afterwards, so step-level F1 is 0.64.
  The detector works on *displacements*, so a constant offset that persists is not visible after the jump itself.
  This is a genuine limitation worth stating.
* The Isolation Forest and autoencoder baselines are weak partly because their thresholds (99.9th percentile of validation
  scores) transferred poorly to the test flights.
* False-alarm rates come from very few events (about 2) so treat them as rough.

## Important: simulated data

The real UAV GPS Spoofing Dataset could not be downloaded while building this (GitHub and Google Drive were not reachable from
the build environment). Everything above is therefore on **simulated flights from `sim.py`**: a quadcopter with
straight/curved/random plans, attitude that follows acceleration and drag, noisy IMU/magnetometer/barometer, GPS with 0.05 m white
noise, and a stealthy perpendicular drift of 0.02–0.05 m per update starting 60–120 s into a 180 s flight.

Consequences you should state to your instructor:

* The simulator is much **easier** than real data. For example, results are equally good on the random plan, whereas the dataset paper
  reports random plans as clearly harder. Do not present these numbers as real-world performance.
* To run on the real data, convert each merged flight to the same arrays (the 13 columns in `sim.RAW_COLS` plus GPS x/y in metres at 10 Hz;
  drop the attack label, ground-truth and un-attacked GPS from the inputs), then reuse `core.py` and `train.py` unchanged
  apart from replacing `sim.make_flight`. I did not write that adapter because I could not see the real file schema and did not
  want to guess column names.

## CSV upload format (in the app)

13 sensor columns + `gps_x`, `gps_y`, at 10 Hz (at least 30 rows); optional `attack_label` (0/1). Use
"Download a sample flight" in the app to get a correctly formatted file.

## Not done (as agreed)

Ablations (window length, sensor subsets, with/without past GPS), the leave-one-plan-out experiment, and the full report.
