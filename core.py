"""
Core of the GPS-spoofing detector: features, windowing, models, CUSUM, metrics.

Detection idea (from the proposal):
  1. A recurrent network (LSTM or GRU) sees ONLY inertial / magnetometer / barometer
     windows (never GPS) and predicts the next GPS displacement of the drone.
  2. The gap between predicted and reported displacement is the residual.
  3. A CUSUM accumulator adds up small residuals so a slow, stealthy drift becomes visible.

Implementation notes / small refinements of the proposal:
  * Displacements are expressed in the drone's heading frame (along-track, cross-track)
    using the yaw from the orientation sensor (not GPS). This makes the target easy to
    predict and keeps a perpendicular drift at a constant sign.
  * The residual is signed, so a two-sided CUSUM is used on each channel.
"""
import json
import os
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import average_precision_score, roc_auc_score

import sim

torch.set_num_threads(2)

W = 20                      # window length in steps (2 s at 10 Hz)
N_FEAT = 14                 # sensor features per step
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

MODEL_NAMES = {
    "lstm": "LSTM predictor (proposed)",
    "gru": "GRU predictor",
    "iforest": "Isolation Forest",
    "ae": "Dense autoencoder",
    "clf": "Supervised LSTM classifier",
}
CUSUM_MODELS = ("lstm", "gru")


# --------------------------------------------------------------------------- data
def build_features(raw):
    """raw (n,13) -> features (n,14): roll, pitch, sin(yaw), cos(yaw), gyro, acc, mag, baro."""
    return np.column_stack([raw[:, 0], raw[:, 1], np.sin(raw[:, 2]), np.cos(raw[:, 2]), raw[:, 3:]])


def body_disp(gps, yaw):
    """GPS displacement t -> t+1 in the heading frame: (along-track, cross-track) in metres."""
    d = np.diff(gps, axis=0)
    y = yaw[:-1]
    fwd = np.column_stack([np.cos(y), np.sin(y)])
    right = np.column_stack([np.sin(y), -np.cos(y)])
    return np.column_stack([(d * fwd).sum(1), (d * right).sum(1)])


def fit_scalers(train_flights):
    """Scalers are fitted on TRAINING flights only (no leakage)."""
    feat = np.vstack([build_features(f["raw"]) for f in train_flights])
    disp = np.vstack([body_disp(f["gps"], f["raw"][:, 2]) for f in train_flights])
    return {"f_mu": feat.mean(0).tolist(), "f_sd": (feat.std(0) + 1e-8).tolist(),
            "d_mu": disp.mean(0).tolist(), "d_sd": (disp.std(0) + 1e-8).tolist()}


def prepare(raw, gps, scal):
    """
    Cut one flight into windows. Sample k uses steps k..k+W-1 of the sensors; its target is the
    GPS displacement from step k+W-1 to k+W. The decision for sample k is made at step k+W.
      Xw : (N,W,14) standardised sensor windows (no GPS)
      Dw : (N,W,2)  standardised reported GPS displacements over the same window
      y  : (N,2)    reported displacement to be predicted, in metres
    """
    feat = build_features(raw)
    disp = body_disp(gps, raw[:, 2])
    n = len(feat)
    Xw = sliding_window_view(feat, W, axis=0)[: n - W].transpose(0, 2, 1)
    Dw = sliding_window_view(disp, W, axis=0)[: n - W].transpose(0, 2, 1)
    y = disp[W - 1: n - 1]
    Xw = ((Xw - np.array(scal["f_mu"])) / np.array(scal["f_sd"])).astype(np.float32)
    Dw = ((Dw - np.array(scal["d_mu"])) / np.array(scal["d_sd"])).astype(np.float32)
    return np.ascontiguousarray(Xw), np.ascontiguousarray(Dw), y


def win_stats(Xw, Dw):
    """Window statistics (mean and std of every channel) used by the Isolation Forest."""
    A = np.concatenate([Xw, Dw], axis=2)
    return np.concatenate([A.mean(1), A.std(1)], axis=1)


# --------------------------------------------------------------------------- models
class Predictor(nn.Module):
    """LSTM/GRU that maps a window to one vector (next displacement, or a class logit)."""

    def __init__(self, n_in=N_FEAT, kind="lstm", hidden=64, n_out=2):
        super().__init__()
        cell = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = cell(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, n_out))

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1])


class Autoencoder(nn.Module):
    """Dense autoencoder: no memory, sees the flattened window."""

    def __init__(self, d=W * (N_FEAT + 2)):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 96), nn.ReLU(), nn.Linear(96, 24), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(24, 96), nn.ReLU(), nn.Linear(96, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def _batched(fn, x, bs=4096):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(fn(torch.from_numpy(x[i:i + bs])).numpy())
    return np.concatenate(out) if out else np.zeros((0,))


# --------------------------------------------------------------------------- CUSUM
def cusum(z, k):
    """Two-sided CUSUM on each channel of z (N,C). Returns the largest accumulator at each step."""
    n, c = z.shape
    sp, sn = np.zeros(c), np.zeros(c)
    out = np.zeros(n)
    for i in range(n):
        sp = np.maximum(0.0, sp + z[i] - k)
        sn = np.maximum(0.0, sn - z[i] - k)
        out[i] = max(sp.max(), sn.max())
    return out


def cusum_events(z, k, h):
    """Alarm events with reset after each alarm (used to count false alarms and delays)."""
    n, c = z.shape
    sp, sn = np.zeros(c), np.zeros(c)
    ev = []
    for i in range(n):
        sp = np.maximum(0.0, sp + z[i] - k)
        sn = np.maximum(0.0, sn - z[i] - k)
        if max(sp.max(), sn.max()) > h:
            ev.append(i)
            sp[:], sn[:] = 0.0, 0.0
    return ev


def rising_edges(state):
    prev = np.concatenate([[False], state[:-1]])
    return list(np.where(state & ~prev)[0])


# --------------------------------------------------------------------------- detectors
class Detectors:
    """All five trained detectors behind one interface (used by train.py and the Flask app)."""

    def __init__(self, models, scal, thr, iforest):
        self.models, self.scal, self.thr, self.iforest = models, scal, thr, iforest

    @classmethod
    def load(cls, art=ART):
        scal = json.load(open(os.path.join(art, "scalers.json")))
        thr = json.load(open(os.path.join(art, "thresholds.json")))
        models = {}
        for kind in ("lstm", "gru"):
            m = Predictor(kind=kind)
            m.load_state_dict(torch.load(os.path.join(art, f"{kind}.pt"), map_location="cpu"))
            models[kind] = m.eval()
        m = Autoencoder()
        m.load_state_dict(torch.load(os.path.join(art, "ae.pt"), map_location="cpu"))
        models["ae"] = m.eval()
        m = Predictor(n_in=N_FEAT + 2, kind="lstm", n_out=1)
        m.load_state_dict(torch.load(os.path.join(art, "clf.pt"), map_location="cpu"))
        models["clf"] = m.eval()
        return cls(models, scal, thr, joblib.load(os.path.join(art, "iforest.joblib")))

    # -- raw model outputs ------------------------------------------------------
    def residual(self, kind, Xw, y):
        pred = _batched(self.models[kind], Xw)
        pred_m = pred * np.array(self.scal["d_sd"]) + np.array(self.scal["d_mu"])
        return y - pred_m

    def score_windows(self, kind, Xw, Dw):
        """Continuous anomaly score for the non-CUSUM models."""
        if kind == "iforest":
            return -self.iforest.score_samples(win_stats(Xw, Dw))
        x = np.concatenate([Xw, Dw], axis=2)
        if kind == "ae":
            flat = x.reshape(len(x), -1)
            rec = _batched(self.models["ae"], flat)
            return ((rec - flat) ** 2).mean(1)
        logit = _batched(self.models["clf"], x)[:, 0]
        return 1.0 / (1.0 + np.exp(-logit))

    # -- full run on one flight ---------------------------------------------------
    def run(self, kind, raw, gps):
        Xw, Dw, y = prepare(raw, gps, self.scal)
        if kind in CUSUM_MODELS:
            c = self.thr[kind]
            z = self.residual(kind, Xw, y) / np.array(c["sigma"])
            score = cusum(z, c["k"])
            thr = c["h"]
            events = cusum_events(z, c["k"], thr)
            extra = {"z": z}
        else:
            score = self.score_windows(kind, Xw, Dw)
            thr = self.thr[kind]
            events = rising_edges(score > thr)
            extra = {}
        return {"score": score, "thr": float(thr), "state": score > thr,
                "events": events, "t": (np.arange(len(score)) + W) * sim.DT, **extra}

    def latency_ms(self, kind, raw, gps, reps=200):
        """Average time to process one new step (single window) on this CPU."""
        Xw, Dw, y = prepare(raw, gps, self.scal)
        xw, dw = Xw[:1], Dw[:1]
        t0 = time.perf_counter()
        for _ in range(reps):
            if kind in CUSUM_MODELS:
                self.residual(kind, xw, y[:1])
            else:
                self.score_windows(kind, xw, dw)
        return (time.perf_counter() - t0) / reps * 1000


# --------------------------------------------------------------------------- metrics
def summarize(items):
    """
    items: list of dicts {det, lab, onset, plan}; onset is None for attack-free flights.
    Step-level precision/recall/F1 and AUCs use every sample; delay and detection rate use
    attacked flights only; false alarms are counted on every attack-free stretch.
    """
    ys, st, sc = [], [], []
    delays, detected, n_att = [], 0, 0
    fa_events, fa_secs = 0, 0.0
    for it in items:
        det, lab, on = it["det"], it["lab"], it["onset"]
        lab_s = lab[W:].astype(int)
        ys.append(lab_s); st.append(det["state"]); sc.append(det["score"])
        ev = np.array(det["events"], dtype=int) + W
        if on is None:
            fa_events += len(ev)
            fa_secs += len(lab_s) * sim.DT
        else:
            n_att += 1
            fa_events += int((ev < on).sum())
            fa_secs += max(on - W, 0) * sim.DT
            post = ev[ev >= on]
            if len(post):
                detected += 1
                delays.append((post[0] - on) * sim.DT)
    y, s, score = np.concatenate(ys), np.concatenate(st), np.concatenate(sc)
    tp = int(((y == 1) & s).sum()); fp = int(((y == 0) & s).sum()); fn = int(((y == 1) & ~s).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    both = len(set(y.tolist())) == 2
    return {
        "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        "roc_auc": round(float(roc_auc_score(y, score)), 3) if both else None,
        "pr_auc": round(float(average_precision_score(y, score)), 3) if both else None,
        "detection_rate": round(detected / n_att, 3) if n_att else None,
        "mean_delay_s": round(float(np.mean(delays)), 1) if delays else None,
        "false_alarms_per_hour": round(fa_events / (fa_secs / 3600), 1) if fa_secs else None,
        "n_flights": len(items),
    }
