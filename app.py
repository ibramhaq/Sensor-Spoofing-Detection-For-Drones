"""
Flask demo for the GPS-spoofing detector.

    python train.py      # once, creates artifacts/
    python app.py        # then open http://127.0.0.1:5000

Pages / endpoints
    GET  /                 the demo page
    GET  /api/meta         models, held-out test flights, attack types
    GET  /api/results      evaluation tables produced by train.py
    POST /api/detect       JSON: simulate a flight (+ optional attack) and run a detector
    POST /api/detect_csv   multipart: run a detector on an uploaded CSV flight log
    GET  /api/sample_csv   download a simulated flight as CSV (the upload format)
"""
import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, Response

import core
import sim
from core import Detectors, MODEL_NAMES, W

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
DET = Detectors.load()
SPLIT = json.load(open(os.path.join(core.ART, "split.json")))
RESULTS = json.load(open(os.path.join(core.ART, "results.json")))
CSV_COLS = sim.RAW_COLS + ["gps_x", "gps_y"]


def _plot(kind, res, gps, clean_gps, onset, title):
    """Left: flight path (reported vs true). Right: detector score vs alarm threshold."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    if clean_gps is not None:
        a1.plot(clean_gps[:, 0], clean_gps[:, 1], color="#1f77b4", lw=1.6, label="true path")
    a1.plot(gps[:, 0], gps[:, 1], color="#d62728", lw=1.0, ls="--", label="GPS as reported")
    ev = res["events"]
    first = (ev[0] + W) if len(ev) else None
    if onset is not None:
        a1.scatter(*gps[onset], marker="x", s=70, color="black", zorder=5, label="attack starts")
    if first is not None:
        a1.scatter(*gps[first], marker="o", s=70, facecolors="none", edgecolors="#ff7f0e",
                   linewidths=2, zorder=6, label="first alarm")
    a1.set_aspect("equal", adjustable="datalim")
    a1.set_xlabel("east (m)"); a1.set_ylabel("north (m)"); a1.set_title("Flight path", fontsize=10)
    a1.legend(fontsize=8, loc="best"); a1.grid(alpha=0.25)

    t = res["t"]
    a2.plot(t, res["score"], color="#4c4c9d", lw=1.2, label="detector score")
    a2.axhline(res["thr"], color="#d62728", ls="--", lw=1.2, label="alarm threshold")
    if onset is not None:
        a2.axvline(onset * sim.DT, color="black", ls=":", lw=1.4, label="attack starts")
    if first is not None:
        a2.axvline(first * sim.DT, color="#ff7f0e", lw=1.6, label="first alarm")
    a2.set_xlabel("time (s)")
    a2.set_ylabel("CUSUM statistic S" if kind in core.CUSUM_MODELS else "anomaly score")
    a2.set_title(title, fontsize=10); a2.legend(fontsize=8, loc="upper left"); a2.grid(alpha=0.25)
    if res["score"].max() > 8 * max(res["thr"], 1e-9):
        a2.set_yscale("symlog", linthresh=max(res["thr"], 1e-6))
    a2.set_ylim(bottom=0)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _analyse(kind, raw, gps, clean_gps=None, onset=None):
    res = DET.run(kind, raw, gps)
    ev = res["events"]
    first = (ev[0] + W) if len(ev) else None
    post = [e + W for e in ev if e + W >= (onset or 0)]
    pre = [e + W for e in ev if onset is not None and e + W < onset]
    summary = {
        "model": MODEL_NAMES[kind],
        "steps": int(len(gps)),
        "duration_s": round(len(gps) * sim.DT, 1),
        "alarm_events": len(ev),
        "alarm_raised": bool(len(ev)),
        "first_alarm_s": round(first * sim.DT, 1) if first is not None else None,
        "attack_onset_s": round(onset * sim.DT, 1) if onset is not None else None,
        "detection_delay_s": round((post[0] - onset) * sim.DT, 1) if (onset is not None and post) else None,
        "false_alarms_before_onset": len(pre) if onset is not None else None,
        "position_error_at_first_alarm_m": (
            round(float(np.linalg.norm(gps[first] - clean_gps[first])), 1)
            if (first is not None and clean_gps is not None) else None),
        "position_error_at_end_m": (
            round(float(np.linalg.norm(gps[-1] - clean_gps[-1])), 1) if clean_gps is not None else None),
    }
    if onset is None and clean_gps is not None and np.allclose(gps, clean_gps):
        summary["verdict"] = "ALARM (false alarm)" if ev else "No spoofing detected"
    elif onset is not None:
        summary["verdict"] = ("SPOOFING DETECTED" if post else "MISSED - no alarm after attack start")
    else:
        summary["verdict"] = "SPOOFING FLAGGED" if ev else "No spoofing detected"
    title = f"{MODEL_NAMES[kind]}"
    return summary, _plot(kind, res, gps, clean_gps, onset, title)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/meta")
def meta():
    return jsonify({
        "models": MODEL_NAMES,
        "attacks": {"none": "No attack (clean flight)",
                    "drift": "Stealthy drift (the dataset's attack)",
                    "ramp": "Fast ramp (unseen)",
                    "step": "Step offset (unseen)",
                    "freeze": "Freeze (unseen)"},
        "plans": list(sim.PLANS),
        "test_flights": SPLIT["test"],
        "csv_columns": CSV_COLS,
    })


@app.get("/api/results")
def results():
    return jsonify(RESULTS)


@app.post("/api/detect")
def detect():
    p = request.get_json(force=True)
    kind = p.get("model", "lstm")
    if kind not in MODEL_NAMES:
        return jsonify(error="unknown model"), 400
    plan, seed = p.get("plan", "straight"), int(p.get("seed", 1))
    if plan not in sim.PLANS:
        return jsonify(error="unknown plan"), 400
    attack = p.get("attack", "drift")
    if attack not in sim.ATTACK_KINDS:
        return jsonify(error="unknown attack"), 400
    onset_s = float(p.get("onset_s", 90))
    onset = int(np.clip(onset_s / sim.DT, 50, sim.N_STEPS - 50))
    rate = p.get("rate")
    rate = float(rate) if rate not in (None, "") else None
    fl = sim.make_flight(plan, seed)
    gps, lab, params = sim.attack(fl, attack, onset, rate=rate, seed=seed + 11)
    summary, img = _analyse(kind, fl["raw"], gps, fl["gps"], None if attack == "none" else onset)
    summary["plan"], summary["attack"] = plan, attack
    summary["attack_params"] = {k: v for k, v in params.items() if k not in ("kind", "onset")}
    return jsonify(summary=summary, image=img)


@app.post("/api/detect_csv")
def detect_csv():
    f = request.files.get("file")
    kind = request.form.get("model", "lstm")
    if f is None or kind not in MODEL_NAMES:
        return jsonify(error="send a CSV file and a valid model"), 400
    try:
        df = pd.read_csv(f)
    except Exception as e:                               # noqa: BLE001
        return jsonify(error=f"could not read CSV: {e}"), 400
    missing = [c for c in CSV_COLS if c not in df.columns]
    if missing:
        return jsonify(error=f"missing columns: {missing}", expected=CSV_COLS), 400
    if len(df) < W + 10:
        return jsonify(error=f"need at least {W + 10} rows (10 Hz samples)"), 400
    if df[CSV_COLS].isna().any().any():
        return jsonify(error="CSV contains missing values"), 400
    raw = df[sim.RAW_COLS].to_numpy(float)
    gps = df[["gps_x", "gps_y"]].to_numpy(float)
    onset = None
    if "attack_label" in df.columns and df["attack_label"].astype(int).any():
        onset = int(np.argmax(df["attack_label"].astype(int).to_numpy()))
    summary, img = _analyse(kind, raw, gps, None, onset)
    summary["source"] = "uploaded CSV"
    return jsonify(summary=summary, image=img)


@app.get("/api/sample_csv")
def sample_csv():
    plan = request.args.get("plan", "curved")
    seed = int(request.args.get("seed", 5))
    attack = request.args.get("attack", "drift")
    if plan not in sim.PLANS or attack not in sim.ATTACK_KINDS:
        return jsonify(error="bad plan or attack"), 400
    fl = sim.make_flight(plan, seed)
    gps, lab, _ = sim.attack(fl, attack, 900, seed=seed + 11)
    df = pd.DataFrame(fl["raw"], columns=sim.RAW_COLS)
    df["gps_x"], df["gps_y"] = gps[:, 0], gps[:, 1]
    df["attack_label"] = lab.astype(int)
    return Response(df.round(5).to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=sample_{plan}_{attack}.csv"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
