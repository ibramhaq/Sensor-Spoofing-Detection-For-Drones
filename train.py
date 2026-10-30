"""
Train and evaluate all five detectors.

    python train.py            # full run (about 60 simulated flights)
    python train.py --quick    # tiny smoke test (a minute or two)

Protocol (follows the proposal):
  * split by WHOLE flight, 60/20/20, balanced across flight plans
  * scalers fitted on training flights only
  * LSTM / GRU / autoencoder / Isolation Forest see attack-free flights only
  * the supervised LSTM classifier is the only model that sees attack labels
  * CUSUM allowance k and alarm threshold h come from attack-free VALIDATION flights only
  * the same test flights and the same metrics for every model
"""
import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

import sim
import core
from core import W, ART, Predictor, Autoencoder, Detectors, prepare, win_stats, cusum, summarize

SEED = 42
ATTACKS = ("drift", "step", "ramp", "freeze")


def set_seed(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ data
def build_split(n_per_plan):
    rng = np.random.default_rng(0)
    split = {"train": [], "val": [], "test": []}
    for pi, plan in enumerate(sim.PLANS):
        seeds = [1000 + pi * 1000 + i for i in range(n_per_plan)]
        rng.shuffle(seeds)
        n_tr, n_va = round(0.6 * n_per_plan), round(0.2 * n_per_plan)
        parts = {"train": seeds[:n_tr], "val": seeds[n_tr:n_tr + n_va], "test": seeds[n_tr + n_va:]}
        for name, ss in parts.items():
            split[name] += [{"seed": int(s), "plan": plan} for s in ss]
    return split


def attacked_copy(fl, kind):
    """Deterministic attack for a flight: onset 60-120 s, parameters drawn from the flight seed."""
    rng = np.random.default_rng(fl["seed"] + 7)
    onset = int(rng.integers(600, 1200))
    gps, lab, params = sim.attack(fl, kind, onset, seed=fl["seed"] + 11)
    return gps, lab, onset


# ------------------------------------------------------------------ training
def fit(model, x, y, xv, yv, loss_fn, name, epochs, bs=256, lr=2e-3, patience=4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, y, xv, yv = map(torch.from_numpy, (x, y, xv, yv))
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x))
        tr = 0.0
        for i in range(0, len(x), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
            tr += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            va = sum(loss_fn(model(xv[i:i + 4096]), yv[i:i + 4096]).item() * len(xv[i:i + 4096])
                     for i in range(0, len(xv), 4096)) / len(xv)
        tr /= len(x)
        hist.append((tr, va))
        log(f"  {name} epoch {ep:2d}  train {tr:.4f}  val {va:.4f}")
        if va < best - 1e-5:
            best, bad = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"  {name}: early stop (best val {best:.4f})")
                break
    model.load_state_dict(best_state)
    model.eval()
    return hist


def stack(flights, scal, attacked=None):
    """Windows for a list of flights; `attacked` = attack kind to apply (or None for clean GPS)."""
    X, D, Y, L = [], [], [], []
    for fl in flights:
        if attacked:
            gps, lab, _ = attacked_copy(fl, attacked)
        else:
            gps, lab = fl["gps"], np.zeros(len(fl["gps"]), bool)
        Xw, Dw, y = prepare(fl["raw"], gps, scal)
        X.append(Xw); D.append(Dw); Y.append(y); L.append(lab[W:].astype(np.float32))
    return np.concatenate(X), np.concatenate(D), np.concatenate(Y), np.concatenate(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_per_plan = 10 if args.quick else 20
    epochs = 2 if args.quick else 20
    os.makedirs(ART, exist_ok=True)
    set_seed()
    t_start = time.time()

    # ---- data
    split = build_split(n_per_plan)
    flights = {name: [sim.make_flight(f["plan"], f["seed"]) for f in lst] for name, lst in split.items()}
    log("flights  train/val/test:", {k: len(v) for k, v in flights.items()})
    scal = core.fit_scalers(flights["train"])
    d_sd, d_mu = np.array(scal["d_sd"]), np.array(scal["d_mu"])

    # ---- attack-free windows
    Xtr, Dtr, Ytr, _ = stack(flights["train"], scal)
    Xva, Dva, Yva, _ = stack(flights["val"], scal)
    Ytr_s = ((Ytr - d_mu) / d_sd).astype(np.float32)
    Yva_s = ((Yva - d_mu) / d_sd).astype(np.float32)
    log("training windows (attack-free):", Xtr.shape, " validation:", Xva.shape)

    history = {}
    models = {}
    mse = nn.MSELoss()

    # ---- LSTM / GRU predictors (attack-free flights only)
    for kind in ("lstm", "gru"):
        set_seed()
        m = Predictor(kind=kind)
        log(f"training {kind.upper()} predictor")
        history[kind] = fit(m, Xtr, Ytr_s, Xva, Yva_s, mse, kind.upper(), epochs)
        models[kind] = m
        torch.save(m.state_dict(), os.path.join(ART, f"{kind}.pt"))

    # ---- dense autoencoder (attack-free flights only)
    set_seed()
    sub = slice(None, None, 2)
    A_tr = np.concatenate([Xtr, Dtr], 2)[sub].reshape(len(Xtr[sub]), -1)
    A_va = np.concatenate([Xva, Dva], 2).reshape(len(Xva), -1)
    ae = Autoencoder()
    log("training dense autoencoder")
    history["ae"] = fit(ae, A_tr, A_tr, A_va, A_va, mse, "AE", epochs)
    models["ae"] = ae
    torch.save(ae.state_dict(), os.path.join(ART, "ae.pt"))

    # ---- Isolation Forest (attack-free flights only)
    log("fitting Isolation Forest")
    iso = IsolationForest(n_estimators=100, random_state=SEED, n_jobs=2)
    iso.fit(win_stats(Xtr[::4], Dtr[::4]))
    joblib.dump(iso, os.path.join(ART, "iforest.joblib"))

    # ---- supervised LSTM classifier (the ONLY model that sees attack labels)
    Xa, Da, _, La = stack(flights["train"], scal, attacked="drift")
    Xc = np.concatenate([np.concatenate([Xtr, Dtr], 2), np.concatenate([Xa, Da], 2)])[::2]
    Lc = np.concatenate([np.zeros(len(Xtr), np.float32), La])[::2][:, None]
    Xav, Dav, _, Lav = stack(flights["val"], scal, attacked="drift")
    Xcv = np.concatenate([np.concatenate([Xva, Dva], 2), np.concatenate([Xav, Dav], 2)])
    Lcv = np.concatenate([np.zeros(len(Xva), np.float32), Lav])[:, None]
    set_seed()
    clf = Predictor(n_in=core.N_FEAT + 2, kind="lstm", n_out=1)
    log("training supervised LSTM classifier")
    history["clf"] = fit(clf, Xc, Lc, Xcv, Lcv, nn.BCEWithLogitsLoss(), "CLF", epochs)
    models["clf"] = clf
    torch.save(clf.state_dict(), os.path.join(ART, "clf.pt"))

    # ---- calibration on attack-free validation flights only
    thr = {}
    for kind in ("lstm", "gru"):
        pred = core._batched(models[kind], Xva) * d_sd + d_mu
        sigma = (Yva - pred).std(0)
        k = 0.25
        smax = 0.0
        for fl in flights["val"]:
            Xw, Dw, y = prepare(fl["raw"], fl["gps"], scal)
            r = (y - (core._batched(models[kind], Xw) * d_sd + d_mu)) / sigma
            smax = max(smax, cusum(r, k).max())
        thr[kind] = {"k": k, "h": float(1.2 * smax), "sigma": sigma.tolist()}
        log(f"{kind}: residual sigma {sigma.round(4)}  k={k}  h={thr[kind]['h']:.2f}")
    thr["iforest"] = float(np.percentile(-iso.score_samples(win_stats(Xva, Dva)), 99.9))
    ae_va = ((core._batched(ae, A_va) - A_va) ** 2).mean(1)
    thr["ae"] = float(np.percentile(ae_va, 99.9))
    thr["clf"] = 0.5
    json.dump(scal, open(os.path.join(ART, "scalers.json"), "w"))
    json.dump(thr, open(os.path.join(ART, "thresholds.json"), "w"), indent=1)
    json.dump(split, open(os.path.join(ART, "split.json"), "w"), indent=1)

    # ---- training curves
    fig, axes = plt.subplots(1, len(history), figsize=(3.2 * len(history), 2.8))
    for ax, (name, h) in zip(np.atleast_1d(axes), history.items()):
        ax.plot([a for a, _ in h], label="train"); ax.plot([b for _, b in h], label="validation")
        ax.set_title(core.MODEL_NAMES[name].split(" (")[0], fontsize=9); ax.set_xlabel("epoch")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(ART, "training_curves.png"), dpi=130); plt.close(fig)

    # ---- evaluation on held-out test flights, via the same load path the app uses
    log("evaluating on held-out test flights")
    det = Detectors.load()
    test = flights["test"]
    results = {"overall": {}, "per_plan": {}, "latency_ms": {}}
    sets = {}
    for kind in ATTACKS:
        rows = []
        for fl in test:
            gps, lab, onset = attacked_copy(fl, kind)
            rows.append((fl, gps, lab, onset))
        sets[kind] = rows
    clean_rows = [(fl, fl["gps"], np.zeros(len(fl["gps"]), bool), None) for fl in test]

    for name in core.MODEL_NAMES:
        t0 = time.time()
        clean_items = [{"det": det.run(name, fl["raw"], g), "lab": lab, "onset": None, "plan": fl["plan"]}
                       for fl, g, lab, _ in clean_rows]
        results["overall"][name] = {}
        for kind in ATTACKS:
            items = [{"det": det.run(name, fl["raw"], g), "lab": lab, "onset": on, "plan": fl["plan"]}
                     for fl, g, lab, on in sets[kind]]
            results["overall"][name][kind] = summarize(items + clean_items)
            if kind == "drift":
                results["per_plan"][name] = {
                    p: summarize([i for i in items + clean_items if i["plan"] == p]) for p in sim.PLANS}
        fl0 = test[0]
        results["latency_ms"][name] = round(det.latency_ms(name, fl0["raw"], fl0["gps"]), 2)
        d = results["overall"][name]["drift"]
        log(f"  {name:8s} drift: F1 {d['f1']}  detected {d['detection_rate']}  delay {d['mean_delay_s']}s  "
            f"FA/h {d['false_alarms_per_hour']}   ({time.time() - t0:.0f}s)")
    results["meta"] = {
        "n_flights": {k: len(v) for k, v in flights.items()}, "window_steps": W, "rate_hz": 1 / sim.DT,
        "data": "simulated (see sim.py)", "quick_run": bool(args.quick),
        "attack_params": "drift 0.02-0.05 m/step, ramp 0.20 m/step, step 8 m, onset 60-120 s",
    }
    json.dump(results, open(os.path.join(ART, "results.json"), "w"), indent=1)
    log(f"done in {(time.time() - t_start) / 60:.1f} min -> artifacts/results.json")


if __name__ == "__main__":
    main()
