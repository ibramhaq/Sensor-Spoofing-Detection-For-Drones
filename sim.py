"""
Quadcopter flight simulator used as a stand-in for the UAV GPS Spoofing Dataset.

The simulator produces data with the same structure as the proposal describes (straight / curved / random plans, IMU,
magnetometer, barometer, GPS, a stealthy perpendicular-drift attack) so that the whole
pipeline can be built, run and demonstrated end to end.

Everything is kept at 10 Hz (the proposal's starting resample rate).

To use the real data instead, produce the same arrays (see `RAW_COLS`) for each flight
and feed them into the pipeline -- nothing downstream depends on the simulator.
"""
import numpy as np
from scipy.signal import lfilter

DT = 0.1                 # seconds per step (10 Hz)
N_STEPS = 1800           # 180 s per flight
PLANS = ("straight", "curved", "random")
G = 9.81
GPS_NOISE = 0.05         # metres, white noise on every GPS position fix

# Raw sensor columns (GPS is stored separately and is NEVER a model input)
RAW_COLS = ["roll", "pitch", "yaw",
            "gyro_x", "gyro_y", "gyro_z",
            "acc_x", "acc_y", "acc_z",
            "mag_x", "mag_y", "mag_z",
            "baro_alt"]

ATTACK_KINDS = ("none", "drift", "step", "ramp", "freeze")


def _ou(rng, n, tau, std):
    """Ornstein-Uhlenbeck style smooth random signal with given time constant and std."""
    a = np.exp(-DT / tau)
    e = rng.normal(0.0, std * np.sqrt(1 - a * a), n)
    return lfilter([1.0], [1.0, -a], e)


def make_flight(plan, seed):
    """Simulate one attack-free flight. Returns a dict of arrays."""
    assert plan in PLANS
    rng = np.random.default_rng(seed)
    n = N_STEPS
    t = np.arange(n) * DT
    cruise = rng.uniform(4.0, 8.0)
    r = np.clip(t / 8.0, 0, 1)
    ramp = r * r * (3 - 2 * r)                      # smooth take-off to cruise speed
    h0 = rng.uniform(-np.pi, np.pi)
    phase = rng.uniform(0, 2 * np.pi)

    if plan == "straight":
        speed = cruise * ramp * (1 + 0.10 * np.sin(2 * np.pi * t / rng.uniform(15, 35) + phase))
        heading = np.full(n, h0)
    elif plan == "curved":
        speed = cruise * ramp * (1 + 0.06 * np.sin(2 * np.pi * t / rng.uniform(15, 35) + phase))
        radius = rng.uniform(40, 120)
        w0 = cruise / radius * rng.choice([-1, 1])
        w = w0 * (1 + 0.3 * np.sin(2 * np.pi * t / rng.uniform(30, 60) + phase))
        heading = h0 + np.cumsum(w) * DT
    else:  # random
        speed = np.clip(cruise * ramp + _ou(rng, n, 6.0, 0.8) * ramp, 0.5, None)
        heading = h0 + np.cumsum(_ou(rng, n, 4.0, 0.12)) * DT

    cosh, sinh = np.cos(heading), np.sin(heading)
    vel = np.column_stack([speed * cosh, speed * sinh])            # east, north
    pos = np.vstack([np.zeros((1, 2)), np.cumsum(vel[:-1] * DT, axis=0)])
    acc = np.gradient(vel, DT, axis=0)

    fwd = np.column_stack([cosh, sinh])
    right = np.column_stack([sinh, -cosh])
    a_f = (acc * fwd).sum(1)
    a_r = (acc * right).sum(1)

    # Attitude: a quadcopter tilts to accelerate and to balance aerodynamic drag,
    # so pitch/roll carry information about speed and acceleration.
    pitch = -(a_f + 0.25 * speed) / G
    roll = a_r / G
    alt = 30.0 + _ou(rng, n, 10.0, 0.6)
    a_z = np.gradient(np.gradient(alt, DT), DT)

    psi = np.pi / 2 - heading                                       # heading from north
    mag = np.column_stack([0.22 * np.cos(psi), -0.22 * np.sin(psi), np.full(n, -0.40)])

    nz = lambda s, shape=n: rng.normal(0, s, shape)
    raw = np.column_stack([
        roll + nz(0.0015), pitch + nz(0.0015),
        np.angle(np.exp(1j * heading)) + nz(0.003),
        np.gradient(roll, DT) + nz(0.003), np.gradient(pitch, DT) + nz(0.003),
        np.gradient(heading, DT) + nz(0.003),
        a_f + nz(0.05), a_r + nz(0.05), a_z + nz(0.05),
        mag[:, 0] + nz(0.004), mag[:, 1] + nz(0.004), mag[:, 2] + nz(0.004),
        alt + nz(0.1),
    ])
    gps = pos + rng.normal(0, GPS_NOISE, (n, 2))
    return {"plan": plan, "seed": int(seed), "raw": raw, "gps": gps, "heading": heading}


def attack(flight, kind, onset, rate=None, side=None, seed=0):
    """
    Apply a GPS attack to a clean flight.

    kind   : drift  - the dataset's stealthy attack: offset grows by `rate` metres per step,
                      perpendicular to the heading
             ramp   - same, but much faster (unseen-attack test)
             step   - position jumps by a fixed offset at `onset` (unseen-attack test)
             freeze - GPS output holds its value from `onset` (unseen-attack test)
             none   - returns the clean GPS
    onset  : step index at which the attack starts
    Returns (gps, labels, params). labels[i] is True once the attack is active.
    """
    rng = np.random.default_rng(seed)
    gps = flight["gps"].copy()
    n = len(gps)
    lab = np.zeros(n, dtype=bool)
    params = {"kind": kind, "onset": int(onset)}
    if kind == "none":
        params["onset"] = None
        return gps, lab, params
    lab[onset:] = True
    h = flight["heading"]
    if kind in ("drift", "ramp"):
        if rate is None:
            rate = rng.uniform(0.02, 0.05) if kind == "drift" else 0.20
        if side is None:
            side = rng.choice([-1, 1])
        inc = side * rate * np.column_stack([np.sin(h), -np.cos(h)])
        inc[: onset + 1] = 0.0
        gps = gps + np.cumsum(inc, axis=0)
        params.update(rate=float(rate), side=int(side))
    elif kind == "step":
        ang = rng.uniform(0, 2 * np.pi)
        size = 8.0 if rate is None else float(rate)
        gps[onset:] += size * np.array([np.cos(ang), np.sin(ang)])
        params.update(size=size)
    elif kind == "freeze":
        gps[onset:] = gps[onset]
    else:
        raise ValueError(f"unknown attack kind {kind}")
    return gps, lab, params
