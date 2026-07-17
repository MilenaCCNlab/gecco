"""Seeded per-participant fitting, mirroring gecco run_fit's protocol
(L-BFGS-B, uniform random starts within bounds, n_starts=10) but fully
reproducible via per-(tag, pid) seeds."""
import hashlib
import math

import numpy as np
from scipy.optimize import minimize

from ..loading import exec_model, participant_inputs

N_STARTS = 10


def bic(nll, k, n):
    return math.log(n) * k + 2.0 * nll


def seed_for(tag, pid):
    return int(hashlib.md5(("%s:%d" % (tag, pid)).encode()).hexdigest()[:8], 16)


def fit_participant(func, inputs, bounds, seed, n_starts=N_STARTS):
    rng = np.random.default_rng(seed)

    def objective(x):
        try:
            v = float(func(*inputs, x))
        except Exception:
            return 1e10
        return v if np.isfinite(v) else 1e10

    best_nll, best_x = np.inf, None
    for _ in range(n_starts):
        x0 = [rng.uniform(lo, hi) for lo, hi in bounds]
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        if res.fun < best_nll:
            best_nll, best_x = float(res.fun), [float(v) for v in res.x]
    return {"nll": best_nll, "params": best_x, "n_starts": n_starts}


def fit_model_on_pids(source_or_func, target, pids, bounds, tag,
                      func_name="cognitive_model", n_starts=N_STARTS):
    func = (exec_model(source_or_func, func_name)
            if isinstance(source_or_func, str) else source_or_func)
    out = {}
    for pid in pids:
        inputs, n = participant_inputs(target, pid)
        seed = seed_for(tag, pid)
        res = fit_participant(func, inputs, bounds, seed, n_starts)
        out[pid] = {"nll": res["nll"], "bic": bic(res["nll"], len(bounds), n),
                    "params": res["params"], "seed": seed}
    return out
