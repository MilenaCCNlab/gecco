"""Behavioral-equivalence verification of library rewrites against originals.

Per participant, four independent checks (plus a WARN-only BIC cross-check):

1. fitted   — NLL at the originally fitted parameters (hard gate, |diff| < tol;
              bitwise equality recorded, and expected given the float-exact
              rewrite policy).
2. probes   — K seeded random parameter vectors inside the docstring bounds.
3. trace    — AST-instrumented trial-level p_choice arrays compared on both
              sides at the fitted params; mismatches report the first
              divergent trial/stage.
4. synthetic— R seeded random datasets (binary sequences for task columns,
              real covariate values otherwise), NLL compared at fitted params
              and one probe vector.

The scalar checks call the participant module imported the normal way
(proving the file works as a real module); the trace check re-execs
instrumented source.

`--baseline` verifies originals against themselves — a harness self-test run
before any rewriting exists, which also snapshots goldens.json.
"""

import json
import math
import time
from typing import Dict, List, Optional

import numpy as np

from .config import BINARY_SEQUENCE_COLUMNS, Target
from . import loading
from . import instrument

FITTED_TOL = 1e-6
PROBE_RTOL = 1e-9
PROBE_ATOL = 1e-6
TRACE_ATOL = 1e-9
BIC_WARN_TOL = 1e-3
SEED_BASE = 20260716


def _nll_close(a: float, b: float, rtol: float = PROBE_RTOL, atol: float = PROBE_ATOL) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return bool(np.isclose(a, b, rtol=rtol, atol=atol))


def _probe_vectors(bounds: List[tuple], rng: np.random.Generator, k: int) -> List[np.ndarray]:
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return [lo + rng.random(len(bounds)) * (hi - lo) for _ in range(k)]


def _synthetic_inputs(
    target: Target,
    real_inputs: List[np.ndarray],
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Random binary sequences for task columns; real values for covariates."""
    synth = []
    for col, real in zip(target.input_columns, real_inputs):
        if col in BINARY_SEQUENCE_COLUMNS:
            synth.append(rng.integers(0, 2, size=len(real), dtype=np.int64))
        else:
            synth.append(real.copy())
    return synth


def verify_participant(
    target: Target,
    pid: int,
    probes: int = 20,
    synthetic: int = 3,
    baseline: bool = False,
) -> dict:
    rec: dict = {"pid": pid}

    code = loading.load_original_code(target, pid)
    func_name, func_args = loading.function_name_and_args(code)
    rec["orig_function"] = func_name
    rec["signature_ok"] = len(func_args) == target.n_model_args
    if not rec["signature_ok"]:
        rec["signature"] = func_args

    f_orig = loading.exec_model(code, func_name)
    unpack = loading.extract_unpack_names(code)
    header, fitted = loading.load_fitted_params(target, pid)
    rec["param_names"] = unpack
    rec["param_order_ok"] = header == unpack
    if not rec["param_order_ok"]:
        rec["param_names_csv"] = header

    inputs, n_rows = loading.participant_inputs(target, pid)
    rec["n_trials"] = n_rows

    if baseline:
        f_lib = f_orig
        lib_source, lib_func_name = code, func_name
    else:
        module = loading.import_participant_module(target, pid)
        f_lib = module.cognitive_model
        lib_source = loading.load_participant_source(target, pid)
        lib_func_name = "cognitive_model"

    # -- check 1: fitted params ------------------------------------------
    nll_orig = float(f_orig(*inputs, fitted))
    nll_lib = float(f_lib(*inputs, fitted))
    rec["orig_nll"] = nll_orig
    rec["lib_nll"] = nll_lib
    rec["fitted_diff"] = abs(nll_orig - nll_lib)
    rec["bitwise_equal"] = nll_orig == nll_lib
    fitted_ok = rec["fitted_diff"] < FITTED_TOL

    # -- check 2: seeded random probes ------------------------------------
    bounds_map = loading.parse_bounds(code, unpack)
    bounds = [bounds_map[name] for name in unpack]
    rng = np.random.default_rng(SEED_BASE + pid)
    probe_thetas = _probe_vectors(bounds, rng, probes)
    probes_passed = 0
    max_probe_diff = 0.0
    probe_nlls = []
    for theta in probe_thetas:
        a = float(f_orig(*inputs, theta))
        b = float(f_lib(*inputs, theta))
        probe_nlls.append(a)
        if _nll_close(a, b):
            probes_passed += 1
        if math.isfinite(a) and math.isfinite(b):
            max_probe_diff = max(max_probe_diff, abs(a - b))
    rec["probes"] = probes
    rec["probes_passed"] = probes_passed
    rec["max_probe_diff"] = max_probe_diff
    probes_ok = probes_passed == probes

    # -- check 3: trial-level traces --------------------------------------
    trace_ok = True
    try:
        t_orig = instrument.exec_traced(code, func_name)
        t_lib = instrument.exec_traced(
            lib_source, lib_func_name, extra_sys_path=str(target.library_dir)
        )
        if t_orig is None or t_lib is None:
            rec["trace_available"] = False
        else:
            rec["trace_available"] = True
            _, p1o, p2o = t_orig(*inputs, fitted)
            _, p1l, p2l = t_lib(*inputs, fitted)
            div = instrument.first_divergence(p1o, p1l, p2o, p2l, atol=TRACE_ATOL)
            rec["trace_max_diff"] = float(
                max(np.max(np.abs(p1o - p1l)), np.max(np.abs(p2o - p2l)))
            )
            if div is not None:
                trace_ok = False
                rec["trace_divergence"] = div
    except Exception as e:  # instrumentation must never mask a scalar verdict
        rec["trace_available"] = False
        rec["trace_error"] = f"{type(e).__name__}: {e}"

    # -- check 4: synthetic datasets --------------------------------------
    synth_passed = 0
    max_synth_diff = 0.0
    synth_nlls = []
    for r in range(synthetic):
        srng = np.random.default_rng(SEED_BASE * 10 + pid * 100 + r)
        sinputs = _synthetic_inputs(target, inputs, srng)
        for theta in (fitted, probe_thetas[0] if probe_thetas else fitted):
            a = float(f_orig(*sinputs, theta))
            b = float(f_lib(*sinputs, theta))
            synth_nlls.append(a)
            if _nll_close(a, b):
                synth_passed += 1
            if math.isfinite(a) and math.isfinite(b):
                max_synth_diff = max(max_synth_diff, abs(a - b))
    rec["synthetic_checks"] = synthetic * 2
    rec["synthetic_passed"] = synth_passed
    rec["max_synthetic_diff"] = max_synth_diff
    synth_ok = synth_passed == synthetic * 2

    # -- WARN-only: stored-BIC consistency ---------------------------------
    stored_bic = loading.load_stored_bic(target, pid)
    if stored_bic is not None:
        recomputed = math.log(n_rows) * len(unpack) + 2.0 * nll_orig
        rec["bic_stored"] = stored_bic
        rec["bic_recomputed"] = recomputed
        rec["bic_consistent"] = abs(stored_bic - recomputed) < BIC_WARN_TOL

    rec["golden"] = {
        "fitted_nll": nll_orig,
        "probe_nlls": probe_nlls,
        "synthetic_nlls": synth_nlls,
    }

    passed = fitted_ok and probes_ok and trace_ok and synth_ok
    rec["status"] = "MATCH" if passed else "MISMATCH"
    return rec


def _fmt_row(rec: dict) -> str:
    if rec.get("status") == "ERROR":
        return f"  p{rec['pid']:<3} ERROR: {rec['error']}"
    trace = (
        "n/a"
        if not rec.get("trace_available")
        else ("ok" if "trace_divergence" not in rec else
              f"DIVERGES t={rec['trace_divergence']['trial']} s{rec['trace_divergence']['stage']}")
    )
    warns = []
    if not rec.get("param_order_ok", True):
        warns.append("param-order!")
    if not rec.get("signature_ok", True):
        warns.append("signature!")
    if rec.get("bic_consistent") is False:
        warns.append("bic-drift")
    return (
        f"  p{rec['pid']:<3} nll={rec['orig_nll']:<12.6f} "
        f"fittedΔ={rec['fitted_diff']:.2e}{'(bit)' if rec['bitwise_equal'] else '     '} "
        f"probes {rec['probes_passed']}/{rec['probes']} "
        f"trace {trace:<18} "
        f"synth {rec['synthetic_passed']}/{rec['synthetic_checks']} "
        f"{'[' + ' '.join(warns) + '] ' if warns else ''}-> {rec['status']}"
    )


def run_verification(
    target: Target,
    pids: Optional[List[int]] = None,
    probes: int = 20,
    synthetic: int = 3,
    baseline: bool = False,
) -> dict:
    all_ids = loading.participant_ids(target)
    pids = pids or all_ids
    mode = "baseline" if baseline else "library"
    print(f"Verifying {len(pids)} participants ({mode} mode) "
          f"[probes={probes}, synthetic={synthetic}] on {target.results_dir.name}")

    results: Dict[str, dict] = {}
    t0 = time.time()
    for pid in pids:
        try:
            rec = verify_participant(
                target, pid, probes=probes, synthetic=synthetic, baseline=baseline
            )
        except Exception as e:
            rec = {"pid": pid, "status": "ERROR", "error": f"{type(e).__name__}: {e}"}
        results[str(pid)] = rec
        print(_fmt_row(rec))

    matched = sum(1 for r in results.values() if r["status"] == "MATCH")
    finite_diffs = [r.get("fitted_diff") for r in results.values() if r.get("fitted_diff") is not None]
    summary = {
        "mode": mode,
        "matched": matched,
        "total": len(pids),
        "match_rate": f"{100.0 * matched / len(pids):.1f}%" if pids else "n/a",
        "max_fitted_diff": max(finite_diffs) if finite_diffs else None,
        "n_bitwise_equal": sum(1 for r in results.values() if r.get("bitwise_equal")),
        "probes_per_participant": probes,
        "synthetic_datasets": synthetic,
        "elapsed_s": round(time.time() - t0, 2),
    }
    print(f"SUMMARY: {matched}/{len(pids)} MATCH ({summary['match_rate']}), "
          f"max fitted diff {summary['max_fitted_diff']}, "
          f"{summary['n_bitwise_equal']} bitwise-equal, {summary['elapsed_s']}s")

    out = {"summary": summary, "participants": results}

    full_run = set(pids) == set(all_ids)
    target.library_dir.mkdir(exist_ok=True)
    if full_run:
        goldens = {
            str(p): r.pop("golden")
            for p, r in ((p, results[str(p)]) for p in pids)
            if "golden" in results[str(p)]
        }
        golden_path = target.library_dir / "goldens.json"
        with open(golden_path, "w") as f:
            json.dump({"seed_base": SEED_BASE, "participants": goldens}, f, indent=2)
        name = "verification_baseline.json" if baseline else "verification_results.json"
        with open(target.library_dir / name, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {target.library_dir / name} and goldens.json")
    else:
        for r in results.values():
            r.pop("golden", None)
    return out
