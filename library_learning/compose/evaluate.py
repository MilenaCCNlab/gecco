"""Stage 3: fit frozen winner + baselines on held-out participants under one
seeded protocol; stats + RESULTS.md."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .fitting import fit_model_on_pids
from .hybrid import HYBRID_BOUNDS, HYBRID_SOURCE
from ..loading import (extract_unpack_names, function_name_and_args,
                       load_original_code, parse_bounds, strip_fences)

TIE_TOL = 1.0
CROSS_CHECK_TOL = 5.0


def _wilcoxon_p(a, b):
    """Wilcoxon signed-rank p; 1.0 when all differences are zero (no signal)."""
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def _bounds_for(code):
    names = extract_unpack_names(code)
    b = parse_bounds(code, names)
    return [b[n] for n in names]


def evaluate_models(target, group_dir, out_dir, pids, set_name):
    out_dir = Path(out_dir)
    composed_path = out_dir / "composed_model.txt"
    if not composed_path.exists():
        raise FileNotFoundError(
            "composed_model.txt not found in %s — run compose-search first "
            "(freeze discipline: evaluation only runs on a frozen winner)" % out_dir)
    composed_src = composed_path.read_text()

    group_src = strip_fences(
        (Path(group_dir) / "models" / "best_model_0.txt").read_text())
    group_func_name, _ = function_name_and_args(group_src)

    results = {}
    results["composed"] = fit_model_on_pids(
        composed_src, target, pids, _bounds_for(composed_src),
        tag="eval:%s:composed" % set_name)
    results["group"] = fit_model_on_pids(
        group_src, target, pids, _bounds_for(group_src),
        tag="eval:%s:group" % set_name, func_name=group_func_name)
    results["hybrid"] = fit_model_on_pids(
        HYBRID_SOURCE, target, pids, HYBRID_BOUNDS,
        tag="eval:%s:hybrid" % set_name)
    individual = {}
    for pid in pids:
        code = load_original_code(target, pid)
        fname, _ = function_name_and_args(code)
        individual.update(fit_model_on_pids(
            code, target, [pid], _bounds_for(code),
            tag="eval:%s:individual" % set_name, func_name=fname))
    results["individual"] = individual
    return results


def cross_checks(results, target, group_dir, pids):
    warnings = []
    stored_path = Path(group_dir) / "bics" / "best_bic_on_test_run0.json"
    if stored_path.exists():
        stored = json.loads(stored_path.read_text())["individual_BIC"]
        for pid in pids:
            idx = pid - 14
            if 0 <= idx < len(stored):
                diff = results["group"][pid]["bic"] - stored[idx]
                if abs(diff) > CROSS_CHECK_TOL:
                    warnings.append(
                        "group refit BIC differs from stored for p%d: %.2f vs %.2f"
                        % (pid, results["group"][pid]["bic"], stored[idx]))
    df = pd.read_csv(target.data_path)
    baseline = df.groupby(target.id_column)["baseline_bic"].first()
    for pid in pids:
        diff = results["hybrid"][pid]["bic"] - float(baseline[pid])
        if abs(diff) > CROSS_CHECK_TOL:
            warnings.append(
                "hybrid refit BIC differs from baseline_bic for p%d: %.2f vs %.2f"
                % (pid, results["hybrid"][pid]["bic"], float(baseline[pid])))
    return warnings


def _rows(results, set_name, oci):
    rows = []
    for model, fits in results.items():
        for pid, f in fits.items():
            rows.append({"set": set_name, "participant": pid,
                         "oci": float(oci[pid]), "model": model,
                         "n_params": len(f["params"]), "nll": f["nll"],
                         "bic": f["bic"], "seed": f["seed"]})
    return rows


def summarize(results_val, results_test, warnings, out_dir, target=None):
    out_dir = Path(out_dir)
    if target is not None:
        df = pd.read_csv(target.data_path)
        oci = df.groupby(target.id_column)["oci"].first()
    else:
        all_pids = {p for r in [results_val, results_test] if r
                    for fits in r.values() for p in fits}
        oci = {p: float("nan") for p in all_pids}

    rows = []
    if results_val:
        rows += _rows(results_val, "validation", oci)
    rows += _rows(results_test, "test", oci)
    pd.DataFrame(rows).to_csv(out_dir / "test_results.csv", index=False)

    test_pids = sorted(next(iter(results_test.values())).keys())
    means = {m: float(np.mean([results_test[m][p]["bic"] for p in test_pids]))
             for m in results_test}
    comp = np.array([results_test["composed"][p]["bic"] for p in test_pids])
    grp = np.array([results_test["group"][p]["bic"] for p in test_pids])
    hyb = np.array([results_test["hybrid"][p]["bic"] for p in test_pids])
    delta = comp - grp
    wins = int((delta < -TIE_TOL).sum())
    ties = int((np.abs(delta) <= TIE_TOL).sum())
    losses = int((delta > TIE_TOL).sum())
    stats = {
        "mean_bic": means,
        "composed_vs_group": {"wilcoxon_p": _wilcoxon_p(comp, grp),
                              "wins": wins, "ties": ties, "losses": losses,
                              "mean_delta": float(delta.mean())},
        "composed_vs_hybrid": {"wilcoxon_p": _wilcoxon_p(comp, hyb),
                               "mean_delta": float((comp - hyb).mean())},
        "warnings": warnings,
    }
    (out_dir / "test_results.json").write_text(json.dumps(
        {"stats": stats, "rows": rows}, indent=2))

    winner = json.loads((out_dir / "winner.json").read_text())
    lines = ["# Library composition results", "",
             "Winner modules: `%s` (%d params)" % (winner["candidate_id"],
                                                   winner["n_params"]), "",
             "## Mean BIC on final test (%d participants)" % len(test_pids), "",
             "| model | mean BIC |", "|---|---|"]
    for m in sorted(means, key=means.get):
        lines.append("| %s | %.2f |" % (m, means[m]))
    lines += ["",
              "composed vs group: mean dBIC %.2f, W/T/L %d/%d/%d, wilcoxon p=%.4f"
              % (stats["composed_vs_group"]["mean_delta"], wins, ties, losses,
                 stats["composed_vs_group"]["wilcoxon_p"]),
              "composed vs hybrid: mean dBIC %.2f, wilcoxon p=%.4f"
              % (stats["composed_vs_hybrid"]["mean_delta"],
                 stats["composed_vs_hybrid"]["wilcoxon_p"]), ""]
    if warnings:
        lines += ["## Cross-check warnings", ""] + ["- " + w for w in warnings]
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return stats
