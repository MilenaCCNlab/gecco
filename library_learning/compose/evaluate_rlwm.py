# library_learning/compose/evaluate_rlwm.py
"""Stage 3 for RLWM: fit frozen winner + baselines on held-out participants
under one seeded protocol; stats + RESULTS.md. Parallel sibling of
evaluate.py — the Daw hybrid is replaced by the canonical Collins & Frank
RLWM baseline, and OCI by age. The stored group-test-BIC cross-check is kept
but auto-skips (results/rlwm has no best_bic_on_test_run0.json)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical_rlwm import CANONICAL_BOUNDS, CANONICAL_SOURCE
from .evaluate import CROSS_CHECK_TOL, TIE_TOL, _wilcoxon_p
from .fitting import fit_model_on_pids
from ..loading import (bounds_for_code, function_name_and_args,
                       load_dataframe, load_original_code, strip_fences)


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
        composed_src, target, pids, bounds_for_code(composed_src),
        tag="eval:%s:composed" % set_name)
    results["group"] = fit_model_on_pids(
        group_src, target, pids, bounds_for_code(group_src),
        tag="eval:%s:group" % set_name, func_name=group_func_name)
    results["canonical"] = fit_model_on_pids(
        CANONICAL_SOURCE, target, pids, CANONICAL_BOUNDS,
        tag="eval:%s:canonical" % set_name)
    individual = {}
    for pid in pids:
        # validation pids 15-19 have no individual gecco fits (only 0-14 and
        # 36-50 were run); the individual baseline covers fitted pids only.
        # All reconstruction/test pids are fitted by construction (splits).
        if not (target.models_dir / ("best_model_0_participant%d.txt" % pid)).exists():
            continue
        code = load_original_code(target, pid)
        fname, _ = function_name_and_args(code)
        individual.update(fit_model_on_pids(
            code, target, [pid], bounds_for_code(code),
            tag="eval:%s:individual" % set_name, func_name=fname))
    results["individual"] = individual
    return results


def cross_checks(results, target, group_dir, pids, heldout_pids=None):
    warnings = []
    stored_path = Path(group_dir) / "bics" / "best_bic_on_test_run0.json"
    if stored_path.exists():
        stored = json.loads(stored_path.read_text())["individual_BIC"]
        if heldout_pids is None:
            heldout_pids = sorted(range(14, 14 + len(stored)))
        for pid in pids:
            if pid not in heldout_pids:
                warnings.append(
                    "cross-check skipped for p%d: not in stored held-out mapping"
                    % pid)
                continue
            idx = heldout_pids.index(pid)
            if 0 <= idx < len(stored):
                diff = results["group"][pid]["bic"] - stored[idx]
                if abs(diff) > CROSS_CHECK_TOL:
                    warnings.append(
                        "group refit BIC differs from stored for p%d: %.2f vs %.2f"
                        % (pid, results["group"][pid]["bic"], stored[idx]))
    df = load_dataframe(target)
    baseline = df.groupby(target.id_column)["baseline_bic"].first()
    for pid in pids:
        diff = results["canonical"][pid]["bic"] - float(baseline[pid])
        if abs(diff) > CROSS_CHECK_TOL:
            warnings.append(
                "canonical refit BIC differs from baseline_bic for p%d: "
                "%.2f vs %.2f (the column's producing variant is unknown)"
                % (pid, results["canonical"][pid]["bic"], float(baseline[pid])))
    return warnings


def _rows(results, set_name, age):
    rows = []
    for model, fits in results.items():
        for pid, f in fits.items():
            rows.append({"set": set_name, "participant": pid,
                         "age": float(age[pid]), "model": model,
                         "n_params": len(f["params"]), "nll": f["nll"],
                         "bic": f["bic"], "seed": f["seed"]})
    return rows


def summarize(results_val, results_test, warnings, out_dir, target=None):
    out_dir = Path(out_dir)
    if target is not None:
        df = load_dataframe(target)
        age = df.groupby(target.id_column)["age"].first()
    else:
        all_pids = {p for r in [results_val, results_test] if r
                    for fits in r.values() for p in fits}
        age = {p: float("nan") for p in all_pids}

    rows = []
    if results_val:
        rows += _rows(results_val, "validation", age)
    rows += _rows(results_test, "test", age)
    pd.DataFrame(rows).to_csv(out_dir / "test_results.csv", index=False)

    test_pids = sorted(next(iter(results_test.values())).keys())
    means = {m: float(np.mean([results_test[m][p]["bic"] for p in test_pids]))
             for m in results_test}
    comp = np.array([results_test["composed"][p]["bic"] for p in test_pids])
    grp = np.array([results_test["group"][p]["bic"] for p in test_pids])
    canon = np.array([results_test["canonical"][p]["bic"] for p in test_pids])
    delta = comp - grp
    wins = int((delta < -TIE_TOL).sum())
    ties = int((np.abs(delta) <= TIE_TOL).sum())
    losses = int((delta > TIE_TOL).sum())
    stats = {
        "mean_bic": means,
        "composed_vs_group": {"wilcoxon_p": _wilcoxon_p(comp, grp),
                              "wins": wins, "ties": ties, "losses": losses,
                              "mean_delta": float(delta.mean())},
        "composed_vs_canonical": {"wilcoxon_p": _wilcoxon_p(comp, canon),
                                  "mean_delta": float((comp - canon).mean())},
        "warnings": warnings,
    }
    (out_dir / "test_results.json").write_text(json.dumps(
        {"stats": stats, "rows": rows}, indent=2))

    winner = json.loads((out_dir / "winner.json").read_text())
    lines = ["# RLWM library composition results", "",
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
              "composed vs canonical: mean dBIC %.2f, wilcoxon p=%.4f"
              % (stats["composed_vs_canonical"]["mean_delta"],
                 stats["composed_vs_canonical"]["wilcoxon_p"]), ""]
    if warnings:
        lines += ["## Cross-check warnings", ""] + ["- " + w for w in warnings]
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return stats
