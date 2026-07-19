"""Leave-one-out ablations of the Daw-hybrid-equivalent module base, per test
participant — the "removals" half of the component ledger (two-step OCI analogue
of analysis/rlwm/compute_baseline_ablations.py).

The Daw hybrid == these 5 library modules on the backbone. "full" fits all 5;
each ablation drops one module. Reuses render_candidate + fit_model_on_pids
(tested machinery); no new model code. Writes baseline_ablation_bics_oci.json:
  {"full": {pid: bic}, "no_<module>": {pid: bic}, ...}

Run: PYTHONPATH=. gecco-env/bin/python analysis/two_step_task/compute_baseline_ablations_oci.py
"""
import json
from pathlib import Path

from library_learning.config import resolve_target
from library_learning.compose.inventory import load_inventory
from library_learning.compose.render import render_candidate, candidate_params
from library_learning.compose.fitting import fit_model_on_pids
from library_learning.compose.splits import load_splits

ROOT = Path(__file__).resolve().parents[2]
IND = ROOT / "results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual"
OUT = ROOT / "analysis/two_step_task/baseline_ablation_bics_oci.json"
BASE = ["stage1_stickiness", "eligibility_trace", "mb_mf_mixture",
        "separate_learning_rates", "separate_stage2_beta"]


def fit_set(inv, target, pids, module_ids, tag):
    src = render_candidate(inv, list(module_ids))
    bounds = [p.bounds for p in candidate_params(inv, list(module_ids))]
    fits = fit_model_on_pids(src, target, pids, bounds, tag=tag)
    return {int(p): fits[p]["bic"] for p in pids}


def main():
    target = resolve_target(str(IND))
    inv = load_inventory(IND / "library_composition" / "module_inventory.json")
    pids = load_splits(target)["test_pids"]

    result = {}
    print("fitting full hybrid base (%d modules) on %d test pids..." % (len(BASE), len(pids)))
    result["full"] = fit_set(inv, target, pids, BASE, "abl:full")
    for m in BASE:
        rem = [x for x in BASE if x != m]
        print("ablation: drop %s (%d modules left)" % (m, len(rem)))
        result["no_%s" % m] = fit_set(inv, target, pids, rem, "abl:no_%s" % m)

    OUT.write_text(json.dumps(result, indent=2))
    # quick summary: mean(full - ablation); positive = removing that module lowers BIC
    import numpy as np
    fm = np.mean(list(result["full"].values()))
    print("full mean BIC: %.1f" % fm)
    for m in BASE:
        am = np.mean(list(result["no_%s" % m].values()))
        print("  drop %-26s ablation %.1f  (full-abl %+.1f)" % (m, am, fm - am))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
