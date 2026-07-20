"""Per-participant mechanism-importance for all 150 participants: for each
library module, the BIC gain of that single module over the bare backbone,
fit on each participant's own data. This is the continuous per-(participant,
mechanism) measure used to test which mechanisms differ with OCI.

Reuses render_candidate + fit_model_on_pids (tested); no new model code.
Writes analysis/two_step_task/module_gains_oci.csv:
  participant, oci, tertile, module, backbone_bic, module_bic, gain
  (gain = backbone_bic - module_bic ; + = the mechanism improves this person's fit)

Run: PYTHONPATH=. gecco-env/bin/python analysis/two_step_task/compute_module_gains_oci.py
"""
import json
from pathlib import Path

import pandas as pd

from library_learning.config import resolve_target
from library_learning.compose.inventory import load_inventory
from library_learning.compose.render import render_candidate, candidate_params
from library_learning.compose.fitting import fit_model_on_pids

ROOT = Path(__file__).resolve().parents[2]
IND = ROOT / "results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual"
OUT = ROOT / "analysis/two_step_task/module_gains_oci.csv"


def main():
    target = resolve_target(str(IND))
    inv = load_inventory(IND / "library_composition" / "module_inventory.json")
    manifest = {p["participant"]: p for p in
                json.loads((ROOT / "data/ocd/ocibalanced150_manifest.json").read_text())["participants"]}
    pids = sorted(manifest)

    # backbone (empty module set) fit on every participant
    bb_src = render_candidate(inv, [])
    bb_bounds = [p.bounds for p in candidate_params(inv, [])]
    print("fitting backbone on %d participants..." % len(pids))
    bb = fit_model_on_pids(bb_src, target, pids, bb_bounds, tag="gain:backbone")

    modules = [m.id for m in inv.modules]
    rows = []
    for mi, mid in enumerate(modules):
        src = render_candidate(inv, [mid])
        bounds = [p.bounds for p in candidate_params(inv, [mid])]
        fits = fit_model_on_pids(src, target, pids, bounds, tag="gain:%s" % mid)
        for pid in pids:
            rows.append({
                "participant": pid,
                "oci": manifest[pid]["oci_total"],
                "tertile": manifest[pid]["tertile"],
                "module": mid,
                "backbone_bic": bb[pid]["bic"],
                "module_bic": fits[pid]["bic"],
                "gain": bb[pid]["bic"] - fits[pid]["bic"],
            })
        print("  [%2d/%2d] %-34s done" % (mi + 1, len(modules), mid), flush=True)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote %s (%d rows = %d modules x %d pids)" % (OUT, len(rows), len(modules), len(pids)))


if __name__ == "__main__":
    main()
