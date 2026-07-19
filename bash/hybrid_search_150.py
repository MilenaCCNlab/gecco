"""Hybrid-base exhaustive search for the ocibalanced150 inventory.

Mirrors library_learning.__main__.cmd_hybrid_search, but with the Daw-hybrid
module set REMAPPED to this inventory's ids (the 150-pid extraction named two
modules differently than the hardcoded hybrid.py HYBRID_MODULES):
  choice_stickiness    -> stage1_stickiness
  separate_stage_betas -> separate_stage2_beta
Outputs under <IND>/library_composition/hybrid_base/. No reviewed code changed.
"""
import sys
from library_learning.config import resolve_target
from library_learning.compose.inventory import load_inventory
from library_learning.compose import search as S
from library_learning.compose.splits import load_splits

IND = "results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual"
PARAM_CAP = 9
BASE = ["stage1_stickiness", "eligibility_trace", "mb_mf_mixture",
        "separate_learning_rates", "separate_stage2_beta"]

target = resolve_target(IND)
out = target.results_dir / "library_composition"
inv = load_inventory(out / "module_inventory.json")
val_pids = load_splits(target)["composition_validation_pids"]
hb_out = out / "hybrid_base"
hb_out.mkdir(exist_ok=True)

cands = S.enumerate_from_base(inv, BASE, param_cap=PARAM_CAP)
print("hybrid base %s: %d candidates (param cap %d)"
      % ("+".join(BASE), len(cands), PARAM_CAP), flush=True)
results = S.score_candidates(inv, target, val_pids, cands, hb_out)
winner = S.select_winner(results)
S.freeze_winner(winner, inv, hb_out)
S.selection_report(results, hb_out)
print("WINNER %s mean validation BIC %.2f -> %s"
      % (winner["candidate_id"], winner["mean_bic"], hb_out / "composed_model.txt"))
