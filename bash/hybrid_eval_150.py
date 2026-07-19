"""Task 9 arm-2: evaluate the frozen hybrid-base composed winner on validation
and test, mirroring library_learning.__main__.cmd_eval but pointed at the
hybrid_base/ out_dir (same pattern as the 45-pid run). No reviewed code changed.
Run: PYTHONPATH=<repo> gecco-env/bin/python bash/hybrid_eval_150.py
"""
import json
from pathlib import Path

from library_learning.config import resolve_target
from library_learning.compose.evaluate import (cross_checks, evaluate_models,
                                               summarize)
from library_learning.compose.figure import plot_comparison
from library_learning.compose.splits import load_splits

IND = "results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual"
GRP = "results/two_step_psychiatry_group_function_ocibalanced150_maxsetting"

target = resolve_target(IND)
hb = target.results_dir / "library_composition" / "hybrid_base"
if not (hb / "composed_model.txt").exists():
    raise SystemExit("hybrid_base/composed_model.txt not frozen yet — run the search first")

splits = load_splits(target)
rv = evaluate_models(target, GRP, hb, splits["composition_validation_pids"], "validation")
rt = evaluate_models(target, GRP, hb, splits["test_pids"], "test")
heldout = sorted(set(splits["reconstruction_pids"]) | set(splits["test_pids"]))
warn = cross_checks(rt, target, GRP, splits["test_pids"], heldout_pids=heldout)
summarize(rv, rt, warn, hb, target=target)
plot_comparison(hb / "test_results.csv", hb)
print("hybrid-base eval done, warnings:", len(warn))
print("  -> %s" % (hb / "RESULTS.md"))
