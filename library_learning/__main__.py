"""Repo-level CLI. Current subcommands cover the composition pipeline; the
compression pipeline (scan/verify/report) gets wired here when resumed.
Subcommands dispatch by the target's resolved task name: task 'rlwm' uses the
parallel *_rlwm modules, everything else the original two-step modules."""
import argparse
import json
import sys
from pathlib import Path

from .config import _task_name_for, resolve_target
from .compose import evaluate as evaluate_two_step
from .compose import evaluate_rlwm
from .compose import extract as extract_two_step
from .compose import extract_rlwm
from .compose import figure as figure_two_step
from .compose import figure_rlwm
from .compose import inventory as inventory_two_step
from .compose import inventory_rlwm
from .compose import reconstruct as reconstruct_two_step
from .compose import reconstruct_rlwm
from .compose import search as search_mod
from .compose import search_rlwm
from .compose.hybrid import HYBRID_MODULES
from .compose.splits import load_splits

DEFAULT_IND = "results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual"
DEFAULT_GRP = "results/two_step_psychiatry_group_function_ocibalanced_maxsetting"
OUT_DIRNAME = "library_composition"

_TWO_STEP_MODS = {"extract": extract_two_step, "search": search_mod,
                  "reconstruct": reconstruct_two_step,
                  "evaluate": evaluate_two_step, "figure": figure_two_step,
                  "inventory": inventory_two_step}
_RLWM_MODS = {"extract": extract_rlwm, "search": search_rlwm,
              "reconstruct": reconstruct_rlwm, "evaluate": evaluate_rlwm,
              "figure": figure_rlwm, "inventory": inventory_rlwm}


def _task_mods(results_dir):
    if _task_name_for(Path(results_dir).resolve()) == "rlwm":
        return _RLWM_MODS
    return _TWO_STEP_MODS


def add_common(p):
    p.add_argument("--results-dir", default=DEFAULT_IND)
    p.add_argument("--group-dir", default=DEFAULT_GRP)
    p.add_argument("--config-dir", default=None)


def out_dir_for(target):
    return target.results_dir / OUT_DIRNAME


def cmd_modules(args):
    mods = _task_mods(args.results_dir)
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    if args.skip_llm:
        obj = json.loads((out / "module_inventory.json").read_text())
        inv, errors = mods["extract"].validate_inventory_obj(obj)
        if errors:
            print("\n".join(errors))
            return 1
        print("inventory valid: %d modules" % len(inv.modules))
        return 0
    inv = mods["extract"].run_extraction(target, args.group_dir, out)
    print("extracted %d modules -> %s" % (len(inv.modules), out))
    return 0


def cmd_count(args):
    mods = _task_mods(args.results_dir)
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    inv = mods["inventory"].load_inventory(out_dir_for(target) / "module_inventory.json")
    print(json.dumps(mods["search"].count_report(inv), indent=2))
    return 0


def cmd_search(args):
    mods = _task_mods(args.results_dir)
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    inv = mods["inventory"].load_inventory(out / "module_inventory.json")
    val_pids = load_splits(target)["composition_validation_pids"]
    if args.mode == "exhaustive":
        cands = mods["search"].enumerate_candidates(inv)
        results = mods["search"].score_candidates(inv, target, val_pids, cands, out)
    else:
        results = mods["search"].greedy_search(inv, target, val_pids, out)
    winner = mods["search"].select_winner(results)
    mods["search"].freeze_winner(winner, inv, out)
    mods["search"].selection_report(results, out)
    print("WINNER %s mean validation BIC %.2f -> composed_model.txt frozen"
          % (winner["candidate_id"], winner["mean_bic"]))
    return 0


def cmd_hybrid_search(args):
    """Exhaustive search over modules missing from the Daw hybrid, using the
    hybrid-equivalent module set as a fixed base. Outputs under hybrid_base/."""
    if _task_mods(args.results_dir) is _RLWM_MODS:
        print("compose-hybrid-search is two-step only; the RLWM hybrid-base "
              "arm is deferred (see the 2026-07-18 spec non-goals)")
        return 2
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    inv = inventory_two_step.load_inventory(out / "module_inventory.json")
    val_pids = load_splits(target)["composition_validation_pids"]
    hb_out = out / "hybrid_base"
    hb_out.mkdir(exist_ok=True)
    cands = search_mod.enumerate_from_base(inv, HYBRID_MODULES,
                                           param_cap=args.param_cap)
    print("hybrid base %s: %d candidates (param cap %d)"
          % ("+".join(HYBRID_MODULES), len(cands), args.param_cap))
    results = search_mod.score_candidates(inv, target, val_pids, cands, hb_out)
    winner = search_mod.select_winner(results)
    search_mod.freeze_winner(winner, inv, hb_out)
    search_mod.selection_report(results, hb_out)
    print("WINNER %s mean validation BIC %.2f -> %s"
          % (winner["candidate_id"], winner["mean_bic"],
             hb_out / "composed_model.txt"))
    return 0


def cmd_reconstruct(args):
    mods = _task_mods(args.results_dir)
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    inv = mods["inventory"].load_inventory(out / "module_inventory.json")
    pids = load_splits(target)["reconstruction_pids"]
    mods["reconstruct"].reconstruct_participants(inv, target, args.group_dir,
                                                 pids, out, mode=args.mode)
    print("reconstruction -> %s" % (out / "reconstruction_results.json"))
    return 0


def cmd_eval(args):
    mods = _task_mods(args.results_dir)
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    splits = load_splits(target)
    sets = args.sets.split(",")
    results_val = None
    if "validation" in sets:
        results_val = mods["evaluate"].evaluate_models(
            target, args.group_dir, out,
            splits["composition_validation_pids"], "validation")
    if "test" not in sets:
        print("validation-only run (--sets=%s): skipping test evaluation, "
              "cross-checks, summary, and figure" % args.sets)
        return 0
    results_test = mods["evaluate"].evaluate_models(
        target, args.group_dir, out, splits["test_pids"], "test")
    heldout_pids = sorted(set(splits["reconstruction_pids"])
                          | set(splits["test_pids"]))
    warnings = mods["evaluate"].cross_checks(results_test, target, args.group_dir,
                                             splits["test_pids"],
                                             heldout_pids=heldout_pids)
    mods["evaluate"].summarize(results_val, results_test, warnings, out,
                               target=target)
    mods["figure"].plot_comparison(out / "test_results.csv", out)
    print("results -> %s (warnings: %d)" % (out / "RESULTS.md", len(warnings)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="library_learning")
    subs = parser.add_subparsers(dest="cmd", required=True)

    p = subs.add_parser("compose-modules", help="Gemini module extraction (logged)")
    add_common(p)
    p.add_argument("--skip-llm", action="store_true")
    p.set_defaults(fn=cmd_modules)

    p = subs.add_parser("compose-count", help="candidate count (user checkpoint)")
    add_common(p)
    p.set_defaults(fn=cmd_count)

    p = subs.add_parser("compose-search", help="fit candidates on validation, freeze winner")
    add_common(p)
    p.add_argument("--mode", choices=["exhaustive", "greedy"], required=True)
    p.set_defaults(fn=cmd_search)

    p = subs.add_parser("compose-hybrid-search",
                        help="exhaustive search over modules missing from the Daw hybrid")
    add_common(p)
    p.add_argument("--param-cap", type=int, default=9)
    p.set_defaults(fn=cmd_hybrid_search)

    p = subs.add_parser("compose-reconstruct",
                        help="per-participant library coverage on the reconstruction set")
    add_common(p)
    p.add_argument("--mode", choices=["exhaustive", "greedy"], default="greedy")
    p.set_defaults(fn=cmd_reconstruct)

    p = subs.add_parser("compose-eval", help="final-test evaluation vs baselines")
    add_common(p)
    p.add_argument("--sets", default="validation,test")
    p.set_defaults(fn=cmd_eval)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
