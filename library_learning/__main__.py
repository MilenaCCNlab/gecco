"""Repo-level CLI. Current subcommands cover the composition pipeline; the
compression pipeline (scan/verify/report) gets wired here when resumed."""
import argparse
import json
import sys
from pathlib import Path

from .config import resolve_target
from .compose import search as search_mod
from .compose.evaluate import cross_checks, evaluate_models, summarize
from .compose.extract import run_extraction, validate_inventory_obj
from .compose.figure import plot_comparison
from .compose.inventory import load_inventory
from .compose.reconstruct import reconstruct_participants
from .compose.splits import load_splits, make_splits

DEFAULT_IND = "results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual"
DEFAULT_GRP = "results/two_step_psychiatry_group_function_ocibalanced_maxsetting"
OUT_DIRNAME = "library_composition"


def add_common(p):
    p.add_argument("--results-dir", default=DEFAULT_IND)
    p.add_argument("--group-dir", default=DEFAULT_GRP)
    p.add_argument("--config-dir", default=None)


def out_dir_for(target):
    return target.results_dir / OUT_DIRNAME


def cmd_modules(args):
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    if args.skip_llm:
        obj = json.loads((out / "module_inventory.json").read_text())
        inv, errors = validate_inventory_obj(obj)
        if errors:
            print("\n".join(errors))
            return 1
        print("inventory valid: %d modules" % len(inv.modules))
        return 0
    inv = run_extraction(target, args.group_dir, out)
    print("extracted %d modules -> %s" % (len(inv.modules), out))
    return 0


def cmd_count(args):
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    inv = load_inventory(out_dir_for(target) / "module_inventory.json")
    print(json.dumps(search_mod.count_report(inv), indent=2))
    return 0


def cmd_search(args):
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    inv = load_inventory(out / "module_inventory.json")
    val_pids = load_splits(target)["composition_validation_pids"]
    if args.mode == "exhaustive":
        cands = search_mod.enumerate_candidates(inv)
        results = search_mod.score_candidates(inv, target, val_pids, cands, out)
    else:
        results = search_mod.greedy_search(inv, target, val_pids, out)
    winner = search_mod.select_winner(results)
    search_mod.freeze_winner(winner, inv, out)
    search_mod.selection_report(results, out)
    print("WINNER %s mean validation BIC %.2f -> composed_model.txt frozen"
          % (winner["candidate_id"], winner["mean_bic"]))
    return 0


def cmd_reconstruct(args):
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    inv = load_inventory(out / "module_inventory.json")
    pids = load_splits(target)["reconstruction_pids"]
    reconstruct_participants(inv, target, args.group_dir, pids, out,
                             mode=args.mode)
    print("reconstruction -> %s" % (out / "reconstruction_results.json"))
    return 0


def cmd_eval(args):
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    splits = load_splits(target)
    sets = args.sets.split(",")
    results_val = None
    if "validation" in sets:
        results_val = evaluate_models(target, args.group_dir, out,
                                      splits["composition_validation_pids"],
                                      "validation")
    results_test = evaluate_models(target, args.group_dir, out,
                                   splits["test_pids"], "test")
    warnings = cross_checks(results_test, target, args.group_dir,
                            splits["test_pids"])
    summarize(results_val, results_test, warnings, out, target=target)
    plot_comparison(out / "test_results.csv", out)
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
