"""Deterministic composition search over module combinations."""
import itertools
import json
from collections import Counter
from pathlib import Path

from .inventory import compatible
from .render import (candidate_id as candidate_id_of, candidate_params,
                     render_candidate, smoke_check)
from .fitting import fit_model_on_pids

PARAM_CAP = 8
TIE_TOL = 1.0


def _n_params(inventory, mods):
    return 2 + sum(inventory.module(m).n_params for m in mods)


def enumerate_candidates(inventory, param_cap=PARAM_CAP):
    ids = sorted(inventory.ids())
    out = []
    for k in range(len(ids) + 1):
        for combo in itertools.combinations(ids, k):
            if _n_params(inventory, combo) > param_cap:
                continue
            ok, _ = compatible(inventory, list(combo))
            if ok:
                out.append(combo)
    return out


def enumerate_from_base(inventory, base, param_cap=PARAM_CAP):
    """All valid supersets of a fixed base module set (base itself included)."""
    base = tuple(sorted(base))
    ok, msg = compatible(inventory, list(base))
    if not ok:
        raise ValueError("base set incompatible: %s" % msg)
    if _n_params(inventory, base) > param_cap:
        raise ValueError("base set alone exceeds param cap %d" % param_cap)
    rest = sorted(set(inventory.ids()) - set(base))
    out = []
    for k in range(len(rest) + 1):
        for extra in itertools.combinations(rest, k):
            cand = tuple(sorted(base + extra))
            if _n_params(inventory, cand) > param_cap:
                continue
            ok, _ = compatible(inventory, list(cand))
            if ok:
                out.append(cand)
    return out


def count_report(inventory, param_cap=PARAM_CAP):
    cands = enumerate_candidates(inventory, param_cap)
    return {
        "n_modules": len(inventory.ids()),
        "param_cap": param_cap,
        "n_candidates": len(cands),
        "by_n_params": dict(Counter(_n_params(inventory, c) for c in cands)),
        "by_n_modules": dict(Counter(len(c) for c in cands)),
    }


def score_candidates(inventory, target, validation_pids, candidates, out_dir):
    out_dir = Path(out_dir)
    log_path = out_dir / "search_log.jsonl"
    results = []
    with open(log_path, "a") as log:
        for mods in candidates:
            cid = candidate_id_of(list(mods))
            src = render_candidate(inventory, list(mods))
            smoke_check(src)  # fail fast on render bugs
            bounds = [p.bounds for p in candidate_params(inventory, list(mods))]
            fits = fit_model_on_pids(src, target, validation_pids, bounds, tag=cid)
            rec = {
                "candidate_id": cid,
                "module_ids": sorted(mods),
                "n_params": len(bounds),
                "per_pid": {str(p): fits[p] for p in fits},
                "mean_bic": sum(f["bic"] for f in fits.values()) / len(fits),
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            results.append(rec)
            print("[search] %-60s mean BIC %.2f" % (cid, rec["mean_bic"]))
    return results


def greedy_search(inventory, target, validation_pids, out_dir, param_cap=PARAM_CAP):
    current = ()
    results = score_candidates(inventory, target, validation_pids, [current], out_dir)
    best = results[0]
    improved = True
    while improved:
        improved = False
        additions = []
        for mid in sorted(set(inventory.ids()) - set(current)):
            cand = tuple(sorted(current + (mid,)))
            if _n_params(inventory, cand) > param_cap:
                continue
            ok, _ = compatible(inventory, list(cand))
            if ok:
                additions.append(cand)
        if not additions:
            break
        round_results = score_candidates(inventory, target, validation_pids,
                                         additions, out_dir)
        results.extend(round_results)
        round_best = min(round_results, key=lambda r: r["mean_bic"])
        if round_best["mean_bic"] < best["mean_bic"]:
            best = round_best
            current = tuple(best["module_ids"])
            improved = True
    return results


def select_winner(results):
    best = min(results, key=lambda r: r["mean_bic"])
    contenders = [r for r in results if r["mean_bic"] <= best["mean_bic"] + TIE_TOL]
    return min(contenders, key=lambda r: (r["n_params"], r["mean_bic"]))


def selection_report(results, out_dir, k=10):
    """Top-k + leave-one-participant-out rank stability (winner's-curse probe)."""
    ranked = sorted(results, key=lambda r: r["mean_bic"])[:k]
    top_ids = [r["candidate_id"] for r in ranked]
    pids = sorted(next(iter(results))["per_pid"].keys(), key=int) if results else []
    loo_ranks = {cid: [] for cid in top_ids}
    for left_out in pids:
        loo = sorted(
            results,
            key=lambda r: (sum(v["bic"] for p, v in r["per_pid"].items()
                               if p != left_out)
                           / max(1, len(r["per_pid"]) - 1)))
        order = [r["candidate_id"] for r in loo]
        for cid in top_ids:
            loo_ranks[cid].append(order.index(cid) + 1)
    report = {
        "top_k": [{"candidate_id": r["candidate_id"],
                   "n_params": r["n_params"],
                   "mean_bic": r["mean_bic"]} for r in ranked],
        "loo_ranks": loo_ranks,
        "loo_pids": pids,
    }
    Path(out_dir, "selection_report.json").write_text(json.dumps(report, indent=2))
    return report


def freeze_winner(result, inventory, out_dir):
    out_dir = Path(out_dir)
    src = render_candidate(inventory, result["module_ids"])
    (out_dir / "composed_model.txt").write_text(src)
    (out_dir / "winner.json").write_text(json.dumps(result, indent=2))
