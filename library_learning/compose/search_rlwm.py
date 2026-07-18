# library_learning/compose/search_rlwm.py
"""Deterministic composition search over RLWM module combinations. Parallel
sibling of search.py: 3 backbone params (learning_rate, beta, wm_weight) and
the RLWM gecco guardrail cap of 6. select_winner/selection_report are reused
from search.py (they only touch result dicts). No enumerate_from_base here —
the hybrid-base arm is deferred (spec non-goal)."""
import itertools
import json
from collections import Counter
from pathlib import Path

from .inventory import compatible
from .render_rlwm import (candidate_id, candidate_params, render_candidate,
                          smoke_check)
from .fitting import fit_model_on_pids
from .search import select_winner, selection_report  # noqa: F401 (re-exported)

PARAM_CAP = 6
N_BACKBONE_PARAMS = 3


def _n_params(inventory, mods):
    return N_BACKBONE_PARAMS + sum(inventory.module(m).n_params for m in mods)


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
            cid = candidate_id(list(mods))
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


def freeze_winner(result, inventory, out_dir):
    out_dir = Path(out_dir)
    src = render_candidate(inventory, result["module_ids"])
    (out_dir / "composed_model.txt").write_text(src)
    (out_dir / "winner.json").write_text(json.dumps(result, indent=2))
