"""Stage 2b: per-participant best library composition on unseen participants
(library-coverage metric; kept separate from the single-program claim)."""
import json
from pathlib import Path

from .fitting import fit_model_on_pids
from .search import (enumerate_candidates, greedy_search, score_candidates,
                     select_winner)
from ..loading import (bounds_for_code, function_name_and_args,
                       load_original_code, strip_fences)


def reconstruct_participants(inventory, target, group_dir, pids, out_dir,
                             mode="greedy"):
    out_dir = Path(out_dir)
    group_src = strip_fences(
        (Path(group_dir) / "models" / "best_model_0.txt").read_text())
    group_fname, _ = function_name_and_args(group_src)

    cands = enumerate_candidates(inventory) if mode == "exhaustive" else None

    results = []
    for pid in pids:
        pid_dir = out_dir / "reconstruction" / ("p%d" % pid)
        pid_dir.mkdir(parents=True, exist_ok=True)
        if mode == "exhaustive":
            recs = score_candidates(inventory, target, [pid], cands, pid_dir)
        else:
            recs = greedy_search(inventory, target, [pid], pid_dir)
        best = select_winner(recs)

        own_code = load_original_code(target, pid)
        own_fname, _ = function_name_and_args(own_code)
        own = fit_model_on_pids(own_code, target, [pid], bounds_for_code(own_code),
                                tag="recon:individual", func_name=own_fname)
        grp = fit_model_on_pids(group_src, target, [pid], bounds_for_code(group_src),
                                tag="recon:group", func_name=group_fname)
        results.append({
            "pid": pid,
            "best_candidate_id": best["candidate_id"],
            "best_module_ids": best["module_ids"],
            "best_n_params": best["n_params"],
            "library_bic": best["mean_bic"],  # mean over one pid == that pid's BIC
            "individual_bic": own[pid]["bic"],
            "group_bic": grp[pid]["bic"],
        })
        print("[reconstruct] p%d library %.2f vs individual %.2f vs group %.2f"
              % (pid, best["mean_bic"], own[pid]["bic"], grp[pid]["bic"]))
    (out_dir / "reconstruction_results.json").write_text(
        json.dumps(results, indent=2))
    return results
