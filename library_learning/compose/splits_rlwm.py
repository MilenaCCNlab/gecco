# library_learning/compose/splits_rlwm.py
"""RLWM participant splits. Parallel sibling of splits.py (two-step).

Differences from two-step: seeds are the group-seen pids (prompt ∪ eval)
intersected with the individually-fitted set (config/rlwm.yaml's eval window
[10:20] includes pids 15-19 that have no individual fits — disclosed);
the held-out pool is fitted-minus-group-seen (this also excludes the
config's eval/test overlap pids 14-19 from testing); stratification uses
age (no OCI column in this dataset)."""
import json
from pathlib import Path

from .splits import group_split_pids
from ..loading import load_dataframe, participant_ids


def make_splits(target, group_dir, out_dir=None, age_column="age"):
    out_dir = Path(out_dir) if out_dir else target.results_dir / "library_composition"
    out_dir.mkdir(parents=True, exist_ok=True)
    pids = group_split_pids(group_dir, data_path=target.data_path)
    fitted = set(participant_ids(target))
    group_seen = sorted(set(pids["prompt"]) | set(pids["eval"]))
    seed = [p for p in group_seen if p in fitted]
    excluded = [p for p in group_seen if p not in fitted]
    if not seed:
        raise ValueError("no group-seen pid has an individual fit in %s"
                         % target.models_dir)
    pool = sorted(fitted - set(group_seen))
    df = load_dataframe(target)
    age = df.groupby(target.id_column)[age_column].first()
    ranked = sorted(pool, key=lambda p: (float(age[p]), p))
    reconstruction = [p for i, p in enumerate(ranked) if i % 3 == 1]
    test = [p for p in pool if p not in reconstruction]
    result = {
        "seed_pids": seed,
        "seed_pids_excluded_unfitted": excluded,
        "composition_validation_pids": sorted(pids["eval"]),
        "reconstruction_pids": sorted(reconstruction),
        "test_pids": sorted(test),
        "method": ("held-out pool = fitted minus group-seen (prompt+eval); "
                   "age-sorted alternation i%3==1"),
        "age_stats": {
            "reconstruction_mean": float(age[reconstruction].mean()),
            "reconstruction_std": float(age[reconstruction].std()),
            "test_mean": float(age[test].mean()),
            "test_std": float(age[test].std()),
        },
    }
    (out_dir / "splits.json").write_text(json.dumps(result, indent=2))
    return result
