# tests/test_compose_splits_rlwm.py
import json

from library_learning.config import resolve_target
from library_learning.compose.splits_rlwm import make_splits

IND = "results/rlwm_individual"
GRP = "results/rlwm"


def test_make_splits_matches_frozen_spec(tmp_path):
    target = resolve_target(IND)
    s1 = make_splits(target, GRP, out_dir=tmp_path)
    s2 = make_splits(target, GRP, out_dir=tmp_path)
    assert s1 == s2                                      # deterministic
    assert s1["seed_pids"] == [1, 2, 3, 10, 11, 12, 13, 14]
    assert s1["seed_pids_excluded_unfitted"] == [15, 16, 17, 18, 19]
    assert s1["composition_validation_pids"] == list(range(10, 20))
    assert s1["reconstruction_pids"] == [0, 5, 37, 40, 45, 46, 49]
    assert s1["test_pids"] == [4, 6, 7, 8, 9, 36, 38, 39,
                               41, 42, 43, 44, 47, 48, 50]
    assert not set(s1["reconstruction_pids"]) & set(s1["test_pids"])
    # age balance: subset means within 10 years of each other
    st = s1["age_stats"]
    assert abs(st["reconstruction_mean"] - st["test_mean"]) < 10.0
    assert json.loads((tmp_path / "splits.json").read_text()) == s1
