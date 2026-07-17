import json
from pathlib import Path

from library_learning.config import resolve_target
from library_learning.compose.splits import group_split_pids, make_splits, parse_split

IND = "results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual"
GRP = "results/two_step_psychiatry_group_function_ocibalanced_maxsetting"


def test_parse_split_slice():
    assert parse_split("[1:3]", list(range(45))) == [1, 2]
    assert parse_split("[14:]", list(range(45))) == list(range(14, 45))


def test_group_split_pids():
    pids = group_split_pids(GRP)
    assert pids["prompt"] == [1, 2]
    assert pids["eval"] == list(range(4, 14))
    assert pids["seed"] == [1, 2] + list(range(4, 14))
    assert pids["heldout"] == list(range(14, 45))


def test_make_splits_deterministic_and_balanced(tmp_path):
    target = resolve_target(IND)
    s1 = make_splits(target, GRP, out_dir=tmp_path)
    s2 = make_splits(target, GRP, out_dir=tmp_path)
    assert s1 == s2
    assert s1["composition_validation_pids"] == list(range(4, 14))
    assert len(s1["reconstruction_pids"]) == 10 and len(s1["test_pids"]) == 21
    assert not set(s1["reconstruction_pids"]) & set(s1["test_pids"])
    assert set(s1["reconstruction_pids"]) | set(s1["test_pids"]) == set(range(14, 45))
    # OCI balance: means within 0.15 of each other
    st = s1["oci_stats"]
    assert abs(st["reconstruction_mean"] - st["test_mean"]) < 0.15
    assert json.loads((tmp_path / "splits.json").read_text()) == s1
