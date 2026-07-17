import json

from library_learning.compose.inventory import parse_inventory
from library_learning.compose.reconstruct import reconstruct_participants
from library_learning.config import resolve_target

IND = "results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual"
GRP = "results/two_step_psychiatry_group_function_ocibalanced_maxsetting"

SMALL_INV = parse_inventory({"modules": [
    {"id": "stick", "name": "stickiness", "description": "d",
     "params": [{"name": "stickiness", "bounds": [0, 5]}],
     "slots": {"init": "stick_last_a1 = -1",
               "stage1_logits_extra": "if stick_last_a1 != -1:\n    logits_1[stick_last_a1] += stickiness",
               "post_trial": "if a1 != -1:\n    stick_last_a1 = a1"},
     "overrides": {}, "provenance": [1], "excludes": []}]})


def test_reconstruct_one_pid(tmp_path):
    target = resolve_target(IND)
    results = reconstruct_participants(SMALL_INV, target, GRP, [14], tmp_path,
                                       mode="greedy")
    assert len(results) == 1
    r = results[0]
    assert r["pid"] == 14
    assert r["library_bic"] > 0 and r["individual_bic"] > 0 and r["group_bic"] > 0
    saved = json.loads((tmp_path / "reconstruction_results.json").read_text())
    assert saved == results
    assert (tmp_path / "reconstruction" / "p14" / "search_log.jsonl").exists()
