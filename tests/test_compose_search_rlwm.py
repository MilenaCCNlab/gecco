# tests/test_compose_search_rlwm.py
from library_learning.compose.inventory_rlwm import parse_inventory
from library_learning.compose import search_rlwm as S

INV = parse_inventory({"modules": [
    {"id": "a", "name": "a", "description": "d",
     "params": [{"name": "pa", "bounds": [0, 1]}],
     "slots": {"post_trial": "pass"}, "overrides": {}, "provenance": [1], "excludes": []},
    {"id": "b", "name": "b", "description": "d",
     "params": [{"name": "pb1", "bounds": [0, 1]}, {"name": "pb2", "bounds": [0, 1]}],
     "slots": {"post_trial": "pass"}, "overrides": {}, "provenance": [2], "excludes": []},
    {"id": "c", "name": "c", "description": "d",
     "params": [{"name": "pc", "bounds": [0, 5]}],
     "slots": {"post_trial": "pass"}, "overrides": {}, "provenance": [4], "excludes": ["a"]},
]})


def test_enumerate_uses_3_backbone_params_and_cap_6():
    cands = S.enumerate_candidates(INV)          # default cap 6
    ids = [tuple(c) for c in cands]
    assert () in ids
    # backbone(3) + a(1) + b(2) = 6 <= 6 -> allowed
    assert ("a", "b") in ids
    assert ("a", "c") not in ids                 # excluded pair
    tight = S.enumerate_candidates(INV, param_cap=4)
    assert ("a", "b") not in [tuple(c) for c in tight]
    assert ("a",) in [tuple(c) for c in tight]   # 3 + 1 = 4


def test_count_report_param_cap_default():
    rep = S.count_report(INV)
    assert rep["param_cap"] == 6
    assert rep["n_candidates"] == len(S.enumerate_candidates(INV))


def test_greedy_uses_stub_scores(tmp_path, monkeypatch):
    calls = []

    def fake_score(inventory, target, pids, candidates, out_dir):
        out = []
        for mods in candidates:
            calls.append(tuple(mods))
            bic = {(): 500.0, ("a",): 480.0, ("b",): 490.0,
                   ("a", "b"): 495.0}.get(tuple(mods), 600.0)
            out.append({"candidate_id": S.candidate_id(list(mods)),
                        "module_ids": sorted(mods),
                        "n_params": 3 + sum(INV.module(m).n_params for m in mods),
                        "per_pid": {"10": {"bic": bic}}, "mean_bic": bic})
        return out

    monkeypatch.setattr(S, "score_candidates", fake_score)
    results = S.greedy_search(INV, None, ["10"], tmp_path)
    best = S.select_winner(results)
    assert best["module_ids"] == ["a"]           # a improves, a+b doesn't
    assert () in calls and ("a",) in calls
