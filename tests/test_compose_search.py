import json

import pytest

from library_learning.compose.inventory import parse_inventory
from library_learning.compose import search as S


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


def test_enumerate_respects_cap_and_excludes():
    cands = S.enumerate_candidates(INV, param_cap=8)
    ids = [S_id for S_id in (tuple(c) for c in cands)]
    assert () in ids                       # backbone
    assert ("a", "c") not in ids           # excluded pair
    assert ("a", "b", "c") not in ids
    # cap: backbone(2) + a(1) + b(2) = 5 <= 8 -> allowed
    assert ("a", "b") in ids
    tight = S.enumerate_candidates(INV, param_cap=3)
    assert ("a", "b") not in tight and ("a",) in tight


def test_count_report():
    rep = S.count_report(INV, param_cap=8)
    assert rep["n_candidates"] == len(S.enumerate_candidates(INV, 8))
    assert rep["n_modules"] == 3


def test_select_winner_tiebreak():
    results = [
        {"candidate_id": "a", "n_params": 3, "mean_bic": 400.5},
        {"candidate_id": "b", "n_params": 4, "mean_bic": 400.0},  # within 1.0 of a
        {"candidate_id": "backbone", "n_params": 2, "mean_bic": 420.0},
    ]
    assert S.select_winner(results)["candidate_id"] == "a"  # fewer params wins tie


def test_selection_report(tmp_path):
    results = []
    for cid, base in [("x", 400.0), ("y", 405.0), ("z", 500.0)]:
        results.append({"candidate_id": cid, "module_ids": [cid], "n_params": 3,
                        "per_pid": {"4": {"bic": base - 5}, "5": {"bic": base + 5}},
                        "mean_bic": base})
    rep = S.selection_report(results, tmp_path, k=2)
    assert [t["candidate_id"] for t in rep["top_k"]] == ["x", "y"]
    assert set(rep["loo_ranks"]) == {"x", "y"}
    assert rep["loo_ranks"]["x"] == [1, 1]
    assert rep["loo_pids"] == ["4", "5"]
    import json as _json
    assert _json.loads((tmp_path / "selection_report.json").read_text()) == rep


def test_greedy_uses_stub_scores(tmp_path, monkeypatch):
    calls = []

    def fake_score(inventory, target, pids, candidates, out_dir):
        out = []
        for mods in candidates:
            mean = 500.0 - 30.0 * ("a" in mods) - 10.0 * ("b" in mods) + 5.0 * ("c" in mods)
            rec = {"candidate_id": S.candidate_id_of(mods), "module_ids": sorted(mods),
                   "n_params": 2 + sum({"a": 1, "b": 2, "c": 1}[m] for m in mods),
                   "per_pid": {}, "mean_bic": mean}
            calls.append(rec["candidate_id"])
            out.append(rec)
        return out

    monkeypatch.setattr(S, "score_candidates", fake_score)
    results = S.greedy_search(INV, target=None, validation_pids=[14],
                              out_dir=tmp_path, param_cap=8)
    best = S.select_winner(results)
    assert best["module_ids"] == ["a", "b"]  # c never helps
