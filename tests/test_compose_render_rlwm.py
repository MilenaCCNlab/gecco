# tests/test_compose_render_rlwm.py
import numpy as np
import pytest

from library_learning.compose.inventory_rlwm import parse_inventory
from library_learning.compose.render_rlwm import (
    candidate_id, candidate_params, render_candidate, smoke_check)
from library_learning.loading import extract_unpack_names, parse_bounds

INV = parse_inventory({"modules": [
    {"id": "wm_decay", "name": "WM decay", "description": "d",
     "params": [{"name": "decay", "bounds": [0, 1]}],
     "slots": {"post_trial": "w += decay * (w_0 - w)"},
     "overrides": {}, "provenance": [1], "excludes": []},
    {"id": "capacity", "name": "set-size scaled WM weight", "description": "d",
     "params": [{"name": "capacity_k", "bounds": [1, 6]}],
     "slots": {},
     "overrides": {"mix_weight": "wm_weight * min(1.0, capacity_k / float(nS))"},
     "provenance": [1, 2], "excludes": []},
    {"id": "lapse", "name": "uniform lapse", "description": "d",
     "params": [{"name": "lapse_p", "bounds": [0, 1]}],
     "slots": {"probs_extra": "probs = (1.0 - lapse_p) * probs + lapse_p / nA"},
     "overrides": {}, "provenance": [2], "excludes": []},
]})


def test_candidate_id_and_params():
    assert candidate_id([]) == "backbone"
    names = [p.name for p in candidate_params(INV, ["lapse", "wm_decay"])]
    assert names == ["learning_rate", "beta", "wm_weight", "lapse_p", "decay"]


def test_backbone_renders_and_runs():
    src = render_candidate(INV, [])
    nll = smoke_check(src)
    assert np.isfinite(nll) and nll > 0
    assert "def cognitive_model(stimulus, actions, rewards, blocks, set_sizes, model_parameters)" in src
    assert extract_unpack_names(src) == ["learning_rate", "beta", "wm_weight"]
    b = parse_bounds(src, ["learning_rate", "beta", "wm_weight"])
    assert b["beta"] == (0.0, 10.0) and b["wm_weight"] == (0.0, 1.0)


def test_full_candidate_runs_with_missed_trials():
    src = render_candidate(INV, ["wm_decay", "capacity", "lapse"])
    assert np.isfinite(smoke_check(src))
    assert extract_unpack_names(src) == [
        "learning_rate", "beta", "wm_weight", "capacity_k", "lapse_p", "decay"]


def test_missed_trials_skip_likelihood():
    # NLL of backbone on smoke data must not change when a missed trial's
    # reward value changes (the trial is fully skipped except post_trial).
    from library_learning.compose import render_rlwm as R
    from library_learning.loading import exec_model
    src = render_candidate(INV, [])
    func = exec_model(src)
    params = [0.5, 5.0, 0.5]
    base = smoke_check(src, params=params)
    data = {k: v.copy() for k, v in R.SMOKE_DATA.items()}
    missed = np.where(data["actions"] == -2)[0]
    assert len(missed) > 0
    data["rewards"][missed[0]] = 1
    nll = float(func(data["stimulus"], data["actions"], data["rewards"],
                     data["blocks"], data["set_sizes"], params))
    assert abs(nll - base) < 1e-12


def test_multiline_update_overrides_render():
    # rl_update/wm_update overrides are statements and may span lines
    # (if/else WM updates were the norm in the first extraction run).
    inv = parse_inventory({"modules": [{
        "id": "wm_two_line", "name": "t", "description": "d", "params": [],
        "slots": {},
        "overrides": {
            "rl_update": "if r > 0:\n    q[s, a] += learning_rate * delta\nelse:\n    q[s, a] += 0.5 * learning_rate * delta",
            "wm_update": "if r > 0.5:\n    w[s, a] = 1.0\nelse:\n    w[s, a] = 0.0"},
        "provenance": [1], "excludes": []}]})
    assert np.isfinite(smoke_check(render_candidate(inv, ["wm_two_line"])))


def test_bad_snippet_fails_smoke():
    inv = parse_inventory({"modules": [
        {"id": "bad", "name": "bad", "description": "d", "params": [],
         "slots": {"init": "undefined_name += 1"}, "overrides": {},
         "provenance": [], "excludes": []}]})
    with pytest.raises(Exception):
        smoke_check(render_candidate(inv, ["bad"]))


def test_multiline_string_snippet_rejected():
    inv = parse_inventory({"modules": [
        {"id": "ml", "name": "ml", "description": "d", "params": [],
         "slots": {"init": 'ml_x = """a\nb"""'}, "overrides": {},
         "provenance": [], "excludes": []}]})
    with pytest.raises(ValueError, match="multi-line string"):
        render_candidate(inv, ["ml"])
