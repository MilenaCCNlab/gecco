# tests/test_compose_render.py
import numpy as np
import pytest

from library_learning.compose.inventory import parse_inventory
from library_learning.compose.render import (
    candidate_id, candidate_params, render_candidate, smoke_check)
from library_learning.loading import (
    exec_model, extract_unpack_names, parse_bounds)

INV = parse_inventory({"modules": [
    {"id": "mixture_w", "name": "MB/MF mixture", "description": "d",
     "params": [{"name": "w", "bounds": [0, 1]}],
     "slots": {},
     "overrides": {"stage1_values": "w * q_stage1_mb + (1 - w) * q_stage1_mf"},
     "provenance": [1], "excludes": []},
    {"id": "stick", "name": "stickiness", "description": "d",
     "params": [{"name": "stickiness", "bounds": [0, 5]}],
     "slots": {"init": "last_action_1 = -1",
               "stage1_logits_extra": "if last_action_1 != -1:\n    logits_1[last_action_1] += stickiness",
               "post_trial": "if a1 != -1:\n    last_action_1 = a1"},
     "overrides": {}, "provenance": [1, 2], "excludes": []},
]})


def test_candidate_id_and_params():
    assert candidate_id([]) == "backbone"
    assert candidate_id(["stick", "mixture_w"]) == "mixture_w+stick"
    names = [p.name for p in candidate_params(INV, ["stick", "mixture_w"])]
    assert names == ["learning_rate", "beta", "w", "stickiness"]


def test_backbone_renders_and_runs():
    src = render_candidate(INV, [])
    nll = smoke_check(src)
    assert np.isfinite(nll) and nll > 0
    # gecco compatibility
    assert extract_unpack_names(src) == ["learning_rate", "beta"]
    b = parse_bounds(src, ["learning_rate", "beta"])
    assert b["beta"] == (0.0, 10.0)


def test_full_candidate_runs_with_missing_trials():
    src = render_candidate(INV, ["mixture_w", "stick"])
    assert "def cognitive_model(" in src
    assert np.isfinite(smoke_check(src))
    assert extract_unpack_names(src) == ["learning_rate", "beta", "w", "stickiness"]


def test_bad_snippet_fails_smoke():
    inv = parse_inventory({"modules": [
        {"id": "bad", "name": "bad", "description": "d", "params": [],
         "slots": {"init": "undefined_name += 1"}, "overrides": {},
         "provenance": [], "excludes": []}]})
    with pytest.raises(Exception):
        smoke_check(render_candidate(inv, ["bad"]))


def test_blank_lines_and_docstring_survive():
    src = render_candidate(INV, ["mixture_w", "stick"])
    assert "###EMPTY_SLOT###" not in src
    doc = src.split('"""')[1]
    assert "Parameters:" in doc
    import ast as _ast
    _ast.parse(src)  # still valid source


def test_multiline_string_snippet_rejected():
    from library_learning.compose.inventory import parse_inventory
    inv = parse_inventory({"modules": [
        {"id": "bad", "name": "bad", "description": "d", "params": [],
         "slots": {"init": 'bad_label = """A\n\nB"""'}, "overrides": {},
         "provenance": [], "excludes": []}]})
    import pytest as _pytest
    with _pytest.raises(ValueError):
        render_candidate(inv, ["bad"])
