# tests/test_compose_extract_rlwm.py
from library_learning.compose.extract_rlwm import (
    CANONICAL_NAMES, PROMPT_ANNOTATE, PROMPT_MERGE, validate_inventory_obj)


def _mod(**kw):
    base = {"id": "m1", "name": "m", "description": "d", "params": [],
            "slots": {"post_trial": "pass"}, "overrides": {},
            "provenance": [1], "excludes": []}
    base.update(kw)
    return base


def test_prompts_are_rlwm_specific():
    for needle in ["set size", "wm_weight", "block"]:
        assert needle in PROMPT_ANNOTATE
    for needle in ["block_init", "wm_update", "probs_extra", "-2",
                   "wm_weight", "stimulus, actions, rewards, blocks, set_sizes"]:
        assert needle in PROMPT_MERGE
    assert "stage1" not in PROMPT_MERGE and "transition_matrix" not in PROMPT_MERGE


def test_canonical_names_cover_backbone_vars():
    for name in ["stimulus", "actions", "rewards", "blocks", "set_sizes",
                 "nA", "nS", "q", "w", "w_0", "probs", "logits_rl",
                 "logits_wm", "mix", "delta", "wm_weight", "np"]:
        assert name in CANONICAL_NAMES
    assert "q_stage1_mb" not in CANONICAL_NAMES


def test_validate_ok_module():
    inv, errors = validate_inventory_obj({"modules": [
        _mod(slots={"post_trial": "w += 0.1 * (w_0 - w)"})]})
    assert errors == [] and inv is not None


def test_validate_flags_unprefixed_state():
    inv, errors = validate_inventory_obj({"modules": [
        _mod(slots={"init": "my_counter = 0"})]})
    assert inv is None
    assert any("unprefixed" in e and "my_counter" in e for e in errors)


def test_validate_allows_prefixed_state_and_params():
    inv, errors = validate_inventory_obj({"modules": [
        _mod(id="stick", params=[{"name": "stick_bonus", "bounds": [0, 5]}],
             slots={"init": "stick_last_a = -1",
                    "rl_logits_extra": "if stick_last_a != -1:\n    logits_rl[stick_last_a] += stick_bonus",
                    "post_trial": "if 0 <= a < nA:\n    stick_last_a = a"})]})
    assert errors == [] and inv is not None
