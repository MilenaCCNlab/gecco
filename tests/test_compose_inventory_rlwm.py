# tests/test_compose_inventory_rlwm.py
import pytest

from library_learning.compose.inventory import InventoryError
from library_learning.compose.inventory_rlwm import (
    APPEND_SLOTS, BACKBONE_PARAM_NAMES, OVERRIDE_SLOTS, parse_inventory)


def _mod(**kw):
    base = {"id": "m1", "name": "m", "description": "d", "params": [],
            "slots": {"post_trial": "pass"}, "overrides": {},
            "provenance": [1], "excludes": []}
    base.update(kw)
    return base


def test_slot_constants_are_rlwm():
    assert "block_init" in APPEND_SLOTS and "wm_logits_extra" in APPEND_SLOTS
    assert "pre_stage1" not in APPEND_SLOTS          # two-step slot must be absent
    assert "wm_update" in OVERRIDE_SLOTS and "mix_weight" in OVERRIDE_SLOTS
    assert BACKBONE_PARAM_NAMES == ("learning_rate", "beta", "wm_weight")


def test_parse_accepts_rlwm_slots_and_rejects_two_step_slots():
    inv = parse_inventory({"modules": [_mod(slots={"block_init": "pass"})]})
    assert inv.ids() == ["m1"]
    with pytest.raises(InventoryError, match="unknown append slot"):
        parse_inventory({"modules": [_mod(slots={"stage1_logits_extra": "pass"})]})
    with pytest.raises(InventoryError, match="unknown override slot"):
        parse_inventory({"modules": [_mod(slots={}, overrides={"stage1_values": "q"})]})


def test_wm_weight_param_collision_rejected():
    bad = _mod(params=[{"name": "wm_weight", "bounds": [0, 1]}])
    with pytest.raises(InventoryError, match="collision"):
        parse_inventory({"modules": [bad]})


def test_string_provenance_coerced_to_int():
    inv = parse_inventory({"modules": [_mod(provenance=["3"])]})
    assert inv.modules[0].provenance == [3]
