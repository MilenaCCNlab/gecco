import pytest

from library_learning.compose.inventory import (
    InventoryError, compatible, parse_inventory)


def make_obj():
    return {"modules": [
        {"id": "mixture_w", "name": "MB/MF mixture", "description": "d",
         "params": [{"name": "w", "bounds": [0, 1]}],
         "slots": {}, "overrides": {"stage1_values": "w * q_stage1_mb + (1 - w) * q_stage1_mf"},
         "provenance": [1, 5], "excludes": []},
        {"id": "beta_mb_mf", "name": "separate MB/MF betas", "description": "d",
         "params": [{"name": "beta_mb", "bounds": [0, 10]}],
         "slots": {}, "overrides": {"stage1_values": "beta_mb * q_stage1_mb / beta"},
         "provenance": [2], "excludes": []},
        {"id": "stick", "name": "stickiness", "description": "d",
         "params": [{"name": "stickiness", "bounds": [0, 5]}],
         "slots": {"init": "last_action_1 = -1",
                   "stage1_logits_extra": "if last_action_1 != -1:\n    logits_1[last_action_1] += stickiness",
                   "post_trial": "if a1 != -1:\n    last_action_1 = a1"},
         "overrides": {}, "provenance": [1, 2, 7], "excludes": []},
    ]}


def test_parse_ok():
    inv = parse_inventory(make_obj())
    assert inv.ids() == ["mixture_w", "beta_mb_mf", "stick"]
    assert inv.module("stick").params[0].bounds == (0.0, 5.0)


def test_param_collision_and_bad_slot():
    obj = make_obj()
    obj["modules"][1]["params"][0]["name"] = "w"          # collides with mixture_w
    obj["modules"][2]["slots"]["nope"] = "x = 1"           # unknown slot
    with pytest.raises(InventoryError) as e:
        parse_inventory(obj)
    assert "w" in str(e.value) and "nope" in str(e.value)


def test_backbone_param_collision():
    obj = make_obj()
    obj["modules"][0]["params"][0]["name"] = "beta"
    with pytest.raises(InventoryError):
        parse_inventory(obj)


def test_compatibility_same_override_slot():
    inv = parse_inventory(make_obj())
    ok, why = compatible(inv, ["mixture_w", "beta_mb_mf"])
    assert not ok and "stage1_values" in why
    ok, _ = compatible(inv, ["mixture_w", "stick"])
    assert ok


def test_excludes():
    obj = make_obj()
    obj["modules"][0]["excludes"] = ["stick"]
    inv = parse_inventory(obj)
    ok, why = compatible(inv, ["mixture_w", "stick"])
    assert not ok and "excludes" in why


def test_provenance_string_coercion():
    obj = make_obj()
    obj["modules"][0]["provenance"] = ["1", "5"]
    inv = parse_inventory(obj)
    assert inv.module("mixture_w").provenance == [1, 5]


def test_provenance_non_integer_rejected():
    obj = make_obj()
    obj["modules"][0]["provenance"] = ["p1"]
    with pytest.raises(InventoryError):
        parse_inventory(obj)
