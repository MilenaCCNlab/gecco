import json

from library_learning.compose.extract import (
    _parse_json_reply, audit_coverage, validate_inventory_obj)


def make_good():
    return {"modules": [
        {"id": "stick", "name": "stickiness", "description": "d",
         "params": [{"name": "stickiness", "bounds": [0, 5]}],
         "slots": {"init": "stick_last_a1 = -1",
                   "stage1_logits_extra": "if stick_last_a1 != -1:\n    logits_1[stick_last_a1] += stickiness",
                   "post_trial": "if a1 != -1:\n    stick_last_a1 = a1"},
         "overrides": {}, "provenance": [1], "excludes": []}]}


def test_parse_json_reply_strips_fences():
    obj = _parse_json_reply('```json\n{"mechanisms": []}\n```')
    assert obj == {"mechanisms": []}
    obj = _parse_json_reply('{"a": 1}')
    assert obj == {"a": 1}


def test_validate_inventory_smoke_catches_bad_code():
    inv, errors = validate_inventory_obj(make_good())
    assert errors == [] and inv is not None

    bad = make_good()
    bad["modules"][0]["slots"]["init"] = "stick_last_a1 = undefined_thing"
    inv, errors = validate_inventory_obj(bad)
    assert errors and "stick" in errors[0]


def test_validate_flags_unprefixed_state_vars():
    bad = make_good()
    bad["modules"][0]["slots"]["init"] = "last_a1 = -1"
    bad["modules"][0]["slots"]["stage1_logits_extra"] = (
        "if last_a1 != -1:\n    logits_1[last_a1] += stickiness")
    bad["modules"][0]["slots"]["post_trial"] = "if a1 != -1:\n    last_a1 = a1"
    inv, errors = validate_inventory_obj(bad)
    assert any("unprefixed" in e for e in errors)


def test_validate_smokes_pairs():
    obj = make_good()
    obj["modules"].append(
        {"id": "decay", "name": "decay", "description": "d",
         "params": [{"name": "decay_rate", "bounds": [0, 1]}],
         "slots": {"post_trial": "q_stage2_mf *= (1.0 - decay_rate)"},
         "overrides": {}, "provenance": [2], "excludes": []})
    inv, errors = validate_inventory_obj(obj)
    assert errors == []  # pair (stick, decay) smoke-tested together


def test_validate_flags_destructured_and_loop_state_vars():
    bad = make_good()
    bad["modules"][0]["slots"]["init"] = "a_var, b_var = -1, 0.0"
    inv, errors = validate_inventory_obj(bad)
    assert any("unprefixed" in e and "a_var" in e for e in errors)

    bad2 = make_good()
    bad2["modules"][0]["slots"]["init"] = "for leaked in range(2):\n    pass"
    inv, errors = validate_inventory_obj(bad2)
    assert any("unprefixed" in e and "leaked" in e for e in errors)


def test_parse_json_reply_prefers_valid_json_fence():
    reply = ("Here is an example:\n```python\nnot json\n```\n"
             "```json\n{\"modules\": []}\n```")
    assert _parse_json_reply(reply) == {"modules": []}


def test_audit_coverage():
    obj = {"modules": [], "coverage": [
        {"pid": 1, "mechanism": "stickiness", "module": "stick",
         "decision": "mapped"}]}
    annotations = {"1": {"mechanisms": [{"name": "stickiness"},
                                        {"name": "decay"}]}}
    errors = audit_coverage(obj, annotations)
    assert len(errors) == 1 and "decay" in errors[0]


def test_audit_coverage_string_pids():
    obj = {"modules": [], "coverage": [
        {"pid": "1", "mechanism": "stickiness", "module": "stick",
         "decision": "mapped"}]}
    annotations = {"1": {"mechanisms": [{"name": "stickiness"}]}}
    assert audit_coverage(obj, annotations) == []
