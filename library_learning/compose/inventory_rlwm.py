# library_learning/compose/inventory_rlwm.py
"""RLWM module inventory: slot constants + parser. Parallel sibling of
inventory.py (two-step) — reuses its task-agnostic dataclasses and
compatibility check; only the slot contract and backbone params differ."""
import json
from pathlib import Path

from .inventory import (Inventory, InventoryError, Module, MODULE_ID_RE,
                        Param, compatible)  # noqa: F401 (compatible re-exported)

APPEND_SLOTS = ("init", "block_init", "pre_choice", "rl_logits_extra",
                "wm_logits_extra", "probs_extra", "update_extra", "post_trial")
OVERRIDE_SLOTS = ("q_init", "w_init", "rl_values", "wm_values", "rl_temp",
                  "wm_temp", "mix_weight", "rl_update", "wm_update")
BACKBONE_PARAM_NAMES = ("learning_rate", "beta", "wm_weight")


def parse_inventory(obj):
    errors = []
    modules = []
    seen_ids = set()
    seen_params = set(BACKBONE_PARAM_NAMES)
    for raw in obj.get("modules", []):
        mid = raw.get("id", "<missing id>")
        if mid in seen_ids:
            errors.append("duplicate module id: %s" % mid)
        if not MODULE_ID_RE.match(mid):
            errors.append("invalid module id '%s' (must be snake_case)" % mid)
        seen_ids.add(mid)
        params = []
        for p in raw.get("params", []):
            name = p.get("name", "")
            try:
                lo, hi = float(p["bounds"][0]), float(p["bounds"][1])
                if not lo < hi:
                    errors.append("%s.%s: bounds lo >= hi" % (mid, name))
            except Exception:
                errors.append("%s.%s: non-numeric bounds" % (mid, name))
                lo, hi = 0.0, 1.0
            if name in seen_params:
                errors.append("param name collision: '%s' (module %s)" % (name, mid))
            seen_params.add(name)
            params.append(Param(name=name, bounds=(lo, hi)))
        slots = dict(raw.get("slots", {}))
        overrides = dict(raw.get("overrides", {}))
        for s in slots:
            if s not in APPEND_SLOTS:
                errors.append("%s: unknown append slot '%s'" % (mid, s))
        for s in overrides:
            if s not in OVERRIDE_SLOTS:
                errors.append("%s: unknown override slot '%s'" % (mid, s))
        if not slots and not overrides:
            errors.append("%s: module has no code (empty slots and overrides)" % mid)
        provenance = []
        for p in raw.get("provenance", []):
            try:
                provenance.append(int(p))
            except (TypeError, ValueError):
                errors.append("%s: non-integer provenance entry %r" % (mid, p))
        modules.append(Module(
            id=mid, name=raw.get("name", mid), description=raw.get("description", ""),
            params=params, slots=slots, overrides=overrides,
            provenance=provenance,
            excludes=list(raw.get("excludes", []))))
    for m in modules:
        for ex in m.excludes:
            if ex not in seen_ids:
                errors.append("%s: excludes unknown module '%s'" % (m.id, ex))
    if errors:
        raise InventoryError("; ".join(errors))
    return Inventory(modules=modules)


def load_inventory(path):
    return parse_inventory(json.loads(Path(path).read_text()))
