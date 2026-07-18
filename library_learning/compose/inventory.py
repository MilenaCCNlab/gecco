"""Module inventory: the validated data model behind module_inventory.json."""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

APPEND_SLOTS = ("init", "pre_stage1", "stage1_logits_extra",
                "stage2_logits_extra", "update_extra", "post_trial")
OVERRIDE_SLOTS = ("q2_init", "stage1_values", "stage2_values",
                  "stage1_temp", "stage2_temp", "stage1_update", "stage2_update")
BACKBONE_PARAM_NAMES = ("learning_rate", "beta")
MODULE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class InventoryError(ValueError):
    pass


@dataclass
class Param:
    name: str
    bounds: Tuple[float, float]


@dataclass
class Module:
    id: str
    name: str
    description: str
    params: List[Param]
    slots: Dict[str, str]
    overrides: Dict[str, str]
    provenance: List[int]
    excludes: List[str] = field(default_factory=list)

    @property
    def n_params(self):
        return len(self.params)


@dataclass
class Inventory:
    modules: List[Module]
    _by_id: Dict[str, Module] = field(default=None, repr=False, compare=False)

    def ids(self):
        return [m.id for m in self.modules]

    def module(self, mid):
        if self._by_id is None:
            self._by_id = {m.id: m for m in self.modules}
        return self._by_id[mid]


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


def compatible(inventory, module_ids):
    mods = [inventory.module(mid) for mid in module_ids]
    for i, a in enumerate(mods):
        for b in mods[i + 1:]:
            if b.id in a.excludes or a.id in b.excludes:
                return False, "%s excludes %s" % (a.id, b.id)
            shared = set(a.overrides) & set(b.overrides)
            if shared:
                return False, "%s and %s both override %s" % (
                    a.id, b.id, sorted(shared)[0])
    return True, ""
