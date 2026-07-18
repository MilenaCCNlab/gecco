# RLWM Library Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replicate the two-step library-composition pipeline on the RLWM aging dataset: extract a cognitive-module library from 8 young seed participants' individual gecco programs, compose a single program, and test generalization on 22 held-out participants (7 young + 15 old).

**Architecture:** Parallel `*_rlwm.py` siblings inside `library_learning/compose/` — the two-step modules (`inventory.py`, `render.py`, `extract.py`, `splits.py`, `search.py`, `reconstruct.py`, `evaluate.py`, `hybrid.py`, `figure.py`) are never edited; task-agnostic pieces (`fitting.py`, `gemini.py`, `loading.py`, `mining.py`, dataclasses, stats helpers) are reused via import. `library_learning/__main__.py` dispatches to the rlwm module set when the target's resolved task name is `rlwm`.

**Tech Stack:** Python 3.9 (`gecco-env/` venv at repo root), numpy/pandas/scipy/pyyaml, pytest, Gemini REST via the existing logged `GeminiClient`.

**Spec:** `docs/superpowers/specs/2026-07-18-rlwm-library-composition-design.md`

## Global Constraints

- NEVER modify these files: `library_learning/compose/{inventory,render,extract,splits,search,reconstruct,evaluate,hybrid,figure}.py`. The ONLY existing file that changes is `library_learning/__main__.py` (dispatch wiring).
- All LLM calls: `gemini-3.1-pro-preview`, temperature 0, via `library_learning/compose/gemini.py` (`GEMINI_API_KEY_LAKELAB` in `.env`; fallback `GEMINI_API_KEY_COCOSCILAB`; plain `GEMINI_API_KEY` is invalid). No OpenAI calls anywhere.
- Coerce LLM-returned pids to `int` at every JSON boundary.
- RLWM model signature: `cognitive_model(stimulus, actions, rewards, blocks, set_sizes, model_parameters)`. Missed trials: `actions == -2`.
- Backbone params: `learning_rate` [0,1], `beta` [0,10], `wm_weight` [0,1]. Parameter cap: **6**.
- Frozen splits (from `config/rlwm.yaml` prompt `[1:4]`, eval `[10:20]`, test `[14:]`, fitted = pids 0–14 ∪ 36–50): seeds `[1,2,3,10,11,12,13,14]`; excluded-unfitted `[15,16,17,18,19]`; validation `[10..19]`; held-out pool = fitted − group-seen = `[0,4,5,6,7,8,9,36..50]`; recon (age-sorted `i%3==1`) `[0,5,37,40,45,46,49]`; test `[4,6,7,8,9,36,38,39,41,42,43,44,47,48,50]`.
- Run tests with `gecco-env/bin/python -m pytest`. Use `set -o pipefail` before any piped shell command.
- Commit after every task with a `feat:`/`test:`/`run:` prefix and the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.
- **Autonomous mode (user directive 2026-07-18, before bed):** when a decision point or surprise comes up, make the most reasonable assumption, log it with rationale in `results/rlwm_individual/library_composition/DECISIONS.md`, and continue. Final deliverable includes a self-contained `report.html` mirroring the two-step run's report.

---

### Task 1: RLWM inventory (slot constants + parser)

**Files:**
- Create: `library_learning/compose/inventory_rlwm.py`
- Test: `tests/test_compose_inventory_rlwm.py`

**Interfaces:**
- Consumes: `Inventory`, `InventoryError`, `Module`, `Param`, `MODULE_ID_RE`, `compatible` from `library_learning/compose/inventory.py` (unmodified imports).
- Produces: `APPEND_SLOTS`, `OVERRIDE_SLOTS`, `BACKBONE_PARAM_NAMES` (tuples), `parse_inventory(obj) -> Inventory`, `load_inventory(path) -> Inventory`. Later tasks import these from `.inventory_rlwm`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_inventory_rlwm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'library_learning.compose.inventory_rlwm'`

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `gecco-env/bin/python -m pytest tests/test_compose_inventory_rlwm.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/inventory_rlwm.py tests/test_compose_inventory_rlwm.py
git commit -m "feat: RLWM inventory slot contract (parallel to two-step inventory)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: RLWM backbone renderer

**Files:**
- Create: `library_learning/compose/render_rlwm.py`
- Test: `tests/test_compose_render_rlwm.py`

**Interfaces:**
- Consumes: `APPEND_SLOTS` from `.inventory_rlwm`; `Param` from `.inventory`; `candidate_id`, `_indent` from `.render` (pure helpers, imported unmodified); `exec_model`, `extract_unpack_names`, `parse_bounds` from `..loading`.
- Produces: `BACKBONE_PARAMS` (list of `Param`), `DEFAULT_OVERRIDES` (dict), `SMOKE_DATA` (dict of np arrays), `candidate_params(inventory, module_ids) -> [Param]`, `render_candidate(inventory, module_ids) -> str`, `smoke_check(source, params=None) -> float`. Also re-exports `candidate_id`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_render_rlwm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'library_learning.compose.render_rlwm'`

- [ ] **Step 3: Write the implementation**

```python
# library_learning/compose/render_rlwm.py
"""Assemble RLWM backbone + module slots into one standalone cognitive_model.

The backbone is the shared skeleton the gecco RLWM fill-in template forced on
every seed model: block loop with per-block q/w/w_0 reset to 1/nA, RL softmax
(beta), near-deterministic WM softmax (temp 50), probability-level RL/WM
mixture (wm_weight), delta-rule Q update — hardened for -2 missed trials
(likelihood and updates skipped; post_trial still runs). Parallel sibling of
render.py (two-step); that file stays untouched.
"""
import ast
import textwrap

import numpy as np

from .inventory import Param
from .inventory_rlwm import APPEND_SLOTS
from .render import _indent, candidate_id  # noqa: F401 (candidate_id re-exported)
from ..loading import exec_model, extract_unpack_names, parse_bounds

EMPTY_SLOT_SENTINEL = "###EMPTY_SLOT###"

BACKBONE_PARAMS = [Param("learning_rate", (0.0, 1.0)),
                   Param("beta", (0.0, 10.0)),
                   Param("wm_weight", (0.0, 1.0))]

DEFAULT_OVERRIDES = {
    "q_init": "(1.0 / nA) * np.ones((nS, nA))",
    "w_init": "(1.0 / nA) * np.ones((nS, nA))",
    "rl_values": "q[s]",
    "wm_values": "w[s]",
    "rl_temp": "beta",
    "wm_temp": "50.0",
    "mix_weight": "wm_weight",
    "rl_update": "q[s, a] += learning_rate * delta",
    "wm_update": "w[s, a] = r",
}

BACKBONE_PARAM_DOCS = {
    "learning_rate": "RL delta-rule learning rate.",
    "beta": "RL softmax inverse temperature.",
    "wm_weight": "WM weight in the RL/WM policy mixture.",
}


def candidate_params(inventory, module_ids):
    params = list(BACKBONE_PARAMS)
    for mid in sorted(module_ids):
        params.extend(inventory.module(mid).params)
    return params


def render_candidate(inventory, module_ids):
    mods = [inventory.module(mid) for mid in sorted(module_ids)]
    params = candidate_params(inventory, module_ids)

    overrides = dict(DEFAULT_OVERRIDES)
    for m in mods:
        overrides.update(m.overrides)
    appends = {slot: [] for slot in APPEND_SLOTS}
    for m in mods:
        for slot, code in m.slots.items():
            appends[slot].append(code)

    for m in mods:
        for slot, code in m.slots.items():
            tree = ast.parse(textwrap.dedent(code))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and "\n" in node.value):
                    raise ValueError(
                        "module snippet contains a multi-line string constant, "
                        "unsupported: %s/%s" % (m.id, slot))

    doc_lines = ["Composed cognitive model: backbone"]
    if mods:
        doc_lines[0] += " + " + ", ".join(m.name for m in mods)
    doc_lines += ["", "Parameters:"]
    for p in params:
        desc = BACKBONE_PARAM_DOCS.get(p.name, "module parameter")
        lo = "%g" % p.bounds[0]
        hi = "%g" % p.bounds[1]
        doc_lines.append("%s: [%s, %s] - %s" % (p.name, lo, hi, desc))
    docstring = "\n    ".join(doc_lines)
    unpack = ", ".join(p.name for p in params)
    if len(params) == 1:
        unpack += ","

    def block(slot, level):
        parts = [_indent(c, level) for c in appends[slot]]
        if not parts:
            return _indent(EMPTY_SLOT_SENTINEL, level)
        return "\n".join(parts) + "\n"

    src = '''def cognitive_model(stimulus, actions, rewards, blocks, set_sizes, model_parameters):
    """
    {docstring}
    """
    {unpack} = model_parameters
    nA = 3
    log_loss = 0.0
    eps = 1e-10
{init}
    for b in np.unique(blocks):
        block_mask = blocks == b
        block_states = stimulus[block_mask]
        block_actions = actions[block_mask]
        block_rewards = rewards[block_mask]
        nS = int(set_sizes[block_mask][0])
        q = {q_init}
        w = {w_init}
        w_0 = (1.0 / nA) * np.ones((nS, nA))
{block_init}
        for trial in range(len(block_states)):
            s = int(block_states[trial])
            a = int(block_actions[trial])
            r = float(block_rewards[trial])
            if 0 <= s < nS and 0 <= a < nA:
{pre_choice}
                rl_values = {rl_values}
                logits_rl = ({rl_temp}) * rl_values
{rl_logits_extra}
                exp_rl = np.exp(logits_rl - np.max(logits_rl))
                probs_rl = exp_rl / np.sum(exp_rl)
                wm_values = {wm_values}
                logits_wm = ({wm_temp}) * wm_values
{wm_logits_extra}
                exp_wm = np.exp(logits_wm - np.max(logits_wm))
                probs_wm = exp_wm / np.sum(exp_wm)
                mix = {mix_weight}
                probs = mix * probs_wm + (1.0 - mix) * probs_rl
{probs_extra}
                log_loss -= np.log(probs[a] + eps)
                delta = r - q[s, a]
                {rl_update}
                {wm_update}
{update_extra}
{post_trial}
    return log_loss
'''.format(
        docstring=docstring,
        unpack=unpack,
        q_init=overrides["q_init"],
        w_init=overrides["w_init"],
        rl_values=overrides["rl_values"],
        rl_temp=overrides["rl_temp"],
        wm_values=overrides["wm_values"],
        wm_temp=overrides["wm_temp"],
        mix_weight=overrides["mix_weight"],
        rl_update=overrides["rl_update"],
        wm_update=overrides["wm_update"],
        init=block("init", 1),
        block_init=block("block_init", 2),
        pre_choice=block("pre_choice", 4),
        rl_logits_extra=block("rl_logits_extra", 4),
        wm_logits_extra=block("wm_logits_extra", 4),
        probs_extra=block("probs_extra", 4),
        update_extra=block("update_extra", 4),
        post_trial=block("post_trial", 3),
    )
    src = "\n".join(
        line for line in src.splitlines() if line.strip() != EMPTY_SLOT_SENTINEL
    ) + "\n"
    return src


# Two blocks (set sizes 3 and 6), -2-coded missed trials in both.
SMOKE_DATA = {
    "stimulus":  np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 3]),
    "actions":   np.array([0, 2, -2, 1, 0, 2, 1, -2, 0, 2, 1, 0, 2, 1]),
    "rewards":   np.array([1, 0, -2, 1, 1, 0, 0, -2, 1, 0, 1, 1, 0, 1]),
    "blocks":    np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
    "set_sizes": np.array([3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6]),
}


def smoke_check(source, params=None):
    """Exec + run on dummy data with -2 missed trials; return NLL or raise."""
    func = exec_model(source, "cognitive_model")
    if params is None:
        names = extract_unpack_names(source)
        bounds_map = parse_bounds(source, names)
        params = [(bounds_map[n][0] + bounds_map[n][1]) / 2.0 for n in names]
    nll = float(func(SMOKE_DATA["stimulus"], SMOKE_DATA["actions"],
                     SMOKE_DATA["rewards"], SMOKE_DATA["blocks"],
                     SMOKE_DATA["set_sizes"], params))
    if not np.isfinite(nll):
        raise ValueError("smoke_check: non-finite NLL %r" % nll)
    return nll
```

- [ ] **Step 4: Run test to verify it passes**

Run: `gecco-env/bin/python -m pytest tests/test_compose_render_rlwm.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/render_rlwm.py tests/test_compose_render_rlwm.py
git commit -m "feat: RLWM backbone renderer with block-looped RL+WM slot contract

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: RLWM splits (age-stratified, fitted-pool intersection)

**Files:**
- Create: `library_learning/compose/splits_rlwm.py`
- Test: `tests/test_compose_splits_rlwm.py`

**Interfaces:**
- Consumes: `group_split_pids` from `.splits` (unmodified); `load_dataframe`, `participant_ids` from `..loading`; `resolve_target` from `..config`.
- Produces: `make_splits(target, group_dir, out_dir=None, age_column="age") -> dict` writing `splits.json` with keys `seed_pids`, `seed_pids_excluded_unfitted`, `composition_validation_pids`, `reconstruction_pids`, `test_pids`, `method`, `age_stats`. Reading back uses the existing generic `splits.load_splits(target)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose_splits_rlwm.py
import json

from library_learning.config import resolve_target
from library_learning.compose.splits_rlwm import make_splits

IND = "results/rlwm_individual"
GRP = "results/rlwm"


def test_make_splits_matches_frozen_spec(tmp_path):
    target = resolve_target(IND)
    s1 = make_splits(target, GRP, out_dir=tmp_path)
    s2 = make_splits(target, GRP, out_dir=tmp_path)
    assert s1 == s2                                      # deterministic
    assert s1["seed_pids"] == [1, 2, 3, 10, 11, 12, 13, 14]
    assert s1["seed_pids_excluded_unfitted"] == [15, 16, 17, 18, 19]
    assert s1["composition_validation_pids"] == list(range(10, 20))
    assert s1["reconstruction_pids"] == [0, 5, 37, 40, 45, 46, 49]
    assert s1["test_pids"] == [4, 6, 7, 8, 9, 36, 38, 39,
                               41, 42, 43, 44, 47, 48, 50]
    assert not set(s1["reconstruction_pids"]) & set(s1["test_pids"])
    # age balance: subset means within 10 years of each other
    st = s1["age_stats"]
    assert abs(st["reconstruction_mean"] - st["test_mean"]) < 10.0
    assert json.loads((tmp_path / "splits.json").read_text()) == s1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_splits_rlwm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'library_learning.compose.splits_rlwm'`

- [ ] **Step 3: Write the implementation**

```python
# library_learning/compose/splits_rlwm.py
"""RLWM participant splits. Parallel sibling of splits.py (two-step).

Differences from two-step: seeds are the group-seen pids (prompt ∪ eval)
intersected with the individually-fitted set (config/rlwm.yaml's eval window
[10:20] includes pids 15-19 that have no individual fits — disclosed);
the held-out pool is fitted-minus-group-seen (this also excludes the
config's eval/test overlap pids 14-19 from testing); stratification uses
age (no OCI column in this dataset)."""
import json
from pathlib import Path

from .splits import group_split_pids
from ..loading import load_dataframe, participant_ids


def make_splits(target, group_dir, out_dir=None, age_column="age"):
    out_dir = Path(out_dir) if out_dir else target.results_dir / "library_composition"
    out_dir.mkdir(parents=True, exist_ok=True)
    pids = group_split_pids(group_dir, data_path=target.data_path)
    fitted = set(participant_ids(target))
    group_seen = sorted(set(pids["prompt"]) | set(pids["eval"]))
    seed = [p for p in group_seen if p in fitted]
    excluded = [p for p in group_seen if p not in fitted]
    if not seed:
        raise ValueError("no group-seen pid has an individual fit in %s"
                         % target.models_dir)
    pool = sorted(fitted - set(group_seen))
    df = load_dataframe(target)
    age = df.groupby(target.id_column)[age_column].first()
    ranked = sorted(pool, key=lambda p: (float(age[p]), p))
    reconstruction = [p for i, p in enumerate(ranked) if i % 3 == 1]
    test = [p for p in pool if p not in reconstruction]
    result = {
        "seed_pids": seed,
        "seed_pids_excluded_unfitted": excluded,
        "composition_validation_pids": sorted(pids["eval"]),
        "reconstruction_pids": sorted(reconstruction),
        "test_pids": sorted(test),
        "method": ("held-out pool = fitted minus group-seen (prompt+eval); "
                   "age-sorted alternation i%3==1"),
        "age_stats": {
            "reconstruction_mean": float(age[reconstruction].mean()),
            "reconstruction_std": float(age[reconstruction].std()),
            "test_mean": float(age[test].mean()),
            "test_std": float(age[test].std()),
        },
    }
    (out_dir / "splits.json").write_text(json.dumps(result, indent=2))
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `gecco-env/bin/python -m pytest tests/test_compose_splits_rlwm.py -q`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/splits_rlwm.py tests/test_compose_splits_rlwm.py
git commit -m "feat: RLWM age-stratified splits over the fitted held-out pool

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Canonical RLWM baseline (Collins & Frank)

**Files:**
- Create: `library_learning/compose/canonical_rlwm.py`
- Test: `tests/test_compose_canonical_rlwm.py`

**Interfaces:**
- Consumes: nothing from other rlwm tasks.
- Produces: `CANONICAL_SOURCE` (str, a full `cognitive_model` definition), `CANONICAL_BOUNDS` (list of 6 `(lo, hi)` tuples, unpack order). `evaluate_rlwm.py` (Task 8) fits this source with these bounds.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose_canonical_rlwm.py
import numpy as np

from library_learning.compose.canonical_rlwm import (CANONICAL_BOUNDS,
                                                     CANONICAL_SOURCE)
from library_learning.compose.render_rlwm import SMOKE_DATA
from library_learning.loading import bounds_for_code, exec_model


def test_source_execs_and_is_finite_on_missed_trials():
    func = exec_model(CANONICAL_SOURCE, "cognitive_model")
    params = [(lo + hi) / 2.0 for lo, hi in CANONICAL_BOUNDS]
    nll = float(func(SMOKE_DATA["stimulus"], SMOKE_DATA["actions"],
                     SMOKE_DATA["rewards"], SMOKE_DATA["blocks"],
                     SMOKE_DATA["set_sizes"], params))
    assert np.isfinite(nll) and nll > 0


def test_docstring_bounds_match_constant():
    assert bounds_for_code(CANONICAL_SOURCE) == CANONICAL_BOUNDS
    assert len(CANONICAL_BOUNDS) == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_canonical_rlwm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'library_learning.compose.canonical_rlwm'`

- [ ] **Step 3: Write the implementation**

```python
# library_learning/compose/canonical_rlwm.py
"""Canonical RLWM baseline (Collins & Frank 2012 style): delta-rule RL mixed
with a one-shot, decaying, capacity-limited WM policy + uniform lapse.
Field-standard baseline, analog of hybrid.py (Daw) on two-step; refits are
cross-checked against the data's baseline_bic column (WARN-level in
evaluate_rlwm — the column's producing variant is unknown)."""

CANONICAL_BOUNDS = [(0.0, 1.0), (0.0, 10.0), (0.0, 1.0), (0.0, 1.0),
                    (1.0, 6.0), (0.0, 1.0)]

CANONICAL_SOURCE = '''def cognitive_model(stimulus, actions, rewards, blocks, set_sizes, model_parameters):
    """
    Canonical RLWM baseline (Collins & Frank 2012 style).

    Parameters:
    learning_rate: [0, 1] - RL delta-rule learning rate
    beta: [0, 10] - RL softmax inverse temperature
    wm_weight: [0, 1] - WM reliance at set sizes within capacity
    wm_decay: [0, 1] - per-trial WM decay toward uniform
    capacity: [1, 6] - WM capacity K; WM reliance scales by min(1, K/nS)
    lapse: [0, 1] - uniform-choice lapse probability
    """
    learning_rate, beta, wm_weight, wm_decay, capacity, lapse = model_parameters
    nA = 3
    log_loss = 0.0
    eps = 1e-10
    for b in np.unique(blocks):
        block_mask = blocks == b
        block_states = stimulus[block_mask]
        block_actions = actions[block_mask]
        block_rewards = rewards[block_mask]
        nS = int(set_sizes[block_mask][0])
        q = (1.0 / nA) * np.ones((nS, nA))
        w = (1.0 / nA) * np.ones((nS, nA))
        w_0 = (1.0 / nA) * np.ones((nS, nA))
        for trial in range(len(block_states)):
            s = int(block_states[trial])
            a = int(block_actions[trial])
            r = float(block_rewards[trial])
            if 0 <= s < nS and 0 <= a < nA:
                exp_rl = np.exp(beta * (q[s] - np.max(q[s])))
                probs_rl = exp_rl / np.sum(exp_rl)
                exp_wm = np.exp(50.0 * (w[s] - np.max(w[s])))
                probs_wm = exp_wm / np.sum(exp_wm)
                mix = wm_weight * min(1.0, capacity / float(nS))
                probs = mix * probs_wm + (1.0 - mix) * probs_rl
                probs = (1.0 - lapse) * probs + lapse / nA
                log_loss -= np.log(probs[a] + eps)
                delta = r - q[s, a]
                q[s, a] += learning_rate * delta
                w[s, a] = r
            w += wm_decay * (w_0 - w)
    return log_loss
'''
```

- [ ] **Step 4: Run test to verify it passes**

Run: `gecco-env/bin/python -m pytest tests/test_compose_canonical_rlwm.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/canonical_rlwm.py tests/test_compose_canonical_rlwm.py
git commit -m "feat: canonical Collins & Frank RLWM baseline source + bounds

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: RLWM composition search

**Files:**
- Create: `library_learning/compose/search_rlwm.py`
- Test: `tests/test_compose_search_rlwm.py`

**Interfaces:**
- Consumes: `compatible` from `.inventory`; `candidate_id`, `candidate_params`, `render_candidate`, `smoke_check` from `.render_rlwm`; `fit_model_on_pids` from `.fitting`; `select_winner`, `selection_report` re-exported from `.search` (they operate on plain result dicts — task-agnostic).
- Produces: `PARAM_CAP = 6`, `enumerate_candidates(inventory, param_cap=6) -> [tuple]`, `count_report(inventory, param_cap=6) -> dict`, `score_candidates(inventory, target, validation_pids, candidates, out_dir) -> [dict]`, `greedy_search(inventory, target, validation_pids, out_dir, param_cap=6) -> [dict]`, `freeze_winner(result, inventory, out_dir)`, plus re-exported `select_winner(results) -> dict`, `selection_report(results, out_dir, k=10) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
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
                   ("a", "b"): 495.0}.get(tuple(mods), 470.0)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_search_rlwm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'library_learning.compose.search_rlwm'`

- [ ] **Step 3: Write the implementation**

```python
# library_learning/compose/search_rlwm.py
"""Deterministic composition search over RLWM module combinations. Parallel
sibling of search.py: 3 backbone params (learning_rate, beta, wm_weight) and
the RLWM gecco guardrail cap of 6. select_winner/selection_report are reused
from search.py (they only touch result dicts). No enumerate_from_base here —
the hybrid-base arm is deferred (spec non-goal)."""
import itertools
import json
from collections import Counter
from pathlib import Path

from .inventory import compatible
from .render_rlwm import (candidate_id, candidate_params, render_candidate,
                          smoke_check)
from .fitting import fit_model_on_pids
from .search import select_winner, selection_report  # noqa: F401 (re-exported)

PARAM_CAP = 6
N_BACKBONE_PARAMS = 3


def _n_params(inventory, mods):
    return N_BACKBONE_PARAMS + sum(inventory.module(m).n_params for m in mods)


def enumerate_candidates(inventory, param_cap=PARAM_CAP):
    ids = sorted(inventory.ids())
    out = []
    for k in range(len(ids) + 1):
        for combo in itertools.combinations(ids, k):
            if _n_params(inventory, combo) > param_cap:
                continue
            ok, _ = compatible(inventory, list(combo))
            if ok:
                out.append(combo)
    return out


def count_report(inventory, param_cap=PARAM_CAP):
    cands = enumerate_candidates(inventory, param_cap)
    return {
        "n_modules": len(inventory.ids()),
        "param_cap": param_cap,
        "n_candidates": len(cands),
        "by_n_params": dict(Counter(_n_params(inventory, c) for c in cands)),
        "by_n_modules": dict(Counter(len(c) for c in cands)),
    }


def score_candidates(inventory, target, validation_pids, candidates, out_dir):
    out_dir = Path(out_dir)
    log_path = out_dir / "search_log.jsonl"
    results = []
    with open(log_path, "a") as log:
        for mods in candidates:
            cid = candidate_id(list(mods))
            src = render_candidate(inventory, list(mods))
            smoke_check(src)  # fail fast on render bugs
            bounds = [p.bounds for p in candidate_params(inventory, list(mods))]
            fits = fit_model_on_pids(src, target, validation_pids, bounds, tag=cid)
            rec = {
                "candidate_id": cid,
                "module_ids": sorted(mods),
                "n_params": len(bounds),
                "per_pid": {str(p): fits[p] for p in fits},
                "mean_bic": sum(f["bic"] for f in fits.values()) / len(fits),
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            results.append(rec)
            print("[search] %-60s mean BIC %.2f" % (cid, rec["mean_bic"]))
    return results


def greedy_search(inventory, target, validation_pids, out_dir, param_cap=PARAM_CAP):
    current = ()
    results = score_candidates(inventory, target, validation_pids, [current], out_dir)
    best = results[0]
    improved = True
    while improved:
        improved = False
        additions = []
        for mid in sorted(set(inventory.ids()) - set(current)):
            cand = tuple(sorted(current + (mid,)))
            if _n_params(inventory, cand) > param_cap:
                continue
            ok, _ = compatible(inventory, list(cand))
            if ok:
                additions.append(cand)
        if not additions:
            break
        round_results = score_candidates(inventory, target, validation_pids,
                                         additions, out_dir)
        results.extend(round_results)
        round_best = min(round_results, key=lambda r: r["mean_bic"])
        if round_best["mean_bic"] < best["mean_bic"]:
            best = round_best
            current = tuple(best["module_ids"])
            improved = True
    return results


def freeze_winner(result, inventory, out_dir):
    out_dir = Path(out_dir)
    src = render_candidate(inventory, result["module_ids"])
    (out_dir / "composed_model.txt").write_text(src)
    (out_dir / "winner.json").write_text(json.dumps(result, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `gecco-env/bin/python -m pytest tests/test_compose_search_rlwm.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/search_rlwm.py tests/test_compose_search_rlwm.py
git commit -m "feat: RLWM composition search (3 backbone params, cap 6)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: RLWM extraction (Gemini prompts + gates + orchestration)

**Files:**
- Create: `library_learning/compose/extract_rlwm.py`
- Test: `tests/test_compose_extract_rlwm.py`

**Interfaces:**
- Consumes: `_parse_json_reply`, `_target_names`, `audit_coverage`, `check_bounds_against_sources`, `render_modules_md` from `.extract` (task-agnostic helpers, unmodified); `GeminiClient` from `.gemini`; `Inventory`, `InventoryError`, `compatible` + RLWM `parse_inventory` from `.inventory_rlwm`; `candidate_params`, `render_candidate`, `smoke_check` from `.render_rlwm`; `make_splits` from `.splits_rlwm`; `fit_model_on_pids` from `.fitting`; `mine_fragments`, `render_mining_report` from `..mining`; `load_original_code`, `load_stored_bic`, `extract_unpack_names`, `parse_bounds` from `..loading`.
- Produces: `run_extraction(target, group_dir, out_dir, client=None) -> Inventory`, `validate_inventory_obj(obj) -> (Inventory|None, [str])`, `reconstruction_gate(inv, obj, target, seed_pids, out_dir, tol=15.0) -> [str]`, `CANONICAL_NAMES` (RLWM variable whitelist), `PROMPT_ANNOTATE`, `PROMPT_MERGE`, `PROMPT_REPAIR`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_extract_rlwm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'library_learning.compose.extract_rlwm'`

- [ ] **Step 3: Write the implementation**

```python
# library_learning/compose/extract_rlwm.py
"""Stage 1 for RLWM: Gemini-driven module extraction. Parallel sibling of
extract.py (two-step) — task-agnostic helpers (_parse_json_reply,
_target_names, audit_coverage, bounds cross-check, MODULES.md renderer) are
imported from it unmodified; prompts, canonical-variable whitelist,
validation, and the reconstruction gate are RLWM-specific."""
import ast
import json
import textwrap
from pathlib import Path

from .extract import (_parse_json_reply, _target_names, audit_coverage,
                      check_bounds_against_sources, render_modules_md)
from .fitting import fit_model_on_pids
from .gemini import GeminiClient
from .inventory import Inventory, InventoryError
from .inventory_rlwm import compatible, parse_inventory
from .render_rlwm import candidate_params, render_candidate, smoke_check
from .splits_rlwm import make_splits
from ..loading import load_original_code, load_stored_bic
from ..mining import mine_fragments, render_mining_report

PROMPT_ANNOTATE = '''You are a renowned cognitive scientist analyzing computational models of a reinforcement learning working memory task (RLWM, Collins & Frank). On each trial the participant sees a stimulus (state) and chooses one of 3 actions; each state has one fixed correct action (reward 1, else 0). States come in blocks with set size 3 (low load) or 6 (high load); states reset between blocks. Below is a Python cognitive model that was fit to participant {pid}'s behavior.

```python
{code}
```

List every distinct psychological mechanism in this model. A mechanism is one separable computational assumption (e.g. "WM decay toward uniform", "set-size-scaled WM weight", "WM capacity limit", "uniform lapse", "chunking interference across states", "load-dependent drift or learning rate", "negative-feedback asymmetry", "uncertainty-adaptive temperature", "choice perseveration").

Return STRICT JSON only (no prose, no markdown fences):
{{"mechanisms": [{{"name": "short snake_case id", "title": "human name", "description": "one sentence", "params": [{{"name": "param name as in code", "bounds": [lo, hi]}}], "evidence": "the exact code lines implementing it", "expressible_in_slots": true, "reason": "only when expressible_in_slots is false: why the slot contract cannot express this mechanism faithfully"}}]}}

Rules: do NOT list backbone machinery shared by all models (the block loop with per-block reset of q/w/w_0 to 1/nA, delta-rule RL update with a single learning rate, RL softmax with one inverse temperature — including the common `softmax_beta *= 10` scaling of a [0,1]-bounded parameter, which the backbone's beta in [0,10] absorbs — near-deterministic WM softmax with fixed temperature ~50, probability-level RL/WM mixture with weight wm_weight, NLL accumulation) as mechanisms. Only list deviations from that backbone. If a mechanism cannot be faithfully expressed as code injected into a fixed backbone (slots described below in the pipeline), set "expressible_in_slots": false and explain — do NOT distort a mechanism to make it fit. Slot contract summary: appended statements at function init / per-block init / per-trial pre-choice / after RL logits / after WM logits / after the mixture probabilities / after the value updates / end of trial (runs on missed trials too); replaceable expressions for q and w initialization, RL and WM value and temperature terms, the mixture weight, and the RL and WM update statements.'''

PROMPT_MERGE = '''You are building a library of cognitive mechanism modules for an RLWM task (reinforcement learning working memory, Collins & Frank; blocks of set size 3 or 6, three actions, one fixed correct action per state). You are given (A) mechanism annotations extracted from {n_seeds} participants' models, (B) a mining report of literally-shared code fragments, and (C) the fixed BACKBONE program that modules will be injected into.

(A) Annotations:
{annotations}

(B) Mining report:
{mining_report}

(C) Backbone (modules inject into the named slots of this exact program):
```python
{backbone}
```

Slot contract — a module may provide:
- "slots": statements APPENDED at these points: "init" (function level, before the block loop), "block_init" (each block, after q/w/w_0 are reset), "pre_choice" (each valid trial, before the RL policy), "rl_logits_extra" (after logits_rl assigned), "wm_logits_extra" (after logits_wm assigned), "probs_extra" (after the mixture probs assigned), "update_extra" (after the RL and WM updates, inside the valid-trial guard), "post_trial" (end of EVERY trial iteration, including missed trials).
- "overrides": expressions/statements REPLACING these defaults: "q_init" (default "(1.0 / nA) * np.ones((nS, nA))"), "w_init" (same default), "rl_values" (default "q[s]"), "wm_values" (default "w[s]"), "rl_temp" (default "beta"), "wm_temp" (default "50.0"), "mix_weight" (default "wm_weight"), "rl_update" (default "q[s, a] += learning_rate * delta"), "wm_update" (default "w[s, a] = r").

Available variable names (use ONLY these plus your own module parameters): stimulus, actions, rewards, blocks, set_sizes, nA, nS, b, block_mask, block_states, block_actions, block_rewards, trial, s, a, r, q, w, w_0, rl_values, wm_values, logits_rl, logits_wm, exp_rl, exp_wm, probs_rl, probs_wm, mix, probs, delta, learning_rate, beta, wm_weight, log_loss, eps, np.

Produce the deduplicated module inventory. Return STRICT JSON only:
{{"modules": [{{"id": "snake_case", "name": "...", "description": "...", "params": [{{"name": "...", "bounds": [lo, hi]}}], "slots": {{...}}, "overrides": {{...}}, "provenance": [participant ids], "excludes": ["ids of incompatible modules"]}}], "coverage": [{{"pid": 1, "mechanism": "annotated mechanism name", "module": "module id or null", "decision": "mapped | merged_into | inexpressible | backbone"}}]}}

Rules:
1. One module per distinct mechanism — merge identical mechanisms across participants (union their provenance). Keep singletons (mechanisms found in only one participant).
2. Parameter names must be globally unique across ALL modules and must not be "learning_rate", "beta", or "wm_weight"; suffix if needed (e.g. "alpha_2", "beta_2").
3. Code must be missed-trial-safe: missed trials have actions == -2; the backbone guards likelihood and value updates with `0 <= s < nS and 0 <= a < nA`, but your "post_trial" snippets run on EVERY trial — guard your own action-indexed state updates like `if 0 <= a < nA:` and never index an array with a possibly negative a.
4. "probs_extra" snippets must leave `probs` a valid probability distribution over the 3 actions (renormalize if needed).
5. Per-block state (anything sized by nS) must be created in "block_init", not "init" — nS changes across blocks.
6. Use plain numpy, no imports, no helper functions.
7. Bounds: use the bounds from the source model's docstring where available; otherwise probabilities/rates/weights [0, 1]; inverse temperatures [0, 10]; additive bonuses (stickiness etc.) [0, 5].
8. List modules that implement alternative versions of the same computation (e.g. two different "wm_update" statements) in each other's "excludes".
9. State isolation: any NEW variable your module creates (in "init", "block_init", "post_trial", etc.) must be prefixed with the module id (e.g. module "chunk" uses "chunk_trace"), so no two modules can collide. Never assign to backbone variables except through your declared overrides or the designated *_extra slot variable (logits_rl / logits_wm / probs).
10. The "coverage" list must account for EVERY mechanism in the annotations — one entry per (pid, mechanism), mapping it to the module that absorbed it, or marking it "inexpressible" (annotator flagged it) or "backbone" (it was actually backbone machinery). Nothing may be silently dropped.'''

PROMPT_REPAIR = '''Your previous module inventory JSON had problems. Fix ALL of them and return the corrected STRICT JSON (same schema, no prose):

Problems:
{errors}

Previous JSON:
{previous}'''

RECONSTRUCTION_GATE_TOL = 15.0

CANONICAL_NAMES = {
    "stimulus", "actions", "rewards", "blocks", "set_sizes", "nA", "nS", "b",
    "block_mask", "block_states", "block_actions", "block_rewards", "trial",
    "s", "a", "r", "q", "w", "w_0", "rl_values", "wm_values", "logits_rl",
    "logits_wm", "exp_rl", "exp_wm", "probs_rl", "probs_wm", "mix", "probs",
    "delta", "learning_rate", "beta", "wm_weight", "log_loss", "eps", "np",
}


def annotate_seed(client, pid, code):
    reply = client.generate(PROMPT_ANNOTATE.format(pid=pid, code=code),
                            tag="annotate_p%d" % pid)
    return _parse_json_reply(reply)


def _unprefixed_assignments(module):
    """Names assigned in append-slot snippets that are neither canonical nor
    '{id}_'-prefixed (state-isolation check). Same AST walk as extract.py's,
    against the RLWM CANONICAL_NAMES."""
    bad = set()
    for code in module.slots.values():
        try:
            tree = ast.parse(textwrap.dedent(code))
        except SyntaxError:
            continue  # caught later by smoke_check
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                targets = [node.target]
            elif isinstance(node, ast.comprehension):
                targets = [node.target]
            elif isinstance(node, ast.withitem):
                if node.optional_vars is not None:
                    targets = [node.optional_vars]
            elif isinstance(node, ast.NamedExpr):
                targets = [node.target]
            for t in targets:
                for name in _target_names(t):
                    if (name not in CANONICAL_NAMES
                            and name not in {p.name for p in module.params}
                            and not name.startswith(module.id + "_")):
                        bad.add(name)
    return sorted(bad)


def validate_inventory_obj(obj):
    try:
        inv = parse_inventory(obj)
    except InventoryError as e:
        return None, str(e).split("; ")
    errors = []
    for m in inv.modules:
        try:
            smoke_check(render_candidate(inv, [m.id]))
        except Exception as e:
            errors.append("module '%s' fails alone with backbone: %s" % (m.id, e))
        bad = _unprefixed_assignments(m)
        if bad:
            errors.append(
                "module '%s' assigns unprefixed state variables %s — prefix "
                "them with '%s_'" % (m.id, bad, m.id))
    ids = inv.ids()
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ok, _ = compatible(inv, [a, b])
            if not ok:
                continue
            try:
                smoke_check(render_candidate(inv, [a, b]))
            except Exception as e:
                errors.append("module pair ('%s', '%s') fails together: %s"
                              % (a, b, e))
    return (inv if not errors else None), errors


def reconstruction_gate(inv, obj, target, seed_pids, out_dir,
                        tol=RECONSTRUCTION_GATE_TOL):
    """Fidelity gate: recompose each seed from its provenance modules, fit to
    that seed's own data, compare BIC to the stored individual-gecco BIC."""
    report, failures = [], []
    for pid in seed_pids:
        mods, dropped = [], []
        for m in inv.modules:
            if pid not in m.provenance:
                continue
            ok, why = compatible(inv, mods + [m.id])
            if ok:
                mods.append(m.id)
            else:
                dropped.append({"module": m.id, "why": why})
        src = render_candidate(inv, mods)
        bounds = [p.bounds for p in candidate_params(inv, mods)]
        fits = fit_model_on_pids(src, target, [pid], bounds, tag="recon")
        stored = load_stored_bic(target, pid)
        delta = (fits[pid]["bic"] - stored) if stored is not None else None
        entry = {"pid": pid, "modules": mods, "dropped": dropped,
                 "recon_bic": fits[pid]["bic"], "stored_bic": stored,
                 "delta": delta}
        report.append(entry)
        if delta is not None and delta > tol:
            failures.append(
                "seed %d reconstruction BIC %.2f exceeds stored %.2f by %.2f "
                "(> %.1f): extraction likely distorted a mechanism (modules %s)"
                % (pid, fits[pid]["bic"], stored, delta, tol, mods))
    Path(out_dir, "reconstruction_report.json").write_text(
        json.dumps(report, indent=2))
    return failures


def merge_inventory(client, annotations, mining_report, target=None,
                    seed_pids=None, out_dir=None, max_repair_rounds=3):
    backbone_src = render_candidate(Inventory(modules=[]), [])
    prompt = PROMPT_MERGE.format(
        n_seeds=len(annotations),
        annotations=json.dumps(annotations, indent=1),
        mining_report=mining_report,
        backbone=backbone_src)
    reply = client.generate(prompt, tag="merge")
    obj = _parse_json_reply(reply)

    def all_errors(obj):
        inv, errors = validate_inventory_obj(obj)
        errors = errors + audit_coverage(obj, annotations)
        if not errors and target is not None and seed_pids and out_dir:
            errors = errors + reconstruction_gate(
                inv, obj, target, seed_pids, out_dir)
        return inv, errors

    inv, errors = all_errors(obj)
    rounds = 0
    while errors and rounds < max_repair_rounds:
        rounds += 1
        reply = client.generate(
            PROMPT_REPAIR.format(errors="\n".join("- " + e for e in errors),
                                 previous=json.dumps(obj, indent=1)),
            tag="merge_repair%d" % rounds)
        obj = _parse_json_reply(reply)
        inv, errors = all_errors(obj)
    if errors:
        raise InventoryError(
            "inventory still invalid after %d repair rounds: %s\n"
            "Fix module_inventory.json by hand and record every edit in "
            "llm_log/MANUAL_EDITS.md" % (max_repair_rounds, "; ".join(errors)))
    return obj


def run_extraction(target, group_dir, out_dir, client=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = make_splits(target, group_dir, out_dir=out_dir)
    seed_pids = splits["seed_pids"]
    client = client or GeminiClient(log_dir=out_dir / "llm_log")

    models = {pid: load_original_code(target, pid) for pid in seed_pids}
    shared = mine_fragments(models)
    report = render_mining_report(shared, len(models))
    (out_dir / "mining_report.md").write_text(report)

    annotations = {pid: annotate_seed(client, pid, code)
                   for pid, code in models.items()}
    (out_dir / "annotations.json").write_text(json.dumps(annotations, indent=2))
    obj = merge_inventory(client, annotations, report, target=target,
                          seed_pids=seed_pids, out_dir=out_dir)
    (out_dir / "module_inventory.json").write_text(json.dumps(obj, indent=2))
    inv, errors = validate_inventory_obj(obj)
    assert not errors
    for w in check_bounds_against_sources(inv, target):
        print(w)
    (out_dir / "MODULES.md").write_text(render_modules_md(inv, seed_pids))
    return inv
```

- [ ] **Step 4: Run test to verify it passes**

Run: `gecco-env/bin/python -m pytest tests/test_compose_extract_rlwm.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/extract_rlwm.py tests/test_compose_extract_rlwm.py
git commit -m "feat: RLWM Gemini extraction prompts, validation, and fidelity gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: RLWM per-participant reconstruction

**Files:**
- Create: `library_learning/compose/reconstruct_rlwm.py`
- Test: `tests/test_compose_reconstruct_rlwm.py`

**Interfaces:**
- Consumes: `enumerate_candidates`, `greedy_search`, `score_candidates`, `select_winner` from `.search_rlwm`; `fit_model_on_pids` from `.fitting`; `bounds_for_code`, `function_name_and_args`, `load_original_code`, `strip_fences` from `..loading`.
- Produces: `reconstruct_participants(inventory, target, group_dir, pids, out_dir, mode="greedy") -> [dict]` writing `reconstruction_results.json` (keys per pid: `pid`, `best_candidate_id`, `best_module_ids`, `best_n_params`, `library_bic`, `individual_bic`, `group_bic`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose_reconstruct_rlwm.py
import json

from library_learning.compose import reconstruct_rlwm as R


def test_reconstruct_uses_stubs(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "greedy_search", lambda inv, t, pids, d: [
        {"candidate_id": "m1", "module_ids": ["m1"], "n_params": 4,
         "per_pid": {str(pids[0]): {"bic": 400.0}}, "mean_bic": 400.0}])
    monkeypatch.setattr(R, "load_original_code", lambda t, pid: "def cognitive_model(a, b, c, d, e, model_parameters):\n    x, = model_parameters\n    return 0.0")
    monkeypatch.setattr(R, "fit_model_on_pids",
                        lambda src, t, pids, bounds, tag, func_name=None:
                        {pids[0]: {"bic": 390.0 if "individual" in tag else 410.0,
                                   "nll": 0.0, "params": [0.5], "seed": 1}})
    monkeypatch.setattr(R, "strip_fences", lambda text: text)
    monkeypatch.setattr(
        R, "function_name_and_args", lambda code: ("cognitive_model", []))
    (tmp_path / "grp" / "models").mkdir(parents=True)
    (tmp_path / "grp" / "models" / "best_model_0.txt").write_text("stub")

    results = R.reconstruct_participants(None, None, tmp_path / "grp", [7], tmp_path)
    assert results[0]["pid"] == 7
    assert results[0]["library_bic"] == 400.0
    assert results[0]["individual_bic"] == 390.0
    assert results[0]["group_bic"] == 410.0
    saved = json.loads((tmp_path / "reconstruction_results.json").read_text())
    assert saved == results
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_reconstruct_rlwm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'library_learning.compose.reconstruct_rlwm'`

- [ ] **Step 3: Write the implementation**

```python
# library_learning/compose/reconstruct_rlwm.py
"""Stage 2b for RLWM: per-participant best library composition on unseen
participants (library-coverage metric; kept separate from the single-program
claim). Parallel sibling of reconstruct.py, wired to the RLWM search."""
import json
from pathlib import Path

from .fitting import fit_model_on_pids
from .search_rlwm import (enumerate_candidates, greedy_search,
                          score_candidates, select_winner)
from ..loading import (bounds_for_code, function_name_and_args,
                       load_original_code, strip_fences)


def reconstruct_participants(inventory, target, group_dir, pids, out_dir,
                             mode="greedy"):
    out_dir = Path(out_dir)
    group_src = strip_fences(
        (Path(group_dir) / "models" / "best_model_0.txt").read_text())
    group_fname, _ = function_name_and_args(group_src)

    cands = enumerate_candidates(inventory) if mode == "exhaustive" else None

    results = []
    for pid in pids:
        pid_dir = out_dir / "reconstruction" / ("p%d" % pid)
        pid_dir.mkdir(parents=True, exist_ok=True)
        if mode == "exhaustive":
            recs = score_candidates(inventory, target, [pid], cands, pid_dir)
        else:
            recs = greedy_search(inventory, target, [pid], pid_dir)
        best = select_winner(recs)

        own_code = load_original_code(target, pid)
        own_fname, _ = function_name_and_args(own_code)
        own = fit_model_on_pids(own_code, target, [pid], bounds_for_code(own_code),
                                tag="recon:individual", func_name=own_fname)
        grp = fit_model_on_pids(group_src, target, [pid], bounds_for_code(group_src),
                                tag="recon:group", func_name=group_fname)
        results.append({
            "pid": pid,
            "best_candidate_id": best["candidate_id"],
            "best_module_ids": best["module_ids"],
            "best_n_params": best["n_params"],
            "library_bic": best["mean_bic"],  # mean over one pid == that pid's BIC
            "individual_bic": own[pid]["bic"],
            "group_bic": grp[pid]["bic"],
        })
        print("[reconstruct] p%d library %.2f vs individual %.2f vs group %.2f"
              % (pid, best["mean_bic"], own[pid]["bic"], grp[pid]["bic"]))
    (out_dir / "reconstruction_results.json").write_text(
        json.dumps(results, indent=2))
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `gecco-env/bin/python -m pytest tests/test_compose_reconstruct_rlwm.py -q`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/reconstruct_rlwm.py tests/test_compose_reconstruct_rlwm.py
git commit -m "feat: RLWM per-participant library reconstruction

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: RLWM evaluation + figure

**Files:**
- Create: `library_learning/compose/evaluate_rlwm.py`
- Create: `library_learning/compose/figure_rlwm.py`
- Test: `tests/test_compose_evaluate_rlwm.py`

**Interfaces:**
- Consumes: `fit_model_on_pids` from `.fitting`; `CANONICAL_BOUNDS`, `CANONICAL_SOURCE` from `.canonical_rlwm`; `_wilcoxon_p`, `TIE_TOL`, `CROSS_CHECK_TOL` from `.evaluate` (constants/helper, unmodified); loaders from `..loading`.
- Produces: `evaluate_models(target, group_dir, out_dir, pids, set_name) -> dict` (keys `composed`, `group`, `canonical`, `individual`); `cross_checks(results, target, group_dir, pids, heldout_pids=None) -> [str]`; `summarize(results_val, results_test, warnings, out_dir, target=None) -> dict` (writes `test_results.csv`, `test_results.json`, `RESULTS.md`; stats keys `mean_bic`, `composed_vs_group`, `composed_vs_canonical`, `warnings`); `figure_rlwm.plot_comparison(test_results_csv, out_dir, rng_seed=7)` (writes `comparison.png` + `comparison.pdf`; models `canonical`/`group`/`composed`/`individual`, teal canonical, blue composed).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose_evaluate_rlwm.py
import json

import pandas as pd

from library_learning.compose.evaluate_rlwm import summarize
from library_learning.compose.figure_rlwm import plot_comparison


def _fits(bic):
    return {p: {"bic": bic + p, "nll": 100.0, "params": [0.5], "seed": 1}
            for p in [4, 6]}


def test_summarize_and_results_md(tmp_path):
    results_test = {"composed": _fits(400.0), "group": _fits(410.0),
                    "canonical": _fits(420.0), "individual": _fits(390.0)}
    (tmp_path / "winner.json").write_text(json.dumps(
        {"candidate_id": "m1+m2", "n_params": 5}))
    stats = summarize(None, results_test, ["a warning"], tmp_path)
    assert stats["composed_vs_group"]["wins"] == 2
    assert stats["composed_vs_canonical"]["mean_delta"] == -20.0
    md = (tmp_path / "RESULTS.md").read_text()
    assert "composed vs canonical" in md and "a warning" in md
    df = pd.read_csv(tmp_path / "test_results.csv")
    assert set(df["model"]) == {"composed", "group", "canonical", "individual"}
    assert "age" in df.columns


def test_figure(tmp_path):
    rows = []
    for pid in [4, 6]:
        for model, b in [("composed", 400.0), ("group", 410.0),
                         ("canonical", 420.0), ("individual", 390.0)]:
            rows.append({"set": "test", "participant": pid, "age": 30.0,
                         "model": model, "n_params": 5, "nll": 100.0,
                         "bic": b + pid, "seed": 1})
    csv = tmp_path / "test_results.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    plot_comparison(csv, tmp_path)
    assert (tmp_path / "comparison.png").exists()
    assert (tmp_path / "comparison.pdf").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_evaluate_rlwm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'library_learning.compose.evaluate_rlwm'`

- [ ] **Step 3: Write the implementation**

```python
# library_learning/compose/evaluate_rlwm.py
"""Stage 3 for RLWM: fit frozen winner + baselines on held-out participants
under one seeded protocol; stats + RESULTS.md. Parallel sibling of
evaluate.py — the Daw hybrid is replaced by the canonical Collins & Frank
RLWM baseline, and OCI by age. The stored group-test-BIC cross-check is kept
but auto-skips (results/rlwm has no best_bic_on_test_run0.json)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical_rlwm import CANONICAL_BOUNDS, CANONICAL_SOURCE
from .evaluate import CROSS_CHECK_TOL, TIE_TOL, _wilcoxon_p
from .fitting import fit_model_on_pids
from ..loading import (bounds_for_code, function_name_and_args,
                       load_dataframe, load_original_code, strip_fences)


def evaluate_models(target, group_dir, out_dir, pids, set_name):
    out_dir = Path(out_dir)
    composed_path = out_dir / "composed_model.txt"
    if not composed_path.exists():
        raise FileNotFoundError(
            "composed_model.txt not found in %s — run compose-search first "
            "(freeze discipline: evaluation only runs on a frozen winner)" % out_dir)
    composed_src = composed_path.read_text()

    group_src = strip_fences(
        (Path(group_dir) / "models" / "best_model_0.txt").read_text())
    group_func_name, _ = function_name_and_args(group_src)

    results = {}
    results["composed"] = fit_model_on_pids(
        composed_src, target, pids, bounds_for_code(composed_src),
        tag="eval:%s:composed" % set_name)
    results["group"] = fit_model_on_pids(
        group_src, target, pids, bounds_for_code(group_src),
        tag="eval:%s:group" % set_name, func_name=group_func_name)
    results["canonical"] = fit_model_on_pids(
        CANONICAL_SOURCE, target, pids, CANONICAL_BOUNDS,
        tag="eval:%s:canonical" % set_name)
    individual = {}
    for pid in pids:
        code = load_original_code(target, pid)
        fname, _ = function_name_and_args(code)
        individual.update(fit_model_on_pids(
            code, target, [pid], bounds_for_code(code),
            tag="eval:%s:individual" % set_name, func_name=fname))
    results["individual"] = individual
    return results


def cross_checks(results, target, group_dir, pids, heldout_pids=None):
    warnings = []
    stored_path = Path(group_dir) / "bics" / "best_bic_on_test_run0.json"
    if stored_path.exists():
        stored = json.loads(stored_path.read_text())["individual_BIC"]
        if heldout_pids is None:
            heldout_pids = sorted(range(14, 14 + len(stored)))
        for pid in pids:
            if pid not in heldout_pids:
                warnings.append(
                    "cross-check skipped for p%d: not in stored held-out mapping"
                    % pid)
                continue
            idx = heldout_pids.index(pid)
            if 0 <= idx < len(stored):
                diff = results["group"][pid]["bic"] - stored[idx]
                if abs(diff) > CROSS_CHECK_TOL:
                    warnings.append(
                        "group refit BIC differs from stored for p%d: %.2f vs %.2f"
                        % (pid, results["group"][pid]["bic"], stored[idx]))
    df = load_dataframe(target)
    baseline = df.groupby(target.id_column)["baseline_bic"].first()
    for pid in pids:
        diff = results["canonical"][pid]["bic"] - float(baseline[pid])
        if abs(diff) > CROSS_CHECK_TOL:
            warnings.append(
                "canonical refit BIC differs from baseline_bic for p%d: "
                "%.2f vs %.2f (the column's producing variant is unknown)"
                % (pid, results["canonical"][pid]["bic"], float(baseline[pid])))
    return warnings


def _rows(results, set_name, age):
    rows = []
    for model, fits in results.items():
        for pid, f in fits.items():
            rows.append({"set": set_name, "participant": pid,
                         "age": float(age[pid]), "model": model,
                         "n_params": len(f["params"]), "nll": f["nll"],
                         "bic": f["bic"], "seed": f["seed"]})
    return rows


def summarize(results_val, results_test, warnings, out_dir, target=None):
    out_dir = Path(out_dir)
    if target is not None:
        df = load_dataframe(target)
        age = df.groupby(target.id_column)["age"].first()
    else:
        all_pids = {p for r in [results_val, results_test] if r
                    for fits in r.values() for p in fits}
        age = {p: float("nan") for p in all_pids}

    rows = []
    if results_val:
        rows += _rows(results_val, "validation", age)
    rows += _rows(results_test, "test", age)
    pd.DataFrame(rows).to_csv(out_dir / "test_results.csv", index=False)

    test_pids = sorted(next(iter(results_test.values())).keys())
    means = {m: float(np.mean([results_test[m][p]["bic"] for p in test_pids]))
             for m in results_test}
    comp = np.array([results_test["composed"][p]["bic"] for p in test_pids])
    grp = np.array([results_test["group"][p]["bic"] for p in test_pids])
    canon = np.array([results_test["canonical"][p]["bic"] for p in test_pids])
    delta = comp - grp
    wins = int((delta < -TIE_TOL).sum())
    ties = int((np.abs(delta) <= TIE_TOL).sum())
    losses = int((delta > TIE_TOL).sum())
    stats = {
        "mean_bic": means,
        "composed_vs_group": {"wilcoxon_p": _wilcoxon_p(comp, grp),
                              "wins": wins, "ties": ties, "losses": losses,
                              "mean_delta": float(delta.mean())},
        "composed_vs_canonical": {"wilcoxon_p": _wilcoxon_p(comp, canon),
                                  "mean_delta": float((comp - canon).mean())},
        "warnings": warnings,
    }
    (out_dir / "test_results.json").write_text(json.dumps(
        {"stats": stats, "rows": rows}, indent=2))

    winner = json.loads((out_dir / "winner.json").read_text())
    lines = ["# RLWM library composition results", "",
             "Winner modules: `%s` (%d params)" % (winner["candidate_id"],
                                                   winner["n_params"]), "",
             "## Mean BIC on final test (%d participants)" % len(test_pids), "",
             "| model | mean BIC |", "|---|---|"]
    for m in sorted(means, key=means.get):
        lines.append("| %s | %.2f |" % (m, means[m]))
    lines += ["",
              "composed vs group: mean dBIC %.2f, W/T/L %d/%d/%d, wilcoxon p=%.4f"
              % (stats["composed_vs_group"]["mean_delta"], wins, ties, losses,
                 stats["composed_vs_group"]["wilcoxon_p"]),
              "composed vs canonical: mean dBIC %.2f, wilcoxon p=%.4f"
              % (stats["composed_vs_canonical"]["mean_delta"],
                 stats["composed_vs_canonical"]["wilcoxon_p"]), ""]
    if warnings:
        lines += ["## Cross-check warnings", ""] + ["- " + w for w in warnings]
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return stats
```

```python
# library_learning/compose/figure_rlwm.py
"""RLWM comparison figure, paper style (teal reference / blue winner / gray
others). Parallel sibling of figure.py with the canonical RLWM baseline in
the reference slot."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"composed": "#2b6cb8", "canonical": "#1b9e91",
          "group": "#9a9a9a", "individual": "#c4c4c4"}
ORDER = ["canonical", "group", "composed", "individual"]
LABELS = {"canonical": "RLWM\n(Collins & Frank)", "group": "Group GeCCo",
          "composed": "Composed (library)", "individual": "Individual GeCCo\n(ceiling)"}


def plot_comparison(test_results_csv, out_dir, rng_seed=7):
    out_dir = Path(out_dir)
    df = pd.read_csv(test_results_csv)
    df = df[df["set"] == "test"]
    rng = np.random.default_rng(rng_seed)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for i, model in enumerate(ORDER):
        vals = df[df.model == model]["bic"].to_numpy()
        if len(vals) == 0:
            raise ValueError("no test rows for model '%s'" % model)
        ax.bar(i, vals.mean(), width=0.62, color=COLORS[model],
               edgecolor="none", zorder=2)
        x = i + rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(x, vals, s=9, color="0.25", alpha=0.55, lw=0, zorder=3)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABELS[m] for m in ORDER], fontsize=8)
    ax.set_ylabel("BIC (test participants)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=300)
    fig.savefig(out_dir / "comparison.pdf")
    plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `gecco-env/bin/python -m pytest tests/test_compose_evaluate_rlwm.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/evaluate_rlwm.py library_learning/compose/figure_rlwm.py tests/test_compose_evaluate_rlwm.py
git commit -m "feat: RLWM held-out evaluation vs group/canonical/individual + figure

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: CLI dispatch by task name

**Files:**
- Modify: `library_learning/__main__.py` (the ONLY existing file this plan touches)
- Test: `tests/test_compose_cli_rlwm.py`

**Interfaces:**
- Consumes: everything produced in Tasks 1–8; `_task_name_for` from `.config` (existing private helper, imported, not modified).
- Produces: all `compose-*` subcommands work with `--results-dir results/rlwm_individual --group-dir results/rlwm`, dispatching to the rlwm module set; two-step defaults/behavior unchanged; `compose-hybrid-search` errors out for rlwm targets.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose_cli_rlwm.py
import subprocess
import sys
from pathlib import Path

from library_learning.__main__ import _task_mods


def test_task_mods_dispatch():
    two_step = _task_mods("results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual")
    rlwm = _task_mods("results/rlwm_individual")
    assert two_step["search"].__name__ == "library_learning.compose.search"
    assert rlwm["search"].__name__ == "library_learning.compose.search_rlwm"
    assert rlwm["extract"].__name__ == "library_learning.compose.extract_rlwm"
    assert rlwm["evaluate"].__name__ == "library_learning.compose.evaluate_rlwm"


def test_hybrid_search_rejected_for_rlwm():
    r = subprocess.run(
        [sys.executable, "-m", "library_learning", "compose-hybrid-search",
         "--results-dir", "results/rlwm_individual",
         "--group-dir", "results/rlwm"],
        capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    assert r.returncode != 0
    assert "deferred" in (r.stderr + r.stdout)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_cli_rlwm.py -q`
Expected: FAIL with `ImportError: cannot import name '_task_mods'`

- [ ] **Step 3: Modify `library_learning/__main__.py`**

Add imports after the existing ones (keep every existing import):

```python
from .compose import evaluate_rlwm, extract_rlwm, reconstruct_rlwm, search_rlwm
from .compose import evaluate as evaluate_two_step
from .compose import extract as extract_two_step
from .compose import figure as figure_two_step
from .compose import figure_rlwm
from .compose import inventory as inventory_two_step
from .compose import inventory_rlwm
from .compose import reconstruct as reconstruct_two_step
from .config import _task_name_for
from pathlib import Path
```

Add the dispatch table + helper right after `OUT_DIRNAME`:

```python
_TWO_STEP_MODS = {"extract": extract_two_step, "search": search_mod,
                  "reconstruct": reconstruct_two_step,
                  "evaluate": evaluate_two_step, "figure": figure_two_step,
                  "inventory": inventory_two_step}
_RLWM_MODS = {"extract": extract_rlwm, "search": search_rlwm,
              "reconstruct": reconstruct_rlwm, "evaluate": evaluate_rlwm,
              "figure": figure_rlwm, "inventory": inventory_rlwm}


def _task_mods(results_dir):
    if _task_name_for(Path(results_dir).resolve()) == "rlwm":
        return _RLWM_MODS
    return _TWO_STEP_MODS
```

Then rewire each `cmd_*` body to go through the table (same logic, module set swapped; the two-step path resolves to the identical functions as before):

```python
def cmd_modules(args):
    mods = _task_mods(args.results_dir)
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    if args.skip_llm:
        obj = json.loads((out / "module_inventory.json").read_text())
        inv, errors = mods["extract"].validate_inventory_obj(obj)
        if errors:
            print("\n".join(errors))
            return 1
        print("inventory valid: %d modules" % len(inv.modules))
        return 0
    inv = mods["extract"].run_extraction(target, args.group_dir, out)
    print("extracted %d modules -> %s" % (len(inv.modules), out))
    return 0


def cmd_count(args):
    mods = _task_mods(args.results_dir)
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    inv = mods["inventory"].load_inventory(out_dir_for(target) / "module_inventory.json")
    print(json.dumps(mods["search"].count_report(inv), indent=2))
    return 0


def cmd_search(args):
    mods = _task_mods(args.results_dir)
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    inv = mods["inventory"].load_inventory(out / "module_inventory.json")
    val_pids = load_splits(target)["composition_validation_pids"]
    if args.mode == "exhaustive":
        cands = mods["search"].enumerate_candidates(inv)
        results = mods["search"].score_candidates(inv, target, val_pids, cands, out)
    else:
        results = mods["search"].greedy_search(inv, target, val_pids, out)
    winner = mods["search"].select_winner(results)
    mods["search"].freeze_winner(winner, inv, out)
    mods["search"].selection_report(results, out)
    print("WINNER %s mean validation BIC %.2f -> composed_model.txt frozen"
          % (winner["candidate_id"], winner["mean_bic"]))
    return 0


def cmd_hybrid_search(args):
    """Exhaustive search over modules missing from the Daw hybrid, using the
    hybrid-equivalent module set as a fixed base. Outputs under hybrid_base/."""
    if _task_mods(args.results_dir) is _RLWM_MODS:
        print("compose-hybrid-search is two-step only; the RLWM hybrid-base "
              "arm is deferred (see the 2026-07-18 spec non-goals)")
        return 2
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    # ... rest of the existing body unchanged ...


def cmd_reconstruct(args):
    mods = _task_mods(args.results_dir)
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    inv = mods["inventory"].load_inventory(out / "module_inventory.json")
    pids = load_splits(target)["reconstruction_pids"]
    mods["reconstruct"].reconstruct_participants(inv, target, args.group_dir,
                                                 pids, out, mode=args.mode)
    print("reconstruction -> %s" % (out / "reconstruction_results.json"))
    return 0


def cmd_eval(args):
    mods = _task_mods(args.results_dir)
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    splits = load_splits(target)
    sets = args.sets.split(",")
    results_val = None
    if "validation" in sets:
        results_val = mods["evaluate"].evaluate_models(
            target, args.group_dir, out,
            splits["composition_validation_pids"], "validation")
    if "test" not in sets:
        print("validation-only run (--sets=%s): skipping test evaluation, "
              "cross-checks, summary, and figure" % args.sets)
        return 0
    results_test = mods["evaluate"].evaluate_models(
        target, args.group_dir, out, splits["test_pids"], "test")
    heldout_pids = sorted(set(splits["reconstruction_pids"])
                          | set(splits["test_pids"]))
    warnings = mods["evaluate"].cross_checks(results_test, target, args.group_dir,
                                             splits["test_pids"],
                                             heldout_pids=heldout_pids)
    mods["evaluate"].summarize(results_val, results_test, warnings, out,
                               target=target)
    mods["figure"].plot_comparison(out / "test_results.csv", out)
    print("results -> %s (warnings: %d)" % (out / "RESULTS.md", len(warnings)))
    return 0
```

Notes for the implementer:
- Keep the existing top-of-file imports that these bodies still use (`resolve_target`, `search_mod`, `load_splits`, `HYBRID_MODULES`, `json`, `sys`); drop now-unused direct function imports (`run_extraction`, `validate_inventory_obj`, `evaluate_models`, `cross_checks`, `summarize`, `plot_comparison`, `reconstruct_participants`, `load_inventory`) — they are replaced by module-level access. That removal is required by the "your changes' orphans" rule, not a refactor.
- Two-step figure: `figure.plot_comparison` and `evaluate` functions are the same objects as before — behavior identical.

- [ ] **Step 4: Run the new test AND the full suite**

Run: `gecco-env/bin/python -m pytest tests/ -q`
Expected: all tests pass (existing two-step tests unchanged and green — this is the "two-step untouched" verification from the spec).

- [ ] **Step 5: Commit**

```bash
git add library_learning/__main__.py tests/test_compose_cli_rlwm.py
git commit -m "feat: dispatch compose subcommands by task name (rlwm vs two-step)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Stage-1 run — extraction + gates + candidate-count CHECKPOINT

This task runs the real pipeline (Gemini calls). It requires `.env` with `GEMINI_API_KEY_LAKELAB` and network access.

**Files:**
- Output (not committed until verified): `results/rlwm_individual/library_composition/{splits.json, mining_report.md, annotations.json, module_inventory.json, MODULES.md, reconstruction_report.json, llm_log/}`

- [ ] **Step 1: Preflight — backbone-vs-seeds sanity**

```bash
source gecco-env/bin/activate
set -o pipefail
python - <<'EOF'
from library_learning.config import resolve_target
from library_learning.compose.render_rlwm import render_candidate, smoke_check
from library_learning.compose.inventory import Inventory
from library_learning.compose.fitting import fit_model_on_pids
from library_learning.loading import load_original_code

target = resolve_target("results/rlwm_individual")
src = render_candidate(Inventory(modules=[]), [])
print("backbone smoke NLL:", smoke_check(src))
fits = fit_model_on_pids(src, target, [1], [(0,1),(0,10),(0,1)], tag="preflight")
print("backbone fit p1 BIC:", fits[1]["bic"])
for pid in [1, 2, 3, 10, 11, 12, 13, 14]:
    code = load_original_code(target, pid)
    assert "def cognitive_model" in code, pid
print("all 8 seed programs load")
EOF
```

Expected: finite smoke NLL, a plausible p1 BIC (roughly 200–700 for 324 trials), "all 8 seed programs load". If the backbone fit errors, STOP and fix render_rlwm before any Gemini call.

- [ ] **Step 2: Read the 8 seed programs against the backbone**

Read each `results/rlwm_individual/models/best_model_0_participant{1,2,3,10,11,12,13,14}.txt`. Confirm the backbone machinery list in `PROMPT_ANNOTATE` (block loop, 1/nA inits, delta-rule RL, softmax ×10 beta scaling, WM temp ~50, wm_weight mixture) matches what they actually share; if a seed deviates structurally (e.g. no block loop), update the prompt's backbone list and the spec note BEFORE calling Gemini, and commit that prompt change.

- [ ] **Step 3: Run extraction (logged Gemini calls)**

```bash
set -o pipefail
python -m library_learning compose-modules \
  --results-dir results/rlwm_individual --group-dir results/rlwm \
  2>&1 | tee results/rlwm_individual/library_composition/extraction_run.log
```

Expected: `extracted N modules -> results/rlwm_individual/library_composition` (N likely 10–25). On `InventoryError` after 3 repair rounds: inspect `llm_log/`, fix `module_inventory.json` by hand, record every edit in `llm_log/MANUAL_EDITS.md`, then re-validate with `python -m library_learning compose-modules --results-dir results/rlwm_individual --group-dir results/rlwm --skip-llm`.

- [ ] **Step 4: Verify the gates**

```bash
python - <<'EOF'
import json
rep = json.load(open("results/rlwm_individual/library_composition/reconstruction_report.json"))
for e in rep:
    print(e["pid"], "delta", e["delta"], "modules", e["modules"], "dropped", e["dropped"])
assert all(e["delta"] is None or e["delta"] <= 15.0 for e in rep), "fidelity gate FAILED"
splits = json.load(open("results/rlwm_individual/library_composition/splits.json"))
assert splits["seed_pids"] == [1, 2, 3, 10, 11, 12, 13, 14]
ann = json.load(open("results/rlwm_individual/library_composition/annotations.json"))
assert all(k.isdigit() or isinstance(k, int) for k in ann)
print("gates OK: %d seeds, all deltas <= 15" % len(rep))
EOF
```

Expected: `gates OK: 8 seeds, all deltas <= 15`. Also skim `MODULES.md` — module names should read as recognizable RLWM mechanisms (decay, capacity, lapse, interference, perseveration…).

- [ ] **Step 5: Candidate count — decide mode autonomously and log it**

```bash
python -m library_learning compose-count \
  --results-dir results/rlwm_individual --group-dir results/rlwm
```

The user is offline (autonomous mode). Measure one mid-size candidate's fit time on the 10 validation pids, estimate exhaustive wall-clock (`n_candidates × per-candidate time`), and decide: **exhaustive if the estimate is ≤ 4 hours, else greedy** (two-step precedent: greedy was chosen when exhaustive ≈ 21 h). Record `n_modules`, `n_candidates`, the per-candidate timing, the estimate, and the chosen mode in `results/rlwm_individual/library_composition/DECISIONS.md`.

- [ ] **Step 6: Commit the extraction artifacts**

```bash
git add results/rlwm_individual/library_composition
git commit -m "run: RLWM module extraction — inventory, gates 8/8, llm_log

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Stage 2/2b/3 runs — search, reconstruction, evaluation

Prerequisite: the exhaustive-vs-greedy decision logged in `DECISIONS.md` at Task 10 Step 5 (`<MODE>` below).

**Files:**
- Output: `results/rlwm_individual/library_composition/{search_log.jsonl, selection_report.json, composed_model.txt, winner.json, reconstruction/, reconstruction_results.json, test_results.csv, test_results.json, RESULTS.md, comparison.png, comparison.pdf}`

- [ ] **Step 1: Composition search on validation pids 10–19**

```bash
source gecco-env/bin/activate
set -o pipefail
python -m library_learning compose-search --mode <MODE> \
  --results-dir results/rlwm_individual --group-dir results/rlwm \
  2>&1 | tee results/rlwm_individual/library_composition/search_run.log
```

Expected: `WINNER <candidate_id> mean validation BIC <x> -> composed_model.txt frozen`. Sanity: the winner's mean validation BIC must be ≤ the backbone-only candidate's (greedy guarantees this by construction).

- [ ] **Step 2: Library reconstruction on the 7 recon pids**

```bash
python -m library_learning compose-reconstruct --mode greedy \
  --results-dir results/rlwm_individual --group-dir results/rlwm \
  2>&1 | tee results/rlwm_individual/library_composition/reconstruct_run.log
```

Expected: 7 `[reconstruct] pN library X vs individual Y vs group Z` lines; `reconstruction_results.json` written.

- [ ] **Step 3: Final-test evaluation**

```bash
python -m library_learning compose-eval --sets validation,test \
  --results-dir results/rlwm_individual --group-dir results/rlwm \
  2>&1 | tee results/rlwm_individual/library_composition/eval_run.log
```

Expected: `results -> .../RESULTS.md (warnings: K)`. `baseline_bic` warnings are expected (unknown producing variant) — read them, note the systematic direction in RESULTS.md, and investigate before making any claim that leans on the canonical baseline.

- [ ] **Step 4: Review results end-to-end**

Read `RESULTS.md`, `selection_report.json` (LOO rank stability of the winner), `reconstruction_results.json` (library vs individual ceiling vs group per pid), and `comparison.png`. Confirm RESULTS.md discloses: 8 seeds (15–19 unfitted), the held-out-pool deviation (reclaimed young 0,4–9), the missed-trial protocol difference, and the small-sample caveat. If any disclosure is missing, append it to RESULTS.md by hand.

- [ ] **Step 5: Commit run artifacts and report**

```bash
git add results/rlwm_individual/library_composition
git commit -m "run: RLWM composition search, reconstruction, and held-out evaluation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Report to Akshay: winner modules + param count, mean test BIC per model, composed-vs-group W/T/L + Wilcoxon p, composed-vs-canonical delta, library-span summary (how many of the 7 recon pids the library matches/beats their individual ceiling), and any cross-check warnings.

---

### Task 12: Self-contained HTML report

Mirror the two-step run's `report.html` (224-line hand-authored page in
`results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual/library_composition/` — read it first and reuse its structure/CSS).

**Files:**
- Create: `results/rlwm_individual/library_composition/report.html`
- Create: `results/rlwm_individual/library_composition/report_fig1_test_bic.png` (+ fig2 reconstruction, fig3 per-pid delta) via a throwaway matplotlib script in the scratchpad (paper style: teal `#1b9e91` = canonical reference, blue `#2b6cb8` = composed winner, grays = others; PNG 300 dpi)

- [ ] **Step 1: Read the two-step report.html for structure**
- [ ] **Step 2: Generate the three report figures from `test_results.csv` and `reconstruction_results.json`** (fig1: mean test BIC bars + per-pid dots; fig2: per-recon-pid library vs individual vs group BIC; fig3: per-test-pid composed-minus-group ΔBIC, sorted, age-annotated)
- [ ] **Step 3: Author `report.html`** — sections: headline result, methods recap (splits incl. disclosures from the spec: 8 seeds, 15–19 unfitted, reclaimed young pids, eval/test overlap, missed-trial protocol, Gemini-only), module library table (from MODULES.md), search + selection stability, library-span reconstruction, final-test stats table, cross-check warnings, decisions log (inline DECISIONS.md), file manifest. All assets referenced relatively; no external URLs.
- [ ] **Step 4: Open-check** — `python -c "import webbrowser"` not needed; verify the HTML references only files that exist (`grep -o 'src="[^"]*"' report.html` and stat each).
- [ ] **Step 5: Commit**

```bash
git add results/rlwm_individual/library_composition/report.html results/rlwm_individual/library_composition/report_fig*.png
git commit -m "results: self-contained HTML report for RLWM library-composition run

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
