# Library Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cognitive-module library from the 12 individual gecco programs seen by group gecco, compose a single program via deterministic module search selected on a validation split, and test generalization on held-out participants against group gecco / hybrid / individual baselines.

**Architecture:** New `library_learning/compose/` subpackage + `library_learning/__main__.py` CLI. Stage 1 extracts modules via logged Gemini API calls grounded in the existing AST fragment miner. Stage 2 renders backbone+module candidates as standalone `cognitive_model` functions and searches combinations (exhaustive or greedy — user decides after seeing the count). Stage 3 fits the frozen winner and baselines on final-test participants under one seeded protocol.

**Tech Stack:** Python 3.9 (`gecco-env/` venv), numpy/pandas/scipy/pyyaml, matplotlib (figure), pytest (tests), Gemini REST API via stdlib `urllib` (no new SDK dependency).

**Spec:** `docs/superpowers/specs/2026-07-17-library-composition-design.md`

## Global Constraints

- Python 3.9 — no `X | Y` annotations, no `match`. Interpreter: `/Users/akshay/projects/gecco/gecco-env/bin/python`; run everything from repo root `/Users/akshay/projects/gecco`.
- Gemini: model `gemini-3.5-flash`, `temperature: 0`, key from `.env` var **`GEMINI_API_KEY_LAKELAB`** (verified working 2026-07-17; plain `GEMINI_API_KEY` is invalid — never use it). Every call logged verbatim to `llm_log/call_{NNN}_{tag}.json`.
- Individual results dir (seeds + output home): `results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual`
- Group results dir: `results/two_step_psychiatry_group_function_ocibalanced_maxsetting` (its config: `config/two_step_psychiatry_group_ocd_maxsetting.yaml`, matched by `task.name`).
- Output dir: `<individual results dir>/library_composition/` (constant `OUT_DIRNAME = "library_composition"`).
- Splits: library seeds = group prompt+eval participants (expect `[1, 2, 4..13]`); validation = 10 of test pids (OCI-stratified, deterministic); final test = remaining 21. Participants 0 and 3 unused.
- Parameter cap: **8** total (backbone 2 + modules). Missed trials are coded `-1` in `choice_1/state/choice_2/reward` — all rendered code must be `-1`-safe.
- Every fit seeded: L-BFGS-B, `n_starts=10`, `seed = int(hashlib.md5(f"{tag}:{pid}".encode()).hexdigest()[:8], 16)`.
- BIC = `log(n_trials)*k + 2*nll`, `n_trials=200` per participant (matches `gecco/offline_evaluation/evaluation_functions.py`).
- Rendered model code: numpy-only, no imports, gecco-compatible (docstring bounds parseable by `library_learning/loading.py::BOUNDS_RE`, unpack line `a, b = model_parameters`).
- Tests: `gecco-env/bin/python -m pytest tests/ -v`. No live-API calls in tests (inject fake transport).
- **USER CHECKPOINT** after Task 11's count step: report candidate count to Akshay; he chooses exhaustive vs greedy. Do not start fitting before that.
- Commit code per task on branch `gecco-individual-differences`. Result artifacts committed only in Task 12.

## Existing utilities to reuse (do not rewrite)

- `library_learning/config.py`: `resolve_target(results_dir) -> Target` (`.data_path`, `.input_columns`, `.id_column`, `.models_dir`)
- `library_learning/loading.py`: `load_original_code(target, pid)`, `participant_ids(target)`, `extract_unpack_names(code)`, `parse_bounds(code, names)`, `participant_inputs(target, pid)`, `exec_model(code, func_name=None)`, `function_name_and_args(code)`, `strip_fences(text)`
- `library_learning/mining.py`: `mine_fragments(models: Dict[int, str])`, `render_mining_report(shared, n_models)`
- `gecco/offline_evaluation/evaluation_functions.py`: reference for BIC formula (reimplement in `compose/fitting.py`, don't import gecco)

## File Structure

```
library_learning/
  __main__.py               # CLI: compose-modules | compose-count | compose-search | compose-eval  (Task 10)
  compose/
    __init__.py             # empty                                                (Task 1)
    gemini.py               # .env loading, logged REST client, retry              (Task 1)
    splits.py               # seed/test pid derivation + OCI-stratified val/test   (Task 2)
    inventory.py            # Module/Inventory dataclasses, JSON validation        (Task 3)
    render.py               # backbone template + slot assembly + smoke check      (Task 4)
    fitting.py              # seeded per-participant L-BFGS-B fits + BIC           (Task 5)
    extract.py              # Gemini prompts, annotate/merge, MODULES.md           (Task 6)
    search.py               # enumeration, exhaustive + greedy search, logs        (Task 7)
    hybrid.py               # Daw hybrid likelihood source (baseline)              (Task 8)
    evaluate.py             # final-test fits, stats, RESULTS.md                   (Task 9)
    figure.py               # comparison figure, paper palette                     (Task 10)
tests/
  test_compose_gemini.py test_compose_splits.py test_compose_inventory.py
  test_compose_render.py test_compose_fitting.py test_compose_search.py
  test_compose_hybrid.py test_compose_evaluate.py test_compose_cli.py
```

---

### Task 1: Environment, package skeleton, logged Gemini client

**Files:**
- Create: `library_learning/compose/__init__.py`, `library_learning/compose/gemini.py`
- Test: `tests/test_compose_gemini.py`

**Interfaces:**
- Produces: `load_env_key(env_path=REPO_ROOT/".env", var="GEMINI_API_KEY_LAKELAB") -> str`;
  `class GeminiClient(log_dir, api_key=None, model="gemini-3.5-flash", temperature=0.0, transport=None)` with
  `generate(prompt: str, tag: str) -> str` (returns response text, writes `log_dir/call_{NNN}_{tag}.json`).
  `transport` is an injectable `fn(url, payload_dict) -> response_dict` for tests; default uses urllib with 5 retries / exponential backoff on HTTP 429/500/503.

- [ ] **Step 1: Install dev deps into the venv**

```bash
/Users/akshay/projects/gecco/gecco-env/bin/pip install pytest matplotlib
```
Expected: both install cleanly (py3.9 wheels exist).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_compose_gemini.py
import json
from pathlib import Path

from library_learning.compose.gemini import GeminiClient, load_env_key


def test_load_env_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text('GEMINI_API_KEY="bad"\nGEMINI_API_KEY_LAKELAB="AIzaGOOD"\n')
    assert load_env_key(env_path=env) == "AIzaGOOD"


def test_generate_logs_verbatim(tmp_path):
    calls = []

    def fake_transport(url, payload):
        calls.append((url, payload))
        return {"candidates": [{"content": {"parts": [{"text": "hello"}]}}],
                "modelVersion": "gemini-3.5-flash"}

    client = GeminiClient(log_dir=tmp_path, api_key="k", transport=fake_transport)
    out = client.generate("say hello", tag="smoke")
    assert out == "hello"
    assert "gemini-3.5-flash" in calls[0][0]
    assert calls[0][1]["generationConfig"]["temperature"] == 0.0

    log_files = sorted(tmp_path.glob("call_*.json"))
    assert len(log_files) == 1 and log_files[0].name == "call_000_smoke.json"
    logged = json.loads(log_files[0].read_text())
    assert logged["prompt"] == "say hello"
    assert logged["response"]["candidates"][0]["content"]["parts"][0]["text"] == "hello"
    assert logged["model"] == "gemini-3.5-flash"

    client.generate("again", tag="smoke")
    assert (tmp_path / "call_001_smoke.json").exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_gemini.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'library_learning.compose'`

- [ ] **Step 4: Implement**

```python
# library_learning/compose/__init__.py
```
(empty file)

```python
# library_learning/compose/gemini.py
"""Logged Gemini REST client (reproducibility: every prompt/response on disk)."""
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ..config import REPO_ROOT

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-3.5-flash"
ENV_VAR = "GEMINI_API_KEY_LAKELAB"  # plain GEMINI_API_KEY in .env is invalid
RETRY_STATUSES = {429, 500, 503}


def load_env_key(env_path=None, var=ENV_VAR):
    env_path = Path(env_path) if env_path else REPO_ROOT / ".env"
    for line in env_path.read_text().splitlines():
        m = re.match(r'\s*%s\s*=\s*"?([^"\s]+)"?' % re.escape(var), line)
        if m:
            return m.group(1)
    raise KeyError("%s not found in %s" % (var, env_path))


def _urllib_transport(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    last_err = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code not in RETRY_STATUSES:
                raise RuntimeError("Gemini HTTP %s: %s" % (e.code, e.read()[:500]))
            time.sleep(2 ** attempt)
    raise RuntimeError("Gemini API failed after 5 retries: %s" % last_err)


class GeminiClient:
    def __init__(self, log_dir, api_key=None, model=DEFAULT_MODEL,
                 temperature=0.0, transport=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or load_env_key()
        self.model = model
        self.temperature = temperature
        self.transport = transport or _urllib_transport
        self._n = len(list(self.log_dir.glob("call_*.json")))

    def generate(self, prompt, tag):
        url = API_URL.format(model=self.model) + "?key=" + self.api_key
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        response = self.transport(url, payload)
        record = {
            "model": self.model,
            "modelVersion": response.get("modelVersion"),
            "generationConfig": payload["generationConfig"],
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        path = self.log_dir / ("call_%03d_%s.json" % (self._n, tag))
        path.write_text(json.dumps(record, indent=2))
        self._n += 1
        return response["candidates"][0]["content"]["parts"][0]["text"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `gecco-env/bin/python -m pytest tests/test_compose_gemini.py -v`
Expected: 2 PASS. (Note: logged URL in test contains the key only in `calls[0][0]` in-memory; the on-disk record deliberately excludes the URL/key.)

- [ ] **Step 6: Commit**

```bash
git add library_learning/compose tests/test_compose_gemini.py
git commit -m "feat(compose): package skeleton + logged Gemini REST client"
```

---

### Task 2: Splits — seed pids from group config, OCI-stratified validation/test

**Files:**
- Create: `library_learning/compose/splits.py`
- Test: `tests/test_compose_splits.py`

**Interfaces:**
- Consumes: `resolve_target` (config.py), group config yaml.
- Produces: `parse_split(value, unique_ids) -> list` (mirror of `gecco/prepare_data/io.py`, reimplemented to avoid importing gecco);
  `group_split_pids(group_dir, config_dir=None, data_path=None) -> dict` with keys `seed` (prompt+eval), `heldout` (test);
  `make_splits(target, group_dir) -> dict` returning and writing `splits.json`:
  `{"seed_pids": [...], "validation_pids": [10], "test_pids": [21], "method": "oci-sorted alternation i%3==1", "oci_stats": {...}}`.
  Stratification rule (deterministic, no RNG): sort held-out pids by `(oci, pid)`; indices with `i % 3 == 1` → validation (10 of 31), rest → final test.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose_splits.py
import json
from pathlib import Path

from library_learning.config import resolve_target
from library_learning.compose.splits import group_split_pids, make_splits, parse_split

IND = "results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual"
GRP = "results/two_step_psychiatry_group_function_ocibalanced_maxsetting"


def test_parse_split_slice():
    assert parse_split("[1:3]", list(range(45))) == [1, 2]
    assert parse_split("[14:]", list(range(45))) == list(range(14, 45))


def test_group_split_pids():
    pids = group_split_pids(GRP)
    assert pids["seed"] == [1, 2] + list(range(4, 14))
    assert pids["heldout"] == list(range(14, 45))


def test_make_splits_deterministic_and_balanced(tmp_path):
    target = resolve_target(IND)
    s1 = make_splits(target, GRP, out_dir=tmp_path)
    s2 = make_splits(target, GRP, out_dir=tmp_path)
    assert s1 == s2
    assert len(s1["validation_pids"]) == 10 and len(s1["test_pids"]) == 21
    assert not set(s1["validation_pids"]) & set(s1["test_pids"])
    assert set(s1["validation_pids"]) | set(s1["test_pids"]) == set(range(14, 45))
    # OCI balance: means within 0.15 of each other
    st = s1["oci_stats"]
    assert abs(st["validation_mean"] - st["test_mean"]) < 0.15
    assert json.loads((tmp_path / "splits.json").read_text()) == s1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_splits.py -v`
Expected: FAIL — no module `splits`.

- [ ] **Step 3: Implement**

```python
# library_learning/compose/splits.py
"""Participant splits: library seeds from the group config; deterministic
OCI-stratified validation/test partition of the held-out participants."""
import json
from pathlib import Path

import pandas as pd
import yaml

from ..config import REPO_ROOT, Target, resolve_target


def parse_split(value, unique_ids):
    """Mirror of gecco/prepare_data/io.py::parse_split (index slice over sorted ids)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        start_str, end_str = value[1:-1].split(":")
        start = int(start_str) if start_str else None
        end = int(end_str) if end_str else None
        return unique_ids[start:end]
    raise ValueError("unsupported split spec: %r" % (value,))


def _group_config(group_dir, config_dir=None):
    config_dir = Path(config_dir) if config_dir else REPO_ROOT / "config"
    task_name = Path(group_dir).name
    for yaml_path in sorted(config_dir.glob("*.yaml")):
        try:
            cfg = yaml.safe_load(yaml_path.read_text())
        except Exception:
            continue
        if isinstance(cfg, dict) and cfg.get("task", {}).get("name") == task_name:
            return cfg
    raise FileNotFoundError("no config with task.name == %r" % task_name)


def group_split_pids(group_dir, config_dir=None, data_path=None):
    cfg = _group_config(group_dir, config_dir)
    data_sec = cfg["data"]
    path = Path(data_path) if data_path else REPO_ROOT / data_sec["path"]
    df = pd.read_csv(path)
    unique_ids = sorted(df[data_sec.get("id_column", "participant")].unique().tolist())
    splits = data_sec["splits"]
    prompt = parse_split(splits["prompt"], unique_ids)
    ev = parse_split(splits["eval"], unique_ids)
    heldout = parse_split(splits["test"], unique_ids)
    return {"seed": sorted(prompt + ev), "heldout": sorted(heldout)}


def make_splits(target, group_dir, out_dir=None, oci_column="oci"):
    """Write splits.json: seeds + OCI-stratified validation(10)/test(21).

    Deterministic: held-out pids sorted by (oci, pid); index i % 3 == 1 ->
    validation. 31 held-out => 10 validation, 21 final test.
    """
    out_dir = Path(out_dir) if out_dir else target.results_dir / "library_composition"
    out_dir.mkdir(parents=True, exist_ok=True)
    pids = group_split_pids(group_dir, data_path=target.data_path)
    df = pd.read_csv(target.data_path)
    oci = df.groupby(target.id_column)[oci_column].first()
    ranked = sorted(pids["heldout"], key=lambda p: (float(oci[p]), p))
    validation = [p for i, p in enumerate(ranked) if i % 3 == 1]
    test = [p for p in pids["heldout"] if p not in validation]
    result = {
        "seed_pids": pids["seed"],
        "validation_pids": sorted(validation),
        "test_pids": sorted(test),
        "method": "oci-sorted alternation i%3==1",
        "oci_stats": {
            "validation_mean": float(oci[validation].mean()),
            "validation_std": float(oci[validation].std()),
            "test_mean": float(oci[test].mean()),
            "test_std": float(oci[test].std()),
        },
    }
    (out_dir / "splits.json").write_text(json.dumps(result, indent=2))
    return result


def load_splits(target):
    path = target.results_dir / "library_composition" / "splits.json"
    return json.loads(path.read_text())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `gecco-env/bin/python -m pytest tests/test_compose_splits.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/splits.py tests/test_compose_splits.py
git commit -m "feat(compose): seed pids from group config + OCI-stratified val/test split"
```

---

### Task 3: Module inventory schema + validation

**Files:**
- Create: `library_learning/compose/inventory.py`
- Test: `tests/test_compose_inventory.py`

**Interfaces:**
- Produces:
  - `APPEND_SLOTS = ("init", "pre_stage1", "stage1_logits_extra", "stage2_logits_extra", "update_extra", "post_trial")`
  - `OVERRIDE_SLOTS = ("q2_init", "stage1_values", "stage2_values", "stage1_temp", "stage2_temp", "stage1_update", "stage2_update")`
  - `class Param(name: str, bounds: Tuple[float, float])`
  - `class Module(id, name, description, params: List[Param], slots: Dict[str, str], overrides: Dict[str, str], provenance: List[int], excludes: List[str])`
  - `class Inventory(modules: List[Module])` with `.module(mid)`, `.ids()`
  - `load_inventory(path) -> Inventory` / `parse_inventory(obj: dict) -> Inventory` — raises `InventoryError` (message lists ALL problems) on: duplicate module ids, param name collisions (across all modules and vs backbone `learning_rate`/`beta`), non-numeric or lo>=hi bounds, unknown slot names, empty modules, unknown ids in `excludes`.
  - `compatible(inventory, module_ids) -> Tuple[bool, str]` — False if any pairwise `excludes` hit OR two modules override the same slot.
- Consumed by: render (Task 4), extract (Task 6), search (Task 7).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose_inventory.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_inventory.py -v`
Expected: FAIL — no module `inventory`.

- [ ] **Step 3: Implement**

```python
# library_learning/compose/inventory.py
"""Module inventory: the validated data model behind module_inventory.json."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

APPEND_SLOTS = ("init", "pre_stage1", "stage1_logits_extra",
                "stage2_logits_extra", "update_extra", "post_trial")
OVERRIDE_SLOTS = ("q2_init", "stage1_values", "stage2_values",
                  "stage1_temp", "stage2_temp", "stage1_update", "stage2_update")
BACKBONE_PARAM_NAMES = ("learning_rate", "beta")


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

    def ids(self):
        return [m.id for m in self.modules]

    def module(self, mid):
        for m in self.modules:
            if m.id == mid:
                return m
        raise KeyError(mid)


def parse_inventory(obj):
    errors = []
    modules = []
    seen_ids = set()
    seen_params = set(BACKBONE_PARAM_NAMES)
    for raw in obj.get("modules", []):
        mid = raw.get("id", "<missing id>")
        if mid in seen_ids:
            errors.append("duplicate module id: %s" % mid)
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
        modules.append(Module(
            id=mid, name=raw.get("name", mid), description=raw.get("description", ""),
            params=params, slots=slots, overrides=overrides,
            provenance=list(raw.get("provenance", [])),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `gecco-env/bin/python -m pytest tests/test_compose_inventory.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/inventory.py tests/test_compose_inventory.py
git commit -m "feat(compose): module inventory schema + validation + compatibility"
```

---

### Task 4: Candidate renderer (backbone + slots → standalone cognitive_model)

**Files:**
- Create: `library_learning/compose/render.py`
- Test: `tests/test_compose_render.py`

**Interfaces:**
- Consumes: `Inventory`, `APPEND_SLOTS`/`OVERRIDE_SLOTS` (Task 3); `exec_model`, `extract_unpack_names`, `parse_bounds` (loading.py).
- Produces:
  - `candidate_id(module_ids) -> str` — `"backbone"` or `"+".join(sorted(module_ids))`
  - `candidate_params(inventory, module_ids) -> List[Param]` — backbone `learning_rate [0,1]`, `beta [0,10]` first, then module params in sorted-module order.
  - `render_candidate(inventory, module_ids) -> str` — full standalone source, gecco-compatible docstring/unpack.
  - `smoke_check(source) -> float` — exec + run on 8-trial dummy data **containing -1 missed trials**, midpoint-of-bounds params; raises on non-finite/exception; returns NLL.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_render.py -v`
Expected: FAIL — no module `render`.

- [ ] **Step 3: Implement**

```python
# library_learning/compose/render.py
"""Assemble backbone + module slots into one standalone cognitive_model.

The backbone is the completed gecco template model (MB stage-1 policy, MF
stage-2, single learning_rate/beta TD updates), hardened for -1 missed
trials. Modules append statements to APPEND_SLOTS and/or replace
OVERRIDE_SLOTS expressions. Everything renders to flat numpy-only source
whose docstring bounds and unpack line stay parseable by gecco's regexes.
"""
import textwrap

import numpy as np

from .inventory import APPEND_SLOTS, OVERRIDE_SLOTS, Param
from ..loading import exec_model

BACKBONE_PARAMS = [Param("learning_rate", (0.0, 1.0)), Param("beta", (0.0, 10.0))]

DEFAULT_OVERRIDES = {
    "q2_init": "np.zeros((2, 2))",
    "stage1_values": "q_stage1_mb",
    "stage2_values": "q_stage2_mf[s_idx]",
    "stage1_temp": "beta",
    "stage2_temp": "beta",
    "stage1_update": "q_stage1_mf[a1] += learning_rate * delta_stage1",
    "stage2_update": "q_stage2_mf[s_idx, a2] += learning_rate * delta_stage2",
}

BACKBONE_PARAM_DOCS = {
    "learning_rate": "TD learning rate.",
    "beta": "Softmax inverse temperature.",
}


def candidate_id(module_ids):
    return "+".join(sorted(module_ids)) if module_ids else "backbone"


def candidate_params(inventory, module_ids):
    params = list(BACKBONE_PARAMS)
    for mid in sorted(module_ids):
        params.extend(inventory.module(mid).params)
    return params


def _indent(code, level):
    return textwrap.indent(textwrap.dedent(code).strip(), "    " * level)


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
        return ("\n".join(parts) + "\n") if parts else ""

    src = '''def cognitive_model(action_1, state, action_2, reward, model_parameters):
    """
    {docstring}
    """
    {unpack} = model_parameters
    n_trials = len(action_1)
    transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    q_stage1_mf = np.zeros(2)
    q_stage2_mf = {q2_init}
    log_loss = 0.0
    eps = 1e-10
{init}
    for trial in range(n_trials):
        a1 = int(action_1[trial])
        s_idx = int(state[trial])
        a2 = int(action_2[trial])
        r = float(reward[trial])

        max_q_stage2 = np.max(q_stage2_mf, axis=1)
        q_stage1_mb = transition_matrix @ max_q_stage2
{pre_stage1}
        stage1_values = {stage1_values}
        logits_1 = ({stage1_temp}) * stage1_values
{stage1_logits_extra}
        exp_q1 = np.exp(logits_1)
        probs_1 = exp_q1 / np.sum(exp_q1)
        if a1 != -1:
            log_loss -= np.log(probs_1[a1] + eps)

        if s_idx != -1 and a2 != -1:
            stage2_values = {stage2_values}
            logits_2 = ({stage2_temp}) * stage2_values
{stage2_logits_extra}
            exp_q2 = np.exp(logits_2)
            probs_2 = exp_q2 / np.sum(exp_q2)
            log_loss -= np.log(probs_2[a2] + eps)

        if a1 != -1 and s_idx != -1 and a2 != -1:
            delta_stage1 = q_stage2_mf[s_idx, a2] - q_stage1_mf[a1]
            {stage1_update}
            delta_stage2 = r - q_stage2_mf[s_idx, a2]
            {stage2_update}
{update_extra}
{post_trial}
    return log_loss
'''.format(
        docstring=docstring,
        unpack=unpack,
        q2_init=overrides["q2_init"],
        stage1_values=overrides["stage1_values"],
        stage1_temp=overrides["stage1_temp"],
        stage2_values=overrides["stage2_values"],
        stage2_temp=overrides["stage2_temp"],
        stage1_update=overrides["stage1_update"],
        stage2_update=overrides["stage2_update"],
        init=block("init", 1),
        pre_stage1=block("pre_stage1", 2),
        stage1_logits_extra=block("stage1_logits_extra", 2),
        stage2_logits_extra=block("stage2_logits_extra", 3),
        update_extra=block("update_extra", 3),
        post_trial=block("post_trial", 2),
    )
    # drop blank slot lines so the source stays tidy
    src = "\n".join(line for line in src.splitlines() if line.strip() != "") + "\n"
    return src


SMOKE_DATA = {
    "action_1": np.array([0, 1, -1, 0, 1, 0, -1, 1]),
    "state":    np.array([0, 1, -1, 1, 0, 0, -1, 1]),
    "action_2": np.array([1, 0, -1, -1, 1, 0, -1, 0]),
    "reward":   np.array([1, 0, -1, -1, 1, 0, -1, 1]),
}


def smoke_check(source, inventory=None, module_ids=None, params=None):
    """Exec + run on dummy data with -1 missed trials; return NLL or raise."""
    func = exec_model(source, "cognitive_model")
    if params is None:
        import re
        bounds = re.findall(r"\[\s*([\-\d.eE+]+)\s*,\s*([\-\d.eE+]+)\s*\]",
                            source.split('"""')[1])
        params = [(float(lo) + float(hi)) / 2.0 for lo, hi in bounds]
    nll = float(func(SMOKE_DATA["action_1"], SMOKE_DATA["state"],
                     SMOKE_DATA["action_2"], SMOKE_DATA["reward"], params))
    if not np.isfinite(nll):
        raise ValueError("smoke_check: non-finite NLL %r" % nll)
    return nll
```

- [ ] **Step 4: Run tests, fix indentation issues until green**

Run: `gecco-env/bin/python -m pytest tests/test_compose_render.py -v`
Expected: 4 PASS. (Slot indentation is the fiddly part — `stage2_logits_extra`/`update_extra` are inside two nested `if`s, hence level 3. Print the rendered source when debugging.)

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/render.py tests/test_compose_render.py
git commit -m "feat(compose): backbone+slot candidate renderer with -1-safe smoke check"
```

---

### Task 5: Seeded fitting

**Files:**
- Create: `library_learning/compose/fitting.py`
- Test: `tests/test_compose_fitting.py`

**Interfaces:**
- Consumes: `participant_inputs` (loading.py).
- Produces:
  - `bic(nll, k, n) -> float` = `math.log(n)*k + 2*nll`
  - `seed_for(tag: str, pid: int) -> int` = `int(hashlib.md5(("%s:%d" % (tag, pid)).encode()).hexdigest()[:8], 16)`
  - `fit_participant(func, inputs, bounds: List[Tuple[float,float]], seed, n_starts=10) -> dict` — `{"nll": float, "params": list, "n_starts": int}`; L-BFGS-B from uniform seeded starts; non-finite objective values mapped to `1e10`.
  - `fit_model_on_pids(source_or_func, target, pids, bounds, tag, func_name="cognitive_model") -> dict` — per-pid `{"nll", "bic", "params", "seed"}` using `n=200` trials each.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose_fitting.py
import numpy as np

from library_learning.compose.fitting import bic, fit_participant, seed_for

RNG_DATA = np.random.default_rng(0)


def biased_coin_model(action_1, state, action_2, reward, model_parameters):
    p, = model_parameters
    eps = 1e-10
    ll = np.where(action_1 == 1, np.log(p + eps), np.log(1 - p + eps))
    return -float(np.sum(ll))


def test_bic():
    assert abs(bic(100.0, 2, 200) - (np.log(200) * 2 + 200.0)) < 1e-12


def test_seed_deterministic():
    assert seed_for("cand", 14) == seed_for("cand", 14)
    assert seed_for("cand", 14) != seed_for("cand", 15)


def test_fit_recovers_bias():
    a1 = (RNG_DATA.random(500) < 0.8).astype(int)
    inputs = [a1, a1, a1, a1]
    res = fit_participant(biased_coin_model, inputs, [(0.001, 0.999)],
                          seed=seed_for("t", 0), n_starts=5)
    assert abs(res["params"][0] - a1.mean()) < 0.02
    res2 = fit_participant(biased_coin_model, inputs, [(0.001, 0.999)],
                           seed=seed_for("t", 0), n_starts=5)
    assert res == res2  # fully deterministic
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_fitting.py -v`
Expected: FAIL — no module `fitting`.

- [ ] **Step 3: Implement**

```python
# library_learning/compose/fitting.py
"""Seeded per-participant fitting, mirroring gecco run_fit's protocol
(L-BFGS-B, uniform random starts within bounds, n_starts=10) but fully
reproducible via per-(tag, pid) seeds."""
import hashlib
import math

import numpy as np
from scipy.optimize import minimize

from ..loading import exec_model, participant_inputs

N_STARTS = 10


def bic(nll, k, n):
    return math.log(n) * k + 2.0 * nll


def seed_for(tag, pid):
    return int(hashlib.md5(("%s:%d" % (tag, pid)).encode()).hexdigest()[:8], 16)


def fit_participant(func, inputs, bounds, seed, n_starts=N_STARTS):
    rng = np.random.default_rng(seed)

    def objective(x):
        try:
            v = float(func(*inputs, x))
        except Exception:
            return 1e10
        return v if np.isfinite(v) else 1e10

    best_nll, best_x = np.inf, None
    for _ in range(n_starts):
        x0 = [rng.uniform(lo, hi) for lo, hi in bounds]
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        if res.fun < best_nll:
            best_nll, best_x = float(res.fun), [float(v) for v in res.x]
    return {"nll": best_nll, "params": best_x, "n_starts": n_starts}


def fit_model_on_pids(source_or_func, target, pids, bounds, tag,
                      func_name="cognitive_model", n_starts=N_STARTS):
    func = (exec_model(source_or_func, func_name)
            if isinstance(source_or_func, str) else source_or_func)
    out = {}
    for pid in pids:
        inputs, n = participant_inputs(target, pid)
        seed = seed_for(tag, pid)
        res = fit_participant(func, inputs, bounds, seed, n_starts)
        out[pid] = {"nll": res["nll"], "bic": bic(res["nll"], len(bounds), n),
                    "params": res["params"], "seed": seed}
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `gecco-env/bin/python -m pytest tests/test_compose_fitting.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/fitting.py tests/test_compose_fitting.py
git commit -m "feat(compose): seeded L-BFGS-B fitting + BIC"
```

---

### Task 6: Gemini module extraction (annotate → merge → validate → render inventory)

**Files:**
- Create: `library_learning/compose/extract.py`
- Test: `tests/test_compose_extract.py`

**Interfaces:**
- Consumes: `GeminiClient` (Task 1), `mine_fragments`/`render_mining_report` (mining.py), `load_original_code` (loading.py), `parse_inventory`/`InventoryError` (Task 3), `render_candidate`/`smoke_check` (Task 4).
- Produces:
  - `annotate_seed(client, pid, code) -> dict` — one logged call, tag `annotate_p{pid}`; returns parsed JSON `{"mechanisms": [...]}`.
  - `merge_inventory(client, annotations, mining_report, max_repair_rounds=3) -> dict` — tag `merge` (+ `merge_repair{i}`); returns raw inventory dict.
  - `validate_inventory_obj(obj) -> Tuple[Inventory, List[str]]` — parses via `parse_inventory`, then smoke-tests **each module rendered alone with the backbone**; returns errors list (empty = good).
  - `run_extraction(target, group_dir, out_dir) -> Inventory` — full pipeline: mine seeds → annotate 12 → merge (with repair loop feeding validation errors back to Gemini, all logged) → write `module_inventory.json`, `mining_report.md`, `MODULES.md`. Raises with instructions to record manual fixes in `llm_log/MANUAL_EDITS.md` if still invalid after repairs.
  - `PROMPT_ANNOTATE`, `PROMPT_MERGE` string templates (kept in this file so the exact prompts are versioned).
- `_parse_json_reply(text)` — strips ``` fences, `json.loads`, raises with the offending text on failure.

**Prompt templates (exact content to put in the file):**

```python
PROMPT_ANNOTATE = '''You are a renowned cognitive scientist analyzing computational models of the two-step decision task (Daw et al.). Below is a Python cognitive model that was fit to participant {pid}'s behavior.

```python
{code}
```

List every distinct psychological mechanism in this model. A mechanism is one separable computational assumption (e.g. "MB/MF mixture weight", "choice stickiness", "reward-dependent stickiness", "separate stage-2 learning rate", "value decay/forgetting", "eligibility trace", "optimistic Q initialization").

Return STRICT JSON only (no prose, no markdown fences):
{{"mechanisms": [{{"name": "short snake_case id", "title": "human name", "description": "one sentence", "params": [{{"name": "param name as in code", "bounds": [lo, hi]}}], "evidence": "the exact code lines implementing it"}}]}}

Rules: do NOT list backbone machinery shared by all models (softmax choice, basic TD updates, MB lookahead with the fixed 0.7/0.3 transition matrix, NLL accumulation) as mechanisms. Only list deviations from that backbone.'''

PROMPT_MERGE = '''You are building a library of cognitive mechanism modules for the two-step task. You are given (A) mechanism annotations extracted from 12 participants' models, (B) a mining report of literally-shared code fragments, and (C) the fixed BACKBONE program that modules will be injected into.

(A) Annotations:
{annotations}

(B) Mining report:
{mining_report}

(C) Backbone (modules inject into the named slots of this exact program):
```python
{backbone}
```

Slot contract — a module may provide:
- "slots": statements APPENDED at these points: "init" (before the trial loop), "pre_stage1" (each trial, after q_stage1_mb computed), "stage1_logits_extra" (after logits_1 assigned), "stage2_logits_extra" (after logits_2 assigned), "update_extra" (after the two TD updates, inside the all-observed guard), "post_trial" (end of each trial iteration).
- "overrides": expressions/statements REPLACING these defaults: "q2_init" (default "np.zeros((2, 2))"), "stage1_values" (default "q_stage1_mb"), "stage2_values" (default "q_stage2_mf[s_idx]"), "stage1_temp" (default "beta"), "stage2_temp" (default "beta"), "stage1_update" (default "q_stage1_mf[a1] += learning_rate * delta_stage1"), "stage2_update" (default "q_stage2_mf[s_idx, a2] += learning_rate * delta_stage2").

Available variable names (use ONLY these plus your own module parameters): action_1, state, action_2, reward, n_trials, trial, a1, s_idx, a2, r, transition_matrix, q_stage1_mf, q_stage2_mf, q_stage1_mb, max_q_stage2, stage1_values, stage2_values, logits_1, logits_2, delta_stage1, delta_stage2, learning_rate, beta, np.

Produce the deduplicated module inventory. Return STRICT JSON only:
{{"modules": [{{"id": "snake_case", "name": "...", "description": "...", "params": [{{"name": "...", "bounds": [lo, hi]}}], "slots": {{...}}, "overrides": {{...}}, "provenance": [participant ids], "excludes": ["ids of incompatible modules"]}}]}}

Rules:
1. One module per distinct mechanism — merge identical mechanisms across participants (union their provenance). Keep singletons (mechanisms found in only one participant).
2. Parameter names must be globally unique across ALL modules and must not be "learning_rate" or "beta"; suffix if needed (e.g. "alpha_2", "beta_2").
3. Code must be -1-safe: missed trials have a1/s_idx/a2 == -1; never index an array with a possibly -1 value inside your snippets (the backbone already guards likelihood and TD updates; guard your own "post_trial"/"init"-state updates like `if a1 != -1:`).
4. Use plain numpy, no imports, no helper functions.
5. Bounds: probabilities/rates/weights [0, 1]; inverse temperatures [0, 10]; additive bonuses (stickiness etc.) [0, 5] unless the source model's docstring says otherwise.
6. List modules that implement alternative versions of the same computation (e.g. two different stage1_values formulas) in each other's "excludes".'''

PROMPT_REPAIR = '''Your previous module inventory JSON had problems. Fix ALL of them and return the corrected STRICT JSON (same schema, no prose):

Problems:
{errors}

Previous JSON:
{previous}'''
```

- [ ] **Step 1: Write the failing test** (fake transport; no live API)

```python
# tests/test_compose_extract.py
import json

from library_learning.compose.extract import (
    _parse_json_reply, validate_inventory_obj)


def test_parse_json_reply_strips_fences():
    obj = _parse_json_reply('```json\n{"mechanisms": []}\n```')
    assert obj == {"mechanisms": []}
    obj = _parse_json_reply('{"a": 1}')
    assert obj == {"a": 1}


def test_validate_inventory_smoke_catches_bad_code():
    good = {"modules": [
        {"id": "stick", "name": "stickiness", "description": "d",
         "params": [{"name": "stickiness", "bounds": [0, 5]}],
         "slots": {"init": "last_a1 = -1",
                   "stage1_logits_extra": "if last_a1 != -1:\n    logits_1[last_a1] += stickiness",
                   "post_trial": "if a1 != -1:\n    last_a1 = a1"},
         "overrides": {}, "provenance": [1], "excludes": []}]}
    inv, errors = validate_inventory_obj(good)
    assert errors == [] and inv is not None

    bad = json.loads(json.dumps(good))
    bad["modules"][0]["slots"]["init"] = "last_a1 = undefined_thing"
    inv, errors = validate_inventory_obj(bad)
    assert errors and "stick" in errors[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_extract.py -v`
Expected: FAIL — no module `extract`.

- [ ] **Step 3: Implement**

```python
# library_learning/compose/extract.py
"""Stage 1: Gemini-driven module extraction. Every call is logged by
GeminiClient; validation and rendering stay deterministic Python."""
import json
import re
from pathlib import Path

from .gemini import GeminiClient
from .inventory import InventoryError, parse_inventory
from .render import render_candidate, smoke_check
from ..loading import load_original_code
from ..mining import mine_fragments, render_mining_report

# PROMPT_ANNOTATE / PROMPT_MERGE / PROMPT_REPAIR exactly as specified in the
# plan's "Prompt templates" block above.

FENCE_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _parse_json_reply(text):
    m = FENCE_JSON_RE.search(text)
    raw = m.group(1) if m else text
    try:
        return json.loads(raw.strip())
    except Exception as e:
        raise ValueError("Gemini reply is not valid JSON (%s):\n%s" % (e, text[:2000]))


def annotate_seed(client, pid, code):
    reply = client.generate(PROMPT_ANNOTATE.format(pid=pid, code=code),
                            tag="annotate_p%d" % pid)
    return _parse_json_reply(reply)


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
    return (inv if not errors else None), errors


def merge_inventory(client, annotations, mining_report, max_repair_rounds=3):
    from .render import render_candidate as _rc
    from .inventory import Inventory
    backbone_src = _rc(Inventory(modules=[]), [])
    prompt = PROMPT_MERGE.format(
        annotations=json.dumps(annotations, indent=1),
        mining_report=mining_report,
        backbone=backbone_src)
    reply = client.generate(prompt, tag="merge")
    obj = _parse_json_reply(reply)
    inv, errors = validate_inventory_obj(obj)
    rounds = 0
    while errors and rounds < max_repair_rounds:
        rounds += 1
        reply = client.generate(
            PROMPT_REPAIR.format(errors="\n".join("- " + e for e in errors),
                                 previous=json.dumps(obj, indent=1)),
            tag="merge_repair%d" % rounds)
        obj = _parse_json_reply(reply)
        inv, errors = validate_inventory_obj(obj)
    if errors:
        raise InventoryError(
            "inventory still invalid after %d repair rounds: %s\n"
            "Fix module_inventory.json by hand and record every edit in "
            "llm_log/MANUAL_EDITS.md" % (max_repair_rounds, "; ".join(errors)))
    return obj


def render_modules_md(inv, seed_pids):
    lines = ["# Cognitive module library", "",
             "Extracted by gemini-3.5-flash (temperature 0, logged in llm_log/) "
             "from the individual best programs of participants %s." % seed_pids, ""]
    for m in inv.modules:
        lines += ["## %s (`%s`)" % (m.name, m.id), "", m.description, "",
                  "- params: " + (", ".join("%s %s" % (p.name, list(p.bounds))
                                            for p in m.params) or "none"),
                  "- provenance: participants %s" % m.provenance,
                  "- excludes: %s" % (m.excludes or "none"), ""]
    return "\n".join(lines)


def run_extraction(target, group_dir, out_dir, client=None):
    from .splits import make_splits
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
    obj = merge_inventory(client, annotations, report)
    (out_dir / "module_inventory.json").write_text(json.dumps(obj, indent=2))
    inv, errors = validate_inventory_obj(obj)
    assert not errors
    (out_dir / "MODULES.md").write_text(render_modules_md(inv, seed_pids))
    return inv
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `gecco-env/bin/python -m pytest tests/test_compose_extract.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/extract.py tests/test_compose_extract.py
git commit -m "feat(compose): Gemini extraction pipeline with validation/repair loop"
```

---

### Task 7: Combination enumeration + search

**Files:**
- Create: `library_learning/compose/search.py`
- Test: `tests/test_compose_search.py`

**Interfaces:**
- Consumes: `compatible` (Task 3), `render_candidate`/`candidate_id`/`candidate_params`/`smoke_check` (Task 4), `fit_model_on_pids` (Task 5).
- Produces:
  - `enumerate_candidates(inventory, param_cap=8) -> List[Tuple[str, ...]]` — all compatibility-valid, cap-respecting module-id tuples (including the empty tuple = backbone), deterministic order.
  - `count_report(inventory, param_cap=8) -> dict` — `{"n_candidates", "n_modules", "by_n_params": {k: count}, "by_n_modules": {...}}`.
  - `score_candidates(inventory, target, validation_pids, candidates, out_dir) -> list` — smoke-check then fit each candidate on validation pids; appends each result to `search_log.jsonl` immediately (crash-safe); returns list of `{"candidate_id", "module_ids", "n_params", "per_pid": {...}, "mean_bic"}`.
  - `greedy_search(inventory, target, validation_pids, out_dir, param_cap=8) -> list` — forward selection from backbone; same result-record shape; logs every evaluated candidate.
  - `select_winner(results) -> dict` — best mean BIC; ties within 1.0 → fewer params.
  - `freeze_winner(result, inventory, out_dir)` — writes `composed_model.txt` (rendered source) and `winner.json` (module ids, validation BICs).

- [ ] **Step 1: Write the failing test** (stub fitting → fast)

```python
# tests/test_compose_search.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_search.py -v`
Expected: FAIL — no module `search`.

- [ ] **Step 3: Implement**

```python
# library_learning/compose/search.py
"""Deterministic composition search over module combinations."""
import itertools
import json
from collections import Counter
from pathlib import Path

from .inventory import compatible
from .render import (candidate_id as candidate_id_of, candidate_params,
                     render_candidate, smoke_check)
from .fitting import fit_model_on_pids

PARAM_CAP = 8
TIE_TOL = 1.0


def _n_params(inventory, mods):
    return 2 + sum(inventory.module(m).n_params for m in mods)


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
            cid = candidate_id_of(list(mods))
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


def select_winner(results):
    best = min(results, key=lambda r: r["mean_bic"])
    contenders = [r for r in results if r["mean_bic"] <= best["mean_bic"] + TIE_TOL]
    return min(contenders, key=lambda r: (r["n_params"], r["mean_bic"]))


def freeze_winner(result, inventory, out_dir):
    out_dir = Path(out_dir)
    src = render_candidate(inventory, result["module_ids"])
    (out_dir / "composed_model.txt").write_text(src)
    (out_dir / "winner.json").write_text(json.dumps(result, indent=2))
```

Note for Step 3: `greedy_search` must call `score_candidates` via module attribute (as written — plain name resolves at module level, which `monkeypatch.setattr(S, ...)` replaces; keep the call unqualified and the test's monkeypatch works).

- [ ] **Step 4: Run tests to verify they pass**

Run: `gecco-env/bin/python -m pytest tests/test_compose_search.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/search.py tests/test_compose_search.py
git commit -m "feat(compose): candidate enumeration, exhaustive/greedy search, winner freeze"
```

---

### Task 8: Hybrid (Daw) baseline likelihood

**Files:**
- Create: `library_learning/compose/hybrid.py`
- Test: `tests/test_compose_hybrid.py`

**Interfaces:**
- Produces: `HYBRID_SOURCE: str` (standalone `cognitive_model`, 7 params) and `HYBRID_BOUNDS: list` in unpack order. Likelihood twin of the simulation template in `config/two_step_psychiatry_group_ocd_maxsetting.yaml` (`simulation_template`): params `learning_rate, learning_rate_2, beta, beta_2, w, lambd, perseveration`; stage-1 `q_net = w*q_mb + (1-w)*q_mf_stage1 + perseveration*pers_array`, softmax `beta`; stage-2 softmax `beta_2` on `q_mf[state]`; updates `q1 += lr*delta1`, `q2 += lr2*delta2`, eligibility `q1 += lambd*lr*delta2`; pers_array one-hot of last stage-1 choice. `-1`-guarded like the backbone.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose_hybrid.py
import numpy as np

from library_learning.compose.fitting import bic, fit_model_on_pids
from library_learning.compose.hybrid import HYBRID_BOUNDS, HYBRID_SOURCE
from library_learning.compose.render import smoke_check
from library_learning.config import resolve_target
from library_learning.loading import extract_unpack_names

IND = "results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual"


def test_hybrid_source_shape():
    assert extract_unpack_names(HYBRID_SOURCE) == [
        "learning_rate", "learning_rate_2", "beta", "beta_2", "w", "lambd", "perseveration"]
    assert len(HYBRID_BOUNDS) == 7
    assert np.isfinite(smoke_check(HYBRID_SOURCE, params=[0.5, 0.5, 2.0, 2.0, 0.5, 0.5, 0.5]))


def test_hybrid_fits_close_to_stored_baseline():
    target = resolve_target(IND)
    import pandas as pd
    df = pd.read_csv(target.data_path)
    stored = float(df[df.participant == 14].baseline_bic.iloc[0])
    fits = fit_model_on_pids(HYBRID_SOURCE, target, [14], HYBRID_BOUNDS, tag="hybrid-test")
    # informational cross-check: same model family, different optimizer runs
    assert abs(fits[14]["bic"] - stored) < 25.0
    print("refit %.2f vs stored %.2f" % (fits[14]["bic"], stored))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_hybrid.py -v`
Expected: FAIL — no module `hybrid`.

- [ ] **Step 3: Implement**

```python
# library_learning/compose/hybrid.py
"""Likelihood twin of the config's hybrid simulation model (Daw hybrid with
eligibility trace + perseveration). Field-standard baseline; cross-checked
against the data's baseline_bic column (WARN-level in evaluate)."""

HYBRID_BOUNDS = [(0.0, 1.0), (0.0, 1.0), (0.0, 10.0), (0.0, 10.0),
                 (0.0, 1.0), (0.0, 1.0), (0.0, 5.0)]

HYBRID_SOURCE = '''def cognitive_model(action_1, state, action_2, reward, model_parameters):
    """
    Hybrid MB/MF model with eligibility trace and perseveration (Daw baseline).

    Parameters:
    learning_rate: [0, 1] - stage-1 learning rate
    learning_rate_2: [0, 1] - stage-2 learning rate
    beta: [0, 10] - stage-1 inverse temperature
    beta_2: [0, 10] - stage-2 inverse temperature
    w: [0, 1] - MB weight
    lambd: [0, 1] - eligibility trace
    perseveration: [0, 5] - stage-1 choice repetition bonus
    """
    learning_rate, learning_rate_2, beta, beta_2, w, lambd, perseveration = model_parameters
    n_trials = len(action_1)
    transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    q_mf = np.zeros((3, 2))
    pers_array = np.zeros(2)
    log_loss = 0.0
    eps = 1e-10
    for trial in range(n_trials):
        a1 = int(action_1[trial])
        s2 = int(state[trial])
        a2 = int(action_2[trial])
        r = float(reward[trial])
        max_q_stage2 = np.max(q_mf[1:], axis=1)
        q_mb = transition_matrix @ max_q_stage2
        q_net = w * q_mb + (1 - w) * q_mf[0] + perseveration * pers_array
        exp_q1 = np.exp(beta * q_net)
        probs_1 = exp_q1 / np.sum(exp_q1)
        if a1 != -1:
            log_loss -= np.log(probs_1[a1] + eps)
        if s2 != -1 and a2 != -1:
            state_idx = s2 + 1
            exp_q2 = np.exp(beta_2 * q_mf[state_idx])
            probs_2 = exp_q2 / np.sum(exp_q2)
            log_loss -= np.log(probs_2[a2] + eps)
        if a1 != -1 and s2 != -1 and a2 != -1:
            state_idx = s2 + 1
            delta1 = q_mf[state_idx, a2] - q_mf[0, a1]
            q_mf[0, a1] += learning_rate * delta1
            delta2 = r - q_mf[state_idx, a2]
            q_mf[state_idx, a2] += learning_rate_2 * delta2
            q_mf[0, a1] += lambd * learning_rate * delta2
            pers_array.fill(0)
            pers_array[a1] = 1
    return log_loss
'''
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `gecco-env/bin/python -m pytest tests/test_compose_hybrid.py -v -s`
Expected: 2 PASS; the printed refit-vs-stored gap should be small (a few BIC points). If the second test's 25-point tolerance trips, investigate (bounds mismatch vs. the original baseline fit) before loosening anything.

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/hybrid.py tests/test_compose_hybrid.py
git commit -m "feat(compose): Daw hybrid baseline likelihood"
```

---

### Task 9: Final-test evaluation + stats + RESULTS.md

**Files:**
- Create: `library_learning/compose/evaluate.py`
- Test: `tests/test_compose_evaluate.py`

**Interfaces:**
- Consumes: `fit_model_on_pids`/`bic` (Task 5), `HYBRID_SOURCE`/`HYBRID_BOUNDS` (Task 8), `load_original_code`/`extract_unpack_names`/`parse_bounds`/`strip_fences` (loading.py), `load_splits` (Task 2).
- Produces:
  - `evaluate_models(target, group_dir, out_dir, pids, set_name) -> dict` — fits four entries on `pids`:
    - `composed`: `out_dir/composed_model.txt` (must exist — freeze discipline: raise `FileNotFoundError` with a clear message otherwise)
    - `group`: `<group_dir>/models/best_model_0.txt` (strip fences; func name from `function_name_and_args`; bounds via `extract_unpack_names` + `parse_bounds`)
    - `hybrid`: `HYBRID_SOURCE`
    - `individual`: per-pid own best model (ceiling; tag `individual`)
    - tags for seeding: `"eval:{set_name}:{model}"`.
  - `cross_checks(results, target, group_dir, pids) -> List[str]` — WARN strings: (a) group refit BIC vs stored `bics/best_bic_on_test_run0.json` `individual_BIC` (index = pid − 14) where |Δ| > 5; (b) hybrid refit vs `baseline_bic` column where |Δ| > 5.
  - `summarize(results_val, results_test, warnings, out_dir)` — writes `test_results.json`, `test_results.csv` (columns: set, participant, oci, model, n_params, nll, bic, seed), `RESULTS.md` with: mean-BIC table per set, ΔBIC (composed − group) per test pid, win/tie/loss (|Δ| ≤ 1 tie), `scipy.stats.wilcoxon` p-values (composed vs group, composed vs hybrid, on test set), cross-check warnings, winner module list from `winner.json`.

- [ ] **Step 1: Write the failing test** (2 pids, stub sources → fast)

```python
# tests/test_compose_evaluate.py
import json
from pathlib import Path

from library_learning.compose.evaluate import evaluate_models, summarize
from library_learning.config import resolve_target

IND = "results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual"
GRP = "results/two_step_psychiatry_group_function_ocibalanced_maxsetting"

TINY_MODEL = '''def cognitive_model(action_1, state, action_2, reward, model_parameters):
    """
    Bias-only stub.
    Parameters:
    p_bias: [0.001, 0.999] - choice bias
    beta_dummy: [0, 10] - unused scale kept for two-param shape
    """
    p_bias, beta_dummy = model_parameters
    log_loss = 0.0
    eps = 1e-10
    for trial in range(len(action_1)):
        if int(action_1[trial]) != -1:
            p = p_bias if int(action_1[trial]) == 1 else 1.0 - p_bias
            log_loss -= np.log(p + eps)
    return log_loss
'''


def test_evaluate_and_summarize(tmp_path):
    target = resolve_target(IND)
    (tmp_path / "composed_model.txt").write_text(TINY_MODEL)
    (tmp_path / "winner.json").write_text(json.dumps(
        {"candidate_id": "stub", "module_ids": [], "n_params": 2, "mean_bic": 0.0}))
    pids = [14, 15]
    res = evaluate_models(target, GRP, tmp_path, pids, set_name="test")
    assert set(res.keys()) == {"composed", "group", "hybrid", "individual"}
    for model, fits in res.items():
        assert set(fits.keys()) == set(pids)
        assert all(f["bic"] > 0 for f in fits.values())

    summarize(results_val=None, results_test=res, warnings=["w1"], out_dir=tmp_path)
    md = (tmp_path / "RESULTS.md").read_text()
    assert "wilcoxon" in md.lower() and "w1" in md
    csv = (tmp_path / "test_results.csv").read_text()
    assert "composed" in csv and "individual" in csv


def test_freeze_discipline(tmp_path):
    target = resolve_target(IND)
    import pytest
    with pytest.raises(FileNotFoundError):
        evaluate_models(target, GRP, tmp_path, [14], set_name="test")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_evaluate.py -v`
Expected: FAIL — no module `evaluate`.

- [ ] **Step 3: Implement**

```python
# library_learning/compose/evaluate.py
"""Stage 3: fit frozen winner + baselines on held-out participants under one
seeded protocol; stats + RESULTS.md."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .fitting import fit_model_on_pids
from .hybrid import HYBRID_BOUNDS, HYBRID_SOURCE
from ..loading import (extract_unpack_names, function_name_and_args,
                       load_original_code, parse_bounds, strip_fences)

TIE_TOL = 1.0
CROSS_CHECK_TOL = 5.0


def _bounds_for(code):
    names = extract_unpack_names(code)
    b = parse_bounds(code, names)
    return [b[n] for n in names]


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
        composed_src, target, pids, _bounds_for(composed_src),
        tag="eval:%s:composed" % set_name)
    results["group"] = fit_model_on_pids(
        group_src, target, pids, _bounds_for(group_src),
        tag="eval:%s:group" % set_name, func_name=group_func_name)
    results["hybrid"] = fit_model_on_pids(
        HYBRID_SOURCE, target, pids, HYBRID_BOUNDS,
        tag="eval:%s:hybrid" % set_name)
    individual = {}
    for pid in pids:
        code = load_original_code(target, pid)
        fname, _ = function_name_and_args(code)
        individual.update(fit_model_on_pids(
            code, target, [pid], _bounds_for(code),
            tag="eval:%s:individual" % set_name, func_name=fname))
    results["individual"] = individual
    return results


def cross_checks(results, target, group_dir, pids):
    warnings = []
    stored_path = Path(group_dir) / "bics" / "best_bic_on_test_run0.json"
    if stored_path.exists():
        stored = json.loads(stored_path.read_text())["individual_BIC"]
        for pid in pids:
            idx = pid - 14
            if 0 <= idx < len(stored):
                diff = results["group"][pid]["bic"] - stored[idx]
                if abs(diff) > CROSS_CHECK_TOL:
                    warnings.append(
                        "group refit BIC differs from stored for p%d: %.2f vs %.2f"
                        % (pid, results["group"][pid]["bic"], stored[idx]))
    df = pd.read_csv(target.data_path)
    baseline = df.groupby(target.id_column)["baseline_bic"].first()
    for pid in pids:
        diff = results["hybrid"][pid]["bic"] - float(baseline[pid])
        if abs(diff) > CROSS_CHECK_TOL:
            warnings.append(
                "hybrid refit BIC differs from baseline_bic for p%d: %.2f vs %.2f"
                % (pid, results["hybrid"][pid]["bic"], float(baseline[pid])))
    return warnings


def _rows(results, set_name, oci):
    rows = []
    for model, fits in results.items():
        for pid, f in fits.items():
            rows.append({"set": set_name, "participant": pid,
                         "oci": float(oci[pid]), "model": model,
                         "n_params": len(f["params"]), "nll": f["nll"],
                         "bic": f["bic"], "seed": f["seed"]})
    return rows


def summarize(results_val, results_test, warnings, out_dir, target=None):
    out_dir = Path(out_dir)
    if target is not None:
        df = pd.read_csv(target.data_path)
        oci = df.groupby(target.id_column)["oci"].first()
    else:
        all_pids = {p for r in [results_val, results_test] if r
                    for fits in r.values() for p in fits}
        oci = {p: float("nan") for p in all_pids}

    rows = []
    if results_val:
        rows += _rows(results_val, "validation", oci)
    rows += _rows(results_test, "test", oci)
    pd.DataFrame(rows).to_csv(out_dir / "test_results.csv", index=False)

    test_pids = sorted(next(iter(results_test.values())).keys())
    means = {m: float(np.mean([results_test[m][p]["bic"] for p in test_pids]))
             for m in results_test}
    comp = np.array([results_test["composed"][p]["bic"] for p in test_pids])
    grp = np.array([results_test["group"][p]["bic"] for p in test_pids])
    hyb = np.array([results_test["hybrid"][p]["bic"] for p in test_pids])
    delta = comp - grp
    wins = int((delta < -TIE_TOL).sum())
    ties = int((np.abs(delta) <= TIE_TOL).sum())
    losses = int((delta > TIE_TOL).sum())
    stats = {
        "mean_bic": means,
        "composed_vs_group": {"wilcoxon_p": float(wilcoxon(comp, grp).pvalue),
                              "wins": wins, "ties": ties, "losses": losses,
                              "mean_delta": float(delta.mean())},
        "composed_vs_hybrid": {"wilcoxon_p": float(wilcoxon(comp, hyb).pvalue),
                               "mean_delta": float((comp - hyb).mean())},
        "warnings": warnings,
    }
    (out_dir / "test_results.json").write_text(json.dumps(
        {"stats": stats, "rows": rows}, indent=2))

    winner = json.loads((out_dir / "winner.json").read_text())
    lines = ["# Library composition results", "",
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
              "composed vs hybrid: mean dBIC %.2f, wilcoxon p=%.4f"
              % (stats["composed_vs_hybrid"]["mean_delta"],
                 stats["composed_vs_hybrid"]["wilcoxon_p"]), ""]
    if warnings:
        lines += ["## Cross-check warnings", ""] + ["- " + w for w in warnings]
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `gecco-env/bin/python -m pytest tests/test_compose_evaluate.py -v`
Expected: 2 PASS (takes a couple of minutes — real fits on 2 participants × 4 models).

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/evaluate.py tests/test_compose_evaluate.py
git commit -m "feat(compose): held-out evaluation, cross-checks, stats, RESULTS.md"
```

---

### Task 10: Figure + CLI

**Files:**
- Create: `library_learning/compose/figure.py`, `library_learning/__main__.py`
- Test: `tests/test_compose_cli.py`

**Interfaces:**
- `figure.py`: `plot_comparison(test_results_csv, out_dir)` — reads `test_results.csv` (test rows), bar of mean BIC per model + jittered per-participant dots; **paper palette: teal `#1b9e91` = hybrid reference, blue `#2b6cb8` = composed winner, gray `#9a9a9a` = group & individual**; `matplotlib.use("Agg")`; saves `comparison.png` (dpi=300) and `comparison.pdf`. Before implementing, load the `dataviz` skill for chart hygiene; keep the memory palette.
- `__main__.py`: argparse with subcommands (all take `--results-dir`, default the psychiatry individual dir; `--group-dir`, default the psychiatry group dir; `--config-dir` optional):
  - `compose-modules` — `run_extraction(...)`; `--skip-llm` re-validates existing `module_inventory.json` without API calls.
  - `compose-count` — load inventory, print `count_report` as JSON. **This is the user-checkpoint command.**
  - `compose-search --mode {exhaustive,greedy}` — `--mode` REQUIRED (the checkpoint decision); runs `score_candidates(enumerate_candidates(...))` or `greedy_search`; `select_winner` + `freeze_winner`; prints winner.
  - `compose-eval [--sets validation,test]` — `evaluate_models` per set + `cross_checks` (test set) + `summarize` + `plot_comparison`.
  - Exit codes: nonzero on any exception; `compose-search` exits nonzero if any candidate fails smoke_check.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose_cli.py
import subprocess
import sys

import pandas as pd


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "library_learning"] + list(args),
        capture_output=True, text=True)


def test_help_lists_subcommands():
    r = run_cli("--help")
    assert r.returncode == 0
    for sub in ["compose-modules", "compose-count", "compose-search", "compose-eval"]:
        assert sub in r.stdout


def test_search_requires_mode():
    r = run_cli("compose-search")
    assert r.returncode != 0
    assert "--mode" in r.stderr


def test_figure(tmp_path):
    from library_learning.compose.figure import plot_comparison
    rows = []
    for pid, boost in [(14, 0.0), (15, 10.0)]:
        for model, b in [("composed", 400.0), ("group", 420.0),
                         ("hybrid", 440.0), ("individual", 380.0)]:
            rows.append({"set": "test", "participant": pid, "oci": 0.5,
                         "model": model, "n_params": 4, "nll": 100.0,
                         "bic": b + boost, "seed": 1})
    csv = tmp_path / "test_results.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    plot_comparison(csv, tmp_path)
    assert (tmp_path / "comparison.png").exists()
    assert (tmp_path / "comparison.pdf").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_compose_cli.py -v`
Expected: FAIL — `__main__` missing / no `figure`.

- [ ] **Step 3: Implement figure.py**

```python
# library_learning/compose/figure.py
"""Comparison figure, paper style (teal reference / blue winner / gray others)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"composed": "#2b6cb8", "hybrid": "#1b9e91",
          "group": "#9a9a9a", "individual": "#c4c4c4"}
ORDER = ["hybrid", "group", "composed", "individual"]
LABELS = {"hybrid": "Hybrid (Daw)", "group": "Group GeCCo",
          "composed": "Composed (library)", "individual": "Individual GeCCo\n(ceiling)"}


def plot_comparison(test_results_csv, out_dir, rng_seed=7):
    out_dir = Path(out_dir)
    df = pd.read_csv(test_results_csv)
    df = df[df["set"] == "test"]
    rng = np.random.default_rng(rng_seed)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for i, model in enumerate(ORDER):
        vals = df[df.model == model]["bic"].to_numpy()
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

- [ ] **Step 4: Implement __main__.py**

```python
# library_learning/__main__.py
"""Repo-level CLI. Current subcommands cover the composition pipeline; the
compression pipeline (scan/verify/report) gets wired here when resumed."""
import argparse
import json
import sys
from pathlib import Path

from .config import resolve_target
from .compose import search as search_mod
from .compose.evaluate import cross_checks, evaluate_models, summarize
from .compose.extract import run_extraction, validate_inventory_obj
from .compose.figure import plot_comparison
from .compose.inventory import load_inventory
from .compose.splits import load_splits, make_splits

DEFAULT_IND = "results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual"
DEFAULT_GRP = "results/two_step_psychiatry_group_function_ocibalanced_maxsetting"
OUT_DIRNAME = "library_composition"


def add_common(p):
    p.add_argument("--results-dir", default=DEFAULT_IND)
    p.add_argument("--group-dir", default=DEFAULT_GRP)
    p.add_argument("--config-dir", default=None)


def out_dir_for(target):
    return target.results_dir / OUT_DIRNAME


def cmd_modules(args):
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    if args.skip_llm:
        obj = json.loads((out / "module_inventory.json").read_text())
        inv, errors = validate_inventory_obj(obj)
        if errors:
            print("\n".join(errors))
            return 1
        print("inventory valid: %d modules" % len(inv.modules))
        return 0
    inv = run_extraction(target, args.group_dir, out)
    print("extracted %d modules -> %s" % (len(inv.modules), out))
    return 0


def cmd_count(args):
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    inv = load_inventory(out_dir_for(target) / "module_inventory.json")
    print(json.dumps(search_mod.count_report(inv), indent=2))
    return 0


def cmd_search(args):
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    inv = load_inventory(out / "module_inventory.json")
    val_pids = load_splits(target)["validation_pids"]
    if args.mode == "exhaustive":
        cands = search_mod.enumerate_candidates(inv)
        results = search_mod.score_candidates(inv, target, val_pids, cands, out)
    else:
        results = search_mod.greedy_search(inv, target, val_pids, out)
    winner = search_mod.select_winner(results)
    search_mod.freeze_winner(winner, inv, out)
    print("WINNER %s mean validation BIC %.2f -> composed_model.txt frozen"
          % (winner["candidate_id"], winner["mean_bic"]))
    return 0


def cmd_eval(args):
    target = resolve_target(args.results_dir, config_dir=args.config_dir)
    out = out_dir_for(target)
    splits = load_splits(target)
    sets = args.sets.split(",")
    results_val = None
    if "validation" in sets:
        results_val = evaluate_models(target, args.group_dir, out,
                                      splits["validation_pids"], "validation")
    results_test = evaluate_models(target, args.group_dir, out,
                                   splits["test_pids"], "test")
    warnings = cross_checks(results_test, target, args.group_dir,
                            splits["test_pids"])
    summarize(results_val, results_test, warnings, out, target=target)
    plot_comparison(out / "test_results.csv", out)
    print("results -> %s (warnings: %d)" % (out / "RESULTS.md", len(warnings)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="library_learning")
    subs = parser.add_subparsers(dest="cmd", required=True)

    p = subs.add_parser("compose-modules", help="Gemini module extraction (logged)")
    add_common(p)
    p.add_argument("--skip-llm", action="store_true")
    p.set_defaults(fn=cmd_modules)

    p = subs.add_parser("compose-count", help="candidate count (user checkpoint)")
    add_common(p)
    p.set_defaults(fn=cmd_count)

    p = subs.add_parser("compose-search", help="fit candidates on validation, freeze winner")
    add_common(p)
    p.add_argument("--mode", choices=["exhaustive", "greedy"], required=True)
    p.set_defaults(fn=cmd_search)

    p = subs.add_parser("compose-eval", help="final-test evaluation vs baselines")
    add_common(p)
    p.add_argument("--sets", default="validation,test")
    p.set_defaults(fn=cmd_eval)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `gecco-env/bin/python -m pytest tests/test_compose_cli.py -v` then the full suite `gecco-env/bin/python -m pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add library_learning/compose/figure.py library_learning/__main__.py tests/test_compose_cli.py
git commit -m "feat(compose): comparison figure + CLI subcommands"
```

---

### Task 11: Live run — extraction, count, **USER CHECKPOINT**

**Files:** no code changes; produces artifacts under `results/.../library_composition/`.

- [ ] **Step 1: Run extraction (live Gemini, logged)**

```bash
gecco-env/bin/python -m library_learning compose-modules
```
Expected: `library_composition/` now contains `splits.json`, `mining_report.md`, `llm_log/` (13+ call files: 12 annotate + 1 merge [+ repairs]), `module_inventory.json`, `MODULES.md`. Sanity: `splits.json` seed pids `[1,2,4..13]`, 10 validation + 21 test pids, OCI means within ~0.1.

- [ ] **Step 2: Review the inventory** — read `MODULES.md` + `module_inventory.json`; spot-check 2–3 modules against their source participants' code (provenance pids). If a module misrepresents its source mechanism, re-run Step 1 (new logged calls) or hand-fix `module_inventory.json` + record in `llm_log/MANUAL_EDITS.md`, then `compose-modules --skip-llm` to re-validate.

- [ ] **Step 3: Count candidates**

```bash
gecco-env/bin/python -m library_learning compose-count
```

- [ ] **Step 4: 🛑 STOP — report to Akshay.** Message must include: number of modules, `n_candidates`, histogram by param count, and a fit-time estimate (`n_candidates × 10 validation pids × ~1–2 s`). **Akshay chooses `--mode exhaustive` or `--mode greedy`. Do not proceed without his reply.**

---

### Task 12: Live run — search, evaluation, results

- [ ] **Step 1: Run the search in Akshay's chosen mode** (background; it can take a while)

```bash
gecco-env/bin/python -m library_learning compose-search --mode <exhaustive|greedy> 2>&1 | tee /tmp/compose_search.log
```
Expected: `search_log.jsonl` grows one line per candidate; ends with `WINNER ... frozen`. Sanity-check the winner: its validation mean BIC must be ≤ backbone's.

- [ ] **Step 2: Run evaluation**

```bash
gecco-env/bin/python -m library_learning compose-eval
```
Expected: `test_results.{json,csv}`, `RESULTS.md`, `comparison.{png,pdf}`. Read every cross-check warning; if group-refit deviations exceed 5 BIC on several pids, investigate (n_starts too low? bounds parse mismatch?) before reporting numbers.

- [ ] **Step 3: Verify end-to-end coherence**
  - `winner.json` module ids ⊆ inventory ids; `composed_model.txt` parses and its `extract_unpack_names` match `winner.json` n_params.
  - `RESULTS.md` test means: individual ≤ composed (ceiling sane); report whether composed < group and composed < hybrid.
  - Full test suite still green: `gecco-env/bin/python -m pytest tests/ -v`.

- [ ] **Step 4: Report + commit artifacts**

Summarize for Akshay: winner modules, validation mean BIC, test mean BICs (all 4 models), W/T/L + Wilcoxon p, warnings. Then:

```bash
git add results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual/library_composition
git commit -m "results: library composition — winner, search log, held-out evaluation"
```

---

## Self-Review Notes

- Spec coverage: splits/no-double-dipping (Task 2), Gemini-logged extraction with singletons+dedup (Task 6), 8-param cap + count checkpoint (Tasks 7, 11), seeded fitting protocol (Task 5), freeze discipline (Task 9 `FileNotFoundError` + Task 12 order), baselines incl. ceiling + cross-checks (Tasks 8–9), figure w/ paper palette (Task 10), reproducibility (seeds everywhere, llm_log, MANUAL_EDITS.md).
- Deliberately out of scope (per spec non-goals): per-participant model assignment, compression-pipeline `scan/verify/report` CLI wiring (the `__main__.py` docstring leaves the door open).
- Spec deviations (intentional): `search_log.jsonl` instead of `search_log.json` (append-per-candidate is crash-safe); no `candidates/` dir — every candidate's source is reproducible from `render_candidate(inventory, module_ids)` and its module ids are in the log; winner's validation params live in `winner.json` (`per_pid`) instead of a separate `winner_params_validation.csv`.
