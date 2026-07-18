# library_learning/compose/extract.py
"""Stage 1: Gemini-driven module extraction. Every call is logged by
GeminiClient; validation and rendering stay deterministic Python."""
import ast
import json
import re
import textwrap
from pathlib import Path

from .gemini import DEFAULT_MODEL, GeminiClient
from .inventory import InventoryError, parse_inventory
from .render import render_candidate, smoke_check
from ..loading import load_original_code
from ..mining import mine_fragments, render_mining_report

PROMPT_ANNOTATE = '''You are a renowned cognitive scientist analyzing computational models of the two-step decision task (Daw et al.). Below is a Python cognitive model that was fit to participant {pid}'s behavior.

```python
{code}
```

List every distinct psychological mechanism in this model. A mechanism is one separable computational assumption (e.g. "MB/MF mixture weight", "choice stickiness", "reward-dependent stickiness", "separate stage-2 learning rate", "value decay/forgetting", "eligibility trace", "optimistic Q initialization").

Return STRICT JSON only (no prose, no markdown fences):
{{"mechanisms": [{{"name": "short snake_case id", "title": "human name", "description": "one sentence", "params": [{{"name": "param name as in code", "bounds": [lo, hi]}}], "evidence": "the exact code lines implementing it", "expressible_in_slots": true, "reason": "only when expressible_in_slots is false: why the slot contract cannot express this mechanism faithfully"}}]}}

Rules: do NOT list backbone machinery shared by all models (softmax choice, basic TD updates, MB lookahead with the fixed 0.7/0.3 transition matrix, NLL accumulation) as mechanisms. Only list deviations from that backbone. If a mechanism cannot be faithfully expressed as code injected into a fixed backbone (slots described below in the pipeline), set "expressible_in_slots": false and explain — do NOT distort a mechanism to make it fit. Slot contract summary: appended statements at init / per-trial pre-stage-1 / after stage-1 logits / after stage-2 logits / after TD updates / end of trial; replaceable expressions for q2 init, stage-1 and stage-2 value and inverse-temperature terms, and the two TD update statements.'''

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
{{"modules": [{{"id": "snake_case", "name": "...", "description": "...", "params": [{{"name": "...", "bounds": [lo, hi]}}], "slots": {{...}}, "overrides": {{...}}, "provenance": [participant ids], "excludes": ["ids of incompatible modules"]}}], "coverage": [{{"pid": 1, "mechanism": "annotated mechanism name", "module": "module id or null", "decision": "mapped | merged_into | inexpressible | backbone"}}]}}

Rules:
1. One module per distinct mechanism — merge identical mechanisms across participants (union their provenance). Keep singletons (mechanisms found in only one participant).
2. Parameter names must be globally unique across ALL modules and must not be "learning_rate" or "beta"; suffix if needed (e.g. "alpha_2", "beta_2").
3. Code must be -1-safe: missed trials have a1/s_idx/a2 == -1; never index an array with a possibly -1 value inside your snippets (the backbone already guards likelihood and TD updates; guard your own "post_trial"/"init"-state updates like `if a1 != -1:`).
4. Use plain numpy, no imports, no helper functions.
5. Bounds: use the bounds from the source model's docstring where available; otherwise probabilities/rates/weights [0, 1]; inverse temperatures [0, 10]; additive bonuses (stickiness etc.) [0, 5].
6. List modules that implement alternative versions of the same computation (e.g. two different stage1_values formulas) in each other's "excludes".
7. State isolation: any NEW variable your module creates (in "init", "post_trial", etc.) must be prefixed with the module id (e.g. module "stick" uses "stick_last_a1"), so no two modules can collide. Never assign to backbone variables except through your declared overrides.
8. The "coverage" list must account for EVERY mechanism in the annotations — one entry per (pid, mechanism), mapping it to the module that absorbed it, or marking it "inexpressible" (annotator flagged it) or "backbone" (it was actually backbone machinery). Nothing may be silently dropped.'''

PROMPT_REPAIR = '''Your previous module inventory JSON had problems. Fix ALL of them and return the corrected STRICT JSON (same schema, no prose):

Problems:
{errors}

Previous JSON:
{previous}'''

FENCE_RE = re.compile(r"```(\w*)[ \t]*\n?(.*?)```", re.DOTALL)


def _parse_json_reply(text):
    """Parse a Gemini reply as JSON, tolerating extra fenced examples in the
    reply (e.g. an illustrative ```python block before the real ```json
    block). Tries, in order: each ```json-tagged fence (in appearance
    order), each other fence, then the whole text stripped."""
    matches = FENCE_RE.findall(text)
    json_tagged = [content for lang, content in matches if lang.lower() == "json"]
    other_fenced = [content for lang, content in matches if lang.lower() != "json"]
    last_err = None
    for candidate in json_tagged + other_fenced + [text]:
        try:
            return json.loads(candidate.strip())
        except Exception as e:
            last_err = e
    raise ValueError("Gemini reply is not valid JSON (%s):\n%s" % (last_err, text[:2000]))


def annotate_seed(client, pid, code):
    reply = client.generate(PROMPT_ANNOTATE.format(pid=pid, code=code),
                            tag="annotate_p%d" % pid)
    return _parse_json_reply(reply)


CANONICAL_NAMES = {
    "action_1", "state", "action_2", "reward", "n_trials", "trial", "a1",
    "s_idx", "a2", "r", "transition_matrix", "q_stage1_mf", "q_stage2_mf",
    "q_stage1_mb", "max_q_stage2", "stage1_values", "stage2_values",
    "logits_1", "logits_2", "delta_stage1", "delta_stage2", "log_loss",
    "eps", "np",
}


def _target_names(target):
    """Recursively collect ast.Name ids bound by an assignment-style target
    expression, following through tuple/list destructuring and starred
    targets. Subscript/Attribute targets are ignored — they mutate existing
    state (allowed), not introduce a new name."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for elt in target.elts:
            names.extend(_target_names(elt))
        return names
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


def _unprefixed_assignments(module):
    """Names assigned/bound in append-slot snippets that are neither
    canonical nor '{id}_'-prefixed (state-isolation check). Covers plain
    assignment targets as well as tuple/list destructuring, starred
    targets, for-loop targets, comprehension targets, and 'with ... as x'."""
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
    from .inventory import compatible
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


def audit_coverage(obj, annotations):
    """Every annotated mechanism (pid, name) must appear in obj['coverage']."""
    covered = set()
    errors = []
    for c in obj.get("coverage", []):
        try:
            covered.add((int(c.get("pid")), c.get("mechanism")))
        except (TypeError, ValueError):
            errors.append(
                "malformed coverage entry (pid=%r, mechanism=%r): pid must "
                "be an integer" % (c.get("pid"), c.get("mechanism")))
    for pid, ann in annotations.items():
        for mech in ann.get("mechanisms", []):
            if (int(pid), mech["name"]) not in covered:
                errors.append("coverage missing for participant %s mechanism "
                              "'%s'" % (pid, mech["name"]))
    return errors


def check_bounds_against_sources(inv, target):
    """WARN where a module param's bounds disagree with a provenance
    participant's docstring bounds for the same param name."""
    from ..loading import load_original_code, extract_unpack_names, parse_bounds
    warnings = []
    for m in inv.modules:
        for pid in m.provenance:
            try:
                code = load_original_code(target, pid)
                names = extract_unpack_names(code)
                src_bounds = parse_bounds(code, names)
            except Exception:
                continue
            for p in m.params:
                base = p.name
                for cand in (base, base.rsplit("_", 1)[0]):
                    if cand in src_bounds and tuple(src_bounds[cand]) != tuple(p.bounds):
                        warnings.append(
                            "WARN %s.%s bounds %s != participant %d docstring %s"
                            % (m.id, p.name, list(p.bounds), pid,
                               list(src_bounds[cand])))
                        break
    return warnings


def reconstruction_gate(inv, obj, target, seed_pids, out_dir, tol=15.0):
    """Fidelity gate: recompose each seed from its provenance modules, fit to
    that seed's own data, compare BIC to the stored individual-gecco BIC."""
    from .inventory import compatible
    from .fitting import fit_model_on_pids
    from .render import candidate_params
    from ..loading import load_stored_bic
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
    from .inventory import Inventory
    backbone_src = render_candidate(Inventory(modules=[]), [])
    prompt = PROMPT_MERGE.format(
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


def render_modules_md(inv, seed_pids):
    lines = ["# Cognitive module library", "",
             "Extracted by %s (temperature 0, logged in llm_log/) "
             "from the individual best programs of participants %s."
             % (DEFAULT_MODEL, seed_pids), ""]
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
