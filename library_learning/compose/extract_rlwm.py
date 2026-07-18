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
