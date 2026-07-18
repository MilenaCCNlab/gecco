"""Final targeted repair for seeds 11 and 13 with explicit diagnoses."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/akshay/projects/gecco")

from library_learning.config import resolve_target
from library_learning.loading import load_original_code
from library_learning.compose.gemini import GeminiClient
from library_learning.compose.extract import (
    _parse_json_reply, audit_coverage, check_bounds_against_sources,
    render_modules_md, reconstruction_gate, validate_inventory_obj)
from library_learning.compose.inventory import Inventory
from library_learning.compose.render import render_candidate

IND = "results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual"
OUT = Path("/Users/akshay/projects/gecco") / IND / "library_composition"

DIAGNOSIS = '''DIAGNOSES (from a line-by-line comparison of the originals against backbone+modules):

Participant 11 (current recon delta +44.6):
1. Their reward-dependent stickiness operates at BOTH stages: stage 1 keyed on (last stage-1 action, last reward), and stage 2 keyed PER SECOND-STAGE STATE on (last action-2 in that state, last reward received in that state) via two length-2 arrays. Your reward_dependent_stickiness module must add the stage-2 per-state bonus into logits_2 (slot "stage2_logits_extra") with its own prefixed state arrays updated in "post_trial", sharing the same stick_win/stick_loss parameters.
2. Their stage-1 update folds an eligibility trace into ONE line: q_stage1[a1] += lr * (delta_1 + lambda * delta_2), with both deltas computed from PRE-update values. In the backbone, delta_stage1 is computed before the stage-2 update, so standard stage1_update plus an "update_extra" line q_stage1_mf[a1] += learning_rate * <lambda param> * delta_stage2 is exactly equivalent. Check the eligibility_trace module does this and nothing else.
3. They use pure model-free stage-1 values (q_net from q_stage1 only, no MB lookahead) and separate beta_1/beta_2 — modules pure_model_free and separate_stage_betas should cover this; verify their overrides.

Participant 13 (current recon delta +35.7):
1. Outcome-dependent inverse temperature applies to BOTH stage 1 AND stage 2 within the same trial (current_beta = beta_win if prev_reward==1 else beta_loss). The module must override BOTH "stage1_temp" AND "stage2_temp" with the same expression, maintaining its prefixed prev-reward state in "post_trial" (init to 0, i.e. loss).
2. Their perseveration parameter is SIGNED with bounds [-3, 3] — the choice_stickiness module's [0, 5] bounds cannot express it. Add a distinct signed-perseveration module with bounds [-3, 3] (mutually exclusive with choice_stickiness) and set participant 13's provenance to it.
3. Their update ORDER differs from the backbone: stage 2 updates FIRST, then the stage-1 TD target uses the UPDATED q_stage2 value. Express this within slots as: override "stage1_update" to "pass", keep the standard stage-2 update, and in "update_extra" (which runs after the stage-2 update) recompute the delta and update: q_stage1_mf[a1] += learning_rate * (q_stage2_mf[s_idx, a2] - q_stage1_mf[a1]).
4. Both Q tables initialize at 0.5 (q_init_05 must set q_stage2_mf via the "q2_init" override AND q_stage1_mf via an "init" statement — assignment to the canonical name q_stage1_mf in init is allowed).
'''

REPAIR_TEMPLATE = '''You previously produced a module inventory for two-step-task cognitive models. A fidelity gate recomposes each seed participant from their provenance modules and fits the result to that participant's own data. Two participants still reconstruct much worse than their originals:

{failures}

{diagnosis}

Original models for reference:

{sources}

The fixed BACKBONE that modules inject into (slot semantics unchanged):
```python
{backbone}
```

Apply the diagnosed fixes. You may edit module code/params, add modules, split variants, and update provenance/excludes/coverage. Do NOT change the computation any other passing participant's provenance modules produce. Parameter names globally unique (never "learning_rate"/"beta"); new state variables prefixed with their module id; -1-safe; canonical backbone variable names only.

Return the FULL corrected inventory as STRICT JSON (same schema: {{"modules": [...], "coverage": [...]}}).

Previous inventory JSON:
{previous}'''


def gate_and_validate(obj, target, seed_pids, annotations):
    inv, errors = validate_inventory_obj(obj)
    if inv is not None:
        errors = errors + audit_coverage(obj, annotations)
    if inv is None or errors:
        return inv, errors, []
    failures = reconstruction_gate(inv, obj, target, seed_pids, OUT)
    return inv, [], failures


def main():
    target = resolve_target("/Users/akshay/projects/gecco/" + IND)
    splits = json.loads((OUT / "splits.json").read_text())
    seed_pids = splits["seed_pids"]
    annotations = json.loads((OUT / "annotations.json").read_text())

    rec = json.loads((OUT / "llm_log" / "call_018_targeted_repair3.json").read_text())
    obj = _parse_json_reply(rec["response"]["candidates"][0]["content"]["parts"][0]["text"])

    client = GeminiClient(log_dir=OUT / "llm_log")

    for round_no in range(1, 3):
        inv, errors, failures = gate_and_validate(obj, target, seed_pids, annotations)
        print("round %d: %d validation errors, %d gate failures" % (round_no, len(errors), len(failures)))
        for e in errors + failures:
            print("  -", e)
        if not errors and not failures:
            break
        sources = []
        for pid in (11, 13):
            code = load_original_code(target, pid)
            sources.append("--- participant %d ---\n```python\n%s\n```\n" % (pid, code))
        backbone_src = render_candidate(Inventory(modules=[]), [])
        prompt = REPAIR_TEMPLATE.format(
            failures="\n".join("- " + f for f in (errors + failures)),
            diagnosis=DIAGNOSIS,
            sources="\n".join(sources),
            backbone=backbone_src,
            previous=json.dumps(obj, indent=1))
        reply = client.generate(prompt, tag="diagnosed_repair%d" % round_no)
        obj = _parse_json_reply(reply)
    else:
        inv, errors, failures = gate_and_validate(obj, target, seed_pids, annotations)
        if errors or failures:
            print("FINAL STATE after diagnosed repairs:")
            for e in errors + failures:
                print("  -", e)
            sys.exit("still failing — stopping for manual fix")

    inv, errors, failures = gate_and_validate(obj, target, seed_pids, annotations)
    assert not errors and not failures
    (OUT / "module_inventory.json").write_text(json.dumps(obj, indent=2))
    (OUT / "MODULES.md").write_text(render_modules_md(inv, seed_pids))
    for w in check_bounds_against_sources(inv, target):
        print(w)
    report = json.loads((OUT / "reconstruction_report.json").read_text())
    for r in report:
        print("seed %2d: recon %.2f stored %.2f delta %+.2f" % (r["pid"], r["recon_bic"], r["stored_bic"], r["delta"]))
    print("SUCCESS: wrote module_inventory.json + MODULES.md; %d modules" % len(inv.modules))


if __name__ == "__main__":
    main()
