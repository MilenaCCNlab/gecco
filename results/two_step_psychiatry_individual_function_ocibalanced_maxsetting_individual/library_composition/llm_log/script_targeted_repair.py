"""Targeted repair rounds for the 5 gate-failing seeds.

Sends Gemini (logged, continuing call numbering) the genuine gate errors PLUS
the failing seeds' original model sources and their annotations, so it can
fix the distorted module definitions. Loops gate+validate up to 3 rounds.
"""
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

REPAIR_TEMPLATE = '''You previously produced a module inventory for two-step-task cognitive models. A fidelity gate recomposed each seed participant from their provenance modules and fit the result to that participant's own data. For these participants the reconstruction fits MUCH worse than their original model, meaning your module definitions distort or omit part of their mechanism:

{failures}

For each failing participant, here is their ORIGINAL model and the mechanisms you annotated for them:

{sources}

The fixed BACKBONE that modules inject into (slot semantics unchanged from before):
```python
{backbone}
```

Fix the inventory so each failing participant's reconstruction (backbone + exactly their provenance modules) reproduces their original computation faithfully. Compare each original model line-by-line against what backbone+modules would compute and find the discrepancy (wrong slot, wrong formula, missing mechanism, wrongly merged variants). You may edit module code/params, split over-merged modules into distinct variants, add new modules, and update provenance/excludes/coverage accordingly. Do NOT change modules in ways that would break participants not listed here. Parameter names must stay globally unique (never "learning_rate"/"beta"); new state variables must be prefixed with their module id; code must be -1-safe and use only the canonical backbone variable names plus your module params.

Return the FULL corrected inventory as STRICT JSON (same schema as before: {{"modules": [...], "coverage": [...]}}).

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

    rec = json.loads((OUT / "llm_log" / "call_015_merge_repair3.json").read_text())
    obj = _parse_json_reply(rec["response"]["candidates"][0]["content"]["parts"][0]["text"])

    client = GeminiClient(log_dir=OUT / "llm_log")

    for round_no in range(1, 4):
        inv, errors, failures = gate_and_validate(obj, target, seed_pids, annotations)
        print("round %d: %d validation errors, %d gate failures" % (round_no, len(errors), len(failures)))
        for e in errors + failures:
            print("  -", e)
        if not errors and not failures:
            break

        failing_pids = []
        report = json.loads((OUT / "reconstruction_report.json").read_text()) if not errors else []
        for r in report:
            if r["delta"] is not None and r["delta"] > 15.0:
                failing_pids.append(r["pid"])

        sources = []
        for pid in failing_pids:
            code = load_original_code(target, pid)
            ann = annotations.get(str(pid), annotations.get(pid, {}))
            sources.append("--- participant %d original model ---\n```python\n%s\n```\nAnnotated mechanisms: %s\n" % (
                pid, code, json.dumps(ann, indent=1)))

        backbone_src = render_candidate(Inventory(modules=[]), [])
        prompt = REPAIR_TEMPLATE.format(
            failures="\n".join("- " + f for f in (errors + failures)),
            sources="\n".join(sources) if sources else "(validation errors only — see list above)",
            backbone=backbone_src,
            previous=json.dumps(obj, indent=1))
        reply = client.generate(prompt, tag="targeted_repair%d" % round_no)
        obj = _parse_json_reply(reply)
    else:
        sys.exit("still failing after 3 targeted repair rounds — stopping for review")

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
