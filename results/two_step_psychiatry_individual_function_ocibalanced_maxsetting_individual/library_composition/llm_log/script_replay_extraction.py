"""Replay run_extraction's tail from the logged merge_repair3 reply.

The live run failed only because of the (now fixed) int/str provenance bug;
this replays validation + coverage audit + reconstruction gate against the
verbatim call_015 inventory and, if clean, writes the same artifacts
run_extraction would have written. No new LLM calls.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/akshay/projects/gecco")

from library_learning.config import resolve_target
from library_learning.compose.extract import (
    _parse_json_reply, audit_coverage, check_bounds_against_sources,
    render_modules_md, reconstruction_gate, validate_inventory_obj)

IND = "results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual"
OUT = Path("/Users/akshay/projects/gecco") / IND / "library_composition"

target = resolve_target("/Users/akshay/projects/gecco/" + IND)

rec = json.loads((OUT / "llm_log" / "call_015_merge_repair3.json").read_text())
obj = _parse_json_reply(rec["response"]["candidates"][0]["content"]["parts"][0]["text"])

annotations = json.loads((OUT / "annotations.json").read_text())

inv, errors = validate_inventory_obj(obj)
errors += audit_coverage(obj, annotations)
print("validation+coverage errors:", len(errors))
for e in errors:
    print(" -", e)

if inv is None:
    sys.exit("inventory invalid; cannot proceed")

gate_failures = reconstruction_gate(inv, obj, target, json.loads((OUT / "splits.json").read_text())["seed_pids"], OUT)
print("reconstruction gate failures:", len(gate_failures))
for f in gate_failures:
    print(" -", f)

report = json.loads((OUT / "reconstruction_report.json").read_text())
for r in report:
    print("seed %2d: modules=%-70s recon %.2f stored %.2f delta %+.2f" % (
        r["pid"], ",".join(r["modules"]) or "(backbone)", r["recon_bic"],
        r["stored_bic"], r["delta"]))

if errors or gate_failures:
    sys.exit("NOT writing inventory artifacts; gate/validation failed")

(OUT / "module_inventory.json").write_text(json.dumps(obj, indent=2))
seed_pids = json.loads((OUT / "splits.json").read_text())["seed_pids"]
(OUT / "MODULES.md").write_text(render_modules_md(inv, seed_pids))
for w in check_bounds_against_sources(inv, target):
    print(w)
print("WROTE module_inventory.json + MODULES.md; %d modules" % len(inv.modules))
