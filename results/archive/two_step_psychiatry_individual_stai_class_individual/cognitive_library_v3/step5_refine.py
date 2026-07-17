"""
Step 5: Refinement
==================

Diagnose and fix mismatches through iterative refinement.
"""

import os
import json
from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v3.step0_config import (
    load_gemini_client,
    call_gemini,
    extract_code_block,
    load_model_files,
    OUTPUT_DIR,
)


REFINEMENT_PROMPT = """
The reconstructed model doesn't match the original. Diagnose and fix.

## Original Model ({pid})
```python
{original_code}
```

## Current Spec
```json
{spec_json}
```

## Primitives Being Used
```python
{primitives_code}
```

## Reconstructor Assembly Logic (relevant section)
The reconstructor uses PARTICIPANT_SPECS to assemble models dynamically.
It routes based on:
- stai_modulation field → appropriate STAI function
- parameter names → perseveration, win-stay, decay detection
- primitives list → MB/MF mixture, value update patterns

## Verification Diagnostics
Fixed-param NLL: Original={orig_nll}, Library={lib_nll}
Difference: {difference}

## Common Issues to Check
1. STAI modulation type mismatch (multiplicative vs additive vs inverse)
2. Missing primitive (e.g., perseveration, win-stay, decay)
3. Parameter routing error (wrong parameter used for modulation)
4. Initialization difference (q_mf presence/absence)
5. Custom bounds affecting fitting
6. Stage-specific logic (stage 1 vs stage 2 behavior)
7. Parameter order mismatch in unpack_parameters

## Instructions
1. Compare original code with reconstruction model in terms of components used and behavior
2. Identify the exact discrepancy
3. Determine if fix is in: spec, primitives, or reconstructor
4. Provide specific fix

Output JSON only (no markdown):
{{
    "diagnosis": "Clear description of the issue",
    "fix_type": "spec|primitives|reconstructor",
    "fixed_spec": {{...}} or null,
    "primitives_fix": "new primitive code" or null,
    "reconstructor_fix": "specific change needed" or null
}}
"""


def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response."""
    import re
    
    # Try to find JSON in code block
    pattern = r"```(?:json)?\s*(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())
    
    # Try to parse entire text as JSON
    start = response.find('{')
    end = response.rfind('}')
    if start != -1 and end != -1:
        return json.loads(response[start:end+1])
    
    raise ValueError(f"Could not extract JSON from: {response[:200]}...")


def refine():
    """Run refinement on mismatched participants."""
    print("=" * 60)
    print("Step 5: Refinement")
    print("=" * 60)
    
    # Load verification results
    results_path = os.path.join(OUTPUT_DIR, "verification_results.json")
    if not os.path.exists(results_path):
        print("No verification_results.json found. Run step4 first.")
        return None
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Find mismatches
    mismatches = []
    for pid, pdata in results["participants"].items():
        if pdata.get("status") in ("MISMATCH", "ERROR"):
            mismatches.append(pid)
    
    if not mismatches:
        print("No mismatches to fix! All models match.")
        return {}
    
    print(f"Found {len(mismatches)} mismatches: {mismatches}")
    
    # Load dependencies
    primitives_path = os.path.join(OUTPUT_DIR, "primitives.py")
    with open(primitives_path) as f:
        primitives_code = f.read()
    
    specs_path = os.path.join(OUTPUT_DIR, "specs.json")
    with open(specs_path) as f:
        specs = json.load(f)
    
    models = load_model_files()
    
    # Initialize LLM
    client = load_gemini_client()
    
    fixes = {}
    for pid in mismatches:
        print(f"\nAnalyzing {pid}...")
        
        if pid not in models:
            print(f"  SKIP: No original model found")
            continue
        
        pdata = results["participants"].get(pid, {})
        spec = specs.get(pid, {})
        
        prompt = REFINEMENT_PROMPT.format(
            pid=pid,
            original_code=models[pid],
            spec_json=json.dumps(spec, indent=2),
            primitives_code=primitives_code,
            orig_nll=pdata.get("orig_nll", "N/A"),
            lib_nll=pdata.get("lib_nll", "N/A"),
            difference=pdata.get("difference", "N/A"),
        )
        
        response = call_gemini(client, prompt)
        
        try:
            fix = extract_json_from_response(response)
            fixes[pid] = fix
            print(f"  Diagnosis: {fix.get('diagnosis', 'unknown')}")
            print(f"  Fix type: {fix.get('fix_type', 'unknown')}")
        except Exception as e:
            print(f"  ERROR parsing response: {e}")
            fixes[pid] = {"error": str(e), "raw_response": response[:500]}
    
    # Save fixes
    output_path = os.path.join(OUTPUT_DIR, "fixes.json")
    with open(output_path, 'w') as f:
        json.dump(fixes, f, indent=2)
    print(f"\nSaved fixes to: {output_path}")
    
    # Apply spec fixes and primitives fixes
    spec_fixes_applied = 0
    primitives_fixes = []
    reconstructor_fixes = []
    
    for pid, fix in fixes.items():
        # Collect spec fixes
        if fix.get("fixed_spec"):
            specs[pid] = fix["fixed_spec"]
            spec_fixes_applied += 1
        
        # Collect primitives fixes
        if fix.get("primitives_fix"):
            primitives_fixes.append((pid, fix["primitives_fix"]))
        
        # Collect reconstructor fixes for regeneration prompt
        if fix.get("reconstructor_fix"):
            reconstructor_fixes.append((pid, fix["reconstructor_fix"]))
    
    # Apply spec fixes
    if spec_fixes_applied > 0:
        print(f"\nApplying {spec_fixes_applied} spec fixes...")
        
        # Regenerate participants.py
        from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v3.step2_generate_specs import generate_participants_code
        participants_code = generate_participants_code(specs)
        
        participants_path = os.path.join(OUTPUT_DIR, "participants.py")
        with open(participants_path, 'w') as f:
            f.write(participants_code)
        print(f"Updated participants.py")
        
        # Update specs.json
        with open(specs_path, 'w') as f:
            json.dump(specs, f, indent=2)
        print(f"Updated specs.json")
    
    # Apply primitives fixes by appending to primitives.py
    if primitives_fixes:
        print(f"\nApplying {len(primitives_fixes)} primitives fixes...")
        
        with open(primitives_path, 'a') as f:
            f.write("\n\n# ============ Auto-generated fixes ============\n")
            for pid, pfix in primitives_fixes:
                f.write(f"\n# Fix for {pid}\n")
                f.write(pfix)
                f.write("\n")
        print(f"Appended primitives to primitives.py")
    
    # Signal if reconstructor needs regeneration
    if reconstructor_fixes:
        print(f"\n{len(reconstructor_fixes)} reconstructor fixes needed.")
        print("Run step3_generate_reconstructor.py to regenerate with fixes.")
        
        # Save reconstructor fix hints for step3
        hints_path = os.path.join(OUTPUT_DIR, "reconstructor_hints.json")
        with open(hints_path, 'w') as f:
            json.dump({pid: fix for pid, fix in reconstructor_fixes}, f, indent=2)
        print(f"Saved reconstructor hints to: {hints_path}")
    
    return fixes


if __name__ == "__main__":
    refine()
