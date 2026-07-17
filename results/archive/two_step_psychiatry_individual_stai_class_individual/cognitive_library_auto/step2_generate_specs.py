"""
Step 2: Generate Specifications
================================

Express each model as primitives + parameters → participants.py
"""

import os
import json
from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_auto.step0_config import (
    load_gemini_client,
    call_gemini,
    extract_code_block,
    load_model_files,
    load_stored_bics,
    OUTPUT_DIR,
)


SPEC_EXTRACTION_PROMPT = """
Extract a specification for this model in terms of library primitives.

## Model Code ({pid})
```python
{model_code}
```

## Available Primitives (with registries)
```python
{primitives_code}
```

## Primitive Registry Reference
The primitives.py contains these registries you MUST use:
- PRIMITIVES: Maps category -> list of function names
- SIGNATURES: Maps "category::name" -> (required_args, optional_args, return_type)
- PARAM_TO_PRIMITIVE: Maps parameter name patterns -> primitive category
- STAI_PATTERNS: Maps modulation type -> code pattern

## Instructions
Analyze the model code and produce a specification that:

1. **Primitives used**: List ALL primitives from the registry that this model uses.
   Format: "category::name" (e.g., "helper::softmax", "policy::add_perseveration_bonus")
   ONLY use primitives that exist in the PRIMITIVES registry.

2. **Parameters**: Extract exact names from `unpack_parameters` method.
   Order matters - must match the tuple unpacking order.

3. **STAI modulation**: For EACH parameter affected by STAI, specify:
   - Which parameter(s)
   - Which modulation type from STAI_PATTERNS
   
4. **Primitive calls**: Map each relevant method to primitive calls:
   - policy_stage1 -> which policy primitives are called
   - policy_stage2 -> which policy primitives are called  
   - value_update -> which value_update primitives are called
   - post_trial -> any decay primitives

5. **Bounds**: Extract from docstring or infer standard bounds.

Output JSON only (no markdown, no code blocks):
{{
    "class": "ClassName",
    "primitives": ["category::name", ...],
    "parameters": ["param1", "param2", ...],
    "parameter_order": ["param1", "param2", ...],
    "stai_modulation": {{
        "param_name": "modulation_type",
        ...
    }},
    "primitive_calls": {{
        "policy_stage1": ["category::name", ...],
        "policy_stage2": ["category::name", ...],
        "value_update": ["category::name", ...],
        "post_trial": ["category::name", ...]
    }},
    "bounds": {{"param_name": [min, max], ...}},
    "bic": {bic}
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
    # Find first { and last }
    start = response.find('{')
    end = response.rfind('}')
    if start != -1 and end != -1:
        return json.loads(response[start:end+1])
    
    raise ValueError(f"Could not extract JSON from: {response[:200]}...")


def load_primitive_registries(primitives_code: str) -> dict:
    """Extract registries from primitives.py code."""
    import ast
    
    registries = {
        "PRIMITIVES": {},
        "SIGNATURES": {},
        "PARAM_TO_PRIMITIVE": {},
        "STAI_PATTERNS": {},
    }
    
    try:
        tree = ast.parse(primitives_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in registries:
                        # Evaluate the value safely
                        try:
                            registries[target.id] = ast.literal_eval(node.value)
                        except:
                            pass
    except:
        print("  Warning: Could not parse primitive registries")
    
    return registries


def validate_spec(spec: dict, registries: dict) -> list:
    """Validate a spec against primitive registries."""
    issues = []
    
    # Check primitives exist
    available = set()
    for cat, funcs in registries.get("PRIMITIVES", {}).items():
        for f in funcs:
            available.add(f"{cat}::{f}")
    
    for prim in spec.get("primitives", []):
        if prim not in available and "::" in prim:
            issues.append(f"Unknown primitive: {prim}")
    
    # Check required fields
    required = ["class", "primitives", "parameters", "stai_modulation", "bounds"]
    for field in required:
        if field not in spec:
            issues.append(f"Missing field: {field}")
    
    # Check stai_modulation is dict (new format)
    if "stai_modulation" in spec and not isinstance(spec["stai_modulation"], dict):
        issues.append("stai_modulation should be a dict mapping param->type")
    
    return issues


def generate_specs():
    """Generate participant specifications."""
    print("=" * 60)
    print("Step 2: Generate Specifications")
    print("=" * 60)
    
    # Load primitives
    primitives_path = os.path.join(OUTPUT_DIR, "primitives.py")
    if not os.path.exists(primitives_path):
        raise FileNotFoundError(f"primitives.py not found. Run step1 first.")
    
    with open(primitives_path) as f:
        primitives_code = f.read()
    print(f"Loaded primitives.py ({len(primitives_code)} chars)")
    
    # Load models and BICs
    models = load_model_files()
    bics = load_stored_bics()
    print(f"Loaded {len(models)} models, {len(bics)} BICs")
    
    # Load primitive registries for validation
    registries = load_primitive_registries(primitives_code)
    print(f"Loaded registries: {list(registries.keys())}")
    
    # Initialize LLM client
    client = load_gemini_client()
    
    # Process each participant
    specs = {}
    validation_issues = {}
    
    for pid, model_code in sorted(models.items()):
        print(f"\nProcessing {pid}...")
        
        bic = bics.get(pid, 0.0)
        
        prompt = SPEC_EXTRACTION_PROMPT.format(
            pid=pid,
            model_code=model_code,
            primitives_code=primitives_code,
            bic=bic,
        )
        
        response = call_gemini(client, prompt)
        
        try:
            spec = extract_json_from_response(response)
            specs[pid] = spec
            
            # Validate against registries
            issues = validate_spec(spec, registries)
            if issues:
                validation_issues[pid] = issues
                print(f"  ⚠️  Validation issues: {issues}")
            
            print(f"  Class: {spec.get('class', 'unknown')}")
            print(f"  Primitives: {len(spec.get('primitives', []))}")
            print(f"  Parameters: {spec.get('parameters', [])}")
            stai_mod = spec.get('stai_modulation', {})
            print(f"  STAI modulations: {stai_mod if isinstance(stai_mod, dict) else 'legacy format'}")
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Response: {response[:200]}...")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processed {len(specs)} specs, {len(validation_issues)} with issues")
    
    # Generate participants.py
    participants_code = generate_participants_code(specs)
    
    output_path = os.path.join(OUTPUT_DIR, "participants.py")
    with open(output_path, 'w') as f:
        f.write(participants_code)
    print(f"\nSaved participants.py: {output_path}")
    
    # Also save raw specs as JSON
    specs_path = os.path.join(OUTPUT_DIR, "specs.json")
    with open(specs_path, 'w') as f:
        json.dump(specs, f, indent=2)
    print(f"Saved specs.json: {specs_path}")
    
    # Save validation issues if any
    if validation_issues:
        issues_path = os.path.join(OUTPUT_DIR, "specs_validation_issues.json")
        with open(issues_path, 'w') as f:
            json.dump(validation_issues, f, indent=2)
        print(f"Saved validation issues: {issues_path}")
    
    return specs


def generate_participants_code(specs: dict) -> str:
    """Generate participants.py from specs."""
    lines = [
        '"""',
        'Participant Specifications',
        '==========================',
        '',
        'Maps each participant to their model specification.',
        'Used by reconstructor.py to assemble models from primitives.',
        '"""',
        '',
        'PARTICIPANT_SPECS = {',
    ]
    
    for pid, spec in sorted(specs.items()):
        lines.append(f'    "{pid}": {{')
        lines.append(f'        "class": "{spec.get("class", "Unknown")}",')
        lines.append(f'        "primitives": {spec.get("primitives", [])},')
        lines.append(f'        "parameters": {spec.get("parameters", [])},')
        lines.append(f'        "parameter_order": {spec.get("parameter_order", spec.get("parameters", []))},')
        
        # Handle stai_modulation - prefer dict format
        stai_mod = spec.get("stai_modulation", {})
        if isinstance(stai_mod, dict):
            lines.append(f'        "stai_modulation": {stai_mod},')
        else:
            # Legacy string format - convert to dict if possible
            lines.append(f'        "stai_modulation": {{"_type": "{stai_mod}"}},')
        
        # Add primitive_calls if present
        if spec.get("primitive_calls"):
            lines.append(f'        "primitive_calls": {spec.get("primitive_calls")},')
        
        if spec.get("bounds"):
            lines.append(f'        "bounds": {spec.get("bounds")},')
        lines.append(f'        "bic": {spec.get("bic", 0.0)},')
        lines.append('    },')
    
    lines.append('}')
    
    return '\n'.join(lines)


if __name__ == "__main__":
    generate_specs()
