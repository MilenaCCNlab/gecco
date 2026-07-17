"""
Step 2: Generate Specifications
================================

Express each model as primitives + parameters → participants.py
"""

import os
import json
from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v3.step0_config import (
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

## Available Primitives
```python
{primitives_code}
```

## Instructions
Analyze the model code and identify:

1. **Primitives used**: Which library functions match the model's behavior
   Use format "category::name" where category is: helper, stai_modulation, policy, value_update, decay
   
2. **Parameters**: Extract from `unpack_parameters` method
   (e.g., self.alpha, self.beta = model_parameters)
   
3. **STAI modulation type**: How does STAI affect parameters?
   - multiplicative: `param * self.stai`
   - additive: `base + slope * self.stai`
   - inverse_linear: `param * (1 - self.stai)`
   - inverse_division: `param / (1 + self.stai)`
   
4. **Custom bounds**: Extract from docstring (e.g., "alpha: 0-1", "beta: 0-10")

Output JSON only (no markdown, no code blocks):
{{
    "class": "ClassName",
    "primitives": ["category::name", ...],
    "parameters": ["param1", "param2", ...],
    "stai_modulation": "type",
    "bounds": {{"param_name": [min, max]}},
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
    
    # Initialize LLM client
    client = load_gemini_client()
    
    # Process each participant
    specs = {}
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
            print(f"  Class: {spec.get('class', 'unknown')}")
            print(f"  Primitives: {len(spec.get('primitives', []))}")
            print(f"  Parameters: {spec.get('parameters', [])}")
            print(f"  STAI: {spec.get('stai_modulation', 'unknown')}")
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Response: {response[:200]}...")
    
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
    
    return specs


def generate_participants_code(specs: dict) -> str:
    """Generate participants.py from specs."""
    lines = [
        '"""',
        'Participant Specifications',
        '==========================',
        '',
        'Maps each participant to their model specification.',
        '"""',
        '',
        'PARTICIPANT_SPECS = {',
    ]
    
    for pid, spec in sorted(specs.items()):
        lines.append(f'    "{pid}": {{')
        lines.append(f'        "class": "{spec.get("class", "Unknown")}",')
        lines.append(f'        "primitives": {spec.get("primitives", [])},')
        lines.append(f'        "parameters": {spec.get("parameters", [])},')
        lines.append(f'        "stai_modulation": "{spec.get("stai_modulation", "none")}",')
        if spec.get("bounds"):
            lines.append(f'        "bounds": {spec.get("bounds")},')
        lines.append(f'        "bic": {spec.get("bic", 0.0)},')
        lines.append('    },')
    
    lines.append('}')
    
    return '\n'.join(lines)


if __name__ == "__main__":
    generate_specs()
