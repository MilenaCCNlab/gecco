"""
Step 1: Discover Primitives
===========================

Analyze participant models and extract reusable cognitive primitives.
Outputs: primitives.py
"""

import os
import sys
from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_auto.step0_config import (
    load_gemini_client,
    call_gemini,
    extract_code_block,
    load_model_files,
    OUTPUT_DIR,
)


PRIMITIVE_DISCOVERY_PROMPT = """
Analyze these cognitive models from a two-step decision task and extract reusable primitives.

## Task Structure
- Stage 1: Binary choice → transition to state
- Stage 2: Binary choice in state → reward (0/1)
- STAI: Anxiety score (0-1) modulates behavior
- Transition matrix: T = [[0.7, 0.3], [0.3, 0.7]]

## Participant Models
{model_code}


## Function Implementation Requirements

Each primitive function MUST:
1. **Pure function**: Return new values, never modify inputs in-place
2. **Type hints**: Include numpy/Optional type hints
3. **Docstring format**:
   ```python
   def function_name(arg1: type, arg2: type) -> return_type:
       \"\"\"
       Brief description.
       
       Args:
           arg1: Description
           arg2: Description
           
       Returns:
           Description of return value
           
       Used by: List participant IDs that use this (e.g., p18, p24)
       \"\"\"

## Categories to Extract

1. **helper**: Core utilities
   - softmax(values: np.ndarray, beta: float) -> np.ndarray
   - compute_mb_values(transition_matrix: np.ndarray, q_stage2: np.ndarray) -> np.ndarray

2. **stai_modulation**: Anxiety parameter modulation
   - Each function takes raw param(s) + stai, returns effective param

3. **policy**: Action selection modifications
   - Functions that modify q_values before softmax
   - Return modified q_values (copy)

4. **value_update**: Learning rules
   - TD updates for stage1 and stage2
   - Return updated q-values (copy)

5. **decay**: Memory/forgetting mechanisms
   - Apply decay to q-values
   - Return decayed q-values (copy)

6. **parameter_unpacking**: Extract parameters from flat arrays
   - Functions to unpack parameter arrays into named params

7. **miscellaneous**: Other utility functions
   - Any additional primitives that don't fit above categories

## Additional Primitives

Scan models for patterns NOT in the core list above. Add them with:
- Unique name following naming convention
- Entry in PRIMITIVES registry
- Entry in SIGNATURES registry
- Entry in PARAM_TO_PRIMITIVE if parameter-related

Output ONLY Python code wrapped in ```python ... ```.
"""


def validate_primitives(code: str) -> dict:
    """Validate that primitives.py contains required registries and functions."""
    issues = []
    
    # Check for required registries
    required_registries = ["PRIMITIVES", "SIGNATURES", "PARAM_TO_PRIMITIVE", "STAI_PATTERNS"]
    for reg in required_registries:
        if f"{reg} = " not in code and f"{reg}=" not in code:
            issues.append(f"Missing registry: {reg}")
    
    # Check for core functions
    core_functions = [
        "def softmax(",
        "def stai_multiplicative(",
        "def stai_additive(",
        "def stai_inverse_linear(",
        "def stai_inverse_division(",
        "def add_perseveration_bonus(",
        "def td_update_stage1(",
        "def td_update_stage2(",
    ]
    for func in core_functions:
        if func not in code:
            issues.append(f"Missing function: {func.replace('def ', '').replace('(', '')}")
    
    # Check for type hints
    if "np.ndarray" not in code and "numpy.ndarray" not in code:
        issues.append("Missing numpy type hints")
    
    if "Optional[" not in code:
        issues.append("Missing Optional type hints")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


def discover_primitives():
    """Run primitive discovery using LLM."""
    print("=" * 60)
    print("Step 1: Primitive Discovery")
    print("=" * 60)
    
    # Load all model files
    models = load_model_files()
    print(f"Loaded {len(models)} participant models")
    
    # Combine models into prompt
    model_code_section = ""
    for pid, code in sorted(models.items()):
        model_code_section += f"\n### {pid}\n```python\n{code}\n```\n"
    
    # Build prompt
    prompt = PRIMITIVE_DISCOVERY_PROMPT.format(model_code=model_code_section)
    
    print(f"Prompt length: {len(prompt)} characters")
    print("Calling Gemini API...")
    
    # Call LLM
    client = load_gemini_client()
    response = call_gemini(client, prompt)
    
    # Extract code
    primitives_code = extract_code_block(response, "python")
    
    # Validate output
    validation = validate_primitives(primitives_code)
    if not validation["valid"]:
        print("\n⚠️  Validation warnings:")
        for issue in validation["issues"]:
            print(f"   - {issue}")
        print("\nNote: These may cause issues in step2/step3. Consider re-running.")
    else:
        print("\n✓ Primitives validated successfully")
    
    # Save to file
    output_path = os.path.join(OUTPUT_DIR, "primitives.py")
    with open(output_path, 'w') as f:
        f.write(primitives_code)
    
    print(f"Saved primitives to: {output_path}")
    print(f"Code length: {len(primitives_code)} characters")
    
    # Save validation results alongside
    validation_path = os.path.join(OUTPUT_DIR, "primitives_validation.json")
    import json
    with open(validation_path, 'w') as f:
        json.dump(validation, f, indent=2)
    
    return primitives_code


if __name__ == "__main__":
    discover_primitives()
