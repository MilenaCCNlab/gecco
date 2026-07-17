"""
Step 1: Discover Primitives
===========================

Analyze participant models and extract reusable cognitive primitives.
Outputs: primitives.py
"""

import os
import sys
from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v3.step0_config import (
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

## Instructions
Create primitives.py with self-contained functions. Each function should:
- Have clear docstrings
- Use numpy
- Return values (not modify in-place)

Categories to extract:
1. **Helper**: softmax(values, beta) → probabilities
2. **STAI modulation**: Functions that combine parameters with STAI
   - stai_multiplicative(base, stai) → param * stai
   - stai_additive(base, slope, stai) → base + slope * stai
   - stai_inverse_linear(param, stai) → param * (1 - stai)
   - stai_inverse_division(param, stai) → param / (1 + stai)
3. **Policy**: Action selection components
   - add_perseveration_bonus(q_values, last_action, bonus) → modified q_values
   - add_win_stay_bonus(q_values, last_action, last_reward, bonus) → modified q_values
   - mb_mf_mixture(q_mb, q_mf, w) → (1-w)*q_mf + w*q_mb
4. **Value update**: Learning rules
   - td_update_stage1(q, action, state, q_stage2, alpha, gamma) → updated q
   - td_update_stage2(q_stage2, state, action, reward, alpha) → updated q_stage2
5. **Decay/memory**: Forgetting mechanisms
   - apply_memory_decay(q, baseline, decay_rate) → decayed q

Look for patterns across models and extract the common mechanisms.
Group similar implementations under one primitive with clear interface.

Output ONLY Python code wrapped in ```python ... ```.
"""


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
    
    # Save to file
    output_path = os.path.join(OUTPUT_DIR, "primitives.py")
    with open(output_path, 'w') as f:
        f.write(primitives_code)
    
    print(f"Saved primitives to: {output_path}")
    print(f"Code length: {len(primitives_code)} characters")
    
    return primitives_code


if __name__ == "__main__":
    discover_primitives()
