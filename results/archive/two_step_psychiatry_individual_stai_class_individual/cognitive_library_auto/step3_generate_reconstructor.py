"""
Step 3: Generate Reconstructor
===============================

Create reconstructor.py that assembles models from primitives + specs.
"""

import os
import json
from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_auto.step0_config import (
    load_gemini_client,
    call_gemini,
    extract_code_block,
    load_model_files,
    OUTPUT_DIR,
)


RECONSTRUCTOR_PROMPT = """
Generate a reconstructor that assembles cognitive models from primitives and specifications.

## Primitives (with registries)
```python
{primitives_code}
```

## Key Registries from Primitives
The primitives.py contains these registries you MUST use:
- `PRIMITIVES`: Maps category -> list of function names (e.g., "helper": ["softmax"])
- `SIGNATURES`: Maps "category::name" -> (required_args, optional_args, return_type)
- `PARAM_TO_PRIMITIVE`: Maps parameter names -> primitive category
- `STAI_PATTERNS`: Maps modulation type -> code pattern

## Specifications (structured format)
```python
{participants_code}
```

## Specification Format
Each participant spec contains:
- `primitives`: List of "category::name" strings for all primitives used
- `parameters`: List of parameter names in order
- `parameter_order`: Explicit tuple unpacking order
- `stai_modulation`: Dict mapping param_name -> modulation_type
- `primitive_calls`: Dict mapping method -> list of primitives called:
  - "policy_stage1": primitives for stage1 action selection
  - "policy_stage2": primitives for stage2 action selection
  - "value_update": primitives for learning
  - "post_trial": primitives for decay/cleanup

## Sample Original Models (CRITICAL - match this interface exactly)
{sample_models}

## CognitiveModelBase Interface (MUST match original models exactly)

The original models inherit from CognitiveModelBase. Your base class MUST have this exact interface:

```python
from abc import ABC, abstractmethod

class CognitiveModelBase(ABC):
    def __init__(self, n_trials: int, stai: float, model_parameters: tuple):
        # Task structure
        self.n_trials = n_trials
        self.n_choices = 2
        self.n_states = 2
        self.stai = stai
        
        # Transition matrix
        self.T = np.array([[0.7, 0.3], [0.3, 0.7]])
        
        # Choice probability sequences
        self.p_choice_1 = np.zeros(n_trials)
        self.p_choice_2 = np.zeros(n_trials)
        
        # Value representations
        self.q_stage1 = np.zeros(self.n_choices)
        self.q_stage2 = np.zeros((self.n_states, self.n_choices))
        
        # Trial tracking
        self.trial = 0
        self.last_action1 = None
        self.last_action2 = None
        self.last_state = None
        self.last_reward = None
        
        # Initialize
        self.unpack_parameters(model_parameters)
        self.init_model()

    @abstractmethod
    def unpack_parameters(self, model_parameters: tuple) -> None:
        pass

    def init_model(self) -> None:
        pass

    def policy_stage1(self) -> np.ndarray:
        return self.softmax(self.q_stage1, self.beta)

    def policy_stage2(self, state: int) -> np.ndarray:
        return self.softmax(self.q_stage2[state], self.beta)

    def value_update(self, action_1: int, state: int, action_2: int, reward: float) -> None:
        delta_2 = reward - self.q_stage2[state, action_2]
        self.q_stage2[state, action_2] += self.alpha * delta_2
        delta_1 = self.q_stage2[state, action_2] - self.q_stage1[action_1]
        self.q_stage1[action_1] += self.alpha * delta_1

    def pre_trial(self) -> None:
        pass

    def post_trial(self, action_1: int, state: int, action_2: int, reward: float) -> None:
        self.last_action1 = action_1
        self.last_action2 = action_2
        self.last_state = state
        self.last_reward = reward

    def run_model(self, action_1, state, action_2, reward) -> float:
        # NOTE: Takes data arrays as arguments, NOT from self
        for self.trial in range(self.n_trials):
            a1, s = int(action_1[self.trial]), int(state[self.trial])
            a2, r = int(action_2[self.trial]), float(reward[self.trial])
            
            self.pre_trial()
            self.p_choice_1[self.trial] = self.policy_stage1()[a1]
            self.p_choice_2[self.trial] = self.policy_stage2(s)[a2]
            self.value_update(a1, s, a2, r)
            self.post_trial(a1, s, a2, r)
        
        return self.compute_nll()
    
    def compute_nll(self) -> float:
        eps = 1e-12
        return -(np.sum(np.log(self.p_choice_1 + eps)) + np.sum(np.log(self.p_choice_2 + eps)))
    
    def softmax(self, values: np.ndarray, beta: float) -> np.ndarray:
        centered = values - np.max(values)
        exp_vals = np.exp(beta * centered)
        return exp_vals / np.sum(exp_vals)
```

## make_cognitive_model Function (MUST match this exactly)

```python
def make_cognitive_model(ModelClass):
    def cognitive_model(action_1, state, action_2, reward, stai, model_parameters):
        n_trials = len(action_1)
        stai_val = float(stai[0]) if hasattr(stai, '__len__') else float(stai)
        model = ModelClass(n_trials, stai_val, model_parameters)
        return model.run_model(action_1, state, action_2, reward)
    return cognitive_model
```

## STAI Modulation Routing (NEW STRUCTURED FORMAT)
Use spec["stai_modulation"] which is now a DICT mapping param_name -> modulation_type:
```python
# Example spec["stai_modulation"]:
{
    "phi": "inverse_division",      # effective_phi = phi / (1 + stai)
    "stick_slope": "additive",      # effective_stick = base + slope * stai
    "w": "inverse_linear"           # effective_w = w * (1 - stai)
}
```

Modulation types and their implementations:
- "multiplicative" → `primitives.stai_multiplicative(param, stai)` → param * stai
- "additive" → `primitives.stai_additive(base, slope, stai)` → base + slope * stai  
- "inverse_linear" → `primitives.stai_inverse_linear(param, stai)` → param * (1 - stai)
- "inverse_division" → `primitives.stai_inverse_division(param, stai)` → param / (1 + stai)
- "affine_amplification" → `primitives.stai_affine_amplification(base, bias, stai)` → base * (1 + bias * stai)

## Primitive Calls Routing (NEW STRUCTURED FORMAT)
Use spec["primitive_calls"] to know exactly which primitives each method should call:
```python
# Example spec["primitive_calls"]:
{
    "policy_stage1": ["helper::softmax", "policy::add_perseveration_bonus"],
    "policy_stage2": ["helper::softmax"],
    "value_update": ["value_update::td_update_stage1", "value_update::td_update_stage2"],
    "post_trial": []
}
```

## Parameter Routing (fallback if primitive_calls not available)
Use PARAM_TO_PRIMITIVE registry from primitives.py or detect from parameter names:
- 'persev' or 'phi' → policy::add_perseveration_bonus
- 'wsls' or 'win_stay' → policy::add_win_stay_bonus
- 'w' or 'w_base' or 'w_max' → policy::mb_mf_mixture
- 'decay' → decay::apply_memory_decay
- 'habit' → policy::add_habit_influence

## Instructions
Create reconstructor.py with:

1. **Import primitives**: `import primitives as P` - use the primitives module, don't redefine functions

2. **CognitiveModelBase**: Exact interface as shown above

3. **reconstruct_model(participant_id)**: Returns a dynamically created model class that:
   - Inherits from CognitiveModelBase
   - Reads spec from participants.PARTICIPANT_SPECS[participant_id]
   - Implements unpack_parameters based on spec["parameter_order"]
   - Implements init_model to compute effective parameters using spec["stai_modulation"]
   - Overrides policy_stage1/policy_stage2/value_update/post_trial based on spec["primitive_calls"]
   - Calls primitives from `import primitives as P` (e.g., `P.softmax(...)`, `P.add_perseveration_bonus(...)`)

3. **make_cognitive_model(ModelClass)**: Wrapper for fitting API (exact signature above)

4. **reconstruct_model_func(participant_id)**: Returns make_cognitive_model(reconstruct_model(pid))

Output ONLY Python code wrapped in ```python ... ```.
"""


def generate_reconstructor():
    """Generate the reconstructor module."""
    print("=" * 60)
    print("Step 3: Generate Reconstructor")
    print("=" * 60)
    
    # Load primitives
    primitives_path = os.path.join(OUTPUT_DIR, "primitives.py")
    if not os.path.exists(primitives_path):
        raise FileNotFoundError("primitives.py not found. Run step1 first.")
    
    with open(primitives_path) as f:
        primitives_code = f.read()
    print(f"Loaded primitives.py ({len(primitives_code)} chars)")
    
    # Load participants
    participants_path = os.path.join(OUTPUT_DIR, "participants.py")
    if not os.path.exists(participants_path):
        raise FileNotFoundError("participants.py not found. Run step2 first.")
    
    with open(participants_path) as f:
        participants_code = f.read()
    print(f"Loaded participants.py ({len(participants_code)} chars)")
    
    # Load sample original models (ALL of them for better coverage)
    models = load_model_files()
    sample_models = ""
    # Include all models for context
    for pid, code in sorted(models.items()):
        sample_models += f"\n### {pid}\n```python\n{code}\n```\n"
    print(f"Using all {len(models)} sample models")
    
    # Load reconstructor hints from refinement (if any)
    hints_path = os.path.join(OUTPUT_DIR, "reconstructor_hints.json")
    hints_section = ""
    if os.path.exists(hints_path):
        with open(hints_path) as f:
            hints = json.load(f)
        if hints:
            print(f"Loaded {len(hints)} reconstructor hints from refinement")
            hints_section = "\n\n## CRITICAL FIXES FROM PREVIOUS ITERATION\n"
            hints_section += "The following issues were identified and MUST be fixed in this iteration:\n\n"
            for pid, fix in hints.items():
                hints_section += f"### {pid}\n{fix}\n\n"
    
    # Load verification results to know what worked before
    verification_path = os.path.join(OUTPUT_DIR, "verification_results.json")
    verification_section = ""
    if os.path.exists(verification_path):
        with open(verification_path) as f:
            ver_results = json.load(f)
        matched = [pid for pid, r in ver_results.get("participants", {}).items() if r.get("status") == "MATCH"]
        if matched:
            verification_section = f"\n\n## PREVIOUSLY MATCHED MODELS (DO NOT BREAK THESE)\n"
            verification_section += f"These models worked correctly: {', '.join(sorted(matched))}\n"
            verification_section += "IMPORTANT: Your reconstructor MUST continue to match these models.\n"
    
    # Build prompt
    prompt = RECONSTRUCTOR_PROMPT.format(
        primitives_code=primitives_code,
        participants_code=participants_code,
        sample_models=sample_models,
    )
    
    # Append hints and verification info
    if verification_section:
        prompt += verification_section
    if hints_section:
        prompt += hints_section
    
    print(f"Prompt length: {len(prompt)} characters")
    print("Calling Gemini API...")
    
    # Call LLM
    client = load_gemini_client()
    response = call_gemini(client, prompt)
    
    # Extract code
    reconstructor_code = extract_code_block(response, "python")
    
    # Save to file
    output_path = os.path.join(OUTPUT_DIR, "reconstructor.py")
    with open(output_path, 'w') as f:
        f.write(reconstructor_code)
    
    print(f"Saved reconstructor to: {output_path}")
    print(f"Code length: {len(reconstructor_code)} characters")
    
    return reconstructor_code


if __name__ == "__main__":
    generate_reconstructor()
