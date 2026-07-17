# Cognitive Library Learning Pipeline

## Overview

Learn a library of cognitive primitives from generated participant-level models. The pipeline:
1. Discovers common mechanisms across models
2. Expresses each model as a composition of primitives
3. Reconstructs models from the library
4. Verifies reconstruction matches original behavior
5. Iteratively refines until 100% match

---

## Task Context

All models implement a **two-step decision task**:
- **Stage 1**: Binary choice (action_1 ∈ {0,1}) → probabilistic transition to state
- **Stage 2**: Binary choice in state (action_2 ∈ {0,1}) → reward (0 or 1)
- **Transition matrix**: T = [[0.7, 0.3], [0.3, 0.7]]
- **STAI**: Anxiety score (0-1) modulates model parameters

All models inherit from **CognitiveModelBase** with:
- `__init__(n_trials, stai, model_parameters)`
- `unpack_parameters(model_parameters)` - assigns named attributes
- `init_model()` - initializes arrays
- `policy_stage1()` → action probabilities
- `policy_stage2(state)` → action probabilities  
- `value_update(action_1, state, action_2, reward)`
- `post_trial(action_1, state, action_2, reward)`
- `run_model(action_1, state, action_2, reward)` → NLL

---

## Step 0: Configuration (`step0_config.py`)

Shared configuration:
- Directory paths: MODELS_DIR, BICS_DIR, DATA_PATH, OUTPUT_DIR
- LLM setup: load_gemini_client(), call_gemini(), extract_code_block()
- Data loaders: load_best_model_scripts(), load_best_model_bics()

---

## Step 1: Primitive Discovery (`step1_discover_primitives.py`)

**Goal**: Extract reusable cognitive mechanisms from participant models → `primitives.py`

**Prompt**:
```
Analyze these cognitive models from a two-step decision task and extract reusable primitives.

## Task Structure
- Stage 1: Binary choice → transition to state
- Stage 2: Binary choice in state → reward (0/1)
- STAI: Anxiety score (0-1) modulates behavior

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
3. **Policy**: Action selection components
4. **Value update**: Learning rules
5. **Decay/memory**: Forgetting mechanisms

Output ONLY Python code wrapped in ```python ... ```.
```

---

## Step 2: Generate Specifications (`step2_generate_specs.py`)

**Goal**: Express each model as primitives + parameters → `participants.py`

**Prompt** (per participant):
```
Extract a specification for this model in terms of library primitives.

## Model Code ({pid})
{model_code}

## Available Primitives
{primitives_code}

## Instructions
Analyze the model code and identify:

1. **Primitives used**: Which library functions match the model's behavior
2. **Parameters**: Extract from `unpack_parameters` method (e.g., self.alpha, self.beta = model_parameters)
3. **STAI modulation type**: How does STAI affect parameters?
   - multiplicative: `param * self.stai`
   - additive: `base + slope * self.stai`
   - inverse_linear: `param * (1 - self.stai)`
   - inverse_division: `param / (1 + self.stai)`
4. **Custom bounds**: Extract from docstring (e.g., "alpha: 0-1", "beta: 0-10")

Output JSON (no markdown):
{
    "class": "ClassName",
    "primitives": ["category::name", ...],
    "parameters": ["param1", "param2", ...],
    "stai_modulation": "type",
    "bounds": {"param_name": [min, max]},
    "bic": stored_bic_value
}
```

**Output Format** (`participants.py`):
```python
PARTICIPANT_SPECS = {
    "p##": {
        "class": "...",
        "primitives": [...],
        "parameters": [...],
        "stai_modulation": "...",
        "bounds": {...},  # only if non-default
        "bic": ###.##,
    },
    ...
}
```

---

## Step 3: Generate Reconstructor (`step3_generate_reconstructor.py`)

**Goal**: Assemble executable models from primitives + specs → `reconstructor.py`

**Prompt**:
```
Generate a reconstructor that assembles cognitive models from primitives and specifications.

## Primitives
{primitives_code}

## Specifications  
{participants_code}

## Sample Original Models (CRITICAL - match this interface exactly)
{sample_models}

## CognitiveModelBase Interface (MUST match original models exactly)

class CognitiveModelBase(ABC):
    def __init__(self, n_trials: int, stai: float, model_parameters: tuple):
        # n_trials, stai, model_parameters - NOT subject_data
        self.n_trials = n_trials
        self.stai = stai
        self.T = np.array([[0.7, 0.3], [0.3, 0.7]])  # transition matrix
        self.p_choice_1 = np.zeros(n_trials)
        self.p_choice_2 = np.zeros(n_trials)
        self.q_stage1 = np.zeros(2)
        self.q_stage2 = np.zeros((2, 2))
        self.last_action1 = None
        self.last_reward = None
        self.unpack_parameters(model_parameters)
        self.init_model()

    def run_model(self, action_1, state, action_2, reward) -> float:
        # Takes data arrays as ARGUMENTS, not from self
        for trial in range(self.n_trials):
            a1, s, a2, r = action_1[trial], state[trial], action_2[trial], reward[trial]
            self.p_choice_1[trial] = self.policy_stage1()[a1]
            self.p_choice_2[trial] = self.policy_stage2(s)[a2]
            self.value_update(a1, s, a2, r)
            self.post_trial(a1, s, a2, r)
        return self.compute_nll()

## make_cognitive_model Function (exact signature)

def make_cognitive_model(ModelClass):
    def cognitive_model(action_1, state, action_2, reward, stai, model_parameters):
        n_trials = len(action_1)
        stai_val = float(stai[0]) if hasattr(stai, '__len__') else float(stai)
        model = ModelClass(n_trials, stai_val, model_parameters)
        return model.run_model(action_1, state, action_2, reward)
    return cognitive_model

## STAI Modulation Routing
- multiplicative → param * stai
- additive → base + slope * stai
- inverse_linear → param * (1 - stai)
- inverse_division → param / (1 + stai)

## Instructions
Create reconstructor.py with:
1. CognitiveModelBase with exact interface above
2. reconstruct_model(participant_id) → model class
3. make_cognitive_model(ModelClass) → callable
4. reconstruct_model_func(pid) → make_cognitive_model(reconstruct_model(pid))

Output ONLY Python code.
```

---

## Step 4: Verification (`step4_verify.py`)

**Goal**: Verify reconstructed models match originals exactly

**Process**:
1. **Fixed-parameter comparison** (CRITICAL first step):
   - Use stored optimal parameters from original fitting
   - Run both original and reconstructed models
   - Compare NLL values - should match within 1e-6
   - If NLL differs, model is incorrect (don't proceed to fitting)

2. **Fitting comparison** (only if fixed-param matches):
   - Fit both models independently using scipy.optimize
   - Compare resulting BICs (within 5% tolerance)

3. **Diagnostics for mismatches**:
   - Print policy outputs at each trial for first 5 trials
   - Print Q-value states after first 5 trials
   - Identify which method diverges (policy_stage1, policy_stage2, value_update)

**Verification Code Pattern**:
```python
# Fixed-parameter test
orig_model = OriginalClass(data, stored_params, stai)
lib_model = reconstruct_model(pid)(data, stored_params, stai)

orig_nll = orig_model.run_model()
lib_nll = lib_model.run_model()

if abs(orig_nll - lib_nll) < 1e-6:
    status = "MATCH"
else:
    status = "MISMATCH"
    # Collect detailed diagnostics...
```

**Output**: `verification_results.json`
```json
{
    "summary": {"matched": N, "total": M, "match_rate": "X%"},
    "participants": {
        "p##": {"status": "MATCH|MISMATCH", "orig_nll": X, "lib_nll": Y, "diagnostics": {...}}
    }
}
```

---

## Step 5: Refinement (`step5_refine.py`)

**Goal**: Diagnose and fix mismatches through iterative refinement

**Prompt** (per mismatch):
```
The reconstructed model doesn't match the original. Diagnose and fix.

## Original Model ({pid})
{original_code}

## Current Spec
{spec}

## Primitives Being Used
{primitives_code}

## Reconstructor Assembly Logic
{reconstructor_code}

## Verification Diagnostics
Fixed-param NLL: Original={orig_nll}, Library={lib_nll}
Difference: {difference}

Trial-by-trial comparison (first 5 trials):
{trial_diagnostics}

## Common Issues to Check
1. STAI modulation type mismatch
2. Missing primitive (e.g., perseveration, win-stay, decay)
3. Parameter routing error (wrong parameter used for modulation)
4. Initialization difference (q_mf presence/absence)
5. Custom bounds affecting fitting
6. Stage-specific logic (stage 1 vs stage 2 behavior)

## Instructions
1. Compare original code with reconstruction model in terms of components used and behavior
2. Identify the exact discrepancy
3. Determine if fix is in: spec, primitives, or reconstructor
4. Provide specific fix

Output JSON:
{
    "diagnosis": "Clear description of the issue",
    "fix_type": "spec|primitives|reconstructor",
    "fixed_spec": {...} or null,
    "primitives_fix": "new primitive code" or null,
    "reconstructor_fix": "specific change needed" or null
}
```

**Iteration Loop**:
```
while match_rate < 100%:
    run step4_verify.py
    for each mismatch:
        run step5 prompt
        apply fixes
    regenerate affected files
```

**Output**: `fixes.json`, updated `participants.py`, `primitives.py`, `reconstructor.py`

---

## Execution

```bash
# Step-by-step execution
python step0_config.py          # Validate config
python step1_discover_primitives.py   # → primitives.py
python step2_generate_specs.py        # → participants.py
python step3_generate_reconstructor.py # → reconstructor.py
python step4_verify.py                # → verification_results.json
python step5_refine.py                # → fixes.json, updated files

# Iterate until 100% match
while [ $(python step4_verify.py | grep "match_rate" | grep -v "100%") ]; do
    python step5_refine.py
done

# Or run full pipeline with auto-iteration
python run_pipeline.py --max-iterations 10
```

---

## Success Criteria

**Primary**:
- 100% match rate: All participants have NLL match within 1e-6 at fixed parameters

**Secondary**:
- BIC within 5% tolerance after independent fitting
- Reconstructed models pass all original test cases

**Verification Checkpoints**:
1. Primitives cover all unique computational patterns
2. Specs correctly capture STAI modulation type for each participant
3. Reconstructor handles all parameter routing cases
4. No hardcoded participant-specific logic in reconstructor

**Output Files**:
- `primitives.py`: Atomic functions
- `participants.py`: PARTICIPANT_SPECS dict
- `reconstructor.py`: CognitiveModelBase + reconstruct_model
- `verification_results.json`: Per-participant match status
- `verify_library_assembly.py`: Standalone verification script
