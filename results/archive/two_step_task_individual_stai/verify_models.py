import sys
import os
import pandas as pd
import numpy as np
import importlib.util

# Add library path
sys.path.append('/home/aj9225/gecco-1/results/two_step_task_individual_stai/library_learned')

# Import reconstructed models
# Since it's a package structure (from .cognitive_library), we need to be careful.
# But we added "from .cognitive_library import *" in the file.
# If we import it as a script, the relative import will fail.
# We should modify reconstructed_models.py to use absolute import or run as module.

# Let's fix the import in reconstructed_models.py first if needed.
# The script learn_cognitive_library_v2.py wrote: "from .cognitive_library import *"
# If we run this verification script from the same dir, we can use "import cognitive_library" if we change the file.

# Let's just load the module dynamically and mock the package.

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load library first
lib_path = '/home/aj9225/gecco-1/results/two_step_task_individual_stai/library_learned/cognitive_library.py'
lib_mod = load_module(lib_path, 'cognitive_library')

# Now load models. But models file has "from .cognitive_library import *"
# We need to patch it or just read it and exec it with lib_mod in context.

models_path = '/home/aj9225/gecco-1/results/two_step_task_individual_stai/library_learned/reconstructed_models.py'
with open(models_path, 'r') as f:
    models_code = f.read()

# Replace relative import with nothing (we will inject lib)
models_code = models_code.replace("from .cognitive_library import *", "")

# Fix syntax errors from astunparse
# T[(:, 0)] -> T[:, 0]
# q2[(s, :)] -> q2[s, :]
import re
# Remove parens around slice tuples inside brackets
models_code = re.sub(r'\[\(([^\]]*?:[^\]]*?)\)\]', r'[\1]', models_code)

# Execute in a context with library functions
ctx = lib_mod.__dict__.copy()
exec(models_code, ctx)

# Now ctx contains all cognitive_modelX functions.

# Load Data
sys.path.append('/home/aj9225/gecco-1')
from gecco.prepare_data.io import load_data

# Load config to get data path
from config.schema import load_config
from pathlib import Path
project_root = Path('/home/aj9225/gecco-1')
cfg = load_config(project_root / "config" / "two_step_psychiatry.yaml")
data_df = load_data(cfg.data.path)

# Parse reconstructed_models.py to find participants
participant_ids = []
import re
p_matches = re.findall(r'# Participant (\d+)', models_code)
participant_ids = [int(p) for p in p_matches]
participant_ids = sorted(list(set(participant_ids)))

print(f"Verifying models for {len(participant_ids)} participants...")

success_count = 0
fail_count = 0

for p in participant_ids:
    # Get data for participant
    p_data = data_df[data_df['participant'] == p]
    if p_data.empty:
        print(f"Skipping participant {p}: No data found.")
        continue
        
    # Prepare inputs
    action_1 = p_data['choice_1'].values.astype(int)
    state = p_data['state'].values.astype(int)
    action_2 = p_data['choice_2'].values.astype(int)
    reward = p_data['reward'].values.astype(float)
    stai = [p_data['stai'].iloc[0]] # Assuming scalar STAI
    
    # Find the block for this participant
    start_marker = f"# Participant {p}\n"
    start_idx = models_code.find(start_marker)
    if start_idx == -1:
        print(f"Model code for participant {p} not found.")
        fail_count += 1
        continue
        
    # Find end (next participant or end of file)
    next_marker = "# Participant"
    end_idx = models_code.find(next_marker, start_idx + len(start_marker))
    if end_idx == -1:
        block = models_code[start_idx:]
    else:
        block = models_code[start_idx:end_idx]
        
    # Execute this block in a fresh context (with library)
    local_ctx = lib_mod.__dict__.copy()
    try:
        exec(block, local_ctx)
    except Exception as e:
        print(f"Error defining model for participant {p}: {e}")
        fail_count += 1
        continue
        
    # Find the function name
    # Let's look for any function in local_ctx that wasn't in lib_mod
    new_keys = set(local_ctx.keys()) - set(lib_mod.__dict__.keys())
    funcs = [k for k in new_keys if callable(local_ctx[k])]
    if funcs:
        func_name = funcs[0] # Pick the first one
    else:
        print(f"No model function found for participant {p}")
        fail_count += 1
        continue
            
    model_func = local_ctx[func_name]
    
    # Inspect function to determine number of parameters
    import inspect
    sig = inspect.signature(model_func)
    # params is usually the last argument
    # We can just pass a dummy list of length 10, usually it unpacks what it needs
    # But if it unpacks (a,b,c) = params, it needs exact length.
    # We can try to parse the code to find the unpacking line?
    # Or just try different lengths?
    # Most models have 2-10 params.
    
    # Let's try to find the unpacking line in the block
    # (alpha, beta, ...) = params
    # or params[0], params[1]...
    
    # Simple heuristic: try lengths 1 to 10
    nll = None
    for n_params in range(1, 15):
        try:
            params = [0.5] * n_params
            nll = model_func(action_1, state, action_2, reward, stai, params)
            break
        except ValueError: # unpacking error
            continue
        except IndexError: # params[i] error
            continue
        except Exception as e:
            # Real error
            # print(f"Error with {n_params} params: {e}")
            pass
            
    if nll is not None:
        if isinstance(nll, (float, np.floating)):
            success_count += 1
        else:
            print(f"Participant {p}: Failed (Returned {type(nll)} instead of float)")
            fail_count += 1
    else:
        print(f"Participant {p}: Failed to find correct parameter count")
        fail_count += 1

print(f"\nVerification Complete.")
print(f"Success: {success_count}")
print(f"Failed: {fail_count}")
