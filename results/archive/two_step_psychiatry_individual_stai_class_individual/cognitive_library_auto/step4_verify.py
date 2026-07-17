"""
Step 4: Verification
====================

Verify reconstructed models match originals exactly.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_auto.step0_config import (
    load_model_files,
    load_stored_bics,
    OUTPUT_DIR,
    DATA_PATH,
    MODELS_DIR,
)


def load_subject_data():
    """Load subject data from CSV."""
    return pd.read_csv(DATA_PATH)


def get_participant_stai(data: pd.DataFrame, pid: str) -> float:
    """Get STAI score for a participant (normalized 0-1)."""
    # Extract participant number
    pid_num = int(pid[1:])
    participant_data = data[data['participant'] == pid_num]
    if participant_data.empty:
        return 0.5  # default
    stai_raw = participant_data['stai'].iloc[0]
    # Normalize STAI to 0-1 range (typically 20-80 scale)
    stai_normalized = (stai_raw - 20) / 60
    return np.clip(stai_normalized, 0, 1)


def get_participant_data(data: pd.DataFrame, pid: str) -> pd.DataFrame:
    """Get trial data for a participant."""
    pid_num = int(pid[1:])
    return data[data['participant'] == pid_num].copy()


def import_original_model(pid: str, model_code: str, CognitiveModelBase, make_cognitive_model):
    """Dynamically import the original model class.
    
    Args:
        pid: Participant ID
        model_code: Python code for the model
        CognitiveModelBase: The base class from reconstructor.py
        make_cognitive_model: Factory function from reconstructor.py
    
    Returns:
        The model class
    """
    # Create namespace with required dependencies
    ns = {
        'np': np,
        'CognitiveModelBase': CognitiveModelBase,
        'make_cognitive_model': make_cognitive_model,
    }
    
    try:
        exec(model_code, ns)
    except Exception as e:
        raise ValueError(f"Error compiling {pid}: {e}")
    
    # Find the model class (ParticipantModel1/2/3 or similar)
    for name, obj in ns.items():
        if isinstance(obj, type) and name.startswith('Participant'):
            return obj
    
    # Fallback: find any class that subclasses CognitiveModelBase
    for name, obj in ns.items():
        if isinstance(obj, type) and issubclass(obj, CognitiveModelBase) and obj != CognitiveModelBase:
            return obj
    
    raise ValueError(f"Could not find model class for {pid}")


def verify_all():
    """Verify all participants."""
    print("=" * 60)
    print("Step 4: Verification")
    print("=" * 60)
    
    # Import reconstructor
    sys.path.insert(0, OUTPUT_DIR)
    try:
        from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_auto.reconstructor import reconstruct_model, make_cognitive_model, CognitiveModelBase
        from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_auto.participants import PARTICIPANT_SPECS
    except ImportError as e:
        print(f"ERROR: Could not import library files: {e}")
        print("Make sure steps 1-3 have been run.")
        return None
    
    # Load data
    data = load_subject_data()
    models = load_model_files()
    bics = load_stored_bics()
    
    print(f"Loaded {len(models)} models")
    print(f"Loaded {len(PARTICIPANT_SPECS)} specs")
    
    results = {
        "summary": {"matched": 0, "total": 0, "match_rate": "0%"},
        "participants": {}
    }
    
    for pid in sorted(PARTICIPANT_SPECS.keys()):
        print(f"\nVerifying {pid}...")
        results["summary"]["total"] += 1
        
        if pid not in models:
            print(f"  SKIP: No original model found")
            results["participants"][pid] = {"status": "SKIP", "reason": "No original model"}
            continue
        
        try:
            # Get participant data
            participant_data = get_participant_data(data, pid)
            stai = get_participant_stai(data, pid)
            n_trials = len(participant_data)
            
            # Extract trial data arrays
            action_1 = participant_data['choice_1'].values
            state = participant_data['state'].values
            action_2 = participant_data['choice_2'].values
            reward = participant_data['reward'].values
            
            # Get the spec
            spec = PARTICIPANT_SPECS[pid]
            n_params = len(spec.get("parameters", []))
            
            # Generate test parameters (random values in [0, 1])
            np.random.seed(42)  # Reproducible
            test_params = tuple(np.random.rand(n_params) * 0.5 + 0.25)
            
            # Import original model
            orig_class = import_original_model(pid, models[pid], CognitiveModelBase, make_cognitive_model)
            
            # Get reconstructed model
            lib_class = reconstruct_model(pid)
            
            # Run both models with correct interface: (n_trials, stai, model_parameters)
            orig_model = orig_class(n_trials, stai, test_params)
            lib_model = lib_class(n_trials, stai, test_params)
            
            # run_model takes data arrays as arguments
            orig_nll = orig_model.run_model(action_1, state, action_2, reward)
            lib_nll = lib_model.run_model(action_1, state, action_2, reward)
            
            diff = abs(orig_nll - lib_nll)
            
            if diff < 1e-6:
                status = "MATCH"
                results["summary"]["matched"] += 1
                print(f"  MATCH: NLL={orig_nll:.6f}")
            else:
                status = "MISMATCH"
                print(f"  MISMATCH: orig={orig_nll:.6f}, lib={lib_nll:.6f}, diff={diff:.6f}")
            
            results["participants"][pid] = {
                "status": status,
                "orig_nll": float(orig_nll),
                "lib_nll": float(lib_nll),
                "difference": float(diff),
                "n_params": n_params,
                "stai": float(stai),
            }
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results["participants"][pid] = {
                "status": "ERROR",
                "error": str(e),
            }
    
    # Calculate match rate
    total = results["summary"]["total"]
    matched = results["summary"]["matched"]
    results["summary"]["match_rate"] = f"{100*matched/total:.1f}%" if total > 0 else "N/A"
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {matched}/{total} matched ({results['summary']['match_rate']})")
    print("=" * 60)
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, "verification_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {output_path}")
    
    return results


if __name__ == "__main__":
    verify_all()
