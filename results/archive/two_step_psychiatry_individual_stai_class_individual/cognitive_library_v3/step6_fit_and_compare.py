"""
Step 6: Fit and Compare
=======================

After verifying that reconstructed models match originals on NLL output,
this script fits BOTH the original and library-assembled models to actual
participant data and compares their fitted BICs.

This is the TRUE test of the library's compression quality - ensuring
that models fit to the same optima when optimizing parameters.
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict

from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v3.step0_config import (
    load_model_files,
    load_stored_bics,
    OUTPUT_DIR,
    DATA_PATH,
    MODELS_DIR,
)

# Import library components
sys.path.insert(0, OUTPUT_DIR)
from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v3.reconstructor import (
    CognitiveModelBase,
    make_cognitive_model,
    reconstruct_model,
    PARTICIPANT_SPECS,
)


# ============================================================
# DATA LOADING
# ============================================================

def load_subject_data() -> pd.DataFrame:
    """Load subject data from CSV."""
    return pd.read_csv(DATA_PATH)


def get_participant_data(data: pd.DataFrame, pid: str) -> pd.DataFrame:
    """Get trial data for a participant."""
    pid_num = int(pid[1:])
    return data[data['participant'] == pid_num].copy()


# ============================================================
# MODEL LOADING
# ============================================================

def load_original_model_func(pid: str, model_code: str) -> Optional[Callable]:
    """Load original model as a callable function.
    
    Returns callable: (action_1, state, action_2, reward, stai, params) -> nll
    """
    ns = {
        'np': np,
        'CognitiveModelBase': CognitiveModelBase,
        'make_cognitive_model': make_cognitive_model,
    }
    
    try:
        exec(model_code, ns)
    except Exception as e:
        print(f"Error compiling {pid}: {e}")
        return None
    
    # Find the cognitive_model function
    for name in ['cognitive_model1', 'cognitive_model2', 'cognitive_model3']:
        if name in ns:
            return ns[name]
    
    # Try to find ParticipantModel class and wrap it
    for name, obj in ns.items():
        if isinstance(obj, type) and issubclass(obj, CognitiveModelBase) and obj != CognitiveModelBase:
            return make_cognitive_model(obj)
    
    return None


def build_library_model_func(pid: str) -> Callable:
    """Build library-assembled model as a callable function.
    
    Returns callable: (action_1, state, action_2, reward, stai, params) -> nll
    """
    model_class = reconstruct_model(pid)
    return make_cognitive_model(model_class)


# ============================================================
# PARAMETER BOUNDS
# ============================================================

def get_parameter_bounds(pid: str) -> List[Tuple[float, float]]:
    """Get parameter bounds from the spec."""
    spec = PARTICIPANT_SPECS[pid]
    bounds_dict = spec.get("bounds", {})
    param_names = spec["parameters"]
    
    bounds = []
    for param in param_names:
        if param in bounds_dict:
            b = bounds_dict[param]
            bounds.append((b[0], b[1]))
        elif 'beta' in param.lower() or 'temperature' in param.lower():
            bounds.append((0, 10))
        else:
            bounds.append((0, 1))
    
    return bounds


# ============================================================
# MODEL FITTING
# ============================================================

def bic(nll: float, k: int, n: int) -> float:
    """Compute Bayesian Information Criterion."""
    return math.log(n) * k + 2 * nll


def fit_model(
    model_func: Callable,
    df: pd.DataFrame,
    param_bounds: List[Tuple[float, float]],
    n_starts: int = 10
) -> Tuple[float, float, Optional[np.ndarray]]:
    """Fit a model to participant data.
    
    Args:
        model_func: Callable (action_1, state, action_2, reward, stai, params) -> nll
        df: Participant data
        param_bounds: List of (min, max) bounds for each parameter
        n_starts: Number of random restarts
    
    Returns:
        (nll, bic, best_params) tuple
    """
    action_1 = df['choice_1'].to_numpy()
    state = df['state'].to_numpy()
    action_2 = df['choice_2'].to_numpy()
    reward = df['reward'].to_numpy()
    stai = df['stai'].to_numpy()
    
    n_trials = len(df)
    n_params = len(param_bounds)
    
    min_nll = np.inf
    best_params = None
    
    for _ in range(n_starts):
        x0 = [np.random.uniform(lo, hi) for lo, hi in param_bounds]
        try:
            res = minimize(
                lambda x: float(model_func(action_1, state, action_2, reward, stai, tuple(x))),
                x0,
                method="L-BFGS-B",
                bounds=param_bounds,
            )
            if res.fun < min_nll:
                min_nll = res.fun
                best_params = res.x
        except Exception:
            continue
    
    if best_params is None:
        return np.inf, np.inf, None
    
    bic_val = bic(min_nll, n_params, n_trials)
    return min_nll, bic_val, best_params


# ============================================================
# COMPARISON RESULT
# ============================================================

@dataclass
class ComparisonResult:
    participant_id: str
    stored_bic: float
    original_nll: float
    original_bic: float
    library_nll: float
    library_bic: float
    diff_bic: float          # library_bic - original_bic
    diff_pct: float          # percent difference
    match: bool              # within tolerance
    error: Optional[str] = None


# ============================================================
# MAIN COMPARISON
# ============================================================

def fit_and_compare(
    tolerance_pct: float = 1.0,
    n_starts: int = 10,
    verbose: bool = True
) -> Dict[str, ComparisonResult]:
    """
    Fit and compare original vs library-assembled models.
    
    Args:
        tolerance_pct: Max percent difference to count as match
        n_starts: Number of optimization restarts per model
        verbose: Print progress
    
    Returns:
        Dictionary of participant_id -> ComparisonResult
    """
    print("=" * 80)
    print("Step 6: Fit and Compare (Library Assembly Verification)")
    print("=" * 80)
    
    # Load data
    df = load_subject_data()
    models = load_model_files()
    stored_bics = load_stored_bics()
    
    print(f"\n📊 Fitting {len(PARTICIPANT_SPECS)} participants")
    print(f"📁 Models: {MODELS_DIR}")
    print(f"🎯 Tolerance: {tolerance_pct}%")
    print(f"🔄 Optimization restarts: {n_starts}\n")
    
    if verbose:
        print(f"{'PID':<6} {'Stored':>10} {'Orig NLL':>10} {'Orig BIC':>10} {'Lib NLL':>10} {'Lib BIC':>10} {'Diff':>10} {'Match':<6}")
        print("-" * 80)
    
    results = {}
    n_match = 0
    n_total = 0
    
    for pid in sorted(PARTICIPANT_SPECS.keys(), key=lambda x: int(x[1:])):
        stored_bic = stored_bics.get(pid, np.nan)
        
        # Get participant data
        df_p = get_participant_data(df, pid)
        if len(df_p) == 0:
            if verbose:
                print(f"{pid:<6} SKIP: No data found")
            continue
        
        n_total += 1
        
        # Get parameter bounds
        param_bounds = get_parameter_bounds(pid)
        
        # Fit original model
        if pid in models:
            original_func = load_original_model_func(pid, models[pid])
            if original_func is not None:
                original_nll, original_bic, _ = fit_model(original_func, df_p, param_bounds, n_starts)
            else:
                original_nll, original_bic = np.nan, np.nan
        else:
            original_nll, original_bic = np.nan, np.nan
        
        # Fit library-assembled model
        try:
            library_func = build_library_model_func(pid)
            library_nll, library_bic, _ = fit_model(library_func, df_p, param_bounds, n_starts)
        except Exception as e:
            library_nll, library_bic = np.nan, np.nan
            if verbose:
                print(f"{pid:<6} ERROR: Library build failed: {e}")
            results[pid] = ComparisonResult(
                participant_id=pid,
                stored_bic=stored_bic,
                original_nll=original_nll,
                original_bic=original_bic,
                library_nll=np.nan,
                library_bic=np.nan,
                diff_bic=np.nan,
                diff_pct=np.nan,
                match=False,
                error=str(e)
            )
            continue
        
        # Compute differences
        diff_bic = library_bic - original_bic if not (np.isnan(library_bic) or np.isnan(original_bic)) else np.nan
        
        if not np.isnan(diff_bic) and original_bic > 0:
            diff_pct = abs(diff_bic) / original_bic * 100
            match = diff_pct <= tolerance_pct
        else:
            diff_pct = np.nan
            match = False
        
        if match:
            n_match += 1
        
        results[pid] = ComparisonResult(
            participant_id=pid,
            stored_bic=stored_bic,
            original_nll=original_nll,
            original_bic=original_bic,
            library_nll=library_nll,
            library_bic=library_bic,
            diff_bic=diff_bic,
            diff_pct=diff_pct,
            match=match
        )
        
        if verbose:
            match_str = "✓" if match else "✗"
            print(f"{pid:<6} {stored_bic:>10.2f} {original_nll:>10.2f} {original_bic:>10.2f} "
                  f"{library_nll:>10.2f} {library_bic:>10.2f} {diff_bic:>+10.2f} {match_str:<6}")
    
    # Summary
    match_rate = 100 * n_match / n_total if n_total > 0 else 0
    
    print("-" * 80)
    print(f"\n✅ Library fit match rate: {n_match}/{n_total} ({match_rate:.1f}%)")
    print("=" * 80)
    
    # Save results
    results_dict = {
        "summary": {
            "matched": n_match,
            "total": n_total,
            "match_rate": f"{match_rate:.1f}%",
            "tolerance_pct": tolerance_pct,
            "n_starts": n_starts
        },
        "participants": {pid: asdict(r) for pid, r in results.items()}
    }
    
    output_path = os.path.join(OUTPUT_DIR, "fit_comparison_results.json")
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=lambda x: None if pd.isna(x) else x)
    print(f"\nSaved results to: {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fit and compare models")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Match tolerance (percent)")
    parser.add_argument("--n-starts", type=int, default=10, help="Optimization restarts")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    fit_and_compare(
        tolerance_pct=args.tolerance,
        n_starts=args.n_starts,
        verbose=not args.quiet
    )
