"""
Library-Assembled Model Verification
=====================================

This verifies the cognitive library by:
1. Loading the participant specs from participants.py
2. Loading the original model code from models/ directory
3. RECONSTRUCTING models by assembling from library primitives (via reconstructor.py)
4. Fitting BOTH original and library-assembled models to human data
5. Comparing BICs to verify the library is lossless

This is the TRUE test of the library's compression quality.
"""

import sys
import os
import json
import glob
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import math

# Add paths
sys.path.insert(0, '/home/aj9225/gecco-1')
sys.path.insert(0, '/home/aj9225/gecco-1/results/two_step_psychiatry_individual_stai_class_individual/cognitive_library_v2')

# Import library components
from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v2.participants import PARTICIPANT_SPECS
import results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v2.primitives as P

# Import model reconstruction from shared reconstructor module
from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v2.reconstructor import (
    CognitiveModelBase,
    make_cognitive_model,
    reconstruct_model,
    reconstruct_model_func,
)


# ============================================================
# LIBRARY-ASSEMBLED MODEL FACTORY
# ============================================================

def build_model_from_library(participant_id: str) -> Callable:
    """
    Build a cognitive model function by ASSEMBLING from library primitives.
    
    This delegates to reconstruct_model_func from reconstructor.py to ensure
    a single source of truth for model assembly logic.
    
    Returns a callable model function (action_1, state, action_2, reward, stai, params) -> nll
    """
    return reconstruct_model_func(participant_id)


def compare_model_classes(participant_id: str, models_dir: str, output_dir: str = None) -> Tuple[str, str]:
    """
    Compare original model class vs library-assembled model class side by side.
    
    Args:
        participant_id: e.g., 'p18'
        models_dir: Directory with original model files
        output_dir: If provided, saves comparison files there
    
    Returns:
        (original_code, assembled_code) tuple
    """
    import inspect
    
    # Get original model code
    pid_num = participant_id[1:]
    model_file = os.path.join(models_dir, f"best_model_0_participant{pid_num}.txt")
    
    with open(model_file) as f:
        original_code = f.read()
    
    # Get library-assembled model class using reconstructor
    spec = PARTICIPANT_SPECS[participant_id]
    ReconstructedModelClass = reconstruct_model(participant_id)
    assembled_code = inspect.getsource(ReconstructedModelClass)
    
    # Print side by side
    print("\n" + "=" * 100)
    print(f"MODEL COMPARISON: {participant_id}")
    print("=" * 100)
    
    print("\n" + "-" * 50 + " ORIGINAL " + "-" * 40)
    print(original_code)
    
    print("\n" + "-" * 50 + " LIBRARY-ASSEMBLED " + "-" * 31)
    print(assembled_code)
    
    print("\n" + "-" * 50 + " KEY DIFFERENCES " + "-" * 33)
    print(f"Spec: primitives={spec['primitives']}")
    print(f"      stai_mod={spec['stai_modulation']}, params={spec['parameters']}")
    
    # Save to files if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{participant_id}_original.py"), 'w') as f:
            f.write(original_code)
        with open(os.path.join(output_dir, f"{participant_id}_assembled.py"), 'w') as f:
            f.write(assembled_code)
        print(f"\n📁 Saved to {output_dir}/{participant_id}_*.py")
    
    return original_code, assembled_code


# ============================================================
# LOAD ORIGINAL MODELS
# ============================================================

def load_original_model_func(models_dir: str, participant_id: str) -> Optional[Callable]:
    """Load the original pre-library model function."""
    pid_num = participant_id[1:]
    model_file = os.path.join(models_dir, f"best_model_0_participant{pid_num}.txt")
    
    if not os.path.exists(model_file):
        return None
    
    with open(model_file) as f:
        code = f.read()
    
    ns = {
        'np': np,
        'CognitiveModelBase': CognitiveModelBase,
        'make_cognitive_model': make_cognitive_model
    }

    try:
        exec(code, ns)
    except Exception as e:
        print(f"Error compiling {participant_id}: {e}")
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


def load_stored_bics(bics_dir: str) -> Dict[str, float]:
    """Load best BIC for each participant."""
    best_bics = {}
    for f in glob.glob(os.path.join(bics_dir, '*.json')):
        with open(f) as fp:
            data = json.load(fp)
        for model in data:
            code_file = model['code_file']
            match = re.search(r'participant(\d+)', code_file)
            if match:
                pid = f"p{match.group(1)}"
                bic_val = model['metric_value']
                if pid not in best_bics or bic_val < best_bics[pid]:
                    best_bics[pid] = bic_val
    return best_bics


def parse_bounds_from_docstring(doc: str) -> Dict[str, Tuple[float, float]]:
    """Extract parameter bounds from docstrings (matching utils.py logic)."""
    if not doc:
        return {}

    bounds = {}
    # Pattern to match: param_name: [lo, hi] or param_name [lo, hi] or param_name (lo, hi)
    explicit_pattern = re.compile(
        r"""
        (?:^|\n)\s*[-*]?\s*
        ([A-Za-z_][A-Za-z0-9_]*)
        [^()\[\]\n]*?
        [\(\[\{]\s*
        ([\-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*
        [,\s]+\s*
        ([\-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*
        [\)\]\}]
        """,
        flags=re.I | re.X | re.M,
    )

    for name, lo, hi in explicit_pattern.findall(doc):
        try:
            bounds[name.lower()] = (float(lo), float(hi))
            bounds[name] = (float(lo), float(hi))
        except ValueError:
            continue

    return bounds


def get_parameter_bounds_from_code(code: str, param_names: List[str]) -> List[Tuple[float, float]]:
    """Extract parameter bounds from model code docstring."""
    # Extract docstring from class
    doc_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
    if not doc_match:
        doc_match = re.search(r"'''(.*?)'''", code, re.DOTALL)
    
    docstring = doc_match.group(1) if doc_match else ""
    bounds_dict = parse_bounds_from_docstring(docstring)
    
    # Build bounds list in parameter order
    bounds = []
    default_bound = (0, 1)
    beta_bound = (0, 10)
    
    for param in param_names:
        if param in bounds_dict:
            bounds.append(bounds_dict[param])
        elif param.lower() in bounds_dict:
            bounds.append(bounds_dict[param.lower()])
        elif 'beta' in param.lower() or 'temperature' in param.lower():
            bounds.append(beta_bound)
        else:
            bounds.append(default_bound)
    
    return bounds


def get_parameter_bounds(participant_id: str, models_dir: str) -> List[Tuple[float, float]]:
    """Get parameter bounds by extracting from original model docstring."""
    spec = PARTICIPANT_SPECS[participant_id]
    param_names = spec["parameters"]
    
    # Load original model code to extract bounds from docstring
    pid_num = participant_id[1:]
    model_file = os.path.join(models_dir, f"best_model_0_participant{pid_num}.txt")
    
    if os.path.exists(model_file):
        with open(model_file) as f:
            code = f.read()
        return get_parameter_bounds_from_code(code, param_names)
    
    # Fallback to defaults if file not found
    bounds = []
    for param in param_names:
        if 'beta' in param.lower() or 'temperature' in param.lower():
            bounds.append((0, 10))
        else:
            bounds.append((0, 1))
    return bounds


def bic(nll: float, k: int, n: int) -> float:
    return math.log(n) * k + 2 * nll


# ============================================================
# FIT AND COMPARE
# ============================================================

def fit_model(
    model_func: Callable,
    df: pd.DataFrame,
    param_bounds: List[Tuple[float, float]],
    n_starts: int = 10
) -> Tuple[float, float, np.ndarray]:
    """Fit a model and return (nll, bic, params)."""
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
        except:
            continue
    
    if best_params is None:
        return np.inf, np.inf, None
    
    bic_val = bic(min_nll, n_params, n_trials)
    return min_nll, bic_val, best_params


@dataclass
class ComparisonResult:
    participant_id: str
    stored_bic: float
    original_bic: float
    library_bic: float
    diff_original_vs_stored: float
    diff_library_vs_stored: float
    diff_library_vs_original: float
    library_match: bool
    error: Optional[str] = None


def compare_models(
    data_path: str,
    models_dir: str,
    bics_dir: str,
    tolerance_pct: float = 5.0,
    n_starts: int = 10,
    verbose: bool = True
) -> Dict[str, ComparisonResult]:
    """
    Compare original models vs library-assembled models.
    
    For each participant:
    1. Load stored BIC
    2. Load & fit original model -> original_bic
    3. Build & fit library model -> library_bic
    4. Compare all three
    """
    df = pd.read_csv(data_path)
    stored_bics = load_stored_bics(bics_dir)
    
    if verbose:
        print("=" * 80)
        print("LIBRARY vs ORIGINAL MODEL COMPARISON")
        print("=" * 80)
        print(f"\n📊 Comparing {len(stored_bics)} participants")
        print(f"📁 Models: {models_dir}")
        print(f"🎯 Tolerance: {tolerance_pct}%")
        print(f"🔄 Optimization starts: {n_starts}\n")
        print(f"{'PID':<6} {'Stored':>10} {'Original':>10} {'Library':>10} {'Orig-Stor':>10} {'Lib-Stor':>10} {'Lib-Orig':>10} {'Match':<6}")
        print("-" * 80)
    
    results = {}
    n_match = 0
    
    for pid in sorted(stored_bics.keys(), key=lambda x: int(x[1:])):
        stored_bic = stored_bics[pid]
        pid_num = int(pid[1:])
        df_p = df[df['participant'] == pid_num].reset_index(drop=True)
        
        if len(df_p) == 0:
            continue
        
        param_bounds = get_parameter_bounds(pid, models_dir)
        
        # Fit original model
        original_func = load_original_model_func(models_dir, pid)

        if original_func is not None:
            _, original_bic, _ = fit_model(original_func, df_p, param_bounds, n_starts)
        else:
            original_bic = np.nan
        
        # Build and fit library-assembled model
        try:
            library_func = build_model_from_library(pid)
            _, library_bic, _ = fit_model(library_func, df_p, param_bounds, n_starts)
        except Exception as e:
            library_bic = np.nan
            if verbose:
                print(f"   ⚠️  {pid}: Library build error: {e}")
     
        # Compute differences
        diff_orig_stored = original_bic - stored_bic if not np.isnan(original_bic) else np.nan
        diff_lib_stored = library_bic - stored_bic if not np.isnan(library_bic) else np.nan
        diff_lib_orig = library_bic - original_bic if not (np.isnan(library_bic) or np.isnan(original_bic)) else np.nan
        
        # Check if library matches (within tolerance of original)
        if not np.isnan(diff_lib_orig):
            pct_diff = abs(diff_lib_orig) / original_bic * 100 if original_bic > 0 else np.inf
            match = pct_diff <= tolerance_pct
        else:
            match = False
        
        if match:
            n_match += 1
        
        results[pid] = ComparisonResult(
            participant_id=pid,
            stored_bic=stored_bic,
            original_bic=original_bic,
            library_bic=library_bic,
            diff_original_vs_stored=diff_orig_stored,
            diff_library_vs_stored=diff_lib_stored,
            diff_library_vs_original=diff_lib_orig,
            library_match=match
        )
        
        if verbose:
            match_str = "✓" if match else "✗"
            print(f"{pid:<6} {stored_bic:>10.2f} {original_bic:>10.2f} {library_bic:>10.2f} "
                  f"{diff_orig_stored:>+10.2f} {diff_lib_stored:>+10.2f} {diff_lib_orig:>+10.2f} {match_str:<6}")
    
    if verbose:
        print("-" * 80)
        print(f"\n✅ Library match rate: {n_match}/{len(results)} ({100*n_match/len(results):.1f}%)")
        print("=" * 80)
    
    return results


# ============================================================
# SYSTEMATIC DIAGNOSTIC CHECKS
# ============================================================

def extract_method_overrides(code: str) -> Dict[str, str]:
    """Extract which methods are overridden in the model class."""
    overrides = {}
    
    # Pattern for method definitions
    method_pattern = re.compile(r'def\s+(\w+)\s*\(self[^)]*\)\s*(?:->.*?)?:', re.MULTILINE)
    
    for match in method_pattern.finditer(code):
        method_name = match.group(1)
        if method_name not in ['__init__']:  # Skip constructor
            # Extract method body (rough approximation)
            start = match.start()
            # Find next method or end of class
            next_def = method_pattern.search(code, match.end())
            end = next_def.start() if next_def else len(code)
            method_body = code[match.end():end].strip()
            overrides[method_name] = method_body[:500]  # First 500 chars
    
    return overrides


def check_bounds_match(participant_id: str, models_dir: str) -> Dict:
    """Check if parameter bounds from original model match what library uses."""
    spec = PARTICIPANT_SPECS[participant_id]
    param_names = spec["parameters"]
    
    # Get bounds from original model docstring
    original_bounds = get_parameter_bounds(participant_id, models_dir)
    
    # Get what library would use as default
    default_bounds = []
    for param in param_names:
        if 'beta' in param.lower() or 'temperature' in param.lower():
            default_bounds.append((0, 10))
        else:
            default_bounds.append((0, 1))
    
    # Compare
    mismatches = []
    for i, (param, orig, default) in enumerate(zip(param_names, original_bounds, default_bounds)):
        if orig != default:
            mismatches.append({
                'param': param,
                'original_bounds': orig,
                'default_bounds': default,
                'issue': 'bounds_differ'
            })
    
    return {
        'param_names': param_names,
        'original_bounds': original_bounds,
        'default_bounds': default_bounds,
        'mismatches': mismatches,
        'bounds_ok': len(mismatches) == 0
    }


def check_structural_differences(participant_id: str, models_dir: str) -> Dict:
    """Check structural differences between original and library assembly."""
    spec = PARTICIPANT_SPECS[participant_id]
    primitives_used = set(spec["primitives"])
    
    # Load original model code
    pid_num = participant_id[1:]
    model_file = os.path.join(models_dir, f"best_model_0_participant{pid_num}.txt")
    
    with open(model_file) as f:
        original_code = f.read()
    
    original_methods = extract_method_overrides(original_code)
    
    issues = []
    
    # Check 1: post_trial override
    if 'post_trial' in original_methods:
        body = original_methods['post_trial'].lower()
        if 'decay' in body or '*=' in body:
            if 'decay::memory_decay' not in primitives_used:
                issues.append({
                    'type': 'missing_primitive',
                    'detail': 'Original has post_trial with decay, but decay::memory_decay not in primitives'
                })
            # Check for unvisited state decay
            if 'unvisited_state' in body or '1 - state' in body:
                issues.append({
                    'type': 'complex_decay',
                    'detail': 'Original decays UNVISITED state - library only decays current state'
                })
        if 'habit' in body:
            issues.append({
                'type': 'missing_habit_update',
                'detail': 'Original has habit trace update in post_trial - not in library'
            })
    
    # Check 2: init_model override
    if 'init_model' in original_methods:
        body = original_methods['init_model'].lower()
        if 'habit' in body:
            issues.append({
                'type': 'missing_init',
                'detail': 'Original initializes habit trace in init_model - not in library'
            })
        if 'alpha_pos' in body or 'alpha_neg' in body:
            issues.append({
                'type': 'asymmetric_alpha',
                'detail': 'Original computes asymmetric learning rates in init_model'
            })
    
    # Check 3: value_update override with reward-based alpha
    if 'value_update' in original_methods:
        body = original_methods['value_update'].lower()
        if 'reward' in body and ('alpha_pos' in body or 'alpha_neg' in body or 'if reward' in body):
            issues.append({
                'type': 'reward_dependent_alpha',
                'detail': 'Original uses reward-dependent learning rate (alpha_pos/alpha_neg)'
            })
    
    # Check 4: policy_stage1 with habit
    if 'policy_stage1' in original_methods:
        body = original_methods['policy_stage1'].lower()
        if 'habit' in body:
            issues.append({
                'type': 'habit_policy',
                'detail': 'Original adds habit trace to policy_stage1 - not in library'
            })
        if 'last_reward' in body and 'win' not in spec.get('primitives', []):
            issues.append({
                'type': 'win_stay_condition',
                'detail': 'Original has last_reward condition in policy - may need policy::win_stay_bonus'
            })
    
    return {
        'original_methods': list(original_methods.keys()),
        'primitives_used': list(primitives_used),
        'issues': issues,
        'structure_ok': len(issues) == 0
    }


def diagnose_participant(participant_id: str, models_dir: str, bic_diff: float = None) -> Dict:
    """Run full diagnostic on a participant."""
    
    bounds_check = check_bounds_match(participant_id, models_dir)
    struct_check = check_structural_differences(participant_id, models_dir)
    
    diagnosis = {
        'participant_id': participant_id,
        'bic_diff': bic_diff,
        'bounds': bounds_check,
        'structure': struct_check,
        'root_causes': []
    }
    
    # Determine root causes
    if not bounds_check['bounds_ok']:
        for m in bounds_check['mismatches']:
            diagnosis['root_causes'].append(f"BOUNDS: {m['param']} has {m['original_bounds']} vs default {m['default_bounds']}")
    
    for issue in struct_check['issues']:
        diagnosis['root_causes'].append(f"{issue['type'].upper()}: {issue['detail']}")
    
    return diagnosis


def run_systematic_diagnostics(
    results: Dict[str, ComparisonResult],
    models_dir: str,
    bic_threshold: float = 10.0
) -> List[Dict]:
    """Run diagnostics on all participants with BIC difference > threshold."""
    
    print("\n" + "=" * 100)
    print(f"SYSTEMATIC DIAGNOSTICS (BIC diff > {bic_threshold})")
    print("=" * 100)
    
    # Find participants with large BIC differences
    failed = []
    for pid, r in results.items():
        diff = abs(r.diff_library_vs_original) if not np.isnan(r.diff_library_vs_original) else 0
        if diff > bic_threshold:
            failed.append((pid, r.diff_library_vs_original, r))
    
    failed.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n📊 Found {len(failed)} participants with |BIC diff| > {bic_threshold}\n")
    
    diagnostics = []
    
    for pid, bic_diff, r in failed:
        print(f"\n{'─' * 100}")
        print(f"🔬 {pid}: BIC diff = {bic_diff:+.2f} (Library={r.library_bic:.2f}, Original={r.original_bic:.2f})")
        print(f"{'─' * 100}")
        
        diag = diagnose_participant(pid, models_dir, bic_diff)
        diagnostics.append(diag)
        
        # Print bounds check
        bc = diag['bounds']
        if bc['bounds_ok']:
            print(f"  ✓ Bounds: OK - {bc['param_names']}")
            print(f"           {bc['original_bounds']}")
        else:
            print(f"  ✗ Bounds: MISMATCH")
            for m in bc['mismatches']:
                print(f"    - {m['param']}: original={m['original_bounds']}, default={m['default_bounds']}")
        
        # Print structure check  
        sc = diag['structure']
        print(f"\n  Original methods: {sc['original_methods']}")
        print(f"  Library primitives: {sc['primitives_used']}")
        
        if sc['structure_ok']:
            print(f"  ✓ Structure: OK")
        else:
            print(f"  ✗ Structure: ISSUES FOUND")
            for issue in sc['issues']:
                print(f"    - [{issue['type']}] {issue['detail']}")
        
        # Summary root causes
        if diag['root_causes']:
            print(f"\n  🔴 ROOT CAUSES:")
            for cause in diag['root_causes']:
                print(f"     • {cause}")
        else:
            print(f"\n  ⚠️  No obvious root cause found - may need deeper inspection")
    
    # Summary table
    print("\n" + "=" * 100)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 100)
    print(f"\n{'PID':<6} {'BIC Δ':>10} {'Bounds':>8} {'Structure':>10} {'Root Causes':<60}")
    print("-" * 100)
    
    for diag in diagnostics:
        pid = diag['participant_id']
        bic_d = diag['bic_diff']
        bounds_ok = "✓" if diag['bounds']['bounds_ok'] else "✗"
        struct_ok = "✓" if diag['structure']['structure_ok'] else "✗"
        causes = "; ".join(diag['root_causes'])[:55] + "..." if len("; ".join(diag['root_causes'])) > 55 else "; ".join(diag['root_causes'])
        print(f"{pid:<6} {bic_d:>+10.2f} {bounds_ok:>8} {struct_ok:>10} {causes:<60}")
    
    # Categorize issues
    issue_counts = {}
    for diag in diagnostics:
        for cause in diag['root_causes']:
            issue_type = cause.split(':')[0]
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
    
    print("\n" + "-" * 100)
    print("ISSUE FREQUENCY:")
    for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {issue_type}: {count} participants")
    
    return diagnostics


def run_comparison(tolerance_pct: float = 5.0, n_starts: int = 10):
    """Run the full comparison with default paths."""
    base_dir = '/home/aj9225/gecco-1'
    results_dir = f'{base_dir}/results/two_step_psychiatry_individual_stai_class_individual'
    
    return compare_models(
        data_path=f'{base_dir}/data/two_step_gillan_2016.csv',
        models_dir=f'{results_dir}/models',
        bics_dir=f'{results_dir}/bics',
        tolerance_pct=tolerance_pct,
        n_starts=n_starts,
        verbose=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare original vs library-assembled models')
    parser.add_argument('--tolerance', type=float, default=5.0,
                        help='Acceptable BIC percentage difference (default: 5.0)')
    parser.add_argument('--n-starts', type=int, default=10,
                        help='Number of optimization starts (default: 10)')
    parser.add_argument('--compare', type=str, default=None,
                        help='Compare specific participant (e.g. p18). Shows side-by-side code comparison.')
    parser.add_argument('--diagnose', action='store_true',
                        help='Run systematic diagnostics on participants with BIC diff > 10')
    parser.add_argument('--bic-threshold', type=float, default=10.0,
                        help='BIC difference threshold for diagnostics (default: 10.0)')
    args = parser.parse_args()
    
    base_dir = '/home/aj9225/gecco-1'
    results_dir = f'{base_dir}/results/two_step_psychiatry_individual_stai_class_individual'
    models_dir = f'{results_dir}/models'
    
    # If --compare specified, just show side-by-side comparison
    if args.compare:
        orig_code, lib_code = compare_model_classes(args.compare, models_dir)
        sys.exit(0)
    
    results = run_comparison(tolerance_pct=args.tolerance, n_starts=args.n_starts)
    
    # Run systematic diagnostics if requested or by default for failures
    if args.diagnose or any(not r.library_match for r in results.values()):
        diagnostics = run_systematic_diagnostics(results, models_dir, args.bic_threshold)
    
    # Summary
    match_count = sum(1 for r in results.values() if r.library_match)
    total = len(results)
    
    # Save results
    output = []
    for pid, r in results.items():
        output.append({
            'participant_id': r.participant_id,
            'stored_bic': r.stored_bic,
            'original_bic': r.original_bic,
            'library_bic': r.library_bic,
            'diff_library_vs_original': r.diff_library_vs_original,
            'library_match': r.library_match,
        })
    
    with open('library_comparison_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📄 Results saved to library_comparison_results.json")

