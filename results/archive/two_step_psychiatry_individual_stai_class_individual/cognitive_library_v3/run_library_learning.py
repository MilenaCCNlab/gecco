import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = {
    "discover": "step1_discover_primitives.py",
    "specs": "step2_generate_specs.py",
    "reconstruct": "step3_generate_reconstructor.py",
    "verify": "step4_verify.py",
    "refine": "step5_refine.py",
    "fit": "step6_fit_and_compare.py",
}


# ============================================================
# UTILITIES
# ============================================================

def log(msg: str, level: str = "INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def run_script(script_name: str, args: list = None) -> subprocess.CompletedProcess:
    """Run a Python script and return the result."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    log(f"Running: {script_name}")
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=SCRIPT_DIR)
    return result


def load_verification_results() -> Optional[Dict[str, Any]]:
    """Load verification results from JSON."""
    results_path = os.path.join(SCRIPT_DIR, "verification_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return None


def get_match_rate(results: Dict[str, Any]) -> tuple:
    """Extract match rate from verification results."""
    if results is None:
        return 0, 0, 0.0
    
    summary = results.get("summary", {})
    matched = summary.get("matched", 0)
    total = summary.get("total", 0)
    rate = matched / total * 100 if total > 0 else 0.0
    return matched, total, rate


def get_mismatches(results: Dict[str, Any]) -> list:
    """Get list of mismatched participant IDs."""
    if results is None:
        return []
    
    mismatches = []
    for pid, data in results.get("participants", {}).items():
        if data.get("status") != "MATCH":
            mismatches.append(pid)
    return mismatches


# ============================================================
# PIPELINE STAGES
# ============================================================

def stage_discovery():
    """Stage 1: Discover primitives from models."""
    print("\n" + "=" * 70)
    print("STAGE 1: PRIMITIVE DISCOVERY")
    print("=" * 70)
    result = run_script(SCRIPTS["discover"])
    if result.returncode != 0:
        log("Primitive discovery failed!", "ERROR")
        return False
    return True


def stage_specs():
    """Stage 2: Generate participant specs."""
    print("\n" + "=" * 70)
    print("STAGE 2: SPEC GENERATION")
    print("=" * 70)
    result = run_script(SCRIPTS["specs"])
    if result.returncode != 0:
        log("Spec generation failed!", "ERROR")
        return False
    return True


def stage_reconstruct():
    """Stage 3: Generate reconstructor."""
    print("\n" + "=" * 70)
    print("STAGE 3: RECONSTRUCTOR GENERATION")
    print("=" * 70)
    result = run_script(SCRIPTS["reconstruct"])
    if result.returncode != 0:
        log("Reconstructor generation failed!", "ERROR")
        return False
    return True


def stage_verify() -> tuple:
    """Stage 4: Verify models match originals.
    
    Returns: (success, matched, total, rate)
    """
    print("\n" + "=" * 70)
    print("STAGE 4: VERIFICATION")
    print("=" * 70)
    result = run_script(SCRIPTS["verify"])
    if result.returncode != 0:
        log("Verification failed!", "ERROR")
        return False, 0, 0, 0.0
    
    results = load_verification_results()
    matched, total, rate = get_match_rate(results)
    return True, matched, total, rate


def stage_refine() -> bool:
    """Stage 5: Refine library based on mismatches.
    
    Returns: True if refinement was applied
    """
    print("\n" + "=" * 70)
    print("STAGE 5: REFINEMENT")
    print("=" * 70)
    result = run_script(SCRIPTS["refine"])
    if result.returncode != 0:
        log("Refinement failed!", "ERROR")
        return False
    return True


def stage_fit(n_starts: int = 10) -> tuple:
    """Stage 6: Fit and compare on actual data.
    
    Returns: (success, matched, total, rate)
    """
    print("\n" + "=" * 70)
    print("STAGE 6: FIT AND COMPARE")
    print("=" * 70)
    result = run_script(SCRIPTS["fit"], [f"--n-starts={n_starts}"])
    if result.returncode != 0:
        log("Fit comparison failed!", "ERROR")
        return False, 0, 0, 0.0
    
    # Load fit results
    fit_path = os.path.join(SCRIPT_DIR, "fit_comparison_results.json")
    if os.path.exists(fit_path):
        with open(fit_path) as f:
            results = json.load(f)
        summary = results.get("summary", {})
        matched = summary.get("matched", 0)
        total = summary.get("total", 0)
        rate = matched / total * 100 if total > 0 else 0.0
        return True, matched, total, rate
    
    return True, 0, 0, 0.0


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    max_iterations: int = 5,
    skip_discovery: bool = False,
    skip_fit: bool = False,
    fit_only: bool = False,
    fit_starts: int = 10,
) -> Dict[str, Any]:
    """
    Run the complete library learning pipeline.
    
    Args:
        max_iterations: Maximum refinement iterations
        skip_discovery: Skip primitive discovery (use existing)
        skip_fit: Skip final fit comparison
        fit_only: Only run verification (no regeneration)
        fit_starts: Number of optimization starts for fitting
    
    Returns:
        Summary of pipeline results
    """
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("COGNITIVE LIBRARY LEARNING PIPELINE")
    print("=" * 70)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max refinement iterations: {max_iterations}")
    print(f"Skip discovery: {skip_discovery}")
    print(f"Skip fit: {skip_fit}")
    print(f"Fit only: {fit_only}")
    
    pipeline_log = {
        "start_time": start_time.isoformat(),
        "config": {
            "max_iterations": max_iterations,
            "skip_discovery": skip_discovery,
            "skip_fit": skip_fit,
            "fit_only": fit_only,
            "fit_starts": fit_starts,
        },
        "stages": [],
        "iterations": [],
        "final_result": None,
    }
    
    # ----- Fit Only Mode -----
    if fit_only:
        log("Running verification only (using existing reconstructor)")
        success, matched, total, rate = stage_verify()
        if success:
            pipeline_log["stages"].append({
                "stage": "verify", 
                "status": "success",
                "matched": matched,
                "total": total,
                "rate": rate
            })
            log(f"Verification: {matched}/{total} ({rate:.1f}%)")
        else:
            pipeline_log["stages"].append({"stage": "verify", "status": "failed"})
        
        # Run fit only if 100% NLL match achieved and not skipped
        if not skip_fit and matched == total:
            log("100% NLL match confirmed - proceeding to fit comparison")
            success, fit_matched, fit_total, fit_rate = stage_fit(fit_starts)
            if success:
                pipeline_log["stages"].append({
                    "stage": "fit",
                    "status": "success",
                    "matched": fit_matched,
                    "total": fit_total,
                    "rate": fit_rate
                })
        elif not skip_fit:
            log("Skipping fit comparison - NLL match not 100%", "WARN")
            pipeline_log["stages"].append({"stage": "fit", "status": "skipped_nll_mismatch"})
        
        # Save and return
        end_time = datetime.now()
        pipeline_log["end_time"] = end_time.isoformat()
        pipeline_log["duration_seconds"] = (end_time - start_time).total_seconds()
        pipeline_log["final_result"] = {"matched": matched, "total": total, "rate": rate}
        
        log_path = os.path.join(SCRIPT_DIR, "pipeline_log.json")
        with open(log_path, 'w') as f:
            json.dump(pipeline_log, f, indent=2)
        return pipeline_log
    
    # ----- Stage 1: Discovery -----
    if not skip_discovery:
        if not stage_discovery():
            pipeline_log["final_result"] = "FAILED at discovery"
            return pipeline_log
        pipeline_log["stages"].append({"stage": "discovery", "status": "success"})
    else:
        log("Skipping primitive discovery (using existing)")
        pipeline_log["stages"].append({"stage": "discovery", "status": "skipped"})
    
    # ----- Stage 2: Specs -----
    if not skip_discovery:
        if not stage_specs():
            pipeline_log["final_result"] = "FAILED at specs"
            return pipeline_log
        pipeline_log["stages"].append({"stage": "specs", "status": "success"})
    else:
        log("Skipping spec generation (using existing)")
        pipeline_log["stages"].append({"stage": "specs", "status": "skipped"})
    
    # ----- Refinement Loop -----
    for iteration in range(max_iterations + 1):
        log(f"\n{'='*30} ITERATION {iteration} {'='*30}")
        
        iter_log = {"iteration": iteration}
        
        # Stage 3: Generate reconstructor
        if not stage_reconstruct():
            pipeline_log["final_result"] = f"FAILED at reconstruct (iter {iteration})"
            return pipeline_log
        
        # Stage 4: Verify
        success, matched, total, rate = stage_verify()
        if not success:
            pipeline_log["final_result"] = f"FAILED at verify (iter {iteration})"
            return pipeline_log
        
        iter_log["matched"] = matched
        iter_log["total"] = total
        iter_log["rate"] = rate
        
        log(f"Verification: {matched}/{total} ({rate:.1f}%)")
        
        # Check for convergence
        if matched == total:
            log(f"✅ 100% match achieved at iteration {iteration}!", "SUCCESS")
            iter_log["status"] = "converged"
            pipeline_log["iterations"].append(iter_log)
            break
        
        # Check if we've hit max iterations
        if iteration >= max_iterations:
            log(f"⚠️ Max iterations ({max_iterations}) reached with {rate:.1f}% match", "WARN")
            iter_log["status"] = "max_iterations"
            pipeline_log["iterations"].append(iter_log)
            break
        
        # Stage 5: Refine
        if not stage_refine():
            log("Refinement produced no fixes, stopping", "WARN")
            iter_log["status"] = "no_fixes"
            pipeline_log["iterations"].append(iter_log)
            break
        
        iter_log["status"] = "refined"
        pipeline_log["iterations"].append(iter_log)
    
    # ----- Stage 6: Final Fit -----
    # Only run fit comparison if 100% NLL match achieved
    final_results = load_verification_results()
    final_matched, final_total, final_rate = get_match_rate(final_results)
    
    if not skip_fit and final_matched == final_total:
        log("100% NLL match confirmed - proceeding to fit comparison")
        success, fit_matched, fit_total, fit_rate = stage_fit(fit_starts)
        if success:
            pipeline_log["stages"].append({
                "stage": "fit",
                "status": "success",
                "matched": fit_matched,
                "total": fit_total,
                "rate": fit_rate
            })
            log(f"Fit comparison: {fit_matched}/{fit_total} ({fit_rate:.1f}%)")
        else:
            pipeline_log["stages"].append({"stage": "fit", "status": "failed"})
    elif not skip_fit:
        log(f"Skipping fit comparison - NLL match is {final_rate:.1f}%, not 100%", "WARN")
        pipeline_log["stages"].append({"stage": "fit", "status": "skipped_nll_mismatch"})
    else:
        log("Skipping fit comparison")
        pipeline_log["stages"].append({"stage": "fit", "status": "skipped"})
    
    # ----- Summary -----
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    pipeline_log["end_time"] = end_time.isoformat()
    pipeline_log["duration_seconds"] = duration
    pipeline_log["final_result"] = {
        "matched": final_matched,
        "total": final_total,
        "rate": final_rate,
        "iterations_used": len(pipeline_log["iterations"]),
    }
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Duration: {duration:.1f} seconds")
    print(f"Iterations: {len(pipeline_log['iterations'])}")
    print(f"Final match rate: {final_matched}/{final_total} ({final_rate:.1f}%)")
    
    # Save pipeline log
    log_path = os.path.join(SCRIPT_DIR, "pipeline_log.json")
    with open(log_path, 'w') as f:
        json.dump(pipeline_log, f, indent=2)
    log(f"Pipeline log saved to: {log_path}")
    
    return pipeline_log


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the complete cognitive library learning pipeline"
    )
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=5,
        help="Maximum refinement iterations (default: 5)"
    )
    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Skip primitive discovery (use existing primitives.py)"
    )
    parser.add_argument(
        "--skip-fit",
        action="store_true",
        help="Skip final fit comparison stage"
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Only run fit comparison (no regeneration)"
    )
    parser.add_argument(
        "--fit-starts",
        type=int,
        default=10,
        help="Number of optimization restarts for fit comparison (default: 10)"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        max_iterations=args.max_iterations,
        skip_discovery=args.skip_discovery,
        skip_fit=args.skip_fit,
        fit_only=args.fit_only,
        fit_starts=args.fit_starts,
    )


if __name__ == "__main__":
    main()
