"""
Run Pipeline
============

Run the full cognitive library learning pipeline.
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add output dir to path
OUTPUT_DIR = "/home/aj9225/gecco-1/results/two_step_psychiatry_individual_stai_class_individual/cognitive_library_v3"
sys.path.insert(0, OUTPUT_DIR)


def run_pipeline(max_iterations: int = 10, start_step: int = 1):
    """Run the full pipeline with optional iteration."""
    print("=" * 70)
    print("COGNITIVE LIBRARY LEARNING PIPELINE v3")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max iterations: {max_iterations}")
    print(f"Starting from step: {start_step}")
    print()
    
    # Step 1: Discover primitives
    if start_step <= 1:
        print("\n" + "=" * 70)
        print("STEP 1: DISCOVER PRIMITIVES")
        print("=" * 70)
        from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v3.step1_discover_primitives import discover_primitives
        discover_primitives()
    
    # Step 2: Generate specs
    if start_step <= 2:
        print("\n" + "=" * 70)
        print("STEP 2: GENERATE SPECIFICATIONS")
        print("=" * 70)
        from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v3.step2_generate_specs import generate_specs
        generate_specs()
    
    # Step 3: Generate reconstructor
    if start_step <= 3:
        print("\n" + "=" * 70)
        print("STEP 3: GENERATE RECONSTRUCTOR")
        print("=" * 70)
        from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v3.step3_generate_reconstructor import generate_reconstructor
        generate_reconstructor()
    
    # Iteration loop: verify and refine
    for iteration in range(max_iterations):
        print("\n" + "=" * 70)
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print("=" * 70)
        
        # Step 4: Verify
        print("\n--- Step 4: Verification ---")
        from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v3.step4_verify import verify_all
        
        # Reload modules to get fresh state
        import importlib
        import results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v3.step4_verify as step4_verify
        importlib.reload(step4_verify)
        
        results = step4_verify.verify_all()
        
        if results is None:
            print("Verification failed to run.")
            break
        
        match_rate = results["summary"]["match_rate"]
        print(f"\nMatch rate: {match_rate}")
        
        if match_rate == "100.0%":
            print("\n" + "=" * 70)
            print("SUCCESS! 100% match rate achieved!")
            print("=" * 70)
            break
        
        # Step 5: Refine
        print("\n--- Step 5: Refinement ---")
        from results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v3.step5_refine import refine
        
        import results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v3.step5_refine as step5_refine
        importlib.reload(step5_refine)
        
        fixes = step5_refine.refine()
        
        if not fixes:
            print("No fixes generated. Stopping iteration.")
            break
        
        # Check if any fixes were applied
        spec_fixes = sum(1 for f in fixes.values() if f.get("fix_type") == "spec")
        prim_fixes = sum(1 for f in fixes.values() if f.get("fix_type") == "primitives")
        recon_fixes = sum(1 for f in fixes.values() if f.get("fix_type") == "reconstructor")
        
        print(f"\nFixes: {spec_fixes} spec, {prim_fixes} primitives, {recon_fixes} reconstructor")
        
        if spec_fixes == 0 and prim_fixes == 0 and recon_fixes == 0:
            print("No actionable fixes. Manual intervention may be needed.")
            break
    
    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load final results
    results_path = os.path.join(OUTPUT_DIR, "verification_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            final_results = json.load(f)
        print(f"Final match rate: {final_results['summary']['match_rate']}")
        print(f"Matched: {final_results['summary']['matched']}/{final_results['summary']['total']}")
    
    print("\nOutput files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(('.py', '.json')):
            path = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(path)
            print(f"  {f}: {size} bytes")


def main():
    parser = argparse.ArgumentParser(description="Run cognitive library learning pipeline")
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=10,
        help="Maximum refinement iterations (default: 10)"
    )
    parser.add_argument(
        "--start-step", "-s",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Start from this step (default: 1)"
    )
    
    args = parser.parse_args()
    run_pipeline(max_iterations=args.max_iterations, start_step=args.start_step)


if __name__ == "__main__":
    main()
