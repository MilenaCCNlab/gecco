# engine/run_fit.py
from utils.extraction import build_model_spec_from_llm_output
from core.fitting import fit_model
import numpy as np

def run_fit(df, code_text):
    """
    Compile an LLM-generated cognitive model (single def ... block),
    adapt it to df, fit it, and return fit stats.
    """
    spec = build_model_spec_from_llm_output(code_text, expected_func_name="cognitive_model")

    print(f"\n[GeCCo] Extracted model: {spec.name}")
    print(f"[GeCCo] Parameters: {spec.param_names}")
    print(f"[GeCCo] Bounds: {spec.bounds}")

    n_trials = len(df)

    fit_res = fit_model(
        nll_func=spec.func,
        param_order=spec.param_names,   # order of model parameters (alpha1, beta1, …)
        bounds=spec.bounds,
        data=df,                        # spec.func consumes df internally (adapter)
        n_trials=n_trials,
        n_starts=8,
        seed=42,
        method="L-BFGS-B",
    )

    print("\n[GeCCo] Fit result:")
    print("  success:", fit_res.success)
    print("  nll:", fit_res.nll)
    print("  aic:", fit_res.aic, "bic:", fit_res.bic)
    print("  params:", fit_res.params)

    return {
        "bics": [float(fit_res.bic)],
        "param_names": list(spec.param_names),
        "model_name": spec.name,
    }
