# engine/run_fit.py

import numpy as np
from scipy.optimize import minimize
from utils.extraction import build_model_spec_from_llm_output
from config.schema import load_config
from core.evaluation import aic as _aic, bic as _bic

directory = "/Users/milena.rmus/Desktop/gecco/"
cfg = load_config(f"{directory}config/two_step.yaml")
data_cfg = cfg.data

# Map evaluation metric name → function
metric_map = {
    "AIC": _aic,
    "BIC": _bic,
}
metric_func = metric_map.get(cfg.evaluation.metric.upper(), _bic)

def run_fit(df, code_text, expected_func_name="cognitive_model"):
    """
    Compile an LLM-generated cognitive model, fit it to participant data,
    and return fit statistics (AIC/BIC) across participants.
    This version automatically adapts to arbitrary input columns.
    """
    spec = build_model_spec_from_llm_output(code_text, expected_func_name=expected_func_name)

    # Execute LLM-generated model function
    exec(code_text, globals())
    model_func = globals()[spec.name]

    participants = df[data_cfg.id_column].unique()
    parameter_bounds = list(spec.bounds.values())
    eval_metrics = []

    n_starts = getattr(cfg.evaluation, "n_starts", 8)

    for p in participants:
        df_p = df[df[data_cfg.id_column] == p].reset_index(drop=True)
        input_cols = list(data_cfg.input_columns)
        inputs = [df_p[c].to_numpy() for c in input_cols]

        min_ll = np.inf
        for _ in range(n_starts):
            x0 = [np.random.uniform(lo, hi) for lo, hi in parameter_bounds]
            # Model function called with variable number of arguments
            res = minimize(
                lambda x: float(model_func(*inputs, x)),
                x0, method="L-BFGS-B", bounds=parameter_bounds
            )
            if res.fun < min_ll:
                min_ll = res.fun

        eval_metrics.append(metric_func(min_ll, len(parameter_bounds), len(df_p)))

    mean_metric = float(np.mean(eval_metrics))
    print(f"[GeCCo] Mean {cfg.evaluation.metric} = {mean_metric:.2f}")

    return {
        "metric_name": cfg.evaluation.metric.upper(),
        "metric_value": mean_metric,
        "param_names": list(spec.param_names),
        "model_name": spec.name,
    }
