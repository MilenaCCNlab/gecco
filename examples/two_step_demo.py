
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from core.fitting import fit_model
from utils.extraction import build_model_spec_from_llm_output
from llm.model_loader import load_llm
from llm.prompt_builder import build_prompt
from llm.generator import generate_model_code
from data.io import load_data
from config.schema import load_config
from data.data2text import get_data2text_function


def run_fit(df, code_text):
    # Build runtime ModelSpec from generated code
    spec = build_model_spec_from_llm_output(code_text, expected_func_name="cognitive_model")
    print(f"\n[GeCCo] Extracted model: {spec.name}")
    print(f"[GeCCo] Parameters: {spec.param_names}")
    print(f"[GeCCo] Bounds: {spec.bounds}")

    # n_trials: pick something meaningful for BIC (e.g., total rows per participant or total rows)
    n_trials = len(df)

    # Fit
    fit_res = fit_model(
        nll_func=spec.func,
        param_order=spec.param_names,
        bounds=spec.bounds,
        data=df,            # the model is expected to consume df
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

def main():
    cfg = load_config("gecco/config/two_step.yaml")
    df = load_data(cfg.data.path, cfg.data.input_columns)
    data2text = get_data2text_function(cfg.data.data2text_function)
    data_text = data2text(df, id_column=cfg.data.id_column, template=cfg.data.narrative_template)

    prompt = build_prompt(cfg, data_text)
    print("Prompt ready. Example snippet:\n", prompt[:600], "\n---")

    tokenizer, model = load_llm(cfg.llm.base_model)
    code = generate_model_code(
        tokenizer,
        model,
        prompt,
        max_tokens=cfg.llm.max_tokens,
        temperature=cfg.llm.temperature
    )

    print("\nGenerated model code (truncated):\n", code[:800], "\n---")

    # Extract + fit
    run_fit(df, code)

if __name__ == "__main__":
    main()
