# engine/model_search.py
import os, json, numpy as np

from llm.generator import generate
from engine.run_fit import run_fit                    # <— fixed
from utils.extraction import extract_full_function    # <— fixed (no gecco.*)
# NOTE: we don't need extract_parameter_names here anymore

class GeCCoModelSearch:
    def __init__(self, model, tokenizer, cfg, df, prompt_builder):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.df = df
        self.prompt_builder = prompt_builder
        self.results_dir = "results"
        os.makedirs(f"{self.results_dir}/models", exist_ok=True)
        os.makedirs(f"{self.results_dir}/bics", exist_ok=True)
        self.best_model = None
        self.best_bic = np.inf
        self.best_params = []
        self.best_iter = -1

    def run(self):
        for it in range(self.cfg.loop.max_iterations):
            print(f"\n[GeCCo] --- Iteration {it} ---")

            feedback = ""
            if self.best_model is not None:
                feedback += (
                    f"The best model so far (iteration {self.best_iter}) "
                    f"has mean BIC = {self.best_bic:.2f}.\n"
                    f"It uses parameters: {', '.join(self.best_params)}.\n"
                    "When generating new models, build on these strengths conceptually, "
                    "but explore alternative mechanisms.\n"
                )

            prompt = self.prompt_builder.build_input_prompt(feedback_text=feedback)

            code_text = generate(self.model, self.tokenizer, prompt, self.cfg)

            for i in range(1, self.cfg.llm.models_per_iteration + 1):
                func_name = f"cognitive_model{i}"
                func_code = extract_full_function(code_text, func_name)
                if not func_code:
                    continue
                try:
                    fit_res = run_fit(self.df, func_code, expected_func_name=f"cognitive_model{i}")  # returns dict
                    mean_bic = float(np.mean(fit_res["bics"]))
                    print(f"[GeCCo] {func_name}: mean BIC = {mean_bic:.2f}")

                    with open(f"{self.results_dir}/models/iter{it}_{func_name}.py", "w") as f:
                        f.write(func_code)

                    if mean_bic < self.best_bic:
                        self.best_bic = mean_bic
                        self.best_model = func_code
                        self.best_iter = it
                        self.best_params = fit_res["param_names"]
                        print(f"[⭐ GeCCo] New best model: {func_name} (BIC={mean_bic:.2f})")
                        with open(f"{self.results_dir}/models/best_model.py", "w") as f:
                            f.write(func_code)

                except Exception as e:
                    print(f"[⚠️ GeCCo] Error fitting {func_name}: {e}")

        print(f"\n[🏁 GeCCo] Finished search. Best model (iter {self.best_iter}) BIC={self.best_bic:.2f}")
        return self.best_model, self.best_bic, self.best_params
