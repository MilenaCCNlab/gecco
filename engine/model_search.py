# engine/model_search.py

import os
import json
import numpy as np

from llm.generator import generate
from engine.run_fit import run_fit
from utils.extraction import extract_full_function
from engine.feedback import FeedbackGenerator  # <— new import
from pathlib import Path


class GeCCoModelSearch:
    def __init__(self, model, tokenizer, cfg, df, prompt_builder):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.df = df
        self.prompt_builder = prompt_builder

        # --- Choose feedback generator based on config ---
        if hasattr(cfg, "feedback") and getattr(cfg.feedback, "type", "manual") == "llm":
            from engine.feedback import LLMFeedbackGenerator
            self.feedback = LLMFeedbackGenerator(cfg, model, tokenizer)
        else:
            from engine.feedback import FeedbackGenerator
            self.feedback = FeedbackGenerator(cfg)

        # --- Set project root ---
        self.project_root = Path(__file__).resolve().parents[1]

        # --- Results directory (absolute path) ---
        self.results_dir = self.project_root / "results" / self.cfg.task.name
        (self.results_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "bics").mkdir(parents=True, exist_ok=True)

        # --- Tracking ---
        self.best_model = None
        self.best_metric = np.inf
        self.best_params = []
        self.best_iter = -1
        self.tried_param_sets = []

    def run(self):
        for it in range(self.cfg.loop.max_iterations):
            print(f"\n[GeCCo] --- Iteration {it} ---")

            # === Feedback generation ===
            feedback = ""
            if self.best_model is not None:
                feedback = self.feedback.get_feedback(
                    self.best_model,
                    self.tried_param_sets,
                )

            prompt = self.prompt_builder.build_input_prompt(feedback_text=feedback)
            code_text = generate(self.model, self.tokenizer, prompt, self.cfg)

            model_file = self.results_dir / "models" / f"iter{it}.py"
            with open(model_file, "w") as f:
                f.write(code_text)

            iteration_results = []

            for i in range(1, self.cfg.llm.models_per_iteration + 1):
                func_name = f"cognitive_model{i}"
                func_code = extract_full_function(code_text, func_name)
                if not func_code:
                    continue

                try:
                    fit_res = run_fit(self.df, func_code, cfg=self.cfg, expected_func_name=func_name)

                    mean_metric = float(fit_res["metric_value"])
                    metric_name = fit_res["metric_name"]
                    params = fit_res["param_names"]
                    self.tried_param_sets.append(params)

                    print(f"[GeCCo] {func_name}: mean {metric_name} = {mean_metric:.2f}")

                    iteration_results.append({
                        "function_name": func_name,
                        "metric_name": metric_name,
                        "metric_value": mean_metric,
                        "param_names": params,
                        "code_file": str(model_file),  # ✅ convert Path -> str
                    })

                    if mean_metric < self.best_metric:
                        self.best_metric = mean_metric
                        self.best_model = func_code
                        self.best_iter = it
                        self.best_params = params
                        print(f"[⭐ GeCCo] New best model: {func_name} ({metric_name}={mean_metric:.2f})")

                        best_model_file = self.results_dir / "models" / "best_model.py"
                        with open(best_model_file, "w") as f:
                            f.write(func_code)

                except Exception as e:
                    print(f"[⚠️ GeCCo] Error fitting {func_name}: {e}")

            # Save iteration results
            bic_file = self.results_dir / "bics" / f"iter{it}.json"
            with open(bic_file, "w") as f:
                json.dump(iteration_results, f, indent=2)

            self.feedback.record_iteration(it, iteration_results)

        print(
            f"\n[🏁 GeCCo] Finished search. "
            f"Best model (iteration {self.best_iter}) "
            f"{self.cfg.evaluation.metric.upper()}={self.best_metric:.2f}"
        )

        return self.best_model, self.best_metric, self.best_params
