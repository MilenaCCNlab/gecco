# engine/model_search.py

import os
import json
import numpy as np

from gecco.offline_evaluation.fit_generated_models import run_fit
from gecco.utils import extract_full_function
from gecco.construct_feedback.feedback import FeedbackGenerator, LLMFeedbackGenerator
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
            self.feedback = LLMFeedbackGenerator(cfg, model, tokenizer)
        else:
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

    def generate(self, model, tokenizer=None, prompt=None):
        """
        Unified text generation function for any supported backend.
        Handles both OpenAI GPT and Hugging Face-style models cleanly.
        """
        if model is None:
            raise ValueError("Model not initialized correctly.")
        provider = self.cfg.llm.provider.lower()

        # -----------------------------
        # OpenAI / GPT-style generation
        # -----------------------------
        if "openai" in provider or "gpt" in provider:
            max_out = self.cfg.llm.max_output_tokens
            reasoning_effort = getattr(self.cfg.llm, "reasoning_effort", "medium")
            text_verbosity = getattr(self.cfg.llm, "text_verbosity", "low")

            print(
                f"[GeCCo] Using GPT model '{self.cfg.llm.base_model}' "
                f"(reasoning={reasoning_effort}, verbosity={text_verbosity}, max_output_tokens={max_out})"
            )

            resp = model.responses.create(
                model=self.cfg.llm.base_model,
                reasoning={"effort": "low"},
                input=[
                    {"role": "developer", "content": self.cfg.llm.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            decoded = resp.output_text.strip()

            return decoded

        # -----------------------------
        # Hugging Face-style generation
        # -----------------------------
        else:
            max_new = getattr(self.cfg.llm, "max_output_tokens", getattr(self.cfg.llm, "max_tokens", 2048))

            print(
                f"[GeCCo] Using HF model '{self.cfg.llm.base_model}' "
                f"(max_new_tokens={max_new}, temperature={self.cfg.llm.temperature})"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=self.cfg.llm.temperature,
                do_sample=True,
            )
            return tokenizer.decode(output[0], skip_special_tokens=True)

    def run_n_shots(self, run_idx):
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
            code_text = self.generate(self.model, self.tokenizer, prompt)

            model_file = self.results_dir / "models" / f"iter{it}_run{run_idx}.py"
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

                        best_model_file = self.results_dir / "models" / f"best_model_{run_idx}.py"
                        with open(best_model_file, "w") as f:
                            f.write(func_code)

                except Exception as e:
                    print(f"[⚠️ GeCCo] Error fitting {func_name}: {e}")

            # Save iteration results
            bic_file = self.results_dir / "bics" / f"iter{it}_run{run_idx}.json"
            with open(bic_file, "w") as f:
                json.dump(iteration_results, f, indent=2)

            self.feedback.record_iteration(it, iteration_results)

        print(
            f"\n[🏁 GeCCo] Finished search. "
            f"Best model (iteration {self.best_iter}) "
            f"{self.cfg.evaluation.metric.upper()}={self.best_metric:.2f}"
        )

        return self.best_model, self.best_metric, self.best_params

    