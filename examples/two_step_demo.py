# examples/two_step_demo.py

import os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.schema import load_config
from data.io import load_data, split_by_participant
from data.data2text import get_data2text_function
from llm.model_loader import load_llm
from llm.prompt_builder import build_prompt
from engine.model_search import GeCCoModelSearch


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
directory = "/Users/milena.rmus/Desktop/gecco/"


# -------------------------------------------------------------------------
# Main entrypoint
# -------------------------------------------------------------------------
def main():
    # --- Load configuration & data ---
    cfg = load_config(f"{directory}config/two_step.yaml")
    data_cfg = cfg.data

    df = load_data(data_cfg.path, data_cfg.input_columns)
    splits = split_by_participant(df, data_cfg.id_column, data_cfg.splits)
    df_prompt, df_eval = splits["prompt"], splits["eval"]

    # --- Convert data to narrative text for the LLM ---
    data2text = get_data2text_function(data_cfg.data2text_function)
    data_text = data2text(
        df_prompt,
        id_col=data_cfg.id_column,
        template=data_cfg.narrative_template,
        max_trials=getattr(data_cfg, "max_trials", None),
    )

    # --- Prepare prompt builder wrapper ---
    class PromptBuilderWrapper:
        """Light adapter so the engine can reuse the build_prompt logic."""
        def __init__(self, cfg, data_text):
            self.cfg = cfg
            self.data_text = data_text

        def build_input_prompt(self, task_text, data_text, template_text, feedback_text=""):
            return build_prompt(self.cfg, data_text, feedback_text=feedback_text)

    prompt_builder = PromptBuilderWrapper(cfg, data_text)

    # --- Load LLM ---
    model, tokenizer = load_llm(cfg.llm.provider, cfg.llm.base_model)

    # --- Run GeCCo iterative model search ---
    search = GeCCoModelSearch(model, tokenizer, cfg, df_eval, prompt_builder)
    best_model, best_bic, best_params = search.run()

    # --- Print final results ---
    print("\n[🏁 GeCCo] Search complete.")
    print(f"Best model parameters: {', '.join(best_params)}")
    print(f"Best mean BIC: {best_bic:.2f}")


# -------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
