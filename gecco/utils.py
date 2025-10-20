import re
from typing import Dict, List
from gecco.prompt_builder.prompt import build_prompt


def extract_full_function(text: str, func_name: str) -> str:
    """
    Extract a full function definition for a given function name from the LLM output.
    Example:
        extract_full_function(llm_output, "cognitive_model1")
    will return:
        def cognitive_model1(...):
            ...
    """
    # Prefer fenced code block first
    match = re.search(r"```(?:python)?(.*?)```", text, re.S)
    if match:
        text = match.group(1).strip()

    # Match the specific function by name (greedy until next def or end)
    pattern = rf"(def\s+{func_name}\s*\([^)]*\)\s*:[\s\S]+?)(?=\n\s*def|\Z)"
    match = re.search(pattern, text, re.M)
    if match:
        func_block = match.group(1).strip()
    else:
        # Fallback: try to find any def block as a last resort
        match = re.search(r"(def\s+\w+\s*\([^)]*\)\s*:[\s\S]+?)(?=\n\s*def|\Z)", text, re.M)
        func_block = match.group(1).strip() if match else text.strip()

    # Clean up markdown or stray comments
    func_block = re.sub(r"^(\s*#+.*$)", "", func_block, flags=re.M)
    return func_block.strip()

class PromptBuilderWrapper:
    def __init__(self, cfg, data_text):
        self.cfg = cfg
        self._data_text = data_text
    def build_input_prompt(self, feedback_text: str = ""):
        return build_prompt(self.cfg, self._data_text, feedback_text)