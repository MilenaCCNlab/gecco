def build_prompt(cfg, data_text, feedback_text=None):
    """
    Construct the structured LLM prompt for cognitive model generation.
    Order:
    1. Task description
    2. Example participant data
    3. Modeling goal and instructions
    4. Guardrails
    5. Template model
    """
    task, llm = cfg.task, cfg.llm
    guardrails = getattr(llm, "guardrails", [])
    include_feedback = getattr(llm, "include_feedback", False)

    # Format goal section dynamically
    names = [f"`cognitive_model{i+1}`" for i in range(llm.models_per_iteration)]
    goal_text = task.goal.format(
        models_per_iteration=llm.models_per_iteration,
        model_names=", ".join(names),
    )

    feedback_section = (
        f"\n\n### Feedback\n{feedback_text.strip()}"
        if (feedback_text and include_feedback)
        else ""
    )

    if cfg.llm.provider in ["openai", "claude", "gemini"]:
        # --- prompt layout for closed models ---
        prompt = f"""
### Task Description
{task.name}
{task.description.strip()}

### Example Participant Data
Here is example data from several participants:
{data_text.strip()}

### Your Task
{goal_text.strip()}

### Guardrails
{chr(10).join(guardrails)}

### Template Model (for reference only — do not reuse its logic)
{llm.template_model.strip()}

{feedback_section}
""".strip()
    else:
        # --- prompt layout for open models ---
        prompt = f"""
{task.description.strip()}

Here's data from several participants:
{data_text.strip()}

{goal_text.strip()}

### Implementation Guidelines
{chr(10).join(guardrails)}

### Initial Model Suggestion
Consider the following code as a function template:
{llm.template_model.strip()}

Your function:

{feedback_section}
        """.strip()

    return prompt

class PromptBuilderWrapper:
    def __init__(self, cfg, data_text):
        self.cfg = cfg
        self._data_text = data_text
    def build_input_prompt(self, feedback_text: str = ""):
        return build_prompt(self.cfg, self._data_text, feedback_text=feedback_text)