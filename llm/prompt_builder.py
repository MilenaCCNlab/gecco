def build_prompt(cfg, data_text: str) -> str:
    """Construct the full LLM prompt from config and data text."""
    return f"""
### Task Description
{cfg.task.description}

### Goal of the Task
{cfg.task.goal}

### Task Instructions
{cfg.task.instructions}

### Additional Explanations
{cfg.task.extra}

### Data
{data_text}

### Model Requirements
Input arguments: {', '.join(cfg.data.input_columns)}
Output: Negative log-likelihood
Guardrails:
{chr(10).join(cfg.llm.guardrails)}
"""
