from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import yaml

class TaskConfig(BaseModel):
    name: str
    description: str
    goal: str
    instructions: str
    extra: Optional[str] = ""

class DataConfig(BaseModel):
    path: str
    id_column: str
    input_columns: List[str]
    data2text_function: str = "narrative"
    narrative_template: Optional[str] = None

class LLMConfig(BaseModel):
    base_model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    guardrails: List[str]

class EvaluationConfig(BaseModel):
    metric: str = "BIC"
    fitting_method: str = "scipy_minimize"
    best_model_path: Optional[str] = None

class GeCCoConfig(BaseModel):
    task: TaskConfig
    data: DataConfig
    llm: LLMConfig
    evaluation: EvaluationConfig

def load_config(path: str) -> GeCCoConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return GeCCoConfig(**raw)
