from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import yaml
import pandas as pd
from gecco.prepare_data.data2text import narrative
from types import SimpleNamespace

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


def load_data_from_config(cfg):
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["path"])
    text_output = narrative(
        df,
        template=data_cfg["narrative_template"],
        id_col=data_cfg.get("id_column", "participant"),
    )
    return text_output

def dict_to_namespace(d):
    """Recursively convert nested dicts into SimpleNamespace objects."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def load_config(path: str):
    """Load YAML config and allow dot notation access."""
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return dict_to_namespace(cfg_dict)
