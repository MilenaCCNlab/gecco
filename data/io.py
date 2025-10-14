import pandas as pd
from typing import List
import os
import numpy as np

def load_data(path, input_columns=None):
    # If path is relative, make it relative to project root
    if not os.path.isabs(path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(project_root, path)

    df = pd.read_csv(path)

    # Optional: keep only specified columns
    if input_columns is not None:
        df =  df[input_columns + ["participant", "trial"]] \
            if "participant" in df.columns and "trial" in df.columns \
            else df[input_columns]
    return df



def split_by_participant(df, id_col, splits_cfg):
    """
    Split data into prompt/eval/test sets based on participant IDs.
    Works whether splits_cfg is a dict or SimpleNamespace.
    """
    # 👇 handle both dicts and namespaces
    if not isinstance(splits_cfg, dict):
        splits_cfg = vars(splits_cfg)  # convert SimpleNamespace to dict
    unique_ids = sorted(np.unique(df[id_col].values).tolist())
    n = len(unique_ids)

    def parse_split(value):
        if isinstance(value, str):
            if value.startswith("first"):
                k = int(value.replace("first", ""))
                return unique_ids[:k]
            elif value.startswith("next"):
                k = int(value.replace("next", ""))
                return unique_ids[k : 2 * k]
            elif value == "remainder":
                return None
        elif isinstance(value, list):
            return value
        else:
            raise ValueError(f"Unsupported split format: {value}")

    prompt_ids = parse_split(splits_cfg.get("prompt", []))
    eval_ids = parse_split(splits_cfg.get("eval", []))
    test_ids = parse_split(splits_cfg.get("test", []))

    used = set((prompt_ids or []) + (eval_ids or []))
    if test_ids is None:
        test_ids = [pid for pid in unique_ids if pid not in used]

    return {
        "prompt": df[df[id_col].isin(prompt_ids)],
        "eval": df[df[id_col].isin(eval_ids)],
        "test": df[df[id_col].isin(test_ids)],
    }
