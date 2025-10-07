# utils/extraction.py

import re
import inspect
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


# -------------------------------------------------------------------------
# ModelSpec dataclass — holds everything needed for fitting
# -------------------------------------------------------------------------
@dataclass
class ModelSpec:
    func: Callable[..., float]
    name: str
    param_names: List[str]
    bounds: Dict[str, List[float]]

    def n_params(self) -> int:
        return len(self.param_names)


# -------------------------------------------------------------------------
# Basic code extraction helpers
# -------------------------------------------------------------------------
def _extract_code_block(text: str) -> str:
    """Extract the first Python fenced code block, or return whole text."""
    m = re.search(r"```python(.*?)```", text, flags=re.S)
    if not m:
        m = re.search(r"```(.*?)```", text, flags=re.S)
    return (m.group(1) if m else text).strip()


def extract_full_function(text: str, func_name: str) -> Optional[str]:
    """Extract a full function definition by name from a text blob."""
    pattern = re.compile(rf"(def {func_name}\(.*?\):.*?)(?=\ndef |\Z)", re.DOTALL)
    m = pattern.search(text)
    return m.group(1).strip() if m else None


# -------------------------------------------------------------------------
# Parameter and bounds extraction helpers
# -------------------------------------------------------------------------
def extract_parameter_names_from_unpack(code: str) -> List[str]:
    """
    Find unpacking lines like:
        alpha, beta, gamma = model_parameters
    Returns the LHS names.
    """
    names: List[str] = []
    for line in code.splitlines():
        s = line.strip()
        m = re.match(r"^([\w\s,]+?)\s*=\s*[A-Za-z_][A-Za-z0-9_]*\s*$", s)
        if not m:
            continue
        lhs = [p.strip() for p in m.group(1).split(",") if p.strip()]
        if len(lhs) >= 2:
            names = lhs
            break
    return names


def _extract_parameters_from_text(code: str) -> Optional[List[str]]:
    """Look for: parameters = ['alpha','beta']"""
    m = re.search(r"parameters\s*=\s*\[([^\]]+)\]", code)
    if not m:
        return None
    raw = m.group(1)
    return [p.strip().strip("'\"") for p in raw.split(",") if p.strip()]


def _extract_bounds_from_text(code: str) -> Dict[str, List[float]]:
    """Parse bounds = {'alpha':[0,1], ...} or inline occurrences."""
    try:
        m = re.search(r"bounds\s*=\s*({.*?})", code, flags=re.S)
        if m:
            safe_ns = {}
            bounds_dict = eval(m.group(1), {"__builtins__": {}}, safe_ns)
            clean = {k: [float(v[0]), float(v[1])] for k, v in bounds_dict.items()}
            return clean
    except Exception:
        pass

    out: Dict[str, List[float]] = {}
    for name, nums in re.findall(r"['\"](\w+)['\"]\s*:\s*\[([^\]]+)\]", code):
        try:
            vals = [float(x) for x in nums.replace("\n", " ").split(",")]
            if len(vals) >= 2:
                out[name] = [vals[0], vals[1]]
        except Exception:
            continue
    return out


def parse_bounds_from_docstring(doc: Optional[str]) -> Dict[str, List[float]]:
    """
    Parse bounds from docstring lines like:
      - alpha : learning rate [0,1]
      - beta  : temperature [0,10]
    """
    if not doc:
        return {}
    bounds: Dict[str, List[float]] = {}
    pattern = re.compile(r"-\s*([A-Za-z0-9_]+)\s*:.*?\[([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\]")
    for name, lo, hi in pattern.findall(doc):
        try:
            bounds[name] = [float(lo), float(hi)]
        except Exception:
            continue
    return bounds


def _default_bounds_for(param: str) -> List[float]:
    """Fallback heuristics for common parameter names."""
    name = param.lower()
    if any(k in name for k in ["alpha", "prob", "rate", "w", "weight", "p_"]):
        return [0.0, 1.0]
    if any(k in name for k in ["beta", "temp", "inverse", "scale", "tau", "omega"]):
        return [1e-3, 100.0]
    if any(k in name for k in ["eps", "epsilon", "eta"]):
        return [0.0, 1.0]
    return [-10.0, 10.0]


# -------------------------------------------------------------------------
# Safe code execution helpers
# -------------------------------------------------------------------------
def _safe_exec_user_code(code: str) -> Dict[str, Any]:
    """Execute LLM-generated Python code in a restricted namespace."""
    ns: Dict[str, Any] = {}
    exec(compile(code, "<model>", "exec"), {"__builtins__": {}}, ns)
    return ns


def _find_first_function(ns: Dict[str, Any]) -> Optional[Callable[..., float]]:
    """Return the first callable found in a namespace."""
    for k, v in ns.items():
        if callable(v):
            return v
    return None


# -------------------------------------------------------------------------
# Core: build_model_spec_from_llm_output
# -------------------------------------------------------------------------
def build_model_spec_from_llm_output(text: str, expected_func_name: str = "cognitive_model") -> ModelSpec:
    """
    Parse LLM output containing a model function definition, extract
    parameter names, bounds, and compile a callable that takes a DataFrame.
    """
    code = _extract_code_block(text)

    # --- Execute code to get function ---
    ns = _safe_exec_user_code(code)
    func = ns.get(expected_func_name, None)
    if func is None:
        func = _find_first_function(ns)
    if func is None:
        raise ValueError("No callable model function found in generated code.")
    name = getattr(func, "__name__", expected_func_name)

    #
