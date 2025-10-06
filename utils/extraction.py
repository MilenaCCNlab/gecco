import ast
import inspect
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any

# ---------- ModelSpec (runtime-adapted) ----------

@dataclass
class ModelSpec:
    func: Callable[..., float]            # callable that returns NLL (float)
    name: str                             # function name (e.g., "cognitive_model")
    param_names: List[str]                # ordered parameter names
    bounds: Dict[str, List[float]]        # {param: [low, high]}

    def n_params(self) -> int:
        return len(self.param_names)

# ---------- Extraction helpers ----------

def _extract_code_block(text: str) -> str:
    """
    Extract the first Python fenced code block, else return the whole text.
    Supports ```python ...``` or ``` ...```.
    """
    m = re.search(r"```python(.*?)```", text, flags=re.S)
    if not m:
        m = re.search(r"```(.*?)```", text, flags=re.S)
    return (m.group(1) if m else text).strip()

def _extract_parameters_from_text(code: str) -> Optional[List[str]]:
    """
    Look for lines like: parameters = ["alpha", "beta"]
    """
    m = re.search(r"parameters\s*=\s*\[([^\]]+)\]", code)
    if not m:
        return None
    raw = m.group(1)
    return [p.strip().strip("'\"") for p in raw.split(",") if p.strip()]

def _extract_bounds_from_text(code: str) -> Dict[str, List[float]]:
    """
    Look for dict-like: bounds = {"alpha": [0,1], "beta":[0.01,10]}
    or inline occurrences.
    """
    # try a quick-and-dirty eval in a safe namespace if we see "bounds = {"
    try:
        m = re.search(r"bounds\s*=\s*({.*?})", code, flags=re.S)
        if m:
            safe_ns = {}  # no builtins
            bounds_dict = eval(m.group(1), {"__builtins__": {}}, safe_ns)  # noqa: S307 (controlled context)
            # ensure values are lists of floats
            clean = {k: [float(v[0]), float(v[1])] for k, v in bounds_dict.items()}
            return clean
    except Exception:
        pass

    # fallback: find " 'param': [low, high] " pairs
    out: Dict[str, List[float]] = {}
    for name, nums in re.findall(r"['\"](\w+)['\"]\s*:\s*\[([^\]]+)\]", code):
        try:
            vals = [float(x) for x in nums.replace("\n", " ").split(",")]
            if len(vals) >= 2:
                out[name] = [vals[0], vals[1]]
        except Exception:
            continue
    return out

def _default_bounds_for(param: str) -> List[float]:
    """
    Heuristics for missing bounds.
    - probs/rates: [0,1]
    - temps/scale/beta: [1e-3, 100]
    - eps/epsilon/eta: [0,1]
    - otherwise: [-10, 10]
    """
    name = param.lower()
    if any(k in name for k in ["alpha", "prob", "rate", "w", "weight", "p_"]):
        return [0.0, 1.0]
    if any(k in name for k in ["beta", "temp", "inverse", "scale", "tau", "omega"]):
        return [1e-3, 100.0]
    if any(k in name for k in ["eps", "epsilon", "eta"]):
        return [0.0, 1.0]
    return [-10.0, 10.0]

def _safe_exec_user_code(code: str) -> Dict[str, Any]:
    """
    Execute generated code in a very restricted namespace.
    We remove builtins; only math/numpy can be injected as needed by you later.
    """
    ns: Dict[str, Any] = {}
    exec(compile(code, "<model>", "exec"), {"__builtins__": {}}, ns)  # noqa: P204
    return ns

def _find_first_function(ns: Dict[str, Any]) -> Optional[Callable[..., float]]:
    for k, v in ns.items():
        if callable(v):
            return v
    return None

def _function_signature_param_names(func: Callable) -> List[str]:
    sig = inspect.signature(func)
    # prefer explicit model parameters (exclude 'data' / 'dataset' / 'df')
    names = [p.name for p in sig.parameters.values()]
    # If data is first arg, keep it for adaptation later but exclude from params list
    filtered = [n for n in names if n.lower() not in ("data", "dataset", "df")]
    return filtered

def build_model_spec_from_llm_output(text: str, expected_func_name: str = "cognitive_model") -> ModelSpec:
    """
    Parse LLM output: extract code block, exec it, locate model function,
    extract param list + bounds, return a ModelSpec.
    """
    code = _extract_code_block(text)

    # Parse text hints first (parameters=..., bounds=...)
    hinted_params = _extract_parameters_from_text(code)
    hinted_bounds = _extract_bounds_from_text(code)

    # Exec to get the function
    ns = _safe_exec_user_code(code)

    # Try to fetch function by expected name, else first callable
    func = ns.get(expected_func_name, None)
    if func is None:
        func = _find_first_function(ns)
    if func is None:
        raise ValueError("No callable model function found in generated code.")

    name = getattr(func, "__name__", expected_func_name)

    # Determine parameter order
    param_names = hinted_params or _function_signature_param_names(func)

    # Fill missing bounds
    bounds = dict(hinted_bounds)
    for p in param_names:
        if p not in bounds:
            bounds[p] = _default_bounds_for(p)

    # Wrap to standard callable form:
    # We expect func may be defined as:
    #   (data, **params) OR (*params, data) OR (param1, param2, ..., data)
    # We'll build an adapter that always calls with: nll = func(data=df, **param_dict) or func(*ordered, data)
    sig = inspect.signature(func)
    names_in_order = [p.name for p in sig.parameters.values()]

    def nll_callable(data, **param_dict) -> float:
        # Build kwargs or args according to signature
        call_kwargs = {}
        call_args = []
        for nm in names_in_order:
            if nm.lower() in ("data", "dataset", "df"):
                call_kwargs[nm] = data
            elif nm in param_dict:
                call_kwargs[nm] = param_dict[nm]
            else:
                # fall back to positional ordered vector if provided that way
                if nm in param_dict:
                    call_args.append(param_dict[nm])
                else:
                    raise ValueError(f"Missing parameter '{nm}' for model call.")
        try:
            return float(func(**call_kwargs)) if call_kwargs else float(func(*call_args))
        except TypeError:
            # try mixed calling: positional ordered by param_names, and data kw
            ordered = [param_dict[p] for p in param_names]
            return float(func(*ordered, data=data))

    return ModelSpec(func=nll_callable, name=name, param_names=param_names, bounds=bounds)
