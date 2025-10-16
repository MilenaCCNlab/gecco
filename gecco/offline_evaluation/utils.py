import re
import types
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# ============================================================
# Dataclass for unified representation of model specifications
# ============================================================

@dataclass
class ModelSpec:
    name: str
    func: Any
    param_names: List[str]
    bounds: Dict[str, List[float]]


# ============================================================
# Helper utilities
# ============================================================

def _extract_code_block(text: str) -> str:
    """Extract the Python function definition from an LLM output text."""
    code_match = re.search(r"```python(.*?)```", text, flags=re.S)
    if code_match:
        return code_match.group(1).strip()
    # fallback: raw text may just contain def ...
    func_match = re.search(r"def\s+\w+\s*\(.*", text, flags=re.S)
    if func_match:
        return func_match.group(0).strip()
    return text.strip()


def _safe_exec_user_code(code: str) -> Dict[str, Any]:
    """Safely execute user code and return namespace."""
    ns: Dict[str, Any] = {}
    try:
        exec(code, {}, ns)
    except Exception as e:
        print(f"[⚠️ GeCCo] Error executing model code: {e}")
    return ns


def _find_first_function(ns: Dict[str, Any]) -> Optional[Any]:
    """Return the first callable function defined in the executed namespace."""
    for k, v in ns.items():
        if isinstance(v, types.FunctionType):
            return v
    return None

def find_softmax_index_in_list(target_list):
    search_terms = {'beta', 'beta1', 'beta2', 'beta_1', 'beta_2', 'softmax', 'softmax_beta', 'theta', 'temperature',
                    'inverse_temperature'}

    indices = [i for i, element in enumerate(target_list) if element in search_terms]

    return indices  # Returns an empty list [] if no matches are found



# ============================================================
# Parameter & bounds extraction
# ============================================================

def extract_parameter_names(text):
    """
    Extract parameter names from a Python code snippet where model
    parameters are unpacked in a single assignment statement.
    Works with indentation and arbitrary variable names (e.g. params, parameters, model_ps, etc.).
    """
    for line in text.splitlines():
        # Strip leading/trailing whitespace to be robust to indentation
        stripped = line.strip()
        # Match lines like: a, b, c = something
        match = re.match(r'^([\w\s,]+?)\s*=\s*[A-Za-z_][A-Za-z0-9_]*$', stripped)
        if match:
            lhs = match.group(1)
            params = [p.strip() for p in lhs.split(',') if p.strip()]
            if len(params) > 1:
                return params
    return []


def parse_bounds_from_docstring(doc: Optional[str]) -> Dict[str, List[float]]:
    """
    Extract parameter bounds from docstrings.
    Supports patterns like:
      - alpha : [0,1]
      - alpha in [0,1]: learning rate
      - alpha (0,1)
      - alpha = [0, 1]
      - Parameters (all in [0,1], except betas in [0,10])
      - alpha range 0–1
    """

    if not doc:
        return {}

    bounds: Dict[str, List[float]] = {}

    # --- Case 1: explicit per-parameter bounds ---
    explicit_pattern = re.compile(
        r"""
        -\s*([A-Za-z0-9_]+)       # parameter name
        [^()\[\]\d\-]*?           # any filler before bounds (e.g. "in", ":", "=")
        [\(\[\{]?                 # optional opening bracket/paren/brace
        \s*([\-+]?\d*\.?\d+(?:e[-+]?\d+)?)   # lower bound
        \s*[,–\-to]+\s*           # comma, dash, or 'to' as separator
        ([\-+]?\d*\.?\d+(?:e[-+]?\d+)?)      # upper bound
        [\)\]\}]?                 # optional closing bracket/paren/brace
        """,
        flags=re.I | re.X,
    )

    for name, lo, hi in explicit_pattern.findall(doc):
        try:
            bounds[name] = [float(lo), float(hi)]
        except ValueError:
            continue

    if bounds:
        return bounds

    # --- Case 2: global rules with exceptions ---
    global_pattern = re.compile(
        r"""
        all\s+in\s*[\(\[\{]?\s*([\-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*[,–\-to]+\s*
        ([\-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*[\)\]\}]?
        (?:[^[]*?except\s+([A-Za-z0-9_ ,/]+?)\s+
        in\s*[\(\[\{]?\s*([\-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*[,–\-to]+\s*
        ([\-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*[\)\]\}]?)?
        """,
        flags=re.I | re.S | re.X,
    )

    m = global_pattern.search(doc)
    if not m:
        return {}

    lo, hi = float(m.group(1)), float(m.group(2))
    bounds = {"__default__": [lo, hi]}

    # handle "except ..." clause
    if m.group(3):
        phrase = m.group(3).lower()
        exc_lo, exc_hi = float(m.group(4)), float(m.group(5))
        tokens = re.findall(r"[A-Za-z0-9_]+", phrase)
        expanded = set(tokens)
        for t in tokens:
            if t.endswith("s") and len(t) > 3:
                expanded.add(t[:-1])
        for tok in expanded:
            bounds[tok] = [exc_lo, exc_hi]

    return bounds

# ============================================================
# High-level builder
# ============================================================

def build_model_spec_from_llm_output(
    text: str, expected_func_name: str = "cognitive_model"
) -> ModelSpec:
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

    # --- Parameter names ---
    param_names = extract_parameter_names(code)

    # --- Bounds from docstring ---
    doc = func.__doc__ or ""
    bounds = parse_bounds_from_docstring(doc)
    if not bounds:
        print(f"[⚠️ GeCCo] No bounds found in {name}; defaulting to [0,1].")
        bounds = {p: [0, 1] for p in param_names}
    else:
        # Fill missing params using default/global bounds
        if "__default__" in bounds:
            default_bound = bounds.pop("__default__")
        else:
            default_bound = [0, 1]

        bounds = {
            p: bounds.get(p, next((b for k, b in bounds.items() if k in p.lower()), default_bound))
            for p in param_names
        }


    return ModelSpec(name=name, func=func, param_names=param_names, bounds=bounds)