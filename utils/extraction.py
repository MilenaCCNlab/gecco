import re
import ast
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


# ============================================================
# Parameter & bounds extraction
# ============================================================

def extract_parameter_names_from_unpack(code: str) -> List[str]:
    """
    Parse model parameter names from unpacking lines like:
      (alpha, beta, epsilon) = model_parameters
    """
    pattern = re.compile(
        r"[(\[]([A-Za-z0-9_,\s]+)[)\]]\s*=\s*model_parameters"
    )
    m = pattern.search(code)
    if not m:
        return []
    raw = m.group(1)
    params = [p.strip() for p in raw.split(",") if p.strip()]
    return params


def _extract_parameters_from_text(code: str) -> List[str]:
    """Fallback for extracting parameter names directly from code."""
    params = extract_parameter_names_from_unpack(code)
    if params:
        return params

    # fallback: parse AST (sometimes parameters appear as model_parameters[0], etc.)
    names = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if node.value.id == "model_parameters":
                    if isinstance(node.slice, ast.Constant):
                        idx = node.slice.value
                        if isinstance(idx, int):
                            names.append(f"param_{idx}")
    except Exception:
        pass
    return sorted(set(names))


def parse_bounds_from_docstring(doc: Optional[str]) -> Dict[str, List[float]]:
    """
    Extract parameter bounds from docstring patterns like:
      - alpha : learning rate [0,1]
      - beta  : temperature [0,10]
    Also supports global rules like:
      'Parameters (all in [0,1], except betas in [0,10])'
    """
    if not doc:
        return {}

    bounds: Dict[str, List[float]] = {}

    # Case 1: explicit per-parameter bounds
    explicit_pattern = re.compile(
        r"-\s*([A-Za-z0-9_]+)\s*:.*?\[([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\]",
        flags=re.I,
    )
    for name, lo, hi in explicit_pattern.findall(doc):
        try:
            bounds[name] = [float(lo), float(hi)]
        except Exception:
            continue

    if bounds:
        return bounds

    # Case 2: global rule with optional exceptions
    global_pattern = re.compile(
        r"all\s+in\s*\[\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\]"
        r"(?:[^[]*?except\s+([A-Za-z0-9_ ,/]+?)\s+in\s*\[\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\])?",
        flags=re.I | re.S,
    )
    m = global_pattern.search(doc)
    if not m:
        return {}

    lo, hi = float(m.group(1)), float(m.group(2))
    bounds = {"__default__": [lo, hi]}

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
    param_names = _extract_parameters_from_text(code)

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

    print(f"[DEBUG] defined callables: {list(ns.keys())}")
    print(f"[GeCCo] Extracted model: {name}")
    print(f"[GeCCo] Parameters: {param_names}")
    print(f"[GeCCo] Bounds: {bounds}")

    return ModelSpec(name=name, func=func, param_names=param_names, bounds=bounds)
