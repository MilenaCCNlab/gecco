"""Loaders and code utilities for original models and library rewrites.

Regex conventions deliberately mirror ``gecco/offline_evaluation/utils.py``
(parameter unpack line, docstring bounds, markdown fences) so anything this
harness accepts stays fittable by gecco's ``run_fit``.
"""

import ast
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Target

FENCE_RE = re.compile(r"```(?:python|plaintext)?\s*(.*?)```", re.DOTALL)
BEST_MODEL_RE = re.compile(r"best_model_0_participant(\d+)\.txt$")
UNPACK_RE = re.compile(r"^([\w\s,]+?)\s*=\s*model_parameters\b")

# Same explicit-bounds pattern as gecco/offline_evaluation/utils.py::parse_bounds_from_docstring
BOUNDS_RE = re.compile(
    r"""
    (?:^|\n)\s*[-*]?\s*
    ([A-Za-z_][A-Za-z0-9_]*)
    [^()\[\]\n]*?
    [\(\[\{]\s*
    ([\-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*
    [,\s]+\s*
    ([\-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*
    [\)\]\}]
    """,
    flags=re.I | re.X | re.M,
)

_df_cache: Dict[str, pd.DataFrame] = {}


def strip_fences(text: str) -> str:
    m = FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


def participant_ids(target: Target) -> List[int]:
    ids = []
    for path in target.models_dir.iterdir():
        m = BEST_MODEL_RE.search(path.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def load_original_code(target: Target, pid: int) -> str:
    path = target.models_dir / f"best_model_0_participant{pid}.txt"
    return strip_fences(path.read_text())


def function_name_and_args(code: str) -> Tuple[str, List[str]]:
    """Name and positional args of the first function definition (AST-based)."""
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name, [a.arg for a in node.args.args]
    raise ValueError("no function definition found")


def extract_unpack_names(code: str) -> List[str]:
    """Parameter names from the `a, b, c = model_parameters` line (gecco Pattern 2)."""
    for line in code.splitlines():
        stripped = line.strip()
        if "model_parameters" in stripped and "=" in stripped and "self." not in stripped:
            m = UNPACK_RE.match(stripped)
            if m:
                params = [p.strip() for p in m.group(1).split(",") if p.strip()]
                if params:
                    return params
    return []


def parse_bounds(code: str, param_names: List[str]) -> Dict[str, Tuple[float, float]]:
    """Docstring bounds per parameter; gecco defaults where absent."""
    tree = ast.parse(code)
    doc = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node)
            break
    found: Dict[str, Tuple[float, float]] = {}
    if doc:
        for name, lo, hi in BOUNDS_RE.findall(doc):
            try:
                found[name] = (float(lo), float(hi))
            except ValueError:
                continue
    found_lower = {k.lower(): v for k, v in found.items()}
    bounds = {}
    for name in param_names:
        if name in found:
            bounds[name] = found[name]
        elif name.lower() in found:
            bounds[name] = found[name.lower()]
        elif name.lower() in found_lower:
            bounds[name] = found_lower[name.lower()]
        elif "beta" in name.lower() or "temperature" in name.lower():
            bounds[name] = (0.0, 10.0)
        else:
            bounds[name] = (0.0, 1.0)
    return bounds


def bounds_for_code(code: str) -> List[Tuple[float, float]]:
    """Bounds list (in unpack order) for a model's own docstring/unpack line."""
    names = extract_unpack_names(code)
    b = parse_bounds(code, names)
    return [b[n] for n in names]


def load_fitted_params(target: Target, pid: int) -> Tuple[List[str], np.ndarray]:
    path = target.params_dir / f"best_params_run0_participant{pid}.csv"
    lines = path.read_text().strip().splitlines()
    header = [h.strip() for h in lines[0].split(",")]
    values = np.array([float(v) for v in lines[1].split(",")], dtype=float)
    return header, values


def load_stored_bic(target: Target, pid: int) -> Optional[float]:
    path = target.bics_dir / f"best_bic_0_participant{pid}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return float(json.load(f)["bic"])


def load_dataframe(target: Target) -> pd.DataFrame:
    key = str(target.data_path)
    if key not in _df_cache:
        _df_cache[key] = pd.read_csv(target.data_path)
    return _df_cache[key]


def participant_inputs(target: Target, pid: int) -> Tuple[List[np.ndarray], int]:
    """Input arrays exactly as gecco's run_fit builds them (dtype-preserving)."""
    df = load_dataframe(target)
    df_p = df[df[target.id_column] == pid]
    if df_p.empty:
        raise ValueError(f"no rows for participant {pid} in {target.data_path}")
    inputs = [df_p[c].to_numpy() for c in target.input_columns]
    return inputs, len(df_p)


def exec_model(code: str, func_name: Optional[str] = None):
    """Exec model source with numpy injected (mirrors _safe_exec_user_code)."""
    ns = {"np": np}
    exec(code, ns)  # noqa: S102 - trusted, locally generated model code
    if func_name is None:
        func_name, _ = function_name_and_args(code)
    func = ns.get(func_name)
    if func is None:
        raise ValueError(f"function '{func_name}' not found after exec")
    return func


def import_participant_module(target: Target, pid: int):
    """Import participant_<pid>.py from the target's cognitive_library dir."""
    lib_dir = str(target.library_dir)
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    importlib.invalidate_caches()
    name = f"participant_{pid}"
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def load_participant_source(target: Target, pid: int) -> str:
    return (target.library_dir / f"participant_{pid}.py").read_text()
