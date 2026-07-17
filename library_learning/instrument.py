"""AST instrumentation for trial-level equivalence tracing.

Every model in this family accumulates per-trial choice probabilities in
local arrays ``p_choice_1`` / ``p_choice_2`` before reducing them to a scalar
NLL. Rewriting the function's ``return`` to also emit those arrays gives
trial-level traces on both the original and the library rewrite without
touching their arithmetic, so a mismatch can be localized to the first
divergent trial and stage instead of a bare scalar difference.
"""

import ast
import sys
from typing import List, Optional

import numpy as np

TRACE_VARS = ("p_choice_1", "p_choice_2")


class _ReturnTracer(ast.NodeTransformer):
    """Rewrite `return expr` -> `return (expr, p_choice_1, p_choice_2)`.

    Applied to the statements of one function body; nested function
    definitions are left untouched.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return node  # do not descend into nested defs

    def visit_AsyncFunctionDef(self, node):
        return node

    def visit_Return(self, node: ast.Return):
        if node.value is None:
            return node
        tup = ast.Tuple(
            elts=[node.value] + [ast.Name(id=v, ctx=ast.Load()) for v in TRACE_VARS],
            ctx=ast.Load(),
        )
        return ast.copy_location(ast.Return(value=tup), node)


def _assigned_names(func: ast.FunctionDef) -> set:
    names = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def instrument_source(code: str, func_name: str) -> Optional[str]:
    """Return transformed source whose func also returns trace arrays.

    Returns None when the function does not define the trace variables
    (trace unavailable; caller falls back to scalar-only checks).
    """
    tree = ast.parse(code)
    func = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            func = node
            break
    if func is None:
        raise ValueError(f"function '{func_name}' not found for instrumentation")
    if not all(v in _assigned_names(func) for v in TRACE_VARS):
        return None
    tracer = _ReturnTracer()
    func.body = [tracer.visit(stmt) for stmt in func.body]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def exec_traced(code: str, func_name: str, extra_sys_path: Optional[str] = None):
    """Exec instrumented source; returns callable or None if trace unavailable.

    ``extra_sys_path`` lets participant files resolve their
    ``from cognitive_library import ...`` while exec'd.
    """
    traced_src = instrument_source(code, func_name)
    if traced_src is None:
        return None
    if extra_sys_path and extra_sys_path not in sys.path:
        sys.path.insert(0, extra_sys_path)
    ns = {"np": np}
    exec(compile(traced_src, "<instrumented>", "exec"), ns)  # noqa: S102
    return ns[func_name]


def first_divergence(
    p1_a: np.ndarray,
    p1_b: np.ndarray,
    p2_a: np.ndarray,
    p2_b: np.ndarray,
    atol: float = 1e-9,
) -> Optional[dict]:
    """Earliest trial (then stage) where traces diverge beyond atol, or None."""
    candidates = []
    for stage, (a, b) in enumerate(((p1_a, p1_b), (p2_a, p2_b)), start=1):
        bad = np.flatnonzero(~np.isclose(a, b, rtol=0.0, atol=atol, equal_nan=True))
        if bad.size:
            t = int(bad[0])
            candidates.append((t, stage, float(a[t]), float(b[t])))
    if not candidates:
        return None
    t, stage, orig_p, lib_p = min(candidates)
    return {"trial": t, "stage": stage, "orig_p": orig_p, "lib_p": lib_p}
