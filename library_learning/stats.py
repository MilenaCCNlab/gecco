"""Compression and primitive-reuse statistics for a learned library."""

import ast
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .config import Target
from . import loading


def _docstring_lines(tree: ast.AST) -> set:
    """Line numbers occupied by module/function/class docstrings."""
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc = body[0]
                lines.update(range(doc.lineno, doc.end_lineno + 1))
    return lines


def count_loc(source: str) -> Dict[str, int]:
    """Non-blank LOC and code-only LOC (excluding comments and docstrings)."""
    all_lines = source.splitlines()
    nonblank = [i + 1 for i, l in enumerate(all_lines) if l.strip()]
    try:
        doc_lines = _docstring_lines(ast.parse(source))
    except SyntaxError:
        doc_lines = set()
    code = [
        i for i in nonblank
        if i not in doc_lines and not all_lines[i - 1].strip().startswith("#")
    ]
    return {"nonblank": len(nonblank), "code": len(code)}


def primitive_usage(target: Target, pid: int) -> Counter:
    """Calls to cognitive_library imports inside participant_<pid>.py."""
    source = loading.load_participant_source(target, pid)
    tree = ast.parse(source)
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "cognitive_library":
            imported.update(alias.asname or alias.name for alias in node.names)
    counts: Counter = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in imported:
            counts[node.func.id] += 1
    return counts


def compute_stats(target: Target) -> Tuple[dict, str]:
    """Returns (library_stats dict, participant x primitive CSV text)."""
    pids = loading.participant_ids(target)

    orig_nonblank = orig_code = 0
    for pid in pids:
        loc = count_loc(loading.load_original_code(target, pid))
        orig_nonblank += loc["nonblank"]
        orig_code += loc["code"]

    lib_loc = count_loc((target.library_dir / "cognitive_library.py").read_text())

    part_nonblank = part_code = 0
    usage: Dict[int, Counter] = {}
    for pid in pids:
        loc = count_loc(loading.load_participant_source(target, pid))
        part_nonblank += loc["nonblank"]
        part_code += loc["code"]
        usage[pid] = primitive_usage(target, pid)

    totals: Counter = Counter()
    users: Dict[str, int] = defaultdict(int)
    for counts in usage.values():
        totals.update(counts)
        for prim in counts:
            users[prim] += 1
    primitives = sorted(totals, key=lambda p: (-users[p], -totals[p], p))

    stats = {
        "n_participants": len(pids),
        "loc": {
            "originals_nonblank": orig_nonblank,
            "originals_code": orig_code,
            "library_nonblank": lib_loc["nonblank"],
            "library_code": lib_loc["code"],
            "participants_nonblank": part_nonblank,
            "participants_code": part_code,
            "compression_ratio_nonblank": round(
                orig_nonblank / (lib_loc["nonblank"] + part_nonblank), 3
            ),
            "compression_ratio_code": round(
                orig_code / (lib_loc["code"] + part_code), 3
            ),
        },
        "primitives": {
            p: {"participants_using": users[p], "total_calls": totals[p]}
            for p in primitives
        },
        "mean_primitives_per_participant": round(
            sum(len(c) for c in usage.values()) / len(pids), 2
        ) if pids else 0,
    }

    header = ["participant"] + primitives
    rows = [",".join(header)]
    for pid in pids:
        rows.append(",".join([str(pid)] + [str(usage[pid].get(p, 0)) for p in primitives]))
    matrix_csv = "\n".join(rows) + "\n"
    return stats, matrix_csv


def write_stats(target: Target) -> dict:
    stats, matrix_csv = compute_stats(target)
    with open(target.library_dir / "library_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    (target.library_dir / "participant_primitive_matrix.csv").write_text(matrix_csv)
    return stats
