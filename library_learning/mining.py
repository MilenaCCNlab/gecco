"""Mechanical discovery of shared code fragments across participant models.

Each candidate fragment (single statements plus 2-3 statement windows, taken
from the function prologue and the trial loop) is normalized by renaming
local variables to positional slots — so `q1[a] += lr * d` and
`q_stage1[act] += alpha * delta` hash identically — then counted across
participants. The ranked output is the empirical evidence for which
mechanisms belong in cognitive_library.py; curation and naming stay a
human/LLM decision.
"""

import ast
import copy
import hashlib
from collections import defaultdict
from typing import Dict, List

# Globals that must keep their identity for fragments to stay comparable.
PRESERVED_NAMES = {
    "np", "len", "range", "float", "int", "abs", "max", "min",
    "enumerate", "zip", "sum", "model_parameters",
}

WINDOW_SIZES = (1, 2, 3)
MIN_NODES = 5  # ignore micro-fragments like `eps = 1e-10`


class _SlotRenamer(ast.NodeTransformer):
    def __init__(self):
        self.mapping: Dict[str, str] = {}

    def _slot(self, name: str) -> str:
        if name in PRESERVED_NAMES:
            return name
        if name not in self.mapping:
            self.mapping[name] = f"v{len(self.mapping)}"
        return self.mapping[name]

    def visit_Name(self, node: ast.Name):
        return ast.copy_location(ast.Name(id=self._slot(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg):
        node.arg = self._slot(node.arg)
        return node


def _normalized_hash(stmts: List[ast.stmt]) -> str:
    block = ast.Module(body=[copy.deepcopy(s) for s in stmts], type_ignores=[])
    block = _SlotRenamer().visit(block)
    dumped = ast.dump(block, annotate_fields=True, include_attributes=False)
    return hashlib.md5(dumped.encode()).hexdigest()


def _n_nodes(stmts: List[ast.stmt]) -> int:
    return sum(1 for s in stmts for _ in ast.walk(s))


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _blocks_of(func: ast.FunctionDef) -> List[List[ast.stmt]]:
    """Statement blocks worth mining: function body and every nested block."""
    blocks = [[s for s in func.body if not _is_docstring(s)]]
    for node in ast.walk(func):
        if isinstance(node, (ast.For, ast.While, ast.If)):
            blocks.append(list(node.body))
            if node.orelse:
                blocks.append(list(node.orelse))
    return blocks


def mine_fragments(models: Dict[int, str]) -> List[dict]:
    """models: pid -> source. Returns fragments shared by >= 2 participants."""
    frags: Dict[str, dict] = {}
    per_pid_hashes: Dict[int, set] = defaultdict(set)

    for pid, code in models.items():
        tree = ast.parse(code)
        func = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        if func is None:
            continue
        for block in _blocks_of(func):
            for size in WINDOW_SIZES:
                for i in range(len(block) - size + 1):
                    window = block[i : i + size]
                    if _n_nodes(window) < MIN_NODES:
                        continue
                    h = _normalized_hash(window)
                    if h not in frags:
                        frags[h] = {
                            "hash": h,
                            "n_statements": size,
                            "n_nodes": _n_nodes(window),
                            "example": "\n".join(ast.unparse(s) for s in window),
                            "pids": set(),
                            "occurrences": 0,
                        }
                    frags[h]["pids"].add(pid)
                    frags[h]["occurrences"] += 1
                    per_pid_hashes[pid].add(h)

    shared = []
    for f in frags.values():
        if len(f["pids"]) >= 2:
            f["pids"] = sorted(f["pids"])
            f["n_participants"] = len(f["pids"])
            shared.append(f)
    shared.sort(key=lambda f: (-f["n_participants"], -f["n_nodes"], f["hash"]))
    return shared


def render_mining_report(shared: List[dict], n_models: int, top: int = 40) -> str:
    lines = [
        "# Shared-fragment mining report",
        "",
        f"Fragments (normalized AST windows of 1-3 statements) shared by >=2 of {n_models} models,",
        "ranked by participant coverage then size. Evidence base for cognitive_library.py primitives.",
        "",
    ]
    for f in shared[:top]:
        lines.append(
            f"## {f['n_participants']}/{n_models} participants "
            f"({f['n_statements']} stmt, {f['n_nodes']} nodes, {f['occurrences']} occurrences)"
        )
        lines.append("")
        lines.append("```python")
        lines.append(f["example"])
        lines.append("```")
        lines.append(f"participants: {f['pids']}")
        lines.append("")
    return "\n".join(lines)
