"""Render REPORT.md from verification results and library stats."""

import json
from datetime import date
from typing import Optional

from .config import Target
from . import stats as stats_mod


def _verification_table(results: dict) -> str:
    lines = [
        "| participant | params | n trials | original NLL | fitted diff | bitwise | probes | trace | synthetic | status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pid_str, r in sorted(results["participants"].items(), key=lambda kv: int(kv[0])):
        if r.get("status") == "ERROR":
            lines.append(f"| {pid_str} | — | — | — | — | — | — | — | — | ERROR: {r['error']} |")
            continue
        trace = "n/a"
        if r.get("trace_available"):
            trace = "ok" if "trace_divergence" not in r else (
                f"trial {r['trace_divergence']['trial']} stage {r['trace_divergence']['stage']}"
            )
        lines.append(
            f"| {pid_str} "
            f"| {', '.join(r.get('param_names', []))} "
            f"| {r.get('n_trials', '—')} "
            f"| {r.get('orig_nll', float('nan')):.4f} "
            f"| {r.get('fitted_diff', float('nan')):.2e} "
            f"| {'yes' if r.get('bitwise_equal') else 'no'} "
            f"| {r.get('probes_passed', 0)}/{r.get('probes', 0)} "
            f"| {trace} "
            f"| {r.get('synthetic_passed', 0)}/{r.get('synthetic_checks', 0)} "
            f"| {r['status']} |"
        )
    return "\n".join(lines)


def _primitive_table(stats: dict) -> str:
    lines = [
        "| primitive | participants using | total calls |",
        "|---|---|---|",
    ]
    for name, info in stats["primitives"].items():
        lines.append(f"| `{name}` | {info['participants_using']}/{stats['n_participants']} | {info['total_calls']} |")
    return "\n".join(lines)


def write_report(target: Target, stats: Optional[dict] = None) -> str:
    ver_path = target.library_dir / "verification_results.json"
    if not ver_path.exists():
        raise FileNotFoundError(
            f"{ver_path} missing - run `python -m library_learning verify` first"
        )
    with open(ver_path) as f:
        results = json.load(f)
    if stats is None:
        stats = stats_mod.write_stats(target)

    s = results["summary"]
    loc = stats["loc"]
    md = f"""# Library-learning report - {target.results_dir.name}

Generated {date.today().isoformat()} by `python -m library_learning report`.

## Verification

- **{s['matched']}/{s['total']} participants MATCH** ({s['match_rate']}), {s['n_bitwise_equal']} bitwise-equal NLLs
- max fitted-parameter NLL difference: {s['max_fitted_diff']}
- checks per participant: NLL at fitted params, {s['probes_per_participant']} seeded in-bounds parameter probes,
  trial-level p_choice traces (AST-instrumented), {s['synthetic_datasets']} synthetic datasets x 2 parameter vectors

{_verification_table(results)}

## Compression

| metric | originals | library + participants | ratio |
|---|---|---|---|
| non-blank LOC | {loc['originals_nonblank']} | {loc['library_nonblank']} + {loc['participants_nonblank']} | **{loc['compression_ratio_nonblank']}x** |
| code-only LOC (no docstrings/comments) | {loc['originals_code']} | {loc['library_code']} + {loc['participants_code']} | **{loc['compression_ratio_code']}x** |

Mean primitives used per participant: {stats['mean_primitives_per_participant']}

## Primitive reuse

{_primitive_table(stats)}

See `participant_primitive_matrix.csv` for the per-participant usage matrix and
`MECHANISMS.md` for the curated mechanism inventory.
"""
    (target.library_dir / "REPORT.md").write_text(md)
    return md
