# tests/test_compose_cli_rlwm.py
import subprocess
import sys
from pathlib import Path

from library_learning.__main__ import _task_mods


def test_task_mods_dispatch():
    two_step = _task_mods("results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual")
    rlwm = _task_mods("results/rlwm_individual")
    assert two_step["search"].__name__ == "library_learning.compose.search"
    assert rlwm["search"].__name__ == "library_learning.compose.search_rlwm"
    assert rlwm["extract"].__name__ == "library_learning.compose.extract_rlwm"
    assert rlwm["evaluate"].__name__ == "library_learning.compose.evaluate_rlwm"


def test_hybrid_search_rejected_for_rlwm():
    r = subprocess.run(
        [sys.executable, "-m", "library_learning", "compose-hybrid-search",
         "--results-dir", "results/rlwm_individual",
         "--group-dir", "results/rlwm"],
        capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    assert r.returncode != 0
    assert "deferred" in (r.stderr + r.stdout)
