import subprocess
import sys

import pandas as pd


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "library_learning"] + list(args),
        capture_output=True, text=True)


def test_help_lists_subcommands():
    r = run_cli("--help")
    assert r.returncode == 0
    for sub in ["compose-modules", "compose-count", "compose-search",
                "compose-reconstruct", "compose-eval"]:
        assert sub in r.stdout


def test_search_requires_mode():
    r = run_cli("compose-search")
    assert r.returncode != 0
    assert "--mode" in r.stderr


def test_figure(tmp_path):
    from library_learning.compose.figure import plot_comparison
    rows = []
    for pid, boost in [(14, 0.0), (15, 10.0)]:
        for model, b in [("composed", 400.0), ("group", 420.0),
                         ("hybrid", 440.0), ("individual", 380.0)]:
            rows.append({"set": "test", "participant": pid, "oci": 0.5,
                         "model": model, "n_params": 4, "nll": 100.0,
                         "bic": b + boost, "seed": 1})
    csv = tmp_path / "test_results.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    plot_comparison(csv, tmp_path)
    assert (tmp_path / "comparison.png").exists()
    assert (tmp_path / "comparison.pdf").exists()
