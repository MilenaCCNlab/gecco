# tests/test_compose_evaluate_rlwm.py
import json

import pandas as pd

from library_learning.compose.evaluate_rlwm import summarize
from library_learning.compose.figure_rlwm import plot_comparison


def _fits(bic):
    return {p: {"bic": bic + p, "nll": 100.0, "params": [0.5], "seed": 1}
            for p in [4, 6]}


def test_summarize_and_results_md(tmp_path):
    results_test = {"composed": _fits(400.0), "group": _fits(410.0),
                    "canonical": _fits(420.0), "individual": _fits(390.0)}
    (tmp_path / "winner.json").write_text(json.dumps(
        {"candidate_id": "m1+m2", "n_params": 5}))
    stats = summarize(None, results_test, ["a warning"], tmp_path)
    assert stats["composed_vs_group"]["wins"] == 2
    assert stats["composed_vs_canonical"]["mean_delta"] == -20.0
    md = (tmp_path / "RESULTS.md").read_text()
    assert "composed vs canonical" in md and "a warning" in md
    df = pd.read_csv(tmp_path / "test_results.csv")
    assert set(df["model"]) == {"composed", "group", "canonical", "individual"}
    assert "age" in df.columns


def test_figure(tmp_path):
    rows = []
    for pid in [4, 6]:
        for model, b in [("composed", 400.0), ("group", 410.0),
                         ("canonical", 420.0), ("individual", 390.0)]:
            rows.append({"set": "test", "participant": pid, "age": 30.0,
                         "model": model, "n_params": 5, "nll": 100.0,
                         "bic": b + pid, "seed": 1})
    csv = tmp_path / "test_results.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    plot_comparison(csv, tmp_path)
    assert (tmp_path / "comparison.png").exists()
    assert (tmp_path / "comparison.pdf").exists()
