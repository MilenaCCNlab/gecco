import json
from pathlib import Path

from library_learning.compose.evaluate import evaluate_models, summarize
from library_learning.config import resolve_target

IND = "results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual"
GRP = "results/two_step_psychiatry_group_function_ocibalanced_maxsetting"

TINY_MODEL = '''def cognitive_model(action_1, state, action_2, reward, model_parameters):
    """
    Bias-only stub.
    Parameters:
    p_bias: [0.001, 0.999] - choice bias
    beta_dummy: [0, 10] - unused scale kept for two-param shape
    """
    p_bias, beta_dummy = model_parameters
    log_loss = 0.0
    eps = 1e-10
    for trial in range(len(action_1)):
        if int(action_1[trial]) != -1:
            p = p_bias if int(action_1[trial]) == 1 else 1.0 - p_bias
            log_loss -= np.log(p + eps)
    return log_loss
'''


def test_evaluate_and_summarize(tmp_path):
    target = resolve_target(IND)
    (tmp_path / "composed_model.txt").write_text(TINY_MODEL)
    (tmp_path / "winner.json").write_text(json.dumps(
        {"candidate_id": "stub", "module_ids": [], "n_params": 2, "mean_bic": 0.0}))
    pids = [14, 15]
    res = evaluate_models(target, GRP, tmp_path, pids, set_name="test")
    assert set(res.keys()) == {"composed", "group", "hybrid", "individual"}
    for model, fits in res.items():
        assert set(fits.keys()) == set(pids)
        assert all(f["bic"] > 0 for f in fits.values())

    summarize(results_val=None, results_test=res, warnings=["w1"], out_dir=tmp_path)
    md = (tmp_path / "RESULTS.md").read_text()
    assert "wilcoxon" in md.lower() and "w1" in md
    csv = (tmp_path / "test_results.csv").read_text()
    assert "composed" in csv and "individual" in csv


def test_freeze_discipline(tmp_path):
    target = resolve_target(IND)
    import pytest
    with pytest.raises(FileNotFoundError):
        evaluate_models(target, GRP, tmp_path, [14], set_name="test")


def test_wilcoxon_p_all_zero_differences():
    import numpy as np
    from library_learning.compose.evaluate import _wilcoxon_p
    a = np.array([1.0, 2.0, 3.0])
    assert _wilcoxon_p(a, a.copy()) == 1.0
    b = a + np.array([0.5, -0.4, 0.9])
    assert 0.0 <= _wilcoxon_p(a, b) <= 1.0
