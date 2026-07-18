# tests/test_compose_reconstruct_rlwm.py
import json

from library_learning.compose import reconstruct_rlwm as R


def test_reconstruct_uses_stubs(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "greedy_search", lambda inv, t, pids, d: [
        {"candidate_id": "m1", "module_ids": ["m1"], "n_params": 4,
         "per_pid": {str(pids[0]): {"bic": 400.0}}, "mean_bic": 400.0}])
    monkeypatch.setattr(R, "load_original_code", lambda t, pid: "def cognitive_model(a, b, c, d, e, model_parameters):\n    x, = model_parameters\n    return 0.0")
    monkeypatch.setattr(R, "fit_model_on_pids",
                        lambda src, t, pids, bounds, tag, func_name=None:
                        {pids[0]: {"bic": 390.0 if "individual" in tag else 410.0,
                                   "nll": 0.0, "params": [0.5], "seed": 1}})
    monkeypatch.setattr(R, "strip_fences", lambda text: text)
    monkeypatch.setattr(
        R, "function_name_and_args", lambda code: ("cognitive_model", []))
    (tmp_path / "grp" / "models").mkdir(parents=True)
    (tmp_path / "grp" / "models" / "best_model_0.txt").write_text("stub")

    results = R.reconstruct_participants(None, None, tmp_path / "grp", [7], tmp_path)
    assert results[0]["pid"] == 7
    assert results[0]["library_bic"] == 400.0
    assert results[0]["individual_bic"] == 390.0
    assert results[0]["group_bic"] == 410.0
    saved = json.loads((tmp_path / "reconstruction_results.json").read_text())
    assert saved == results
