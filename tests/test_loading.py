from library_learning.loading import parse_bounds

CODE_MIXED_CASE = '''def cognitive_model(action_1, state, action_2, reward, model_parameters):
    """
    Bias-only stub.
    Parameters:
    Alpha: [0.1, 0.9] - learning rate (docstring uses different case than unpack)
    """
    alpha, beta = model_parameters
    return 0.0
'''


def test_parse_bounds_case_insensitive_fallback():
    # docstring names the bound "Alpha" but the unpack line uses "alpha";
    # gecco's parse_bounds_from_docstring falls back case-insensitively
    # instead of silently defaulting to [0, 1].
    bounds = parse_bounds(CODE_MIXED_CASE, ["alpha", "beta"])
    assert bounds["alpha"] == (0.1, 0.9)
    assert bounds["beta"] == (0.0, 10.0)  # no docstring entry -> beta default
