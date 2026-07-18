# tests/test_compose_canonical_rlwm.py
import numpy as np

from library_learning.compose.canonical_rlwm import (CANONICAL_BOUNDS,
                                                     CANONICAL_SOURCE)
from library_learning.compose.render_rlwm import SMOKE_DATA
from library_learning.loading import bounds_for_code, exec_model


def test_source_execs_and_is_finite_on_missed_trials():
    func = exec_model(CANONICAL_SOURCE, "cognitive_model")
    params = [(lo + hi) / 2.0 for lo, hi in CANONICAL_BOUNDS]
    nll = float(func(SMOKE_DATA["stimulus"], SMOKE_DATA["actions"],
                     SMOKE_DATA["rewards"], SMOKE_DATA["blocks"],
                     SMOKE_DATA["set_sizes"], params))
    assert np.isfinite(nll) and nll > 0


def test_docstring_bounds_match_constant():
    assert bounds_for_code(CANONICAL_SOURCE) == CANONICAL_BOUNDS
    assert len(CANONICAL_BOUNDS) == 6
