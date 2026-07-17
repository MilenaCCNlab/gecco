import numpy as np

from library_learning.compose.fitting import bic, fit_model_on_pids
from library_learning.compose.hybrid import HYBRID_BOUNDS, HYBRID_SOURCE
from library_learning.compose.render import smoke_check
from library_learning.config import resolve_target
from library_learning.loading import extract_unpack_names

IND = "results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual"


def test_hybrid_source_shape():
    assert extract_unpack_names(HYBRID_SOURCE) == [
        "learning_rate", "learning_rate_2", "beta", "beta_2", "w", "lambd", "perseveration"]
    assert len(HYBRID_BOUNDS) == 7
    assert np.isfinite(smoke_check(HYBRID_SOURCE, params=[0.5, 0.5, 2.0, 2.0, 0.5, 0.5, 0.5]))


def test_hybrid_fits_close_to_stored_baseline():
    target = resolve_target(IND)
    import pandas as pd
    df = pd.read_csv(target.data_path)
    stored = float(df[df.participant == 14].baseline_bic.iloc[0])
    fits = fit_model_on_pids(HYBRID_SOURCE, target, [14], HYBRID_BOUNDS, tag="hybrid-test")
    # informational cross-check: same model family, different optimizer runs
    assert abs(fits[14]["bic"] - stored) < 25.0
    print("refit %.2f vs stored %.2f" % (fits[14]["bic"], stored))
