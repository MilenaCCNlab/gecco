import numpy as np

from library_learning.compose.fitting import bic, fit_participant, seed_for

RNG_DATA = np.random.default_rng(0)


def biased_coin_model(action_1, state, action_2, reward, model_parameters):
    p, = model_parameters
    eps = 1e-10
    ll = np.where(action_1 == 1, np.log(p + eps), np.log(1 - p + eps))
    return -float(np.sum(ll))


def test_bic():
    assert abs(bic(100.0, 2, 200) - (np.log(200) * 2 + 200.0)) < 1e-12


def test_seed_deterministic():
    assert seed_for("cand", 14) == seed_for("cand", 14)
    assert seed_for("cand", 14) != seed_for("cand", 15)


def test_fit_recovers_bias():
    a1 = (RNG_DATA.random(500) < 0.8).astype(int)
    inputs = [a1, a1, a1, a1]
    res = fit_participant(biased_coin_model, inputs, [(0.001, 0.999)],
                          seed=seed_for("t", 0), n_starts=5)
    assert abs(res["params"][0] - a1.mean()) < 0.02
    res2 = fit_participant(biased_coin_model, inputs, [(0.001, 0.999)],
                           seed=seed_for("t", 0), n_starts=5)
    assert res == res2  # fully deterministic
