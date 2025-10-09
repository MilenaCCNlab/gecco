def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Weighted-additive with trust-transformed validities, feature salience, bias, and lapse.
    Process:
    - Each expert’s stated validity (0.9,0.8,0.7,0.6) is transformed by a trust parameter (power-law).
    - Feature-specific salience parameters modulate how much each expert influences the value.
    - A global response bias shifts preference toward A vs. B, independent of features.
    - A logistic choice with inverse temperature converts value differences into choice probabilities.
    - A lapse parameter mixes decisions with random choice.

    Parameters (all used):
    salience1: [0,1] — Attention/salience for expert 1 (validity 0.9)
    salience2: [0,1] — Attention/salience for expert 2 (validity 0.8)
    salience3: [0,1] — Attention/salience for expert 3 (validity 0.7)
    salience4: [0,1] — Attention/salience for expert 4 (validity 0.6)
    trust:     [0,1] — Transforms validities via power: eff_v = validity^(1 + 4*trust)
    bias_A:    [0,1] — Response bias favoring A (0.5 = neutral; >0.5 favors A, <0.5 favors B)
    lapse:     [0,1] — Probability of random choice (mixed with 0.5)
    temperature:[0,10] — Inverse temperature for logistic choice (higher = more deterministic)

    Returns:
    Negative log-likelihood of observed choices (decisions coded as 1 for A, 0 for B).
    """
    sal1, sal2, sal3, sal4, trust, bias_A, lapse, temperature = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    eff_v = validities ** (1.0 + 4.0 * trust)
    sal = np.array([sal1, sal2, sal3, sal4], dtype=float)
    weights = eff_v * sal

    denom = np.sum(weights)
    if denom <= 0:

        weights = np.ones(4, dtype=float) / 4.0
    else:
        weights = weights / denom

    A = np.vstack([A_feature1, A_feature2, A_feature3, A_feature4]).T.astype(float)
    B = np.vstack([B_feature1, B_feature2, B_feature3, B_feature4]).T.astype(float)
    decisions = np.array(decisions, dtype=float)

    bias_shift = bias_A - 0.5  # in [-0.5, 0.5]
    d = (A - B) @ weights + bias_shift

    x = np.clip(temperature * d, -700, 700)  # numerical safety
    pA = 1.0 / (1.0 + np.exp(-x))

    pA = (1.0 - lapse) * pA + 0.5 * lapse

    eps = 1e-12
    pA = np.clip(pA, eps, 1.0 - eps)
    nll = -np.sum(decisions * np.log(pA) + (1.0 - decisions) * np.log(1.0 - pA))
    return nll