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

    # Trust-transformed validities (monotone with trust), then modulated by salience
    eff_v = validities ** (1.0 + 4.0 * trust)
    sal = np.array([sal1, sal2, sal3, sal4], dtype=float)
    weights = eff_v * sal
    # Normalize to sum to 1 (avoid zero-division)
    denom = np.sum(weights)
    if denom <= 0:
        # If all weights collapsed to zero, fall back to equal weighting
        weights = np.ones(4, dtype=float) / 4.0
    else:
        weights = weights / denom

    # Prepare arrays
    A = np.vstack([A_feature1, A_feature2, A_feature3, A_feature4]).T.astype(float)
    B = np.vstack([B_feature1, B_feature2, B_feature3, B_feature4]).T.astype(float)
    decisions = np.array(decisions, dtype=float)

    # Value difference (A - B) weighted, plus bias (centered around 0)
    bias_shift = bias_A - 0.5  # in [-0.5, 0.5]
    d = (A - B) @ weights + bias_shift

    # Logistic choice for P(choose A)
    x = np.clip(temperature * d, -700, 700)  # numerical safety
    pA = 1.0 / (1.0 + np.exp(-x))

    # Lapse mixture
    pA = (1.0 - lapse) * pA + 0.5 * lapse

    # Negative log-likelihood
    eps = 1e-12
    pA = np.clip(pA, eps, 1.0 - eps)
    nll = -np.sum(decisions * np.log(pA) + (1.0 - decisions) * np.log(1.0 - pA))
    return nll


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Probabilistic lexicographic search with noisy cue perception, mixture with compensatory fallback, bias, and lapse.
    Process:
    - With mixture weight m, choices follow a lexicographic (take-the-best-like) process scanning experts in validity order.
      Each cue is perceived with independent flip noise (epsilon). At the first discriminating cue, the process stops
      with probability s and chooses the option favored by that cue; otherwise it continues.
    - If no stop occurs after all cues, or with probability (1 - m), a compensatory fallback is used: weighted additive
      with trust-transformed validities and logistic choice.
    - A response bias toward A and a lapse component are applied.

    Parameters (all used):
    m:          [0,1]  — Mixture weight: probability of using lexicographic process (vs. fallback)
    s:          [0,1]  — Stop probability at the first discriminating cue in lexicographic scan
    epsilon:    [0,1]  — Perception noise (probability each cue is flipped)
    bias_A:     [0,1]  — Response bias favoring A (0.5 neutral)
    trust:      [0,1]  — Trust exponent for fallback weights: eff_v = validity^(1 + 4*trust)
    lapse:      [0,1]  — Lapse rate (mixture with random choice)
    temperature:[0,10] — Inverse temperature for fallback logistic choice

    Returns:
    Negative log-likelihood of observed choices (decisions coded as 1 for A, 0 for B).
    """
    m, s, epsilon, bias_A, trust, lapse, temperature = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    # Fallback compensatory weights
    eff_v = validities ** (1.0 + 4.0 * trust)
    weights = eff_v / np.sum(eff_v)

    A = np.vstack([A_feature1, A_feature2, A_feature3, A_feature4]).T.astype(float)
    B = np.vstack([B_feature1, B_feature2, B_feature3, B_feature4]).T.astype(float)
    decisions = np.array(decisions, dtype=float)

    # Precompute fallback P(A)
    bias_shift = bias_A - 0.5
    d_fallback = (A - B) @ weights + bias_shift
    x_fb = np.clip(temperature * d_fallback, -700, 700)
    pA_fallback = 1.0 / (1.0 + np.exp(-x_fb))

    # Lexicographic probabilities with noisy cue perception
    # For each cue i, compute P(A' = 1) and P(B' = 1) after flip noise
    # P(bit' = 1) = epsilon + (1 - 2*epsilon)*bit
    A_prob1 = epsilon + (1.0 - 2.0 * epsilon) * A  # shape (n_trials, 4)
    B_prob1 = epsilon + (1.0 - 2.0 * epsilon) * B
    A_prob0 = 1.0 - A_prob1
    B_prob0 = 1.0 - B_prob1

    # Probability cue i is discriminating after noise, and favors A or B
    pAwin = A_prob1 * B_prob0  # P(A'=1, B'=0)
    pBwin = A_prob0 * B_prob1  # P(A'=0, B'=1)
    pDisc = pAwin + pBwin      # P(discriminating)
    pSame = 1.0 - pDisc

    # Scan in validity order (columns already in descending validity order)
    n_trials = A.shape[0]
    pA_lexi = np.zeros(n_trials)
    continue_prob = np.ones(n_trials)

    for i in range(4):
        # Probability we stop at cue i and choose A (or B)
        stop_A_i = continue_prob * s * pAwin[:, i]
        stop_B_i = continue_prob * s * pBwin[:, i]
        pA_lexi += stop_A_i
        # Update probability we haven't stopped yet after cue i
        continue_prob = continue_prob * (1.0 - s * pDisc[:, i])

    # If we never stopped, fall back (within lexicographic branch)
    pA_lexi += continue_prob * pA_fallback

    # Mixture of lexicographic and fallback
    pA = m * pA_lexi + (1.0 - m) * pA_fallback

    # Apply bias (as an additional shift toward A) by blending toward biased choice
    # We implement bias by shifting probabilities toward A: p <- (1-bias_weight)*p + bias_weight*I(A)
    # Here bias_A in [0,1] with 0.5 neutral; map to bias_strength in [-0.5, 0.5], then convert to blend
    bias_strength = bias_A - 0.5
    if np.any(bias_strength != 0):
        # Convert to a mild linear push; cap effect to preserve bounds
        pA = np.clip(pA + bias_strength * 0.5, 0.0, 1.0)

    # Lapse
    pA = (1.0 - lapse) * pA + 0.5 * lapse

    # NLL
    eps = 1e-12
    pA = np.clip(pA, eps, 1.0 - eps)
    nll = -np.sum(decisions * np.log(pA) + (1.0 - decisions) * np.log(1.0 - pA))
    return nll


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Bayesian cue integration with subjective reliability scaling, prior bias, temperature, and lapse.
    Process:
    - Each expert i has stated validity v_i in {0.9,0.8,0.7,0.6}. Participants subjectively compress/expand
      these toward 0.5 using gamma_i: v_i' = 0.5 + gamma_i*(v_i - 0.5).
    - Assuming independent cues, the log-likelihood ratio favoring A over B equals
      sum_i log(v_i'/(1 - v_i')) * (A_i - B_i), plus a prior log-odds bias.
    - A logistic with inverse temperature converts this evidence into P(choose A).
    - A lapse component mixes in random choice.

    Parameters (all used):
    gamma1: [0,1] — Reliability scaling for expert 1 (0.9)
    gamma2: [0,1] — Reliability scaling for expert 2 (0.8)
    gamma3: [0,1] — Reliability scaling for expert 3 (0.7)
    gamma4: [0,1] — Reliability scaling for expert 4 (0.6)
    priorA: [0,1] — Prior bias toward A (0.5 neutral); converted to log-odds via logit
    lapse:  [0,1] — Lapse rate (mixture with random choice)
    temperature: [0,10] — Inverse temperature applied to total log-odds (decision noise)

    Returns:
    Negative log-likelihood of observed choices (decisions coded as 1 for A, 0 for B).
    """
    gamma1, gamma2, gamma3, gamma4, priorA, lapse, temperature = parameters
    base_v = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    gammas = np.array([gamma1, gamma2, gamma3, gamma4], dtype=float)
    # Subjective validities pulled toward 0.5 based on gamma
    subj_v = 0.5 + gammas * (base_v - 0.5)

    # Convert to log-odds weights
    # Clip to avoid log(0) / division by zero in extreme parameterizations
    eps = 1e-12
    subj_v = np.clip(subj_v, eps, 1.0 - eps)
    cue_weights = np.log(subj_v / (1.0 - subj_v))  # log-odds weight per cue

    # Prior log-odds from priorA
    priorA = np.clip(priorA, eps, 1.0 - eps)
    prior_logodds = np.log(priorA / (1.0 - priorA))

    A = np.vstack([A_feature1, A_feature2, A_feature3, A_feature4]).T.astype(float)
    B = np.vstack([B_feature1, B_feature2, B_feature3, B_feature4]).T.astype(float)
    decisions = np.array(decisions, dtype=float)

    # Total log-odds evidence: sum_i w_i*(A_i - B_i) + prior
    evidence = (A - B) @ cue_weights + prior_logodds

    # Temperature-scaled logistic for P(A)
    x = np.clip(temperature * evidence, -700, 700)
    pA = 1.0 / (1.0 + np.exp(-x))

    # Lapse
    pA = (1.0 - lapse) * pA + 0.5 * lapse

    # NLL
    pA = np.clip(pA, eps, 1.0 - eps)
    nll = -np.sum(decisions * np.log(pA) + (1.0 - decisions) * np.log(1.0 - pA))
    return nll