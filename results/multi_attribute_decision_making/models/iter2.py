def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Probabilistic serial/lexicographic integration with geometric decay, gating, and bias.

    Core idea:
    - Cues are prioritized by a preference for stated validities vs. equal weighting (focus).
    - Earlier (higher-priority) cues get geometrically larger influence (serial_decay).
    - Per-cue discriminability is “compressed” toward chance by cue_confidence (mistrust of cue validities).
    - Evidence blends a noncompensatory max component and a compensatory sum (gate).
    - A baseline side bias toward B is included (bias_B).
    - Evidence is mapped to choice via a logistic with inverse temperature and lapse.

    Parameters (bounds):
    - serial_decay in [0,1]: Geometric decay per rank position (0 = only top cue matters; 1 = no decay).
    - focus in [0,1]: Interpolation between equal cue priority (0) and stated validities (1) for ranking/priority.
    - cue_confidence in [0,1]: Shrinks cue validities toward 0.5; 0 = ignore validity (treat as 0.5), 1 = trust stated validities.
    - gate in [0,1]: Blend between compensatory sum (0) and relying on the single most weighted cue (1).
    - bias_B in [0,1]: Baseline bias toward choosing B (0.5 = no bias); mapped to [-1,1] in evidence space.
    - temperature in [0,10]: Inverse temperature for the logistic choice rule.
    - lapse in [0,1]: Lapse (random choice) probability mixed with the model’s prediction.

    Returns:
    - Negative log-likelihood of observed choices (decisions: 1 = choose B, 0 = choose A).
    """
    serial_decay, focus, cue_confidence, gate, bias_B, temperature, lapse = parameters

    # Assemble matrices
    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    decisions = np.asarray(decisions).astype(int)

    n_trials = A.shape[0]
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    # Priority scores interpolate between equal (1.0) and actual validities
    priority_scores = (1.0 - focus) * 1.0 + focus * validities

    # Create rank positions (higher priority gets lower rank index)
    # Ties cannot occur given monotonic validities; still handle generally
    rank_order = np.argsort(-priority_scores)  # indices sorted by descending priority
    ranks = np.empty_like(rank_order)
    ranks[rank_order] = np.arange(len(validities))  # 0,1,2,3 according to priority

    # Geometric serial discount by rank: weight per cue position
    serial_weights = (serial_decay ** ranks).astype(float)

    # Perceived cue validity (compress toward 0.5 by cue_confidence)
    perceived_validity = 0.5 + cue_confidence * (validities - 0.5)

    # Signed cue signal (+1 if B> A, -1 if A>B, 0 otherwise)
    disc = (B > A).astype(float) - (A > B).astype(float)  # shape (n_trials,4)

    # Per-cue weights combine serial priority and perceived validity
    w = serial_weights * (2.0 * perceived_validity - 1.0)  # map validity in [0.5,1] to [0,1]

    # Compensatory sum component
    sum_component = np.dot(disc, w)

    # Noncompensatory max component (most weighted discriminating cue)
    weighted_disc = disc * w[None, :]
    # If all zeros on a trial, max is 0; that’s fine
    max_component = np.max(np.abs(weighted_disc), axis=1) * np.sign(
        np.sum(weighted_disc * (np.abs(weighted_disc) == np.max(np.abs(weighted_disc), axis=1, keepdims=True)), axis=1)
    )
    # The above sign selection: if multiple ties, sums identical-signed maxima; if mixed, sign may cancel; acceptable heuristic.

    # Blend components
    evidence = (1.0 - gate) * sum_component + gate * max_component

    # Add baseline side bias toward B mapped to [-1,1]
    evidence += (bias_B - 0.5) * 2.0

    # Map to probability with temperature and lapse
    dv = np.clip(temperature, 0.0, 10.0) * evidence
    pB = 1.0 / (1.0 + np.exp(-np.clip(dv, -700, 700)))
    pB = lapse * 0.5 + (1.0 - lapse) * pB

    # Negative log-likelihood
    eps = 1e-12
    pB = np.clip(pB, eps, 1.0 - eps)
    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1.0 - pB)
    return -np.sum(ll)


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Bayesian cue integration with miscalibrated reliabilities and asymmetric utility.

    Core idea:
    - Treat each cue as a noisy expert providing evidence about whether B > A.
    - Convert cues to log-likelihood ratios (LLRs) from perceived reliabilities (calibration).
    - Combine with a prior bias toward B (prior_mean) with adjustable prior strength.
    - Allow asymmetric utility scaling of favoring B vs A (utility_asymmetry).
    - Map posterior evidence to choice with temperature and lapse.

    Parameters (bounds):
    - prior_mean in [0,1]: Prior preference toward B (0.5 = neutral), mapped to prior log-odds.
    - prior_strength in [0,1]: Weight of prior relative to cumulative cue evidence (0 = ignore prior; 1 = full prior).
    - calibration in [0,1]: Compresses/expands stated validities toward 0.5 (0 -> 0.5; 1 -> stated validity).
    - likelihood_sensitivity in [0,1]: Scales all cue LLRs (0 = ignore cues, 1 = full strength).
    - utility_asymmetry in [0,1]: Multiplicative scaling of evidence in favor of B; mapped to [0.5, 1.5].
    - temperature in [0,10]: Inverse temperature for logistic choice.
    - lapse in [0,1]: Lapse (random choice) probability mixed with model prediction.

    Returns:
    - Negative log-likelihood of observed choices (decisions: 1 = choose B, 0 = choose A).
    """
    prior_mean, prior_strength, calibration, likelihood_sensitivity, utility_asymmetry, temperature, lapse = parameters

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    decisions = np.asarray(decisions).astype(int)

    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    # Perceived reliabilities compressed toward 0.5 by calibration
    r = 0.5 + calibration * (validities - 0.5)  # in [0.5,1]
    # Log-likelihood ratio per cue if it favors B vs A
    # LLR = ln(r/(1-r)) for supportive cue; negative for opposing
    # Avoid log(0) via clipping
    r = np.clip(r, 1e-6, 1.0 - 1e-6)
    llr_per_cue = np.log(r / (1.0 - r))  # shape (4,)

    # Trial-wise signed evidence from cues
    disc = (B > A).astype(float) - (A > B).astype(float)  # -1,0,1
    cue_evidence = np.dot(disc, llr_per_cue)  # sum of LLRs

    # Apply global sensitivity to cue evidence
    cue_evidence *= likelihood_sensitivity

    # Prior log-odds for B vs A from prior_mean
    prior_mean = np.clip(prior_mean, 0.0, 1.0)
    prior_log_odds = np.log(np.clip(prior_mean, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - prior_mean, 1e-6, 1.0))
    # Mix prior strength
    combined_evidence = (1.0 - prior_strength) * cue_evidence + prior_strength * prior_log_odds

    # Asymmetric utility scaling favoring B (utility_asymmetry in [0.5,1.5])
    u = 0.5 + utility_asymmetry  # in [0.5, 1.5]
    combined_evidence *= u

    # Map to choice probability with temperature and lapse
    dv = np.clip(temperature, 0.0, 10.0) * combined_evidence
    pB = 1.0 / (1.0 + np.exp(-np.clip(dv, -700, 700)))
    pB = lapse * 0.5 + (1.0 - lapse) * pB

    eps = 1e-12
    pB = np.clip(pB, eps, 1.0 - eps)
    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1.0 - pB)
    return -np.sum(ll)


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Contextual contrast model with polarity bias, crowding penalty, and validity shift.

    Core idea:
    - Start with a weighted sum of cue directions using validities shifted toward equal weighting.
    - Apply polarity bias so cues supporting B and cues supporting A get different gains.
    - Apply a contrast nonlinearity to amplify strong consensus and compress weak evidence.
    - Penalize crowded evidence when many cues are discriminating (crowding).
    - Include an overall side bias toward B.
    - Map to choice via a logistic with temperature and lapse.

    Parameters (bounds):
    - weight_shift in [0,1]: Interpolates cue weights between stated validities (0) and equal weights (1).
    - contrast in [0,1]: Exponent controlling nonlinearity on evidence magnitude (1 + contrast as the power).
    - polarity in [0,1]: Bias multiplier for cues favoring B vs A; higher = more gain for B-supporting cues.
    - crowding in [0,1]: Strength of divisive normalization by the number of discriminating cues.
    - side_bias in [0,1]: Baseline bias toward B (0.5 = none), mapped to [-1,1] evidence.
    - temperature in [0,10]: Inverse temperature for logistic choice.
    - lapse in [0,1]: Lapse (random choice) probability mixed with model prediction.

    Returns:
    - Negative log-likelihood of observed choices (decisions: 1 = choose B, 0 = choose A).
    """
    weight_shift, contrast, polarity, crowding, side_bias, temperature, lapse = parameters

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    decisions = np.asarray(decisions).astype(int)

    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    # Interpolate weights between validities and equal weighting
    w = (1.0 - weight_shift) * validities + weight_shift * 1.0

    disc = (B > A).astype(float) - (A > B).astype(float)  # -1,0,1

    # Polarity gains: g_pos for B-supporting cues, g_neg for A-supporting cues (balanced around 1)
    g_pos = 0.5 + polarity         # [0.5, 1.5]
    g_neg = 1.5 - g_pos            # [1.0, 0.0] mirrored so total potential gain is balanced

    gains = (disc > 0).astype(float) * g_pos + (disc < 0).astype(float) * g_neg

    # Base evidence
    weighted_disc = disc * w[None, :] * gains
    base = np.sum(weighted_disc, axis=1)

    # Contrast nonlinearity on magnitude
    mag = np.abs(base)
    sgn = np.sign(base)
    amplified = sgn * (mag ** (1.0 + contrast))

    # Crowding penalty based on number of discriminating cues
    n_nonzero = np.sum(disc != 0.0, axis=1).astype(float)
    denom = 1.0 + crowding * n_nonzero
    evidence = amplified / denom

    # Add side bias mapped to [-1,1]
    evidence += (side_bias - 0.5) * 2.0

    # Logistic choice with temperature and lapse
    dv = np.clip(temperature, 0.0, 10.0) * evidence
    pB = 1.0 / (1.0 + np.exp(-np.clip(dv, -700, 700)))
    pB = lapse * 0.5 + (1.0 - lapse) * pB

    eps = 1e-12
    pB = np.clip(pB, eps, 1.0 - eps)
    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1.0 - pB)
    return -np.sum(ll)