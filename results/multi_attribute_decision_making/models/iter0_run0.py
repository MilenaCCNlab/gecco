def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """
    Probabilistic Take-The-Best (PTTB) heuristic with lapse.
    The model inspects cues in order of expert validity (0.9, 0.8, 0.7, 0.6). If a cue discriminates,
    the decision is driven primarily by that cue's direction and its utilization strength. If no cue
    discriminates, the model falls back to an additive weighted difference. Choices are transformed
    by a logistic with inverse temperature, with a small lapse to allow stimulus-independent errors.

    Parameters (and bounds):
    - util1: [0,1] utilization strength for cue 1 (validity 0.9)
    - util2: [0,1] utilization strength for cue 2 (validity 0.8)
    - util3: [0,1] utilization strength for cue 3 (validity 0.7)
    - util4: [0,1] utilization strength for cue 4 (validity 0.6)
    - temperature: [0,10] inverse temperature (choice sensitivity); higher = more deterministic
    - lapse: [0,1] lapse/guessing rate that mixes 50/50 with the model choice

    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose option B, 0 = chose option A)
    - A_feature1..4, B_feature1..4: arrays of 0/1 expert ratings per trial

    Returns:
    - negative log-likelihood of observed choices under the model
    """
    util1, util2, util3, util4, temperature, lapse = parameters
    n_trials = len(decisions)
    validities = [0.9, 0.8, 0.7, 0.6]

    log_likelihood = 0.0
    eps = 1e-12

    for t in range(n_trials):
        A = [A_feature1[t], A_feature2[t], A_feature3[t], A_feature4[t]]
        B = [B_feature1[t], B_feature2[t], B_feature3[t], B_feature4[t]]
        utils = [util1, util2, util3, util4]

        # Identify first discriminating cue in validity order
        score = 0.0
        first_found = False
        for i in range(4):
            d = B[i] - A[i]  # -1, 0, or 1
            if d != 0:
                # Use sign of the first discriminating cue scaled by its utilization
                score = utils[i] * (1.0 if d > 0 else -1.0)
                first_found = True
                break

        if not first_found:
            # Fall back to additive weighted difference using the same utilization weights
            score = 0.0
            for i in range(4):
                score += utils[i] * (B[i] - A[i])

        # Logistic choice with lapse
        z = temperature * score
        pB = 1.0 / (1.0 + np.exp(-z))
        pB = (1.0 - lapse) * pB + 0.5 * lapse

        # Clamp for numerical stability
        pB = min(max(pB, eps), 1.0 - eps)
        if decisions[t] == 1:
            log_likelihood += np.log(pB)
        else:
            log_likelihood += np.log(1.0 - pB)

    return -log_likelihood


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """
    Validity-weighted additive integration with expert trust and lapse.
    Each cue contributes additively to the subjective value difference, weighted by the cue's expert validity
    raised to an exponent and by a cue-specific trust parameter. The resultant value difference is passed
    through a logistic with inverse temperature, with a lapse component.

    Parameters (and bounds):
    - trust1: [0,1] trust in cue 1 (multiplier for validity 0.9)
    - trust2: [0,1] trust in cue 2 (multiplier for validity 0.8)
    - trust3: [0,1] trust in cue 3 (multiplier for validity 0.7)
    - trust4: [0,1] trust in cue 4 (multiplier for validity 0.6)
    - alpha: [0,1] exponent shaping sensitivity to validity (effective weight ~ validity^alpha)
    - temperature: [0,10] inverse temperature (choice sensitivity)
    - lapse: [0,1] lapse/guessing rate

    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose option B, 0 = chose option A)
    - A_feature1..4, B_feature1..4: arrays of 0/1 expert ratings per trial

    Returns:
    - negative log-likelihood of observed choices under the model
    """
    trust1, trust2, trust3, trust4, alpha, temperature, lapse = parameters
    n_trials = len(decisions)
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    trusts = np.array([trust1, trust2, trust3, trust4], dtype=float)
    # Effective weights combine validity sensitivity and trust
    eff_weights = trusts * (validities ** (alpha + 1e-9))  # add tiny to avoid 0**0 edge cases

    log_likelihood = 0.0
    eps = 1e-12

    for t in range(n_trials):
        A = np.array([A_feature1[t], A_feature2[t], A_feature3[t], A_feature4[t]], dtype=float)
        B = np.array([B_feature1[t], B_feature2[t], B_feature3[t], B_feature4[t]], dtype=float)

        dv = np.sum(eff_weights * (B - A))
        z = temperature * dv
        pB = 1.0 / (1.0 + np.exp(-z))
        pB = (1.0 - lapse) * pB + 0.5 * lapse

        pB = min(max(pB, eps), 1.0 - eps)
        if decisions[t] == 1:
            log_likelihood += np.log(pB)
        else:
            log_likelihood += np.log(1.0 - pB)

    return -log_likelihood


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """
    Robust majority integration with perceptual noise, validity-attention mixture, nonlinearity, and confirmation bias.
    The model first applies feature noise (probability of misperceiving a cue), then computes a weighted difference
    combining expert validities with equal-attention mixing. Differences are passed through a nonlinear transform,
    and a confirmation bias pushes the choice toward the option favored by the most valid cue. Logistic choice with lapse.

    Parameters (and bounds):
    - feature_noise: [0,1] probability that a cue is internally flipped (0 -> 1, 1 -> 0); implemented in expectation
    - mix_validity: [0,1] mixture between equal attention (0) and validity-based attention (1)
    - nonlinearity: [0,1] exponent controlling superlinearity of differences (applied as |x|^(1+nonlinearity) with sign)
    - confirm_bias: [0,1] additive bias toward the option favored by cue 1 (most valid)
    - temperature: [0,10] inverse temperature (choice sensitivity)
    - lapse: [0,1] lapse/guessing rate

    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose option B, 0 = chose option A)
    - A_feature1..4, B_feature1..4: arrays of 0/1 expert ratings per trial

    Returns:
    - negative log-likelihood of observed choices under the model
    """
    feature_noise, mix_validity, nonlinearity, confirm_bias, temperature, lapse = parameters
    n_trials = len(decisions)
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    equal_w = np.ones_like(validities) / 4.0
    # Attention weights as convex combination of equal and validity-based attention, normalized
    attn = mix_validity * validities + (1.0 - mix_validity) * equal_w
    attn = attn / np.sum(attn)

    log_likelihood = 0.0
    eps = 1e-12

    # Helper: expected value of a bit after symmetric flip with prob feature_noise
    # E[x'] = (1 - 2*noise) * x + noise
    flip_scale = (1.0 - 2.0 * feature_noise)
    flip_bias = feature_noise

    for t in range(n_trials):
        A = np.array([A_feature1[t], A_feature2[t], A_feature3[t], A_feature4[t]], dtype=float)
        B = np.array([B_feature1[t], B_feature2[t], B_feature3[t], B_feature4[t]], dtype=float)

        # Expected internal representation under feature noise
        A_int = flip_scale * A + flip_bias
        B_int = flip_scale * B + flip_bias

        # Weighted cue differences
        diff = (B_int - A_int)  # can be in [-1,1]
        # Apply nonlinear sensitivity with preserved sign
        sign = np.sign(diff)
        mag = np.abs(diff) ** (1.0 + nonlinearity)
        transformed = sign * mag

        dv = np.sum(attn * transformed)

        # Confirmation bias based on most valid cue's (noisy) direction
        cue1_dir = np.sign(B_int[0] - A_int[0])
        dv += confirm_bias * cue1_dir

        z = temperature * dv
        pB = 1.0 / (1.0 + np.exp(-z))
        pB = (1.0 - lapse) * pB + 0.5 * lapse

        pB = min(max(pB, eps), 1.0 - eps)
        if decisions[t] == 1:
            log_likelihood += np.log(pB)
        else:
            log_likelihood += np.log(1.0 - pB)

    return -log_likelihood