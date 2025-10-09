def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Validity-weighted additive integration with subjective cue weights, response bias, and lapse.
    
    Choice rule:
    - Compute a weighted value difference between options using expert validities (0.9,0.8,0.7,0.6)
      modulated by subjective cue weights.
    - Convert the difference to a choice probability via a logistic with inverse temperature.
    - Add a response bias toward B in logit space (simple shift).
    - Apply a lapse rate that mixes the model probability with random choice.
    
    Parameters (all used):
    - w1: [0,1] subjective weight for cue 1 (most valid; 0.9)
    - w2: [0,1] subjective weight for cue 2 (0.8)
    - w3: [0,1] subjective weight for cue 3 (0.7)
    - w4: [0,1] subjective weight for cue 4 (least valid; 0.6)
    - bias: [0,1] response bias toward choosing option B (0.5 = no bias; >0.5 favors B)
    - temperature: [0,10] inverse softmax temperature; higher -> more deterministic
    - lapse: [0,1] probability of a random (uniform) lapse on each trial
    
    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B, 0 = chose A)
    - A_feature1..A_feature4: arrays of 0/1 cue values for option A
    - B_feature1..B_feature4: arrays of 0/1 cue values for option B
    
    Returns:
    - negative log-likelihood of the observed decisions under the model
    """
    w1, w2, w3, w4, bias, temperature, lapse = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    # Subjective cue weights interact with objective validity; normalize to avoid arbitrary scale
    subj = np.array([w1, w2, w3, w4], dtype=float)
    eff_w = validities * (1e-6 + subj)
    eff_w = eff_w / (np.sum(eff_w) + 1e-12)

    # Prepare arrays
    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    d = (B - A)  # positive values favor B

    # Deterministic value difference
    delta = np.dot(d, eff_w)

    # Add bias in logit space as a shift in evidence
    bias_shift = (bias - 0.5) * 2.0  # in [-1,1]
    logits = temperature * delta + bias_shift

    # Logistic choice probability for choosing B
    pB = 1.0 / (1.0 + np.exp(-logits))

    # Lapse mixture
    pB = (1.0 - lapse) * pB + lapse * 0.5

    # Likelihood
    decisions = np.asarray(decisions).astype(float)
    p = decisions * pB + (1.0 - decisions) * (1.0 - pB)
    nll = -np.sum(np.log(np.clip(p, 1e-12, 1.0)))
    return nll


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Probabilistic Take-The-Best (TTB) with learned search priorities, stopping sharpness, bias, temperature, and lapse.
    
    Process:
    - Assign a priority to each cue via a softmax over attention parameters (modulated by validity).
    - On each trial, identify discriminating cues (where A != B).
    - The probability that a cue is the first inspected discriminating cue is proportional to its
      priority raised to a sharpness exponent (controls stopping/priority steepness).
    - The chosen option is determined by the direction of the first discriminating cue, passed
      through a logistic with temperature and bias. If no cue discriminates, choice is driven by bias.
    - Apply a lapse rate to mix with random choice.
    
    Parameters (all used):
    - a1: [0,1] attention strength for cue 1
    - a2: [0,1] attention strength for cue 2
    - a3: [0,1] attention strength for cue 3
    - a4: [0,1] attention strength for cue 4
    - phi: [0,1] stopping/priority sharpness; 0 ~ flat priorities, 1 ~ very steep to top cue
            (implemented as exponent s = 1 + 9*phi on priorities)
    - bias: [0,1] baseline bias toward choosing B (0.5 = neutral)
    - temperature: [0,10] inverse temperature in the logistic at the discriminating cue
    - lapse: [0,1] lapse probability mixing with random choice
    
    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B, 0 = chose A)
    - A_feature1..A_feature4: arrays of 0/1 cue values for option A
    - B_feature1..B_feature4: arrays of 0/1 cue values for option B
    
    Returns:
    - negative log-likelihood of the observed decisions under the model
    """
    a1, a2, a3, a4, phi, bias, temperature, lapse = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    # Priority via softmax over attention * validity (more valid cues generally higher priority)
    att = np.array([a1, a2, a3, a4], dtype=float)
    base_pri = att * validities
    # Softmax
    x = base_pri - np.max(base_pri)
    pri = np.exp(x)
    pri = pri / (np.sum(pri) + 1e-12)

    # Sharpness exponent for prioritizing the first discriminating cue
    s_exp = 1.0 + 9.0 * phi

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    decisions = np.asarray(decisions).astype(float)

    bias_shift = (bias - 0.5) * 2.0  # additive logit shift
    n_trials = len(decisions)
    pB_all = np.zeros(n_trials, dtype=float)

    for t in range(n_trials):
        a = A[t]
        b = B[t]
        discrim = (a != b)
        if not np.any(discrim):
            # No discriminating cues: rely on bias only
            pB = 1.0 / (1.0 + np.exp(-(bias_shift)))
        else:
            # Probabilities over which discriminating cue is encountered first
            pri_disc = pri[discrim]
            # Apply sharpness
            pri_disc_sharp = np.power(pri_disc + 1e-12, s_exp)
            pri_disc_sharp = pri_disc_sharp / (np.sum(pri_disc_sharp) + 1e-12)

            # Directions for discriminating cues in original indexing
            dirs = np.where(b[discrim] > a[discrim], 1.0, -1.0)  # +1 favors B, -1 favors A

            # Logistic decision at the first discriminating cue, mixed over which cue is first
            logits = temperature * dirs + bias_shift
            pB_cue = 1.0 / (1.0 + np.exp(-logits))
            pB = np.sum(pri_disc_sharp * pB_cue)

        pB_all[t] = pB

    # Lapse mixture
    pB_all = (1.0 - lapse) * pB_all + lapse * 0.5

    # Likelihood
    p = decisions * pB_all + (1.0 - decisions) * (1.0 - pB_all)
    nll = -np.sum(np.log(np.clip(p, 1e-12, 1.0)))
    return nll


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Hybrid reliability model: blends objective validities with subjective reliabilities,
    includes feature nonlinearity, bias, temperature, and lapse.
    
    Process:
    - Each cue's effective weight is a convex combination of objective validity and a subjective reliability:
        w_i = g * validity_i + (1 - g) * s_i
      where s_i are learned/reported subjective reliabilities.
    - Feature nonlinearity alpha warps 0/1 features toward or away from positivity:
        x' = alpha if original x=1, and x' = (1 - alpha) if x=0.
      Thus alpha>0.5 overweights positive cues; alpha<0.5 overweights negative cues.
    - Compute weighted additive value difference using warped features and effective weights.
    - Add response bias in logit space, transform via logistic with inverse temperature.
    - Apply lapse rate.
    
    Parameters (all used):
    - s1: [0,1] subjective reliability for cue 1
    - s2: [0,1] subjective reliability for cue 2
    - s3: [0,1] subjective reliability for cue 3
    - s4: [0,1] subjective reliability for cue 4
    - g: [0,1] blending parameter between objective validities (g=1) and subjective reliabilities (g=0)
    - alpha: [0,1] feature nonlinearity (0.5 = linear; >0.5 increases weight of positive cue values)
    - bias: [0,1] response bias toward choosing B (0.5 = neutral)
    - temperature: [0,10] inverse temperature for the logistic choice function
    - lapse: [0,1] lapse probability mixing with random choice
    
    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B, 0 = chose A)
    - A_feature1..A_feature4: arrays of 0/1 cue values for option A
    - B_feature1..B_feature4: arrays of 0/1 cue values for option B
    
    Returns:
    - negative log-likelihood of the observed decisions under the model
    """
    s1, s2, s3, s4, g, alpha, bias, temperature, lapse = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    subj_rel = np.array([s1, s2, s3, s4], dtype=float)

    # Effective weights blending validity and subjective reliability; normalize to unit sum
    eff_w = g * validities + (1.0 - g) * subj_rel
    eff_w = eff_w / (np.sum(eff_w) + 1e-12)

    # Warp feature values using alpha
    # Map 1 -> alpha; 0 -> (1 - alpha)
    def warp(x):
        return alpha * x + (1.0 - alpha) * (1.0 - x)

    A_raw = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B_raw = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    A = warp(A_raw)
    B = warp(B_raw)

    # Value difference
    delta = np.dot(B - A, eff_w)

    # Add bias in logit space
    bias_shift = (bias - 0.5) * 2.0
    logits = temperature * delta + bias_shift

    # Choice probability of B
    pB = 1.0 / (1.0 + np.exp(-logits))

    # Lapse mixture
    pB = (1.0 - lapse) * pB + lapse * 0.5

    # Likelihood
    decisions = np.asarray(decisions).astype(float)
    p = decisions * pB + (1.0 - decisions) * (1.0 - pB)
    nll = -np.sum(np.log(np.clip(p, 1e-12, 1.0)))
    return nll