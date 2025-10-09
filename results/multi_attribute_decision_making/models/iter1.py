def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Lexicographic-stopping mixture: random single-cue vs. compensatory integration.

    Rationale:
    - On each trial, with probability stop_prob the decision-maker inspects only a single cue
      (sampled from an attention distribution over cue ranks) and decides based on that cue.
    - Otherwise, they integrate all cues compensatorily with attention-modulated, validity-scaled weights.
    - A trust parameter compresses/expands the given validities (0.9, 0.8, 0.7, 0.6).
    - A response bias toward B shifts the decision in logit space.
    - Choices are passed through a logistic with inverse temperature, and a lapse rate mixes in random choice.

    Parameters (all used):
    - att1: [0,1] attention propensity for cue 1 (most valid; influences both single-cue and compensatory weights)
    - att2: [0,1] attention propensity for cue 2
    - att3: [0,1] attention propensity for cue 3
    - att4: [0,1] attention propensity for cue 4 (least valid)
    - trust: [0,1] compress/expand validities; effective v' = 0.5 + (v - 0.5)^(trust + 1e-6)
             (smaller -> more extreme weighting of validities; larger -> more linear)
    - stop_prob: [0,1] probability of using a single, attended cue instead of full integration
    - biasB: [0,1] response bias toward choosing B (0.5 = unbiased; >0.5 favors B)
    - temperature: [0,10] inverse softmax temperature; higher -> more deterministic mapping from evidence to choice
    - lapse: [0,1] probability of random (uniform) lapse on each trial

    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B, 0 = chose A)
    - A_feature1..A_feature4: arrays of 0/1 cue values for option A
    - B_feature1..B_feature4: arrays of 0/1 cue values for option B

    Returns:
    - negative log-likelihood of the observed decisions under the model
    """
    att1, att2, att3, att4, trust, stop_prob, biasB, temperature, lapse = parameters

    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    # Trust-modulated validities: power around 0.5
    eps = 1e-8
    v_eff = 0.5 + np.power(validities - 0.5, trust + eps)

    # Attention propensities to probabilities
    att = np.array([att1, att2, att3, att4], dtype=float) + eps
    att = att / np.sum(att)

    # Trial-wise differences per cue (positive favors B)
    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    D = B - A  # shape (n_trials, 4); entries in {-1,0,1}

    # Single-cue route: expected evidence is the attention-weighted signed difference times validity
    # Only discriminating cues contribute (nonzero D)
    single_cue_evidence = np.sum(att[None, :] * v_eff[None, :] * D, axis=1)

    # Compensatory route: attention-weighted, trust-modulated validity weights
    w_comp = att * v_eff
    w_comp = w_comp / (np.sum(w_comp) + eps)
    compensatory_evidence = np.dot(D, w_comp)

    # Mixture of routes
    delta = stop_prob * single_cue_evidence + (1.0 - stop_prob) * compensatory_evidence

    # Add bias in logit space
    bias_shift = (biasB - 0.5) * 2.0  # in [-1,1]

    logits = temperature * delta + bias_shift
    pB = 1.0 / (1.0 + np.exp(-logits))

    # Lapse
    pB = (1.0 - lapse) * pB + lapse * 0.5

    # Negative log-likelihood
    decisions = np.asarray(decisions).astype(float)
    p = decisions * pB + (1.0 - decisions) * (1.0 - pB)
    nll = -np.sum(np.log(np.clip(p, 1e-12, 1.0)))
    return nll


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Asymmetric gain–loss integration with attentional gradient and zero-penalty reference.

    Rationale:
    - Each cue comparison is treated as a gain (+) for B if B=1 and A=0, or a loss (-) if B=0 and A=1.
    - Losses are over-weighted by a loss-aversion parameter.
    - Cue weights combine stated validities transformed by a nonlinearity and an attentional gradient
      across ranks (emphasize earlier or later cues).
    - A "reference sensitivity" penalizes options with more zeros (interpreting 0 as a shortfall).
    - Response bias, logistic temperature, and lapse complete the choice rule.

    Parameters (all used):
    - alpha: [0,1] nonlinearity on validity; eff v_k = 0.5 + (v_k - 0.5)^(alpha + 1e-6)
              (smaller alpha -> stronger separation of high vs. low validities)
    - grad: [0,1] attentional gradient; ratio r = 0.5 + grad in [0.5, 1.5]; weight multiplier r^(rank-1)
            (grad < 0.5 emphasizes earlier cues; >0.5 emphasizes later cues)
    - loss_aversion: [0,1] transforms to lambda = 1 + loss_aversion in [1,2] (extra weight on losses)
    - ref_sensitivity: [0,1] penalty strength for zeros in B relative to A (reference-based shortfall)
    - biasB: [0,1] response bias toward B (0.5 = none; >0.5 favors B)
    - temperature: [0,10] inverse softmax temperature; higher -> more deterministic
    - lapse: [0,1] probability of random (uniform) lapse

    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B, 0 = chose A)
    - A_feature1..A_feature4: arrays of 0/1 cue values for option A
    - B_feature1..B_feature4: arrays of 0/1 cue values for option B

    Returns:
    - negative log-likelihood of the observed decisions under the model
    """
    alpha, grad, loss_aversion, ref_sensitivity, biasB, temperature, lapse = parameters

    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    eps = 1e-8

    # Validity nonlinearity
    v_eff = 0.5 + np.power(validities - 0.5, alpha + eps)

    # Attentional gradient across ranks (rank order: 1..4 = most to least valid)
    r = 0.5 + grad  # in [0.5, 1.5]
    ranks = np.array([0, 1, 2, 3], dtype=float)
    g = np.power(r, ranks)

    # Combine into normalized weights
    w = v_eff * g
    w = w / (np.sum(w) + eps)

    # Trial arrays
    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)

    # Gain-loss coding per cue for B relative to A
    # gain_k = 1 if (B=1, A=0), loss_k = 1 if (B=0, A=1)
    gain = np.clip(B - A, 0.0, 1.0)  # 1 where B=1 & A=0
    loss = np.clip(A - B, 0.0, 1.0)  # 1 where B=0 & A=1

    lam = 1.0 + loss_aversion  # in [1,2]
    cue_contrib = np.dot(gain - lam * loss, w)  # positive favors B

    # Reference (zero) penalty: penalize B for having more zeros than A
    zeros_A = np.sum(1.0 - A, axis=1)
    zeros_B = np.sum(1.0 - B, axis=1)
    ref_term = ref_sensitivity * (zeros_A - zeros_B)  # positive favors B if A has more zeros

    delta = cue_contrib + ref_term

    # Bias and logistic mapping
    bias_shift = (biasB - 0.5) * 2.0
    logits = temperature * delta + bias_shift
    pB = 1.0 / (1.0 + np.exp(-logits))

    # Lapse
    pB = (1.0 - lapse) * pB + lapse * 0.5

    decisions = np.asarray(decisions).astype(float)
    p = decisions * pB + (1.0 - decisions) * (1.0 - pB)
    nll = -np.sum(np.log(np.clip(p, 1e-12, 1.0)))
    return nll


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Random-attention single-cue choice with calibrated cue reliability.

    Rationale:
    - On each trial, the decision-maker attends a single cue drawn from an attention distribution
      over the four ranks and decides based on that cue alone.
    - The discriminability contributed by the attended cue depends on a calibration of its stated validity.
    - If the attended cue does not discriminate (A_k == B_k), the decision reduces to the bias/noise rule.
    - A response bias toward B shifts the logit; temperature controls determinism; lapse mixes in random choice.

    Parameters (all used):
    - q1: [0,1] attention propensity for cue 1 (most valid)
    - q2: [0,1] attention propensity for cue 2
    - q3: [0,1] attention propensity for cue 3
    - q4: [0,1] attention propensity for cue 4 (least valid)
    - calib: [0,1] validity calibration; strength s_k = ((v_k - 0.5) / 0.5)^(calib + 1e-6)
             (smaller -> more extreme reliance on higher validities)
    - biasB: [0,1] response bias toward B (0.5 = none; >0.5 favors B)
    - temperature: [0,10] inverse softmax temperature for the single-cue logistic choice
    - lapse: [0,1] probability of random (uniform) lapse

    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B, 0 = chose A)
    - A_feature1..A_feature4: arrays of 0/1 cue values for option A
    - B_feature1..B_feature4: arrays of 0/1 cue values for option B

    Returns:
    - negative log-likelihood of the observed decisions under the model
    """
    q1, q2, q3, q4, calib, biasB, temperature, lapse = parameters

    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    eps = 1e-8

    # Attention distribution over cues
    q = np.array([q1, q2, q3, q4], dtype=float) + eps
    att = q / np.sum(q)

    # Calibrated discriminability per cue (scaled around 0)
    # base strength in (0,1]: (v-0.5)/0.5 in (0,1); raise to calib -> in (0,1]
    base = (validities - 0.5) / 0.5
    strength = np.power(base, calib + eps)  # higher for more valid cues; in (0,1]

    # Data arrays
    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    D = B - A  # in {-1,0,1}

    # For each trial, compute the attention-weighted probability of choosing B
    # given potentially non-discriminating attended cues.
    bias_shift = (biasB - 0.5) * 2.0  # in [-1,1]

    # Precompute per-cue logits for d in {-1,0,1}
    # If non-discriminating (d=0), only bias applies.
    # If discriminating, contribution is +/- strength_k depending on sign.
    logits_pos = temperature * (strength + 0.0) + bias_shift        # when D=+1
    logits_neg = temperature * (-strength + 0.0) + bias_shift       # when D=-1
    logits_zero = np.full(4, bias_shift, dtype=float)               # when D=0

    pB_pos = 1.0 / (1.0 + np.exp(-logits_pos))
    pB_neg = 1.0 / (1.0 + np.exp(-logits_neg))
    pB_zero = 1.0 / (1.0 + np.exp(-logits_zero))

    # Map each trial's D to per-cue pB and average under attention
    # Build masks
    D_pos = (D > 0).astype(float)
    D_neg = (D < 0).astype(float)
    D_zero = (D == 0).astype(float)

    # Attention-weighted mixture per trial
    # pB_trial = sum_k att_k * pB_k(D_k)
    pB = (
        np.dot(D_pos, att * pB_pos) +
        np.dot(D_neg, att * pB_neg) +
        np.dot(D_zero, att * pB_zero)
    )

    # Lapse
    pB = (1.0 - lapse) * pB + lapse * 0.5

    # Negative log-likelihood
    decisions = np.asarray(decisions).astype(float)
    p = decisions * pB + (1.0 - decisions) * (1.0 - pB)
    nll = -np.sum(np.log(np.clip(p, 1e-12, 1.0)))
    return nll