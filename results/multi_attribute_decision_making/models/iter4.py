Below are three distinct cognitive models as standalone Python functions. Each returns the negative log-likelihood of the observed choices (decisions: 1 = choose B, 0 = choose A). All parameters are used meaningfully, and bounds are indicated in the docstrings.

def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Bayesian log-odds integration with per-cue calibration, diagnosticity control, and prior bias.

    Idea:
    - Each cue produces a log-likelihood ratio favoring A or B, based on its (calibrated) validity.
    - Per-cue calibration shrinks stated validities toward 0.5, allowing under/over-trust in any specific cue.
    - A global diagnosticity parameter scales all cue log-odds.
    - A prior bias (in log-odds) favors B vs A.
    - Evidence is transformed to choice probabilities via a logistic with inverse temperature and mixed with a lapse.

    Parameters (bounds):
    - calib1 in [0,1]: Shrinkage toward 0.5 for cue 1 (validity 0.9). 0 = ignore stated validity (0.5), 1 = trust fully.
    - calib2 in [0,1]: Shrinkage toward 0.5 for cue 2 (validity 0.8).
    - calib3 in [0,1]: Shrinkage toward 0.5 for cue 3 (validity 0.7).
    - calib4 in [0,1]: Shrinkage toward 0.5 for cue 4 (validity 0.6).
    - diagnosticity in [0,1]: Scales overall cue log-odds magnitude (0 = insensitive; 1 = full strength).
    - bias_B in [0,1]: Prior bias toward B, converted to log-odds via logit(bias_B). 0.5 is neutral.
    - temperature in [0,10]: Inverse temperature for the logistic choice rule (maps evidence to P(choose B)).
    - lapse in [0,1]: Lapse (random choice) probability mixed with model prediction.

    Returns:
    - Negative log-likelihood of observed decisions.
    """
    calib1, calib2, calib3, calib4, diagnosticity, bias_B, temperature, lapse = parameters

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    decisions = np.asarray(decisions).astype(int)

    stated_validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    calib = np.array([calib1, calib2, calib3, calib4], dtype=float)

    # Calibrate validities toward 0.5 (no information)
    v_eff = 0.5 + (stated_validities - 0.5) * calib
    v_eff = np.clip(v_eff, 1e-6, 1 - 1e-6)

    # Cue signals: +1 if B=1 & A=0; -1 if A=1 & B=0; 0 otherwise
    disc = (B > A).astype(float) - (A > B).astype(float)  # shape (n_trials, 4)

    # Per-cue log-likelihood ratios
    llr_per_cue = np.log(v_eff / (1.0 - v_eff))[None, :] * disc
    # Global diagnosticity scaling
    llr = diagnosticity * np.sum(llr_per_cue, axis=1)

    # Prior bias in log-odds
    bias_B = np.clip(bias_B, 1e-6, 1 - 1e-6)
    lo_prior = np.log(bias_B / (1.0 - bias_B))

    # Decision variable and choice probability
    dv = np.clip(temperature, 0.0, 10.0) * (llr + lo_prior)
    dv = np.clip(dv, -700, 700)
    pB = 1.0 / (1.0 + np.exp(-dv))

    # Lapse mixture
    pB = lapse * 0.5 + (1.0 - lapse) * pB
    pB = np.clip(pB, 1e-12, 1.0 - 1e-12)

    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1 - pB)
    return -np.sum(ll)


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Consensus-contradiction enhanced compensatory integration.

    Idea:
    - Start with a compensatory sum of signed cue differences weighted by a reliability-reliance scheme.
    - Add a consensus bonus when more cues align in the same direction.
    - Subtract a contradiction penalty proportional to the strength of the minority-opposed cues.
    - Emphasize the top-validity cue via top_emphasis.
    - Add a constant side bias toward B.
    - Map to choice via logistic with inverse temperature and lapse.

    Parameters (bounds):
    - reliance in [0,1]: Interpolates cue weights between equal-weight (0) and stated validities (1).
    - contradiction in [0,1]: Strength of penalty for conflicting (minority) cues.
    - consensus in [0,1]: Strength of bonus for majority alignment.
    - top_emphasis in [0,1]: Extra gain on cue 1 (validity 0.9); 0 = none, 1 = strong.
    - side_bias in [0,1]: Constant bias toward choosing B; 0.5 is neutral.
    - temperature in [0,10]: Inverse temperature for logistic choice.
    - lapse in [0,1]: Lapse (random choice) probability.

    Returns:
    - Negative log-likelihood of observed decisions.
    """
    reliance, contradiction, consensus, top_emphasis, side_bias, temperature, lapse = parameters

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    decisions = np.asarray(decisions).astype(int)

    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    # Interpolate weights between equal-weight and stated validities
    w = (1.0 - reliance) * np.ones_like(validities) + reliance * validities
    # Emphasize the top cue
    w = w.copy()
    w[0] *= (1.0 + 2.0 * top_emphasis)

    # Signed cue signal
    disc = (B > A).astype(float) - (A > B).astype(float)  # -1,0,+1 per cue

    # Base compensatory component
    base = np.dot(disc, w)

    # Compute majority/minority structure for enhancements
    weighted_disc = disc * w[None, :]
    pos = np.sum(np.maximum(weighted_disc, 0.0), axis=1)
    neg = np.sum(np.maximum(-weighted_disc, 0.0), axis=1)

    # Contradiction penalty: subtract minority mass
    conflict_mass = np.minimum(pos, neg)  # how much weight sits in the opposing minority
    penalty = -contradiction * conflict_mass

    # Consensus bonus: boost along the majority direction, scaled by agreement proportion
    majority_dir = np.sign(pos - neg)  # +1 favors B, -1 favors A, 0 tie
    total_mass = pos + neg + 1e-12
    agreement_strength = np.abs(pos - neg) / total_mass  # in [0,1]
    bonus = consensus * agreement_strength * majority_dir

    # Constant side bias toward B
    bias = (side_bias - 0.5)

    evidence = base + penalty + bonus + bias

    dv = np.clip(temperature, 0.0, 10.0) * evidence
    dv = np.clip(dv, -700, 700)
    pB = 1.0 / (1.0 + np.exp(-dv))

    pB = lapse * 0.5 + (1.0 - lapse) * pB
    pB = np.clip(pB, 1e-12, 1.0 - 1e-12)

    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1 - pB)
    return -np.sum(ll)


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Noisy one-reason sampler blended with compensatory evidence, with misperception and tie handling.

    Idea:
    - A "one-reason" component samples a single cue according to a noisy ordering favoring more valid cues,
      then uses that cue's direction as evidence (in expectation).
    - A compensatory component sums all cue signals weighted by validity.
    - Misperception flips cue comparisons with some probability, implemented as attenuation of signals.
    - A tie preference nudges decisions only when no cues discriminate.
    - A side bias adds a constant shift toward B.
    - Final evidence is a convex blend of one-reason and compensatory components, passed through a logistic with lapse.

    Parameters (bounds):
    - one_reason in [0,1]: Mixture weight on the one-reason sampler (1 = pure one-reason; 0 = pure compensatory).
    - ordering_noise in [0,1]: Controls how strongly cue selection favors high-validity cues
                               (0 = nearly uniform; 1 = very peaked toward high validity).
    - misperception in [0,1]: Probability of misperceiving a cue comparison; implemented as signal attenuation (1-2p).
    - tie_preference in [0,1]: Bias toward B applied only when all cues tie; 0.5 is neutral.
    - side_bias in [0,1]: Constant bias toward B on all trials; 0.5 is neutral.
    - temperature in [0,10]: Inverse temperature mapping evidence to choice probability.
    - lapse in [0,1]: Lapse (random choice) probability.

    Returns:
    - Negative log-likelihood of observed decisions.
    """
    one_reason, ordering_noise, misperception, tie_preference, side_bias, temperature, lapse = parameters

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    decisions = np.asarray(decisions).astype(int)

    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    # Signed cue signals per trial
    disc_raw = (B > A).astype(float) - (A > B).astype(float)  # -1,0,+1
    # Misperception attenuates the effective signal: E[sign * flip] = (1-2p)*sign
    atten = (1.0 - 2.0 * misperception)
    disc = atten * disc_raw

    # Weighting for compensatory component
    w = validities  # fixed by task

    # Compensatory evidence
    evidence_comp = np.dot(disc, w)

    # One-reason sampler: probability of sampling each cue based on noisy ordering
    # Softmax over validities with adjustable inverse temperature
    beta_order = 1e-6 + 10.0 * ordering_noise  # maps [0,1] to [~0,10]
    logits = beta_order * (validities - np.max(validities))  # stabilize
    exp_logits = np.exp(logits)
    p_cue = exp_logits / np.sum(exp_logits)

    # Expected one-reason evidence: expected weighted sign of sampled cue
    # Use validity as the cue's strength
    evidence_or = np.dot(disc, w * p_cue)

    # Blend components
    evidence = one_reason * evidence_or + (1.0 - one_reason) * evidence_comp

    # Tie handling: when all cues tie for a trial, add a tie preference shift
    all_tie = (np.sum(np.abs(disc_raw), axis=1) == 0).astype(float)
    tie_shift = (tie_preference - 0.5) * all_tie

    # Constant side bias
    bias_shift = (side_bias - 0.5)

    dv = np.clip(temperature, 0.0, 10.0) * (evidence + tie_shift + bias_shift)
    dv = np.clip(dv, -700, 700)
    pB = 1.0 / (1.0 + np.exp(-dv))

    # Lapse mixture
    pB = lapse * 0.5 + (1.0 - lapse) * pB
    pB = np.clip(pB, 1e-12, 1.0 - 1e-12)

    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1 - pB)
    return -np.sum(ll)