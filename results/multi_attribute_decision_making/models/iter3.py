def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Probabilistic Take-The-Best with stopping, serial decay, and confirmation reweighting.

    Core idea:
    - Cues (experts) have given validities v = [0.9, 0.8, 0.7, 0.6].
    - Effective cue trust is shaped by two parameters (trust_scale, trust_shift).
      This maps stated validities toward 0.5 (agnostic) or accentuates stronger cues.
    - The model blends a pure Take-The-Best (only the strongest discriminating cue) with
      a multi-cue sum that decays across the validity order (serial_decay).
    - A confirmation parameter amplifies cues that agree with the aggregate sign and attenuates
      those that oppose it (capturing confirmatory use of information).
    - Evidence is mapped to P(B) via a logistic with inverse temperature; lapse mixes in random choice.

    Parameters (bounds):
    - trust_scale in [0,1]: Interpolates between agnostic 0.5 (0) and stated validities (1).
    - trust_shift in [0,1]: Curvature on effective validity (0 = linear, 1 = square to accentuate high).
    - stopping in [0,1]: Weight on the first (strongest) discriminating cue vs. the decayed multi-cue sum.
                         1 = pure TTB; 0 = pure decayed sum across all discriminating cues.
    - serial_decay in [0,1]: Geometric decay across the validity order for the decayed multi-cue sum.
    - confirmation in [0,1]: Reweights cues that agree with the aggregate sign (+) vs. disagree (-).
    - temperature in [0,10]: Inverse temperature for the logistic choice rule.
    - lapse in [0,1]: Lapse (random choice) probability mixed with the model’s prediction.

    Returns:
    - Negative log-likelihood of observed choices (decisions: 1 = choose B, 0 = choose A).
    """
    trust_scale, trust_shift, stopping, serial_decay, confirmation, temperature, lapse = parameters

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    decisions = np.asarray(decisions).astype(int)

    # Stated validities in descending order
    v = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    # Effective validities shaped by trust_scale and trust_shift
    # Pull toward 0.5 when trust_scale small; accentuate strong cues with trust_shift via squaring.
    v_lin = (1.0 - trust_scale) * 0.5 + trust_scale * v
    v_eff = (1.0 - trust_shift) * v_lin + trust_shift * (v_lin ** 2)

    # Discrimination signal per cue: +1 if B positive and A negative, -1 if A pos and B neg, else 0
    disc = (B > A).astype(float) - (A > B).astype(float)  # shape (n_trials, 4)

    # Sort cues by effectiveness (descending), but keep stable original columns
    order = np.argsort(-v_eff)
    v_sorted = v_eff[order]
    disc_sorted = disc[:, order]

    # Pure TTB component: take the first non-zero discriminating cue (if any)
    # If none discriminate, TTB evidence is 0.
    first_nonzero_idx = np.argmax(np.abs(disc_sorted) > 0, axis=1)  # index of first nonzero; zero if none
    has_disc = np.any(np.abs(disc_sorted) > 0, axis=1).astype(float)
    ttb_sign = np.take_along_axis(disc_sorted, first_nonzero_idx[:, None], axis=1).squeeze()
    ttb_weight = np.take_along_axis(v_sorted[None, :], first_nonzero_idx[:, None], axis=1).squeeze()
    ttb_evidence = has_disc * (ttb_sign * ttb_weight)

    # Decayed multi-cue sum over discriminating cues
    positions = np.arange(4, dtype=float)  # 0 for strongest, 3 for weakest (after sorting)
    pos_decay = (serial_decay ** positions)[None, :]  # shape (1,4)
    decayed_weights = v_sorted[None, :] * pos_decay
    multi_sum = np.sum(disc_sorted * decayed_weights, axis=1)

    # Pre-confirmation aggregate sign from the blended (for defining agreement)
    pre_blend = stopping * ttb_evidence + (1.0 - stopping) * multi_sum
    aggregate_sign = np.sign(pre_blend)

    # Confirmation reweight: cues agreeing with aggregate_sign get boosted; opposing are dampened
    agree = np.sign(disc_sorted) == aggregate_sign[:, None]
    disagree = np.sign(disc_sorted) == -aggregate_sign[:, None]
    # Neutral (zero) cues are unaffected
    conf_factor = np.ones_like(disc_sorted)
    conf_factor[agree] *= (1.0 + confirmation)
    conf_factor[disagree] *= (1.0 - confirmation)

    # Recompute the decayed sum with confirmation
    multi_sum_conf = np.sum(disc_sorted * decayed_weights * conf_factor, axis=1)

    # Final blended evidence
    evidence = stopping * ttb_evidence + (1.0 - stopping) * multi_sum_conf

    # Map to choice probability
    dv = np.clip(temperature, 0.0, 10.0) * evidence
    pB = 1.0 / (1.0 + np.exp(-np.clip(dv, -700, 700)))
    pB = lapse * 0.5 + (1.0 - lapse) * pB

    # Negative log-likelihood
    eps = 1e-12
    pB = np.clip(pB, eps, 1.0 - eps)
    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1 - pB)
    return -np.sum(ll)


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Leaky urgency-weighted evidence integration with asymmetric sensitivity.

    Core idea:
    - Integration approximates a drift process where discriminating cues add evidence,
      with stronger amplification for high-validity cues vs. lower-validity cues.
    - A leaky nonlinearity compresses very large absolute evidence (self-normalization).
    - Urgency increases the impact when more cues discriminate (parallel to collapsing bounds).
    - Asymmetric sensitivity allows different gain for evidence in favor of B vs. A.

    Parameters (bounds):
    - gain_high in [0,1]: Weight multiplier for high-validity cues (v >= 0.75).
    - gain_low in [0,1]: Weight multiplier for low-validity cues (v < 0.75).
    - leak in [0,1]: Leak/compression; higher leak compresses large |evidence| (0 = none, 1 = strong).
    - urgency in [0,1]: Scales with the fraction of discriminating cues to emphasize decisive trials.
    - asymmetry in [0,1]: Relative sensitivity for evidence sign (+ favors B) vs. (− favors A).
                           Effective gain for positive evidence is 1 + asymmetry, for negative is 1 - asymmetry.
    - temperature in [0,10]: Inverse temperature for the logistic choice rule.
    - lapse in [0,1]: Lapse (random choice) probability mixed with the model’s prediction.

    Returns:
    - Negative log-likelihood of observed choices (decisions: 1 = choose B, 0 = choose A).
    """
    gain_high, gain_low, leak, urgency, asymmetry, temperature, lapse = parameters

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    decisions = np.asarray(decisions).astype(int)

    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    high_mask = (validities >= 0.75).astype(float)
    low_mask = 1.0 - high_mask
    cue_gain = gain_high * high_mask + gain_low * low_mask  # per-cue gain

    # Signed discrimination per cue
    disc = (B > A).astype(float) - (A > B).astype(float)  # shape (n_trials, 4)

    # Base evidence from discriminating cues with cue-specific gain
    base = np.dot(disc, cue_gain)

    # Urgency factor: proportional to the fraction of discriminating cues on the trial
    n_disc = np.sum(np.abs(disc) > 0, axis=1).astype(float)
    frac_disc = n_disc / 4.0
    urg_factor = 1.0 + urgency * frac_disc  # in [1, 2] depending on urgency and discriminability
    base *= urg_factor

    # Asymmetric sensitivity: amplify evidence toward B, attenuate toward A
    pos = base > 0
    neg = base < 0
    base_adj = np.zeros_like(base)
    base_adj[pos] = base[pos] * (1.0 + asymmetry)
    base_adj[neg] = base[neg] * (1.0 - asymmetry)
    # zero stays zero

    # Leaky compression to avoid unrealistically large magnitudes
    # Map x -> x / (1 + leak * |x|), with leak=0 = identity; leak=1 => strong compression
    evidence = base_adj / (1.0 + leak * np.abs(base_adj))

    # Choice probability
    dv = np.clip(temperature, 0.0, 10.0) * evidence
    pB = 1.0 / (1.0 + np.exp(-np.clip(dv, -700, 700)))
    pB = lapse * 0.5 + (1.0 - lapse) * pB

    # Negative log-likelihood
    eps = 1e-12
    pB = np.clip(pB, eps, 1.0 - eps)
    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1 - pB)
    return -np.sum(ll)


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Stochastic cue sampling with lexicographic-vs-compensatory blending and contrast nonlinearity.

    Core idea:
    - Cues are sampled with probabilities shaped by their stated validities using a soft weighting
      controlled by sample_gain and a uniform mixing via sample_bias.
    - A lexicographic process: sample cues without replacement in a probabilistic order until a
      discriminating cue is found. With probability stop_prob, decide based on that cue.
    - Otherwise, fall back on a compensatory scheme that uses a contrast nonlinearity on cue signals.
    - A mixture of the two (noncomp_weight) yields the final evidence. tie_bias adds a constant bias toward B.
    - Evidence is mapped via logistic with inverse temperature and lapse.

    Parameters (bounds):
    - sample_gain in [0,1]: Sharpness of sampling preference for higher-validity cues (0 ~ uniform, 1 ~ proportional to validity).
    - sample_bias in [0,1]: Mixes the sampling distribution toward uniform (1 = fully uniform).
    - stop_prob in [0,1]: Probability of stopping at the first discriminating cue (lexicographic use).
    - noncomp_weight in [0,1]: Weight on the lexicographic component (1) vs. compensatory contrast (0) in final blend.
    - contrast_power in [0,1]: Nonlinearity on compensatory contrast; raises absolute cue signal to this power (0 = sign only).
    - tie_bias in [0,1]: Additive bias toward B in evidence (centered so 0.5 means no bias; >0.5 favors B).
    - temperature in [0,10]: Inverse temperature for the logistic choice rule.
    - lapse in [0,1]: Lapse (random choice) probability mixed with the model’s prediction.

    Returns:
    - Negative log-likelihood of observed choices (decisions: 1 = choose B, 0 = choose A).
    """
    sample_gain, sample_bias, stop_prob, noncomp_weight, contrast_power, tie_bias, temperature, lapse = parameters

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    decisions = np.asarray(decisions).astype(int)

    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    # Sampling probabilities over cues: soften validities via sample_gain and mix toward uniform via sample_bias
    # Base proportional to validity^gamma where gamma in [0,1] moves toward uniform as it approaches 0.
    gamma = 1e-8 + sample_gain  # avoid zero exactly
    base = validities ** gamma
    base /= np.sum(base)
    p_sample = (1.0 - sample_bias) * base + sample_bias * 0.25  # convex mixture with uniform

    # Precompute a deterministic approximation of the lexicographic decision using expected-first-discriminator heuristic:
    # We weight each cue's contribution by its probability of being the first discriminating cue in a Plackett-Luce sampling.
    # For simplicity, approximate the probability that cue i is encountered before j by p_i / (p_i + p_j).
    # Expected "first discriminating cue" weight for cue i:
    pair_pref = p_sample[None, :] / (p_sample[None, :] + p_sample[:, None] + 1e-12)  # i vs j
    # Probability cue i precedes all others: product over j!=i of p_i/(p_i+p_j)
    precede_prob = np.prod(pair_pref, axis=1)

    # Signed cue signal per trial
    disc = (B > A).astype(float) - (A > B).astype(float)  # shape (n_trials, 4)

    # Lexicographic component: use only the (expected) first discriminating cue
    # Multiply by the chance that a cue is first; then zero out non-discriminating cues per trial
    lex_weights = precede_prob  # shape (4,)
    # For each trial, weight discriminating cues by lex_weights and pick the strongest by absolute value
    lex_signal = disc * lex_weights[None, :]
    # If multiple discriminate, expected first is approximated by taking the max by absolute weighted signal
    # But since signals are +-1 or 0, weighted max reduces to the cue with highest lex_weight among discriminators.
    # Compute per-trial chosen cue contribution:
    abs_weighted = np.abs(lex_signal)
    max_idx = np.argmax(abs_weighted, axis=1)
    chosen = np.take_along_axis(lex_signal, max_idx[:, None], axis=1).squeeze()
    # Probability of stopping at first discriminator:
    lex_evidence = stop_prob * chosen

    # Compensatory contrast component: power-nonlinearity on cue signals
    # Transform: f(s) = sign(s) * |s|^contrast_power; for binary signals this becomes sign(s) if power>0; when power=0, sign(s)^0 -> 1 for |s|>0
    # Implement robustly for 0: use |s|^p with p in [0,1]
    transformed = np.sign(disc) * (np.abs(disc) ** contrast_power)
    # Weight cues by their sampling probabilities (as attention proxy)
    comp_evidence = np.dot(transformed, p_sample)

    # Blend lexicographic and compensatory components
    blended = noncomp_weight * lex_evidence + (1.0 - noncomp_weight) * comp_evidence

    # Add centered tie bias (0.5 = no bias)
    bias = (tie_bias - 0.5)

    evidence = blended + bias

    # Choice probability
    dv = np.clip(temperature, 0.0, 10.0) * evidence
    pB = 1.0 / (1.0 + np.exp(-np.clip(dv, -700, 700)))
    pB = lapse * 0.5 + (1.0 - lapse) * pB

    # Negative log-likelihood
    eps = 1e-12
    pB = np.clip(pB, eps, 1.0 - eps)
    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1 - pB)
    return -np.sum(ll)