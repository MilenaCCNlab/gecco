def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Attentive blend of compensatory and max-heuristic with validity interpolation and lapse.

    Core idea:
    - Each cue contributes a signed signal (+1 if B=1 & A=0; -1 if A=1 & B=0; 0 otherwise).
    - A per-cue attention parameter scales whether that cue is effectively used.
    - Cue weights interpolate between equal-weight and stated validities via validity_weight.
    - The final evidence blends a compensatory sum and a noncompensatory max rule via alpha.
    - Evidence is mapped to P(choose B) through a logistic with inverse temperature, with lapse.

    Parameters (bounds):
    - attn1 in [0,1]: Attention probability/strength for cue 1 (validity 0.9).
    - attn2 in [0,1]: Attention probability/strength for cue 2 (validity 0.8).
    - attn3 in [0,1]: Attention probability/strength for cue 3 (validity 0.7).
    - attn4 in [0,1]: Attention probability/strength for cue 4 (validity 0.6).
    - validity_weight in [0,1]: Interpolates between equal cue weights (0) and stated validities (1).
    - alpha in [0,1]: Blend between compensatory sum (0) and max-heuristic (1).
    - temperature in [0,10]: Inverse temperature for the logistic choice rule.
    - lapse in [0,1]: Lapse (random choice) probability mixed with the model’s prediction.

    Returns:
    - Negative log-likelihood of the observed choices (decisions: 1 = choose B, 0 = choose A).
    """
    attn1, attn2, attn3, attn4, validity_weight, alpha, temperature, lapse = parameters

    # Inputs to arrays
    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    decisions = np.asarray(decisions).astype(int)

    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    # Interpolate between equal weights (1.0) and stated validities
    v_eff = (1.0 - validity_weight) * 1.0 + validity_weight * validities

    # Attention-scaled effective weights
    attn = np.array([attn1, attn2, attn3, attn4], dtype=float)
    w = attn * v_eff  # per-cue weight

    # Signed discrimination per trial and cue: +1 if B>A, -1 if A>B, 0 otherwise
    disc = (B > A).astype(float) - (A > B).astype(float)  # shape (n_trials, 4)

    # Compensatory sum component
    sum_component = np.dot(disc, w)  # shape (n_trials,)

    # Max-heuristic component (take the strongest single-cue push)
    # Note: max over signed weighted discriminations across cues
    weighted_disc = disc * w[None, :]
    max_component = np.max(weighted_disc, axis=1)

    # Blend
    evidence = (1.0 - alpha) * sum_component + alpha * max_component

    # Choice rule with lapse
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
    """Bayesian evidence accumulation with prior, consistency gain, redundancy suppression, and lapse.

    Core idea:
    - Each cue produces a log-likelihood ratio (LLR) favoring B versus A based on cue validity.
    - Validities are softened toward 0.5 using validity_scale to capture imperfect trust.
    - Cues are processed from highest to lowest validity; their LLR contributions are attenuated by:
        • noise (reduces all cue impact),
        • redundancy_suppression (downweights later discriminating cues),
        • consistency_gain (amplifies cues consistent with current cumulative belief, attenuates inconsistent).
    - A prior bias contributes an initial log-odds favoring either A or B.
    - The final cumulative LLR is mapped to choice with a logistic temperature and lapse.

    Parameters (bounds):
    - prior_strength in [0,1]: Prior bias toward B (>0.5) or A (<0.5); mapped to prior log-odds with data-scaled magnitude.
    - noise in [0,1]: Global attenuation of cue information (1 = no information, 0 = full information).
    - consistency_gain in [0,1]: Degree to boost cues that agree with current cumulated sign (and damp disagreeing cues).
    - redundancy_suppression in [0,1]: Geometric down-weighting for later discriminating cues.
    - validity_scale in [0,1]: Interpolates cue reliability between 0.5 (uninformative) and stated validity.
    - temperature in [0,10]: Inverse temperature for the logistic choice rule.
    - lapse in [0,1]: Lapse (random choice) probability.

    Returns:
    - Negative log-likelihood of the observed choices (decisions: 1 = choose B, 0 = choose A).
    """
    prior_strength, noise, consistency_gain, redundancy_suppression, validity_scale, temperature, lapse = parameters

    # Prepare arrays
    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    decisions = np.asarray(decisions).astype(int)

    # Fixed cue validities (descending)
    base_v = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    # Soften toward 0.5
    v_eff = (1.0 - validity_scale) * 0.5 + validity_scale * base_v
    # Convert to LLR magnitude per cue
    # Avoid division by zero by clipping away from 0 and 1.
    v_eff = np.clip(v_eff, 1e-6, 1 - 1e-6)
    lambda_i = np.log(v_eff / (1.0 - v_eff))  # shape (4,)

    n_trials = A.shape[0]
    evidence = np.zeros(n_trials, dtype=float)

    # Discriminations per trial and cue: +1 (B>A), -1 (A>B), 0 (tie)
    disc = (B > A).astype(float) - (A > B).astype(float)  # (n_trials, 4)

    # Process cues in fixed validity order: indices [0,1,2,3]
    order = np.array([0, 1, 2, 3], dtype=int)

    # Prior log-odds: scale by potential signal magnitude of the trial to keep comparable units.
    # Trial-specific scale C = sum |lambda_i| over all cues (constant across trials here).
    C = np.sum(np.abs(lambda_i))
    prior_logodds = (prior_strength - 0.5) * C  # positive favors B, negative favors A

    for t in range(n_trials):
        cum = prior_logodds
        seen_disc = 0  # number of discriminating cues encountered so far
        for idx in order:
            d = disc[t, idx]
            if d == 0.0:
                # No information from a tie on this cue
                continue
            # Base weight reduced by noise and redundancy (for later discriminating cues)
            w = (1.0 - noise) * ((1.0 - redundancy_suppression) ** seen_disc)

            # Consistency modulation relative to current cumulative evidence
            if cum == 0:
                mod = 1.0
            else:
                sign_cum = 1.0 if cum > 0 else -1.0
                if np.sign(d) == sign_cum:
                    mod = 1.0 + consistency_gain
                else:
                    mod = 1.0 - consistency_gain

            cum += d * lambda_i[idx] * w * mod
            seen_disc += 1

        evidence[t] = cum

    # Choice rule with lapse
    dv = np.clip(temperature, 0.0, 10.0) * evidence
    pB = 1.0 / (1.0 + np.exp(-np.clip(dv, -700, 700)))
    pB = lapse * 0.5 + (1.0 - lapse) * pB

    eps = 1e-12
    pB = np.clip(pB, eps, 1.0 - eps)
    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1 - pB)
    return -np.sum(ll)


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Context-dependent attention with inhibition, nonlinear sensitivity, and side bias.

    Core idea:
    - Two weight pools: high-validity cues (1–2) share weight high_w; low-validity cues (3–4) share weight low_w.
    - Trial-wise attention gain depends on sparsity of discriminating cues (fewer discriminations → stronger gain).
    - Cross-cue inhibition reduces effective signal as more cues discriminate (divisive normalization).
    - A nonlinear power transform captures diminishing sensitivity to aggregated evidence.
    - A side_bias tilts decisions toward B (>0.5) or A (<0.5), independent of evidence strength.
    - Logistic choice with inverse temperature, plus lapse.

    Parameters (bounds):
    - high_w in [0,1]: Base weight for cues 1–2 (higher validity).
    - low_w in [0,1]: Base weight for cues 3–4 (lower validity).
    - inhibition in [0,1]: Divisive normalization factor based on number of discriminating cues.
    - sparsity_gain in [0,1]: Strength of attention amplification when few cues discriminate.
    - power in [0,1]: Nonlinear sensitivity exponent applied to aggregated evidence magnitude.
    - side_bias in [0,1]: Bias toward B (>0.5) or A (<0.5), added as signed shift in evidence.
    - temperature in [0,10]: Inverse temperature for the logistic choice rule.
    - lapse in [0,1]: Lapse (random choice) probability.

    Returns:
    - Negative log-likelihood of the observed choices (decisions: 1 = choose B, 0 = choose A).
    """
    high_w, low_w, inhibition, sparsity_gain, power, side_bias, temperature, lapse = parameters

    # Prepare arrays
    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    decisions = np.asarray(decisions).astype(int)

    # Signed discrimination and discriminating-cue indicator
    disc = (B > A).astype(float) - (A > B).astype(float)  # +1, -1, or 0
    discrim = (B != A).astype(float)  # 1 if discriminating, else 0

    # Base weights per cue (two pools)
    w = np.array([high_w, high_w, low_w, low_w], dtype=float)

    # Trial-level computations
    n_trials = A.shape[0]
    evidence = np.zeros(n_trials, dtype=float)

    # Bias term scaled by overall weight scale (keeps bias comparable across settings)
    weight_scale = np.sum(w)
    bias_term = (side_bias - 0.5) * max(weight_scale, 1e-6)

    for t in range(n_trials):
        d = disc[t, :]
        r = discrim[t, :]
        # Sparsity: fraction of discriminating cues
        sparsity = 1.0 - (np.sum(r) / 4.0)  # 1 when no cues discriminate; 0 when all do
        # Attention amplification with sparsity
        attn_factor = (1.0 - sparsity_gain) + sparsity_gain * (1.0 * sparsity)
        # Aggregate raw signal
        raw_sum = np.sum((w * attn_factor) * d)
        # Divisive normalization by number of discriminating cues
        denom = 1.0 + inhibition * np.sum(r)
        adj = raw_sum / denom
        # Nonlinear sensitivity (power on magnitude)
        pe = max(power, 1e-6)
        nonlin = np.sign(adj) * (np.abs(adj) ** pe)
        # Add side bias
        evidence[t] = nonlin + bias_term

    # Choice rule with lapse
    dv = np.clip(temperature, 0.0, 10.0) * evidence
    pB = 1.0 / (1.0 + np.exp(-np.clip(dv, -700, 700)))
    pB = lapse * 0.5 + (1.0 - lapse) * pB

    # Negative log-likelihood
    eps = 1e-12
    pB = np.clip(pB, eps, 1.0 - eps)
    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1 - pB)
    return -np.sum(ll)