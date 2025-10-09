def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Bayesian-like weighted cue integration with calibration, redundancy penalty, momentum, and lapses.

    Mechanism:
    - Computes signed evidence from all cues (expert ratings) favoring B over A.
    - Cue weights are a calibrated linear function of validity, then normalized to form an attention distribution.
    - A redundancy penalty downscales evidence when many cues align strongly (to capture diminishing returns).
    - A momentum term pulls the current choice toward the previous observed decision (choice autocorrelation).
    - The final choice probability is passed through a logistic with inverse temperature and mixed with lapses.

    Parameters (all used):
    - base: [0,1] baseline attention to cues independent of validity (higher -> more uniform weighting).
    - slope: [0,1] calibration slope mapping validity deviations into attention (0 -> validity ignored; 1 -> fully used).
    - rho: [0,1] redundancy penalty; scales down evidence as cues align (0 none; 1 strong penalty).
    - momentum: [0,1] strength of autocorrelation toward previous observed decision (0 none; 1 strong).
    - epsilon: [0,1] lapse rate; mixes the model probability with random choice (0.5).
    - temperature: [0,10] inverse temperature for the logistic choice rule.

    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B, 0 = chose A).
    - A_feature1..A_feature4: arrays of 0/1 ratings for option A from experts [validities 0.9, 0.8, 0.7, 0.6].
    - B_feature1..B_feature4: arrays of 0/1 ratings for option B from the same experts.

    Returns:
    - Negative log-likelihood of observed decisions under the model.
    """
    base, slope, rho, momentum, epsilon, temperature = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    mean_v = np.mean(validities)

    # Stack features
    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)

    # Signed cue signals: +1 favors B, -1 favors A, 0 tie
    sign = np.sign(B - A)

    n_trials = len(decisions)
    decisions = np.asarray(decisions).astype(float)

    # Calibrated nonnegative weights from validity
    raw_w = base + slope * (validities - mean_v)  # may be small/negative if base tiny; clip to nonneg
    raw_w = np.clip(raw_w, 0.0, None)
    if np.sum(raw_w) <= 0:
        # If degenerate, fall back to uniform
        w = np.ones(4, dtype=float) / 4.0
    else:
        w = raw_w / np.sum(raw_w)

    nll = 0.0
    prev_choice = 0.5  # neutral initial previous-choice belief

    for t in range(n_trials):
        s = sign[t]

        # Base weighted evidence
        base_ev = float(np.sum(w * s))

        # Redundancy penalty: stronger down-weight when cues align
        nonzero = (s != 0).astype(float)
        k = int(np.sum(nonzero))
        if k > 0:
            align_strength = abs(np.sum(s)) / k  # in [0,1]; 1 if all discriminating cues align
            penalty = 1.0 - rho * align_strength  # in [1 - rho, 1]
        else:
            penalty = 1.0

        ev = base_ev * penalty

        # Convert to logits with temperature
        logits = temperature * ev

        # Momentum toward previous choice in logit space
        # Map prev_choice in [0,1] to signed pull in [-1,1]
        prev_pull = (prev_choice - 0.5) * 2.0
        logits += 5.0 * momentum * prev_pull  # scale 5 provides a meaningful dynamic range

        # Prob of choosing B
        pB = 1.0 / (1.0 + np.exp(-logits))

        # Lapses
        pB = (1.0 - epsilon) * pB + 0.5 * epsilon

        # Likelihood contribution
        p_obs = decisions[t] * pB + (1.0 - decisions[t]) * (1.0 - pB)
        p_obs = np.clip(p_obs, 1e-12, 1.0)
        nll += -np.log(p_obs)

        prev_choice = decisions[t]

    return nll


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Sequential satisficing with position-dependent stopping, asymmetric trust, noisy cue reading, and lapses.

    Mechanism:
    - Cues are inspected in descending validity order (0.9 -> 0.6).
    - When a discriminating cue is encountered at position k, the model stops with probability s_k = 1 - (1 - tau)^k.
      If it does not stop, it continues to search for the next discriminating cue.
    - Discriminating cues that favor B receive a multiplicative boost relative to those favoring A (asymmetry).
    - Cue observation is noisy: with probability nu, the sign of a discriminating cue is flipped.
      We integrate over this noise by using the expected signed impact.
    - If no stop occurs across all discriminating cues, a centered initial bias contributes in logit space.
    - The expected signed evidence across possible stop points is mapped to choice via logistic with temperature and lapses.

    Parameters (all used):
    - tau: [0,1] stopping growth parameter; larger values increase stopping probability at earlier positions.
    - nu: [0,1] perceptual noise; probability of flipping the sign of a discriminating cue upon inspection.
    - asym: [0,1] asymmetric trust; boosts evidence when cue favors B (1+asym) and reduces when favors A (1-asym).
    - init_bias: [0,1] baseline preference for B when no decisive stop occurs; centered around 0.5 and applied in logits.
    - epsilon: [0,1] lapse rate; mixes the model probability with random choice (0.5).
    - temperature: [0,10] inverse temperature scaling of expected evidence.

    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B, 0 = chose A).
    - A_feature1..A_feature4: arrays of 0/1 ratings for option A from experts [validities 0.9, 0.8, 0.7, 0.6].
    - B_feature1..B_feature4: arrays of 0/1 ratings for option B from the same experts.

    Returns:
    - Negative log-likelihood of observed decisions under the model.
    """
    tau, nu, asym, init_bias, epsilon, temperature = parameters
    # Order indices by validity descending (fixed order here)
    order = np.array([0, 1, 2, 3], dtype=int)  # corresponds to validities [0.9, 0.8, 0.7, 0.6]

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    sign = np.sign(B - A)  # +1 favors B, -1 favors A, 0 tie

    n_trials = len(decisions)
    decisions = np.asarray(decisions).astype(float)

    nll = 0.0
    bias_logit = 5.0 * (init_bias - 0.5) * 2.0  # centered bias mapped to logits

    flip_factor = (1.0 - 2.0 * nu)  # E[sign with flip noise] = sign * flip_factor

    for t in range(n_trials):
        s = sign[t][order]  # inspect in descending validity order

        # Collect indices of discriminating cues in the order encountered
        discrim_idx = np.where(s != 0)[0]

        expected_ev = 0.0
        mass_continue = 1.0

        if discrim_idx.size > 0:
            for idx in discrim_idx:
                k = idx + 1  # 1-based position in the ordered sequence
                # Asymmetric trust scaling
                if s[idx] > 0:
                    s_eff = (1.0 + asym) * 1.0
                else:
                    s_eff = (1.0 - asym) * (-1.0)

                # Expected signed impact after noise
                s_exp = s_eff * flip_factor

                stop_p = 1.0 - (1.0 - tau) ** k
                stop_p = np.clip(stop_p, 0.0, 1.0)

                expected_ev += mass_continue * stop_p * s_exp
                mass_continue *= (1.0 - stop_p)

        # Map expected evidence to logits
        logits = temperature * expected_ev

        # If never stopped, add centered initial bias in logits
        if mass_continue > 0.0:
            logits += mass_continue * bias_logit

        # Choice probability for B
        pB = 1.0 / (1.0 + np.exp(-logits))

        # Lapses
        pB = (1.0 - epsilon) * pB + 0.5 * epsilon

        # Log-likelihood
        p_obs = decisions[t] * pB + (1.0 - decisions[t]) * (1.0 - pB)
        p_obs = np.clip(p_obs, 1e-12, 1.0)
        nll += -np.log(p_obs)

    return nll


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Nonlinear cue weighting with power-transformed validities, sparsity, conflict-adaptive temperature, and lapses.

    Mechanism:
    - Cues contribute signed evidence favoring B vs A.
    - Weights are derived from validities via a power transform and a sparsity mixer:
        w_power ∝ validity^power; normalize to sum=1.
        final_w = (1 - sparsity) * w_power + sparsity * one_hot(argmax(w_power)).
      Thus sparsity shifts mass toward the single most valid cue (winner-take-all) as it increases.
    - A conflict-adaptive mechanism reduces effective temperature when cues strongly disagree.
    - A baseline preference for B (baselineB) acts as a prior in logit space.
    - Lapses mix probability with random choice.

    Parameters (all used):
    - power: [0,1] curvature on validity; lower values flatten cue weights, higher values sharpen them.
    - sparsity: [0,1] mixes distributed vs winner-take-all attention across cues.
    - conflict_aversion: [0,1] reduces effective temperature in proportion to cue conflict.
    - baselineB: [0,1] baseline preference for B; centered around 0.5 and applied in logits.
    - epsilon: [0,1] lapse rate; mixes the model probability with random choice (0.5).
    - temperature: [0,10] base inverse temperature before conflict adaptation.

    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B, 0 = chose A).
    - A_feature1..A_feature4: arrays of 0/1 ratings for option A from experts [validities 0.9, 0.8, 0.7, 0.6].
    - B_feature1..B_feature4: arrays of 0/1 ratings for option B from the same experts.

    Returns:
    - Negative log-likelihood of observed decisions under the model.
    """
    power, sparsity, conflict_aversion, baselineB, epsilon, temperature = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    # Power-transformed weights
    # Avoid 0^0 by ensuring strictly positive base; validities already >0
    w_power = validities ** (1e-9 + power)  # tiny offset keeps derivative stable near 0
    w_power = w_power / np.sum(w_power)

    # Winner-take-all component
    winner = np.zeros_like(w_power)
    winner[int(np.argmax(w_power))] = 1.0

    # Final weights with sparsity mixing
    w = (1.0 - sparsity) * w_power + sparsity * winner
    w = w / np.sum(w)  # ensure normalization

    # Data matrices
    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)

    # Signed cues
    sign = np.sign(B - A)

    n_trials = len(decisions)
    decisions = np.asarray(decisions).astype(float)

    # Baseline prior mapped to logits
    bias_logit = 5.0 * (baselineB - 0.5) * 2.0

    nll = 0.0
    for t in range(n_trials):
        s = sign[t]

        # Evidence
        ev = float(np.sum(w * s))

        # Conflict measure: fraction of disagreement among discriminating cues
        nonzero = (s != 0)
        k = int(np.sum(nonzero))
        if k > 0:
            n_pos = int(np.sum(s[nonzero] > 0))
            n_neg = k - n_pos
            if max(n_pos, n_neg) > 0:
                conflict = min(n_pos, n_neg) / max(n_pos, n_neg)  # 0 (no conflict) .. 1 (balanced conflict)
            else:
                conflict = 0.0
        else:
            conflict = 0.0

        # Conflict-adaptive temperature
        eff_temp = temperature * (1.0 - conflict_aversion * conflict)

        logits = eff_temp * ev + bias_logit

        pB = 1.0 / (1.0 + np.exp(-logits))

        # Lapses
        pB = (1.0 - epsilon) * pB + 0.5 * epsilon

        # Likelihood
        p_obs = decisions[t] * pB + (1.0 - decisions[t]) * (1.0 - pB)
        p_obs = np.clip(p_obs, 1e-12, 1.0)
        nll += -np.log(p_obs)

    return nll