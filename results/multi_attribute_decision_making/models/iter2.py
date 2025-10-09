def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Stochastic take-the-best with noisy cue inspection, probabilistic stopping, response inertia, and lapses.
    
    Mechanism:
    - On each trial, the model inspects cues one-by-one until a discriminating cue is found.
    - The next cue to inspect is chosen stochastically with a bias toward higher-validity cues.
    - When a discriminating cue is found, the model stops with probability theta; otherwise it continues to search.
    - Cue observations are noisy: with probability eta, a cue's comparative sign is flipped.
    - The signed advantage from the last discriminating cue drives choice via a logistic with inverse temperature.
    - A response-inertia term pulls choice probability toward the previous observed decision.
    - A lapse rate mixes the model probability with uniform random choice.
    
    Parameters (all used):
    - p_high: [0,1] probability of choosing the highest-validity remaining cue on each inspection (vs uniform among remaining)
    - theta:  [0,1] stopping probability upon encountering a discriminating cue (theta=1 -> pure take-the-best)
    - eta:    [0,1] perceptual noise; probability that the sign of a discriminating cue is flipped
    - inertia:[0,1] response inertia; weight placed on repeating the previous choice (0 = none, 1 = full stickiness)
    - epsilon:[0,1] lapse probability; mixes with random choice
    - temperature: [0,10] inverse temperature; scales the impact of the final discriminating cue on choice
    
    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B, 0 = chose A)
    - A_feature1..A_feature4: arrays of 0/1 cue values for option A
    - B_feature1..B_feature4: arrays of 0/1 cue values for option B
    
    Returns:
    - negative log-likelihood of the observed decisions under the model
    """
    p_high, theta, eta, inertia, epsilon, temperature = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    # Precompute arrays
    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    # Signed comparative evidence per cue: +1 favors B, -1 favors A, 0 tie
    sign = np.sign(B - A)

    n_trials = len(decisions)
    decisions = np.asarray(decisions).astype(float)
    prev_choice = 0.5  # neutral for the very first trial

    nll = 0.0
    for t in range(n_trials):
        s = sign[t].copy()
        # If all cues tie, fallback to 0 net evidence
        if np.all(s == 0):
            final_ev = 0.0
        else:
            # Maintain a mask of remaining cues to inspect
            remaining = np.ones(4, dtype=bool)
            final_ev = 0.0  # signed evidence from last discriminating cue found
            # We compute the expected signed evidence under the stochastic inspection policy.
            # To avoid stochastic simulation, we analytically compute the expected contribution
            # of the first discriminating cue encountered under the policy and stopping rule.
            # We approximate by iterating up to 4 inspections and aggregating probabilities.
            # At each step: choose a cue index according to p_high policy among remaining cues.
            # If discriminating, with prob theta we stop and take its (noisy) sign; otherwise continue.
            rem_valid = validities.copy()
            rem_s = s.copy()

            # Keep track of probability mass that we reach each step
            mass_continue = 1.0
            expected_ev = 0.0

            # Work on copies for index selection each step
            rem_idx = np.arange(4)

            for step in range(4):
                # If no remaining cues, break
                if not np.any(remaining):
                    break

                # Get current remaining indices
                idxs = rem_idx[remaining[rem_idx]]
                vals = rem_valid[idxs]
                signs = rem_s[idxs]

                # Build selection probabilities: with prob p_high pick highest-validity cue,
                # with prob (1-p_high) pick uniformly among remaining
                if len(idxs) == 1:
                    sel_probs = np.array([1.0], dtype=float)
                else:
                    # Candidate highest-validity set (handle ties by spreading mass equally among maxima)
                    max_val = np.max(vals)
                    is_max = (vals == max_val).astype(float)
                    if np.sum(is_max) > 0:
                        high_probs = is_max / np.sum(is_max)
                    else:
                        high_probs = np.ones_like(vals) / len(vals)
                    uni_probs = np.ones_like(vals) / len(vals)
                    sel_probs = p_high * high_probs + (1.0 - p_high) * uni_probs

                # For each candidate, if discriminating (sign != 0),
                # its contribution if we stop is noisy_sign = sign*(1-2*eta)
                # Probability to stop upon discriminating is theta; else we continue without committing.
                # Aggregate expected contribution at this step:
                # expected increment = sum_over_cues sel_probs * I(sign!=0) * theta * noisy_sign
                if np.any(signs != 0):
                    noisy_factor = (1.0 - 2.0 * eta)
                    step_ev = np.sum(sel_probs * (signs != 0).astype(float) * theta * signs * noisy_factor)
                else:
                    step_ev = 0.0

                expected_ev += mass_continue * step_ev

                # Update continuation mass:
                # We continue if either we selected a tie cue (sign==0), or we selected a discriminating cue and decided not to stop (1-theta).
                # The probability of continuing from this step is:
                cont_prob = 0.0
                if len(idxs) > 0:
                    cont_prob = np.sum(sel_probs * (((signs == 0).astype(float)) + ((signs != 0).astype(float) * (1.0 - theta))))
                mass_continue *= cont_prob

                # Remove one cue from remaining in expectation by updating remaining deterministically is tricky.
                # To maintain tractability without simulation, we approximate by removing the most likely selected cue.
                # This keeps the process finite and captures the bias toward high-validity inspection.
                sel_idx_local = int(np.argmax(sel_probs))  # index within idxs
                chosen_global = idxs[sel_idx_local]
                remaining[chosen_global] = False

            final_ev = expected_ev  # expected signed evidence from the first effective discriminating cue

        # Convert evidence to choice probability toward B
        logits = temperature * final_ev

        # Add inertia toward previous observed decision in probability space
        pB = 1.0 / (1.0 + np.exp(-logits))
        pB = (1.0 - inertia) * pB + inertia * prev_choice

        # Lapse
        pB = (1.0 - epsilon) * pB + 0.5 * epsilon

        # Likelihood update
        p_obs = decisions[t] * pB + (1.0 - decisions[t]) * (1.0 - pB)
        p_obs = np.clip(p_obs, 1e-12, 1.0)
        nll += -np.log(p_obs)

        # Update inertia baseline with actual observed decision
        prev_choice = decisions[t]

    return nll


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Nonlinear configural integration with asymmetry, context normalization, prior preference, and lapses.
    
    Mechanism:
    - Each cue contributes signed evidence proportional to its (possibly compressed) validity and whether it favors A or B.
    - Positive (favoring B) and negative (favoring A) evidence are scaled differently (asymmetric sensitivity).
    - Evidence is divisively normalized by the amount of cue conflict present on the trial.
    - A prior preference toward B shifts the decision variable in logit space.
    - The inverse temperature scales the normalized decision variable, and lapses mix in random choice.
    
    Parameters (all used):
    - gamma:     [0,1] validity compression exponent; effective validity = validity^gamma (gamma<1 expands differences, >1 compresses; here 0..1 gives expansion)
    - lambda_p:  [0,1] gain applied to cues favoring B (positives)
    - lambda_m:  [0,1] gain applied to cues favoring A (negatives)
    - norm:      [0,1] context normalization strength; divides by (1 + norm * total_conflict)
    - priorB:    [0,1] prior preference toward B; 0.5 is neutral; implemented as a logit shift
    - temperature: [0,10] inverse temperature; higher -> more deterministic choices
    - epsilon:   [0,1] lapse probability; mixes with uniform choice
    
    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B, 0 = chose A)
    - A_feature1..A_feature4: arrays of 0/1 cue values for option A
    - B_feature1..B_feature4: arrays of 0/1 cue values for option B
    
    Returns:
    - negative log-likelihood of the observed decisions under the model
    """
    gamma, lambda_p, lambda_m, norm, priorB, temperature, epsilon = parameters

    base_valid = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    # Validity compression/expansion
    eff_valid = np.power(base_valid, np.clip(gamma, 0.0, 1.0))

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)

    # Signed cue signals: +1 favors B, -1 favors A, 0 tie
    sgn = np.sign(B - A)

    # Gains for positive vs negative
    gain = np.where(sgn > 0, lambda_p, np.where(sgn < 0, lambda_m, 0.0))
    # Effective signed evidences
    cue_ev = sgn * eff_valid * gain

    # Sum and normalize by conflict magnitude
    total_conflict = np.sum(np.abs(sgn), axis=1).astype(float)  # number of discriminating cues (0..4)
    raw_delta = np.sum(cue_ev, axis=1)
    denom = 1.0 + norm * total_conflict
    delta = raw_delta / denom

    # Prior shift in logit space from priorB in [0,1]
    prior_shift = (np.clip(priorB, 0.0, 1.0) - 0.5) * 2.0  # in [-1,1]

    logits = temperature * delta + prior_shift
    pB = 1.0 / (1.0 + np.exp(-logits))

    # Lapse
    pB = (1.0 - epsilon) * pB + 0.5 * epsilon

    decisions = np.asarray(decisions).astype(float)
    p = decisions * pB + (1.0 - decisions) * (1.0 - pB)
    nll = -np.sum(np.log(np.clip(p, 1e-12, 1.0)))
    return nll


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Attentional sampling accumulator with validity-biased selection, leak, perceptual noise, start bias, and lapses.
    
    Mechanism:
    - On each trial, the decision maker samples cues K times.
    - Each sample selects a cue with probability proportional to validity^alpha (alpha controls attentional selectivity).
    - The sampled cue contributes +1 if it favors B, -1 if it favors A, 0 if tied. With probability noise, the sign is flipped.
    - Evidence is leaky: later samples are down-weighted by (1 - leak)^(t-1). We use the closed-form expected effective number of samples.
    - The expected accumulated evidence equals E = K_eff * sum_i w_i * sign_i * (1 - 2*noise), where w_i are normalized attention weights.
    - A starting bias toward B shifts the decision variable.
    - A logistic with inverse temperature maps evidence to choice probability; a lapse rate mixes with random choice.
    
    Parameters (all used):
    - att:   [0,1] attentional selectivity; alpha = 5*att controls how strongly selection favors high-validity cues
    - samp:  [0,1] sampling extent; maps to integer K = 1 + round(49*samp) samples per trial
    - noise: [0,1] perceptual noise; probability that a sampled cue's sign is flipped
    - leak:  [0,1] evidence leak per step; higher leak -> stronger down-weighting of later samples
    - startB:[0,1] starting point bias toward B; 0.5 neutral; implemented as an additive shift in logit space
    - temperature: [0,10] inverse temperature; scales the accumulated evidence
    - epsilon:[0,1] lapse probability; mixes with random choice
    
    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B, 0 = chose A)
    - A_feature1..A_feature4: arrays of 0/1 cue values for option A
    - B_feature1..B_feature4: arrays of 0/1 cue values for option B
    
    Returns:
    - negative log-likelihood of the observed decisions under the model
    """
    att, samp, noise, leak, startB, temperature, epsilon = parameters

    # Validities and attention weights
    base_valid = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    alpha = 5.0 * np.clip(att, 0.0, 1.0)
    att_weights_raw = np.power(base_valid, alpha)
    att_weights = att_weights_raw / (np.sum(att_weights_raw) + 1e-12)

    # Number of samples and effective sample mass under leak
    K = int(np.round(49.0 * np.clip(samp, 0.0, 1.0))) + 1  # 1..50
    # Effective cumulative weight under geometric leak: sum_{t=0}^{K-1} (1 - leak)^t
    decay = 1.0 - np.clip(leak, 0.0, 1.0)
    if abs(decay - 1.0) < 1e-12:
        K_eff = float(K)
    else:
        K_eff = (1.0 - decay**K) / (1.0 - decay)

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)

    # Signed cue signals: +1 favors B, -1 favors A, 0 tie
    sgn = np.sign(B - A)

    # Expected per-sample signed increment
    # For ties, contribution is 0 regardless of noise; for discriminating cues, expected sign is sign*(1-2*noise)
    noisy_factor = (1.0 - 2.0 * np.clip(noise, 0.0, 1.0))
    per_sample_ev = np.dot(sgn, att_weights) * noisy_factor  # shape (n_trials,)

    # Expected accumulated evidence with leak
    delta = K_eff * per_sample_ev

    # Start bias as logit shift from startB in [0,1]
    start_shift = (np.clip(startB, 0.0, 1.0) - 0.5) * 2.0  # [-1,1]

    logits = temperature * delta + start_shift
    pB = 1.0 / (1.0 + np.exp(-logits))

    # Lapse
    pB = (1.0 - epsilon) * pB + 0.5 * epsilon

    decisions = np.asarray(decisions).astype(float)
    p = decisions * pB + (1.0 - decisions) * (1.0 - pB)
    nll = -np.sum(np.log(np.clip(p, 1e-12, 1.0)))
    return nll