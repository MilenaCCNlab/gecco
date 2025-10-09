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

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)

    sign = np.sign(B - A)

    n_trials = len(decisions)
    decisions = np.asarray(decisions).astype(float)
    prev_choice = 0.5  # neutral for the very first trial

    nll = 0.0
    for t in range(n_trials):
        s = sign[t].copy()

        if np.all(s == 0):
            final_ev = 0.0
        else:

            remaining = np.ones(4, dtype=bool)
            final_ev = 0.0  # signed evidence from last discriminating cue found






            rem_valid = validities.copy()
            rem_s = s.copy()

            mass_continue = 1.0
            expected_ev = 0.0

            rem_idx = np.arange(4)

            for step in range(4):

                if not np.any(remaining):
                    break

                idxs = rem_idx[remaining[rem_idx]]
                vals = rem_valid[idxs]
                signs = rem_s[idxs]


                if len(idxs) == 1:
                    sel_probs = np.array([1.0], dtype=float)
                else:

                    max_val = np.max(vals)
                    is_max = (vals == max_val).astype(float)
                    if np.sum(is_max) > 0:
                        high_probs = is_max / np.sum(is_max)
                    else:
                        high_probs = np.ones_like(vals) / len(vals)
                    uni_probs = np.ones_like(vals) / len(vals)
                    sel_probs = p_high * high_probs + (1.0 - p_high) * uni_probs





                if np.any(signs != 0):
                    noisy_factor = (1.0 - 2.0 * eta)
                    step_ev = np.sum(sel_probs * (signs != 0).astype(float) * theta * signs * noisy_factor)
                else:
                    step_ev = 0.0

                expected_ev += mass_continue * step_ev



                cont_prob = 0.0
                if len(idxs) > 0:
                    cont_prob = np.sum(sel_probs * (((signs == 0).astype(float)) + ((signs != 0).astype(float) * (1.0 - theta))))
                mass_continue *= cont_prob



                sel_idx_local = int(np.argmax(sel_probs))  # index within idxs
                chosen_global = idxs[sel_idx_local]
                remaining[chosen_global] = False

            final_ev = expected_ev  # expected signed evidence from the first effective discriminating cue

        logits = temperature * final_ev

        pB = 1.0 / (1.0 + np.exp(-logits))
        pB = (1.0 - inertia) * pB + inertia * prev_choice

        pB = (1.0 - epsilon) * pB + 0.5 * epsilon

        p_obs = decisions[t] * pB + (1.0 - decisions[t]) * (1.0 - pB)
        p_obs = np.clip(p_obs, 1e-12, 1.0)
        nll += -np.log(p_obs)

        prev_choice = decisions[t]

    return nll