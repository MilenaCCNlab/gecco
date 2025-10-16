def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Validity-weighted additive utility with attentional weights, side bias, and lapse.
    
    Cognitive idea:
    - Each expert's rating contributes to a utility via its known validity (0.9,0.8,0.7,0.6), modulated by an attention weight.
    - Choices are made via a softmax on the difference in additive utilities (B minus A).
    - A side bias toward B can shift choices independently of attributes.
    - A lapse parameter captures random responding.

    Parameters (in order):
    - w1: [0,1] Attention multiplier for cue 1 (most valid, 0.9).
    - w2: [0,1] Attention multiplier for cue 2 (0.8).
    - w3: [0,1] Attention multiplier for cue 3 (0.7).
    - w4: [0,1] Attention multiplier for cue 4 (0.6).
    - biasB: [0,1] Baseline bias toward choosing option B (0.5 = no bias). Internally centered around zero.
    - lapse: [0,1] Probability of random choice (uniform 0.5), mixed with the model-based probability.
    - beta: [0,10] Inverse temperature controlling choice determinism.

    Inputs:
    - decisions: array-like of 0/1 where 1 indicates choosing B, 0 indicates choosing A.
    - A_feature1..4, B_feature1..4: arrays of 0/1 expert ratings for options A and B per trial.
    - parameters: iterable of length 7 as described above.

    Returns:
    - Negative log-likelihood of observed choices under the model.
    """
    w1, w2, w3, w4, biasB, lapse, beta = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6])
    attn = np.array([w1, w2, w3, w4])
    # Center bias around zero, scaled modestly to be comparable to one cue
    bias_term = (biasB - 0.5) * 2.0  # in [-1,1]

    A = np.vstack([A_feature1, A_feature2, A_feature3, A_feature4]).T
    B = np.vstack([B_feature1, B_feature2, B_feature3, B_feature4]).T
    decisions = np.asarray(decisions).astype(int)

    # Compute weighted utilities
    cue_weights = validities * attn  # shape (4,)
    uA = A @ cue_weights
    uB = B @ cue_weights

    dv = (uB - uA) + bias_term  # add bias toward B
    pB_model = 1.0 / (1.0 + np.exp(-beta * dv))
    # Mix with lapse (uniform random = 0.5)
    pB = lapse * 0.5 + (1.0 - lapse) * pB_model
    pB = np.clip(pB, 1e-9, 1.0 - 1e-9)

    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1.0 - pB)
    return -np.sum(ll)


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Probabilistic Take-The-Best with noisy stopping, bias, and lapse.
    
    Cognitive idea:
    - The decision maker inspects cues in a personalized priority order (modulating known validities).
    - At each discriminating cue (one option has 1, the other 0), there is a probability to stop and decide based on that cue.
    - The choice at the selected cue is stochastic and scaled by both cue validity and inverse temperature.
    - If no cue is used (no discrimination or all skipped), the choice falls back to a side bias.
    - A lapse parameter captures random responding.

    Parameters (in order):
    - p1: [0,1] Priority multiplier for cue 1 (0.9 validity).
    - p2: [0,1] Priority multiplier for cue 2 (0.8 validity).
    - p3: [0,1] Priority multiplier for cue 3 (0.7 validity).
    - p4: [0,1] Priority multiplier for cue 4 (0.6 validity).
    - stop: [0,1] Probability to stop and decide at the first encountered discriminating cue; else continue to next.
    - biasB: [0,1] Baseline bias toward B when no cue is used (0.5 = no bias).
    - lapse: [0,1] Probability of random choice (uniform 0.5), mixed with the model-based probability.
    - beta: [0,10] Inverse temperature shaping stochasticity of cue-based decisions.

    Inputs:
    - decisions: array-like of 0/1 where 1 indicates choosing B, 0 indicates choosing A.
    - A_feature1..4, B_feature1..4: arrays of 0/1 expert ratings for options A and B per trial.
    - parameters: iterable of length 8 as described above.

    Returns:
    - Negative log-likelihood of observed choices under the model.
    """
    p1, p2, p3, p4, stop, biasB, lapse, beta = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6])
    priorities = validities * np.array([p1, p2, p3, p4])

    A = np.vstack([A_feature1, A_feature2, A_feature3, A_feature4]).T
    B = np.vstack([B_feature1, B_feature2, B_feature3, B_feature4]).T
    decisions = np.asarray(decisions).astype(int)
    n_trials = len(decisions)

    # Precompute cue order (highest priority first)
    order = np.argsort(-priorities)  # descending

    pB = np.zeros(n_trials)
    for t in range(n_trials):
        # Discriminations: +1 supports B, -1 supports A, 0 no info
        d = B[t, :] - A[t, :]  # length 4
        pB_t = None

        # Iterate in priority order and apply stopping policy
        for idx in order:
            if d[idx] == 0:
                continue
            # With probability 'stop', decide using this cue; otherwise continue searching
            # We implement the expected choice probability under this stochastic stopping:
            # p(use this cue) = stop; p(continue) = 1 - stop to potentially use later cues.
            # To keep it simple and analytic, we compute the probability as if the agent stops here deterministically with prob 'stop'.
            cue_strength = validities[idx] * np.sign(d[idx])  # positive supports B, negative supports A
            pB_cue = 1.0 / (1.0 + np.exp(-beta * cue_strength))
            if pB_t is None:
                pB_t = stop * pB_cue + (1.0 - stop) * None  # placeholder for continuation
                # We'll propagate continuation multiplicatively below
                cont_prob = (1.0 - stop)
                cont_val = 0.0  # accumulates weighted pB from later cues
            else:
                # shouldn't happen because we handle the first discriminating cue specially
                pass

            # Look for next discriminating cue to allocate continuation probability
            # If none exists, continuation falls back to bias.
            # Find next discriminating cue in the remaining ordered list
            next_pB = None
            for jdx in order:
                if jdx == idx:
                    continue
                if d[jdx] == 0:
                    continue
                cue_strength2 = validities[jdx] * np.sign(d[jdx])
                next_pB = 1.0 / (1.0 + np.exp(-beta * cue_strength2))
                break

            if next_pB is None:
                fallback = biasB
            else:
                fallback = next_pB

            # Combine stop decision at first discriminating cue with continuation outcome
            pB_t = stop * pB_cue + (1.0 - stop) * fallback
            break  # we have handled first discriminating cue

        # If no discriminating cue found, use bias
        if pB_t is None:
            pB_t = biasB

        # Mix with lapse
        pB[t] = lapse * 0.5 + (1.0 - lapse) * pB_t

    pB = np.clip(pB, 1e-9, 1.0 - 1e-9)
    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1.0 - pB)
    return -np.sum(ll)


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Selective integration with polarity asymmetry, conflict suppression, bias, and lapse.
    
    Cognitive idea:
    - Attributes contribute proportionally to their validities, modulated by feature-specific sensitivities.
    - Positive evidence for B (B=1, A=0) and negative evidence (B=0, A=1) are weighted asymmetrically (polarity asymmetry).
    - When multiple cues conflict, lower-validity cues can be selectively suppressed relative to the strongest discriminating cue.
    - Choices follow a logistic function of the resulting utility difference, with bias and lapse.

    Parameters (in order):
    - s1: [0,1] Sensitivity to cue 1 (0.9 validity).
    - s2: [0,1] Sensitivity to cue 2 (0.8 validity).
    - s3: [0,1] Sensitivity to cue 3 (0.7 validity).
    - s4: [0,1] Sensitivity to cue 4 (0.6 validity).
    - alpha: [0,1] Polarity asymmetry; weight for positive evidence favoring B. Weight for evidence favoring A is (1 - alpha).
    - k_suppress: [0,1] Degree of suppression applied to cues with validity lower than the strongest discriminating cue (0=no suppression, 1=full suppression).
    - biasB: [0,1] Baseline bias toward choosing B (0.5 = no bias), added as an intercept to the value difference.
    - lapse: [0,1] Probability of random choice (uniform 0.5), mixed with the model-based probability.
    - beta: [0,10] Inverse temperature controlling choice determinism.

    Inputs:
    - decisions: array-like of 0/1 where 1 indicates choosing B, 0 indicates choosing A.
    - A_feature1..4, B_feature1..4: arrays of 0/1 expert ratings for options A and B per trial.
    - parameters: iterable of length 9 as described above.

    Returns:
    - Negative log-likelihood of observed choices under the model.
    """
    s1, s2, s3, s4, alpha, k_suppress, biasB, lapse, beta = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6])
    sens = np.array([s1, s2, s3, s4])

    A = np.vstack([A_feature1, A_feature2, A_feature3, A_feature4]).T
    B = np.vstack([B_feature1, B_feature2, B_feature3, B_feature4]).T
    decisions = np.asarray(decisions).astype(int)
    n_trials = len(decisions)

    # Bias term centered to [-1,1] for intercept-like effect
    bias_term = (biasB - 0.5) * 2.0

    pB = np.zeros(n_trials)
    for t in range(n_trials):
        d = B[t, :] - A[t, :]  # +1 supports B, -1 supports A, 0 no info

        # Identify discriminating cues and the strongest one by validity
        discr_mask = d != 0
        if np.any(discr_mask):
            discr_validities = validities[discr_mask]
            max_v = np.max(discr_validities)
        else:
            max_v = 0.0

        contrib = np.zeros(4)
        for i in range(4):
            if d[i] == 0:
                continue
            # Polarity asymmetry
            if d[i] > 0:
                pol_weight = alpha
            else:
                pol_weight = (1.0 - alpha)

            # Selective suppression for cues weaker than the strongest discriminating cue
            if validities[i] < max_v:
                suppress_factor = (1.0 - k_suppress)
            else:
                suppress_factor = 1.0

            # Contribution to B-minus-A value
            contrib[i] = d[i] * validities[i] * sens[i] * pol_weight * suppress_factor

        dv = np.sum(contrib) + bias_term
        pB_model = 1.0 / (1.0 + np.exp(-beta * dv))
        pB[t] = lapse * 0.5 + (1.0 - lapse) * pB_model

    pB = np.clip(pB, 1e-9, 1.0 - 1e-9)
    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1.0 - pB)
    return -np.sum(ll)