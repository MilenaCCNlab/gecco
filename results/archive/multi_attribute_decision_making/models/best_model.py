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

        score = 0.0
        first_found = False
        for i in range(4):
            d = B[i] - A[i]  # -1, 0, or 1
            if d != 0:

                score = utils[i] * (1.0 if d > 0 else -1.0)
                first_found = True
                break

        if not first_found:

            score = 0.0
            for i in range(4):
                score += utils[i] * (B[i] - A[i])

        z = temperature * score
        pB = 1.0 / (1.0 + np.exp(-z))
        pB = (1.0 - lapse) * pB + 0.5 * lapse

        pB = min(max(pB, eps), 1.0 - eps)
        if decisions[t] == 1:
            log_likelihood += np.log(pB)
        else:
            log_likelihood += np.log(1.0 - pB)

    return -log_likelihood