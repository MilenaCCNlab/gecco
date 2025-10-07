def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """Asymmetric learning with surprise-modulated MB weighting and learned transitions.

    This model implements:
    - Asymmetric stage-2 learning rates for positive vs negative outcomes.
    - Surprise-adaptive effective learning rate and dynamic arbitration:
      the weight on model-based control increases with transition surprise.
    - A single stickiness term applied at both stages.
    - Online learning of the transition matrix.

    Parameters (model_parameters):
    - alpha_pos: [0,1] stage-2 learning rate when reward=1
    - alpha_neg: [0,1] stage-2 learning rate when reward=0
    - beta1:     [0,10] inverse temperature at stage 1
    - beta2:     [0,10] inverse temperature at stage 2
    - w0:        [0,1] baseline MB weight (at zero surprise)
    - phi:       [0,1] surprise gain; increases MB weight with transition surprise
    - trans_alpha: [0,1] learning rate for transition probabilities
    - kappa:     [0,1] choice stickiness applied at both stages

    Inputs:
    - action_1: array-like, int in {0,1}
    - state:    array-like, int in {0,1}
    - action_2: array-like, int in {0,1}
    - reward:   array-like, float in {0,1}
    - model_parameters: list/array of 8 floats within the bounds above

    Returns:
    - Negative log-likelihood of the observed choices.
    """
    alpha_pos, alpha_neg, beta1, beta2, w0, phi, trans_alpha, kappa = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    trans = np.array([[0.7, 0.3],
                      [0.3, 0.7]], dtype=float)

    q1_mf = np.zeros(2)
    q2 = np.full((2, 2), 0.5)

    prev_a1 = None
    prev_a2_in_state = [None, None]

    loglik = 0.0

    for t in range(n_trials):
        s = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r = reward[t]

        surprise = 1.0 - trans[a1, s]

        w = w0 + phi * surprise
        w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)

        max_q2 = np.max(q2, axis=1)
        q1_mb = trans @ max_q2

        stick1 = np.zeros(2)
        if prev_a1 is not None:
            stick1[prev_a1] = 1.0

        stick2 = np.zeros(2)
        if prev_a2_in_state[s] is not None:
            stick2[prev_a2_in_state[s]] = 1.0

        q1 = w * q1_mb + (1.0 - w) * q1_mf + kappa * stick1
        pref1 = beta1 * q1
        pref1 -= np.max(pref1)
        probs1 = np.exp(pref1)
        probs1 /= np.sum(probs1)
        loglik += np.log(probs1[a1] + eps)

        q2_pref = q2[s] + kappa * stick2
        pref2 = beta2 * q2_pref
        pref2 -= np.max(pref2)
        probs2 = np.exp(pref2)
        probs2 /= np.sum(probs2)
        loglik += np.log(probs2[a2] + eps)


        for st in (0, 1):
            target = 1.0 if st == s else 0.0
            trans[a1, st] = (1 - trans_alpha) * trans[a1, st] + trans_alpha * target
        trans[a1] /= np.sum(trans[a1])

        alpha = alpha_pos if r > q2[s, a2] else alpha_neg

        alpha_eff = alpha * (0.5 + 0.5 * (surprise if surprise < 1.0 else 1.0))
        alpha_eff = 0.0 if alpha_eff < 0.0 else (1.0 if alpha_eff > 1.0 else alpha_eff)
        pe2 = r - q2[s, a2]
        q2[s, a2] += alpha_eff * pe2

        target1 = q2[s, a2]
        pe1 = target1 - q1_mf[a1]
        q1_mf[a1] += (alpha_pos * 0.5 + alpha_neg * 0.5) * pe1  # mean of pos/neg rates to keep both used

        prev_a1 = a1
        prev_a2_in_state[s] = a2

    return -loglik