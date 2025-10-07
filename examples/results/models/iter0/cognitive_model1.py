def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """Hybrid model-based/model-free RL with learned transitions, eligibility traces, and perseveration.
    
    This model blends a learned model-based action value with model-free values at stage 1, 
    learns stage-2 values from reward, propagates reward to stage 1 via an eligibility trace, 
    learns the transition structure online, and includes perseveration (choice stickiness) at both stages.
    A small lapse parameter mixes the softmax policy with uniform random choice.

    Parameters (model_parameters):
    - alpha1: [0,1] learning rate for stage-1 model-free Q-values
    - alpha2: [0,1] learning rate for stage-2 model-free Q-values
    - beta1:  [0,10] inverse temperature for stage-1 softmax
    - beta2:  [0,10] inverse temperature for stage-2 softmax
    - w_mb:   [0,1] weight of model-based value in stage-1 action values (1=fully model-based)
    - lam:    [0,1] eligibility trace; propagates stage-2 reward prediction error to stage-1 MF values
    - kappa1: [0,1] perseveration weight (stage-1): bias to repeat previous spaceship choice
    - kappa2: [0,1] perseveration weight (stage-2): bias to repeat previous alien choice (within the visited state)
    - trans_alpha: [0,1] transition learning rate to update P(state | action_1)
    - lapse:  [0,1] lapse rate; with probability lapse, choices are uniformly random (2 options)

    Inputs:
    - action_1: array-like, int in {0,1} chosen spaceship each trial (0=A, 1=U)
    - state:    array-like, int in {0,1} visited planet each trial (0=X, 1=Y)
    - action_2: array-like, int in {0,1} chosen alien on visited planet each trial
    - reward:   array-like, float in {0,1} coin outcome
    - model_parameters: list/array of 10 floats within the bounds above

    Returns:
    - Negative log-likelihood of the observed stage-1 and stage-2 choices under the model.
    """
    alpha1, alpha2, beta1, beta2, w_mb, lam, kappa1, kappa2, trans_alpha, lapse = model_parameters

    n_trials = len(action_1)
    eps = 1e-12


    trans = np.array([[0.7, 0.3],
                      [0.3, 0.7]], dtype=float)

    q1_mf = np.zeros(2)               # stage-1 MF values for actions {0,1}
    q2_mf = np.full((2, 2), 0.5)      # stage-2 MF values for states {0,1} and actions {0,1}

    prev_a1 = None
    prev_a2_in_state = [None, None]   # track previous a2 for each state separately

    loglik = 0.0

    for t in range(n_trials):
        s = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r = reward[t]

        max_q2 = np.max(q2_mf, axis=1)  # per state
        q1_mb = trans @ max_q2          # shape (2,)

        stick1 = np.zeros(2)
        if prev_a1 is not None:
            stick1[prev_a1] = 1.0

        q1 = w_mb * q1_mb + (1.0 - w_mb) * q1_mf + kappa1 * stick1

        pref1 = beta1 * q1
        pref1 -= np.max(pref1)  # numerical stability
        soft1 = np.exp(pref1)
        soft1 /= np.sum(soft1)
        probs1 = (1.0 - lapse) * soft1 + lapse * 0.5

        p1 = probs1[a1]
        loglik += np.log(p1 + eps)

        stick2 = np.zeros(2)
        if prev_a2_in_state[s] is not None:
            stick2[prev_a2_in_state[s]] = 1.0
        q2_pref = q2_mf[s] + kappa2 * stick2

        pref2 = beta2 * q2_pref
        pref2 -= np.max(pref2)
        soft2 = np.exp(pref2)
        soft2 /= np.sum(soft2)
        probs2 = (1.0 - lapse) * soft2 + lapse * 0.5

        p2 = probs2[a2]
        loglik += np.log(p2 + eps)



        for st in (0, 1):
            target = 1.0 if st == s else 0.0
            trans[a1, st] = (1 - trans_alpha) * trans[a1, st] + trans_alpha * target

        trans[a1] /= np.sum(trans[a1])

        pe2 = r - q2_mf[s, a2]
        q2_mf[s, a2] += alpha2 * pe2



        backup = q2_mf[s, a2]
        pe1_boot = backup - q1_mf[a1]
        q1_mf[a1] += alpha1 * pe1_boot

        q1_mf[a1] += alpha1 * lam * pe2

        prev_a1 = a1
        prev_a2_in_state[s] = a2

    return -loglik