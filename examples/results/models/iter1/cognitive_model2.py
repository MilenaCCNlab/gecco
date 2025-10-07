def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """
    Successor-Representation hybrid with asymmetric learning, learned transitions, habit, perseveration, and loss-boosted temperature.
    Concept:
      - Stage 2: asymmetric learning rates for wins vs losses with forgetting.
      - Stage 1: blend of (a) model-based value using learned transitions and (b) a cached SR-like mapping from actions to states.
      - SR component is a learned action->state occupancy estimate; sr_lambda weights cached SR vs transition-based MB.
      - Habit accumulation and perseveration bias at stage 1.
      - Loss-contingent temperature boost based on previous trial.
      - Bias toward spaceship A.

    Parameters (in order):
      alpha1: [0,1] Learning rate for SR cache (action->state) and MF Q1
      alpha2_win: [0,1] Stage-2 learning rate after reward=1
      alpha2_loss: [0,1] Stage-2 learning rate after reward=0
      beta1: [0,10] Stage-1 softmax inverse temperature
      beta2: [0,10] Stage-2 softmax inverse temperature
      sr_lambda: [0,1] Weight of cached SR mapping vs transition-based MB in stage-1 value
      omega_mb: [0,1] Weight on MB/SR mixture vs MF at stage 1
      habit_gain: [0,1] Habit increment for chosen stage-1 action
      habit_decay: [0,1] Habit exponential decay per trial
      beta_loss_boost: [0,1] Beta boost after losses for both stages (based on previous trial)
      forget: [0,1] Forgetting for Q-values and SR cache
      biasA: [0,1] Bias toward spaceship A; additive bias b = (biasA - 0.5)*2
      alphaT: [0,1] Transition learning rate (EWMA)
      perseveration: [0,1] Additive bias to repeat stage-1 action

    Returns:
      Negative log-likelihood of the observed choices.
    """
    import numpy as np  # assumed already imported

    (alpha1, alpha2_win, alpha2_loss, beta1, beta2, sr_lambda, omega_mb,
     habit_gain, habit_decay, beta_loss_boost, forget, biasA, alphaT, perseveration) = model_parameters

    n_trials = len(action_1)

    q1_mf = np.zeros(2)
    q2 = np.zeros((2, 2))
    habit = np.zeros(2)

    T_est = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)

    M = np.full((2, 2), 0.5, dtype=float)

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    bias_add = (biasA - 0.5) * 2.0
    prev_a1 = None
    prev_reward = 1.0
    eps = 1e-12

    for t in range(n_trials):

        q1_mf *= (1.0 - forget)
        q2 *= (1.0 - forget)
        M *= (1.0 - forget)
        habit *= (1.0 - habit_decay)

        beta2_eff = beta2 * (1.0 + beta_loss_boost * (1.0 - prev_reward))

        max_q2 = np.max(q2, axis=1)
        q1_mb = T_est @ max_q2               # transition-based model-based
        q1_sr = M @ max_q2                   # cached SR mapping
        q1_cache = (1.0 - sr_lambda) * q1_mb + sr_lambda * q1_sr
        q1_val = (1.0 - omega_mb) * q1_mf + omega_mb * q1_cache + habit
        q1_val[0] += bias_add
        if prev_a1 is not None:
            q1_val[prev_a1] += perseveration

        beta1_eff = beta1 * (1.0 + beta_loss_boost * (1.0 - prev_reward))

        v1 = q1_val - np.max(q1_val)
        probs1 = np.exp(beta1_eff * v1)
        probs1 /= (np.sum(probs1) + eps)
        a1 = action_1[t]
        p_choice_1[t] = probs1[a1]

        s = state[t]
        q2_val = q2[s].copy()
        v2 = q2_val - np.max(q2_val)
        probs2 = np.exp(beta2_eff * v2)
        probs2 /= (np.sum(probs2) + eps)
        a2 = action_2[t]
        p_choice_2[t] = probs2[a2]

        r = reward[t]

        alpha2 = alpha2_win if r > 0.5 else alpha2_loss
        delta2 = r - q2[s, a2]
        q2[s, a2] += alpha2 * delta2

        target1 = q2[s, a2]
        delta1 = target1 - q1_mf[a1]
        q1_mf[a1] += alpha1 * delta1

        habit[a1] += habit_gain

        onehot_state = np.array([1.0 - s, float(s)])
        T_est[a1] = (1.0 - alphaT) * T_est[a1] + alphaT * onehot_state
        T_est[a1] /= (np.sum(T_est[a1]) + eps)

        M[a1] += alpha1 * (onehot_state - M[a1])

        prev_a1 = a1
        prev_reward = r

    neg_ll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return neg_ll