def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """
    Hybrid MB/MF with learned transitions, dynamic habit, perseveration, forgetting, and loss-contingent choice temperature.
    Concept:
      - Stage 2: model-free Q-learning with forgetting.
      - Stage 1: hybrid of model-based (using learned transition model) and model-free values.
      - Habit strength accumulates for chosen stage-1 action and decays over time.
      - Perseveration bias to repeat the previous stage-1 choice.
      - Transition model is learned online (EWMA).
      - Loss-contingent increase in choice determinism (beta boost after losses).
      - Bias toward spaceship A.

    Parameters (in order):
      alpha1: [0,1] Stage-1 MF learning rate
      alpha2: [0,1] Stage-2 learning rate
      beta1: [0,10] Softmax inverse temperature at stage 1
      beta2: [0,10] Softmax inverse temperature at stage 2
      omega: [0,1] Weight of model-based value in stage-1 hybrid value
      habit_gain: [0,1] Increment added to habit for the chosen stage-1 action each trial
      habit_decay: [0,1] Exponential decay of habit each trial
      beta_loss_boost: [0,1] Multiplier for beta after prior loss: beta_eff = beta * (1 + boost*(1 - prev_reward))
      forget: [0,1] Exponential decay toward zero for MF Q-values each trial (both stages)
      biasA: [0,1] Bias toward spaceship A; transformed to centered additive bias b = (biasA - 0.5)*2
      alphaT: [0,1] Transition learning rate (EWMA toward observed state given chosen stage-1 action)
      perseveration: [0,1] Additive bias for repeating previous stage-1 action

    Returns:
      Negative log-likelihood of the observed stage-1 and stage-2 choices.
    """
    import numpy as np  # assumed already imported per guardrail; included here only for clarity of usage

    alpha1, alpha2, beta1, beta2, omega, habit_gain, habit_decay, beta_loss_boost, forget, biasA, alphaT, perseveration = model_parameters

    n_trials = len(action_1)

    q1_mf = np.zeros(2)                 # model-free Q-values at stage 1
    q2 = np.zeros((2, 2))               # stage-2 Q-values: [state, action]
    habit = np.zeros(2)                 # habit strength for stage-1 actions

    T_est = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    bias_add = (biasA - 0.5) * 2.0
    prev_a1 = None
    prev_reward = 1.0  # initialize as a "win" so first-trial betas are unboosted

    eps = 1e-12

    for t in range(n_trials):

        q1_mf *= (1.0 - forget)
        q2 *= (1.0 - forget)
        habit *= (1.0 - habit_decay)

        max_q2 = np.max(q2, axis=1)               # shape (2,)
        q1_mb = T_est @ max_q2                    # shape (2,)

        q1_val = (1.0 - omega) * q1_mf + omega * q1_mb + habit

        q1_val[0] += bias_add

        if prev_a1 is not None:
            q1_val[prev_a1] += perseveration

        beta1_eff = beta1 * (1.0 + beta_loss_boost * (1.0 - prev_reward))

        v = q1_val - np.max(q1_val)
        expv = np.exp(beta1_eff * v)
        probs1 = expv / (np.sum(expv) + eps)
        a1 = action_1[t]
        p_choice_1[t] = probs1[a1]

        s = state[t]
        q2_val = q2[s].copy()

        beta2_eff = beta2 * (1.0 + beta_loss_boost * (1.0 - prev_reward))
        v2 = q2_val - np.max(q2_val)
        expv2 = np.exp(beta2_eff * v2)
        probs2 = expv2 / (np.sum(expv2) + eps)
        a2 = action_2[t]
        p_choice_2[t] = probs2[a2]

        r = reward[t]

        delta2 = r - q2[s, a2]
        q2[s, a2] += alpha2 * delta2

        target1 = q2[s, a2]
        delta1 = target1 - q1_mf[a1]
        q1_mf[a1] += alpha1 * delta1

        habit[a1] += habit_gain

        onehot_state = np.array([1.0 - s, float(s)])  # 2 states, s in {0,1}
        T_est[a1] = (1.0 - alphaT) * T_est[a1] + alphaT * onehot_state

        T_est[a1] /= (np.sum(T_est[a1]) + eps)

        prev_a1 = a1
        prev_reward = r

    neg_ll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return neg_ll