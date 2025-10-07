def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """Transition-learning model combining Successor-like cached prediction with MF control, plus habit and asymmetry.
    
    Mechanism:
    - Learns transition matrix T_hat online with learning rate trans_alpha.
    - Stage-2 MF learning with asymmetric learning rates for reward vs omission.
    - Stage-1 value is a dynamic mixture of:
        • Cached "successor-like" predictor V_SR[a] := m[a]·v, where m[a] is a learned state-occupancy vector
          updated from experienced states (sr_alpha, sr_lambda), and v is current max Q2 per state.
        • MF bootstrap value Q1_MF.
      Mixture weight theta_t is higher when transitions are certain (low entropy of T_hat row for chosen action).
    - Choice habit kernel at stage 1 (habit_gain, habit_decay).
    - Loss-dependent temperature boost (beta_loss_boost) for both stages.
    - Forgetting toward 0.5 for unchosen Q-values (both stages).
    - Bias toward spaceship A (biasA).
    
    Parameters (all in [0,1] except betas in [0,10]):
    - alpha_r: [0,1] learning rate for MF when reward=1.
    - alpha_pun: [0,1] learning rate for MF when reward=0 (omission).
    - beta1: [0,10] inverse temperature at stage 1.
    - beta2: [0,10] inverse temperature at stage 2.
    - sr_alpha: [0,1] learning rate for the SR-like state-occupancy vector m[a].
    - sr_lambda: [0,1] eligibility persistence for m[a] (controls carry-over from previous m[a]).
    - trans_alpha: [0,1] learning rate for transition matrix T_hat updates.
    - habit_gain: [0,1] gain for stage-1 habit kernel.
    - habit_decay: [0,1] decay of habit kernel.
    - beta_loss_boost: [0,1] fractional increase of inverse temperature on loss.
    - forget: [0,1] forgetting rate toward 0.5 for unchosen values.
    - biasA: [0,1] bias toward action A (index 0), mapped to [-bmax, +bmax].
    
    Inputs:
    - action_1, state, action_2, reward: arrays of length T.
    - model_parameters: list/array of 12 parameters in the order above.
    
    Returns:
    - Negative log-likelihood of observed choices.
    """
    (alpha_r, alpha_pun, beta1, beta2, sr_alpha, sr_lambda, trans_alpha,
     habit_gain, habit_decay, beta_loss_boost, forget, biasA) = model_parameters

    n_trials = len(action_1)

    T_hat = np.array([[0.5, 0.5],
                      [0.5, 0.5]], dtype=float)

    q1_mf = np.zeros(2)
    q2_mf = np.zeros((2, 2))
    habit = np.zeros(2)

    m = np.zeros((2, 2))  # each row a: m[a, s] ~ expected occupancy of state s after choosing action a

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    bmax = 2.0
    bias_term = (biasA - 0.5) * 2 * bmax

    eps = 1e-12

    for t in range(n_trials):
        s = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r = reward[t]

        v_states = np.max(q2_mf, axis=1)  # shape (2,)

        v_sr = m @ v_states  # shape (2,)