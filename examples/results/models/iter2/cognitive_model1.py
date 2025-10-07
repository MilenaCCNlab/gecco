def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """Dual-controller arbitration via surprise-driven weighting with habit, forgetting, and loss-dependent temperature.
    
    Mechanism:
    - Stage-2 model-free learning with learning rates alpha1 (for stage-1 MF bootstrap) and alpha2 (stage-2 MF).
    - Model-based evaluation at stage 1 from a fixed transition model.
    - Dynamic arbitration weight w_t between MB and MF driven by recent surprise (unsigned stage-2 RPE):
        when surprise is high, rely more on MF; when low, rely more on MB.
    - Choice habit (choice kernel) at stage 1 with gain and decay.
    - Forgetting for unchosen Q-values toward a neutral baseline (0.5).
    - Loss-dependent boost of inverse temperature on both stages.
    - Action bias toward spaceship A (action 0).
    
    Parameters (all in [0,1] except betas in [0,10]):
    - alpha1: [0,1] stage-1 MF learning rate via bootstrapping from stage-2 value.
    - alpha2: [0,1] stage-2 MF learning rate from reward.
    - beta1: [0,10] inverse temperature at stage 1.
    - beta2: [0,10] inverse temperature at stage 2.
    - w0: [0,1] initial arbitration weight toward model-based control at stage 1.
    - kappa_arbitration: [0,1] update rate for arbitration weight based on surprise.
    - habit_gain: [0,1] gain for the choice habit signal at stage 1.
    - habit_decay: [0,1] decay of habit across trials; higher means slower decay.
    - beta_loss_boost: [0,1] fractional increase of inverse temperature on loss (reward==0).
    - forget: [0,1] forgetting rate toward 0.5 for unchosen Q-values (both stages).
    - biasA: [0,1] bias toward action A (index 0); mapped to signed bias in [-bmax, +bmax].
    
    Inputs:
    - action_1: array-like of length T with values in {0,1} for stage-1 choices.
    - state: array-like of length T with values in {0,1} for planet X/Y.
    - action_2: array-like of length T with values in {0,1} for stage-2 alien choices.
    - reward: array-like of length T with scalar rewards (e.g., 0/1).
    - model_parameters: list or array of 11 parameters in the order above.
    
    Returns:
    - Negative log-likelihood of the observed stage-1 and stage-2 choices under the model.
    """
    alpha1, alpha2, beta1, beta2, w0, kappa, habit_gain, habit_decay, beta_loss_boost, forget, biasA = model_parameters
    n_trials = len(action_1)

    transition_matrix = np.array([[0.7, 0.3],
                                  [0.3, 0.7]])

    q1_mf = np.zeros(2)               # stage-1 MF values for actions A/U
    q2_mf = np.zeros((2, 2))          # stage-2 MF values for states X/Y and actions (aliens)
    habit = np.zeros(2)               # stage-1 habit kernel
    w = w0                            # arbitration weight toward MB

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    bmax = 2.0
    bias_term = (biasA - 0.5) * 2 * bmax  # add to action-0 only

    eps = 1e-12

    for t in range(n_trials):
        s = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r = reward[t]

        max_q2 = np.max(q2_mf, axis=1)  # shape (2,)
        q1_mb = transition_matrix @ max_q2  # shape (2,)


        q1_combined = w * q1_mb + (1 - w) * q1_mf + habit_gain * habit

        q1_with_bias = q1_combined.copy()
        q1_with_bias[0] += bias_term

        loss_boost = 1.0 + beta_loss_boost * (1.0 - r)  # if r=0, boost; if r=1, no boost
        beta1_eff = beta1 * loss_boost
        beta2_eff = beta2 * loss_boost

        q1s = q1_with_bias - np.max(q1_with_bias)
        exp_q1 = np.exp(beta1_eff * q1s)
        probs_1 = exp_q1 / (np.sum(exp_q1) + eps)
        p_choice_1[t] = max(probs_1[a1], eps)

        q2s = q2_mf[s].copy()
        q2s -= np.max(q2s)
        exp_q2 = np.exp(beta2_eff * q2s)
        probs_2 = exp_q2 / (np.sum(exp_q2) + eps)
        p_choice_2[t] = max(probs_2[a2], eps)




        pe2 = r - q2_mf[s, a2]
        q2_mf[s, a2] += alpha2 * pe2

        baseline = 0.5
        for a in range(2):
            if a != a2:
                q2_mf[s, a] = (1 - forget) * q2_mf[s, a] + forget * baseline

        s_other = 1 - s
        q2_mf[s_other, :] = (1 - forget) * q2_mf[s_other, :] + forget * baseline

        target1 = q2_mf[s, a2]
        pe1 = target1 - q1_mf[a1]
        q1_mf[a1] += alpha1 * pe1

        a1_other = 1 - a1
        q1_mf[a1_other] = (1 - forget) * q1_mf[a1_other] + forget * baseline

        habit *= habit_decay
        habit[a1] += (1 - habit_decay)  # accumulate to chosen action


        surprise = min(abs(pe2), 1.0)
        desired_w = 1.0 - surprise
        w = (1 - kappa) * w + kappa * desired_w

    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll