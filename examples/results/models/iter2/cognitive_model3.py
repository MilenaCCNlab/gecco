def cognitive_model3(action_1, state, action_2, reward, model_parameters):
    """Affect-modulated control with risk sensitivity, lapse, and transition-dependent credit assignment.
    
    Mechanism:
    - Latent "mood/motivation" m_t in [0,1] integrates recent signed prediction errors and scales both learning and choice temperature.
      High mood increases effective beta and learning rates; low mood reduces them (mood_lr, mood_volatility).
    - Risk sensitivity transforms reward via a concave/convex utility u(r) = r^rho (rho in [0,1]).
    - Mixture of MB and MF at stage 1 depends on whether the observed transition was common vs rare:
      on common transitions weight = lambda_credit toward MB; on rare transitions weight shifts toward MF.
    - Choice habit kernel at stage 1 with decay.
    - Loss-dependent temperature boost.
    - Lapse epsilon mixes softmax with uniform choice at both stages.
    - Forgetting toward 0.5 for unchosen values and planet-specific decay.
    - Bias toward spaceship A.
    
    Parameters (all in [0,1] except betas in [0,10]):
    - alpha1: [0,1] base learning rate for stage-1 MF (bootstrapped).
    - alpha2: [0,1] base learning rate for stage-2 MF (from reward utility).
    - beta1: [0,10] base inverse temperature at stage 1.
    - beta2: [0,10] base inverse temperature at stage 2.
    - mood_lr: [0,1] integration rate of mood toward recent signed RPEs.
    - mood_volatility: [0,1] leak/decay of mood to neutral 0.5; higher -> more stable mood.
    - rho: [0,1] risk sensitivity exponent for reward utility u = r^rho.
    - epsilon: [0,1] lapse rate; mixes softmax with uniform choice.
    - habit_gain: [0,1] habit kernel gain at stage 1.
    - habit_decay: [0,1] decay of habit kernel.
    - beta_loss_boost: [0,1] fractional increase of inverse temperature on loss.
    - forget: [0,1] forgetting rate toward 0.5 for unchosen values.
    - biasA: [0,1] bias toward action A (index 0), mapped to [-bmax, +bmax].
    - lambda_credit: [0,1] weight toward MB on common transitions; on rare transitions weight flips to (1-lambda_credit).
    
    Inputs:
    - action_1, state, action_2, reward: arrays of length T.
    - model_parameters: list/array of 14 parameters in the order above.
    
    Returns:
    - Negative log-likelihood of observed choices.
    """
    (alpha1, alpha2, beta1, beta2, mood_lr, mood_volatility, rho, epsilon,
     habit_gain, habit_decay, beta_loss_boost, forget, biasA, lambda_credit) = model_parameters

    n_trials = len(action_1)

    T = np.array([[0.7, 0.3],
                  [0.3, 0.7]])

    q1_mf = np.zeros(2)
    q2_mf = np.zeros((2, 2))
    habit = np.zeros(2)

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    bmax = 2.0
    bias_term = (biasA - 0.5) * 2 * bmax

    m = 0.5

    eps = 1e-12

    for t in range(n_trials):
        s = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r_raw = reward[t]

        r = (r_raw + 0.0) ** max(rho, 1e-6)


        is_common = 1.0 if T[a1, s] >= 0.5 else 0.0

        max_q2 = np.max(q2_mf, axis=1)
        q1_mb = T @ max_q2

        w_mb = lambda_credit * is_common + (1 - lambda_credit) * (1 - is_common)

        mood_gain = 0.5 + m  # in [0.5, 1.5]

        loss_boost = 1.0 + beta_loss_boost * (1.0 - r_raw)
        beta1_eff = beta1 * mood_gain * loss_boost
        beta2_eff = beta2 * mood_gain * loss_boost
        alpha1_eff = np.clip(alpha1 * mood_gain, 0.0, 1.0)
        alpha2_eff = np.clip(alpha2 * mood_gain, 0.0, 1.0)

        q1_comp = w_mb * q1_mb + (1 - w_mb) * q1_mf + habit_gain * habit
        q1_with_bias = q1_comp.copy()
        q1_with_bias[0] += bias_term

        q1s = q1_with_bias - np.max(q1_with_bias)
        exp_q1 = np.exp(beta1_eff * q1s)
        soft_1 = exp_q1 / (np.sum(exp_q1) + eps)
        probs_1 = (1 - epsilon) * soft_1 + epsilon * 0.5
        p_choice_1[t] = max(probs_1[a1], eps)

        q2s = q2_mf[s].copy()
        q2s -= np.max(q2s)
        exp_q2 = np.exp(beta2_eff * q2s)
        soft_2 = exp_q2 / (np.sum(exp_q2) + eps)
        probs_2 = (1 - epsilon) * soft_2 + epsilon * 0.5
        p_choice_2[t] = max(probs_2[a2], eps)




        pe2 = r - q2_mf[s, a2]
        q2_mf[s, a2] += alpha2_eff * pe2

        baseline = 0.5
        for a in range(2):
            if a != a2:
                q2_mf[s, a] = (1 - forget) * q2_mf[s, a] + forget * baseline
        s_other = 1 - s
        q2_mf[s_other, :] = (1 - forget) * q2_mf[s_other, :] + forget * baseline

        target1 = q2_mf[s, a2]
        pe1 = target1 - q1_mf[a1]
        q1_mf[a1] += alpha1_eff * pe1
        a1_other = 1 - a1
        q1_mf[a1_other] = (1 - forget) * q1_mf[a1_other] + forget * baseline

        habit *= habit_decay
        habit[a1] += (1 - habit_decay)

        signed_pe = 0.5 * (np.clip(pe2, -1, 1) + np.clip(pe1, -1, 1))

        m = (mood_volatility * m + (1 - mood_volatility) * 0.5) + mood_lr * signed_pe
        m = float(np.clip(m, 0.0, 1.0))

    eps = 1e-12
    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll