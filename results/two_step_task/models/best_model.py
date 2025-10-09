def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """Hybrid MB/MF with learned transitions, eligibility trace, forgetting, and stickiness.
    Returns the negative log-likelihood of the observed choices.

    Cognitive assumptions:
    - Stage-2 (alien) values are learned model-free from rewards.
    - The agent learns the action→state transition matrix from experience.
    - Stage-1 decisions blend model-based (planning via learned transitions) and model-free values.
    - An eligibility trace lets reward update flow back to the stage-1 choice.
    - Values decay over time (forgetting).
    - Choices exhibit perseveration (stickiness) at both stages.

    Parameters (all in [0,1] except betas in [0,10]):
    - alpha_q: [0,1] learning rate for Q-values at both stages.
    - alpha_t: [0,1] learning rate for the transition matrix.
    - w_mb:    [0,1] weight for model-based value at stage 1 (1-w_mb is MF weight).
    - lam:     [0,1] eligibility trace controlling backpropagation of reward to stage 1.
    - decay:   [0,1] forgetting rate (per trial value decay).
    - beta1:   [0,10] inverse temperature for stage-1 softmax.
    - beta2:   [0,10] inverse temperature for stage-2 softmax.
    - stick1:  [0,1] perseveration strength at stage 1.
    - stick2:  [0,1] perseveration strength at stage 2.

    Inputs:
    - action_1: array-like of length n_trials, 0/1 indicating spaceship A/U chosen.
    - state:    array-like of length n_trials, 0/1 indicating planet X/Y reached.
    - action_2: array-like of length n_trials, 0/1 indicating alien chosen on that planet.
    - reward:   array-like of length n_trials, 0/1 coin outcome.
    - model_parameters: iterable with parameters in the order above.

    Notes:
    - Two actions at stage 1, two states, two actions at stage 2.
    - The transition matrix T has shape (2,2): T[a1, s] = P(state=s | action_1=a1).
    """
    alpha_q, alpha_t, w_mb, lam, decay, beta1, beta2, stick1, stick2 = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    T = np.ones((2, 2)) * 0.5

    Q1_mf = np.zeros(2)          # stage-1 MF values for A/U
    Q2_mf = np.zeros((2, 2))     # stage-2 MF values for each planet's two aliens

    prev_a1 = None
    prev_a2_by_state = [None, None]

    loglik = 0.0

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        max_Q2 = np.max(Q2_mf, axis=1)          # shape (2,)
        Q1_mb = T @ max_Q2                      # shape (2,)

        Q1_comb = w_mb * Q1_mb + (1.0 - w_mb) * Q1_mf

        bias1 = np.zeros(2)
        if prev_a1 is not None:
            bias1[prev_a1] += stick1

        logits1 = beta1 * Q1_comb + bias1
        logits1 -= np.max(logits1)  # numerical stability
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        loglik += np.log(p1[a1] + eps)

        bias2 = np.zeros(2)
        if prev_a2_by_state[s] is not None:
            bias2[prev_a2_by_state[s]] += stick2

        logits2 = beta2 * Q2_mf[s] + bias2
        logits2 -= np.max(logits2)
        p2 = np.exp(logits2)
        p2 /= np.sum(p2)
        loglik += np.log(p2[a2] + eps)


        delta2 = r - Q2_mf[s, a2]
        Q2_mf[s, a2] += alpha_q * delta2


        target_s1 = (1.0 - lam) * Q2_mf[s, a2] + lam * r
        delta1 = target_s1 - Q1_mf[a1]
        Q1_mf[a1] += alpha_q * delta1

        T[a1, :] *= (1.0 - alpha_t)
        T[a1, s] += alpha_t

        T[a1, :] /= np.sum(T[a1, :])

        Q1_mf *= (1.0 - decay)
        Q2_mf *= (1.0 - decay)

        prev_a1 = a1
        prev_a2_by_state[s] = a2

    return -loglik