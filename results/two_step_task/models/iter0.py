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

    # Initialize learned transition model; start with symmetric (uninformative) 0.5/0.5
    T = np.ones((2, 2)) * 0.5

    # Model-free values
    Q1_mf = np.zeros(2)          # stage-1 MF values for A/U
    Q2_mf = np.zeros((2, 2))     # stage-2 MF values for each planet's two aliens

    # Perseveration trackers
    prev_a1 = None
    prev_a2_by_state = [None, None]

    loglik = 0.0

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # Model-based stage-1 values from learned transitions and current MF stage-2 values
        max_Q2 = np.max(Q2_mf, axis=1)          # shape (2,)
        Q1_mb = T @ max_Q2                      # shape (2,)

        # Combine MB and MF for stage-1 decision
        Q1_comb = w_mb * Q1_mb + (1.0 - w_mb) * Q1_mf

        # Add perseveration biases (as additive choice biases)
        bias1 = np.zeros(2)
        if prev_a1 is not None:
            bias1[prev_a1] += stick1

        # Stage-1 choice probability
        logits1 = beta1 * Q1_comb + bias1
        logits1 -= np.max(logits1)  # numerical stability
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        loglik += np.log(p1[a1] + eps)

        # Stage-2 choice probability within the observed state
        bias2 = np.zeros(2)
        if prev_a2_by_state[s] is not None:
            bias2[prev_a2_by_state[s]] += stick2

        logits2 = beta2 * Q2_mf[s] + bias2
        logits2 -= np.max(logits2)
        p2 = np.exp(logits2)
        p2 /= np.sum(p2)
        loglik += np.log(p2[a2] + eps)

        # TD updates
        # Stage-2 MF update
        delta2 = r - Q2_mf[s, a2]
        Q2_mf[s, a2] += alpha_q * delta2

        # Stage-1 MF update with eligibility trace
        # Use a combination of bootstrapping to the chosen stage-2 action and direct reward via lam
        target_s1 = (1.0 - lam) * Q2_mf[s, a2] + lam * r
        delta1 = target_s1 - Q1_mf[a1]
        Q1_mf[a1] += alpha_q * delta1

        # Transition learning: update chosen row toward the observed state (one-hot)
        T[a1, :] *= (1.0 - alpha_t)
        T[a1, s] += alpha_t
        # Ensure normalization (for numerical safety)
        T[a1, :] /= np.sum(T[a1, :])

        # Forgetting (value decay toward 0)
        Q1_mf *= (1.0 - decay)
        Q2_mf *= (1.0 - decay)

        # Update perseveration trackers
        prev_a1 = a1
        prev_a2_by_state[s] = a2

    return -loglik


def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """Bayes-inspired alien learning with surprise-modulated learning, perceived transition reliability,
    risk-weighted expectations, perseveration, and lapses.
    Returns the negative log-likelihood of the observed choices.

    Cognitive assumptions:
    - The agent tracks each alien's reward probability p and updates it with a learning rate
      amplified by outcome surprise (|r - p|).
    - The action→state transition model is learned, but the agent discounts its reliability (gamma),
      blending it with a uniform transition belief.
    - Expected values are risk-transformed using a probability weighting parameter (rho).
    - Choices include perseveration and a small lapse probability (epsilon).

    Parameters:
    - alpha:  [0,1] base learning rate for alien reward probabilities and transition learning.
    - surprise_gain: [0,1] scales the boost of learning rate by surprise magnitude.
    - gamma:  [0,1] perceived transition reliability (1=trust learned transitions, 0=uniform).
    - rho:    [0,1] risk/probability-weighting strength for transforming p into subjective value.
    - beta1:  [0,10] inverse temperature for stage-1 softmax.
    - beta2:  [0,10] inverse temperature for stage-2 softmax.
    - persev: [0,1] perseveration bias added to the previously chosen action (per stage).
    - epsilon: [0,1] lapse probability; with probability epsilon, choose uniformly at random.

    Inputs:
    - action_1: array-like of length n_trials, 0/1 for spaceship choice.
    - state:    array-like of length n_trials, 0/1 for planet reached.
    - action_2: array-like of length n_trials, 0/1 for alien choice within state.
    - reward:   array-like of length n_trials, 0/1 outcome.
    - model_parameters: iterable with parameters in the order above.
    """
    alpha, surprise_gain, gamma, rho, beta1, beta2, persev, epsilon = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Learned transition model (initialized uninformative)
    T = np.ones((2, 2)) * 0.5

    # Alien success probability estimates p[s, a2]
    P_alien = np.ones((2, 2)) * 0.5

    # Perseveration trackers
    prev_a1 = None
    prev_a2_by_state = [None, None]

    loglik = 0.0

    def prob_weight(p, rho_):
        # Symmetric probability weighting mapping in [0,1] with rho controlling curvature.
        # For rho=0, identity; as rho->1, curve approaches step-like sharpening around 0.5.
        p = np.clip(p, eps, 1.0 - eps)
        w = p**(1.0 - rho_) / (p**(1.0 - rho_) + (1.0 - p)**(1.0 - rho_))
        return w

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # Effective transition beliefs (discount toward uniform by gamma)
        T_eff = gamma * T + (1.0 - gamma) * 0.5

        # Risk-weighted (probability-weighted) alien values
        V2 = prob_weight(P_alien, rho)  # shape (2,2)

        # Model-based stage-1 values via planning
        max_V2 = np.max(V2, axis=1)
        Q1_mb = T_eff @ max_V2

        # Stage-1 softmax with perseveration and lapses
        bias1 = np.zeros(2)
        if prev_a1 is not None:
            bias1[prev_a1] += persev
        logits1 = beta1 * Q1_mb + bias1
        logits1 -= np.max(logits1)
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        p1 = (1.0 - epsilon) * p1 + epsilon * 0.5  # mixture with random choice
        loglik += np.log(p1[a1] + eps)

        # Stage-2 softmax with perseveration and lapses
        bias2 = np.zeros(2)
        if prev_a2_by_state[s] is not None:
            bias2[prev_a2_by_state[s]] += persev
        logits2 = beta2 * V2[s] + bias2
        logits2 -= np.max(logits2)
        p2 = np.exp(logits2)
        p2 /= np.sum(p2)
        p2 = (1.0 - epsilon) * p2 + epsilon * 0.5
        loglik += np.log(p2[a2] + eps)

        # Surprise-modulated learning for alien success probabilities
        pe = r - P_alien[s, a2]
        lr_eff = alpha * (1.0 + surprise_gain * np.abs(pe))
        P_alien[s, a2] += lr_eff * pe
        # Keep in [0,1]
        P_alien[s, a2] = np.clip(P_alien[s, a2], 0.0, 1.0)

        # Transition learning toward observed state
        T[a1, :] *= (1.0 - alpha)
        T[a1, s] += alpha
        T[a1, :] /= np.sum(T[a1, :])

        # Update perseveration trackers
        prev_a1 = a1
        prev_a2_by_state[s] = a2

    return -loglik


def cognitive_model3(action_1, state, action_2, reward, model_parameters):
    """Adaptive meta-control between MB and MF with uncertainty-driven temperature,
    eligibility trace, transition learning, perseveration, and lapses.
    Returns the negative log-likelihood of the observed choices.

    Cognitive assumptions:
    - A meta-controller maintains a dynamic weight w_t over MB vs MF control at stage 1.
      This weight adapts based on recent surprise and reward rate (captured via volatility-like updates).
    - The softmax temperature at stage 1 adapts to decision uncertainty (entropy of action preferences).
    - Stage-2 values are learned model-free; stage-1 MF values bootstrap from stage-2.
    - Transitions are learned online.
    - Choices include perseveration and lapses.

    Parameters:
    - alpha_mf: [0,1] learning rate for MF Q-values (both stages).
    - alpha_s1: [0,1] learning rate specifically for stage-1 MF values.
    - alpha_t:  [0,1] transition learning rate.
    - lam:      [0,1] eligibility trace for propagating reward to stage-1 MF values.
    - w0:       [0,1] initial MB weight of the meta-controller.
    - tau:      [0,1] adaptation rate of the MB weight (meta-controller volatility).
    - beta0:    [0,10] baseline inverse temperature at stage 1.
    - beta_gain:[0,10] gain scaling how much low uncertainty increases beta at stage 1.
    - beta2:    [0,10] inverse temperature at stage 2.
    - kappa:    [0,1] perseveration strength (both stages).
    - epsilon:  [0,1] lapse probability (random choice mixture).

    Inputs:
    - action_1: array-like of length n_trials, 0/1 for spaceship choice.
    - state:    array-like of length n_trials, 0/1 for planet reached.
    - action_2: array-like of length n_trials, 0/1 for alien choice.
    - reward:   array-like of length n_trials, 0/1 outcome.
    - model_parameters: iterable with parameters in the order above.
    """
    (alpha_mf, alpha_s1, alpha_t, lam, w0, tau,
     beta0, beta_gain, beta2, kappa, epsilon) = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Learned transition model
    T = np.ones((2, 2)) * 0.5

    # MF values
    Q1_mf = np.zeros(2)
    Q2_mf = np.zeros((2, 2))

    # Meta-controller state
    w = float(w0)     # current MB weight
    avg_rew = 0.5     # running reward rate (start neutral)
    rr_alpha = 0.1    # fixed small step for reward-rate tracking (used to drive meta-control)

    # Perseveration trackers
    prev_a1 = None
    prev_a2_by_state = [None, None]

    loglik = 0.0

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # Model-based Q at stage 1 from current MF stage-2 values and learned transitions
        max_Q2 = np.max(Q2_mf, axis=1)   # (2,)
        Q1_mb = T @ max_Q2               # (2,)

        # Combine MF and MB according to current meta-controller weight
        Q1_comb = w * Q1_mb + (1.0 - w) * Q1_mf

        # Uncertainty (entropy) over stage-1 actions at unit temperature, normalized to [0,1]
        logits_tmp = Q1_comb - np.max(Q1_comb)
        p_tmp = np.exp(logits_tmp)
        p_tmp /= np.sum(p_tmp)
        entropy = -np.sum(p_tmp * (np.log(p_tmp + eps)))
        entropy /= np.log(2.0)  # normalize by log(2)

        # Adaptive temperature: higher when uncertainty high, lower when certainty high
        beta1_t = beta0 + beta_gain * (1.0 - entropy)

        # Stage-1 softmax with perseveration and lapses
        bias1 = np.zeros(2)
        if prev_a1 is not None:
            bias1[prev_a1] += kappa
        logits1 = beta1_t * Q1_comb + bias1
        logits1 -= np.max(logits1)
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        p1 = (1.0 - epsilon) * p1 + epsilon * 0.5
        loglik += np.log(p1[a1] + eps)

        # Stage-2 softmax with perseveration and lapses
        bias2 = np.zeros(2)
        if prev_a2_by_state[s] is not None:
            bias2[prev_a2_by_state[s]] += kappa
        logits2 = beta2 * Q2_mf[s] + bias2
        logits2 -= np.max(logits2)
        p2 = np.exp(logits2)
        p2 /= np.sum(p2)
        p2 = (1.0 - epsilon) * p2 + epsilon * 0.5
        loglik += np.log(p2[a2] + eps)

        # MF updates
        # Stage-2
        delta2 = r - Q2_mf[s, a2]
        Q2_mf[s, a2] += alpha_mf * delta2

        # Stage-1 MF with eligibility trace
        target_s1 = (1.0 - lam) * Q2_mf[s, a2] + lam * r
        delta1 = target_s1 - Q1_mf[a1]
        Q1_mf[a1] += alpha_s1 * delta1

        # Transition learning
        T[a1, :] *= (1.0 - alpha_t)
        T[a1, s] += alpha_t
        T[a1, :] /= np.sum(T[a1, :])

        # Meta-control update: adjust w toward higher MB when surprises are large and reward rate is high
        pe_mag = abs(delta2)
        # Drive signal combines surprise and reward rate relative to neutral 0.5
        drive = 0.5 * pe_mag + 0.5 * max(0.0, avg_rew - 0.5)
        # Move w toward drive (bounded in [0,1])
        w = w + tau * (drive - w)
        w = min(1.0, max(0.0, w))

        # Update running reward rate
        avg_rew = (1.0 - rr_alpha) * avg_rew + rr_alpha * r

        # Perseveration trackers
        prev_a1 = a1
        prev_a2_by_state[s] = a2

    return -loglik