def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """Hybrid MB/MF with learned transitions, entropy-driven exploration, and eligibility.
    Returns the negative log-likelihood of the observed choices.

    Cognitive assumptions:
    - Stage-2 (alien) values are learned model-free from rewards.
    - The agent learns the action→state transition matrix from experience.
    - Stage-1 decisions blend model-based (planning via learned transitions) and model-free values.
    - An eligibility trace propagates immediate reward vs. second-stage value to the stage-1 MF value.
    - Directed exploration at stage 1: a bonus proportional to transition uncertainty (row entropy).
    - Perseveration (stickiness) at both stages.

    Parameters (all in [0,1] except betas in [0,10]):
    - alpha_q: [0,1] learning rate for Q-values at both stages.
    - alpha_t: [0,1] learning rate for the transition matrix (exponential moving average).
    - prior_t: [0,1] convex weight toward a uniform transition prior (0.5, 0.5) when planning.
    - w_mb:    [0,1] weight for model-based value at stage 1 (1-w_mb is MF weight).
    - lam:     [0,1] eligibility mixing of reward vs. second-stage value for stage-1 MF update.
    - kappa_unc1: [0,1] weight of entropy-based directed exploration bonus at stage 1.
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
    - Transition uncertainty per action is quantified by normalized entropy: H(a) in [0,1].
    """
    alpha_q, alpha_t, prior_t, w_mb, lam, kappa_unc1, beta1, beta2, stick1, stick2 = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Initialize learned transition matrix (start at uniform)
    T_learned = np.ones((2, 2)) * 0.5

    # Model-free values
    Q1_mf = np.zeros(2)          # stage-1 MF values for A/U
    Q2_mf = np.zeros((2, 2))     # stage-2 MF values for each planet's two aliens

    # Stickiness memory
    prev_a1 = None
    prev_a2_by_state = [None, None]

    loglik = 0.0

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # Planning with a shrinkage prior toward uniform transitions
        T_plan = (1.0 - prior_t) * T_learned + prior_t * 0.5

        # Entropy-based directed exploration bonus for each stage-1 action
        # Normalize entropy by log(2) to map to [0,1]
        H_rows = np.zeros(2)
        for a in range(2):
            row = T_plan[a]
            row = np.clip(row, eps, 1.0)
            H = -np.sum(row * np.log(row))
            H_rows[a] = H / np.log(2.0)

        # Model-based value via planning: expected max Q2 under T_plan
        max_Q2 = np.max(Q2_mf, axis=1)          # shape (2,)
        Q1_mb = T_plan @ max_Q2                 # shape (2,)

        # Combine MB and MF at stage 1, add uncertainty bonus
        Q1_comb = w_mb * Q1_mb + (1.0 - w_mb) * Q1_mf + kappa_unc1 * H_rows

        # Stage-1 choice with perseveration
        bias1 = np.zeros(2)
        if prev_a1 is not None:
            bias1[prev_a1] += stick1

        logits1 = beta1 * Q1_comb + bias1
        logits1 -= np.max(logits1)
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        loglik += np.log(p1[a1] + eps)

        # Stage-2 choice with perseveration within reached state
        bias2 = np.zeros(2)
        if prev_a2_by_state[s] is not None:
            bias2[prev_a2_by_state[s]] += stick2

        logits2 = beta2 * Q2_mf[s] + bias2
        logits2 -= np.max(logits2)
        p2 = np.exp(logits2)
        p2 /= np.sum(p2)
        loglik += np.log(p2[a2] + eps)

        # Update stage-2 MF values
        delta2 = r - Q2_mf[s, a2]
        Q2_mf[s, a2] += alpha_q * delta2

        # Eligibility-based update to stage-1 MF values
        target_s1 = (1.0 - lam) * Q2_mf[s, a2] + lam * r
        delta1 = target_s1 - Q1_mf[a1]
        Q1_mf[a1] += alpha_q * delta1

        # Update learned transitions via exponential moving average and renormalize
        T_learned[a1, :] *= (1.0 - alpha_t)
        T_learned[a1, s] += alpha_t
        T_learned[a1, :] /= np.sum(T_learned[a1, :])

        # Update stickiness memory
        prev_a1 = a1
        prev_a2_by_state[s] = a2

    return -loglik


def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """Successor-Representation (SR) + Model-Free hybrid with asymmetric learning and lapses.
    Returns the negative log-likelihood of the observed choices.

    Cognitive assumptions:
    - Stage-2 (alien) values are learned model-free with asymmetric learning rates for wins/losses.
    - The agent learns a successor representation M over states for each stage-1 action.
      Here, M[a, s] approximates the discounted expected occupancy of planet s after choosing action a.
    - Stage-1 values are a weighted combination of SR-derived values (planning-free) and MF stage-1 values.
    - Eligibility trace mixes immediate reward and second-stage value in the MF stage-1 update.
    - Lapse (stimulus-independent) choice noise at both stages plus a static bias toward spaceship A.

    Parameters (all in [0,1] except betas in [0,10]):
    - alpha_pos: [0,1] learning rate when reward=1 updates Q2 and MF Q1.
    - alpha_neg: [0,1] learning rate when reward=0 updates Q2 and MF Q1.
    - gamma_sr:  [0,1] discount factor for SR (controls spread beyond immediate state).
    - lam_sr:    [0,1] eligibility mixing of reward vs. second-stage value for MF stage-1 update.
    - w_sr:      [0,1] weight of SR value at stage 1 (1-w_sr is MF stage-1 weight).
    - beta1:     [0,10] inverse temperature for stage-1 softmax.
    - beta2:     [0,10] inverse temperature for stage-2 softmax.
    - eps1:      [0,1] lapse rate at stage 1 (probability of random choice).
    - eps2:      [0,1] lapse rate at stage 2 (probability of random choice).
    - bias_side: [0,1] static bias toward spaceship A (action 0); applied as +/- bias in logits.

    Inputs:
    - action_1: array-like of length n_trials, 0/1 indicating spaceship A/U chosen.
    - state:    array-like of length n_trials, 0/1 indicating planet X/Y reached.
    - action_2: array-like of length n_trials, 0/1 indicating alien chosen on that planet.
    - reward:   array-like of length n_trials, 0/1 coin outcome.
    - model_parameters: iterable with parameters in the order above.

    Notes:
    - SR update uses TD(0) toward the one-step successor e_s plus discounted continuation (here terminal).
      M[a, :] <- M[a, :] + alpha_sr * (e_s + gamma_sr*0 - M[a, :]),
      where alpha_sr is set to the mean of alpha_pos and alpha_neg to keep parameter parsimony.
    - Lapse implements p = (1-eps)*softmax + eps*uniform.
    """
    alpha_pos, alpha_neg, gamma_sr, lam_sr, w_sr, beta1, beta2, eps1, eps2, bias_side = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Learning rates
    alpha_sr = 0.5 * (alpha_pos + alpha_neg)

    # Successor representation over planets for each stage-1 action
    M = np.zeros((2, 2))  # M[a1, s]

    # Model-free values
    Q1_mf = np.zeros(2)
    Q2_mf = np.zeros((2, 2))

    loglik = 0.0

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # SR-derived value: expected max second-stage value under M
        max_Q2 = np.max(Q2_mf, axis=1)      # shape (2,)
        Q1_sr = M @ max_Q2                  # shape (2,)

        # Combine SR and MF for stage 1, add static side bias toward action 0
        bias_vec = np.array([bias_side, -bias_side])
        Q1_comb = w_sr * Q1_sr + (1.0 - w_sr) * Q1_mf

        # Stage-1 choice with lapse
        logits1 = beta1 * Q1_comb + bias_vec
        logits1 -= np.max(logits1)
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        p1 = (1.0 - eps1) * p1 + eps1 * 0.5
        loglik += np.log(p1[a1] + eps)

        # Stage-2 choice with lapse
        logits2 = beta2 * Q2_mf[s]
        logits2 -= np.max(logits2)
        p2 = np.exp(logits2)
        p2 /= np.sum(p2)
        p2 = (1.0 - eps2) * p2 + eps2 * 0.5
        loglik += np.log(p2[a2] + eps)

        # Asymmetric learning rate for stage-2 MF update
        alpha_q = alpha_pos if r > 0.5 else alpha_neg
        delta2 = r - Q2_mf[s, a2]
        Q2_mf[s, a2] += alpha_q * delta2

        # Eligibility-based update to stage-1 MF values with asymmetry
        target_s1 = (1.0 - lam_sr) * Q2_mf[s, a2] + lam_sr * r
        delta1 = target_s1 - Q1_mf[a1]
        Q1_mf[a1] += alpha_q * delta1

        # SR TD(0) update toward one-step successor of the observed state
        e_s = np.zeros(2)
        e_s[s] = 1.0
        # No continuation after reaching planet (terminal), but allow small gamma_sr to regularize
        target_sr = e_s + gamma_sr * 0.0
        M[a1, :] += alpha_sr * (target_sr - M[a1, :])

    return -loglik


def cognitive_model3(action_1, state, action_2, reward, model_parameters):
    """Kalman-filtered reward beliefs with fixed transition belief, directed exploration, and decaying stickiness.
    Returns the negative log-likelihood of the observed choices.

    Cognitive assumptions:
    - Reward contingencies for each alien drift over time; the agent tracks them with a (scalar) Kalman filter.
      This yields both a mean value and an uncertainty for each alien.
    - Stage-2 choices use softmax over mean values plus an uncertainty bonus (directed exploration).
    - Stage-1 planning uses a fixed subjective common-transition parameter c_comm to compute MB values,
      and also propagates an uncertainty bonus from second-stage beliefs.
    - Perseveration kernels at both stages decay over trials (leaky stickiness).

    Parameters (all in [0,1] except betas in [0,10]):
    - q_vol:   [0,1] process (diffusion) variance controlling volatility of rewards.
    - r_obs:   [0,1] observation noise variance used by the Kalman filter.
    - c_comm:  [0,1] subjective probability of common transition (A->X, U->Y).
    - beta1:   [0,10] inverse temperature for stage-1 softmax.
    - beta2:   [0,10] inverse temperature for stage-2 softmax.
    - eta1:    [0,1] weight of uncertainty bonus at stage 1 (propagated from stage 2).
    - eta2:    [0,1] weight of uncertainty bonus at stage 2 (for chosen state's aliens).
    - stick1:  [0,1] strength scaling the stage-1 perseveration kernel.
    - stick2:  [0,1] strength scaling the stage-2 perseveration kernel.
    - decay_k: [0,1] per-trial decay of the perseveration kernels (higher = faster decay).

    Inputs:
    - action_1: array-like of length n_trials, 0/1 indicating spaceship A/U chosen.
    - state:    array-like of length n_trials, 0/1 indicating planet X/Y reached.
    - action_2: array-like of length n_trials, 0/1 indicating alien chosen on that planet.
    - reward:   array-like of length n_trials, 0/1 coin outcome.
    - model_parameters: iterable with parameters in the order above.

    Notes:
    - Kalman filter (scalar per option):
        Prior:  mu <- mu,    P <- P + q_vol
        Gain:   K = P / (P + r_obs + eps)
        Update: mu <- mu + K*(r - mu),   P <- (1 - K)*P
    - Stage-1 transition belief (fixed):
        T = [[c_comm, 1-c_comm], [1-c_comm, c_comm]] for actions A and U respectively.
    - Uncertainty bonuses:
        Stage 2: add eta2 * sqrt(P[s, :]) to the logits.
        Stage 1: add eta1 * E_T[max_a (mu + eta2*sqrt(P))] under T.
    - Perseveration kernels:
        K1 (size 2) and K2 (size 2x2) decay each trial by (1 - decay_k), then increment at chosen entries by 1.
    """
    q_vol, r_obs, c_comm, beta1, beta2, eta1, eta2, stick1, stick2, decay_k = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Transition belief
    T = np.array([[c_comm, 1.0 - c_comm],
                  [1.0 - c_comm, c_comm]], dtype=float)

    # Kalman filter states for each planet's two aliens
    mu = np.zeros((2, 2))     # mean reward belief
    P = np.ones((2, 2)) * 0.25  # initial uncertainty (moderate)

    # Perseveration kernels
    K1 = np.zeros(2)
    K2 = np.zeros((2, 2))

    loglik = 0.0

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # Prior diffusion for all aliens
        P = P + q_vol

        # Stage-2 values with uncertainty bonus
        bonus2 = eta2 * np.sqrt(np.maximum(P, 0.0))
        V2 = mu + bonus2

        # Stage-1 MB value: expected max over second-stage options under T
        max_V2 = np.max(V2, axis=1)  # shape (2,)
        Q1_mb = T @ max_V2

        # Add stage-1 uncertainty bonus propagated from stage 2:
        # Use the expected max uncertainty term under T as additional directed exploration
        max_bonus2 = np.max(bonus2, axis=1)
        Q1_bonus = T @ max_bonus2
        Q1 = Q1_mb + eta1 * Q1_bonus

        # Stage-1 softmax with decaying perseveration kernel
        K1 *= (1.0 - decay_k)
        logits1 = beta1 * Q1 + stick1 * K1
        logits1 -= np.max(logits1)
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        loglik += np.log(p1[a1] + eps)

        # Stage-2 softmax with decaying perseveration kernel
        K2 *= (1.0 - decay_k)
        logits2 = beta2 * V2[s] + stick2 * K2[s]
        logits2 -= np.max(logits2)
        p2 = np.exp(logits2)
        p2 /= np.sum(p2)
        loglik += np.log(p2[a2] + eps)

        # Update perseveration kernels with the chosen actions
        K1[a1] += 1.0
        K2[s, a2] += 1.0

        # Kalman filter update for the observed alien only
        # Observation variance
        S = P[s, a2] + r_obs + eps
        K_gain = P[s, a2] / S
        mu[s, a2] = mu[s, a2] + K_gain * (r - mu[s, a2])
        P[s, a2] = (1.0 - K_gain) * P[s, a2]

    return -loglik