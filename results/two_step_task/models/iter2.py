def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """Asymmetric learning, counterfactual updating, learned transitions, and lapses.
    Returns the negative log-likelihood of the observed choices.

    Cognitive assumptions:
    - Stage-2 (alien) values are learned with asymmetric learning rates for wins vs. losses.
    - The agent learns the action→state transition matrix from experience.
    - Stage-1 decisions blend model-based (planning via learned transitions) and model-free values.
    - An eligibility trace lets reward update flow back to the stage-1 choice.
    - Counterfactual learning updates the unchosen alien in the visited state toward the guessed
      counterfactual outcome (assumed 1 - r) at a controllable rate.
    - Lapse (tremble) noise mixes softmax choice with uniform choice at both stages.

    Parameters (all in [0,1], except betas in [0,10]):
    - alpha_pos: [0,1] learning rate for positive RPEs (r - Q > 0) at stage 2.
    - alpha_neg: [0,1] learning rate for negative RPEs (r - Q < 0) at stage 2.
    - alpha_t:   [0,1] learning rate for the transition matrix (row-wise delta rule).
    - w_mb:      [0,1] weight for model-based value at stage 1 (1 - w_mb is MF weight).
    - lam:       [0,1] eligibility trace controlling backpropagation of reward to stage 1 MF.
    - cf:        [0,1] counterfactual learning strength for unchosen stage-2 action.
    - beta1:     [0,10] inverse temperature for stage-1 softmax.
    - beta2:     [0,10] inverse temperature for stage-2 softmax.
    - lapse1:    [0,1] lapse rate mixing stage-1 softmax with uniform choice.
    - lapse2:    [0,1] lapse rate mixing stage-2 softmax with uniform choice.

    Inputs:
    - action_1: array-like of length n_trials, 0/1 indicating spaceship A/U chosen.
    - state:    array-like of length n_trials, 0/1 indicating planet X/Y reached.
    - action_2: array-like of length n_trials, 0/1 indicating alien chosen on that planet.
    - reward:   array-like of length n_trials, 0/1 coin outcome.
    - model_parameters: iterable with parameters in the order above.
    """
    alpha_pos, alpha_neg, alpha_t, w_mb, lam, cf, beta1, beta2, lapse1, lapse2 = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Initialize learned transition matrix and Q-values
    T = np.ones((2, 2)) * 0.5  # start ignorant about transitions
    Q1_mf = np.zeros(2)        # stage-1 model-free values
    Q2 = np.zeros((2, 2))      # stage-2 values (per state, per alien)

    loglik = 0.0

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # Model-based planning at stage 1
        max_Q2 = np.max(Q2, axis=1)  # shape (2,)
        Q1_mb = T @ max_Q2           # shape (2,)

        # Combine MB and MF
        Q1 = w_mb * Q1_mb + (1.0 - w_mb) * Q1_mf

        # Stage-1 choice probability with lapse
        logits1 = beta1 * Q1
        logits1 -= np.max(logits1)
        p1_soft = np.exp(logits1)
        p1_soft /= np.sum(p1_soft)
        p1 = (1.0 - lapse1) * p1_soft + lapse1 * 0.5
        loglik += np.log(p1[a1] + eps)

        # Stage-2 choice probability with lapse
        logits2 = beta2 * Q2[s]
        logits2 -= np.max(logits2)
        p2_soft = np.exp(logits2)
        p2_soft /= np.sum(p2_soft)
        p2 = (1.0 - lapse2) * p2_soft + lapse2 * 0.5
        loglik += np.log(p2[a2] + eps)

        # Stage-2 update (asymmetric learning)
        pe2 = r - Q2[s, a2]
        alpha2 = alpha_pos if pe2 >= 0.0 else alpha_neg
        Q2[s, a2] += alpha2 * pe2

        # Counterfactual update for unchosen alien on the visited state
        a2_other = 1 - a2
        r_cf = 1.0 - r  # simple heuristic: counterfactual assumed opposite of observed
        pe2_cf = r_cf - Q2[s, a2_other]
        alpha2_cf = (alpha_pos if pe2_cf >= 0.0 else alpha_neg) * cf
        Q2[s, a2_other] += alpha2_cf * pe2_cf

        # Backpropagate to stage-1 MF via eligibility trace
        # Target mixes immediate reward and learned stage-2 value
        target_s1 = (1.0 - lam) * Q2[s, a2] + lam * r
        pe1 = target_s1 - Q1_mf[a1]
        # Use symmetric learning at stage 1 with rate equal to mean of pos/neg
        alpha1 = 0.5 * (alpha_pos + alpha_neg)
        Q1_mf[a1] += alpha1 * pe1

        # Learn transitions row for selected action toward observed state
        T[a1, :] *= (1.0 - alpha_t)
        T[a1, s] += alpha_t
        # Normalize row to ensure valid probabilities
        T[a1, :] /= np.sum(T[a1, :])

    return -loglik


def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """Uncertainty-guided arbitration with choice kernels, learned transitions, and forgetting.
    Returns the negative log-likelihood of the observed choices.

    Cognitive assumptions:
    - Stage-2 values are learned model-free from rewards.
    - The agent learns the transition matrix from experience.
    - Arbitration between model-based and model-free control at stage 1 depends on online uncertainty:
      higher transition/reward uncertainty increases reliance on model-based planning.
    - Simple choice kernels capture recency-driven perseveration independent of value.
    - Value forgetting decays learned values over time.

    Parameters (all in [0,1], except betas in [0,10]):
    - alpha_q:  [0,1] learning rate for Q-values at both stages.
    - alpha_t:  [0,1] learning rate for the transition matrix.
    - decay:    [0,1] forgetting/decay factor applied to Q-values each trial.
    - kappa_u:  [0,1] relative weight on transition vs. reward uncertainty in arbitration.
    - alpha_k:  [0,1] learning rate for choice kernels at both stages.
    - kappa_k:  [0,1] strength of choice-kernel bias in logits (scaled internally).
    - w0:       [0,1] baseline arbitration inclination (mapped to [-5,5] inside).
    - beta1:    [0,10] inverse temperature for stage-1 softmax.
    - beta2:    [0,10] inverse temperature for stage-2 softmax.

    Inputs:
    - action_1: array-like of length n_trials, 0/1 indicating spaceship A/U chosen.
    - state:    array-like of length n_trials, 0/1 indicating planet X/Y reached.
    - action_2: array-like of length n_trials, 0/1 indicating alien chosen on that planet.
    - reward:   array-like of length n_trials, 0/1 coin outcome.
    - model_parameters: iterable with parameters in the order above.
    """
    alpha_q, alpha_t, decay, kappa_u, alpha_k, kappa_k, w0, beta1, beta2 = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Initialize structures
    T = np.ones((2, 2)) * 0.5
    Q1_mf = np.zeros(2)
    Q2 = np.zeros((2, 2))

    # Choice kernels (recency bias) for stage 1 and per-state stage 2
    K1 = np.zeros(2)
    K2 = np.zeros((2, 2))  # separate kernel per state

    loglik = 0.0

    # Helper for entropy of a 2-probability row
    def row_entropy(p_row):
        pr = np.clip(p_row, eps, 1 - eps)
        return -(pr[0] * np.log(pr[0]) + pr[1] * np.log(pr[1]))

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # Model-based value from current Q2 and T
        max_Q2 = np.max(Q2, axis=1)   # (2,)
        Q1_mb = T @ max_Q2            # (2,)

        # Compute uncertainty signals for arbitration on each action
        # Transition uncertainty: entropy of each row T[a]
        H = np.array([row_entropy(T[0]), row_entropy(T[1])])  # (2,)

        # Reward/value uncertainty per state: variance across aliens' Q within state
        var_Q2 = np.var(Q2, axis=1)  # (2,)

        # For each action, expected reward uncertainty = T[a] • var_Q2
        U_reward = T @ var_Q2  # (2,)

        # Combine uncertainties with kappa_u mixing coefficient
        U = kappa_u * H + (1.0 - kappa_u) * U_reward  # (2,)

        # Map baseline arbitration w0 in [0,1] to [-5,5] and produce action-specific weights via sigmoid
        w0_lin = 10.0 * (w0 - 0.5)
        w_mb_vec = 1.0 / (1.0 + np.exp(-(w0_lin + U)))  # higher U -> more model-based

        # Combine model-based and model-free values action-wise
        Q1_comb = w_mb_vec * Q1_mb + (1.0 - w_mb_vec) * Q1_mf

        # Choice kernels added as biases (scaled by kappa_k)
        # Scale kappa_k to a comparable range with values by multiplying by 2.0
        bias1 = 2.0 * kappa_k * (K1 - np.mean(K1))
        logits1 = beta1 * Q1_comb + bias1
        logits1 -= np.max(logits1)
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        loglik += np.log(p1[a1] + eps)

        # Stage 2 choice with state-specific kernel
        bias2 = 2.0 * kappa_k * (K2[s] - np.mean(K2[s]))
        logits2 = beta2 * Q2[s] + bias2
        logits2 -= np.max(logits2)
        p2 = np.exp(logits2)
        p2 /= np.sum(p2)
        loglik += np.log(p2[a2] + eps)

        # Updates
        # Stage-2 TD update
        pe2 = r - Q2[s, a2]
        Q2[s, a2] += alpha_q * pe2

        # Stage-1 MF bootstraps from max Q2 at visited state (TD(0) back-up)
        target_s1 = Q2[s, a2]
        pe1 = target_s1 - Q1_mf[a1]
        Q1_mf[a1] += alpha_q * pe1

        # Learn transitions row for chosen action
        T[a1, :] *= (1.0 - alpha_t)
        T[a1, s] += alpha_t
        T[a1, :] /= np.sum(T[a1, :])

        # Choice kernel updates (recency)
        K1 *= (1.0 - alpha_k)
        K1[a1] += alpha_k
        K2[s] *= (1.0 - alpha_k)
        K2[s, a2] += alpha_k

        # Forgetting
        Q1_mf *= (1.0 - decay)
        Q2 *= (1.0 - decay)

    return -loglik


def cognitive_model3(action_1, state, action_2, reward, model_parameters):
    """Optimistic uncertainty bonuses, transition confidence, and motor biases with eligibility trace.
    Returns the negative log-likelihood of the observed choices.

    Cognitive assumptions:
    - Stage-2 values are learned model-free, with optimistic bonuses for uncertain (rarely sampled) aliens.
    - The agent learns the transition matrix from experience and is attracted to uncertain spaceships
      via a transition uncertainty bonus at stage 1.
    - Stage-1 planning is model-based over learned transitions and current stage-2 values.
    - Stage-1 also has a model-free component updated via an eligibility trace from reward.
    - Action-specific motor biases favor each spaceship independently.

    Parameters (all in [0,1], except betas in [0,10]):
    - alpha_q:   [0,1] learning rate for Q-values (both stages).
    - alpha_t:   [0,1] learning rate for transition rows.
    - lam:       [0,1] eligibility trace weighting for backpropagation to stage-1 MF.
    - bonus_r:   [0,1] scale of optimistic bonus for uncertain stage-2 actions (1/(1+N)).
    - bonus_t:   [0,1] scale of uncertainty bonus for stage-1 actions based on transition entropy.
    - bias_a:    [0,1] motor bias toward spaceship A (mapped to [-0.5, 0.5] in logits).
    - bias_u:    [0,1] motor bias toward spaceship U (mapped to [-0.5, 0.5] in logits).
    - beta1:     [0,10] inverse temperature at stage 1.
    - beta2:     [0,10] inverse temperature at stage 2.

    Inputs:
    - action_1: array-like of length n_trials, 0/1 indicating spaceship A/U chosen.
    - state:    array-like of length n_trials, 0/1 indicating planet X/Y reached.
    - action_2: array-like of length n_trials, 0/1 indicating alien chosen on that planet.
    - reward:   array-like of length n_trials, 0/1 coin outcome.
    - model_parameters: iterable with parameters in the order above.
    """
    alpha_q, alpha_t, lam, bonus_r, bonus_t, bias_a, bias_u, beta1, beta2 = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Learned structures
    T = np.ones((2, 2)) * 0.5
    Q1_mf = np.zeros(2)
    Q2 = np.zeros((2, 2))

    # Visit counts for uncertainty bonuses
    N2 = np.zeros((2, 2))  # counts for (state, alien) pairs
    Ntrans = np.zeros(2)   # counts of selecting each spaceship

    loglik = 0.0

    def entropy_row(p_row):
        pr = np.clip(p_row, eps, 1 - eps)
        return -(pr[0] * np.log(pr[0]) + pr[1] * np.log(pr[1]))

    # Map motor biases from [0,1] to [-0.5, 0.5]
    motor_bias = np.array([bias_a - 0.5, bias_u - 0.5])

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # Stage-2 optimistic bonus based on uncertainty (smaller counts => larger bonus)
        u2 = 1.0 / (1.0 + N2)  # elementwise
        Q2_bonus = Q2 + bonus_r * u2

        # Model-based value for stage 1 using optimistic Q2 (agent plans with optimism)
        max_Q2_bonus = np.max(Q2_bonus, axis=1)
        Q1_mb = T @ max_Q2_bonus

        # Transition uncertainty bonus at stage 1 from entropy
        UT = np.array([entropy_row(T[0]), entropy_row(T[1])])
        bonus_stage1 = bonus_t * UT

        # Combine MB value, MF value, and bonus; equal weighting of MB and MF (implicit arbitration)
        Q1 = 0.5 * Q1_mb + 0.5 * Q1_mf + bonus_stage1

        # Stage-1 choice
        logits1 = beta1 * Q1 + motor_bias
        logits1 -= np.max(logits1)
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        loglik += np.log(p1[a1] + eps)

        # Stage-2 choice with optimistic bonus in logits
        logits2 = beta2 * Q2_bonus[s]
        logits2 -= np.max(logits2)
        p2 = np.exp(logits2)
        p2 /= np.sum(p2)
        loglik += np.log(p2[a2] + eps)

        # Updates
        # Count updates
        N2[s, a2] += 1.0
        Ntrans[a1] += 1.0

        # Stage-2 TD update (no optimism in learning, only in choice)
        pe2 = r - Q2[s, a2]
        Q2[s, a2] += alpha_q * pe2

        # Stage-1 MF eligibility-trace update
        target_s1 = (1.0 - lam) * Q2[s, a2] + lam * r
        pe1 = target_s1 - Q1_mf[a1]
        Q1_mf[a1] += alpha_q * pe1

        # Learn transitions for chosen action
        T[a1, :] *= (1.0 - alpha_t)
        T[a1, s] += alpha_t
        T[a1, :] /= np.sum(T[a1, :])

    return -loglik