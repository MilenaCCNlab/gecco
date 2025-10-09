def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """Hybrid MB/MF with eligibility trace, learned transitions, perseveration, decay, and lapses.
    Returns negative log-likelihood of observed choices.

    Parameters
    ----------
    action_1 : array-like of int (0 or 1)
        First-stage choices: 0 = spaceship A, 1 = spaceship U.
    state : array-like of int (0 or 1)
        Second-stage state reached: 0 = planet X, 1 = planet Y.
    action_2 : array-like of int (0 or 1)
        Second-stage choices on the planet: 0 or 1 (e.g., alien indices).
    reward : array-like of float
        Obtained reward on each trial (e.g., 0/1 coins).
    model_parameters : array-like
        Model parameters (all used):
        - alpha1 in [0,1]: learning rate for first-stage model-free updates via eligibility trace.
        - alpha2 in [0,1]: learning rate for second-stage Q updates.
        - lam in [0,1]: eligibility trace strength from stage 2 to stage 1 MF.
        - w_mb in [0,1]: weight of model-based relative to model-free at stage 1.
        - beta1 in [0,10]: softmax inverse temperature at stage 1.
        - beta2 in [0,10]: softmax inverse temperature at stage 2.
        - kappa1 in [0,1]: perseveration strength at stage 1 (choice kernel weight).
        - kappa2 in [0,1]: perseveration strength at stage 2 (choice kernel weight).
        - alpha_T in [0,1]: learning rate for transition probabilities.
        - lapse in [0,1]: lapse rate (mixture with uniform choice).
        - decay in [0,1]: forgetting/decay toward neutral value 0.5 for Q-values each trial.

    Notes
    -----
    - Transition structure is learned online (T) from observed action-state pairs.
    - Stage 1 values combine MB (via T @ max(Q2)) and MF (Q1) using w_mb.
    - Perseveration implemented via decaying choice kernels added to action preferences.
    - Lapse mixes softmax probabilities with uniform random responding.
    """
    alpha1, alpha2, lam, w_mb, beta1, beta2, kappa1, kappa2, alpha_T, lapse, decay = model_parameters
    n_trials = len(action_1)

    # Initialize learned transition matrix T[a, s2]
    T = np.array([[0.7, 0.3],
                  [0.3, 0.7]], dtype=float)

    # Q-values
    Q1_mf = np.zeros(2) + 0.5  # neutral start
    Q2 = np.zeros((2, 2)) + 0.5

    # Choice kernels for perseveration (decay each trial)
    K1 = np.zeros(2)
    K2 = np.zeros((2, 2))

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    eps = 1e-12

    for t in range(n_trials):
        s2 = state[t]

        # Model-based evaluation for stage 1: expected max second-stage value under learned T
        max_Q2 = np.max(Q2, axis=1)  # shape (2,)
        Q1_mb = T @ max_Q2  # shape (2,)

        # Hybrid valuation with perseveration kernels
        pref1 = w_mb * Q1_mb + (1.0 - w_mb) * Q1_mf + kappa1 * K1

        # Stage 1 choice probabilities (softmax + lapse)
        exp1 = np.exp(beta1 * (pref1 - np.max(pref1)))
        probs1 = exp1 / (np.sum(exp1) + eps)
        probs1 = (1.0 - lapse) * probs1 + lapse * 0.5

        a1 = action_1[t]
        p_choice_1[t] = probs1[a1]

        # Stage 2 preferences with perseveration
        pref2 = Q2[s2].copy() + kappa2 * K2[s2]
        exp2 = np.exp(beta2 * (pref2 - np.max(pref2)))
        probs2 = exp2 / (np.sum(exp2) + eps)
        probs2 = (1.0 - lapse) * probs2 + lapse * 0.5

        a2 = action_2[t]
        p_choice_2[t] = probs2[a2]

        r = reward[t]

        # Learning: Second stage Q update (prediction error at stage 2)
        pe2 = r - Q2[s2, a2]
        Q2[s2, a2] += alpha2 * pe2

        # Eligibility trace update for first-stage MF
        # Target is the immediate updated second-stage value (bootstrapped), propagated via lambda
        target1 = Q2[s2, a2]
        pe1 = target1 - Q1_mf[a1]
        Q1_mf[a1] += alpha1 * lam * pe1

        # Decay unchosen Q-values slightly toward 0.5 (forgetting)
        Q2 = (1.0 - decay) * Q2 + decay * 0.5
        Q1_mf = (1.0 - decay) * Q1_mf + decay * 0.5

        # Update transition model for chosen action toward observed state
        oh = np.array([1.0 if s2 == 0 else 0.0, 1.0 if s2 == 1 else 0.0])
        T[a1] = (1.0 - alpha_T) * T[a1] + alpha_T * oh
        # Keep rows normalized (small numeric safeguard)
        T[a1] = T[a1] / (np.sum(T[a1]) + eps)

        # Update perseveration kernels (recency)
        K1 = 0.8 * K1  # fixed decay of kernel memory (structure; value modulated by kappa1)
        K1[a1] += 1.0

        K2[s2] = 0.8 * K2[s2]
        K2[s2, a2] += 1.0

    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll


def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """Successor-representation flavored model with asymmetric learning, reward sensitivity,
    learned transitions, and a decaying choice kernel with bias.

    Returns negative log-likelihood of observed choices.

    Parameters
    ----------
    action_1 : array-like of int (0 or 1)
        First-stage choices: 0 = A, 1 = U.
    state : array-like of int (0 or 1)
        Second-stage state: 0 = X, 1 = Y.
    action_2 : array-like of int (0 or 1)
        Second-stage choices: 0 or 1.
    reward : array-like of float
        Reward outcome per trial.
    model_parameters : array-like
        - alpha_pos in [0,1]: learning rate when reward prediction error is positive.
        - alpha_neg in [0,1]: learning rate when reward prediction error is negative.
        - beta in [0,10]: softmax inverse temperature (both stages).
        - phi in [0,1]: decay rate for choice kernels (recency effect).
        - pi in [0,1]: weight of choice kernels on preferences (both stages).
        - biasA in [0,1]: baseline bias toward spaceship A at stage 1 (added as +bias to A, -bias to U).
        - rho in [0,1]: reward sensitivity (scales rewards before learning).
        - alpha_T in [0,1]: transition learning rate (for T).
        - sr_lambda in [0,1]: discount/horizon parameter shaping SR weighting at stage 1.
        - lapse in [0,1]: lapse rate (mixture with uniform choice).

    Notes
    -----
    - We construct an SR-like evaluation at stage 1 using the learned one-step transition model T.
      For two-step tasks, SR reduces to a weighted expectation over next states; sr_lambda
      scales the contribution of the max second-stage values (approaches MB when near 1).
    - Asymmetric learning at stage 2 uses alpha_pos for positive PE and alpha_neg for negative PE.
    - Choice kernels accumulate recent choices and influence both stages.
    - A static bias toward action A is included at stage 1.
    """
    alpha_pos, alpha_neg, beta, phi, pi, biasA, rho, alpha_T, sr_lambda, lapse = model_parameters
    n_trials = len(action_1)

    # Transition matrix learned
    T = np.array([[0.7, 0.3],
                  [0.3, 0.7]], dtype=float)

    # Q-values at stage 2 (state x action)
    Q2 = np.zeros((2, 2)) + 0.5

    # Choice kernels
    K1 = np.zeros(2)
    K2 = np.zeros((2, 2))

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    eps = 1e-12

    for t in range(n_trials):
        s2 = state[t]

        # SR-like evaluation: expected value over next states with a controllability scaling
        max_Q2 = np.max(Q2, axis=1)  # value of best alien on each planet
        # SR weighting: interpolate between neutral expectation (0.5) and full MB using sr_lambda
        sr_value = (1.0 - sr_lambda) * 0.5 + sr_lambda * max_Q2  # shape (2,)
        Q1_sr = T @ sr_value  # expected value of each spaceship

        # Add choice kernel and bias (biasA as +b for A, -b for U)
        bias_vec = np.array([biasA, -biasA])
        pref1 = Q1_sr + pi * K1 + bias_vec

        # Stage 1 choice distribution
        exp1 = np.exp(beta * (pref1 - np.max(pref1)))
        probs1 = exp1 / (np.sum(exp1) + eps)
        probs1 = (1.0 - lapse) * probs1 + lapse * 0.5

        a1 = action_1[t]
        p_choice_1[t] = probs1[a1]

        # Stage 2 preferences (kernel-influenced)
        pref2 = Q2[s2] + pi * K2[s2]
        exp2 = np.exp(beta * (pref2 - np.max(pref2)))
        probs2 = exp2 / (np.sum(exp2) + eps)
        probs2 = (1.0 - lapse) * probs2 + lapse * 0.5

        a2 = action_2[t]
        p_choice_2[t] = probs2[a2]

        # Learning at stage 2 with asymmetric rates and reward sensitivity
        r = rho * reward[t]
        pe2 = r - Q2[s2, a2]
        alpha = alpha_pos if pe2 >= 0.0 else alpha_neg
        Q2[s2, a2] += alpha * pe2

        # Update transition model for chosen action
        oh = np.array([1.0 if s2 == 0 else 0.0, 1.0 if s2 == 1 else 0.0])
        T[a1] = (1.0 - alpha_T) * T[a1] + alpha_T * oh
        T[a1] = T[a1] / (np.sum(T[a1]) + eps)

        # Update choice kernels (recency with decay phi)
        K1 = (1.0 - phi) * K1
        K1[a1] += 1.0

        K2[s2] = (1.0 - phi) * K2[s2]
        K2[s2, a2] += 1.0

    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll


def cognitive_model3(action_1, state, action_2, reward, model_parameters):
    """Dynamic arbitration between cached MB and MF with transition-sensitive eligibility,
    learnable transitions, decay, and lapses.

    Returns negative log-likelihood of observed choices.

    Parameters
    ----------
    action_1 : array-like of int (0 or 1)
        First-stage choices: 0 = A, 1 = U.
    state : array-like of int (0 or 1)
        Second-stage state: 0 = X, 1 = Y.
    action_2 : array-like of int (0 or 1)
        Second-stage choice on the planet.
    reward : array-like of float
        Reward per trial.
    model_parameters : array-like
        - alpha_mf in [0,1]: MF learning rate for stage 1 via eligibility trace from stage 2.
        - alpha_mb in [0,1]: learning rate for updating a cached MB value approximation at stage 1.
        - beta1 in [0,10]: inverse temperature for stage 1 choices.
        - beta2 in [0,10]: inverse temperature for stage 2 choices.
        - eta in [0,1]: adaptation rate for arbitration weight w_t.
        - w0 in [0,1]: initial arbitration weight favoring MB at t=0.
        - kappa_tr in [0,1]: first-stage action stickiness (tendency to repeat same spaceship).
        - zeta in [0,1]: down-weighting of MF eligibility after rare transitions (0=no effect, 1=fully suppress).
        - lapse in [0,1]: lapse rate (mixture with uniform).
        - alpha_T in [0,1]: transition learning rate.
        - gamma_forgot in [0,1]: forgetting rate toward 0.5 for unchosen Q-values.

    Notes
    -----
    - Arbitration weight w_t is updated each trial toward a sigmoid of the instantaneous advantage
      of the MB over MF valuations (based on their max action preferences).
    - MF eligibility is reduced after rare transitions relative to the most-likely transition
      predicted by the learned T for the chosen action (controlled by zeta).
    - A cached MB estimate at stage 1 is updated with alpha_mb toward the instantaneous MB value.
    """
    (alpha_mf, alpha_mb, beta1, beta2, eta, w0,
     kappa_tr, zeta, lapse, alpha_T, gamma_forgot) = model_parameters

    n_trials = len(action_1)

    # Learned transitions
    T = np.array([[0.7, 0.3],
                  [0.3, 0.7]], dtype=float)

    # Q values
    Q2 = np.zeros((2, 2)) + 0.5
    Q1_mf = np.zeros(2) + 0.5
    Q1_mb_cached = np.zeros(2) + 0.5

    # Stickiness kernel for stage 1
    last_a1 = None

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    # Arbitration weight
    w = w0

    eps = 1e-12

    for t in range(n_trials):
        s2 = state[t]

        # Instantaneous MB evaluation from current Q2 and T
        max_Q2 = np.max(Q2, axis=1)
        Q1_mb_inst = T @ max_Q2

        # Update cached MB estimate
        Q1_mb_cached = (1.0 - alpha_mb) * Q1_mb_cached + alpha_mb * Q1_mb_inst

        # Compute MF vs MB advantage for arbitration update (based on max preference)
        mf_max = np.max(Q1_mf)
        mb_max = np.max(Q1_mb_cached)
        adv = mb_max - mf_max
        # Squashing to [0,1] via logistic with fixed slope for stability
        sig_adv = 1.0 / (1.0 + np.exp(-10.0 * adv))
        # Update arbitration weight
        w = (1.0 - eta) * w + eta * sig_adv

        # Combine MB and MF at stage 1, plus stickiness
        stickiness = np.zeros(2)
        if last_a1 is not None:
            stickiness[last_a1] = kappa_tr

        pref1 = w * Q1_mb_cached + (1.0 - w) * Q1_mf + stickiness

        exp1 = np.exp(beta1 * (pref1 - np.max(pref1)))
        probs1 = exp1 / (np.sum(exp1) + eps)
        probs1 = (1.0 - lapse) * probs1 + lapse * 0.5

        a1 = action_1[t]
        p_choice_1[t] = probs1[a1]

        # Stage 2 softmax
        exp2 = np.exp(beta2 * (Q2[s2] - np.max(Q2[s2])))
        probs2 = exp2 / (np.sum(exp2) + eps)
        probs2 = (1.0 - lapse) * probs2 + lapse * 0.5

        a2 = action_2[t]
        p_choice_2[t] = probs2[a2]

        r = reward[t]

        # Stage 2 learning
        pe2 = r - Q2[s2, a2]
        Q2[s2, a2] += alpha_mf * pe2  # reuse alpha_mf here to keep param count minimal and used

        # Transition rarity for MF eligibility
        # Identify whether observed transition s2 was the most probable under T[a1]
        most_prob_state = int(np.argmax(T[a1]))
        is_rare = 1.0 if s2 != most_prob_state else 0.0
        eligibility_scale = 1.0 - zeta * is_rare

        # MF eligibility update of stage 1
        target1 = Q2[s2, a2]
        pe1 = target1 - Q1_mf[a1]
        Q1_mf[a1] += alpha_mf * eligibility_scale * pe1

        # Forgetting toward 0.5
        Q2 = (1.0 - gamma_forgot) * Q2 + gamma_forgot * 0.5
        Q1_mf = (1.0 - gamma_forgot) * Q1_mf + gamma_forgot * 0.5
        Q1_mb_cached = (1.0 - gamma_forgot) * Q1_mb_cached + gamma_forgot * 0.5

        # Learn transitions for chosen action
        oh = np.array([1.0 if s2 == 0 else 0.0, 1.0 if s2 == 1 else 0.0])
        T[a1] = (1.0 - alpha_T) * T[a1] + alpha_T * oh
        T[a1] = T[a1] / (np.sum(T[a1]) + eps)

        # Update stickiness memory
        last_a1 = a1

    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll