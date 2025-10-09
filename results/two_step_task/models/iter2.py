def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """Successor-Representation (SR) stage-1 valuation with uncertainty-tempered choice,
    MF eligibility to stage-1, transition-free SR learning, choice kernels, and decay.
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
        - alpha_sr in [0,1]: learning rate for SR rows M[a,:] (from action to second-stage state occupancies).
        - alpha_R in [0,1]: learning rate for second-stage Q-values.
        - eta in [0,1]: eligibility trace strength to update stage-1 model-free value from stage-2 outcome.
        - w_sr in [0,1]: weight of SR-based value vs MF value at stage 1.
        - beta_base in [0,10]: base inverse temperature at stage 1.
        - u_temp in [0,1]: uncertainty gain; higher increases temperature softening with higher transition entropy.
        - gamma in [0,1]: SR discount factor (here controls persistence; terminal so mainly scales SR magnitude).
        - kappa1 in [0,1]: perseveration (choice kernel) strength at stage 1.
        - lapse in [0,1]: lapse rate applied to both stages (mixture with uniform choice).
        - decay in [0,1]: decay of Q-values toward 0.5 each trial.

    Notes
    -----
    - SR M is learned directly from observed action->state outcomes without an explicit transition model.
    - Stage-1 value for action a is V_SR[a] = sum_s M[a,s] * max_a2 Q2[s,a2].
    - Uncertainty-tempered choice: effective beta1 is reduced as the entropy of M[a,:] increases.
    """
    alpha_sr, alpha_R, eta, w_sr, beta_base, u_temp, gamma, kappa1, lapse, decay = model_parameters
    n_trials = len(action_1)

    # Successor representation over (action -> second-stage states)
    # Initialize as nearly uniform to avoid zero-entropy edge cases
    M = np.ones((2, 2)) * 0.5  # rows: actions (A,U); cols: states (X,Y)

    # Model-free stage-1 cached values (via eligibility from stage-2)
    Q1_mf = np.zeros(2) + 0.5

    # Second-stage Q-values
    Q2 = np.zeros((2, 2)) + 0.5

    # Choice kernel for perseveration at stage 1
    K1 = np.zeros(2)

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    eps = 1e-12

    for t in range(n_trials):
        s2 = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r = reward[t]

        # Stage-1 SR-based values: use current Q2 to compute value features
        max_Q2 = np.max(Q2, axis=1)  # value of each second-stage state
        V_sr = M @ max_Q2  # shape (2,): value per action via SR

        # Uncertainty-tempered softmax: compute entropy of M rows, normalize to [0,1]
        # Entropy of a Bernoulli p is H(p) = -p*log p - (1-p) log(1-p); normalize by log(2).
        p_rows = np.clip(M, eps, 1 - eps)
        p = p_rows[:, 0]  # prob of state 0; two-state entropy is symmetric
        H = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2.0)  # in [0,1]
        # Effective beta per action: lower when uncertainty high; combine to scalar by averaging
        beta1_eff = beta_base * (1.0 - u_temp * np.mean(H))

        # Combine SR and MF values at stage 1, add choice kernel bias
        pref1 = w_sr * V_sr + (1.0 - w_sr) * Q1_mf + kappa1 * K1

        # Stage-1 choice probabilities with lapse
        pref1_centered = pref1 - np.max(pref1)
        exp1 = np.exp(beta1_eff * pref1_centered)
        probs1 = exp1 / (np.sum(exp1) + eps)
        probs1 = (1.0 - lapse) * probs1 + lapse * 0.5
        p_choice_1[t] = probs1[a1]

        # Stage-2 choice probabilities with lapse
        pref2 = Q2[s2].copy()
        pref2_centered = pref2 - np.max(pref2)
        exp2 = np.exp(beta_base * pref2_centered)  # reuse beta_base for simplicity at stage 2
        probs2 = exp2 / (np.sum(exp2) + eps)
        probs2 = (1.0 - lapse) * probs2 + lapse * 0.5
        p_choice_2[t] = probs2[a2]

        # Learning: Stage 2 Q-values
        pe2 = r - Q2[s2, a2]
        Q2[s2, a2] += alpha_R * pe2

        # MF eligibility to stage 1 (bootstrapping from obtained stage-2 value)
        target1 = Q2[s2, a2]
        pe1 = target1 - Q1_mf[a1]
        Q1_mf[a1] += eta * pe1

        # SR update: move row a1 toward observed one-hot of reached state with a terminal continuation via gamma
        oh_s = np.array([1.0 if s2 == 0 else 0.0, 1.0 if s2 == 1 else 0.0])
        # Since the episode ends after stage-2, next SR expectation is 0; include gamma scaling for completeness
        td_sr = oh_s + gamma * np.zeros(2) - M[a1]
        M[a1] += alpha_sr * td_sr
        # Keep rows normalized to sum near 1 for interpretability (not required but stabilizes entropy)
        M[a1] = M[a1] / (np.sum(M[a1]) + eps)

        # Decay Q-values toward neutral 0.5
        Q1_mf = (1.0 - decay) * Q1_mf + decay * 0.5
        Q2 = (1.0 - decay) * Q2 + decay * 0.5

        # Update choice kernel (simple recency; fixed decay factor embedded in kappa scaling)
        K1 *= 0.8
        K1[a1] += 1.0

    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll


def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """Volatility-gated hybrid MB/MF with asymmetric reward learning, leaky transition learning,
    dynamic MB weight from transition volatility, perseveration, action bias, and lapses.
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
        - alpha_pos in [0,1]: positive RPE learning rate at stage 2.
        - alpha_neg in [0,1]: negative RPE learning rate at stage 2.
        - beta1 in [0,10]: stage-1 inverse temperature.
        - beta2 in [0,10]: stage-2 inverse temperature.
        - w_mb0 in [0,1]: baseline weight of model-based value at stage 1.
        - vol_sens in [0,1]: sensitivity of MB weight to transition volatility.
        - kappa1 in [0,1]: perseveration (choice kernel) strength at stage 1.
        - kappa2 in [0,1]: perseveration (choice kernel) strength at stage 2.
        - biasA in [0,1]: static bias toward spaceship A (mapped to [-0.5, 0.5]).
        - alpha_T in [0,1]: leaky learning rate for transition probabilities per action.
        - nu in [0,1]: volatility learning rate (EMA of absolute transition changes).
        - lapse in [0,1]: lapse rate applied to both stages (mixture with uniform choice).

    Notes
    -----
    - Transition matrix T is learned per action with exponential averaging and renormalization.
    - Volatility per action is an EMA of absolute updates to T[a], increasing MB weighting when high.
    - Stage-1 value blends MB (T @ max(Q2)) and MF (cached Q1 from eligibility) using dynamic weight.
    """
    alpha_pos, alpha_neg, beta1, beta2, w_mb0, vol_sens, kappa1, kappa2, biasA, alpha_T, nu, lapse = model_parameters
    n_trials = len(action_1)

    # Initialize transition beliefs moderately structured
    T = np.array([[0.7, 0.3],
                  [0.3, 0.7]], dtype=float)

    # Track volatility per action (EMA of |delta T[a]|)
    vol = np.zeros(2)

    # Stage-1 MF cache and stage-2 Q-values
    Q1_mf = np.zeros(2) + 0.5
    Q2 = np.zeros((2, 2)) + 0.5

    # Choice kernels
    K1 = np.zeros(2)
    K2 = np.zeros((2, 2))

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    eps = 1e-12
    # Map biasA in [0,1] to additive preference in [-0.5, 0.5]
    bias_term = biasA - 0.5

    for t in range(n_trials):
        s2 = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r = reward[t]

        # Model-based stage-1 values from learned T
        max_Q2 = np.max(Q2, axis=1)  # shape (2,)
        Q1_mb = T @ max_Q2  # shape (2,)

        # Dynamic MB weight based on average volatility (scaled and clipped)
        vol_avg = 0.5 * (vol[0] + vol[1])
        w_mb = np.clip(w_mb0 + vol_sens * (vol_avg - 0.25), 0.0, 1.0)

        # Stage-1 preferences: weighted MB/MF + perseveration + static bias toward A
        pref1 = w_mb * Q1_mb + (1.0 - w_mb) * Q1_mf + kappa1 * K1 + np.array([bias_term, -bias_term])

        # Stage-1 softmax with lapse
        pref1_centered = pref1 - np.max(pref1)
        exp1 = np.exp(beta1 * pref1_centered)
        probs1 = exp1 / (np.sum(exp1) + eps)
        probs1 = (1.0 - lapse) * probs1 + lapse * 0.5
        p_choice_1[t] = probs1[a1]

        # Stage-2 preferences and softmax with perseveration and lapse
        pref2 = Q2[s2].copy() + kappa2 * K2[s2]
        pref2_centered = pref2 - np.max(pref2)
        exp2 = np.exp(beta2 * pref2_centered)
        probs2 = exp2 / (np.sum(exp2) + eps)
        probs2 = (1.0 - lapse) * probs2 + lapse * 0.5
        p_choice_2[t] = probs2[a2]

        # Learning: stage-2 asymmetric RPE
        pe2 = r - Q2[s2, a2]
        alpha2 = alpha_pos if pe2 >= 0 else alpha_neg
        Q2[s2, a2] += alpha2 * pe2

        # Eligibility to stage-1 MF cache
        target1 = Q2[s2, a2]
        pe1 = target1 - Q1_mf[a1]
        # Use symmetric rate equal to mean of alpha_pos/alpha_neg for MF cache to avoid extra params
        alpha1_mf = 0.5 * (alpha_pos + alpha_neg)
        Q1_mf[a1] += alpha1_mf * pe1

        # Update transition beliefs for the taken action
        oh = np.array([1.0 if s2 == 0 else 0.0, 1.0 if s2 == 1 else 0.0])
        T_old = T[a1].copy()
        T[a1] = (1.0 - alpha_T) * T[a1] + alpha_T * oh
        # Renormalize
        T[a1] = T[a1] / (np.sum(T[a1]) + eps)
        # Update volatility as EMA of absolute change magnitude
        delta_T = np.abs(T[a1] - T_old).sum()  # L1 change
        vol[a1] = (1.0 - nu) * vol[a1] + nu * delta_T

        # Update choice kernels (fixed recency decay)
        K1 *= 0.8
        K1[a1] += 1.0
        K2[s2] *= 0.8
        K2[s2, a2] += 1.0

    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll


def cognitive_model3(action_1, state, action_2, reward, model_parameters):
    """Arbitrated mixture of Model-Based (MB) planning and Win-Stay/Lose-Shift (WSLS),
    with learned transitions, count-based novelty bonus, strategy credit assignment,
    stickiness, and lapses. Returns negative log-likelihood.

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
        - alpha2 in [0,1]: learning rate for second-stage Q-values.
        - beta1 in [0,10]: inverse temperature for stage-1 MB softmax.
        - beta2 in [0,10]: inverse temperature for stage-2 softmax.
        - alpha_T in [0,1]: learning rate for transition probabilities.
        - alpha_arbit in [0,1]: learning rate for strategy value (arbitrator) updates.
        - beta_arbit in [0,10]: inverse temperature mapping strategy values to mixing weight.
        - stick1 in [0,1]: stage-1 stickiness strength (perseveration).
        - novelty_bonus in [0,1]: scaling of count-based novelty bonus at stage 1.
        - wsls_weight in [0,1]: baseline weight for WSLS policy within the arbitration.
        - lapse in [0,1]: lapse rate applied to both stages (mixture with uniform choice).

    Notes
    -----
    - MB policy uses learned T and Q2 (max over stage-2 actions) plus an exploration bonus equal
      to expected novelty of next states (inverse visitation count).
    - WSLS policy recommends repeating previous a1 if last reward was 1, switching if 0.
    - Arbitration weight is a softmax of strategy values SV_mb and SV_wsls; these are updated
      by crediting a strategy proportional to whether it supported the chosen action on that trial.
    """
    alpha2, beta1, beta2, alpha_T, alpha_arbit, beta_arbit, stick1, novelty_bonus, wsls_weight, lapse = model_parameters
    n_trials = len(action_1)

    # Transition model
    T = np.array([[0.7, 0.3],
                  [0.3, 0.7]], dtype=float)

    # Second-stage Q-values
    Q2 = np.zeros((2, 2)) + 0.5

    # Visit counts per second-stage state for novelty
    visits = np.ones(2)  # start at 1 to avoid div by zero

    # Strategy values for arbitration
    SV_mb = 0.0
    SV_wsls = 0.0

    # Stage-1 stickiness kernel
    K1 = np.zeros(2)

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    eps = 1e-12

    # For WSLS, track previous a1 and reward
    prev_a1 = None
    prev_r = None

    for t in range(n_trials):
        s2 = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r = reward[t]

        # MB policy: compute stage-1 action values from T @ (max Q2 + novelty bonus)
        max_Q2 = np.max(Q2, axis=1)
        novelty_value = novelty_bonus * (1.0 / (1.0 + visits))  # per state
        state_value = max_Q2 + novelty_value
        Q1_mb = T @ state_value

        # MB softmax policy
        mb_pref = Q1_mb + stick1 * K1
        mb_pref_centered = mb_pref - np.max(mb_pref)
        mb_exp = np.exp(beta1 * mb_pref_centered)
        mb_probs = mb_exp / (np.sum(mb_exp) + eps)

        # WSLS policy as a probability distribution
        if prev_a1 is None:
            wsls_probs = np.array([0.5, 0.5])
        else:
            if prev_r is None:
                wsls_probs = np.array([0.5, 0.5])
            else:
                if prev_r > 0.0:
                    # stay
                    wsls_probs = np.array([1.0 if prev_a1 == 0 else 0.0,
                                           1.0 if prev_a1 == 1 else 0.0])
                else:
                    # shift
                    wsls_probs = np.array([1.0 if prev_a1 == 1 else 0.0,
                                           1.0 if prev_a1 == 0 else 0.0])

        # Arbitration: compute mixing weight via softmax of strategy values with baseline wsls_weight
        # Convert wsls_weight in [0,1] to prior logit offset
        prior_logit = np.log((1e-6 + wsls_weight) / (1e-6 + (1.0 - wsls_weight)))
        logits = beta_arbit * np.array([SV_mb, SV_wsls]) + np.array([0.0, prior_logit])
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        strat_probs = exp_logits / (np.sum(exp_logits) + eps)
        w_mb = strat_probs[0]  # probability of using MB policy
        w_wsls = strat_probs[1]

        # Mixture policy for stage-1
        probs1 = w_mb * mb_probs + w_wsls * wsls_probs
        probs1 = (1.0 - lapse) * probs1 + lapse * 0.5
        p_choice_1[t] = probs1[a1]

        # Stage-2 policy
        pref2 = Q2[s2].copy()
        pref2_centered = pref2 - np.max(pref2)
        exp2 = np.exp(beta2 * pref2_centered)
        probs2 = exp2 / (np.sum(exp2) + eps)
        probs2 = (1.0 - lapse) * probs2 + lapse * 0.5
        p_choice_2[t] = probs2[a2]

        # Learning: stage-2 Q-values
        pe2 = r - Q2[s2, a2]
        Q2[s2, a2] += alpha2 * pe2

        # Update transitions based on observed action->state
        oh = np.array([1.0 if s2 == 0 else 0.0, 1.0 if s2 == 1 else 0.0])
        T[a1] = (1.0 - alpha_T) * T[a1] + alpha_T * oh
        T[a1] = T[a1] / (np.sum(T[a1]) + eps)

        # Update visits for novelty
        visits[s2] += 1.0

        # Strategy credit assignment:
        # - MB support: how much more probability MB assigned to chosen action vs alternative
        mb_support = mb_probs[a1] - mb_probs[1 - a1]
        mb_support = np.clip(mb_support, 0.0, 1.0)
        # - WSLS support: 1 if WSLS recommended chosen action, else 0 (or 0.5 if undefined)
        if prev_a1 is None or prev_r is None:
            wsls_support = 0.5
        else:
            if prev_r > 0.0:
                wsls_choice = prev_a1
            else:
                wsls_choice = 1 - prev_a1
            wsls_support = 1.0 if wsls_choice == a1 else 0.0

        # Update strategy values toward obtained reward weighted by support
        SV_mb += alpha_arbit * (mb_support * r - SV_mb)
        SV_wsls += alpha_arbit * (wsls_support * r - SV_wsls)

        # Update stickiness kernel
        K1 *= 0.8
        K1[a1] += 1.0

        # Cache for next trial
        prev_a1 = a1
        prev_r = r

    eps = 1e-12
    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll