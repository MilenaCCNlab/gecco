def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """Surprise-modulated hybrid planner with choice-kernels and learned transitions.
    Returns the negative log-likelihood of the observed choices.

    Cognitive assumptions:
    - Stage-2 values are learned model-free from reward.
    - The transition matrix (spaceship -> planet) is learned from experience.
    - Stage-1 uses a weighted combination of model-based (planning via learned transitions)
      and model-free values, where the model-based weight is boosted on surprising
      transitions (transition-surprise arbitration).
    - A small eligibility trace lets reward flow back to the stage-1 MF value.
    - A dynamic choice kernel (separate from values) captures short-term choice perseveration/alternation.
      It is updated with its own learning rate and weighted into the logits.
    
    Parameters (bounds):
    - alpha_r: [0,1] learning rate for stage-2 Q-values.
    - alpha_t: [0,1] learning rate for the transition matrix.
    - w0:      [0,1] baseline weight on model-based value at stage 1.
    - k_sur:   [0,1] strength of surprise-based increase of model-based weight (trial-by-trial).
    - lam:     [0,1] eligibility trace for backpropagating value/reward to stage 1 MF.
    - beta1:   [0,10] inverse temperature for stage-1 softmax.
    - beta2:   [0,10] inverse temperature for stage-2 softmax.
    - k_ck:    [0,1] learning rate for the (two-action) choice-kernel at both stages.
    - w_ck:    [0,1] weight of the choice-kernel contribution in the logits at both stages.

    Inputs:
    - action_1: array-like (n_trials,) of 0/1 spaceship choices.
    - state:    array-like (n_trials,) of 0/1 planet reached.
    - action_2: array-like (n_trials,) of 0/1 alien choices.
    - reward:   array-like (n_trials,) of 0/1 coin outcomes.
    - model_parameters: iterable with parameters in the order above.

    Notes:
    - Surprise is computed as 1 - predicted probability of the observed transition on that trial.
      The model-based weight is w_t = clip(w0 + k_sur * surprise, 0, 1).
    - Choice-kernel is a two-armed, recency-weighted bias that is updated toward the chosen action.
    """
    alpha_r, alpha_t, w0, k_sur, lam, beta1, beta2, k_ck, w_ck = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Learned transition matrix T[a1, s] = P(planet=s | spaceship=a1)
    T = np.ones((2, 2)) * 0.5

    # Model-free values
    Q1_mf = np.zeros(2)
    Q2_mf = np.zeros((2, 2))

    # Choice kernels (separate recency biases) for stage 1 and each planet at stage 2
    K1 = np.zeros(2)
    K2 = np.zeros((2, 2))

    loglik = 0.0

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # Model-based planning uses learned transitions to back up max over stage-2 values
        max_Q2 = np.max(Q2_mf, axis=1)   # shape (2,)
        Q1_mb = T @ max_Q2               # shape (2,)

        # Compute surprise for the realized transition
        pred_p = T[a1, s]
        surprise = 1.0 - pred_p
        w_t = w0 + k_sur * surprise
        if w_t < 0.0:
            w_t = 0.0
        elif w_t > 1.0:
            w_t = 1.0

        Q1_comb = w_t * Q1_mb + (1.0 - w_t) * Q1_mf

        # Stage-1 softmax with choice-kernel bias
        logits1 = beta1 * Q1_comb + w_ck * K1
        logits1 -= np.max(logits1)
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        loglik += np.log(p1[a1] + eps)

        # Stage-2 softmax with planet-specific choice-kernel bias
        logits2 = beta2 * Q2_mf[s] + w_ck * K2[s]
        logits2 -= np.max(logits2)
        p2 = np.exp(logits2)
        p2 /= np.sum(p2)
        loglik += np.log(p2[a2] + eps)

        # Learning updates

        # Stage-2 MF update
        pe2 = r - Q2_mf[s, a2]
        Q2_mf[s, a2] += alpha_r * pe2

        # Stage-1 MF update via eligibility trace mixture (value and immediate reward)
        target_s1 = (1.0 - lam) * Q2_mf[s, a2] + lam * r
        pe1 = target_s1 - Q1_mf[a1]
        Q1_mf[a1] += alpha_r * pe1

        # Transition learning (row-wise)
        T[a1, :] *= (1.0 - alpha_t)
        T[a1, s] += alpha_t
        # Normalize for numerical safety
        T[a1, :] /= (np.sum(T[a1, :]) + eps)

        # Update choice kernels (recency-weighted toward chosen action)
        # Stage-1
        for a in (0, 1):
            target = 1.0 if a == a1 else 0.0
            K1[a] += k_ck * (target - K1[a])
        # Stage-2 (planet-specific kernel)
        for a in (0, 1):
            target = 1.0 if a == a2 else 0.0
            K2[s, a] += k_ck * (target - K2[s, a])

    return -loglik


def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """Asymmetric learning, soft backup planning, WSLS heuristic blend, and selective decay.
    Returns the negative log-likelihood of observed choices.

    Cognitive assumptions:
    - Stage-2 MF values use separate learning rates for positive vs negative outcomes.
    - Stage-1 planning uses a soft backup over stage-2 action values (not a hard max),
      controlled by a softness parameter. Transitions are assumed known but parameterized
      by the agent's belief in a "common transition" probability.
    - A win-stay/lose-switch (WSLS) heuristic is blended into stage-1 choice probabilities.
    - Selective decay (forgetting) applies to unchosen actions' values at both stages.

    Parameters (bounds):
    - alpha_pos: [0,1] learning rate for positive PE (r - Q > 0) at stage 2.
    - alpha_neg: [0,1] learning rate for negative PE (r - Q < 0) at stage 2.
    - p_comm:    [0,1] belief about common transition probability (maps to 0.1..0.9).
    - soft_back: [0,1] softness of backup; 0=hard max, 1=uniform average over actions.
    - omega_wsls:[0,1] weight of WSLS heuristic in stage-1 logits.
    - phi_sel:   [0,1] selective decay rate applied to unchosen actions each trial.
    - beta1:     [0,10] inverse temperature for stage-1 softmax.
    - beta2:     [0,10] inverse temperature for stage-2 softmax.

    Inputs:
    - action_1: array-like (n_trials,) of 0/1 spaceship choices.
    - state:    array-like (n_trials,) of 0/1 planet reached.
    - action_2: array-like (n_trials,) of 0/1 alien choices.
    - reward:   array-like (n_trials,) of 0/1 coin outcomes.
    - model_parameters: iterable with parameters in the order above.

    Notes:
    - Transition structure is symmetric: for a1 in {0,1}, P(common state | a1) = pc, P(rare) = 1-pc
      with pc mapping from p_comm via pc = 0.1 + 0.8 * p_comm (range 0.1..0.9).
      Action 0's common state is 0; action 1's common state is 1.
    - Soft backup: V(s) = (1-soft_back)*max_a Q2[s,a] + soft_back*mean_a Q2[s,a].
    - WSLS heuristic: if previous trial won (r=1), bias toward repeating previous a1;
      if lost (r=0), bias toward switching. The bias is added to logits with weight omega_wsls.
    - Selective decay: all unchosen Q-values are shrunk by (1-phi_sel) each trial.
    """
    alpha_pos, alpha_neg, p_comm, soft_back, omega_wsls, phi_sel, beta1, beta2 = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Stage-2 MF values
    Q2 = np.zeros((2, 2))
    # Stage-1 cached MF values for completeness (receives selective decay only; not learned directly here)
    Q1_mf = np.zeros(2)

    # Transition belief (fixed structure parameterized by p_comm)
    pc = 0.1 + 0.8 * p_comm  # in [0.1, 0.9]
    T = np.array([[pc, 1.0 - pc],
                  [1.0 - pc, pc]])

    loglik = 0.0

    prev_a1 = None
    prev_r = None

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # Soft backup values for each potential spaceship action
        # For each candidate a1', expected value is sum_s T[a1', s] * V_soft(s)
        V_soft = (1.0 - soft_back) * np.max(Q2, axis=1) + soft_back * np.mean(Q2, axis=1)  # shape (2,)
        Q1_mb = T @ V_soft  # shape (2,)

        # WSLS heuristic bias vector
        wsls_bias = np.zeros(2)
        if prev_a1 is not None and prev_r is not None:
            if prev_r > 0.5:
                # win-stay: bias the previous action up, the other down
                wsls_bias[prev_a1] += 1.0
                wsls_bias[1 - prev_a1] -= 1.0
            else:
                # lose-switch: bias the other action up
                wsls_bias[prev_a1] -= 1.0
                wsls_bias[1 - prev_a1] += 1.0

        # Combine MB value, a (small) MF cache, and WSLS in logits
        logits1 = beta1 * (Q1_mb + 0.1 * Q1_mf) + omega_wsls * wsls_bias
        logits1 -= np.max(logits1)
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        loglik += np.log(p1[a1] + eps)

        # Stage-2 choice
        logits2 = beta2 * Q2[s]
        logits2 -= np.max(logits2)
        p2 = np.exp(logits2)
        p2 /= np.sum(p2)
        loglik += np.log(p2[a2] + eps)

        # Learning: asymmetric at stage 2
        pe2 = r - Q2[s, a2]
        alpha = alpha_pos if pe2 >= 0.0 else alpha_neg
        Q2[s, a2] += alpha * pe2

        # Selective decay: shrink unchosen actions' values
        # Stage-2: within visited state, decay the unchosen action
        Q2[s, 1 - a2] *= (1.0 - phi_sel)
        # Stage-2: in unvisited state, decay both actions (no new info)
        other_s = 1 - s
        Q2[other_s, :] *= (1.0 - phi_sel)

        # Stage-1 MF cache: decay unchosen and slightly reinforce chosen toward observed stage-2 value
        Q1_mf[1 - a1] *= (1.0 - phi_sel)
        target1 = V_soft[s]  # use current state's soft value as target
        Q1_mf[a1] += 0.5 * (alpha_pos + alpha_neg) * (target1 - Q1_mf[a1])

        prev_a1 = a1
        prev_r = r

    return -loglik


def cognitive_model3(action_1, state, action_2, reward, model_parameters):
    """MB/MF hybrid with PE-scaled temperature, confidence in known transitions, run-length suppression, and counterfactual learning.
    Returns the negative log-likelihood of observed choices.

    Cognitive assumptions:
    - Stage-2 values are learned model-free from reward, with counterfactual learning on the
      unchosen alien in the visited state.
    - Stage-1 combines model-based planning (assuming a partially trusted known transition matrix)
      and model-free values via an eligibility trace.
    - Decision noise (temperature) decreases after predictable outcomes and increases after surprising ones
      via unsigned prediction-error scaling.
    - Repeating the same stage-1 action for many trials accrues a penalty (run-length suppression),
      discouraging long perseveration streaks beyond simple stickiness.

    Parameters (bounds):
    - alpha_r:  [0,1] learning rate for stage-2 Q-values.
    - w_mb:     [0,1] weight on model-based value at stage 1 (1-w_mb on MF).
    - lam:      [0,1] eligibility trace for backprop to stage-1 MF.
    - conf_t:   [0,1] confidence in known transitions (0.7/0.3); blends with an uninformed 0.5/0.5 prior.
    - beta1:    [0,10] baseline inverse temperature at stage 1.
    - beta2:    [0,10] baseline inverse temperature at stage 2.
    - k_pe:     [0,1] scaling for PE-driven temperature modulation; higher -> more noise after surprise.
    - run_sup:  [0,1] strength of run-length suppression at stage 1.
    - cf:       [0,1] counterfactual learning rate fraction for the unchosen alien (visited state).

    Inputs:
    - action_1: array-like (n_trials,) of 0/1 spaceship choices.
    - state:    array-like (n_trials,) of 0/1 planet reached.
    - action_2: array-like (n_trials,) of 0/1 alien choices.
    - reward:   array-like (n_trials,) of 0/1 coin outcomes.
    - model_parameters: iterable with parameters in the order above.

    Notes:
    - Known transition structure: action 0 commonly -> state 0, action 1 commonly -> state 1, with prob 0.7.
      Effective transition: T_eff = conf_t * T_known + (1-conf_t) * T_flat where T_flat has 0.5 in each row.
    - PE-scaled temperature: beta_eff = beta * max(0, 1 - k_pe * |PE|), applied separately at each stage.
      PE at stage 2 is used for stage-2 beta; a value PE for stage 1 is derived from current planning target.
    - Run-length suppression: subtract run_sup * tanh(run_len/5) from the logit of the repeated action.
    - Counterfactual update: in the visited state, the unchosen alien is nudged toward (1 - r) with rate alpha_r * cf.
    """
    alpha_r, w_mb, lam, conf_t, beta1, beta2, k_pe, run_sup, cf = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Transition structure
    T_known = np.array([[0.7, 0.3],
                        [0.3, 0.7]])
    T_flat = np.ones((2, 2)) * 0.5
    T = conf_t * T_known + (1.0 - conf_t) * T_flat

    # Values
    Q1_mf = np.zeros(2)
    Q2 = np.zeros((2, 2))

    loglik = 0.0

    # Track run length of repeating stage-1 actions
    prev_a1 = None
    run_len = 0

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # Model-based backup (hard max over stage-2)
        max_Q2 = np.max(Q2, axis=1)
        Q1_mb = T @ max_Q2

        # Combine MB and MF for stage-1 value
        Q1_comb = w_mb * Q1_mb + (1.0 - w_mb) * Q1_mf

        # Run-length suppression bias
        bias1 = np.zeros(2)
        if prev_a1 is not None:
            # Penalize repeating the same action with a saturating function of run length
            penalty = run_sup * np.tanh(run_len / 5.0)
            bias1[prev_a1] -= penalty

        # Stage-2 choice probabilities use PE-scaled temperature
        # Compute logits with baseline beta2 first to get probabilities for likelihood
        logits2_base = beta2 * Q2[s]
        logits2_base -= np.max(logits2_base)
        p2_base = np.exp(logits2_base)
        p2_base /= np.sum(p2_base)
        loglik += np.log(p2_base[a2] + eps)

        # After observing reward, compute PE and update Q2
        pe2 = r - Q2[s, a2]
        Q2[s, a2] += alpha_r * pe2

        # Counterfactual update for unchosen alien in visited state toward 1 - r
        unchosen = 1 - a2
        cf_target = 1.0 - r
        Q2[s, unchosen] += alpha_r * cf * (cf_target - Q2[s, unchosen])

        # Stage-1: compute a target value to define a stage-1 PE for temperature modulation
        target_s1 = (1.0 - lam) * Q2[s, a2] + lam * r
        pe1 = target_s1 - Q1_mf[a1]
        Q1_mf[a1] += alpha_r * pe1

        # PE-scaled effective temperatures for reporting choice likelihood (stage-1 next)
        beta1_eff = beta1 * max(0.0, 1.0 - k_pe * abs(pe1))
        # Use the immediate PE at stage 2 for the stage-2 effective temperature on next trial (not needed here for likelihood already computed)

        # Stage-1 choice likelihood
        logits1 = beta1_eff * Q1_comb + bias1
        logits1 -= np.max(logits1)
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        loglik += np.log(p1[a1] + eps)

        # Update run length tracker
        if prev_a1 is None or a1 != prev_a1:
            run_len = 1
        else:
            run_len += 1
        prev_a1 = a1

    return -loglik