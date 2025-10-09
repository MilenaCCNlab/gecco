def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """Hybrid MB/MF with surprise-adaptive transition learning, uncertainty-directed exploration,
    asymmetric Stage-2 learning, eligibility trace to Stage-1 MF, perseveration, decay, and lapses.
    Returns the negative log-likelihood (NLL) of observed choices.

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
        All parameters are used and bounded as:
        - alpha2_pos in [0,1]: Stage-2 learning rate for positive prediction errors.
        - alpha2_neg in [0,1]: Stage-2 learning rate for negative prediction errors.
        - alpha1 in [0,1]: Stage-1 model-free learning rate via eligibility trace.
        - lam in [0,1]: Eligibility trace strength from Stage-2 to Stage-1 MF.
        - w_mb in [0,1]: Weight of model-based relative to model-free at Stage-1.
        - beta1 in [0,10]: Softmax inverse temperature at Stage-1.
        - beta2 in [0,10]: Softmax inverse temperature at Stage-2.
        - kappa in [0,1]: Perseveration (choice kernel) weight applied at both stages.
        - alpha_T_base in [0,1]: Base learning rate for transitions (Dirichlet count update via surprise).
        - phi_dir in [0,1]: Directed exploration bonus scale from transition uncertainty (row entropy).
        - lapse in [0,1]: Lapse rate mixing with uniform choice.
        - decay in [0,1]: Forgetting toward 0.5 for Q-values (both stages).

    Notes
    -----
    - Transitions are learned via Dirichlet counts (initialized to 1's). Surprise-adaptive update:
      effective alpha_T = alpha_T_base + (1 - T[a1, s2]) * (1 - alpha_T_base).
    - Directed exploration: add phi_dir * H(T[a]) to Stage-1 MB action values (H = entropy of transition row).
    - Stage-2 learning is asymmetric for positive vs negative PEs.
    - Stage-1 MF receives eligibility-trace update from Stage-2 chosen Q.
    - Perseveration via decaying choice kernels at both stages.
    - Lapse mixes softmax with uniform (0.5).
    """
    (alpha2_pos, alpha2_neg, alpha1, lam, w_mb,
     beta1, beta2, kappa, alpha_T_base, phi_dir, lapse, decay) = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Dirichlet counts and implied transition estimates
    dir_counts = np.ones((2, 2), dtype=float)  # prior counts
    T = dir_counts / dir_counts.sum(axis=1, keepdims=True)

    # Q-values
    Q1_mf = np.zeros(2) + 0.5
    Q2 = np.zeros((2, 2)) + 0.5

    # Choice kernels (perseveration)
    K1 = np.zeros(2)
    K2 = np.zeros((2, 2))
    kernel_decay = 0.8  # structural decay factor (kappa scales its impact on preference)

    p1 = np.zeros(n_trials)
    p2 = np.zeros(n_trials)

    for t in range(n_trials):
        s2 = state[t]

        # Model-based values via learned transitions
        max_Q2 = np.max(Q2, axis=1)  # shape (2,)
        Q1_mb = T @ max_Q2  # shape (2,)

        # Transition uncertainty bonus: row entropy for each action
        row_ent = -np.sum(T * (np.log(T + eps)), axis=1)  # entropy in nats
        # Normalize entropy to [0,1] for two outcomes: max entropy = ln 2
        row_ent_norm = row_ent / np.log(2.0)

        # Stage 1 preference
        pref1 = w_mb * Q1_mb + (1.0 - w_mb) * Q1_mf + phi_dir * row_ent_norm + kappa * K1
        exp1 = np.exp(beta1 * (pref1 - np.max(pref1)))
        probs1 = exp1 / (np.sum(exp1) + eps)
        probs1 = (1.0 - lapse) * probs1 + lapse * 0.5

        a1 = action_1[t]
        p1[t] = probs1[a1]

        # Stage 2 preference
        pref2 = Q2[s2] + kappa * K2[s2]
        exp2 = np.exp(beta2 * (pref2 - np.max(pref2)))
        probs2 = exp2 / (np.sum(exp2) + eps)
        probs2 = (1.0 - lapse) * probs2 + lapse * 0.5

        a2 = action_2[t]
        p2[t] = probs2[a2]

        r = reward[t]

        # Stage-2 update (asymmetric)
        pe2 = r - Q2[s2, a2]
        alpha2 = alpha2_pos if pe2 >= 0.0 else alpha2_neg
        Q2[s2, a2] += alpha2 * pe2

        # Stage-1 MF via eligibility trace (bootstrap on immediate Stage-2 chosen value)
        target1 = Q2[s2, a2]
        pe1 = target1 - Q1_mf[a1]
        Q1_mf[a1] += alpha1 * lam * pe1

        # Surprise-adaptive transition learning (Dirichlet)
        surprise = 1.0 - T[a1, s2]
        eff_alpha_T = alpha_T_base + surprise * (1.0 - alpha_T_base)
        # Implement via interpolation between prior counts and increment-by-one
        # Equivalent to adding eff_alpha_T "pseudo-counts" to the observed outcome
        dir_counts[a1, s2] = (1.0 - eff_alpha_T) * dir_counts[a1, s2] + (eff_alpha_T) * (dir_counts[a1, s2] + 1.0)
        # Light decay for the unobserved outcome in that row to maintain normalization behavior
        other = 1 - s2
        dir_counts[a1, other] = (1.0 - eff_alpha_T) * dir_counts[a1, other] + (eff_alpha_T) * (dir_counts[a1, other] + 0.0)
        # Recompute T
        T = dir_counts / (dir_counts.sum(axis=1, keepdims=True) + eps)

        # Decay of Q-values toward neutral 0.5 (chosen and unchosen)
        Q2 = (1.0 - decay) * Q2 + decay * 0.5
        Q1_mf = (1.0 - decay) * Q1_mf + decay * 0.5

        # Update perseveration kernels
        K1 = kernel_decay * K1
        K1[a1] += 1.0

        K2[s2] = kernel_decay * K2[s2]
        K2[s2, a2] += 1.0

    nll = -(np.sum(np.log(p1 + eps)) + np.sum(np.log(p2 + eps)))
    return nll


def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """MB/MF hybrid with miscrediting at Stage-1 (confusion), planet-specific perseveration,
    fixed transitions, and lapses. Returns negative log-likelihood (NLL).

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
        All parameters are used and bounded as:
        - alpha2 in [0,1]: Stage-2 learning rate for Q-values.
        - alpha1_mf in [0,1]: Stage-1 model-free learning rate (from Stage-2 target).
        - xi_conf in [0,1]: Miscrediting strength to the unchosen Stage-1 action.
        - w_mb in [0,1]: Weight of model-based relative to model-free at Stage-1.
        - beta1 in [0,10]: Softmax inverse temperature at Stage-1.
        - beta2 in [0,10]: Softmax inverse temperature at Stage-2.
        - kappa1 in [0,1]: Perseveration weight at Stage-1 (choice kernel).
        - kappa2 in [0,1]: Perseveration weight at Stage-2 (planet-specific kernels).
        - bias_first in [0,1]: Constant bias toward spaceship A (adds to its preference).
        - lapse in [0,1]: Lapse rate mixing with uniform choice.

    Notes
    -----
    - Fixed transition structure is used for planning: T = [[0.7, 0.3], [0.3, 0.7]].
    - Miscrediting (confusion): after observing outcome, a fraction (xi_conf) of credit is
      also applied to the unchosen Stage-1 action, modeling imperfect attribution of outcome to spaceship.
    - Perseveration at Stage-2 is planet-specific; at Stage-1 it is global across spaceships.
    """
    (alpha2, alpha1_mf, xi_conf, w_mb, beta1, beta2,
     kappa1, kappa2, bias_first, lapse) = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Fixed transition matrix for MB component
    T = np.array([[0.7, 0.3],
                  [0.3, 0.7]], dtype=float)

    # Q-values
    Q2 = np.zeros((2, 2)) + 0.5
    Q1_mf = np.zeros(2) + 0.5

    # Choice kernels
    K1 = np.zeros(2)
    K2 = np.zeros((2, 2))
    kernel_decay = 0.8

    p1 = np.zeros(n_trials)
    p2 = np.zeros(n_trials)

    for t in range(n_trials):
        s2 = state[t]

        # Model-based values
        max_Q2 = np.max(Q2, axis=1)
        Q1_mb = T @ max_Q2

        # Stage 1 preference with bias toward A (action 0)
        bias_vec = np.array([bias_first, 0.0])
        pref1 = w_mb * Q1_mb + (1.0 - w_mb) * Q1_mf + kappa1 * K1 + bias_vec
        exp1 = np.exp(beta1 * (pref1 - np.max(pref1)))
        probs1 = exp1 / (np.sum(exp1) + eps)
        probs1 = (1.0 - lapse) * probs1 + lapse * 0.5

        a1 = action_1[t]
        p1[t] = probs1[a1]

        # Stage 2 preference (planet-specific perseveration)
        pref2 = Q2[s2] + kappa2 * K2[s2]
        exp2 = np.exp(beta2 * (pref2 - np.max(pref2)))
        probs2 = exp2 / (np.sum(exp2) + eps)
        probs2 = (1.0 - lapse) * probs2 + lapse * 0.5

        a2 = action_2[t]
        p2[t] = probs2[a2]

        r = reward[t]

        # Stage 2 Q update
        pe2 = r - Q2[s2, a2]
        Q2[s2, a2] += alpha2 * pe2

        # Stage 1 MF update using Stage-2 chosen value as target
        target1 = Q2[s2, a2]
        pe1 = target1 - Q1_mf[a1]
        Q1_mf[a1] += alpha1_mf * pe1

        # Miscrediting to unchosen Stage-1 action
        a1_alt = 1 - a1
        pe1_alt = target1 - Q1_mf[a1_alt]
        Q1_mf[a1_alt] += alpha1_mf * xi_conf * pe1_alt

        # Update perseveration kernels
        K1 = kernel_decay * K1
        K1[a1] += 1.0

        K2[s2] = kernel_decay * K2[s2]
        K2[s2, a2] += 1.0

    nll = -(np.sum(np.log(p1 + eps)) + np.sum(np.log(p2 + eps)))
    return nll


def cognitive_model3(action_1, state, action_2, reward, model_parameters):
    """Two-step policy-gradient (REINFORCE) with dynamic inverse temperature, entropy-controlled
    exploration, baseline for variance reduction, and lapses. Returns negative log-likelihood (NLL).

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
        All parameters are used and bounded as:
        - eta1 in [0,1]: Learning rate for Stage-1 policy weights.
        - eta2 in [0,1]: Learning rate for Stage-2 policy weights.
        - beta0 in [0,10]: Base inverse temperature for softmax.
        - beta_gain in [0,10]: Gain scaling of inverse temperature based on running reward rate.
        - entropy in [0,1]: Exploration control that downscales effective inverse temperature.
        - baseline_lr in [0,1]: Learning rate for scalar reward baseline (variance reduction).
        - lam_pg in [0,1]: Eligibility factor scaling Stage-1 update relative to Stage-2 advantage.
        - lapse in [0,1]: Lapse rate mixing with uniform choice at each stage.
        - bias_first in [0,1]: Constant bias added to Stage-1 logit for spaceship A.
        - alpha_rw in [0,1]: Learning rate for running reward average (drives beta dynamics).

    Notes
    -----
    - Policies are parameterized by weights (theta) per action; probabilities via softmax of scaled logits.
    - Dynamic inverse temperature: beta_t = min(beta0 + beta_gain * running_reward, 10), then
      beta_eff = beta_t * (1 - entropy).
    - REINFORCE updates with advantage = r - baseline. Stage-1 update scaled by lam_pg.
    - Lapse mixes model probabilities with uniform (0.5).
    """
    (eta1, eta2, beta0, beta_gain, entropy,
     baseline_lr, lam_pg, lapse, bias_first, alpha_rw) = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Policy weights
    theta1 = np.zeros(2)  # stage-1
    theta2 = np.zeros((2, 2))  # stage-2 per planet

    # Running averages
    baseline = 0.0
    run_rew = 0.0

    p1 = np.zeros(n_trials)
    p2 = np.zeros(n_trials)

    for t in range(n_trials):
        s2 = state[t]

        # Dynamic inverse temperature
        beta_t = beta0 + beta_gain * run_rew
        beta_t = min(max(beta_t, 0.0), 10.0)  # clamp to [0,10]
        beta_eff = beta_t * (1.0 - entropy)

        # Stage 1 probabilities (include fixed bias toward action 0)
        logits1 = beta_eff * (theta1 + np.array([bias_first, 0.0]))
        logits1 -= np.max(logits1)
        exp1 = np.exp(logits1)
        probs1 = exp1 / (np.sum(exp1) + eps)
        probs1 = (1.0 - lapse) * probs1 + lapse * 0.5

        a1 = action_1[t]
        p1[t] = probs1[a1]

        # Stage 2 probabilities
        logits2 = beta_eff * theta2[s2]
        logits2 -= np.max(logits2)
        exp2 = np.exp(logits2)
        probs2 = exp2 / (np.sum(exp2) + eps)
        probs2 = (1.0 - lapse) * probs2 + lapse * 0.5

        a2 = action_2[t]
        p2[t] = probs2[a2]

        r = reward[t]

        # Advantage (variance-reduced REINFORCE)
        adv = r - baseline

        # Policy gradient updates (using pre-lapse model probs for gradient; lapse only affects likelihood)
        # Recompute non-lapse probabilities for gradient calculation
        base_probs1 = exp1 / (np.sum(exp1) + eps)
        base_probs2 = exp2 / (np.sum(exp2) + eps)

        # Gradients of log-softmax: onehot - probs
        g1 = np.array([0.0, 0.0])
        g1[a1] = 1.0
        g1 -= base_probs1

        g2 = np.array([0.0, 0.0])
        g2[a2] = 1.0
        g2 -= base_probs2

        # Updates scaled by effective temperature (absorbed in logits already) and learning rates
        theta1 += eta1 * lam_pg * adv * g1
        theta2[s2] += eta2 * adv * g2

        # Update baseline and running reward average
        baseline += baseline_lr * (r - baseline)
        run_rew += alpha_rw * (r - run_rew)

    nll = -(np.sum(np.log(p1 + eps)) + np.sum(np.log(p2 + eps)))
    return nll