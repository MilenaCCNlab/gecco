Below are three standalone Python functions implementing distinct cognitive models for the two-step task. Each returns the negative log-likelihood of the observed choices, and uses all parameters meaningfully within the specified bounds.

def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """Bayesian reward learner with leaky Beta posteriors, arbitration-by-uncertainty, MF backup, and lapses.
    Returns the negative log-likelihood of observed stage-1 and stage-2 choices.

    Cognitive assumptions:
    - Each alien’s reward probability is tracked with a Beta posterior that leaks back toward a prior (volatility).
    - Stage-1 uses a mixture of model-based (planning via fixed-parameter transitions) and model-free values.
    - Arbitration: the MB/MF mixture weight increases with global uncertainty in stage-2 reward estimates.
    - Stage-2 values also updated model-free with a learning rate.
    - Simple eligibility-like backpropagation from stage-2 to stage-1 MF values.
    - Choice perseveration via a learned choice kernel updated with its own learning rate.
    - Lapse noise (independent of value) at each stage.

    Parameters (all in [0,1] except betas in [0,10]):
    - prior_strength: [0,1] strength of symmetric Beta prior for each alien (maps to pseudo-count 1..10).
    - leak:           [0,1] leaky pull of posteriors back to prior on each trial.
    - trans_common:   [0,1] assumed “common” transition prob for A→X and U→Y (the complement goes to the other planet).
    - mb_base:        [0,1] baseline weight for model-based value at stage 1.
    - arb_unc:        [0,1] additional MB weight modulated by global reward uncertainty (entropy/variance).
    - alpha2:         [0,1] model-free learning rate for stage-2 Q-values.
    - backprop:       [0,1] strength of backup from stage-2 value/reward to stage-1 MF value.
    - beta1:          [0,10] inverse temperature for stage-1 softmax.
    - beta2:          [0,10] inverse temperature for stage-2 softmax.
    - lapse:          [0,1] lapse probability mixed uniformly into both stages’ policies.
    - kappa:          [0,1] learning rate for a choice kernel (perseveration) shared across stages.

    Inputs:
    - action_1: array-like of length n_trials, 0/1 for spaceship A/U chosen.
    - state:    array-like of length n_trials, 0/1 for planet X/Y reached.
    - action_2: array-like of length n_trials, 0/1 for alien chosen on that planet.
    - reward:   array-like of length n_trials, 0/1 coin outcome.
    - model_parameters: iterable with parameters in the order above.
    """
    prior_strength, leak, trans_common, mb_base, arb_unc, alpha2, backprop, beta1, beta2, lapse, kappa = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Transition matrix parameterized by trans_common (A->X, U->Y are "common")
    T = np.array([
        [trans_common, 1.0 - trans_common],  # P(X|A), P(Y|A)
        [1.0 - trans_common, trans_common],  # P(X|U), P(Y|U)
    ])

    # Beta-Bernoulli reward posteriors for each planet's two aliens
    prior_cnt = 1.0 + 9.0 * prior_strength  # map [0,1] to [1,10]
    a_post = np.ones((2, 2)) * prior_cnt
    b_post = np.ones((2, 2)) * prior_cnt

    # Model-free values
    Q2_mf = np.zeros((2, 2))
    Q1_mf = np.zeros(2)

    # Choice kernels (perseveration)
    CK1 = np.zeros(2)
    CK2 = np.zeros((2, 2))

    loglik = 0.0

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # Current Bayesian means and uncertainties
        p_hat = a_post / (a_post + b_post)  # means
        var_hat = (a_post * b_post) / ((a_post + b_post) ** 2 * (a_post + b_post + 1.0))  # variances

        # MB stage-1 values via planning over max alien on each planet
        max_Q2_est = np.max(p_hat, axis=1)  # proxy for expected reward
        Q1_mb = T @ max_Q2_est

        # Global uncertainty: mean variance of the best alien per planet
        idx_best = np.argmax(p_hat, axis=1)
        best_vars = var_hat[np.arange(2), idx_best]
        unc_glob = np.mean(best_vars)  # in [0, ~0.25]

        # Arbitration weight
        w_mb = mb_base + arb_unc * np.clip((unc_glob / 0.25), 0.0, 1.0)
        w_mb = np.clip(w_mb, 0.0, 1.0)

        # Combine MF and MB at stage 1
        Q1_comb = w_mb * Q1_mb + (1.0 - w_mb) * Q1_mf

        # Stage-1 choice with choice kernel and lapse
        logits1 = beta1 * Q1_comb + CK1
        logits1 -= np.max(logits1)
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        p1 = (1.0 - lapse) * p1 + lapse * 0.5
        loglik += np.log(p1[a1] + eps)

        # Stage-2 choice from MF values plus kernel (use MF estimates, not p_hat, to avoid double-count)
        logits2 = beta2 * Q2_mf[s] + CK2[s]
        logits2 -= np.max(logits2)
        p2 = np.exp(logits2)
        p2 /= np.sum(p2)
        p2 = (1.0 - lapse) * p2 + lapse * 0.5
        loglik += np.log(p2[a2] + eps)

        # Updates:
        # 1) Leaky Beta posteriors pulled toward prior, then add current outcome
        a_post = (1.0 - leak) * a_post + leak * prior_cnt
        b_post = (1.0 - leak) * b_post + leak * prior_cnt
        a_post[s, a2] += r
        b_post[s, a2] += (1.0 - r)

        # 2) Stage-2 MF update
        delta2 = r - Q2_mf[s, a2]
        Q2_mf[s, a2] += alpha2 * delta2

        # 3) Backprop to stage-1 MF value (combining value and immediate reward)
        target_s1 = (1.0 - backprop) * Q2_mf[s, a2] + backprop * r
        Q1_mf[a1] += alpha2 * (target_s1 - Q1_mf[a1])

        # 4) Update choice kernels toward chosen actions
        CK1 *= (1.0 - kappa)
        CK1[a1] += kappa
        CK2[s] *= (1.0 - kappa)
        CK2[s, a2] += kappa

    return -loglik


def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """Successor-like transition learner with MF stage-2, asymmetric learning, and decaying perseveration.
    Returns the negative log-likelihood of observed choices.

    Cognitive assumptions:
    - Stage-1 learns an action→planet “successor” mapping M (like a 1-step SR) with discount gamma.
      M[a] encodes expected discounted occupancy of planets after choosing action a.
    - The mapping updates from observed states with a controllable “sharpness” toward the reached state.
    - Stage-1 action values are a mixture of SR-based planning (M @ max Q2) and MF Q1 values.
    - Stage-2 MF values learn asymmetrically from positive vs negative outcomes.
    - Action perseveration at both stages decays over time.

    Parameters (all in [0,1] except betas in [0,10]):
    - alpha_sr:   [0,1] learning rate for the SR-like mapping M.
    - sharp:      [0,1] sharpness of the SR update toward the reached state (0=blunt/flat, 1=one-hot).
    - omega:      [0,1] mixture weight for SR-based value at stage 1 (1-omega is MF).
    - gamma:      [0,1] discount factor in SR (short horizon here, but controls mass toward reached state).
    - alpha_pos2: [0,1] stage-2 MF learning rate for positive prediction errors.
    - alpha_neg2: [0,1] stage-2 MF learning rate for negative prediction errors.
    - beta1:      [0,10] inverse temperature for stage-1 softmax.
    - beta2:      [0,10] inverse temperature for stage-2 softmax.
    - rho_stick:  [0,1] perseveration strength added to last chosen action at both stages.
    - decay_st:   [0,1] decay of perseveration traces each trial.

    Inputs:
    - action_1: array-like of length n_trials, 0/1 for spaceship A/U chosen.
    - state:    array-like of length n_trials, 0/1 for planet X/Y reached.
    - action_2: array-like of length n_trials, 0/1 for alien chosen on that planet.
    - reward:   array-like of length n_trials, 0/1 coin outcome.
    - model_parameters: iterable with parameters in the order above.
    """
    alpha_sr, sharp, omega, gamma, alpha_pos2, alpha_neg2, beta1, beta2, rho_stick, decay_st = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # SR-like mapping: M[a, s] ~ P(s | a) with discount; initialized uniform
    M = np.ones((2, 2)) * 0.5

    # Stage-2 MF values and stage-1 MF values
    Q2_mf = np.zeros((2, 2))
    Q1_mf = np.zeros(2)

    # Perseveration traces
    prev_a1 = None
    prev_a2_by_state = [None, None]
    stick1 = np.zeros(2)
    stick2 = np.zeros((2, 2))

    loglik = 0.0

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # Planning via SR: expected best alien per planet
        max_Q2 = np.max(Q2_mf, axis=1)  # (2,)
        Q1_sr = M @ max_Q2

        # Mix SR and MF at stage 1
        Q1 = omega * Q1_sr + (1.0 - omega) * Q1_mf

        # Perseveration biases
        stick1 *= (1.0 - decay_st)
        stick2 *= (1.0 - decay_st)
        if prev_a1 is not None:
            stick1[prev_a1] += rho_stick
        if prev_a2_by_state[s] is not None:
            stick2[s, prev_a2_by_state[s]] += rho_stick

        # Stage-1 choice
        logits1 = beta1 * Q1 + stick1
        logits1 -= np.max(logits1)
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        loglik += np.log(p1[a1] + eps)

        # Stage-2 choice
        logits2 = beta2 * Q2_mf[s] + stick2[s]
        logits2 -= np.max(logits2)
        p2 = np.exp(logits2)
        p2 /= np.sum(p2)
        loglik += np.log(p2[a2] + eps)

        # Stage-2 MF learning with asymmetry
        pe2 = r - Q2_mf[s, a2]
        alpha2 = alpha_pos2 if pe2 >= 0.0 else alpha_neg2
        Q2_mf[s, a2] += alpha2 * pe2

        # Stage-1 MF backup from stage-2 (no separate lambda; use gamma as bootstrapping weight)
        target1 = (1.0 - gamma) * Q2_mf[s, a2] + gamma * r
        Q1_mf[a1] += alpha2 * (target1 - Q1_mf[a1])  # reuse alpha2 from sign of PE to keep asymmetry aligned

        # SR-like update toward reached state with sharpness, and discount shaping
        # Construct target distribution for the chosen action: blend flat and one-hot at reached state
        flat = np.array([0.5, 0.5])
        onehot = np.array([1.0 if i == s else 0.0 for i in range(2)])
        target_M = (1.0 - sharp) * flat + sharp * onehot
        # Apply discount mass toward reached state
        target_M = (1.0 - gamma) * flat + gamma * target_M
        # Update only the chosen row
        M[a1, :] += alpha_sr * (target_M - M[a1, :])
        # Renormalize row to a proper distribution
        row_sum = np.sum(M[a1, :])
        if row_sum > 0:
            M[a1, :] /= row_sum

        # Update perseveration memory
        prev_a1 = a1
        prev_a2_by_state[s] = a2

    return -loglik


def cognitive_model3(action_1, state, action_2, reward, model_parameters):
    """Volatile reward with hazard-rate resets, utility curvature, novelty bonus, and planet-bias planning.
    Returns the negative log-likelihood of observed choices.

    Cognitive assumptions:
    - Each alien’s reward probability is tracked via Beta posteriors subject to change-points with hazard h.
      Hazard acts like a probabilistic reset toward a symmetric prior each trial (volatility).
    - Outcomes are evaluated via a utility function u(r) = r^alpha_u (risk sensitivity).
    - Stage-2 choices rely on MF utilities plus a novelty bonus that decays with visitation.
    - Stage-1 planning uses a fixed prior over transitions (common vs rare) and adds a planet-utility bias:
      actions leading (in expectation) to higher-utility planets gain additional logit bias.
    - Both stages include epsilon-greedy lapses in addition to softmax control.
    - A constant side bias toward spaceship A is included.

    Parameters (all in [0,1] except betas in [0,10]):
    - hazard:     [0,1] change-point hazard; higher values reset reward posteriors more strongly to prior.
    - prior_str:  [0,1] strength of symmetric Beta prior for each alien (maps to pseudo-count 1..10).
    - alpha_u:    [0,1] utility curvature exponent in u(r) = r^alpha_u.
    - trans_c:    [0,1] assumed “common” transition prob (A→X, U→Y common).
    - k_planet:   [0,1] scale of planet-utility bias added to stage-1 logits.
    - eta_nov:    [0,1] novelty bonus magnitude added to stage-2 logits (inverse with sqrt visit count).
    - beta1:      [0,10] inverse temperature for stage-1 softmax.
    - beta2:      [0,10] inverse temperature for stage-2 softmax.
    - eps1:       [0,1] epsilon-greedy lapse at stage 1.
    - eps2:       [0,1] epsilon-greedy lapse at stage 2.
    - bias_side:  [0,1] constant bias toward spaceship A (mapped to ± around zero for logits).

    Inputs:
    - action_1: array-like of length n_trials, 0/1 for spaceship A/U chosen.
    - state:    array-like of length n_trials, 0/1 for planet X/Y reached.
    - action_2: array-like of length n_trials, 0/1 for alien chosen on that planet.
    - reward:   array-like of length n_trials, 0/1 coin outcome.
    - model_parameters: iterable with parameters in the order above.
    """
    hazard, prior_str, alpha_u, trans_c, k_planet, eta_nov, beta1, beta2, eps1, eps2, bias_side = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Transitions
    T = np.array([
        [trans_c, 1.0 - trans_c],
        [1.0 - trans_c, trans_c],
    ])

    # Volatile Beta posteriors for each alien with hazard resets
    prior_cnt = 1.0 + 9.0 * prior_str
    a_post = np.ones((2, 2)) * prior_cnt
    b_post = np.ones((2, 2)) * prior_cnt

    # MF utility values for stage-2 (initialized from prior mean 0.5^alpha_u, but start at 0)
    Q2 = np.zeros((2, 2))

    # Novelty counters
    N_vis = np.zeros((2, 2))

    # Constant side bias toward A in logits
    bias_logit = (bias_side - 0.5) * 2.0  # maps 0..1 to -1..+1

    loglik = 0.0

    for t in range(n_trials):
        a1 = int(action_1[t])
        s = int(state[t])
        a2 = int(action_2[t])
        r = float(reward[t])

        # Posterior means per alien from Beta
        p_hat = a_post / (a_post + b_post)
        # Utility-transformed values for each alien
        U_hat = p_hat ** max(alpha_u, eps)  # use exponent on expectation as a simple risk transform proxy

        # Planet utilities = best alien on each planet
        planet_util = np.max(U_hat, axis=1)

        # Stage-1 logits: softmax over planned utilities + planet-bias term + side bias
        Q1_mb = T @ planet_util  # expected planet utility under each spaceship
        # Add planet-bias term that favors actions leading to better planet differentials
        # Compute bias for each action as expected (util_X - util_Y) weighted by the action's transition row
        diff = planet_util[0] - planet_util[1]
        planet_bias = k_planet * np.array([T[0, 0] - T[0, 1], T[1, 0] - T[1, 1]]) * diff  # action-specific

        logits1 = beta1 * Q1_mb + planet_bias
        logits1[0] += bias_logit  # bias toward A
        logits1 -= np.max(logits1)
        p1 = np.exp(logits1)
        p1 /= np.sum(p1)
        # Epsilon-greedy mixture
        p1 = (1.0 - eps1) * p1 + eps1 * 0.5
        loglik += np.log(p1[a1] + eps)

        # Stage-2 logits: MF Q2 plus novelty bonus
        novelty = eta_nov / np.sqrt(N_vis[s] + 1.0)
        logits2 = beta2 * Q2[s] + novelty
        logits2 -= np.max(logits2)
        p2 = np.exp(logits2)
        p2 /= np.sum(p2)
        p2 = (1.0 - eps2) * p2 + eps2 * 0.5
        loglik += np.log(p2[a2] + eps)

        # Observe utility-transformed outcome
        u = r ** max(alpha_u, eps)

        # Hazard-based posterior reset toward prior in expectation, then add current evidence
        a_post = (1.0 - hazard) * a_post + hazard * prior_cnt
        b_post = (1.0 - hazard) * b_post + hazard * prior_cnt
        a_post[s, a2] += r
        b_post[s, a2] += (1.0 - r)

        # Update MF stage-2 utilities with a simple delta rule toward u
        pe2 = u - Q2[s, a2]
        # Use a PE-dependent adaptive rate derived from posterior confidence: smaller updates when confident
        # Confidence ~ 1 / variance of Beta; var in (0, ~0.25), normalize to [0,1]
        var_sa = (a_post[s, a2] * b_post[s, a2]) / ((a_post[s, a2] + b_post[s, a2]) ** 2 * (a_post[s, a2] + b_post[s, a2] + 1.0))
        conf = 1.0 - np.clip(var_sa / 0.25, 0.0, 1.0)
        alpha_eff = conf  # in [0,1]
        Q2[s, a2] += alpha_eff * pe2

        # Update novelty counters
        N_vis[s, a2] += 1.0

    return -loglik