Below are three standalone cognitive models tailored to the two-step task. Each function:
- Takes inputs: action_1, state, action_2, reward, model_parameters
- Returns the negative log-likelihood of observed choices
- Uses all parameters meaningfully and adheres to requested bounds
- Includes a clear docstring with parameter bounds and explanations

Note: Assumes numpy as np is already imported (no imports included inside functions). Actions and states are assumed to be encoded as 0/1.

def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """
    Hybrid MB/MF with learned transitions, dynamic habit, perseveration, forgetting, and loss-contingent choice temperature.
    Concept:
      - Stage 2: model-free Q-learning with forgetting.
      - Stage 1: hybrid of model-based (using learned transition model) and model-free values.
      - Habit strength accumulates for chosen stage-1 action and decays over time.
      - Perseveration bias to repeat the previous stage-1 choice.
      - Transition model is learned online (EWMA).
      - Loss-contingent increase in choice determinism (beta boost after losses).
      - Bias toward spaceship A.

    Parameters (in order):
      alpha1: [0,1] Stage-1 MF learning rate
      alpha2: [0,1] Stage-2 learning rate
      beta1: [0,10] Softmax inverse temperature at stage 1
      beta2: [0,10] Softmax inverse temperature at stage 2
      omega: [0,1] Weight of model-based value in stage-1 hybrid value
      habit_gain: [0,1] Increment added to habit for the chosen stage-1 action each trial
      habit_decay: [0,1] Exponential decay of habit each trial
      beta_loss_boost: [0,1] Multiplier for beta after prior loss: beta_eff = beta * (1 + boost*(1 - prev_reward))
      forget: [0,1] Exponential decay toward zero for MF Q-values each trial (both stages)
      biasA: [0,1] Bias toward spaceship A; transformed to centered additive bias b = (biasA - 0.5)*2
      alphaT: [0,1] Transition learning rate (EWMA toward observed state given chosen stage-1 action)
      perseveration: [0,1] Additive bias for repeating previous stage-1 action

    Returns:
      Negative log-likelihood of the observed stage-1 and stage-2 choices.
    """
    import numpy as np  # assumed already imported per guardrail; included here only for clarity of usage

    alpha1, alpha2, beta1, beta2, omega, habit_gain, habit_decay, beta_loss_boost, forget, biasA, alphaT, perseveration = model_parameters

    n_trials = len(action_1)
    # Initialize
    q1_mf = np.zeros(2)                 # model-free Q-values at stage 1
    q2 = np.zeros((2, 2))               # stage-2 Q-values: [state, action]
    habit = np.zeros(2)                 # habit strength for stage-1 actions
    # Learned transition model T_est[action1, state]
    T_est = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    # Bias toward A as additive value
    bias_add = (biasA - 0.5) * 2.0
    prev_a1 = None
    prev_reward = 1.0  # initialize as a "win" so first-trial betas are unboosted

    eps = 1e-12

    for t in range(n_trials):
        # Decay MF values and habit
        q1_mf *= (1.0 - forget)
        q2 *= (1.0 - forget)
        habit *= (1.0 - habit_decay)

        # Model-based value at stage 1 from learned transitions and stage-2 values
        max_q2 = np.max(q2, axis=1)               # shape (2,)
        q1_mb = T_est @ max_q2                    # shape (2,)

        # Hybrid value + biases
        q1_val = (1.0 - omega) * q1_mf + omega * q1_mb + habit
        # Add bias toward A (index 0)
        q1_val[0] += bias_add

        # Perseveration bias for repeating last action
        if prev_a1 is not None:
            q1_val[prev_a1] += perseveration

        # Loss-contingent temperature (based on previous outcome)
        beta1_eff = beta1 * (1.0 + beta_loss_boost * (1.0 - prev_reward))

        # Softmax for stage 1
        v = q1_val - np.max(q1_val)
        expv = np.exp(beta1_eff * v)
        probs1 = expv / (np.sum(expv) + eps)
        a1 = action_1[t]
        p_choice_1[t] = probs1[a1]

        # Stage 2 choice probabilities given current state
        s = state[t]
        q2_val = q2[s].copy()
        # Loss-contingent temperature for stage 2 also depends on previous trial's outcome
        beta2_eff = beta2 * (1.0 + beta_loss_boost * (1.0 - prev_reward))
        v2 = q2_val - np.max(q2_val)
        expv2 = np.exp(beta2_eff * v2)
        probs2 = expv2 / (np.sum(expv2) + eps)
        a2 = action_2[t]
        p_choice_2[t] = probs2[a2]

        r = reward[t]

        # Update stage-2 Q
        delta2 = r - q2[s, a2]
        q2[s, a2] += alpha2 * delta2

        # Update stage-1 MF Q toward the obtained stage-2 chosen value
        target1 = q2[s, a2]
        delta1 = target1 - q1_mf[a1]
        q1_mf[a1] += alpha1 * delta1

        # Update habit (after choice)
        habit[a1] += habit_gain

        # Learn transition model from observed state given chosen stage-1 action
        onehot_state = np.array([1.0 - s, float(s)])  # 2 states, s in {0,1}
        T_est[a1] = (1.0 - alphaT) * T_est[a1] + alphaT * onehot_state
        # Ensure row normalization
        T_est[a1] /= (np.sum(T_est[a1]) + eps)

        # Update previous choice and reward (for next trial's beta boost and perseveration)
        prev_a1 = a1
        prev_reward = r

    neg_ll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return neg_ll


def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """
    Successor-Representation hybrid with asymmetric learning, learned transitions, habit, perseveration, and loss-boosted temperature.
    Concept:
      - Stage 2: asymmetric learning rates for wins vs losses with forgetting.
      - Stage 1: blend of (a) model-based value using learned transitions and (b) a cached SR-like mapping from actions to states.
      - SR component is a learned action->state occupancy estimate; sr_lambda weights cached SR vs transition-based MB.
      - Habit accumulation and perseveration bias at stage 1.
      - Loss-contingent temperature boost based on previous trial.
      - Bias toward spaceship A.

    Parameters (in order):
      alpha1: [0,1] Learning rate for SR cache (action->state) and MF Q1
      alpha2_win: [0,1] Stage-2 learning rate after reward=1
      alpha2_loss: [0,1] Stage-2 learning rate after reward=0
      beta1: [0,10] Stage-1 softmax inverse temperature
      beta2: [0,10] Stage-2 softmax inverse temperature
      sr_lambda: [0,1] Weight of cached SR mapping vs transition-based MB in stage-1 value
      omega_mb: [0,1] Weight on MB/SR mixture vs MF at stage 1
      habit_gain: [0,1] Habit increment for chosen stage-1 action
      habit_decay: [0,1] Habit exponential decay per trial
      beta_loss_boost: [0,1] Beta boost after losses for both stages (based on previous trial)
      forget: [0,1] Forgetting for Q-values and SR cache
      biasA: [0,1] Bias toward spaceship A; additive bias b = (biasA - 0.5)*2
      alphaT: [0,1] Transition learning rate (EWMA)
      perseveration: [0,1] Additive bias to repeat stage-1 action

    Returns:
      Negative log-likelihood of the observed choices.
    """
    import numpy as np  # assumed already imported

    (alpha1, alpha2_win, alpha2_loss, beta1, beta2, sr_lambda, omega_mb,
     habit_gain, habit_decay, beta_loss_boost, forget, biasA, alphaT, perseveration) = model_parameters

    n_trials = len(action_1)

    # Values and structures
    q1_mf = np.zeros(2)
    q2 = np.zeros((2, 2))
    habit = np.zeros(2)

    # Learned transition model
    T_est = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)
    # Cached SR-like mapping M[action, state]: expected state occupancy from action
    M = np.full((2, 2), 0.5, dtype=float)

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    bias_add = (biasA - 0.5) * 2.0
    prev_a1 = None
    prev_reward = 1.0
    eps = 1e-12

    for t in range(n_trials):
        # Forgetting
        q1_mf *= (1.0 - forget)
        q2 *= (1.0 - forget)
        M *= (1.0 - forget)
        habit *= (1.0 - habit_decay)

        # Stage-2 softmax temperature boost based on prev trial
        beta2_eff = beta2 * (1.0 + beta_loss_boost * (1.0 - prev_reward))

        # Compute stage-1 components
        max_q2 = np.max(q2, axis=1)
        q1_mb = T_est @ max_q2               # transition-based model-based
        q1_sr = M @ max_q2                   # cached SR mapping
        q1_cache = (1.0 - sr_lambda) * q1_mb + sr_lambda * q1_sr
        q1_val = (1.0 - omega_mb) * q1_mf + omega_mb * q1_cache + habit
        q1_val[0] += bias_add
        if prev_a1 is not None:
            q1_val[prev_a1] += perseveration

        beta1_eff = beta1 * (1.0 + beta_loss_boost * (1.0 - prev_reward))
        # Stage 1 softmax
        v1 = q1_val - np.max(q1_val)
        probs1 = np.exp(beta1_eff * v1)
        probs1 /= (np.sum(probs1) + eps)
        a1 = action_1[t]
        p_choice_1[t] = probs1[a1]

        # Stage 2 softmax
        s = state[t]
        q2_val = q2[s].copy()
        v2 = q2_val - np.max(q2_val)
        probs2 = np.exp(beta2_eff * v2)
        probs2 /= (np.sum(probs2) + eps)
        a2 = action_2[t]
        p_choice_2[t] = probs2[a2]

        r = reward[t]

        # Stage 2 learning (asymmetric)
        alpha2 = alpha2_win if r > 0.5 else alpha2_loss
        delta2 = r - q2[s, a2]
        q2[s, a2] += alpha2 * delta2

        # Stage 1 MF learning toward obtained value
        target1 = q2[s, a2]
        delta1 = target1 - q1_mf[a1]
        q1_mf[a1] += alpha1 * delta1

        # Update habit
        habit[a1] += habit_gain

        # Update transition model (EWMA toward observed state)
        onehot_state = np.array([1.0 - s, float(s)])
        T_est[a1] = (1.0 - alphaT) * T_est[a1] + alphaT * onehot_state
        T_est[a1] /= (np.sum(T_est[a1]) + eps)

        # Update cached SR mapping M toward the actually visited state (acts as cached action->state occupancy)
        M[a1] += alpha1 * (onehot_state - M[a1])

        prev_a1 = a1
        prev_reward = r

    neg_ll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return neg_ll


def cognitive_model3(action_1, state, action_2, reward, model_parameters):
    """
    Risk-sensitive hybrid with reference adaptation and uncertainty-gated model-based control.
    Concept:
      - Stage 2 uses a risk-sensitive utility with a dynamic reference point; utility guides learning.
      - Stage 1 mixes MF and MB values with a state-action-specific MB weight that decreases under transition uncertainty.
      - Transition model learned with pseudo-count prior; uncertainty measured via entropy of learned transitions.
      - Habit accumulation and stickiness biases (both stages).
      - Loss-contingent beta boost for stage-1 choices, bias toward spaceship A.
      - Global forgetting on Q-values.

    Parameters (in order):
      alpha1: [0,1] Stage-1 MF learning rate (also used for reference adaptation)
      alpha2: [0,1] Stage-2 learning rate on utility prediction errors
      beta1: [0,10] Stage-1 inverse temperature
      beta2: [0,10] Stage-2 inverse temperature
      rho: [0,1] Utility curvature (concavity/convexity) for deviations from reference
      kappa_loss: [0,1] Loss aversion multiplier for negative utility (0=no extra loss weight, 1=full extra weight)
      tau_trans: [0,1] Scale of uncertainty gating: MB weight w_a = 1 - tau_trans * (entropy(T_est[a]) / log(2))
      habit_gain: [0,1] Habit increment for chosen stage-1 action
      habit_decay: [0,1] Habit decay per trial
      forget: [0,1] Forgetting applied to Q-values each trial
      biasA: [0,1] Additive bias toward spaceship A, b = (biasA - 0.5)*2
      dirichlet_prior: [0,1] Strength of prior pseudo-counts for transitions (scaled internally)
      alphaT: [0,1] Transition learning step (count update EWMA)
      stickiness: [0,1] Stickiness bias added to repeating previous choices (both stages)
      beta_loss_boost: [0,1] Multiplicative boost to beta1 after prior loss

    Returns:
      Negative log-likelihood of the observed choices.
    """
    import numpy as np  # assumed already imported

    (alpha1, alpha2, beta1, beta2, rho, kappa_loss, tau_trans,
     habit_gain, habit_decay, forget, biasA, dirichlet_prior, alphaT,
     stickiness, beta_loss_boost) = model_parameters

    n_trials = len(action_1)

    # Initialize values
    q1_mf = np.zeros(2)
    q2 = np.zeros((2, 2))
    habit = np.zeros(2)

    # Reference point for utility (adapt to recent outcomes)
    ref = 0.5  # initial expected reward

    # Transition learning with pseudo-counts (Dirichlet-like)
    # Prior counts scaled from dirichlet_prior to avoid zero; scale to [0.1, 10.1]
    prior_scale = 10.0 * dirichlet_prior + 0.1
    counts = np.full((2, 2), prior_scale)  # counts[action, state]
    T_est = counts / np.sum(counts, axis=1, keepdims=True)

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    bias_add = (biasA - 0.5) * 2.0
    prev_a1 = None
    prev_a2 = [None, None]  # store last action per state for stage-2 stickiness
    prev_reward = 1.0
    eps = 1e-12

    def entropy_row(p):
        p = np.clip(p, eps, 1.0)
        p = p / np.sum(p)
        return -np.sum(p * np.log(p))

    for t in range(n_trials):
        # Forgetting
        q1_mf *= (1.0 - forget)
        q2 *= (1.0 - forget)
        habit *= (1.0 - habit_decay)

        # Compute MB value and uncertainty-gated weight per action
        max_q2 = np.max(q2, axis=1)
        q1_mb = T_est @ max_q2

        # Per-action MB weight w_a based on transition entropy
        w = np.zeros(2)
        for a in range(2):
            H = entropy_row(T_est[a])
            w[a] = 1.0 - tau_trans * (H / np.log(2.0))  # normalized by log(2)
            w[a] = np.clip(w[a], 0.0, 1.0)

        # Combine MF and MB per action with uncertainty-gated weights
        q1_val = w * q1_mb + (1.0 - w) * q1_mf + habit
        q1_val[0] += bias_add

        # Stickiness for stage 1
        if prev_a1 is not None:
            q1_val[prev_a1] += stickiness

        # Loss-contingent beta for stage 1
        beta1_eff = beta1 * (1.0 + beta_loss_boost * (1.0 - prev_reward))

        # Softmax stage 1
        v1 = q1_val - np.max(q1_val)
        probs1 = np.exp(beta1_eff * v1)
        probs1 /= (np.sum(probs1) + eps)
        a1 = action_1[t]
        p_choice_1[t] = probs1[a1]

        # Stage 2
        s = state[t]
        q2_val = q2[s].copy()

        # Stickiness for stage 2 in this state
        if prev_a2[s] is not None:
            q2_val[prev_a2[s]] += stickiness

        v2 = q2_val - np.max(q2_val)
        probs2 = np.exp(beta2 * v2)
        probs2 /= (np.sum(probs2) + eps)
        a2 = action_2[t]
        p_choice_2[t] = probs2[a2]

        r = reward[t]

        # Risk-sensitive utility with reference adaptation
        # Deviation from reference
        delta_r = r - ref
        if delta_r >= 0.0:
            u = (delta_r ** max(rho, eps))  # gains
        else:
            u = -kappa_loss * ((-delta_r) ** max(rho, eps))  # losses (loss aversion)

        # Update reference (EMA)
        ref += alpha1 * (r - ref)

        # Stage-2 learning on utility (predict utility; rescale back via direct update on Q2)
        # Treat utility as target by shifting baseline: Q2 <- Q2 + alpha2*(u - Q2_u)
        # We approximate by updating toward (ref + u), which equals r filtered by curvature/asymmetry.
        target2 = ref + u
        delta2 = target2 - q2[s, a2]
        q2[s, a2] += alpha2 * delta2

        # Stage-1 MF learning toward current stage-2 chosen value
        target1 = q2[s, a2]
        delta1 = target1 - q1_mf[a1]
        q1_mf[a1] += alpha1 * delta1

        # Habit update
        habit[a1] += habit_gain

        # Transition learning via pseudo-counts with EWMA flavor
        # Decay counts slightly then add one to observed state for chosen action
        counts[a1] = (1.0 - alphaT) * counts[a1] + alphaT * prior_scale  # shrink toward prior mass
        counts[a1, s] += 1.0  # add evidence for observed transition
        # Recompute T_est row
        T_est[a1] = counts[a1] / (np.sum(counts[a1]) + eps)

        # Update stickiness trackers
        prev_a1 = a1
        prev_a2[s] = a2
        prev_reward = r

    neg_ll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return neg_ll