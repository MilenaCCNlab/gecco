Below are three standalone cognitive models that compute the negative log-likelihood of observed two-step choices. Each model uses all provided parameters meaningfully, adheres to the requested bounds, and explores different mechanisms that build on the strengths you highlighted (habit, decay, asymmetric sensitivity to loss via beta, forgetting, priors/bias at stage 1), while introducing alternative control/learning dynamics.

Note: Assume numpy is already imported as np externally (no imports inside functions).

def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """Dual-controller arbitration via surprise-driven weighting with habit, forgetting, and loss-dependent temperature.
    
    Mechanism:
    - Stage-2 model-free learning with learning rates alpha1 (for stage-1 MF bootstrap) and alpha2 (stage-2 MF).
    - Model-based evaluation at stage 1 from a fixed transition model.
    - Dynamic arbitration weight w_t between MB and MF driven by recent surprise (unsigned stage-2 RPE):
        when surprise is high, rely more on MF; when low, rely more on MB.
    - Choice habit (choice kernel) at stage 1 with gain and decay.
    - Forgetting for unchosen Q-values toward a neutral baseline (0.5).
    - Loss-dependent boost of inverse temperature on both stages.
    - Action bias toward spaceship A (action 0).
    
    Parameters (all in [0,1] except betas in [0,10]):
    - alpha1: [0,1] stage-1 MF learning rate via bootstrapping from stage-2 value.
    - alpha2: [0,1] stage-2 MF learning rate from reward.
    - beta1: [0,10] inverse temperature at stage 1.
    - beta2: [0,10] inverse temperature at stage 2.
    - w0: [0,1] initial arbitration weight toward model-based control at stage 1.
    - kappa_arbitration: [0,1] update rate for arbitration weight based on surprise.
    - habit_gain: [0,1] gain for the choice habit signal at stage 1.
    - habit_decay: [0,1] decay of habit across trials; higher means slower decay.
    - beta_loss_boost: [0,1] fractional increase of inverse temperature on loss (reward==0).
    - forget: [0,1] forgetting rate toward 0.5 for unchosen Q-values (both stages).
    - biasA: [0,1] bias toward action A (index 0); mapped to signed bias in [-bmax, +bmax].
    
    Inputs:
    - action_1: array-like of length T with values in {0,1} for stage-1 choices.
    - state: array-like of length T with values in {0,1} for planet X/Y.
    - action_2: array-like of length T with values in {0,1} for stage-2 alien choices.
    - reward: array-like of length T with scalar rewards (e.g., 0/1).
    - model_parameters: list or array of 11 parameters in the order above.
    
    Returns:
    - Negative log-likelihood of the observed stage-1 and stage-2 choices under the model.
    """
    alpha1, alpha2, beta1, beta2, w0, kappa, habit_gain, habit_decay, beta_loss_boost, forget, biasA = model_parameters
    n_trials = len(action_1)

    # Fixed transition structure: A->X common (0.7), U->Y common (0.7)
    transition_matrix = np.array([[0.7, 0.3],
                                  [0.3, 0.7]])

    # Initialize value functions
    q1_mf = np.zeros(2)               # stage-1 MF values for actions A/U
    q2_mf = np.zeros((2, 2))          # stage-2 MF values for states X/Y and actions (aliens)
    habit = np.zeros(2)               # stage-1 habit kernel
    w = w0                            # arbitration weight toward MB

    # For likelihood
    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    # Bias scale: map biasA in [0,1] to [-bmax, +bmax]
    bmax = 2.0
    bias_term = (biasA - 0.5) * 2 * bmax  # add to action-0 only

    eps = 1e-12

    for t in range(n_trials):
        s = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r = reward[t]

        # Model-based value at stage 1: expected max over second-stage actions per planet
        max_q2 = np.max(q2_mf, axis=1)  # shape (2,)
        q1_mb = transition_matrix @ max_q2  # shape (2,)

        # Arbitration weight updated by surprise of last trial (use last available w)
        # Compute current combined Q at stage 1
        q1_combined = w * q1_mb + (1 - w) * q1_mf + habit_gain * habit
        # Add bias to action 0
        q1_with_bias = q1_combined.copy()
        q1_with_bias[0] += bias_term

        # Loss-dependent beta scaling
        loss_boost = 1.0 + beta_loss_boost * (1.0 - r)  # if r=0, boost; if r=1, no boost
        beta1_eff = beta1 * loss_boost
        beta2_eff = beta2 * loss_boost

        # Softmax stage 1
        q1s = q1_with_bias - np.max(q1_with_bias)
        exp_q1 = np.exp(beta1_eff * q1s)
        probs_1 = exp_q1 / (np.sum(exp_q1) + eps)
        p_choice_1[t] = max(probs_1[a1], eps)

        # Softmax stage 2
        q2s = q2_mf[s].copy()
        q2s -= np.max(q2s)
        exp_q2 = np.exp(beta2_eff * q2s)
        probs_2 = exp_q2 / (np.sum(exp_q2) + eps)
        p_choice_2[t] = max(probs_2[a2], eps)

        # -------------------------
        # Learning updates
        # -------------------------

        # Stage-2 MF learning
        pe2 = r - q2_mf[s, a2]
        q2_mf[s, a2] += alpha2 * pe2

        # Forgetting for unchosen stage-2 actions toward 0.5
        baseline = 0.5
        for a in range(2):
            if a != a2:
                q2_mf[s, a] = (1 - forget) * q2_mf[s, a] + forget * baseline
        # Also decay unvisited state's actions slightly toward baseline
        s_other = 1 - s
        q2_mf[s_other, :] = (1 - forget) * q2_mf[s_other, :] + forget * baseline

        # Stage-1 MF bootstrapping from stage-2 value (SARSA-style)
        target1 = q2_mf[s, a2]
        pe1 = target1 - q1_mf[a1]
        q1_mf[a1] += alpha1 * pe1
        # Forgetting for unchosen stage-1 action
        a1_other = 1 - a1
        q1_mf[a1_other] = (1 - forget) * q1_mf[a1_other] + forget * baseline

        # Habit update (choice kernel)
        habit *= habit_decay
        habit[a1] += (1 - habit_decay)  # accumulate to chosen action

        # Arbitration update: w moves opposite to surprise (high surprise -> lower w)
        # Surprise proxy: unsigned stage-2 RPE normalized to [0,1] by clipping
        surprise = min(abs(pe2), 1.0)
        desired_w = 1.0 - surprise
        w = (1 - kappa) * w + kappa * desired_w

    # Negative log-likelihood
    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll


def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """Transition-learning model combining Successor-like cached prediction with MF control, plus habit and asymmetry.
    
    Mechanism:
    - Learns transition matrix T_hat online with learning rate trans_alpha.
    - Stage-2 MF learning with asymmetric learning rates for reward vs omission.
    - Stage-1 value is a dynamic mixture of:
        • Cached "successor-like" predictor V_SR[a] := m[a]·v, where m[a] is a learned state-occupancy vector
          updated from experienced states (sr_alpha, sr_lambda), and v is current max Q2 per state.
        • MF bootstrap value Q1_MF.
      Mixture weight theta_t is higher when transitions are certain (low entropy of T_hat row for chosen action).
    - Choice habit kernel at stage 1 (habit_gain, habit_decay).
    - Loss-dependent temperature boost (beta_loss_boost) for both stages.
    - Forgetting toward 0.5 for unchosen Q-values (both stages).
    - Bias toward spaceship A (biasA).
    
    Parameters (all in [0,1] except betas in [0,10]):
    - alpha_r: [0,1] learning rate for MF when reward=1.
    - alpha_pun: [0,1] learning rate for MF when reward=0 (omission).
    - beta1: [0,10] inverse temperature at stage 1.
    - beta2: [0,10] inverse temperature at stage 2.
    - sr_alpha: [0,1] learning rate for the SR-like state-occupancy vector m[a].
    - sr_lambda: [0,1] eligibility persistence for m[a] (controls carry-over from previous m[a]).
    - trans_alpha: [0,1] learning rate for transition matrix T_hat updates.
    - habit_gain: [0,1] gain for stage-1 habit kernel.
    - habit_decay: [0,1] decay of habit kernel.
    - beta_loss_boost: [0,1] fractional increase of inverse temperature on loss.
    - forget: [0,1] forgetting rate toward 0.5 for unchosen values.
    - biasA: [0,1] bias toward action A (index 0), mapped to [-bmax, +bmax].
    
    Inputs:
    - action_1, state, action_2, reward: arrays of length T.
    - model_parameters: list/array of 12 parameters in the order above.
    
    Returns:
    - Negative log-likelihood of observed choices.
    """
    (alpha_r, alpha_pun, beta1, beta2, sr_alpha, sr_lambda, trans_alpha,
     habit_gain, habit_decay, beta_loss_boost, forget, biasA) = model_parameters

    n_trials = len(action_1)

    # Initialize transition estimates (rows are actions A/U, columns are states X/Y)
    T_hat = np.array([[0.5, 0.5],
                      [0.5, 0.5]], dtype=float)

    # Values
    q1_mf = np.zeros(2)
    q2_mf = np.zeros((2, 2))
    habit = np.zeros(2)

    # Successor-like occupancy vectors for each stage-1 action over states X/Y
    m = np.zeros((2, 2))  # each row a: m[a, s] ~ expected occupancy of state s after choosing action a

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    # Bias mapping
    bmax = 2.0
    bias_term = (biasA - 0.5) * 2 * bmax

    eps = 1e-12

    for t in range(n_trials):
        s = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r = reward[t]

        # Current state value vector from stage-2 MF
        v_states = np.max(q2_mf, axis=1)  # shape (2,)

        # SR-like cached value for each stage-1 action: m[a] dot v_states
        v_sr = m @ v_states  # shape (2,)

        # Entropy-based arbitration weight (higher certainty -> more SR/MB reliance)
        # Use row entropy of T_hat for each action; weight is 1 - normalized entropy
        def row_weight(row):
            p = np.clip(row, 1e-6, 1 - 1e-6)
            H = -np.sum(p * np.log(p))
            Hmax = np.log(2.0)
            return 1.0 - (H / Hmax)  # in [0,1]

        theta_rows = np.array([row_weight(T_hat[0]), row_weight(T_hat[1])])
        # Action-specific mixture: here we approximate by averaging rows to form a vector
        theta = theta_rows  # vector per action

        # Stage-1 composite value
        q1_comp = theta * v_sr + (1 - theta) * q1_mf + habit_gain * habit
        q1_with_bias = q1_comp.copy()
        q1_with_bias[0] += bias_term

        # Loss-dependent beta
        loss_boost = 1.0 + beta_loss_boost * (1.0 - r)
        beta1_eff = beta1 * loss_boost
        beta2_eff = beta2 * loss_boost

        # Softmax stage 1
        q1s = q1_with_bias - np.max(q1_with_bias)
        exp_q1 = np.exp(beta1_eff * q1s)
        probs_1 = exp_q1 / (np.sum(exp_q1) + eps)
        p_choice_1[t] = max(probs_1[a1], eps)

        # Softmax stage 2
        q2s = q2_mf[s].copy()
        q2s -= np.max(q2s)
        exp_q2 = np.exp(beta2_eff * q2s)
        probs_2 = exp_q2 / (np.sum(exp_q2) + eps)
        p_choice_2[t] = max(probs_2[a2], eps)

        # -------------------------
        # Learning updates
        # -------------------------

        # Transition learning for chosen action: move row toward observed state
        target_row = np.array([0.0, 0.0])
        target_row[s] = 1.0
        T_hat[a1] = (1 - trans_alpha) * T_hat[a1] + trans_alpha * target_row
        # Keep rows normalized (they already are convex combos)

        # Stage-2 MF learning with asymmetric rates
        alpha2 = alpha_r if r > 0 else alpha_pun
        pe2 = r - q2_mf[s, a2]
        q2_mf[s, a2] += alpha2 * pe2

        # Forgetting at stage 2 toward 0.5
        baseline = 0.5
        for a in range(2):
            if a != a2:
                q2_mf[s, a] = (1 - forget) * q2_mf[s, a] + forget * baseline
        s_other = 1 - s
        q2_mf[s_other, :] = (1 - forget) * q2_mf[s_other, :] + forget * baseline

        # Stage-1 MF bootstrapping target: expected value under learned transitions for chosen action
        v_states_post = np.max(q2_mf, axis=1)
        target1 = np.dot(T_hat[a1], v_states_post)
        pe1 = target1 - q1_mf[a1]
        q1_mf[a1] += alpha2 * pe1  # use same asymmetric alpha2 to couple credit across stages
        a1_other = 1 - a1
        q1_mf[a1_other] = (1 - forget) * q1_mf[a1_other] + forget * baseline

        # SR-like occupancy update for chosen action:
        # decay previous occupancy and add current observed state as target
        m[a1] = (1 - sr_alpha) * (sr_lambda * m[a1]) + sr_alpha * target_row
        # mild decay for unchosen action toward uniform (agnostic prior)
        m[a1_other] = (1 - forget) * m[a1_other] + forget * 0.5

        # Habit update
        habit *= habit_decay
        habit[a1] += (1 - habit_decay)

    eps = 1e-12
    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll


def cognitive_model3(action_1, state, action_2, reward, model_parameters):
    """Affect-modulated control with risk sensitivity, lapse, and transition-dependent credit assignment.
    
    Mechanism:
    - Latent "mood/motivation" m_t in [0,1] integrates recent signed prediction errors and scales both learning and choice temperature.
      High mood increases effective beta and learning rates; low mood reduces them (mood_lr, mood_volatility).
    - Risk sensitivity transforms reward via a concave/convex utility u(r) = r^rho (rho in [0,1]).
    - Mixture of MB and MF at stage 1 depends on whether the observed transition was common vs rare:
      on common transitions weight = lambda_credit toward MB; on rare transitions weight shifts toward MF.
    - Choice habit kernel at stage 1 with decay.
    - Loss-dependent temperature boost.
    - Lapse epsilon mixes softmax with uniform choice at both stages.
    - Forgetting toward 0.5 for unchosen values and planet-specific decay.
    - Bias toward spaceship A.
    
    Parameters (all in [0,1] except betas in [0,10]):
    - alpha1: [0,1] base learning rate for stage-1 MF (bootstrapped).
    - alpha2: [0,1] base learning rate for stage-2 MF (from reward utility).
    - beta1: [0,10] base inverse temperature at stage 1.
    - beta2: [0,10] base inverse temperature at stage 2.
    - mood_lr: [0,1] integration rate of mood toward recent signed RPEs.
    - mood_volatility: [0,1] leak/decay of mood to neutral 0.5; higher -> more stable mood.
    - rho: [0,1] risk sensitivity exponent for reward utility u = r^rho.
    - epsilon: [0,1] lapse rate; mixes softmax with uniform choice.
    - habit_gain: [0,1] habit kernel gain at stage 1.
    - habit_decay: [0,1] decay of habit kernel.
    - beta_loss_boost: [0,1] fractional increase of inverse temperature on loss.
    - forget: [0,1] forgetting rate toward 0.5 for unchosen values.
    - biasA: [0,1] bias toward action A (index 0), mapped to [-bmax, +bmax].
    - lambda_credit: [0,1] weight toward MB on common transitions; on rare transitions weight flips to (1-lambda_credit).
    
    Inputs:
    - action_1, state, action_2, reward: arrays of length T.
    - model_parameters: list/array of 14 parameters in the order above.
    
    Returns:
    - Negative log-likelihood of observed choices.
    """
    (alpha1, alpha2, beta1, beta2, mood_lr, mood_volatility, rho, epsilon,
     habit_gain, habit_decay, beta_loss_boost, forget, biasA, lambda_credit) = model_parameters

    n_trials = len(action_1)

    # Fixed transitions for common vs rare determination
    T = np.array([[0.7, 0.3],
                  [0.3, 0.7]])

    # Values
    q1_mf = np.zeros(2)
    q2_mf = np.zeros((2, 2))
    habit = np.zeros(2)

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    # Bias mapping
    bmax = 2.0
    bias_term = (biasA - 0.5) * 2 * bmax

    # Latent mood in [0,1], initialize neutral
    m = 0.5

    eps = 1e-12

    for t in range(n_trials):
        s = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r_raw = reward[t]

        # Utility-transformed reward (risk sensitivity)
        r = (r_raw + 0.0) ** max(rho, 1e-6)

        # Determine if transition was common for chosen action
        # Common if T[a1, s] >= 0.5
        is_common = 1.0 if T[a1, s] >= 0.5 else 0.0

        # Compute MB value at stage 1
        max_q2 = np.max(q2_mf, axis=1)
        q1_mb = T @ max_q2

        # Mixture weight depends on transition type (trial-wise credit)
        w_mb = lambda_credit * is_common + (1 - lambda_credit) * (1 - is_common)

        # Mood-modulated effective learning rates and temperatures
        mood_gain = 0.5 + m  # in [0.5, 1.5]
        # Loss-dependent temperature scaling
        loss_boost = 1.0 + beta_loss_boost * (1.0 - r_raw)
        beta1_eff = beta1 * mood_gain * loss_boost
        beta2_eff = beta2 * mood_gain * loss_boost
        alpha1_eff = np.clip(alpha1 * mood_gain, 0.0, 1.0)
        alpha2_eff = np.clip(alpha2 * mood_gain, 0.0, 1.0)

        # Stage-1 composite value
        q1_comp = w_mb * q1_mb + (1 - w_mb) * q1_mf + habit_gain * habit
        q1_with_bias = q1_comp.copy()
        q1_with_bias[0] += bias_term

        # Softmax stage 1 with lapse
        q1s = q1_with_bias - np.max(q1_with_bias)
        exp_q1 = np.exp(beta1_eff * q1s)
        soft_1 = exp_q1 / (np.sum(exp_q1) + eps)
        probs_1 = (1 - epsilon) * soft_1 + epsilon * 0.5
        p_choice_1[t] = max(probs_1[a1], eps)

        # Softmax stage 2 with lapse
        q2s = q2_mf[s].copy()
        q2s -= np.max(q2s)
        exp_q2 = np.exp(beta2_eff * q2s)
        soft_2 = exp_q2 / (np.sum(exp_q2) + eps)
        probs_2 = (1 - epsilon) * soft_2 + epsilon * 0.5
        p_choice_2[t] = max(probs_2[a2], eps)

        # -------------------------
        # Learning updates
        # -------------------------

        # Stage-2 MF learning with utility reward
        pe2 = r - q2_mf[s, a2]
        q2_mf[s, a2] += alpha2_eff * pe2

        # Forgetting at stage 2 toward 0.5
        baseline = 0.5
        for a in range(2):
            if a != a2:
                q2_mf[s, a] = (1 - forget) * q2_mf[s, a] + forget * baseline
        s_other = 1 - s
        q2_mf[s_other, :] = (1 - forget) * q2_mf[s_other, :] + forget * baseline

        # Stage-1 MF bootstrapping from visited state's chosen action
        target1 = q2_mf[s, a2]
        pe1 = target1 - q1_mf[a1]
        q1_mf[a1] += alpha1_eff * pe1
        a1_other = 1 - a1
        q1_mf[a1_other] = (1 - forget) * q1_mf[a1_other] + forget * baseline

        # Habit update
        habit *= habit_decay
        habit[a1] += (1 - habit_decay)

        # Mood update: integrate signed RPEs (combine stage-2 and stage-1 signals)
        signed_pe = 0.5 * (np.clip(pe2, -1, 1) + np.clip(pe1, -1, 1))
        # Leaky integration toward neutral 0.5 with volatility controlling leak
        m = (mood_volatility * m + (1 - mood_volatility) * 0.5) + mood_lr * signed_pe
        m = float(np.clip(m, 0.0, 1.0))

    eps = 1e-12
    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll