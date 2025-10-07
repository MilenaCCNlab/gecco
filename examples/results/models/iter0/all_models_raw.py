def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """Hybrid model-based/model-free RL with learned transitions, eligibility traces, and perseveration.
    
    This model blends a learned model-based action value with model-free values at stage 1, 
    learns stage-2 values from reward, propagates reward to stage 1 via an eligibility trace, 
    learns the transition structure online, and includes perseveration (choice stickiness) at both stages.
    A small lapse parameter mixes the softmax policy with uniform random choice.

    Parameters (model_parameters):
    - alpha1: [0,1] learning rate for stage-1 model-free Q-values
    - alpha2: [0,1] learning rate for stage-2 model-free Q-values
    - beta1:  [0,10] inverse temperature for stage-1 softmax
    - beta2:  [0,10] inverse temperature for stage-2 softmax
    - w_mb:   [0,1] weight of model-based value in stage-1 action values (1=fully model-based)
    - lam:    [0,1] eligibility trace; propagates stage-2 reward prediction error to stage-1 MF values
    - kappa1: [0,1] perseveration weight (stage-1): bias to repeat previous spaceship choice
    - kappa2: [0,1] perseveration weight (stage-2): bias to repeat previous alien choice (within the visited state)
    - trans_alpha: [0,1] transition learning rate to update P(state | action_1)
    - lapse:  [0,1] lapse rate; with probability lapse, choices are uniformly random (2 options)

    Inputs:
    - action_1: array-like, int in {0,1} chosen spaceship each trial (0=A, 1=U)
    - state:    array-like, int in {0,1} visited planet each trial (0=X, 1=Y)
    - action_2: array-like, int in {0,1} chosen alien on visited planet each trial
    - reward:   array-like, float in {0,1} coin outcome
    - model_parameters: list/array of 10 floats within the bounds above

    Returns:
    - Negative log-likelihood of the observed stage-1 and stage-2 choices under the model.
    """
    alpha1, alpha2, beta1, beta2, w_mb, lam, kappa1, kappa2, trans_alpha, lapse = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Initialize learned transition model P(s | a1). Start with common=0.7, rare=0.3 prior.
    # Shape (2 actions, 2 states)
    trans = np.array([[0.7, 0.3],
                      [0.3, 0.7]], dtype=float)

    # Model-free Q-values
    q1_mf = np.zeros(2)               # stage-1 MF values for actions {0,1}
    q2_mf = np.full((2, 2), 0.5)      # stage-2 MF values for states {0,1} and actions {0,1}

    # For perseveration (stickiness)
    prev_a1 = None
    prev_a2_in_state = [None, None]   # track previous a2 for each state separately

    loglik = 0.0

    for t in range(n_trials):
        s = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r = reward[t]

        # Model-based stage-1 value: V_MB(a1) = sum_s P(s|a1) * max_a2 Q2_MF(s, a2)
        max_q2 = np.max(q2_mf, axis=1)  # per state
        q1_mb = trans @ max_q2          # shape (2,)

        # Add perseveration at stage 1
        stick1 = np.zeros(2)
        if prev_a1 is not None:
            stick1[prev_a1] = 1.0

        # Combine MB and MF at stage 1
        q1 = w_mb * q1_mb + (1.0 - w_mb) * q1_mf + kappa1 * stick1

        # Stage-1 choice probabilities (with lapse)
        pref1 = beta1 * q1
        pref1 -= np.max(pref1)  # numerical stability
        soft1 = np.exp(pref1)
        soft1 /= np.sum(soft1)
        probs1 = (1.0 - lapse) * soft1 + lapse * 0.5

        p1 = probs1[a1]
        loglik += np.log(p1 + eps)

        # Stage-2: add within-state perseveration
        stick2 = np.zeros(2)
        if prev_a2_in_state[s] is not None:
            stick2[prev_a2_in_state[s]] = 1.0
        q2_pref = q2_mf[s] + kappa2 * stick2

        pref2 = beta2 * q2_pref
        pref2 -= np.max(pref2)
        soft2 = np.exp(pref2)
        soft2 /= np.sum(soft2)
        probs2 = (1.0 - lapse) * soft2 + lapse * 0.5

        p2 = probs2[a2]
        loglik += np.log(p2 + eps)

        # Learning updates
        # 1) Transition learning for chosen a1 towards observed state s
        #    Move probability mass towards the observed state
        for st in (0, 1):
            target = 1.0 if st == s else 0.0
            trans[a1, st] = (1 - trans_alpha) * trans[a1, st] + trans_alpha * target
        # Renormalize to avoid drift due to numerical issues
        trans[a1] /= np.sum(trans[a1])

        # 2) Stage-2 TD update
        pe2 = r - q2_mf[s, a2]
        q2_mf[s, a2] += alpha2 * pe2

        # 3) Stage-1 MF update with eligibility trace
        #    Two components: classic SARSA-style backup and eligibility from reward PE
        #    (a) bootstrapped value from chosen second-stage action
        backup = q2_mf[s, a2]
        pe1_boot = backup - q1_mf[a1]
        q1_mf[a1] += alpha1 * pe1_boot
        #    (b) propagate reward PE with lambda
        q1_mf[a1] += alpha1 * lam * pe2

        # Update perseveration trackers
        prev_a1 = a1
        prev_a2_in_state[s] = a2

    return -loglik


def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """Asymmetric learning with surprise-modulated MB weighting and learned transitions.

    This model implements:
    - Asymmetric stage-2 learning rates for positive vs negative outcomes.
    - Surprise-adaptive effective learning rate and dynamic arbitration:
      the weight on model-based control increases with transition surprise.
    - A single stickiness term applied at both stages.
    - Online learning of the transition matrix.

    Parameters (model_parameters):
    - alpha_pos: [0,1] stage-2 learning rate when reward=1
    - alpha_neg: [0,1] stage-2 learning rate when reward=0
    - beta1:     [0,10] inverse temperature at stage 1
    - beta2:     [0,10] inverse temperature at stage 2
    - w0:        [0,1] baseline MB weight (at zero surprise)
    - phi:       [0,1] surprise gain; increases MB weight with transition surprise
    - trans_alpha: [0,1] learning rate for transition probabilities
    - kappa:     [0,1] choice stickiness applied at both stages

    Inputs:
    - action_1: array-like, int in {0,1}
    - state:    array-like, int in {0,1}
    - action_2: array-like, int in {0,1}
    - reward:   array-like, float in {0,1}
    - model_parameters: list/array of 8 floats within the bounds above

    Returns:
    - Negative log-likelihood of the observed choices.
    """
    alpha_pos, alpha_neg, beta1, beta2, w0, phi, trans_alpha, kappa = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Learned transition model initialized to common=0.7
    trans = np.array([[0.7, 0.3],
                      [0.3, 0.7]], dtype=float)

    # Stage-1 MF values and Stage-2 Q-values
    q1_mf = np.zeros(2)
    q2 = np.full((2, 2), 0.5)

    prev_a1 = None
    prev_a2_in_state = [None, None]

    loglik = 0.0

    for t in range(n_trials):
        s = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r = reward[t]

        # Surprise = 1 - P(observed state | chosen a1)
        surprise = 1.0 - trans[a1, s]
        # Dynamic MB weight increases with surprise: w = clip(w0 + phi * surprise, 0, 1)
        w = w0 + phi * surprise
        w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)

        # Model-based Q1 via learned transitions
        max_q2 = np.max(q2, axis=1)
        q1_mb = trans @ max_q2

        # Stickiness features
        stick1 = np.zeros(2)
        if prev_a1 is not None:
            stick1[prev_a1] = 1.0

        stick2 = np.zeros(2)
        if prev_a2_in_state[s] is not None:
            stick2[prev_a2_in_state[s]] = 1.0

        # Stage-1 action values and choice prob
        q1 = w * q1_mb + (1.0 - w) * q1_mf + kappa * stick1
        pref1 = beta1 * q1
        pref1 -= np.max(pref1)
        probs1 = np.exp(pref1)
        probs1 /= np.sum(probs1)
        loglik += np.log(probs1[a1] + eps)

        # Stage-2 preferences and choice prob
        q2_pref = q2[s] + kappa * stick2
        pref2 = beta2 * q2_pref
        pref2 -= np.max(pref2)
        probs2 = np.exp(pref2)
        probs2 /= np.sum(probs2)
        loglik += np.log(probs2[a2] + eps)

        # Learning updates
        # Transition learning (online Bayesian-like delta rule)
        for st in (0, 1):
            target = 1.0 if st == s else 0.0
            trans[a1, st] = (1 - trans_alpha) * trans[a1, st] + trans_alpha * target
        trans[a1] /= np.sum(trans[a1])

        # Stage-2 asymmetric learning, modulated by surprise (attention)
        alpha = alpha_pos if r > q2[s, a2] else alpha_neg
        # Scale learning rate by 1 + surprise (ensuring within [0,1] via clipping)
        alpha_eff = alpha * (0.5 + 0.5 * (surprise if surprise < 1.0 else 1.0))
        alpha_eff = 0.0 if alpha_eff < 0.0 else (1.0 if alpha_eff > 1.0 else alpha_eff)
        pe2 = r - q2[s, a2]
        q2[s, a2] += alpha_eff * pe2

        # Stage-1 MF bootstrapped update towards current state max
        target1 = q2[s, a2]
        pe1 = target1 - q1_mf[a1]
        q1_mf[a1] += (alpha_pos * 0.5 + alpha_neg * 0.5) * pe1  # mean of pos/neg rates to keep both used

        # Update stickiness trackers
        prev_a1 = a1
        prev_a2_in_state[s] = a2

    return -loglik


def cognitive_model3(action_1, state, action_2, reward, model_parameters):
    """Habit formation with MB guidance, Q-value forgetting, exploration control, and action bias.

    This model combines:
    - Model-based guidance from a fixed transition structure (common=0.7).
    - Habit system at stage 1 that strengthens the tendency to repeat rewarded spaceship choices,
      with decay over time.
    - Stage-2 model-free learning with forgetting toward a neutral prior.
    - Meta-control that increases exploration after losses (beta scaling).
    - A static bias favoring spaceship A.

    Parameters (model_parameters):
    - alpha2:         [0,1] stage-2 learning rate for Q-values
    - beta1:          [0,10] base inverse temperature at stage 1
    - beta2:          [0,10] inverse temperature at stage 2
    - habit_gain:     [0,1] increment of habit strength assigned to chosen stage-1 action when rewarded
    - habit_decay:    [0,1] decay of habit preferences each trial (applied to both actions)
    - beta_loss_boost:[0,1] fractional decrease in beta1 after a loss; beta1_eff = beta1 * (1 - beta_loss_boost) if last reward=0
    - forget:         [0,1] forgetting rate toward 0.5 baseline for stage-2 Q-values each trial
    - biasA:          [0,1] constant additive bias for spaceship A (action_1==0)

    Inputs:
    - action_1: array-like, int in {0,1}
    - state:    array-like, int in {0,1}
    - action_2: array-like, int in {0,1}
    - reward:   array-like, float in {0,1}
    - model_parameters: list/array of 8 floats within the bounds above

    Returns:
    - Negative log-likelihood of the observed choices.
    """
    alpha2, beta1, beta2, habit_gain, habit_decay, beta_loss_boost, forget, biasA = model_parameters

    n_trials = len(action_1)
    eps = 1e-12

    # Fixed transition structure (common=0.7)
    trans = np.array([[0.7, 0.3],
                      [0.3, 0.7]], dtype=float)

    # Stage-2 Q-values with forgetting (initialize neutral)
    q2 = np.full((2, 2), 0.5)

    # Habit preferences over stage-1 actions
    habit_pref = np.zeros(2)  # additive preference toward actions

    # Optional: prior reward for beta scaling
    prev_reward = 1.0  # start neutral; setting 1 avoids initial exploration penalty

    loglik = 0.0

    for t in range(n_trials):
        s = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r = reward[t]

        # Model-based stage-1 value via fixed transitions
        max_q2 = np.max(q2, axis=1)
        q1_mb = trans @ max_q2

        # Effective beta1 depending on previous outcome: explore more after loss
        beta1_eff = beta1 * (1.0 - beta_loss_boost * (1.0 - prev_reward))

        # Stage-1 total preference: MB value + habit + static bias for A
        bias_vec = np.array([biasA, 0.0])
        q1_pref = q1_mb + habit_pref + bias_vec

        # Stage-1 choice probability
        pref1 = beta1_eff * q1_pref
        pref1 -= np.max(pref1)
        probs1 = np.exp(pref1)
        probs1 /= np.sum(probs1)
        loglik += np.log(probs1[a1] + eps)

        # Stage-2 choice probability
        pref2 = beta2 * q2[s]
        pref2 -= np.max(pref2)
        probs2 = np.exp(pref2)
        probs2 /= np.sum(probs2)
        loglik += np.log(probs2[a2] + eps)

        # Learning
        # 1) Stage-2 forgetting toward 0.5, then TD update
        # Forgetting pulls values toward neutral before incorporating new outcome
        q2 = (1.0 - forget) * q2 + forget * 0.5
        pe2 = r - q2[s, a2]
        q2[s, a2] += alpha2 * pe2

        # 2) Habit dynamics at stage 1: decay, then reinforce chosen action if rewarded
        habit_pref *= (1.0 - habit_decay)
        habit_pref[a1] += habit_gain * r

        # Track previous reward for next trial's beta scaling
        prev_reward = r

    return -loglik