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

    trans = np.array([[0.7, 0.3],
                      [0.3, 0.7]], dtype=float)

    q2 = np.full((2, 2), 0.5)

    habit_pref = np.zeros(2)  # additive preference toward actions

    prev_reward = 1.0  # start neutral; setting 1 avoids initial exploration penalty

    loglik = 0.0

    for t in range(n_trials):
        s = state[t]
        a1 = action_1[t]
        a2 = action_2[t]
        r = reward[t]

        max_q2 = np.max(q2, axis=1)
        q1_mb = trans @ max_q2

        beta1_eff = beta1 * (1.0 - beta_loss_boost * (1.0 - prev_reward))

        bias_vec = np.array([biasA, 0.0])
        q1_pref = q1_mb + habit_pref + bias_vec

        pref1 = beta1_eff * q1_pref
        pref1 -= np.max(pref1)
        probs1 = np.exp(pref1)
        probs1 /= np.sum(probs1)
        loglik += np.log(probs1[a1] + eps)

        pref2 = beta2 * q2[s]
        pref2 -= np.max(pref2)
        probs2 = np.exp(pref2)
        probs2 /= np.sum(probs2)
        loglik += np.log(probs2[a2] + eps)



        q2 = (1.0 - forget) * q2 + forget * 0.5
        pe2 = r - q2[s, a2]
        q2[s, a2] += alpha2 * pe2

        habit_pref *= (1.0 - habit_decay)
        habit_pref[a1] += habit_gain * r

        prev_reward = r

    return -loglik