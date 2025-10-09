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

    T = np.array([[0.7, 0.3],
                  [0.3, 0.7]], dtype=float)

    Q1_mf = np.zeros(2) + 0.5  # neutral start
    Q2 = np.zeros((2, 2)) + 0.5

    K1 = np.zeros(2)
    K2 = np.zeros((2, 2))

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    eps = 1e-12

    for t in range(n_trials):
        s2 = state[t]

        max_Q2 = np.max(Q2, axis=1)  # shape (2,)
        Q1_mb = T @ max_Q2  # shape (2,)

        pref1 = w_mb * Q1_mb + (1.0 - w_mb) * Q1_mf + kappa1 * K1

        exp1 = np.exp(beta1 * (pref1 - np.max(pref1)))
        probs1 = exp1 / (np.sum(exp1) + eps)
        probs1 = (1.0 - lapse) * probs1 + lapse * 0.5

        a1 = action_1[t]
        p_choice_1[t] = probs1[a1]

        pref2 = Q2[s2].copy() + kappa2 * K2[s2]
        exp2 = np.exp(beta2 * (pref2 - np.max(pref2)))
        probs2 = exp2 / (np.sum(exp2) + eps)
        probs2 = (1.0 - lapse) * probs2 + lapse * 0.5

        a2 = action_2[t]
        p_choice_2[t] = probs2[a2]

        r = reward[t]

        pe2 = r - Q2[s2, a2]
        Q2[s2, a2] += alpha2 * pe2


        target1 = Q2[s2, a2]
        pe1 = target1 - Q1_mf[a1]
        Q1_mf[a1] += alpha1 * lam * pe1

        Q2 = (1.0 - decay) * Q2 + decay * 0.5
        Q1_mf = (1.0 - decay) * Q1_mf + decay * 0.5

        oh = np.array([1.0 if s2 == 0 else 0.0, 1.0 if s2 == 1 else 0.0])
        T[a1] = (1.0 - alpha_T) * T[a1] + alpha_T * oh

        T[a1] = T[a1] / (np.sum(T[a1]) + eps)

        K1 = 0.8 * K1  # fixed decay of kernel memory (structure; value modulated by kappa1)
        K1[a1] += 1.0

        K2[s2] = 0.8 * K2[s2]
        K2[s2, a2] += 1.0

    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll