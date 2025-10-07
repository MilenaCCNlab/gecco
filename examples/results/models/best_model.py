def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    """Model-based RL with learned transitions, reward sensitivity, forgetting,
    choice kernels at both stages, and lapses.

    This model learns the transition probabilities online and plans using the learned matrix.
    Second-stage MF values are learned with a forgetting term. Choice kernels capture
    short-term choice recency at both stages and act as additive biases in the logits.
    A lapse parameter mixes softmax policy with a uniform policy. Reward sensitivity scales
    the effective reward used for learning.

    Parameters (model_parameters):
    - alpha_T: [0,1] Transition learning rate for updating the chosen row of the transition matrix.
    - alpha_Q: [0,1] Learning rate for stage-2 MF Q-value updates.
    - forget: [0,1] Forgetting rate pulling stage-2 Q-values toward 0.5 each trial.
    - beta1: [0,10] Inverse temperature for stage-1 softmax.
    - beta2: [0,10] Inverse temperature for stage-2 softmax.
    - kappa1: [0,1] Choice-kernel learning rate and weight at stage 1 (recency bias magnitude).
    - kappa2: [0,1] Choice-kernel learning rate and weight at stage 2 (per planet).
    - eta: [0,1] Lapse probability; mixes softmax with uniform choices at each stage.
    - rho: [0,1] Reward sensitivity; scales the reward used for learning (r_eff = rho * r).

    Inputs:
    - action_1: array of ints in {0,1}, chosen spaceship at stage 1 (0=A, 1=U).
    - state: array of ints in {0,1}, reached planet at stage 2 (0=X, 1=Y).
    - action_2: array of ints in {0,1}, chosen alien at stage 2 (per planet; 0/1 indexing).
    - reward: array of floats in [0,1], obtained coins (0/1).

    Returns:
    - Negative log-likelihood of the observed stage-1 and stage-2 choices under the model.
    """
    import numpy as np  # assumed available

    alpha_T, alpha_Q, forget, beta1, beta2, kappa1, kappa2, eta, rho = model_parameters
    n_trials = len(action_1)

    T = np.full((2, 2), 0.5)

    q2 = np.full((2, 2), 0.5)

    kernel1 = np.zeros(2)       # stage-1 kernel across spaceships
    kernel2 = np.zeros((2, 2))  # stage-2 kernel per planet

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    eps = 1e-12

    for t in range(n_trials):

        q2 = (1.0 - forget) * q2 + forget * 0.5

        max_q2 = np.max(q2, axis=1)     # best alien per planet
        q1_mb = T @ max_q2

        logits1 = beta1 * q1_mb + kappa1 * kernel1
        logits1 -= np.max(logits1)
        probs1 = np.exp(logits1)
        probs1 /= np.sum(probs1)

        probs1 = (1.0 - eta) * probs1 + eta * 0.5

        a1 = action_1[t]
        p_choice_1[t] = probs1[a1]

        s = state[t]
        logits2 = beta2 * q2[s] + kappa2 * kernel2[s]
        logits2 -= np.max(logits2)
        probs2 = np.exp(logits2)
        probs2 /= np.sum(probs2)
        probs2 = (1.0 - eta) * probs2 + eta * 0.5

        a2 = action_2[t]
        p_choice_2[t] = probs2[a2]

        r_eff = rho * reward[t]

        target = np.array([0.0, 0.0])
        target[s] = 1.0
        T[a1] = (1.0 - alpha_T) * T[a1] + alpha_T * target

        T[a1] = np.clip(T[a1], eps, 1.0)
        T[a1] /= np.sum(T[a1])

        delta2 = r_eff - q2[s, a2]
        q2[s, a2] += alpha_Q * delta2

        kernel1 *= (1.0 - kappa1)
        kernel1[a1] += kappa1
        kernel2[s] *= (1.0 - kappa2)
        kernel2[s, a2] += kappa2

    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll