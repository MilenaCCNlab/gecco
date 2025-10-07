def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    """Hybrid model-based/model-free RL with eligibility traces, separate stage temperatures,
    transition-confidence blending, and perseveration at both stages.
    
    This model blends model-based (MB) and model-free (MF) control at stage 1, learns MF action values
    at both stages, and propagates second-stage TD errors to stage 1 via an eligibility trace.
    The MB planner uses a transition matrix that is a convex mixture of the canonical task transitions
    and an agnostic (uniform) matrix, controlled by a confidence parameter. Perseveration biases
    at each stage capture choice stickiness.

    Parameters (model_parameters):
    - alpha1: [0,1] Learning rate for stage-1 MF value updates (via eligibility trace).
    - alpha2: [0,1] Learning rate for stage-2 MF value updates.
    - lambda_elig: [0,1] Eligibility trace strength to propagate stage-2 TD error back to stage 1.
    - w_mb: [0,1] Weight of model-based vs. model-free values at stage 1 (0 = pure MF, 1 = pure MB).
    - beta1: [0,10] Inverse temperature for stage-1 softmax.
    - beta2: [0,10] Inverse temperature for stage-2 softmax.
    - stick1: [0,1] Perseveration bias magnitude at stage 1 (added to the last chosen action's logit).
    - stick2: [0,1] Perseveration bias magnitude at stage 2.
    - tau_conf: [0,1] Transition confidence; blends canonical transitions with an agnostic (uniform) matrix.
                 tau_conf=1 uses canonical transitions exactly; tau_conf=0 uses uniform transitions.

    Inputs:
    - action_1: array of ints in {0,1}, chosen spaceship at stage 1 (0=A, 1=U).
    - state: array of ints in {0,1}, reached planet at stage 2 (0=X, 1=Y).
    - action_2: array of ints in {0,1}, chosen alien at stage 2 (per planet; 0/1 indexing).
    - reward: array of floats in [0,1], obtained coins (0/1).
    
    Returns:
    - Negative log-likelihood of the observed stage-1 and stage-2 choices under the model.
    """
    import numpy as np  # assumed available per guardrails

    alpha1, alpha2, lambda_elig, w_mb, beta1, beta2, stick1, stick2, tau_conf = model_parameters
    n_trials = len(action_1)

    T_canonical = np.array([[0.7, 0.3],
                            [0.3, 0.7]])
    T_uniform = np.full((2, 2), 0.5)
    T_eff = tau_conf * T_canonical + (1.0 - tau_conf) * T_uniform

    q1_mf = np.zeros(2)        # stage-1 MF values for spaceships A/U
    q2_mf = np.zeros((2, 2))   # stage-2 MF values per planet (X/Y) and alien (0/1)

    last_a1 = None
    last_a2 = [None, None]  # keep last chosen alien for each planet independently

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    for t in range(n_trials):

        max_q2 = np.max(q2_mf, axis=1)              # length-2: best alien per planet
        q1_mb = T_eff @ max_q2                      # length-2 MB values per spaceship

        q1 = w_mb * q1_mb + (1.0 - w_mb) * q1_mf

        bias1 = np.zeros(2)
        if last_a1 is not None:
            bias1[last_a1] += stick1

        logits1 = beta1 * q1 + bias1
        logits1 -= np.max(logits1)  # numerical stability
        probs1 = np.exp(logits1)
        probs1 /= np.sum(probs1)
        a1 = action_1[t]
        p_choice_1[t] = probs1[a1]

        s = state[t]

        bias2 = np.zeros(2)
        if last_a2[s] is not None:
            bias2[last_a2[s]] += stick2

        logits2 = beta2 * q2_mf[s] + bias2
        logits2 -= np.max(logits2)
        probs2 = np.exp(logits2)
        probs2 /= np.sum(probs2)
        a2 = action_2[t]
        p_choice_2[t] = probs2[a2]

        r = reward[t]

        delta2 = r - q2_mf[s, a2]
        q2_mf[s, a2] += alpha2 * delta2

        q1_mf[a1] += alpha1 * lambda_elig * delta2

        last_a1 = a1
        last_a2[s] = a2

    eps = 1e-12
    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll