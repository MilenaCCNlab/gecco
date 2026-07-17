"""Likelihood twin of the config's hybrid simulation model (Daw hybrid with
eligibility trace + perseveration). Field-standard baseline; cross-checked
against the data's baseline_bic column (WARN-level in evaluate)."""

HYBRID_BOUNDS = [(0.0, 1.0), (0.0, 1.0), (0.0, 10.0), (0.0, 10.0),
                 (0.0, 1.0), (0.0, 1.0), (0.0, 5.0)]

HYBRID_SOURCE = '''def cognitive_model(action_1, state, action_2, reward, model_parameters):
    """
    Hybrid MB/MF model with eligibility trace and perseveration (Daw baseline).

    Parameters:
    learning_rate: [0, 1] - stage-1 learning rate
    learning_rate_2: [0, 1] - stage-2 learning rate
    beta: [0, 10] - stage-1 inverse temperature
    beta_2: [0, 10] - stage-2 inverse temperature
    w: [0, 1] - MB weight
    lambd: [0, 1] - eligibility trace
    perseveration: [0, 5] - stage-1 choice repetition bonus
    """
    learning_rate, learning_rate_2, beta, beta_2, w, lambd, perseveration = model_parameters
    n_trials = len(action_1)
    transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    q_mf = np.zeros((3, 2))
    pers_array = np.zeros(2)
    log_loss = 0.0
    eps = 1e-10
    for trial in range(n_trials):
        a1 = int(action_1[trial])
        s2 = int(state[trial])
        a2 = int(action_2[trial])
        r = float(reward[trial])
        max_q_stage2 = np.max(q_mf[1:], axis=1)
        q_mb = transition_matrix @ max_q_stage2
        q_net = w * q_mb + (1 - w) * q_mf[0] + perseveration * pers_array
        exp_q1 = np.exp(beta * q_net)
        probs_1 = exp_q1 / np.sum(exp_q1)
        if a1 != -1:
            log_loss -= np.log(probs_1[a1] + eps)
        if s2 != -1 and a2 != -1:
            state_idx = s2 + 1
            exp_q2 = np.exp(beta_2 * q_mf[state_idx])
            probs_2 = exp_q2 / np.sum(exp_q2)
            log_loss -= np.log(probs_2[a2] + eps)
        if a1 != -1 and s2 != -1 and a2 != -1:
            state_idx = s2 + 1
            delta1 = q_mf[state_idx, a2] - q_mf[0, a1]
            q_mf[0, a1] += learning_rate * delta1
            delta2 = r - q_mf[state_idx, a2]
            q_mf[state_idx, a2] += learning_rate_2 * delta2
            q_mf[0, a1] += lambd * learning_rate * delta2
            pers_array.fill(0)
            pers_array[a1] = 1
    return log_loss
'''
