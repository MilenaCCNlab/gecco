# library_learning/compose/canonical_rlwm.py
"""Canonical RLWM baseline (Collins & Frank 2012 style): delta-rule RL mixed
with a one-shot, decaying, capacity-limited WM policy + uniform lapse.
Field-standard baseline, analog of hybrid.py (Daw) on two-step; refits are
cross-checked against the data's baseline_bic column (WARN-level in
evaluate_rlwm — the column's producing variant is unknown)."""

CANONICAL_BOUNDS = [(0.0, 1.0), (0.0, 10.0), (0.0, 1.0), (0.0, 1.0),
                    (1.0, 6.0), (0.0, 1.0)]

CANONICAL_SOURCE = '''def cognitive_model(stimulus, actions, rewards, blocks, set_sizes, model_parameters):
    """
    Canonical RLWM baseline (Collins & Frank 2012 style).

    Parameters:
    learning_rate: [0, 1] - RL delta-rule learning rate
    beta: [0, 10] - RL softmax inverse temperature
    wm_weight: [0, 1] - WM reliance at set sizes within capacity
    wm_decay: [0, 1] - per-trial WM decay toward uniform
    capacity: [1, 6] - WM capacity K; WM reliance scales by min(1, K/nS)
    lapse: [0, 1] - uniform-choice lapse probability
    """
    learning_rate, beta, wm_weight, wm_decay, capacity, lapse = model_parameters
    nA = 3
    log_loss = 0.0
    eps = 1e-10
    for b in np.unique(blocks):
        block_mask = blocks == b
        block_states = stimulus[block_mask]
        block_actions = actions[block_mask]
        block_rewards = rewards[block_mask]
        nS = int(set_sizes[block_mask][0])
        q = (1.0 / nA) * np.ones((nS, nA))
        w = (1.0 / nA) * np.ones((nS, nA))
        w_0 = (1.0 / nA) * np.ones((nS, nA))
        for trial in range(len(block_states)):
            s = int(block_states[trial])
            a = int(block_actions[trial])
            r = float(block_rewards[trial])
            if 0 <= s < nS and 0 <= a < nA:
                exp_rl = np.exp(beta * (q[s] - np.max(q[s])))
                probs_rl = exp_rl / np.sum(exp_rl)
                exp_wm = np.exp(50.0 * (w[s] - np.max(w[s])))
                probs_wm = exp_wm / np.sum(exp_wm)
                mix = wm_weight * min(1.0, capacity / float(nS))
                probs = mix * probs_wm + (1.0 - mix) * probs_rl
                probs = (1.0 - lapse) * probs + lapse / nA
                log_loss -= np.log(probs[a] + eps)
                delta = r - q[s, a]
                q[s, a] += learning_rate * delta
                w[s, a] = r
            w += wm_decay * (w_0 - w)
    return log_loss
'''
