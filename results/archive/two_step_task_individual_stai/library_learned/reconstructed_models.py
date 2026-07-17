import numpy as np
from .cognitive_library import *

# Participant 0


def cognitive_model3(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, spill, rare_bias, beta, biasA) = params
    (Q2, T) = init_q_and_T()
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai = float(stai_arr[0])
    rare_eff = (((0.5 * stai) + 0.5) * rare_bias)
    spill_eff = (((0.5 * stai) + 0.5) * spill)
    w_mb = np.clip((1.0 - stai), 0.0, 1.0)
    for t in range(num_trials):
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1 = integrate_mf_mb(q1, q1_mb, w_mb)
        logits = q1.copy()
        logits[0] += biasA
        logits = ((logits - logsumexp_max(logits)) * beta)
        probs1 = exp_logits(logits)
        probs1 /= np.sum(probs1)
        a1 = a1_seq[t]
        s = record_stage1(a1, p1_seq, probs1, s_seq, t)
        logits = Q2[s].copy()
        logits = ((logits - logsumexp_max(logits)) * beta)
        probs2 = exp_logits(logits)
        probs2 /= np.sum(probs2)
        a2 = a2_seq[t]
        is_common = (((a1 == 0) and (s == 0)) or ((a1 == 1) and (s == 1)))
        r = record_stage2(a2, p2_seq, probs2, r_seq, t)
        delta2 = update_q2(Q2, a2, alpha, r, s)
        other_s = (1 - s)
        Q2[(other_s, a2)] += ((alpha * spill_eff) * (r - Q2[(other_s, a2)]))
        delta1 = update_q1(Q2, a1, a2, alpha, q1, s)
        if (not is_common):
            a1_other = (1 - a1)
            q1[a1_other] += ((Q2[(s, a2)] - q1[a1_other]) * (alpha * rare_eff))
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return nll

# Participant 1


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, alpha_t, kappa_stick, phi_anx) = params
    (Q2, T) = init_q_and_T()
    (last_a1, num_trials, p1_seq, p2_seq) = init_with_last_action(a1_seq)
    stai = float(stai_arr[0])
    stickiness = np.zeros(2)
    for t in range(num_trials):
        bonus = np.zeros(2)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        if (t > 0):
            last_a1 = a1_seq[(t - 1)]
            s2_prev = s_seq[(t - 1)]
            p_obs = T[(last_a1, s2_prev)]
            surprise = (1.0 - p_obs)
            a1_now = a1_seq[t]
            bonus[a1_now] += ((phi_anx * stai) * surprise)
        if (last_a1 is not None):
            stickiness = np.zeros(2)
            stickiness[last_a1] = ((1.0 + stai) * kappa_stick)
        q1_combined = ((bonus + q1_mb) + stickiness)
        q1_combined -= np.max(q1_combined)
        a1 = a1_seq[t]
        exp_q1 = np.exp((beta * q1_combined))
        probs_1 = (exp_q1 / np.sum(exp_q1))
        p1_seq[t] = probs_1[a1]
        s2 = s_seq[t]
        q2_row = Q2[s2].copy()
        q2_row -= np.max(q2_row)
        a2 = a2_seq[t]
        exp_q2 = np.exp((beta * q2_row))
        probs_2 = (exp_q2 / np.sum(exp_q2))
        p2_seq[t] = probs_2[a2]
        r = r_seq[t]
        delta2 = (r - Q2[(s2, a2)])
        Q2[(s2, a2)] += td_step(alpha, delta2)
        a1_row = T[a1]
        for s_idx in (0, 1):
            target = (1.0 if (s_idx == s2) else 0.0)
            a1_row[s_idx] += ((target - a1_row[s_idx]) * alpha_t)
        T[a1] = (a1_row / np.sum(a1_row))
        last_a1 = a1
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return float(nll)

# Participant 10


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, alpha_T, w0, decay) = params
    Q2 = np.zeros((2, 2))
    T = np.full((2, 2), 0.5)
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai = float(stai_arr[0])
    decay_eff = np.clip((((0.5 * np.clip(stai, 0.0, 1.0)) + 0.5) * decay), 0.0, 1.0)
    for t in range(num_trials):
        Q2 = ((1.0 - decay_eff) * Q2)
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        c_actions = (2.0 * np.abs((T[:, 0] - 0.5)))
        certainty = np.mean(c_actions)
        max_q2 = max_q2_per_state(Q2)
        q1 = ((1.0 - decay_eff) * q1)
        q1_mb = (T @ max_q2)
        w_mb_eff = ((certainty ** np.clip((1.0 - stai), 0.0, 1.0)) * w0)
        w_mb_eff = np.clip(w_mb_eff, 0.0, 1.0)
        q1 = integrate_mf_mb(q1, q1_mb, w_mb_eff)
        q1_centered = center_values(q1)
        probs1 = np.exp((beta * q1_centered))
        probs1 = (probs1 / np.sum(probs1))
        p1_seq[t] = probs1[a1]
        s = int(s_seq[t])
        q2 = Q2[s]
        q2_centered = (q2 - np.max(q2))
        probs2 = np.exp((beta * q2_centered))
        probs2 = (probs2 / np.sum(probs2))
        r = record_stage2(a2, p2_seq, probs2, r_seq, t)
        for sp in range(2):
            target = (1.0 if (sp == s) else 0.0)
            T[(a1, sp)] += ((target - T[(a1, sp)]) * alpha_T)
        T[a1] = (T[a1] / np.sum(T[a1]))
        delta2 = update_q2(Q2, a2, alpha, r, s)
        v2 = Q2[(s, a2)]
        delta1 = (v2 - q1[a1])
        q1[a1] += td_step(alpha, delta1)
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return nll

# Participant 11


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, psi, gamma, pers) = params
    T = np.array([[0.7, 0.3], [0.3, 0.7]])
    epsilon = 1e-12
    H = (- np.sum((T * np.log((T + epsilon))), axis=1))
    Q2 = np.zeros((2, 2))
    last_a1 = 0
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai = float(stai_arr[0])
    pers_eff = ((1.0 + stai) * pers)
    w_mb_eff = np.clip(((1.0 - (0.6 * stai)) * psi), 0.0, 1.0)
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        bias1 = np.zeros(2)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1_combined = integrate_mf_mb(q1, q1_mb, w_mb_eff)
        r = float(r_seq[t])
        s2 = int(s_seq[t])
        bias1[last_a1] += pers_eff
        unc_penalty = ((gamma * stai) * H)
        logits = (((beta * q1_combined) + bias1) - unc_penalty)
        logits -= logsumexp_max(logits)
        p1 = exp_logits(logits)
        p1 /= np.sum(p1)
        logits = (Q2[s2] * beta)
        p1_seq[t] = p1[a1]
        logits -= logsumexp_max(logits)
        p2 = exp_logits(logits)
        p2 /= np.sum(p2)
        delta2 = (r - Q2[(s2, a2)])
        p2_seq[t] = p2[a2]
        Q2[(s2, a2)] += td_step(alpha, delta2)
        target1 = Q2[(s2, a2)]
        delta1 = (target1 - q1[a1])
        q1[a1] += td_step(alpha, delta1)
        last_a1 = a1
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return nll

# Participant 12


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha_base, beta, pi_base, stai_weight, lambda_) = params
    Q2 = np.zeros((2, 2))
    last_a1 = None
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai_arr = stai_arr[0]
    mod = (((2.0 * stai_arr) - 1.0) * ((2.0 * stai_weight) - 1.0))
    alpha_neg = np.clip(((1.0 - mod) * alpha_base), 0.0, 1.0)
    alpha_pos = np.clip(((1.0 + mod) * alpha_base), 0.0, 1.0)
    pers_eff = ((1.0 + mod) * pi_base)
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        q1 = q1.copy()
        s = s_seq[t]
        if (last_a1 is not None):
            pers = np.zeros(2)
            pers[last_a1] = 1.0
            q1 = ((pers * pers_eff) + q1)
        delta2 = (r_seq[t] - Q2[(s, a2)])
        q1_centered = center_values(q1)
        exp_q1 = np.exp((beta * q1_centered))
        probs_1 = (exp_q1 / np.sum(exp_q1))
        p1_seq[t] = probs_1[a1]
        q2_row = Q2[s].copy()
        q2_centered = center_values(q2_row)
        exp_q2 = np.exp((beta * q2_centered))
        probs_2 = (exp_q2 / np.sum(exp_q2))
        p2_seq[t] = probs_2[a2]
        if (delta2 >= 0):
            a2_lr = alpha_pos
        else:
            a2_lr = alpha_neg
        Q2[(s, a2)] += (a2_lr * delta2)
        q1[a1] += ((a2_lr * lambda_) * delta2)
        alpha1 = np.clip((((0.5 * mod) + 1.0) * alpha_base), 0.0, 1.0)
        delta1 = (Q2[(s, a2)] - q1[a1])
        q1[a1] += (alpha1 * delta1)
        last_a1 = a1
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return nll

# Participant 13


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, lambda_, w_mb, stickiness) = params
    (Q2, T) = init_q_and_T()
    epsilon = 1e-10
    last_a1 = (- 1)
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai_arr = float(stai_arr[0])
    w_mb = np.clip(w_mb, epsilon, (1 - epsilon))
    logit = (np.log(w_mb) - np.log((1 - w_mb)))
    w_mb_eff = (1 / (1 + np.exp((- (((0.5 - stai_arr) * 2.0) + logit)))))
    w_mb_eff = float(np.clip(w_mb_eff, 0.0, 1.0))
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1 = (((1 - w_mb_eff) * q1) + (q1_mb * w_mb_eff))
        r = float(r_seq[t])
        s = int(s_seq[t])
        if (last_a1 >= 0):
            pers_eff = ((0.5 + stai_arr) * stickiness)
            bias = np.array([0.0, 0.0])
            bias[last_a1] += pers_eff
            q1 = (bias + q1)
        delta2 = (r - Q2[(s, a2)])
        exp_q1 = np.exp((center_values(q1) * beta))
        probs_1 = (exp_q1 / (epsilon + np.sum(exp_q1)))
        p1_seq[t] = probs_1[a1]
        q2_row = Q2[s]
        exp_q2 = np.exp((center_values(q2_row) * beta))
        probs_2 = (exp_q2 / (epsilon + np.sum(exp_q2)))
        p2_seq[t] = probs_2[a2]
        Q2[(s, a2)] += td_step(alpha, delta2)
        td_to_stage1 = (Q2[(s, a2)] - q1[a1])
        q1[a1] += (((alpha * lambda_) * delta2) + (alpha * td_to_stage1))
        last_a1 = a1
    epsilon = 1e-10
    nll = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return float(nll)

# Participant 14


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, alpha_t, beta, pers, gamma_anx) = params
    Q2 = np.zeros((2, 2))
    T = (0.5 * np.ones((2, 2)))
    (last_a1, num_trials, p1_seq, p2_seq) = init_with_last_action(a1_seq)
    stai_arr = stai_arr[0]
    stick_eff = (np.clip((((stai_arr - 0.31) * gamma_anx) + 1.0), 0.0, 2.0) * pers)
    for t in range(num_trials):
        bias = np.zeros(2)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        if (last_a1 is not None):
            bias[last_a1] = stick_eff
        logits = ((beta * q1_mb) + bias)
        logits -= logsumexp_max(logits)
        a1 = a1_seq[t]
        exp1 = exp_logits(logits)
        probs1 = (exp1 / (1e-12 + np.sum(exp1)))
        s = record_stage1(a1, p1_seq, probs1, s_seq, t)
        logits = (Q2[s] * beta)
        logits -= logsumexp_max(logits)
        a2 = a2_seq[t]
        exp2 = exp_logits(logits)
        probs2 = (exp2 / (1e-12 + np.sum(exp2)))
        r = record_stage2(a2, p2_seq, probs2, r_seq, t)
        delta2 = update_q2(Q2, a2, alpha, r, s)
        target = np.array([0.0, 0.0])
        target[s] = 1.0
        T[a1] += ((target - T[a1]) * alpha_t)
        T[a1] = np.clip(T[a1], 1e-06, 1.0)
        T[a1] /= np.sum(T[a1])
        last_a1 = a1
    epsilon = 1e-10
    nll = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return nll

# Participant 15


def cognitive_model3(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, alpha_t, decay, omega_safe) = params
    Q2 = (0.5 * np.ones((2, 2)))
    T = np.array([[0.6, 0.4], [0.4, 0.6]])
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai_arr = float(stai_arr[0])
    safety_scale = (omega_safe * stai_arr)
    w_mb = (((1.0 - stai_arr) * 0.5) + 0.5)
    for t in range(num_trials):
        ent = np.zeros(2)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        s = s_seq[t]
        for a in range(2):
            p = np.clip(T[(a, 0)], 1e-08, (1.0 - 1e-08))
            q = (1.0 - p)
            H = (- ((np.log(p) * p) + (np.log(q) * q)))
            ent[a] = (H / np.log(2.0))
        bias = ((- safety_scale) * ent)
        q1 = (integrate_mf_mb(q1, q1_mb, w_mb) + bias)
        z1 = (center_values(q1) * beta)
        probs1 = np.exp(z1)
        probs1 /= np.sum(probs1)
        a1 = a1_seq[t]
        p1_seq[t] = probs1[a1]
        z2 = ((Q2[s] - np.max(Q2[s])) * beta)
        probs2 = np.exp(z2)
        probs2 /= np.sum(probs2)
        a2 = a2_seq[t]
        r = record_stage2(a2, p2_seq, probs2, r_seq, t)
        delta2 = update_q2(Q2, a2, alpha, r, s)
        (delta1, target1) = update_q1_full(Q2, a1, a2, alpha, q1, s)
        T = ((memory_retention(decay) * T) + (0.5 * decay))
        T[(a1, s)] += ((1.0 - T[(a1, s)]) * alpha_t)
        other = (1 - s)
        T[(a1, other)] += ((0.0 - T[(a1, other)]) * alpha_t)
        for a in range(2):
            row = T[a]
            row = np.clip(row, 1e-08, 1.0)
            T[a] = (row / np.sum(row))
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return nll

# Participant 16


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, beta, pi_base, chi_base) = params
    (Q2, T) = init_q_and_T()
    num_trials = len(a1_seq)
    q1 = np.zeros(2)
    stai_arr = float(stai_arr[0])
    chi = np.clip((stai_modulate(stai_arr) * chi_base), 0.0, 5.0)
    lambda_ = np.clip((stai_scale(stai_arr) + 0.4), 0.0, 1.0)
    last_a1 = None
    (p1_seq, p2_seq) = init_prob_arrays(num_trials)
    pers = np.clip((((0.8 * stai_arr) + 1.0) * pi_base), 0.0, 5.0)
    prev_sigma = 0.0
    w_mb = np.clip((0.7 - (0.4 * stai_arr)), 0.0, 1.0)
    for t in range(num_trials):
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1 = integrate_mf_mb(q1, q1_mb, w_mb)
        logits = (beta * q1)
        if (last_a1 is not None):
            logits[last_a1] += pers
            logits[last_a1] += (chi * prev_sigma)
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        maxl1 = logsumexp_max(logits)
        probs1 = np.exp((logits - maxl1))
        probs1 = (probs1 / np.sum(probs1))
        s = record_stage1(a1, p1_seq, probs1, s_seq, t)
        logits = (Q2[s] * beta)
        maxl2 = logsumexp_max(logits)
        probs2 = np.exp((logits - maxl2))
        probs2 = (probs2 / np.sum(probs2))
        r = record_stage2(a2, p2_seq, probs2, r_seq, t)
        delta2 = update_q2(Q2, a2, alpha, r, s)
        q1[a1] += ((alpha * lambda_) * delta2)
        common = int((((a1 == 0) and (s == 0)) or ((a1 == 1) and (s == 1))))
        last_a1 = a1
        sigma = (1 if ((common and (r > 0.5)) or ((1 - common) and (r <= 0.5))) else (- 1))
        prev_sigma = float(sigma)
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return float(nll)

# Participant 17


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, w_slope, decay, tau_stay) = params
    (Q2, T) = init_q_and_T()
    (num_trials, p1_seq, p2_seq) = init_trial_arrays_basic(a1_seq)
    prev_a2 = None
    prev_s = None
    q1 = np.zeros(2)
    stai_arr = float(stai_arr[0])
    decay = (1.0 - (decay * stai_arr))
    w_mb = (((1.0 - stai_arr) * w_slope) + ((1.0 - w_slope) * stai_arr))
    w_mb = (0.0 if (w_mb < 0.0) else (1.0 if (w_mb > 1.0) else w_mb))
    if (decay < 0.0):
        decay = 0.0
    if (decay > 1.0):
        decay = 1.0
    for t in range(num_trials):
        a1 = a1_seq[t]
        bias2 = np.zeros(2)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1 = integrate_mf_mb(q1, q1_mb, w_mb)
        z1 = center_values(q1)
        exp1 = np.exp((beta * z1))
        probs1 = (exp1 / np.sum(exp1))
        s = record_stage1(a1, p1_seq, probs1, s_seq, t)
        q2_row = Q2[s].copy()
        if ((prev_a2 is not None) and (prev_s == s)):
            bias2[prev_a2] += (stai_arr * tau_stay)
        a2 = a2_seq[t]
        z2 = ((bias2 + q2_row) - np.max((bias2 + q2_row)))
        exp2 = np.exp((beta * z2))
        probs2 = (exp2 / np.sum(exp2))
        r = record_stage2(a2, p2_seq, probs2, r_seq, t)
        q1 *= decay
        Q2 *= decay
        delta2 = update_q2(Q2, a2, alpha, r, s)
        delta1 = update_q1(Q2, a1, a2, alpha, q1, s)
        prev_a2 = a2
        prev_s = s
    epsilon = 1e-10
    nll = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return nll

# Participant 18


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, k_temp, decay, pers) = params
    (Q2, T) = init_q_and_T()
    num_trials = len(a1_seq)
    q1 = np.zeros(2)
    stai = float(stai_arr[0])
    beta = max(0.001, ((1.0 - (k_temp * stai)) * beta))
    decay = np.clip(decay, 0.0, 1.0)
    epsilon = 1e-10
    (p1_seq, p2_seq) = init_prob_arrays(num_trials)
    stick_strength = ((1.0 - stai) * pers)
    trace1 = np.zeros(2)
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        pref1 = (((0.5 * q1) + (0.5 * q1_mb)) + (stick_strength * trace1))
        centered1 = (pref1 - np.max(pref1))
        exp1 = np.exp((beta * centered1))
        probs1 = (exp1 / (epsilon + np.sum(exp1)))
        p1_seq[t] = probs1[a1]
        r = float(r_seq[t])
        s = int(s_seq[t])
        q2_row = Q2[s]
        centered2 = center_values(q2_row)
        exp2 = np.exp((beta * centered2))
        probs2 = (exp2 / (epsilon + np.sum(exp2)))
        p2_seq[t] = probs2[a2]
        Q2 *= memory_retention(decay)
        delta2 = update_q2(Q2, a2, alpha, r, s)
        delta1 = update_q1(Q2, a1, a2, alpha, q1, s)
        trace1 *= memory_retention(decay)
        trace1[a1] += 1.0
    nll = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return nll

# Participant 19


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, lambda_loss, kappa_anx, phi_forget) = params
    (Q2, T) = init_q_and_T()
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai_arr = float(stai_arr[0])
    decay = np.clip(((0.5 + stai_arr) * phi_forget), 0.0, 1.0)
    epsilon = 1e-12
    lambda_eff = ((1.0 + stai_arr) * lambda_loss)
    xi = np.clip((1.0 - (kappa_anx * stai_arr)), 0.0, 1.0)
    for t in range(num_trials):
        max_q2 = max_q2_per_state(Q2)
        vmin = np.min(Q2, axis=1)
        V_state = (((1.0 - xi) * vmin) + (max_q2 * xi))
        Q1_mb_pess = (T @ V_state)
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        w_mb_eff = np.clip(((stai_scale(stai_arr) * (1.0 - lambda_loss)) + 0.5), 0.0, 1.0)
        q1 = (((1.0 - w_mb_eff) * q1) + (Q1_mb_pess * w_mb_eff))
        q_centered = center_values(q1)
        probs_1 = softmax_exp(beta, q_centered)
        probs_1 = (probs_1 / (epsilon + np.sum(probs_1)))
        p1_seq[t] = probs_1[a1]
        s = int(s_seq[t])
        q_centered = (Q2[s] - np.max(Q2[s]))
        probs_2 = softmax_exp(beta, q_centered)
        probs_2 = (probs_2 / (epsilon + np.sum(probs_2)))
        p2_seq[t] = probs_2[a2]
        r = float(r_seq[t])
        if (r >= 0):
            u = r
        else:
            u = ((- (1.0 + lambda_eff)) * (- r))
        delta2 = (u - Q2[(s, a2)])
        Q2[(s, a2)] += td_step(alpha, delta2)
        other_a2 = (1 - a2)
        Q2[(s, other_a2)] *= memory_retention(decay)
        delta1 = update_q1(Q2, a1, a2, alpha, q1, s)
        other_a1 = (1 - a1)
        q1[other_a1] *= memory_retention(decay)
    nll = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return nll

# Participant 2


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, beta, lambda_, w0) = params
    (Q2, T) = init_q_and_T()
    epsilon = 1e-10
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai = float(np.asarray(stai_arr)[0])
    w0 = min(max(w0, 0.0), 1.0)
    w0_logit = np.log(((epsilon + w0) / (1.0 - (epsilon + w0))))
    w_logit = (((0.5 - stai) * 3.0) + w0_logit)
    w_mb = (1.0 / (1.0 + np.exp((- w_logit))))
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1 = integrate_mf_mb(q1, q1_mb, w_mb)
        q1_shift = center_values(q1)
        exp_q1 = np.exp((beta * q1_shift))
        probs1 = (exp_q1 / np.sum(exp_q1))
        p1_seq[t] = probs1[a1]
        r = float(r_seq[t])
        s = int(s_seq[t])
        delta2 = (r - Q2[(s, a2)])
        q2_row = Q2[s]
        q2_shift = center_values(q2_row)
        exp_q2 = np.exp((beta * q2_shift))
        probs2 = (exp_q2 / np.sum(exp_q2))
        p2_seq[t] = probs2[a2]
        Q2[(s, a2)] += td_step(alpha, delta2)
        delta1 = update_q1(Q2, a1, a2, alpha, q1, s)
        q1[a1] += ((alpha * lambda_) * delta2)
    neg_log_likelihood = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return float(neg_log_likelihood)

# Participant 20


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, w0, lambda_, pers) = params
    (Q2, T) = init_q_and_T()
    (num_trials, p1_seq, p2_seq) = init_trial_arrays_basic(a1_seq)
    prev_choice1 = None
    q1 = np.zeros(2)
    stai_arr = float(stai_arr[0])
    for t in range(num_trials):
        max_q2 = max_q2_per_state(Q2)
        pers = ((0.5 + stai_arr) * pers)
        q1_mb = (T @ max_q2)
        w_mb_eff = np.clip(((1.0 - (0.6 * stai_arr)) * w0), 0.0, 1.0)
        q1_combined = integrate_mf_mb(q1, q1_mb, w_mb_eff)
        pref1 = (beta * q1_combined)
        if (prev_choice1 is not None):
            pers = np.zeros(2)
            pers[prev_choice1] = 1.0
            pref1 = ((pers * pers) + pref1)
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        exp_q1 = np.exp((pref1 - np.max(pref1)))
        probs_1 = (exp_q1 / np.sum(exp_q1))
        p1_seq[t] = probs_1[a1]
        s = s_seq[t]
        delta1 = (Q2[(s, a2)] - q1[a1])
        delta2 = (r_seq[t] - Q2[(s, a2)])
        q2_row = (Q2[s] * beta)
        exp_q2 = np.exp(center_values(q2_row))
        probs_2 = (exp_q2 / np.sum(exp_q2))
        p2_seq[t] = probs_2[a2]
        Q2[(s, a2)] += td_step(alpha, delta2)
        q1[a1] += (((delta2 * lambda_) + delta1) * alpha)
        prev_choice1 = a1
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return nll

# Participant 21


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, omega0, anx_mod, pers) = params
    (Q2, T) = init_q_and_T()
    (last_a1, num_trials, p1_seq, p2_seq) = init_with_last_action(a1_seq)
    prev_a2_by_state = [None, None]
    q1 = np.zeros(2)
    stai = float(stai_arr[0])
    stick_strength = ((1.0 - stai) * pers)
    w_mb_eff = np.clip((((0.5 - stai) * anx_mod) + omega0), 0.0, 1.0)
    for t in range(num_trials):
        bias1 = np.zeros(2)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        if (last_a1 is not None):
            bias1[last_a1] += stick_strength
        a1 = a1_seq[t]
        bias2 = np.zeros(2)
        q1_combined = (integrate_mf_mb(q1, q1_mb, w_mb_eff) + bias1)
        q_centered = (q1_combined - np.max(q1_combined))
        probs_1 = softmax_exp(beta, q_centered)
        probs_1 = (probs_1 / np.sum(probs_1))
        p1_seq[t] = probs_1[a1]
        s = s_seq[t]
        if (prev_a2_by_state[s] is not None):
            bias2[prev_a2_by_state[s]] += stick_strength
        a2 = a2_seq[t]
        q2_net = (Q2[s] + bias2)
        q_centered = (q2_net - np.max(q2_net))
        probs_2 = softmax_exp(beta, q_centered)
        probs_2 = (probs_2 / np.sum(probs_2))
        p2_seq[t] = probs_2[a2]
        r = r_seq[t]
        delta2 = update_q2(Q2, a2, alpha, r, s)
        delta1 = update_q1(Q2, a1, a2, alpha, q1, s)
        last_a1 = a1
        prev_a2_by_state[s] = a2
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return nll

# Participant 22


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, w0, k_unc, k_decay) = params
    (Q2, T) = init_q_and_T()
    num_trials = len(a1_seq)
    p1 = np.zeros(num_trials)
    p2 = np.zeros(num_trials)
    q1 = np.zeros(2)
    stai_arr = float(stai_arr[0])
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        diff_x = abs((Q2[(0, 0)] - Q2[(0, 1)]))
        diff_y = abs((Q2[(1, 0)] - Q2[(1, 1)]))
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        r = r_seq[t]
        s = int(s_seq[t])
        unc = (1.0 - ((np.tanh(diff_x) + np.tanh(diff_y)) * 0.5))
        w_mb = (((1.0 - (0.6 * stai_arr)) * w0) + (k_unc * unc))
        w_mb = min(1.0, max(0.0, w_mb))
        q1_combined = integrate_mf_mb(q1, q1_mb, w_mb)
        logits = (beta * q1_combined)
        logits -= logsumexp_max(logits)
        probs1 = exp_logits(logits)
        probs1 /= (1e-16 + np.sum(probs1))
        logits = (Q2[s] * beta)
        p1[t] = probs1[a1]
        logits -= logsumexp_max(logits)
        probs2 = exp_logits(logits)
        probs2 /= (1e-16 + np.sum(probs2))
        delta2 = (r - Q2[(s, a2)])
        p2[t] = probs2[a2]
        Q2[(s, a2)] += td_step(alpha, delta2)
        (delta1, target1) = update_q1_full(Q2, a1, a2, alpha, q1, s)
        other_a1 = (1 - a1)
        q1[other_a1] *= (1.0 - k_decay)
        other_a2 = (1 - a2)
        Q2[(s, other_a2)] *= (1.0 - k_decay)
        other_s = (1 - s)
        Q2[other_s] *= (1.0 - (0.5 * k_decay))
    epsilon = 1e-12
    nll = (- (np.sum(np.log((epsilon + p1))) + np.sum(np.log((epsilon + p2)))))
    return nll

# Participant 23


def cognitive_model3(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, zeta0, phi0, decay_base) = params
    Q2 = np.zeros((2, 2))
    Q2 = np.ones((2, 2))
    T = np.ones((2, 2))
    num_trials = len(a1_seq)
    stai_arr = float(stai_arr[0])
    decay = max(0.0, min(1.0, (stai_modulate(stai_arr) * decay_base)))
    (p1_seq, p2_seq) = init_prob_arrays(num_trials)
    q1 = np.zeros(2)
    w_mb = max(0.0, min(1.0, ((1.0 - stai_scale(stai_arr)) * phi0)))
    zeta = max(0.0, min(1.0, ((1.0 - stai_arr) * zeta0)))
    for t in range(num_trials):
        q1 *= memory_retention(decay)
        Q2 *= memory_retention(decay)
        T = (T / np.sum(T, axis=1, keepdims=True))
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1_combined = integrate_mf_mb(q1, q1_mb, w_mb)
        logits = (beta * q1_combined)
        logits -= logsumexp_max(logits)
        probs1 = exp_logits(logits)
        probs1 /= np.sum(probs1)
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        p1_seq[t] = probs1[a1]
        r = r_seq[t]
        s = int(s_seq[t])
        bonus = (zeta / np.sqrt((1e-08 + Q2[s])))
        logits = ((Q2[s] + bonus) * beta)
        logits -= logsumexp_max(logits)
        probs2 = exp_logits(logits)
        probs2 /= np.sum(probs2)
        p2_seq[t] = probs2[a2]
        Q2[(s, a2)] += 1.0
        delta2 = update_q2(Q2, a2, alpha, r, s)
        backup = Q2[(s, a2)]
        delta1 = (backup - q1[a1])
        q1[a1] += td_step(alpha, delta1)
        T[(a1, s)] += 1.0
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return float(nll)

# Participant 24


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha_pos, alpha_neg, beta, pers, kappa_stai) = params
    last_a1 = None
    last_a2_by_state = {0: None, 1: None}
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    q2 = np.zeros((2, 2))
    s = float(stai_arr[0])
    pers = ((kappa_stai * s) + pers)
    for t in range(num_trials):
        bias1 = np.zeros(2)
        if (last_a1 is not None):
            bias1[last_a1] += pers
        bias2 = np.zeros(2)
        stai = s_seq[t]
        if (last_a2_by_state[stai] is not None):
            bias2[last_a2_by_state[stai]] += pers
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        alpha_pos = (((1 - s) * alpha_pos) + (alpha_neg * s))
        alpha_neg = (((1 - s) * alpha_neg) + (alpha_pos * s))
        prefs1 = (bias1 + q1)
        exp1 = np.exp(((prefs1 - np.max(prefs1)) * beta))
        probs1 = (exp1 / np.sum(exp1))
        p1_seq[t] = probs1[a1]
        prefs2 = (bias2 + q2[stai])
        exp2 = np.exp(((prefs2 - np.max(prefs2)) * beta))
        probs2 = (exp2 / np.sum(exp2))
        r = record_stage2(a2, p2_seq, probs2, r_seq, t)
        delta2 = (r - q2[(stai, a2)])
        a2_lr = (alpha_pos if (delta2 >= 0) else alpha_neg)
        q2[(stai, a2)] += (a2_lr * delta2)
        delta1 = (q2[(stai, a2)] - q1[a1])
        a1_lr = (alpha_pos if (delta1 >= 0) else alpha_neg)
        q1[a1] += (a1_lr * delta1)
        last_a1 = a1
        last_a2_by_state[stai] = a2
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return nll

# Participant 25


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, w_mb, lambda_, pers) = params
    (Q2, T) = init_q_and_T()
    epsilon = 1e-10
    last_a1 = (- 1)
    (num_trials, p1_seq, p2_seq) = init_trial_arrays_basic(a1_seq)
    prev_a2_state0 = (- 1)
    prev_a2_state1 = (- 1)
    q1 = np.zeros(2)
    stai = (float(stai_arr[0]) if hasattr(stai_arr, '__len__') else float(stai_arr))
    pers_eff = ((1.0 + stai) * pers)
    w_mb_eff = ((1.0 - stai) * w_mb)
    for t in range(num_trials):
        bias1 = np.zeros(2)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1_combined = integrate_mf_mb(q1, q1_mb, w_mb_eff)
        s = s_seq[t]
        if (last_a1 in (0, 1)):
            bias1[last_a1] += pers_eff
        logits = ((beta * q1_combined) + bias1)
        logits -= logsumexp_max(logits)
        probs1 = exp_logits(logits)
        probs1 /= np.sum(probs1)
        a1 = a1_seq[t]
        bias2 = np.zeros(2)
        p1_seq[t] = max(probs1[a1], epsilon)
        prev_a2 = (prev_a2_state0 if (s == 0) else prev_a2_state1)
        if (prev_a2 in (0, 1)):
            bias2[prev_a2] += pers_eff
        logits = ((Q2[s] * beta) + bias2)
        logits -= logsumexp_max(logits)
        probs2 = exp_logits(logits)
        probs2 /= np.sum(probs2)
        a2 = a2_seq[t]
        p2_seq[t] = max(probs2[a2], epsilon)
        q2_old = Q2[(s, a2)]
        r = r_seq[t]
        delta2 = (r - q2_old)
        Q2[(s, a2)] += td_step(alpha, delta2)
        delta1 = (q2_old - q1[a1])
        q1[a1] += (((delta2 * lambda_) + delta1) * alpha)
        last_a1 = a1
        if (s == 0):
            prev_a2_state0 = a2
        else:
            prev_a2_state1 = a2
    neg_ll = (- (np.sum(np.log(p1_seq)) + np.sum(np.log(p2_seq))))
    return float(neg_ll)

# Participant 26


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, pers, w_mb, a_neg) = params
    (Q2, T) = init_q_and_T()
    epsilon = 1e-12
    last_a1 = None
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai = float(stai_arr[0])
    alpha_neg = np.clip((((a_neg * stai) + 1.0) * alpha), 0.0, 1.0)
    alpha_pos = np.clip(((1.0 - (a_neg * stai)) * alpha), 0.0, 1.0)
    pers = (((0.75 * stai) + 0.25) * pers)
    w_mf = stai
    w_mb = (1.0 - w_mf)
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        max_q2 = max_q2_per_state(Q2)
        persev_bias = np.zeros(2)
        q1_mb = (T @ max_q2)
        r = r_seq[t]
        s = int(s_seq[t])
        if (last_a1 is not None):
            persev_bias[last_a1] = (((0.5 * stai) + 0.5) * w_mb)
        q1_combined = (((q1 * w_mf) + (q1_mb * w_mb)) + persev_bias)
        exp_q1 = np.exp(((q1_combined - np.max(q1_combined)) * beta))
        probs_1 = (exp_q1 / (epsilon + np.sum(exp_q1)))
        p1_seq[t] = probs_1[a1]
        q2_row = Q2[s]
        exp_q2 = np.exp((center_values(q2_row) * beta))
        probs_2 = (exp_q2 / (epsilon + np.sum(exp_q2)))
        p2_seq[t] = probs_2[a2]
        target = np.array([(1.0 if (i == s) else 0.0) for i in range(2)])
        T[a1] = (((1.0 - pers) * T[a1]) + (pers * target))
        T[a1] = (T[a1] / (epsilon + np.sum(T[a1])))
        delta2 = (r - Q2[(s, a2)])
        lr2 = (alpha_pos if (delta2 >= 0.0) else alpha_neg)
        Q2[(s, a2)] += (delta2 * lr2)
        delta1 = (Q2[(s, a2)] - q1[a1])
        q1[a1] += (delta1 * lr2)
        last_a1 = a1
    log_loss = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return log_loss

# Participant 27


def cognitive_model3(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, w_mb, bias_safe, decay) = params
    Q2 = (0.5 * np.ones((2, 2)))
    T = np.array([[0.7, 0.3], [0.3, 0.7]])
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai_arr = float(stai_arr[0])
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        beta = ((1.0 - stai_scale(stai_arr)) * beta)
        r = r_seq[t]
        s = int(s_seq[t])
        is_common = (((a1 == 0) and (s == 0)) or ((a1 == 1) and (s == 1)))
        max_q2 = max_q2_per_state(Q2)
        mb_w_eff = (w_mb if is_common else ((1.0 - (0.7 * stai_arr)) * w_mb))
        mb_w_eff = np.clip(mb_w_eff, 0.0, 1.0)
        q1_mb = (T @ max_q2)
        q1 = (((1.0 - mb_w_eff) * q1) + (mb_w_eff * q1_mb))
        logits = (beta * q1)
        logits -= logsumexp_max(logits)
        probs1 = exp_logits(logits)
        probs1 /= np.sum(probs1)
        p1_seq[t] = probs1[a1]
        uncertainty = (1.0 - abs((Q2[(s, 0)] - Q2[(s, 1)])))
        bias_vec = np.array([((bias_safe * stai_arr) * uncertainty), 0.0])
        logits = ((Q2[s] * beta) + bias_vec)
        logits -= logsumexp_max(logits)
        probs2 = exp_logits(logits)
        probs2 /= np.sum(probs2)
        Q2 = ((memory_retention(decay) * Q2) + (0.5 * decay))
        delta2 = (r - Q2[(s, a2)])
        p2_seq[t] = probs2[a2]
        q1 = ((memory_retention(decay) * q1) + (0.0 * decay))
        Q2[(s, a2)] += td_step(alpha, delta2)
        delta1 = (Q2[(s, a2)] - q1[a1])
        q1[a1] += (((0.5 * delta1) + (0.5 * delta2)) * alpha)
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return nll

# Participant 28


def cognitive_model3(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, c_asym, decay, curv) = params
    num_trials = len(a1_seq)
    stai = float(stai_arr[0])
    beta = ((1.0 - ((stai - 0.5) * 0.5)) * beta)
    if (beta < 1e-06):
        beta = 1e-06
    Q2 = np.zeros((2, 2))
    alpha_neg = ((1.0 - (c_asym * stai)) * alpha)
    alpha_pos = (((c_asym * stai) + 1.0) * alpha)
    epsilon = 1e-10
    (p1_seq, p2_seq) = init_prob_arrays(num_trials)
    q1 = np.zeros(2)
    if (alpha_pos > 1.0):
        alpha_pos = 1.0
    if (alpha_neg < 0.0):
        alpha_neg = 0.0
    for t in range(num_trials):
        Q2 = ((memory_retention(decay) * Q2) + (0.5 * decay))
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        q1 = ((memory_retention(decay) * q1) + (0.5 * decay))
        q_centered = center_values(q1)
        ps1 = softmax_exp(beta, q_centered)
        ps1 = (ps1 / (epsilon + np.sum(ps1)))
        p1_seq[t] = ps1[a1]
        s = int(s_seq[t])
        q_centered = (Q2[s] - np.max(Q2[s]))
        ps2 = softmax_exp(beta, q_centered)
        ps2 = (ps2 / (epsilon + np.sum(ps2)))
        p2_seq[t] = ps2[a2]
        r = r_seq[t]
        u = (r ** curv)
        delta2 = (u - Q2[(s, a2)])
        a2_lr = (alpha_pos if (delta2 >= 0.0) else alpha_neg)
        Q2[(s, a2)] += (a2_lr * delta2)
        delta1 = (Q2[(s, a2)] - q1[a1])
        a1_lr = (alpha_pos if (delta1 >= 0.0) else alpha_neg)
        q1[a1] += (a1_lr * delta1)
    neg_log_lik = (- (np.sum(np.log((1e-10 + p1_seq))) + np.sum(np.log((1e-10 + p2_seq)))))
    return neg_log_lik

# Participant 29


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, kernel_lr, risk_penalty, anx_risk_gain) = params
    Q2 = np.zeros((2, 2))
    Q2 = np.zeros((2, 2))
    epsilon = 1e-12
    k1 = np.zeros(2)
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai = float(stai_arr[0])
    kernel_strength = ((((0.5 * anx_risk_gain) * stai) + 1.0) * kernel_lr)
    lambda_ = (((anx_risk_gain * stai) + 1.0) * risk_penalty)
    for t in range(num_trials):
        logits = ((((2.0 * k1) - 1.0) * kernel_strength) + q1)
        logits = ((logits - logsumexp_max(logits)) * beta)
        probs1 = exp_logits(logits)
        var2 = ((1.0 - Q2) * Q2)
        unc2 = np.sqrt(np.clip(var2, 0.0, 0.25))
        probs1 /= (epsilon + np.sum(probs1))
        a1 = a1_seq[t]
        s = record_stage1(a1, p1_seq, probs1, s_seq, t)
        logits = ((((2.0 * Q2[s]) - 1.0) * kernel_strength) + (Q2[s] - (lambda_ * unc2[s])))
        logits = ((logits - logsumexp_max(logits)) * beta)
        probs2 = exp_logits(logits)
        probs2 /= (epsilon + np.sum(probs2))
        a2 = a2_seq[t]
        r = record_stage2(a2, p2_seq, probs2, r_seq, t)
        var_chosen = ((1.0 - Q2[(s, a2)]) * Q2[(s, a2)])
        unc_chosen = np.sqrt(max(0.0, min(0.25, var_chosen)))
        r_subj = (r - (lambda_ * unc_chosen))
        delta2 = (r_subj - Q2[(s, a2)])
        Q2[(s, a2)] += td_step(alpha, delta2)
        (delta1, target1) = update_q1_full(Q2, a1, a2, alpha, q1, s)
        k1 = ((1.0 - kernel_lr) * k1)
        k1[a1] += kernel_lr
        Q2[s] = ((1.0 - kernel_lr) * Q2[s])
        Q2[(s, a2)] += kernel_lr
    neg_ll = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return float(neg_ll)

# Participant 3


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, k1, omega0, xi_unc) = params
    Q2 = np.zeros((2, 2))
    T_counts = np.ones((2, 2))
    epsilon = 1e-12
    last_a1 = (- 1)
    (num_trials, p1_seq, p2_seq) = init_trial_arrays_basic(a1_seq)
    prev_a2 = (- 1)
    q1 = np.zeros(2)
    stai = float(stai_arr[0])
    for t in range(num_trials):
        T = (T_counts / (epsilon + np.sum(T_counts, axis=1, keepdims=True)))
        ent = ((- np.sum((T * np.log((T + epsilon))), axis=1)) / np.log((2 + epsilon)))
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        unc = ((ent[0] + ent[1]) * 0.5)
        w_mb_eff = ((((0.5 * stai) + 0.5) * (unc * xi_unc)) + omega0)
        w_mb_eff = float(np.clip(w_mb_eff, 0.0, 1.0))
        q1 = integrate_mf_mb(q1, q1_mb, w_mb_eff)
        stick1 = np.zeros(2)
        if (last_a1 in (0, 1)):
            stick1[last_a1] = 1.0
        logits = ((beta * q1) + (k1 * stick1))
        logits -= logsumexp_max(logits)
        a1 = int(a1_seq[t])
        probs1 = (exp_logits(logits) / (epsilon + np.sum(exp_logits(logits))))
        p1_seq[t] = probs1[a1]
        s2 = int(s_seq[t])
        stick2 = np.zeros(2)
        if (prev_a2 in (0, 1)):
            stick2[prev_a2] = 1.0
        k2_eff = ((1.0 - (0.7 * stai)) * k1)
        logits = ((Q2[s2] * beta) + (k2_eff * stick2))
        logits -= logsumexp_max(logits)
        a2 = int(a2_seq[t])
        probs2 = (exp_logits(logits) / (epsilon + np.sum(exp_logits(logits))))
        p2_seq[t] = probs2[a2]
        r = float(r_seq[t])
        T_counts[(a1, s2)] += 1.0
        delta2 = (r - Q2[(s2, a2)])
        Q2[(s2, a2)] += td_step(alpha, delta2)
        target1 = Q2[(s2, a2)]
        delta1 = (target1 - q1[a1])
        q1[a1] += td_step(alpha, delta1)
        q1[a1] += td_step(alpha, delta2)
        last_a1 = a1
        prev_a2 = a2
    nll = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return float(nll)

# Participant 30


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, w_mb, lambda_, pers) = params
    (Q2, T) = init_q_and_T()
    (last_a1, num_trials, p1_seq, p2_seq) = init_with_last_action(a1_seq)
    prev_a2_by_state = [None, None]
    q1 = np.zeros(2)
    stai_arr = float(stai_arr[0])
    w_mb_eff = np.clip(((1.0 - (0.8 * stai_arr)) * w_mb), 0.0, 1.0)
    for t in range(num_trials):
        bias1 = np.zeros(2)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1 = integrate_mf_mb(q1, q1_mb, w_mb_eff)
        s = int(s_seq[t])
        if (last_a1 is not None):
            bias1[last_a1] += pers
        logits = ((beta * q1) + bias1)
        logits -= logsumexp_max(logits)
        a1 = int(a1_seq[t])
        bias2 = np.zeros(2)
        probs_1 = normalize_probs(logits)
        p1_seq[t] = probs_1[a1]
        q2 = Q2[s].copy()
        if (prev_a2_by_state[s] is not None):
            bias2[prev_a2_by_state[s]] += pers
        logits = ((beta * q2) + bias2)
        logits -= logsumexp_max(logits)
        a2 = int(a2_seq[t])
        delta1 = (Q2[(s, a2)] - q1[a1])
        probs_2 = normalize_probs(logits)
        p2_seq[t] = probs_2[a2]
        r = float(r_seq[t])
        delta2 = update_q2(Q2, a2, alpha, r, s)
        q1[a1] += (((delta2 * lambda_) + delta1) * alpha)
        last_a1 = a1
        prev_a2_by_state[s] = a2
    epsilon = 1e-10
    nll = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return nll

# Participant 31


def cognitive_model3(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, decay, k_anx_decay, tr_bias) = params
    num_trials = len(a1_seq)
    stai = float(stai_arr[0])
    decay_eff = ((k_anx_decay * stai) + decay)
    if (decay_eff < 0.0):
        decay_eff = 0.0
    if (decay_eff > 1.0):
        decay_eff = 1.0
    Q2 = np.zeros((2, 2))
    keep = (1.0 - decay_eff)
    (p1_seq, p2_seq) = init_prob_arrays(num_trials)
    q1 = np.zeros(2)
    for t in range(num_trials):
        v_planet = max_q2_per_state(Q2)
        bias1 = np.array([((v_planet[0] - v_planet[1]) * tr_bias), ((- tr_bias) * (v_planet[0] - v_planet[1]))])
        logits = ((beta * q1) + bias1)
        logits -= logsumexp_max(logits)
        a1 = int(a1_seq[t])
        p1 = exp_logits(logits)
        p1 = (p1 / (1e-12 + np.sum(p1)))
        p1_seq[t] = p1[a1]
        s = int(s_seq[t])
        logits = (Q2[s] * beta)
        logits -= logsumexp_max(logits)
        a2 = int(a2_seq[t])
        p2 = exp_logits(logits)
        p2 = (p2 / (1e-12 + np.sum(p2)))
        p2_seq[t] = p2[a2]
        r = r_seq[t]
        q1 *= keep
        Q2 *= keep
        delta2 = update_q2(Q2, a2, alpha, r, s)
        delta1 = update_q1(Q2, a1, a2, alpha, q1, s)
    epsilon = 1e-10
    nll = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return nll

# Participant 32


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha_pos, alpha_neg, beta, z_forget, psi_persist) = params
    Q2 = np.zeros((2, 2))
    (last_a1, num_trials, p1_seq, p2_seq) = init_with_last_action(a1_seq)
    prev_a2_state = {0: None, 1: None}
    prior_q1 = 0.5
    prior_q2 = 0.5
    q1 = np.zeros(2)
    stai_arr = stai_arr[0]
    for t in range(num_trials):
        bias1 = np.zeros(2)
        pers = (stai_modulate(stai_arr) * psi_persist)
        if (last_a1 is not None):
            bias1[last_a1] = 1.0
        a1 = a1_seq[t]
        bias2 = np.zeros(2)
        logits = ((center_values(q1) * beta) + (bias1 * pers))
        probs1 = np.exp((logits - logsumexp_max(logits)))
        probs1 = (probs1 / np.sum(probs1))
        s = record_stage1(a1, p1_seq, probs1, s_seq, t)
        q2_row = Q2[s].copy()
        if (prev_a2_state[s] is not None):
            bias2[prev_a2_state[s]] = 1.0
        a2 = a2_seq[t]
        logits = ((center_values(q2_row) * beta) + (bias2 * pers))
        probs2 = np.exp((logits - logsumexp_max(logits)))
        probs2 = (probs2 / np.sum(probs2))
        r = record_stage2(a2, p2_seq, probs2, r_seq, t)
        delta2 = (r - Q2[(s, a2)])
        lr2 = (alpha_pos if (delta2 >= 0.0) else alpha_neg)
        Q2[(s, a2)] += (delta2 * lr2)
        delta1 = (Q2[(s, a2)] - q1[a1])
        lr1 = (alpha_pos if (delta1 >= 0.0) else alpha_neg)
        q1[a1] += (delta1 * lr1)
        decay = np.clip((stai_modulate(stai_arr) * z_forget), 0.0, 1.0)
        Q2 = ((memory_retention(decay) * Q2) + (decay * prior_q2))
        last_a1 = a1
        prev_a2_state[s] = a2
        q1 = ((memory_retention(decay) * q1) + (decay * prior_q1))
    epsilon = 1e-12
    neg_log_lik = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return neg_log_lik

# Participant 33


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, w_base, lambda_, pers) = params
    (Q2, T) = init_q_and_T()
    last_a1 = None
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai_arr = float(stai_arr[0])
    w_mb = ((1.0 - (0.7 * stai_arr)) * w_base)
    w_mb = max(0.0, min(1.0, w_mb))
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        bias = np.zeros(2)
        mb_values = (T @ max_q2_per_state(Q2))
        pers_eff = ((stai_scale(stai_arr) + 1.0) * pers)
        r = float(r_seq[t])
        s = int(s_seq[t])
        if (last_a1 is not None):
            bias[last_a1] += pers_eff
        delta2 = (r - Q2[(s, a2)])
        q1 = ((((1.0 - w_mb) * q1) + (mb_values * w_mb)) + bias)
        q1_shift = center_values(q1)
        exp_q1 = np.exp((beta * q1_shift))
        probs_1 = (exp_q1 / (1e-12 + np.sum(exp_q1)))
        p1_seq[t] = probs_1[a1]
        q2 = Q2[s].copy()
        q2_shift = (q2 - np.max(q2))
        exp_q2 = np.exp((beta * q2_shift))
        probs_2 = (exp_q2 / (1e-12 + np.sum(exp_q2)))
        p2_seq[t] = probs_2[a2]
        Q2[(s, a2)] += td_step(alpha, delta2)
        delta1 = update_q1(Q2, a1, a2, alpha, q1, s)
        q1[a1] += ((alpha * lambda_) * delta2)
        last_a1 = a1
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return float(nll)

# Participant 34


def cognitive_model3(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, tau_T, k_anx_trans_bias, psi_perseverate) = params
    Q2 = np.zeros((2, 2))
    T = np.full((2, 2), 0.5)
    last_a1 = None
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai = float(stai_arr[0])
    for t in range(num_trials):
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        stick_vec = np.zeros(2)
        if (last_a1 is not None):
            stick_vec[last_a1] = 1.0
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        perseveration_bias = ((psi_perseverate * stai) * stick_vec)
        q1_combined = ((0.5 * q1) + (0.5 * q1_mb))
        logits = (perseveration_bias + q1_combined)
        logits = (logits - logsumexp_max(logits))
        probs1 = np.exp((beta * logits))
        probs1 = (probs1 / np.sum(probs1))
        s = record_stage1(a1, p1_seq, probs1, s_seq, t)
        logits = (Q2[s] - np.max(Q2[s]))
        probs2 = np.exp((beta * logits))
        probs2 = (probs2 / np.sum(probs2))
        p2_seq[t] = probs2[a2]
        p_obs = T[(a1, s)]
        is_rare = (p_obs < 0.5)
        r = r_seq[t]
        if is_rare:
            tau_eff = (((k_anx_trans_bias * stai) + 1.0) * tau_T)
        else:
            tau_eff = ((1.0 - (k_anx_trans_bias * stai)) * tau_T)
        if (tau_eff < 0.0):
            tau_eff = 0.0
        if (tau_eff > 1.0):
            tau_eff = 1.0
        T[(a1, s)] += ((1.0 - T[(a1, s)]) * tau_eff)
        other = (1 - s)
        T[(a1, other)] = (1.0 - T[(a1, s)])
        delta2 = update_q2(Q2, a2, alpha, r, s)
        boot = Q2[(s, a2)]
        delta1 = (boot - q1[a1])
        q1[a1] += td_step(alpha, delta1)
        last_a1 = a1
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return float(nll)

# Participant 35


def cognitive_model3(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta_base, phi_leak, zeta_wsls, nu_loss) = params
    Q2 = np.zeros((2, 2))
    num_trials = len(a1_seq)
    stai_arr = float(stai_arr[0])
    beta = max(0.0, min(10.0, ((1.0 - stai_scale(stai_arr)) * beta_base)))
    has_prev = np.zeros(2, dtype=bool)
    leak = max(0.0, min(1.0, phi_leak))
    nu_eff = max(0.0, min(2.0, ((1.0 + stai_arr) * nu_loss)))
    (p1_seq, p2_seq) = init_prob_arrays(num_trials)
    prev_a2 = np.zeros(2, dtype=int)
    prev_sign = np.zeros(2)
    q1 = np.zeros(2)
    w_wsls = max(0.0, min(1.0, (stai_modulate(stai_arr) * zeta_wsls)))
    for t in range(num_trials):
        q_centered = center_values(q1)
        probs1 = softmax_exp(beta, q_centered)
        probs1 /= np.sum(probs1)
        a1 = int(a1_seq[t])
        p1_seq[t] = probs1[a1]
        s = int(s_seq[t])
        q_centered = (Q2[s] - np.max(Q2[s]))
        probs2_soft = softmax_exp(beta, q_centered)
        probs2_soft /= np.sum(probs2_soft)
        if has_prev[s]:
            if (prev_sign[s] >= 0.0):
                wsls_probs = np.array([0.0, 0.0])
                wsls_probs[prev_a2[s]] = 1.0
            else:
                wsls_probs = np.array([0.0, 0.0])
                wsls_probs[(1 - prev_a2[s])] = 1.0
        else:
            wsls_probs = np.array([0.5, 0.5])
        probs2 = (((1.0 - w_wsls) * probs2_soft) + (w_wsls * wsls_probs))
        probs2 /= np.sum(probs2)
        a2 = int(a2_seq[t])
        p2_seq[t] = probs2[a2]
        r = float(r_seq[t])
        util = (r if (r >= 0.0) else ((- nu_eff) * (- r)))
        Q2 *= (1.0 - leak)
        q1 *= (1.0 - leak)
        delta2 = (util - Q2[(s, a2)])
        Q2[(s, a2)] += td_step(alpha, delta2)
        (delta1, target1) = update_q1_full(Q2, a1, a2, alpha, q1, s)
        has_prev[s] = True
        prev_a2[s] = a2
        prev_sign[s] = (1.0 if (r >= 0.0) else (- 1.0))
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return nll

# Participant 36


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta_base, k_vol, w_MB, stickiness) = params
    (Q2, T) = init_q_and_T()
    last_a1 = 0
    (num_trials, p1_seq, p2_seq) = init_trial_arrays_basic(a1_seq)
    prev_a2 = 0
    q1 = np.zeros(2)
    stai_arr = float(stai_arr[0])
    beta = ((1.0 - stai_scale(stai_arr)) * beta_base)
    v = 0.0
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        beta = (beta / (1.0 + v))
        bias1 = np.array([0.0, 0.0])
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1 = (((1.0 - w_MB) * q1) + (q1_mb * w_MB))
        r = float(r_seq[t])
        s = int(s_seq[t])
        bias1[last_a1] += stickiness
        bias2 = np.array([0.0, 0.0])
        logits = ((center_values(q1) * beta) + bias1)
        probs1 = np.exp((logits - logsumexp_max(logits)))
        probs1 = (probs1 / np.sum(probs1))
        p1_seq[t] = probs1[a1]
        bias2[prev_a2] += stickiness
        delta2 = (r - Q2[(s, a2)])
        logits = (((Q2[s] - np.max(Q2[s])) * beta) + bias2)
        probs2 = np.exp((logits - logsumexp_max(logits)))
        probs2 = (probs2 / np.sum(probs2))
        p2_seq[t] = probs2[a2]
        Q2[(s, a2)] += td_step(alpha, delta2)
        target1 = Q2[(s, a2)]
        delta1 = (target1 - q1[a1])
        v = (((1.0 - k_vol) * v) + ((delta2 * delta2) * k_vol))
        q1[a1] += td_step(alpha, delta1)
        last_a1 = a1
        prev_a2 = a2
    epsilon = 1e-12
    neg_log_likelihood = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return float(neg_log_likelihood)

# Participant 37


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, alpha_T0, alpha_Tgain, pers) = params
    Q2 = np.zeros((2, 2))
    T = np.full((2, 2), 0.5)
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai_arr = float(stai_arr[0])
    alpha_T = ((alpha_Tgain * stai_arr) + alpha_T0)
    alpha_T = min(1.0, max(0.0, alpha_T))
    last_a1 = (- 1)
    pers = ((((stai_arr - 0.51) * 0.5) + 1.0) * pers)
    w_mb = (((0.51 - stai_arr) * 0.4) + 0.5)
    w_mb = min(1.0, max(0.0, w_mb))
    for t in range(num_trials):
        bias = np.zeros(2)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1 = integrate_mf_mb(q1, q1_mb, w_mb)
        if (last_a1 >= 0):
            bias[last_a1] = pers
        logits = ((beta * q1) + bias)
        logits -= logsumexp_max(logits)
        probs1 = exp_logits(logits)
        probs1 /= np.sum(probs1)
        a1 = a1_seq[t]
        s = record_stage1(a1, p1_seq, probs1, s_seq, t)
        q2 = Q2[s]
        logits = (beta * q2)
        logits -= logsumexp_max(logits)
        probs2 = exp_logits(logits)
        probs2 /= np.sum(probs2)
        a2 = a2_seq[t]
        r = record_stage2(a2, p2_seq, probs2, r_seq, t)
        T[(a1, s)] += ((1.0 - T[(a1, s)]) * alpha_T)
        T[(a1, (1 - s))] = (1.0 - T[(a1, s)])
        delta2 = update_q2(Q2, a2, alpha, r, s)
        delta1 = (Q2[(s, a2)] - q1[a1])
        lambda_ = stai_arr
        q1[a1] += (((delta2 * lambda_) + delta1) * alpha)
        last_a1 = a1
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return nll

# Participant 38


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, mb_w, asym, decay) = params
    Q2 = np.full((2, 2), 0.5)
    T = np.array([[0.7, 0.3], [0.3, 0.7]])
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai_arr = float(stai_arr[0])
    decay_eff = np.clip((stai_modulate(stai_arr) * decay), 0.0, 1.0)
    decay_eff = np.clip((stai_modulate(stai_arr) * decay), 0.0, 1.0)
    w_mb = ((1.0 - stai_arr) * mb_w)
    w_mb = np.clip(w_mb, 0.0, 1.0)
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        alpha_neg = np.clip((((asym * stai_arr) + 1.0) * alpha), 0.0, 1.0)
        alpha_pos = np.clip(((((1.0 - stai_arr) * asym) + 1.0) * alpha), 0.0, 1.0)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1 = integrate_mf_mb(q1, q1_mb, w_mb)
        exp_q1 = np.exp((center_values(q1) * beta))
        probs1 = (exp_q1 / np.sum(exp_q1))
        s = record_stage1(a1, p1_seq, probs1, s_seq, t)
        exp_q2 = np.exp(((Q2[s] - np.max(Q2[s])) * beta))
        probs2 = (exp_q2 / np.sum(exp_q2))
        r = record_stage2(a2, p2_seq, probs2, r_seq, t)
        delta2 = (r - Q2[(s, a2)])
        lr = (alpha_pos if (delta2 >= 0) else alpha_neg)
        Q2[(s, a2)] += (delta2 * lr)
        target1 = Q2[(s, a2)]
        q1[a1] += ((target1 - q1[a1]) * alpha)
        Q2 = (((1.0 - decay_eff) * Q2) + (0.5 * decay_eff))
        q1 = ((1.0 - decay_eff) * q1)
    epsilon = 1e-12
    neg_log_likelihood = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return neg_log_likelihood

# Participant 39


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, rho_surp0, kappa_rep0, zeta_pers2) = params
    (Q2, T) = init_q_and_T()
    epsilon = 1e-12
    (last_a1, num_trials, p1_seq, p2_seq) = init_with_last_action(a1_seq)
    prev_a2_by_state = {0: None, 1: None}
    q1 = np.zeros(2)
    stai = float(stai_arr[0])
    w_mb = max(0.0, min(1.0, (1.0 - stai)))
    for t in range(num_trials):
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1_combined = integrate_mf_mb(q1, q1_mb, w_mb)
        if (last_a1 is not None):
            pers = np.zeros(2)
            pers[last_a1] = 1.0
            pers_eff = ((1.0 + stai) * kappa_rep0)
            q1_combined = ((pers * pers_eff) + q1_combined)
        a1 = a1_seq[t]
        q_centered = (q1_combined - np.max(q1_combined))
        probs_1 = softmax_exp(beta, q_centered)
        probs_1 = (probs_1 / (epsilon + np.sum(probs_1)))
        p1_seq[t] = probs_1[a1]
        s2 = s_seq[t]
        prev_a2 = prev_a2_by_state[s2]
        q2 = Q2[s2].copy()
        if (prev_a2 is not None):
            stick2 = np.zeros(2)
            stick2[prev_a2] = 1.0
            zeta_eff = ((1.0 - (0.5 * stai)) * zeta_pers2)
            q2 = ((stick2 * zeta_eff) + q2)
        a2 = a2_seq[t]
        q_centered = (q2 - np.max(q2))
        probs_2 = softmax_exp(beta, q_centered)
        probs_2 = (probs_2 / (epsilon + np.sum(probs_2)))
        p2_seq[t] = probs_2[a2]
        r = r_seq[t]
        delta2 = (r - Q2[(s2, a2)])
        Q2[(s2, a2)] += td_step(alpha, delta2)
        td_target1 = Q2[(s2, a2)]
        delta1 = (td_target1 - q1[a1])
        q1[a1] += td_step(alpha, delta1)
        last_a1 = a1
        p_trans = T[(a1, s2)]
        prev_a2_by_state[s2] = a2
        surprise = (1.0 - p_trans)
        w_mb = ((((1.0 + stai) * rho_surp0) * (surprise - (w_mb - (1.0 - stai)))) + w_mb)
        w_mb = max(0.0, min(1.0, w_mb))
    nll = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return nll

# Participant 4


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta_base, eta_surp, kappa_pers) = params
    Q2 = (0.5 + np.zeros((2, 2)))
    T = np.array([[0.7, 0.3], [0.3, 0.7]])
    (last_a1, num_trials, p1_seq, p2_seq) = init_with_last_action(a1_seq)
    prev_a2 = np.array([None, None], dtype=object)
    stai_arr = float(stai_arr[0])
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        r = float(r_seq[t])
        s = int(s_seq[t])
        is_rare = (((a1 == 0) and (s == 1)) or ((a1 == 1) and (s == 0)))
        beta = ((1.0 - ((1.0 if is_rare else 0.0) * (eta_surp * stai_arr))) * beta_base)
        beta = max(1e-06, beta)
        bias1 = np.zeros(2)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        if (last_a1 is not None):
            bias1[last_a1] += kappa_pers
        bias2 = np.zeros(2)
        if (prev_a2[s] is not None):
            bias2[int(prev_a2[s])] += kappa_pers
        delta2 = (r - Q2[(s, a2)])
        logits = (((q1_mb - np.max(q1_mb)) * beta) + bias1)
        logits = (logits - logsumexp_max(logits))
        probs1 = normalize_probs(logits)
        p1_seq[t] = probs1[a1]
        q2_row = Q2[s]
        logits = ((center_values(q2_row) * beta) + bias2)
        logits = (logits - logsumexp_max(logits))
        probs2 = normalize_probs(logits)
        p2_seq[t] = probs2[a2]
        Q2[(s, a2)] += td_step(alpha, delta2)
        last_a1 = a1
        prev_a2[s] = a2
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return float(nll)

# Participant 40


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, tau, pers) = params
    Q2 = np.zeros((2, 2))
    T = np.full((2, 2), 0.5)
    (last_a1, num_trials, p1_seq, p2_seq) = init_with_last_action(a1_seq)
    stai = float(stai_arr[0])
    tau_eff = ((1.0 - stai) * tau)
    for t in range(num_trials):
        bias1 = np.zeros(2)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        if (last_a1 is not None):
            bias1[last_a1] = pers
        a1 = int(a1_seq[t])
        logits = ((beta * q1_mb) + bias1)
        logits -= logsumexp_max(logits)
        probs1 = exp_logits(logits)
        probs1 /= np.sum(probs1)
        a2 = int(a2_seq[t])
        p1_seq[t] = probs1[a1]
        s2 = int(s_seq[t])
        logits = (Q2[s2] * beta)
        logits -= logsumexp_max(logits)
        probs2 = exp_logits(logits)
        probs2 /= np.sum(probs2)
        r = record_stage2(a2, p2_seq, probs2, r_seq, t)
        if (tau_eff > 0.0):
            T[a1] = ((1.0 - tau_eff) * T[a1])
            T[(a1, s2)] += tau_eff
            row_sum = np.sum(T[a1])
            if (row_sum > 0):
                T[a1] /= row_sum
        delta2 = (r - Q2[(s2, a2)])
        Q2[(s2, a2)] += td_step(alpha, delta2)
        last_a1 = a1
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return nll

# Participant 41


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, lambda_, beta, w_base, pers_base) = params
    (Q2, T) = init_q_and_T()
    last_a1 = (- 1)
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai = stai_arr[0]
    pers_eff = (((0.5 * stai) + 0.5) * pers_base)
    w_eff_scale = (1.0 - (0.5 * stai))
    w_eff_base = (w_base * w_eff_scale)
    for t in range(num_trials):
        bias = np.zeros(2)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        w_mb_eff = w_eff_base
        q1_combined = integrate_mf_mb(q1, q1_mb, w_mb_eff)
        if (last_a1 >= 0):
            bias[last_a1] = pers_eff
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        q1_combined = (bias + q1_combined)
        exp_q1 = np.exp(((q1_combined - np.max(q1_combined)) * beta))
        probs_1 = (exp_q1 / np.sum(exp_q1))
        p1_seq[t] = probs_1[a1]
        s = s_seq[t]
        q2_row = Q2[s]
        exp_q2 = np.exp((center_values(q2_row) * beta))
        probs_2 = (exp_q2 / np.sum(exp_q2))
        p2_seq[t] = probs_2[a2]
        r = r_seq[t]
        delta2 = update_q2(Q2, a2, alpha, r, s)
        delta1 = update_q1(Q2, a1, a2, alpha, q1, s)
        q1[a1] += ((alpha * lambda_) * delta2)
        last_a1 = a1
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return nll

# Participant 42


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha_g, alpha_l, beta, beta, lapse_base) = params
    Q2 = np.zeros((2, 2))
    num_trials = len(a1_seq)
    stai_arr = float(stai_arr[0])
    lambda_ = (1.0 - stai_arr)
    lambda_ = min(max(lambda_, 0.0), 1.0)
    lapse = min(0.5, max(0.0, (lapse_base * stai_arr)))
    (p1_seq, p2_seq) = init_prob_arrays(num_trials)
    q1 = np.zeros(2)
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        pref1 = (beta * q1)
        z1 = np.max(pref1)
        exp1 = np.exp((pref1 - z1))
        soft1 = (exp1 / np.sum(exp1))
        probs_1 = (((1.0 - lapse) * soft1) + (0.5 * lapse))
        p1_seq[t] = probs_1[a1]
        s = s_seq[t]
        q2_row = (Q2[s] * beta)
        z2 = np.max(q2_row)
        exp2 = np.exp((q2_row - z2))
        soft2 = (exp2 / np.sum(exp2))
        probs_2 = (((1.0 - lapse) * soft2) + (0.5 * lapse))
        p2_seq[t] = probs_2[a2]
        r = r_seq[t]
        delta2 = (r - Q2[(s, a2)])
        alpha = (alpha_g if (delta2 > 0.0) else alpha_l)
        Q2[(s, a2)] += td_step(alpha, delta2)
        delta1 = (Q2[(s, a2)] - q1[a1])
        q1[a1] += (((delta2 * lambda_) + delta1) * alpha)
    (epsilon, nll) = compute_nll(p1_seq, p2_seq)
    return nll

# Participant 43


def cognitive_model3(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, rho_base, k_anx_rho, decay) = params
    num_trials = len(a1_seq)
    stai = float(stai_arr[0])
    decay = ((k_anx_rho * stai) + rho_base)
    if (decay < 0.0):
        decay = 0.0
    if (decay > 1.0):
        decay = 1.0
    Q2 = np.zeros((2, 2))
    Q2 = np.zeros((2, 2))
    epsilon = 1e-10
    (p1_seq, p2_seq) = init_prob_arrays(num_trials)
    q1 = np.zeros(2)
    for t in range(num_trials):
        logits = (beta * q1)
        var = np.maximum((Q2 - (Q2 ** 2)), 0.0)
        std = np.sqrt((1e-12 + var))
        u = (Q2 - (decay * std))
        logits -= logsumexp_max(logits)
        a1 = int(a1_seq[t])
        soft1 = exp_logits(logits)
        probs1 = (soft1 / (epsilon + np.sum(soft1)))
        p1_seq[t] = probs1[a1]
        s = int(s_seq[t])
        logits = (beta * u[s])
        logits -= logsumexp_max(logits)
        a2 = int(a2_seq[t])
        other_a2 = (1 - a2)
        Q2[(s, other_a2)] = (memory_retention(decay) * Q2[(s, other_a2)])
        Q2[(s, other_a2)] = (memory_retention(decay) * Q2[(s, other_a2)])
        other_state = (1 - s)
        Q2[(other_state, 0)] = (memory_retention(decay) * Q2[(other_state, 0)])
        Q2[(other_state, 0)] = (memory_retention(decay) * Q2[(other_state, 0)])
        Q2[(other_state, 1)] = (memory_retention(decay) * Q2[(other_state, 1)])
        Q2[(other_state, 1)] = (memory_retention(decay) * Q2[(other_state, 1)])
        soft2 = exp_logits(logits)
        probs2 = (soft2 / (epsilon + np.sum(soft2)))
        r = record_stage2(a2, p2_seq, probs2, r_seq, t)
        Q2[(s, a2)] = ((((r * r) - Q2[(s, a2)]) * alpha) + Q2[(s, a2)])
        Q2[(s, a2)] = (((r - Q2[(s, a2)]) * alpha) + Q2[(s, a2)])
        q1[(1 - a1)] = (memory_retention(decay) * q1[(1 - a1)])
        var_sa = max((Q2[(s, a2)] - (Q2[(s, a2)] ** 2)), 0.0)
        u_sa = (Q2[(s, a2)] - (decay * np.sqrt((1e-12 + var_sa))))
        delta1 = (u_sa - q1[a1])
        q1[a1] = (td_step(alpha, delta1) + q1[a1])
    nll = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return nll

# Participant 44


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, phi_trans, eta_forget, xi_anxTemp) = params
    (Q2, T) = init_q_and_T()
    num_trials = len(a1_seq)
    stai_arr = stai_arr[0]
    beta = max(0.001, ((1.0 - (stai_arr * xi_anxTemp)) * beta))
    epsilon = 1e-12
    forget_eff = np.clip((eta_forget * stai_arr), 0.0, 1.0)
    lambda_anx = np.clip(stai_arr, 0.0, 1.0)
    (p1_seq, p2_seq) = init_prob_arrays(num_trials)
    phi_eff = np.clip(((stai_scale(stai_arr) + 1.0) * phi_trans), 0.0, 1.0)
    q1 = np.zeros(2)
    for t in range(num_trials):
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        row_ent = []
        for r in range(2):
            h = 0.0
            p = T[r]
            for x in p:
                if (x > 0):
                    h -= ((np.log(x) / np.log(2)) * x)
            row_ent.append(h)
        ent_mean = ((row_ent[0] + row_ent[1]) * 0.5)
        w_mb = np.clip(((1.0 - ent_mean) * (1.0 - stai_arr)), 0.0, 1.0)
        q1 = integrate_mf_mb(q1, q1_mb, w_mb)
        logits = (beta * q1)
        logits -= logsumexp_max(logits)
        probs1 = exp_logits(logits)
        probs1 /= np.sum(probs1)
        a1 = a1_seq[t]
        s = record_stage1(a1, p1_seq, probs1, s_seq, t)
        logits = (Q2[s] * beta)
        logits -= logsumexp_max(logits)
        probs2 = exp_logits(logits)
        probs2 /= np.sum(probs2)
        Q2 = ((1.0 - forget_eff) * Q2)
        a2 = a2_seq[t]
        p2_seq[t] = probs2[a2]
        q1 = ((1.0 - forget_eff) * q1)
        delta1 = (Q2[(s, a2)] - q1[a1])
        r = r_seq[t]
        q1[a1] += td_step(alpha, delta1)
        delta2 = update_q2(Q2, a2, alpha, r, s)
        q1[a1] += ((alpha * lambda_anx) * delta2)
        obs = np.array([(1.0 if (i == s) else 0.0) for i in range(2)])
        T[a1] = (((1.0 - phi_eff) * T[a1]) + (obs * phi_eff))
        T[a1] = np.clip(T[a1], 1e-06, 1.0)
        T[a1] /= np.sum(T[a1])
    neg_log_lik = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return neg_log_lik

# Participant 5


def cognitive_model3(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta_base, decay, mbw0) = params
    Q2 = (0.5 * np.ones((2, 2)))
    T = np.array([[0.7, 0.3], [0.3, 0.7]])
    num_trials = len(a1_seq)
    q1 = np.zeros(2)
    stai = float(stai_arr[0])
    beta = np.clip(((((1.0 - stai) * 0.5) + 0.5) * beta_base), 0.0, 10.0)
    epsilon = 1e-12
    forget_eff = np.clip((((0.5 * stai) + 0.5) * decay), 0.0, 1.0)
    p1 = np.zeros(num_trials)
    p2 = np.zeros(num_trials)
    w_mb = np.clip(((((1.0 - mbw0) * (1.0 - stai)) * 0.5) + mbw0), 0.0, 1.0)
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1 = integrate_mf_mb(q1, q1_mb, w_mb)
        logits = (center_values(q1) * beta)
        probs1 = exp_logits(logits)
        r = float(r_seq[t])
        s = int(s_seq[t])
        probs1 /= (epsilon + np.sum(probs1))
        logits = ((Q2[s] - np.max(Q2[s])) * beta)
        p1[t] = probs1[a1]
        probs2 = exp_logits(logits)
        probs2 /= (epsilon + np.sum(probs2))
        Q2 = (((1.0 - forget_eff) * Q2) + (0.5 * forget_eff))
        delta2 = (r - Q2[(s, a2)])
        p2[t] = probs2[a2]
        Q2[(s, a2)] += td_step(alpha, delta2)
        alpha1 = ((((1.0 - stai) * 0.2) + 0.8) * alpha)
        target1 = Q2[(s, a2)]
        delta1 = (target1 - q1[a1])
        q1[a1] += (alpha1 * delta1)
    neg_ll = (- (np.sum(np.log((epsilon + p1))) + np.sum(np.log((epsilon + p2)))))
    return float(neg_ll)

# Participant 6


def cognitive_model1(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, w_mb, zeta, bias_prev) = params
    (Q2, T) = init_q_and_T()
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai_arr = float(stai_arr[0])
    bias_prev_eff_base = np.clip(((1.0 - stai_arr) * bias_prev), 0.0, 1.0)
    last_a1 = None
    r_prev = None
    s_prev = None
    w_mb_eff = np.clip(((1.0 - (0.4 * stai_arr)) * w_mb), 0.0, 1.0)
    for t in range(num_trials):
        max_q2_by_state = max_q2_per_state(Q2)
        pref = np.zeros(2)
        q1_plan = (T @ max_q2_by_state)
        if (last_a1 is not None):
            was_common = (1 if (((last_a1 == 0) and (s_prev == 0)) or ((last_a1 == 1) and (s_prev == 1))) else 0)
            sign = (1.0 if (r_prev > 0) else ((- 1.0) if (r_prev < 0) else 0.0))
            bias = ((1.0 if was_common else (- 1.0)) * (bias_prev_eff_base * sign))
            pref[last_a1] += bias
            pref[(1 - last_a1)] -= bias
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        q1 = ((((1.0 - w_mb_eff) * q1) + (q1_plan * w_mb_eff)) + pref)
        q1_shift = center_values(q1)
        probs_1 = np.exp((beta * q1_shift))
        probs_1 = (probs_1 / np.sum(probs_1))
        p1_seq[t] = probs_1[a1]
        s = int(s_seq[t])
        q2_row = Q2[s]
        q2_shift = center_values(q2_row)
        probs_2 = np.exp((beta * q2_shift))
        probs_2 = (probs_2 / np.sum(probs_2))
        p2_seq[t] = probs_2[a2]
        r = r_seq[t]
        Q2 *= (1.0 - zeta)
        q1 *= (1.0 - zeta)
        delta2 = update_q2(Q2, a2, alpha, r, s)
        (delta1, target1) = update_q1_full(Q2, a1, a2, alpha, q1, s)
        (last_a1, s_prev, r_prev) = (a1, s, r)
    epsilon = 1e-10
    nll = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return float(nll)

# Participant 7


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, w_mb, z0, k_z) = params
    Q2 = np.zeros((2, 2))
    S = np.zeros(2)
    num_trials = len(a1_seq)
    p_common = 0.7
    T = np.array([[p_common, (1 - p_common)], [(1 - p_common), p_common]])
    (p1_seq, p2_seq) = init_prob_arrays(num_trials)
    q1 = np.zeros(2)
    stai_arr = float(stai_arr[0])
    z_eff = ((k_z * stai_arr) + z0)
    if (z_eff < 0.0):
        z_eff = 0.0
    if (z_eff > 1.0):
        z_eff = 1.0
    epsilon = 1e-12
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        (max_q2, q1_mb) = compute_mb_values(Q2, T)
        q1 = (integrate_mf_mb(q1, q1_mb, w_mb) + (S * z_eff))
        q_centered = center_values(q1)
        probs_1 = softmax_exp(beta, q_centered)
        probs_1 = (probs_1 / np.sum(probs_1))
        p1_seq[t] = probs_1[a1]
        s = s_seq[t]
        q_centered = (Q2[s] - np.max(Q2[s]))
        probs_2 = softmax_exp(beta, q_centered)
        probs_2 = (probs_2 / np.sum(probs_2))
        p2_seq[t] = probs_2[a2]
        r = r_seq[t]
        delta2 = update_q2(Q2, a2, alpha, r, s)
        boot = Q2[(s, a2)]
        delta1 = (boot - q1[a1])
        q1[a1] += td_step(alpha, delta1)
        p_s_given_a = T[(a1, s)]
        if (p_s_given_a < 1e-08):
            p_s_given_a = 1e-08
        surprise = (- np.log(p_s_given_a))
        S[a1] += ((surprise - S[a1]) * alpha)
    nll = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return nll

# Participant 8


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (rho_decay, beta, phi0, pi0) = params
    Q2 = (1.0 + np.zeros((2, 2)))
    Q2 = (0.5 + np.zeros((2, 2)))
    T = np.array([[0.7, 0.3], [0.3, 0.7]])
    (last_a1, num_trials, p1_seq, p2_seq) = init_with_last_action(a1_seq)
    stai = float(stai_arr[0])
    pers = ((1.0 - (0.5 * stai)) * pi0)
    w_mb = ((0.5 + stai) * phi0)
    for t in range(num_trials):
        Q2 *= rho_decay
        Q2 *= rho_decay
        bias = np.zeros(2)
        m = (Q2 / (1e-08 + Q2))
        u = (1.0 / (1.0 + Q2))
        q2 = ((u * w_mb) + m)
        vmax2 = np.max(q2, axis=1)
        q1 = (T @ vmax2)
        if (last_a1 is not None):
            bias[last_a1] += pers
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        logits = (bias + q1)
        logits = (logits - logsumexp_max(logits))
        probs1 = np.exp((beta * logits))
        probs1 = (probs1 / np.sum(probs1))
        p1_seq[t] = probs1[a1]
        s2 = s_seq[t]
        logits = (q2[s2] - np.max(q2[s2]))
        probs2 = np.exp((beta * logits))
        probs2 = (probs2 / np.sum(probs2))
        r = record_stage2(a2, p2_seq, probs2, r_seq, t)
        Q2[(s2, a2)] += 1.0
        Q2[(s2, a2)] += r
        last_a1 = a1
    epsilon = 1e-10
    nll = (- log_likelihood(epsilon, p1_seq, p2_seq))
    return nll

# Participant 9


def cognitive_model2(a1_seq, s_seq, a2_seq, r_seq, stai_arr, params):
    (alpha, beta, decay, wsls_gain, xi) = params
    Q2 = np.zeros((2, 2))
    last_a1 = None
    last_is_common = 0
    last_reward = 0.0
    (num_trials, p1_seq, p2_seq, q1) = init_trial_arrays(a1_seq)
    stai = float(stai_arr[0])
    for t in range(num_trials):
        (a1, a2) = get_trial_actions(a1_seq, a2_seq, t)
        decay_eff = (((0.5 * stai) + 0.5) * decay)
        r = r_seq[t]
        s = s_seq[t]
        q1 *= (1.0 - decay_eff)
        Q2 *= (1.0 - decay_eff)
        bias1 = np.zeros(2)
        if (last_a1 is not None):
            reward_term = ((2.0 * np.clip(last_reward, 0.0, 1.0)) - 1.0)
            trans_term = (1.0 if (last_is_common == 1) else (- 1.0))
            base_signal = (((1.0 - xi) * trans_term) + (reward_term * xi))
            anxiety_gain = ((0.5 * stai) + 0.5)
            signed_signal = (((1.0 - stai) * base_signal) - ((anxiety_gain - (1.0 - stai)) * reward_term))
            bias_strength = (signed_signal * wsls_gain)
            bias1[last_a1] += bias_strength
        logits = ((center_values(q1) * beta) + bias1)
        probs1 = exp_logits(logits)
        probs1 /= np.sum(probs1)
        logits = ((Q2[s] - np.max(Q2[s])) * beta)
        p1_seq[t] = probs1[a1]
        probs2 = exp_logits(logits)
        probs2 /= np.sum(probs2)
        delta2 = (r - Q2[(s, a2)])
        is_common = int((((a1 == 0) and (s == 0)) or ((a1 == 1) and (s == 1))))
        p2_seq[t] = probs2[a2]
        Q2[(s, a2)] += td_step(alpha, delta2)
        (delta1, target1) = update_q1_full(Q2, a1, a2, alpha, q1, s)
        last_a1 = a1
        last_is_common = is_common
        last_reward = r
    epsilon = 1e-12
    return (- log_likelihood(epsilon, p1_seq, p2_seq))

