from /home/aj9225/gecco-1/results/two_step_task_individual_stai/cognitive_library import *
import numpy as np

models = {}

def participant_0(action_1, state, action_2, reward, stai, model_parameters):
        Idea:
        - Standard stage-2 MF learning from rewards.
        - Stage-1 MF receives TD backup from obtained second-stage value.
        - Additionally, after rare transitions, a fraction of the TD signal is misassigned
          to the unchosen first-stage action (transition-dependent MF). This fraction grows
          with anxiety, capturing stress-driven miscrediting.
        - Stage-2 "spillover" generalization: the reward also partially updates the same-index
          alien in the unvisited planet; spillover increases with anxiety.
        - Stage-1 policy includes a baseline bias toward spaceship A.
        - Stage-1 choice value is a mixture of MB (known transitions) and MF with weight
          w = 1 - stai (more MF under higher anxiety).
        Parameters (bounds):
        - alpha: [0,1] learning rate for all TD updates
        - spill: [0,1] baseline fraction of reward that generalizes to the other state's same action
        - rare_bias: [0,1] baseline fraction of TD signal credited to the unchosen first-stage action after rare transitions
        - beta: [0,10] inverse temperature for both stages
        - biasA: [0,1] additive bias toward choosing spaceship A at stage 1
        Inputs:
        - action_1, state, action_2, reward: arrays of length T
        - stai: array-like with one element in [0,1]
        - model_parameters: [alpha, spill, rare_bias, beta, biasA]
        Returns:
        - Negative log-likelihood of observed choices (stage 1 + stage 2).
        alpha, spill, rare_bias, beta, biasA = model_parameters
        n_trials = len(action_1)
        stai_score = float(stai[0])
        T_known = np.array([[0.7, 0.3], [0.3, 0.7]])
        w = np.clip(1.0 - stai_score, 0.0, 1.0)
        spill_eff = spill * (0.5 + 0.5 * stai_score)
        rare_eff = rare_bias * (0.5 + 0.5 * stai_score)
        q1_mf = np.zeros(2)
        q2 = np.zeros((2, 2))
    shared_mechanism_0()
        for t in range(n_trials):
            max_q2 = np.max(q2, axis=1)
            q1_mb = T_known @ max_q2
            q1 = w * q1_mb + (1.0 - w) * q1_mf
            logits1 = q1.copy()
            logits1[0] += biasA
            l1 = beta * (logits1 - np.max(logits1))
            probs1 = np.exp(l1)
            probs1 /= np.sum(probs1)
            a1 = action_1[t]
            p_choice_1[t] = probs1[a1]
            s = state[t]
            logits2 = q2[s].copy()
            l2 = beta * (logits2 - np.max(logits2))
            probs2 = np.exp(l2)
            probs2 /= np.sum(probs2)
            a2 = action_2[t]
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            is_common = (a1 == 0 and s == 0) or (a1 == 1 and s == 1)
            pe2 = r - q2[s, a2]
            q2[s, a2] += alpha * pe2
            other_s = 1 - s
            q2[other_s, a2] += alpha * spill_eff * (r - q2[other_s, a2])
            td1 = q2[s, a2] - q1_mf[a1]
            q1_mf[a1] += alpha * td1
            if not is_common:
                a1_other = 1 - a1
                q1_mf[a1_other] += alpha * rare_eff * (q2[s, a2] - q1_mf[a1_other])
    shared_mechanism_3()

models[0] = participant_0

def participant_1(action_1, state, action_2, reward, stai, model_parameters):
        Core idea:
        - First-stage policy is purely model-based using a learned transition matrix.
        - Anxiety increases the impact of transition surprise on first-stage preferences.
        - Also includes first-stage choice stickiness (perseveration), scaled by anxiety.
        - Second-stage policy is model-free Q-learning.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices (0=A, 1=U).
        state : array-like of int (0 or 1)
            Second-stage states (0=planet X, 1=planet Y).
        action_2 : array-like of int (0 or 1)
            Second-stage choices (0 or 1; aliens).
        reward : array-like of float (0 or 1)
            Coins received each trial.
        stai : array-like of float in [0,1]
            Anxiety score used to scale surprise bonus and stickiness.
        model_parameters : list or array
            [alpha, beta, alpha_t, kappa_stick, phi_anx]
            - alpha in [0,1]: learning rate for second-stage Q-value updates.
            - beta in [0,10]: inverse temperature at both stages.
            - alpha_t in [0,1]: transition learning rate for updating the transition matrix.
            - kappa_stick in [0,1]: strength of first-stage choice perseveration.
            - phi_anx in [0,1]: scales the surprise-to-bonus mapping as a function of anxiety.
    shared_mechanism_1()
            Negative log-likelihood of observed first- and second-stage choices.
        alpha, beta, alpha_t, kappa_stick, phi_anx = model_parameters
        n_trials = len(action_1)
        stai_val = float(stai[0])
        T = np.array([[0.7, 0.3],
                      [0.3, 0.7]], dtype=float)  # rows: actions (A,U), cols: states (X,Y)
        q2 = np.zeros((2, 2), dtype=float)
        p_choice_1 = np.zeros(n_trials, dtype=float)
        p_choice_2 = np.zeros(n_trials, dtype=float)
        prev_a1 = None
        stickiness = np.zeros(2, dtype=float)
        for t in range(n_trials):
            max_q2 = np.max(q2, axis=1)  # per state
            q1_mb = T @ max_q2  # shape (2,)
            bonus = np.zeros(2, dtype=float)
            if t > 0:
                a1_prev = action_1[t - 1]
                s2_prev = state[t - 1]
                p_obs = T[a1_prev, s2_prev]
                surprise = 1.0 - p_obs  # higher when transition was rare/unexpected
                a1_now = action_1[t]
                bonus[a1_now] += phi_anx * stai_val * surprise
            if prev_a1 is not None:
                stickiness = np.zeros(2, dtype=float)
                stickiness[prev_a1] = kappa_stick * (1.0 + stai_val)
            q1_eff = q1_mb + bonus + stickiness
            q1_eff -= np.max(q1_eff)  # softmax stability
            exp_q1 = np.exp(beta * q1_eff)
            probs_1 = exp_q1 / np.sum(exp_q1)
            a1 = action_1[t]
            p_choice_1[t] = probs_1[a1]
            s2 = state[t]
            q2_s = q2[s2].copy()
            q2_s -= np.max(q2_s)
            exp_q2 = np.exp(beta * q2_s)
            probs_2 = exp_q2 / np.sum(exp_q2)
            a2 = action_2[t]
            p_choice_2[t] = probs_2[a2]
            r = reward[t]
            pe2 = r - q2[s2, a2]
            q2[s2, a2] += alpha * pe2
            a1_row = T[a1]
            for s_idx in (0, 1):
                target = 1.0 if s_idx == s2 else 0.0
                a1_row[s_idx] += alpha_t * (target - a1_row[s_idx])
            T[a1] = a1_row / np.sum(a1_row)
            prev_a1 = a1
    shared_mechanism_3()

models[1] = participant_1

def participant_2(action_1, state, action_2, reward, stai, model_parameters):
        This model combines model-free Q-values with model-based estimates from the known transition
        structure. The arbitration weight favoring model-based control is modulated by STAI (anxiety)
        such that lower anxiety increases model-based weighting. Stage-1 Q-values are updated with an
        eligibility trace from the stage-2 prediction error.
    shared_mechanism_2()
        action_1 : array-like, shape (n_trials,)
            First-stage choices (0 = spaceship A, 1 = spaceship U).
        state : array-like, shape (n_trials,)
            Second-stage states (0 = planet X, 1 = planet Y) actually reached after action_1.
        action_2 : array-like, shape (n_trials,)
            Second-stage choices (0/1; X: W/S, Y: P/H).
        reward : array-like, shape (n_trials,)
            Obtained reward (e.g., coins; typically 0/1).
        stai : array-like, shape (1,) or scalar-like
            Participant STAI score in [0,1]. Lower values indicate lower anxiety.
        model_parameters : iterable
            Model parameters (total <= 5):
            - alpha in [0,1]: learning rate for value updates
            - beta1 in [0,10]: inverse temperature at stage 1
            - beta2 in [0,10]: inverse temperature at stage 2
            - lam in [0,1]: eligibility trace strength from stage 2 PE to stage 1 MF value
            - w0 in [0,1]: baseline arbitration weight that is modulated by STAI
        Returns
        -------
        neg_log_likelihood : float
            Negative log-likelihood of observed choices under the model.
        alpha, beta1, beta2, lam, w0 = model_parameters
        n_trials = len(action_1)
        stai_val = float(np.asarray(stai)[0])
        transition_matrix = np.array([[0.7, 0.3],
                                      [0.3, 0.7]], dtype=float)
        q1_mf = np.zeros(2)           # model-free stage-1 action values (A,U)
        q2 = np.zeros((2, 2))         # stage-2 state-action values: rows X,Y; cols actions 0/1
    shared_mechanism_0()
        eps = 1e-10
        w0 = min(max(w0, 0.0), 1.0)
        w0_logit = np.log((w0 + eps) / (1.0 - (w0 + eps)))
        w_logit = w0_logit + 3.0 * (0.5 - stai_val)  # lower STAI -> larger w
        w = 1.0 / (1.0 + np.exp(-w_logit))
        for t in range(n_trials):
            s = int(state[t])
            a1 = int(action_1[t])
            a2 = int(action_2[t])
            r = float(reward[t])
            max_q2 = np.max(q2, axis=1)  # shape (2,)
            q1_mb = transition_matrix @ max_q2  # shape (2,)
            q1 = (1.0 - w) * q1_mf + w * q1_mb
            q1_shift = q1 - np.max(q1)
            exp_q1 = np.exp(beta1 * q1_shift)
            probs1 = exp_q1 / np.sum(exp_q1)
            p_choice_1[t] = probs1[a1]
            q2_s = q2[s]
            q2_shift = q2_s - np.max(q2_s)
            exp_q2 = np.exp(beta2 * q2_shift)
            probs2 = exp_q2 / np.sum(exp_q2)
            p_choice_2[t] = probs2[a2]
            delta2 = r - q2[s, a2]
            q2[s, a2] += alpha * delta2
            delta1 = q2[s, a2] - q1_mf[a1]
            q1_mf[a1] += alpha * delta1
            q1_mf[a1] += alpha * lam * delta2
        neg_log_likelihood = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[2] = participant_2

def participant_3(action_1, state, action_2, reward, stai, model_parameters):
        Mechanism overview:
        - Stage-2 values Q2(s2, a2) learned with a single learning rate.
        - Stage-1 hybrid action values combine a model-based (MB) projection using a learned
          transition model T(a1->s2) and a model-free (MF) backup from the last reached Q2.
        - A Dirichlet transition posterior is tracked from observed transitions; the entropy of T
          indexes transition uncertainty. Higher uncertainty increases MB reliance, and this
          effect is amplified by anxiety (stai).
        - A single perseveration parameter k1 acts at both stages; its effect is attenuated
          at stage-2 by anxiety (higher anxiety reduces stage-2 stickiness).
        Parameters and bounds:
        - action_1: int array (n_trials,) in {0,1}; first-stage choices (A=0, U=1)
        - state:    int array (n_trials,) in {0,1}; reached second-stage planet (X=0, Y=1)
        - action_2: int array (n_trials,) in {0,1}; second-stage alien choice
        - reward:   float array (n_trials,) in [0,1]; coins received
        - stai:     float array with single element in [0,1]; anxiety score
        - model_parameters: tuple/list with five params:
            rho_v   in [0,1]: value learning rate for Q2 and MF backup to Q1
            beta    in [0,10]: inverse temperature for softmax at both stages
            k1      in [0,1]: perseveration strength (shared); anxiety scales its stage-2 effect
            omega0  in [0,1]: baseline weight on model-based control at stage-1
            xi_unc  in [0,1]: strength of uncertainty-driven boost to MB weight
        Returns:
        - Negative log-likelihood of observed stage-1 and stage-2 choices.
        rho_v, beta, k1, omega0, xi_unc = model_parameters
        n_trials = len(action_1)
        s_anx = float(stai[0])
        q2 = np.zeros((2, 2), dtype=float)   # Q2[s2, a2]
        q1_mf = np.zeros(2, dtype=float)     # model-free Q at stage-1
        trans_counts = np.ones((2, 2), dtype=float)  # symmetric prior -> starts at 0.5/0.5
        eps = 1e-12
    shared_mechanism_0()
        prev_a1 = -1
        prev_a2 = -1
        for t in range(n_trials):
            T = trans_counts / (np.sum(trans_counts, axis=1, keepdims=True) + eps)
            ent = -np.sum(T * (np.log(T + eps)), axis=1) / np.log(2 + eps)
            unc = 0.5 * (ent[0] + ent[1])  # scalar in [0,1]
            max_q2 = np.max(q2, axis=1)        # value of each second-stage state
            q1_mb = T @ max_q2                 # MB value for each first-stage action
            omega_eff = omega0 + xi_unc * unc * (0.5 + 0.5 * s_anx)
            omega_eff = float(np.clip(omega_eff, 0.0, 1.0))
            q1 = omega_eff * q1_mb + (1.0 - omega_eff) * q1_mf
            stick1 = np.zeros(2)
            if prev_a1 in (0, 1):
                stick1[prev_a1] = 1.0
            logits1 = beta * q1 + k1 * stick1
            logits1 -= np.max(logits1)
            probs1 = np.exp(logits1) / (np.sum(np.exp(logits1)) + eps)
            a1 = int(action_1[t])
            p_choice_1[t] = probs1[a1]
            s2 = int(state[t])
            stick2 = np.zeros(2)
            if prev_a2 in (0, 1):
                stick2[prev_a2] = 1.0
            k2_eff = k1 * (1.0 - 0.7 * s_anx)  # stronger anxiety -> less stage-2 perseveration
            logits2 = beta * q2[s2] + k2_eff * stick2
            logits2 -= np.max(logits2)
            probs2 = np.exp(logits2) / (np.sum(np.exp(logits2)) + eps)
            a2 = int(action_2[t])
            p_choice_2[t] = probs2[a2]
            r = float(reward[t])
            trans_counts[a1, s2] += 1.0  # simple Bayesian counting update
            delta2 = r - q2[s2, a2]
            q2[s2, a2] += rho_v * delta2
            target1 = q2[s2, a2]  # could include immediate r via delta2 already embedded
            delta1 = target1 - q1_mf[a1]
            q1_mf[a1] += rho_v * delta1
            q1_mf[a1] += rho_v * delta2
            prev_a1 = a1
            prev_a2 = a2
        nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[3] = participant_3

def participant_4(action_1, state, action_2, reward, stai, model_parameters):
        Overview
        - Stage 2 learns Q-values via Rescorla-Wagner (model-free).
        - Stage 1 uses purely model-based planning using the known transition matrix.
        - Anxious surprise effect: effective inverse temperature decreases after rare transitions,
          scaled by stai and eta_surp. This increases stochasticity when a rare transition occurs.
          Applied to both stages on that trial.
        - Includes a perseveration (stay) bias applied to both stages, parameterized by kappa_pers.
        Parameters (all used)
        - alpha:       [0,1]   Learning rate for stage-2 Q-values.
        - beta_base:   [0,10]  Baseline inverse temperature.
        - eta_surp:    [0,1]   Magnitude by which surprise (rare transition) reduces beta.
        - kappa_pers:  [0,1]   Additive bias to repeat previous action at each stage.
                               Implemented as an additive term in the logits.
        Inputs
        - action_1: int array (n_trials,), chosen spaceship per trial (0=A, 1=U).
        - state:    int array (n_trials,), reached planet per trial (0=X, 1=Y).
        - action_2: int array (n_trials,), chosen alien per trial (0/1).
        - reward:   float array (n_trials,), coins obtained per trial in [0,1].
        - stai:     float array with one element in [0,1], anxiety score.
        - model_parameters: iterable/list/array with [alpha, beta_base, eta_surp, kappa_pers].
        Returns
        - Negative log-likelihood (float) of observed stage-1 and stage-2 choices.
        alpha, beta_base, eta_surp, kappa_pers = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        T_known = np.array([[0.7, 0.3],
                            [0.3, 0.7]], dtype=float)
        q2 = np.zeros((2, 2), dtype=float) + 0.5
        p_choice_1 = np.zeros(n_trials, dtype=float)
        p_choice_2 = np.zeros(n_trials, dtype=float)
        prev_a1 = None
        prev_a2 = np.array([None, None], dtype=object)
        for t in range(n_trials):
            a1 = int(action_1[t])
            s = int(state[t])
            a2 = int(action_2[t])
            r = float(reward[t])
            is_rare = (a1 == 0 and s == 1) or (a1 == 1 and s == 0)
            beta_eff = beta_base * (1.0 - eta_surp * stai * (1.0 if is_rare else 0.0))
            beta_eff = max(1e-6, beta_eff)
            max_q2 = np.max(q2, axis=1)  # per state
            q1_mb = T_known @ max_q2
            bias1 = np.zeros(2, dtype=float)
            if prev_a1 is not None:
                bias1[prev_a1] += kappa_pers
            bias2 = np.zeros(2, dtype=float)
            if prev_a2[s] is not None:
                bias2[int(prev_a2[s])] += kappa_pers
            logits1 = beta_eff * (q1_mb - np.max(q1_mb)) + bias1
            logits1 = logits1 - np.max(logits1)
            probs1 = np.exp(logits1) / np.sum(np.exp(logits1))
            p_choice_1[t] = probs1[a1]
            q2_s = q2[s]
            logits2 = beta_eff * (q2_s - np.max(q2_s)) + bias2
            logits2 = logits2 - np.max(logits2)
            probs2 = np.exp(logits2) / np.sum(np.exp(logits2))
            p_choice_2[t] = probs2[a2]
            pe2 = r - q2[s, a2]
            q2[s, a2] += alpha * pe2
            prev_a1 = a1
            prev_a2[s] = a2
    shared_mechanism_3()

models[4] = participant_4

def participant_5(action_1, state, action_2, reward, stai, model_parameters):
        Meta-control via anxiety-gated temperature, MB weight, and reward forgetting.
        Idea:
        - Stage 1 uses a hybrid of MB and MF, with MB weight higher when anxiety is low.
        - Softmax temperature increases when anxiety is low (more exploitation).
        - Stage 2 values decay (forget) over trials; decay grows with anxiety,
          modeling difficulty maintaining stable reward beliefs under high anxiety.
        Parameters (all in [0,1] except beta in [0,10])
        ----------
        action_1 : array-like of int {0,1}
            First-stage choices (0=spaceship A, 1=spaceship U).
        state : array-like of int {0,1}
            Reached second-stage state (0=planet X, 1=planet Y).
        action_2 : array-like of int {0,1}
            Second-stage choices (0/1=alien within the visited planet).
        reward : array-like of float
            Coins received.
        stai : array-like of float in [0,1]
            Anxiety score; higher values reduce beta (more randomness), reduce MB weight,
            and increase forgetting of stage-2 values.
        model_parameters : array-like of float
            [alpha, beta_base, forget, mbw0]
            - alpha in [0,1]: learning rate for Q updates.
            - beta_base in [0,10]: baseline inverse temperature.
            - forget in [0,1]: base forgetting factor for stage-2 values.
            - mbw0 in [0,1]: baseline MB weight at stage 1 (boosted when anxiety is low).
    shared_mechanism_1()
            Negative log-likelihood of observed choices at both stages.
        alpha, beta_base, forget, mbw0 = model_parameters
        n_trials = len(action_1)
        st = float(stai[0])
        transition_matrix = np.array([[0.7, 0.3],
                                      [0.3, 0.7]])
        q1_mf = np.zeros(2)
        q2 = 0.5 * np.ones((2, 2))
        w_mb = np.clip(mbw0 + (1.0 - st) * (1.0 - mbw0) * 0.5, 0.0, 1.0)
        beta_eff = np.clip(beta_base * (0.5 + 0.5 * (1.0 - st)), 0.0, 10.0)
        forget_eff = np.clip(forget * (0.5 + 0.5 * st), 0.0, 1.0)
        p1 = np.zeros(n_trials)
        p2 = np.zeros(n_trials)
        eps = 1e-12
        for t in range(n_trials):
            s = int(state[t])
            a1 = int(action_1[t])
            a2 = int(action_2[t])
            r = float(reward[t])
            max_q2 = np.max(q2, axis=1)
            q1_mb = transition_matrix @ max_q2
            q1 = w_mb * q1_mb + (1.0 - w_mb) * q1_mf
            logits1 = beta_eff * (q1 - np.max(q1))
            probs1 = np.exp(logits1)
            probs1 /= (np.sum(probs1) + eps)
            p1[t] = probs1[a1]
            logits2 = beta_eff * (q2[s] - np.max(q2[s]))
            probs2 = np.exp(logits2)
            probs2 /= (np.sum(probs2) + eps)
            p2[t] = probs2[a2]
            q2 = (1.0 - forget_eff) * q2 + forget_eff * 0.5
            pe2 = r - q2[s, a2]
            q2[s, a2] += alpha * pe2
            target1 = q2[s, a2]
            pe1 = target1 - q1_mf[a1]
            alpha1 = alpha * (0.8 + 0.2 * (1.0 - st))
            q1_mf[a1] += alpha1 * pe1
        neg_ll = -(np.sum(np.log(p1 + eps)) + np.sum(np.log(p2 + eps)))

models[5] = participant_5

def participant_6(action_1, state, action_2, reward, stai, model_parameters):
        Core idea:
        - Stage 1 values blend a successor-like (transition-based) planner and a model-free cache.
        - Low anxiety increases planning weight and a structure-based stay/shift bias tied to prior common/rare transitions.
        - Soft forgetting (decay) prevents overcommitment to stale values.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices (0=A, 1=U).
        state : array-like of int (0 or 1)
            Second-stage states reached (0=planet X, 1=planet Y).
        action_2 : array-like of int (0 or 1)
            Second-stage choices (0 or 1).
        reward : array-like of float
            Coins received on each trial.
        stai : array-like of float
            Anxiety score in [0,1]; use stai[0].
        model_parameters : sequence of floats
            [alpha, beta, omega, zeta, bias_prev]
            - alpha in [0,1]: learning rate for value updates (both stages).
            - beta in [0,10]: inverse temperature for both stages.
            - omega in [0,1]: base weight on transition-based planning at stage 1.
            - zeta in [0,1]: forgetting rate toward zero for unrefreshed Q-values.
            - bias_prev in [0,1]: magnitude of structure-based stay/shift bias at stage 1:
                after a common rewarded trial, bias to repeat; after a rare punished trial, bias to switch.
                Anxiety reduces the impact of this bias.
    shared_mechanism_1()
            Negative log-likelihood of the observed first- and second-stage choices.
        alpha, beta, omega, zeta, bias_prev = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        T = np.array([[0.7, 0.3],
                      [0.3, 0.7]], dtype=float)
        q1_mf = np.zeros(2)        # model-free cache for stage 1 actions
        q2 = np.zeros((2, 2))      # stage 2 action values per state
    shared_mechanism_0()
        omega_eff = np.clip(omega * (1.0 - 0.4 * stai), 0.0, 1.0)
        bias_prev_eff_base = np.clip(bias_prev * (1.0 - stai), 0.0, 1.0)  # lower with anxiety
        a1_prev = None
        s_prev = None
        r_prev = None
        for t in range(n_trials):
            max_q2_by_state = np.max(q2, axis=1)     # best attainable value on each planet
            q1_plan = T @ max_q2_by_state           # expected value of each spaceship
            pref = np.zeros(2)
            if a1_prev is not None:
                was_common = 1 if ((a1_prev == 0 and s_prev == 0) or (a1_prev == 1 and s_prev == 1)) else 0
                sign = 1.0 if r_prev > 0 else (-1.0 if r_prev < 0 else 0.0)
                bias = bias_prev_eff_base * sign * (1.0 if was_common else -1.0)
                pref[a1_prev] += bias
                pref[1 - a1_prev] -= bias
            q1 = omega_eff * q1_plan + (1.0 - omega_eff) * q1_mf + pref
            q1_shift = q1 - np.max(q1)
            probs_1 = np.exp(beta * q1_shift)
            probs_1 = probs_1 / np.sum(probs_1)
            a1 = int(action_1[t])
            p_choice_1[t] = probs_1[a1]
            s = int(state[t])
            q2_s = q2[s]
            q2_shift = q2_s - np.max(q2_s)
            probs_2 = np.exp(beta * q2_shift)
            probs_2 = probs_2 / np.sum(probs_2)
            a2 = int(action_2[t])
            p_choice_2[t] = probs_2[a2]
            r = reward[t]
            q2 *= (1.0 - zeta)
            q1_mf *= (1.0 - zeta)
            delta2 = r - q2[s, a2]
            q2[s, a2] += alpha * delta2
            target1 = q2[s, a2]
            delta1 = target1 - q1_mf[a1]
            q1_mf[a1] += alpha * delta1
            a1_prev, s_prev, r_prev = a1, s, r
        eps = 1e-10
        nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[6] = participant_6

def participant_7(action_1, state, action_2, reward, stai, model_parameters):
        Overview:
        - Stage 2: standard TD(0) learning on rewards.
        - Stage 1: mixture of model-based and model-free values.
        - Additionally, an intrinsic "surprise" signal is maintained per first-stage action:
            surprise_t = -log P(observed_state | chosen_ship) under the known transition model.
          A running trace of surprise per action S[a] is updated and added as a bonus to stage-1 values.
        - Trait anxiety scales the strength of the surprise bonus:
            z_eff = clip(z0 + k_z * stai, [0,1]).
          Higher z_eff increases preference for actions that recently produced surprising transitions.
        Parameters (model_parameters):
        - alpha: [0,1] learning rate for Q-value updates and surprise trace updates.
        - beta: [0,10] inverse temperature for both stages.
        - w: [0,1] weight on model-based control at stage 1.
        - z0: [0,1] base weight of surprise bonus.
        - k_z: [0,1] how strongly anxiety increases surprise bonus (z_eff = z0 + k_z*stai).
        Inputs:
        - action_1: array-like of ints {0,1}, first-stage choices (0=A, 1=U).
        - state: array-like of ints {0,1}, second-stage states (0=X, 1=Y).
        - action_2: array-like of ints {0,1}, second-stage choices.
        - reward: array-like of floats.
        - stai: array-like with a single float in [0,1].
        - model_parameters: tuple/list of 5 params (alpha, beta, w, z0, k_z).
        Returns:
        - Negative log-likelihood of the observed first- and second-stage choices.
        alpha, beta, w, z0, k_z = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        p_common = 0.7
        T = np.array([[p_common, 1 - p_common],
                      [1 - p_common, p_common]])
        Q2 = np.zeros((2, 2))   # stage-2
        Q1_mf = np.zeros(2)     # stage-1 MF
        S = np.zeros(2)         # surprise trace per first-stage action
    shared_mechanism_0()
        z_eff = z0 + k_z * stai
        if z_eff < 0.0:
            z_eff = 0.0
        if z_eff > 1.0:
            z_eff = 1.0
        eps = 1e-12
        for t in range(n_trials):
            max_Q2 = np.max(Q2, axis=1)
            Q1_mb = T @ max_Q2
            Q1 = w * Q1_mb + (1.0 - w) * Q1_mf + z_eff * S
            q1c = Q1 - np.max(Q1)
            probs_1 = np.exp(beta * q1c)
            probs_1 = probs_1 / np.sum(probs_1)
            a1 = action_1[t]
            p_choice_1[t] = probs_1[a1]
            s = state[t]
            q2c = Q2[s] - np.max(Q2[s])
            probs_2 = np.exp(beta * q2c)
            probs_2 = probs_2 / np.sum(probs_2)
            a2 = action_2[t]
            p_choice_2[t] = probs_2[a2]
            r = reward[t]
            delta2 = r - Q2[s, a2]
            Q2[s, a2] += alpha * delta2
            boot = Q2[s, a2]
            delta1 = boot - Q1_mf[a1]
            Q1_mf[a1] += alpha * delta1
            p_s_given_a = T[a1, s]
            if p_s_given_a < 1e-8:
                p_s_given_a = 1e-8
            surprise = -np.log(p_s_given_a)
            S[a1] += alpha * (surprise - S[a1])
        nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[7] = participant_7

def participant_8(action_1, state, action_2, reward, stai, model_parameters):
        Summary
        -------
        This model encourages sampling uncertain aliens by adding an uncertainty bonus
        to second-stage values, with leaky accumulation of successes and counts to track
        slowly drifting probabilities. Anxiety increases the directed exploration bonus and
        slightly reduces perseveration bias.
        Mechanics
        ---------
        - Maintain leaky counts N[s,a] and successes S[s,a].
        - Estimated mean: m = S / (N + tiny); uncertainty: u = 1 / (N + 1).
        - Stage 2 value: Q2 = m + phi(stai) * u, where phi increases with anxiety.
        - Stage 1 value: MB projection of max Q2 via known transitions.
        - Perseveration at stage 1 reduced by anxiety.
        - Policies: softmax with inverse temperature beta at both stages.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices (0 = A, 1 = U).
        state : array-like of int (0 or 1)
            Reached second-stage planet (0 = X, 1 = Y).
        action_2 : array-like of int (0 or 1)
            Second-stage alien choices within the reached planet.
        reward : array-like of float
            Trial outcomes (e.g., 0/1 coins).
        stai : array-like of float
            Anxiety score(s); uses stai[0] in [0,1]. Higher anxiety increases exploration bonus.
        model_parameters : iterable of floats
            [rho_decay, beta, phi0, pi0]
            - rho_decay: leaky decay factor applied each trial to counts and successes [0,1]
                         (larger means slower forgetting; 0 = full reset, 1 = no decay)
            - beta: inverse temperature for both stages [0,10]
            - phi0: baseline directed exploration bonus weight [0,1]
            - pi0: baseline perseveration bias magnitude at stage 1 [0,1]
    shared_mechanism_1()
            Negative log-likelihood of observed first- and second-stage choices.
        rho_decay, beta, phi0, pi0 = model_parameters
        n_trials = len(action_1)
        st = float(stai[0])
        T = np.array([[0.7, 0.3],
                      [0.3, 0.7]], dtype=float)
    shared_mechanism_0()
        N = np.zeros((2, 2)) + 1.0  # start with weak prior count to avoid zero
        S = np.zeros((2, 2)) + 0.5  # weak prior successes for ~0.5 mean
        prev_a1 = None
        phi = phi0 * (0.5 + st)           # higher anxiety -> larger exploration bonus
        persev = pi0 * (1.0 - 0.5 * st)   # higher anxiety -> less perseveration
        for t in range(n_trials):
            N *= rho_decay
            S *= rho_decay
            m = S / (N + 1e-8)
            u = 1.0 / (N + 1.0)
            q2 = m + phi * u
            vmax2 = np.max(q2, axis=1)
            q1 = T @ vmax2
            bias = np.zeros(2)
            if prev_a1 is not None:
                bias[prev_a1] += persev
            logits1 = q1 + bias
            logits1 = logits1 - np.max(logits1)
            probs1 = np.exp(beta * logits1)
            probs1 = probs1 / np.sum(probs1)
            a1 = action_1[t]
            p_choice_1[t] = probs1[a1]
            s2 = state[t]
            logits2 = q2[s2] - np.max(q2[s2])
            probs2 = np.exp(beta * logits2)
            probs2 = probs2 / np.sum(probs2)
            a2 = action_2[t]
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            N[s2, a2] += 1.0
            S[s2, a2] += r
            prev_a1 = a1
        eps = 1e-10
        nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[8] = participant_8

def participant_9(action_1, state, action_2, reward, stai, model_parameters):
        Core idea:
        - Pure MF Q-learning at both stages with trial-by-trial forgetting (decay).
        - A first-stage win-stay/lose-shift (WSLS) bias influences action selection:
            - After rewarded-common or unrewarded-rare transitions, bias to repeat is stronger when anxiety is low.
            - After unrewarded-common or rewarded-rare transitions, bias to switch is stronger when anxiety is high.
        - Anxiety also increases forgetting (higher decay when stai is high).
    shared_mechanism_2()
        action_1 : array-like of int {0,1}
            First-stage choices: 0=A, 1=U.
        state : array-like of int {0,1}
            Reached second-stage state: 0=X, 1=Y.
        action_2 : array-like of int {0,1}
            Second-stage choices.
        reward : array-like of float
            Rewards in [0,1].
        stai : array-like with single float in [0,1]
            Anxiety score; higher -> more forgetting and stronger lose-shift after certain outcomes.
        model_parameters : list/tuple of 5 floats
            [alpha, beta, decay, wsls_gain, xi]
            - alpha: [0,1] learning rate for both stages
            - beta: [0,10] inverse temperature (both stages)
            - decay: [0,1] baseline forgetting rate applied each trial to all Q-values
            - wsls_gain: [0,1] magnitude of WSLS bias on first-stage logits
            - xi: [0,1] mixes reward vs transition contributions to the WSLS signal
    shared_mechanism_1()
            Negative log-likelihood of observed choices.
        alpha, beta, decay, wsls_gain, xi = model_parameters
        n_trials = len(action_1)
        stai_val = float(stai[0])
        q1 = np.zeros(2)        # first-stage MF values
        q2 = np.zeros((2, 2))   # second-stage MF values
    shared_mechanism_0()
        last_a1 = None
        last_reward = 0.0
        last_is_common = 0  # 1 common, 0 rare
        for t in range(n_trials):
            a1 = action_1[t]
            s = state[t]
            a2 = action_2[t]
            r = reward[t]
            decay_eff = decay * (0.5 + 0.5 * stai_val)
            q1 *= (1.0 - decay_eff)
            q2 *= (1.0 - decay_eff)
            bias1 = np.zeros(2)
            if last_a1 is not None:
                reward_term = (2.0 * np.clip(last_reward, 0.0, 1.0)) - 1.0
                trans_term = 1.0 if last_is_common == 1 else -1.0
                base_signal = xi * reward_term + (1.0 - xi) * trans_term
                anxiety_gain = (0.5 + 0.5 * stai_val)  # scales switching on adverse signals
                signed_signal = base_signal * (1.0 - stai_val) - reward_term * (anxiety_gain - (1.0 - stai_val))
                bias_strength = wsls_gain * signed_signal
                bias1[last_a1] += bias_strength
            logits1 = beta * (q1 - np.max(q1)) + bias1
            probs1 = np.exp(logits1)
            probs1 /= np.sum(probs1)
            p_choice_1[t] = probs1[a1]
            logits2 = beta * (q2[s] - np.max(q2[s]))
            probs2 = np.exp(logits2)
            probs2 /= np.sum(probs2)
            p_choice_2[t] = probs2[a2]
            is_common = int((a1 == 0 and s == 0) or (a1 == 1 and s == 1))
            pe2 = r - q2[s, a2]
            q2[s, a2] += alpha * pe2
            target1 = q2[s, a2]
            pe1 = target1 - q1[a1]
            q1[a1] += alpha * pe1
            last_a1 = a1
            last_reward = r
            last_is_common = is_common
        eps = 1e-12

models[9] = participant_9

def participant_10(action_1, state, action_2, reward, stai, model_parameters):
        This model learns the first-stage transition structure online and arbitrates
        between model-based (MB) and model-free (MF) control at stage 1. The arbitration
        weight increases with transition certainty but is down-weighted by anxiety.
        Both stage-1 and stage-2 Q-values undergo anxiety-scaled forgetting.
    shared_mechanism_2()
        action_1 : array-like of int
            First-stage choices per trial (0 = spaceship A, 1 = spaceship U).
        state : array-like of int
            Second-stage state per trial (0 = planet X, 1 = planet Y).
        action_2 : array-like of int
            Second-stage choices per trial on the planet (0/1 for the two aliens).
        reward : array-like of float
            Obtained reward per trial (e.g., 0 or 1).
        stai : array-like of float
            Anxiety score(s). Uses stai[0] in [0,1]. Higher indicates more anxious.
        model_parameters : array-like of float
            [alpha2, beta, alpha_T, w0, decay]
            - alpha2 in [0,1]: stage-2 MF learning rate.
            - beta in [0,10]: inverse temperature for softmax at both stages.
            - alpha_T in [0,1]: learning rate for the transition matrix.
            - w0 in [0,1]: baseline arbitration weight for MB control at stage 1.
            - decay in [0,1]: base forgetting rate applied each trial.
    shared_mechanism_1()
            Negative log-likelihood of the observed choices across both stages.
        alpha2, beta, alpha_T, w0, decay = model_parameters
        n_trials = len(action_1)
        st = float(stai[0])
        T = np.full((2, 2), 0.5)
        q1_mf = np.zeros(2)        # model-free stage-1
        q2_mf = np.zeros((2, 2))   # model-free stage-2
    shared_mechanism_0()
        decay_eff = np.clip(decay * (0.5 + 0.5 * np.clip(st, 0.0, 1.0)), 0.0, 1.0)
        for t in range(n_trials):
            q1_mf = (1.0 - decay_eff) * q1_mf
            q2_mf = (1.0 - decay_eff) * q2_mf
            max_q2 = np.max(q2_mf, axis=1)      # value of each planet
            q1_mb = T @ max_q2                  # MB value of each spaceship
            c_actions = 2.0 * np.abs(T[:, 0] - 0.5)
            certainty = np.mean(c_actions)  # overall certainty in [0,1]
            w_eff = w0 * (certainty ** np.clip(1.0 - st, 0.0, 1.0))
            w_eff = np.clip(w_eff, 0.0, 1.0)
            q1 = w_eff * q1_mb + (1.0 - w_eff) * q1_mf
            q1_centered = q1 - np.max(q1)
            probs1 = np.exp(beta * q1_centered)
            probs1 = probs1 / np.sum(probs1)
            a1 = int(action_1[t])
            p_choice_1[t] = probs1[a1]
            s = int(state[t])
            q2 = q2_mf[s]
            q2_centered = q2 - np.max(q2)
            probs2 = np.exp(beta * q2_centered)
            probs2 = probs2 / np.sum(probs2)
            a2 = int(action_2[t])
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            for sp in range(2):
                target = 1.0 if sp == s else 0.0
                T[a1, sp] += alpha_T * (target - T[a1, sp])
            T[a1] = T[a1] / np.sum(T[a1])
            delta2 = r - q2_mf[s, a2]
            q2_mf[s, a2] += alpha2 * delta2
            v2 = q2_mf[s, a2]
            delta1 = v2 - q1_mf[a1]
            q1_mf[a1] += alpha2 * delta1
    shared_mechanism_3()

models[10] = participant_10

def participant_11(action_1, state, action_2, reward, stai, model_parameters):
        This model blends model-free (MF) and model-based (MB) values at stage 1, and penalizes
        first-stage actions that lead to higher transition uncertainty. Anxiety reduces the
        arbitration weight on MB control and amplifies a stickiness bias.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            Observed first-stage choices (0=A, 1=U).
        state : array-like of int (0 or 1)
            Observed second-stage state (0=X, 1=Y).
        action_2 : array-like of int (0 or 1)
            Observed second-stage choices within the encountered state.
        reward : array-like of float
            Rewards received on each trial (e.g., 0.0 or 1.0).
        stai : array-like of float
            Anxiety score in [0,1]; uses stai[0] as scalar.
        model_parameters : array-like of float
            Parameters with bounds:
            - alpha in [0,1]: learning rate for Q-value updates (both stages).
            - beta in [0,10]: inverse temperature for softmax (both stages).
            - psi in [0,1]: baseline MB arbitration weight.
            - gamma in [0,1]: weight on uncertainty penalty at stage 1.
            - kappa1 in [0,1]: baseline first-stage stickiness strength.
    shared_mechanism_1()
            Negative log-likelihood of the observed first- and second-stage choices.
        alpha, beta, psi, gamma, kappa1 = model_parameters
        n_trials = len(action_1)
        stai_val = float(stai[0])
        T = np.array([[0.7, 0.3],
                      [0.3, 0.7]])
        eps_h = 1e-12
        H = -np.sum(T * np.log(T + eps_h), axis=1)  # shape (2,)
        q1_mf = np.zeros(2)     # MF values for A,U
        q2 = np.zeros((2, 2))   # Q2[state, action]
    shared_mechanism_0()
        prev_a1 = 0
        w_eff = np.clip(psi * (1.0 - 0.6 * stai_val), 0.0, 1.0)
        kappa_eff = kappa1 * (1.0 + stai_val)
        for t in range(n_trials):
            s2 = int(state[t])
            a1 = int(action_1[t])
            a2 = int(action_2[t])
            r = float(reward[t])
            max_q2 = np.max(q2, axis=1)       # value of X and Y
            q1_mb = T @ max_q2               # MB value for A,U
            q1_hybrid = w_eff * q1_mb + (1.0 - w_eff) * q1_mf
            bias1 = np.zeros(2)
            bias1[prev_a1] += kappa_eff
            unc_penalty = gamma * stai_val * H
            logits1 = beta * q1_hybrid + bias1 - unc_penalty
            logits1 -= np.max(logits1)
            p1 = np.exp(logits1)
            p1 /= np.sum(p1)
            p_choice_1[t] = p1[a1]
            logits2 = beta * q2[s2]
            logits2 -= np.max(logits2)
            p2 = np.exp(logits2)
            p2 /= np.sum(p2)
            p_choice_2[t] = p2[a2]
            pe2 = r - q2[s2, a2]
            q2[s2, a2] += alpha * pe2
            target1 = q2[s2, a2]
            pe1 = target1 - q1_mf[a1]
            q1_mf[a1] += alpha * pe1
            prev_a1 = a1
    shared_mechanism_3()

models[11] = participant_11

def participant_12(action_1, state, action_2, reward, stai, model_parameters):
        This purely model-free controller learns second-stage values and backs them up to first-stage
        values via an eligibility trace. Anxiety (stai) creates asymmetry between positive and negative
        prediction-error learning rates and scales a perseveration (choice stickiness) bias at stage 1.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices per trial (0=A, 1=U).
        state : array-like of int (0 or 1)
            Second-stage states per trial (0=planet X, 1=planet Y).
        action_2 : array-like of int (0 or 1)
            Second-stage choices per trial (0=first alien, 1=second alien).
        reward : array-like of float (0 or 1)
            Reward outcomes per trial.
        stai : array-like of float
            Anxiety score array of length 1. Used to modulate learning-rate asymmetry and perseveration.
        model_parameters : list or array
            [alpha_base, beta, pi_base, stai_weight, lambd]
            Bounds:
              alpha_base in [0,1]  : base learning rate
              beta in [0,10]       : inverse temperature
              pi_base in [0,1]     : baseline perseveration strength
              stai_weight in [0,1] : sensitivity of parameters to STAI (0=no effect, 1=max effect)
              lambd in [0,1]       : eligibility trace from stage 2 to stage 1
    shared_mechanism_1()
            Negative log-likelihood of the observed choices under the model.
        alpha_base, beta, pi_base, stai_weight, lambd = model_parameters
        n_trials = len(action_1)
        stai = stai[0]
        q_stage1_mf = np.zeros(2)
        q_stage2_mf = np.zeros((2, 2))
    shared_mechanism_0()
        prev_a1 = None
        mod = (2.0 * stai - 1.0) * (2.0 * stai_weight - 1.0)
        alpha_pos = np.clip(alpha_base * (1.0 + mod), 0.0, 1.0)
        alpha_neg = np.clip(alpha_base * (1.0 - mod), 0.0, 1.0)
        pi_eff = pi_base * (1.0 + mod)  # can vary approximately in [0, 2*pi_base]
        for t in range(n_trials):
            s = state[t]
            a1 = action_1[t]
            a2 = action_2[t]
            q1 = q_stage1_mf.copy()
            if prev_a1 is not None:
                stick = np.zeros(2)
                stick[prev_a1] = 1.0
                q1 = q1 + pi_eff * stick
            q1_centered = q1 - np.max(q1)
            exp_q1 = np.exp(beta * q1_centered)
            probs_1 = exp_q1 / np.sum(exp_q1)
            p_choice_1[t] = probs_1[a1]
            q2_s = q_stage2_mf[s].copy()
            q2_centered = q2_s - np.max(q2_s)
            exp_q2 = np.exp(beta * q2_centered)
            probs_2 = exp_q2 / np.sum(exp_q2)
            p_choice_2[t] = probs_2[a2]
            pe2 = reward[t] - q_stage2_mf[s, a2]
            if pe2 >= 0:
                a2_lr = alpha_pos
            else:
                a2_lr = alpha_neg
            q_stage2_mf[s, a2] += a2_lr * pe2
            q_stage1_mf[a1] += a2_lr * lambd * pe2
            td1 = q_stage2_mf[s, a2] - q_stage1_mf[a1]
            alpha1 = np.clip(alpha_base * (1.0 + 0.5 * mod), 0.0, 1.0)
            q_stage1_mf[a1] += alpha1 * td1
            prev_a1 = a1
    shared_mechanism_3()

models[12] = participant_12

def participant_13(action_1, state, action_2, reward, stai, model_parameters):
        This model learns second-stage Q-values (per planet/alien) and a model-free first-stage value.
        First-stage choices are driven by a convex combination of model-based and model-free values,
        where the arbitration weight is modulated by the participant's anxiety (STAI). A perseveration
        bias at the first stage is also scaled by anxiety.
        Parameters (model_parameters):
        - alpha in [0,1]: Learning rate for value updates at both stages.
        - beta in [0,10]: Inverse temperature for softmax choice at both stages.
        - lam in [0,1]: Eligibility trace strength propagating second-stage RPE to first stage.
        - w_mb in [0,1]: Baseline weight for model-based values at first stage (before anxiety modulation).
        - stickiness in [0,1]: Strength of first-stage perseveration bias (scaled by STAI).
        Inputs:
        - action_1: array of length n_trials with first-stage actions (0 = A, 1 = U).
        - state: array of length n_trials with observed second-stage state (0 = X, 1 = Y).
        - action_2: array of length n_trials with second-stage actions (0/1 for the two aliens on that planet).
        - reward: array of length n_trials with received reward (e.g., 0 or 1).
        - stai: array-like with a single float in [0,1], the participant's anxiety score.
        - model_parameters: array-like with the parameters [alpha, beta, lam, w_mb, stickiness].
        Returns:
        - Negative log-likelihood of the observed first- and second-stage choices under the model.
        alpha, beta, lam, w_mb, stickiness = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        transition_matrix = np.array([[0.7, 0.3],
                                      [0.3, 0.7]])
        q_stage2 = np.zeros((2, 2))      # Q2[state, action2]
        q_stage1_mf = np.zeros(2)        # model-free Q at stage 1
    shared_mechanism_0()
        eps = 1e-10
        w_mb = np.clip(w_mb, eps, 1 - eps)
        logit = np.log(w_mb) - np.log(1 - w_mb)
        w_eff = 1 / (1 + np.exp(-(logit + 2.0 * (0.5 - stai))))  # increases when stai < 0.5, decreases when stai > 0.5
        w_eff = float(np.clip(w_eff, 0.0, 1.0))
        prev_a1 = -1  # for stickiness
        for t in range(n_trials):
            s = int(state[t])
            a1 = int(action_1[t])
            a2 = int(action_2[t])
            r = float(reward[t])
            max_q2 = np.max(q_stage2, axis=1)                  # best alien per planet
            q1_mb = transition_matrix @ max_q2                 # expected value per spaceship
            q1 = w_eff * q1_mb + (1 - w_eff) * q_stage1_mf     # arbitration
            if prev_a1 >= 0:
                kappa_eff = stickiness * (0.5 + stai)          # stronger bias with higher anxiety
                bias = np.array([0.0, 0.0])
                bias[prev_a1] += kappa_eff
                q1 = q1 + bias
            exp_q1 = np.exp(beta * (q1 - np.max(q1)))
            probs_1 = exp_q1 / (np.sum(exp_q1) + eps)
            p_choice_1[t] = probs_1[a1]
            q2_s = q_stage2[s]
            exp_q2 = np.exp(beta * (q2_s - np.max(q2_s)))
            probs_2 = exp_q2 / (np.sum(exp_q2) + eps)
            p_choice_2[t] = probs_2[a2]
            delta2 = r - q_stage2[s, a2]
            q_stage2[s, a2] += alpha * delta2
            td_to_stage1 = (q_stage2[s, a2] - q_stage1_mf[a1])  # bootstrapping from stage 2
            q_stage1_mf[a1] += alpha * td_to_stage1 + alpha * lam * delta2
            prev_a1 = a1
        eps = 1e-10
        nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[13] = participant_13

def participant_14(action_1, state, action_2, reward, stai, model_parameters):
        This model learns the transition matrix from experience and uses a purely
        model-based (MB) first-stage policy computed from the learned transition and
        second-stage Q-values. Stage 2 uses MF Q-learning.
        Anxiety (stai) amplifies a perseveration bias to repeat the previous
        spaceship choice, capturing heightened habit-like inertia under higher
        anxiety while still planning via MB values.
        Parameters (model_parameters):
        - alpha_r: reward learning rate for Q2 updates, in [0,1]
        - alpha_t: transition learning rate, in [0,1]
        - beta: inverse temperature for both stages, in [0,10]
        - stick: base perseveration strength added to stage-1 logits, in [0,1]
        - gamma_anx: anxiety gain on perseveration, in [0,1]
        Inputs:
        - action_1: array of first-stage choices (0: A, 1: U)
        - state: array of reached planets (0: X, 1: Y)
        - action_2: array of second-stage choices (0/1)
        - reward: array of rewards per trial
        - stai: array-like with single float (0-1) anxiety score
        - model_parameters: array-like of parameters as above
        Returns:
        - Negative log-likelihood of observed choices across both stages.
        alpha_r, alpha_t, beta, stick, gamma_anx = model_parameters
        n_trials = len(action_1)
        stai = stai[0]
        T = np.ones((2, 2)) * 0.5  # rows: spaceships [A,U], cols: planets [X,Y]
        q2 = np.zeros((2, 2))      # second-stage values per planet-alien
    shared_mechanism_0()
        prev_a1 = None
        stick_eff = stick * np.clip(1.0 + gamma_anx * (stai - 0.31), 0.0, 2.0)
        for t in range(n_trials):
            max_q2 = np.max(q2, axis=1)  # best alien per planet
            q1_mb = T @ max_q2
            bias = np.zeros(2)
            if prev_a1 is not None:
                bias[prev_a1] = stick_eff
            logits1 = beta * q1_mb + bias
            logits1 -= np.max(logits1)
            exp1 = np.exp(logits1)
            probs1 = exp1 / (np.sum(exp1) + 1e-12)
            a1 = action_1[t]
            p_choice_1[t] = probs1[a1]
            s = state[t]
            logits2 = beta * q2[s]
            logits2 -= np.max(logits2)
            exp2 = np.exp(logits2)
            probs2 = exp2 / (np.sum(exp2) + 1e-12)
            a2 = action_2[t]
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            delta2 = r - q2[s, a2]
            q2[s, a2] += alpha_r * delta2
            target = np.array([0.0, 0.0])
            target[s] = 1.0
            T[a1] += alpha_t * (target - T[a1])
            T[a1] = np.clip(T[a1], 1e-6, 1.0)
            T[a1] /= np.sum(T[a1])
            prev_a1 = a1
        eps = 1e-10
        nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[14] = participant_14

def participant_15(action_1, state, action_2, reward, stai, model_parameters):
        Core ideas:
        - The agent learns transition probabilities online (alpha_t) with forgetting (f_forget).
        - Stage-1 uses a hybrid of learned MB values and Stage-1 MF values, with MB weight higher when
          anxiety is low: w_mb = 0.5 + 0.5*(1 - stai).
        - An anxiety-weighted safety bias penalizes actions with higher transition entropy (uncertainty),
          capturing preference for predictable outcomes under anxiety.
        - Stage-2 values are learned model-free.
        Parameters (all used; total=5):
        - eta: [0,1] Learning rate for Stage-2 MF values and Stage-1 MF bootstrapping.
        - beta: [0,10] Inverse temperature for both stages.
        - alpha_t: [0,1] Learning rate for updating the transition matrix rows upon observing transitions.
        - f_forget: [0,1] Per-trial forgetting toward uniform for transition probabilities.
        - omega_safe: [0,1] Strength of safety bias; effective bias scales with stai.
        Inputs:
        - action_1: int array of shape (T,), first-stage choices in {0,1} (0=A, 1=U).
        - state: int array of shape (T,), second-stage planet in {0,1}.
        - action_2: int array of shape (T,), second-stage choices in {0,1}.
        - reward: float array of shape (T,), rewards in [0,1].
        - stai: array-like with a single float in [0,1], anxiety score.
        - model_parameters: iterable [eta, beta, alpha_t, f_forget, omega_safe].
        Returns:
        - Negative log-likelihood of observed Stage-1 and Stage-2 choices.
        eta, beta, alpha_t, f_forget, omega_safe = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        T_learn = np.array([[0.6, 0.4],
                            [0.4, 0.6]], dtype=float)
    shared_mechanism_0()
        q2 = 0.5 * np.ones((2, 2))
        q1_mf = np.zeros(2)
        w_mb = 0.5 + 0.5 * (1.0 - stai)          # in [0.5,1.0]; more anxiety => closer to 0.5
        safety_scale = omega_safe * stai          # stronger bias with higher anxiety
        for t in range(n_trials):
            s = state[t]
            max_q2 = np.max(q2, axis=1)
            q1_mb = T_learn @ max_q2
            ent = np.zeros(2)
            for a in range(2):
                p = np.clip(T_learn[a, 0], 1e-8, 1.0 - 1e-8)
                q = 1.0 - p
                H = -(p * np.log(p) + q * np.log(q))
                ent[a] = H / np.log(2.0)
            bias = -safety_scale * ent  # penalize uncertainty
            q1 = w_mb * q1_mb + (1.0 - w_mb) * q1_mf + bias
            z1 = beta * (q1 - np.max(q1))
            probs1 = np.exp(z1)
            probs1 /= np.sum(probs1)
            a1 = action_1[t]
            p_choice_1[t] = probs1[a1]
            z2 = beta * (q2[s] - np.max(q2[s]))
            probs2 = np.exp(z2)
            probs2 /= np.sum(probs2)
            a2 = action_2[t]
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            pe2 = r - q2[s, a2]
            q2[s, a2] += eta * pe2
            target1 = q2[s, a2]
            pe1 = target1 - q1_mf[a1]
            q1_mf[a1] += eta * pe1
            T_learn = (1.0 - f_forget) * T_learn + f_forget * 0.5
            T_learn[a1, s] += alpha_t * (1.0 - T_learn[a1, s])
            other = 1 - s
            T_learn[a1, other] += alpha_t * (0.0 - T_learn[a1, other])
            for a in range(2):
                row = T_learn[a]
                row = np.clip(row, 1e-8, 1.0)
                T_learn[a] = row / np.sum(row)
    shared_mechanism_3()

models[15] = participant_15

def participant_16(action_1, state, action_2, reward, stai, model_parameters):
        This model combines:
          - Model-free updates at stage 2 with TD learning.
          - An eligibility trace from stage 2 to stage 1 MF values, scaled up by anxiety.
          - A model-based (MB) planner using the known transition matrix.
          - Perseveration (stickiness) at stage 1 that increases with anxiety.
          - A transition-outcome bias that promotes stay after common+reward or rare+no-reward,
            and switch after common+no-reward or rare+reward; its strength increases with anxiety.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices (0=A, 1=U).
        state : array-like of int (0 or 1)
            Reached state (0=X, 1=Y).
        action_2 : array-like of int (0 or 1)
            Second-stage choices (0 or 1 within the reached state).
        reward : array-like of float in [0,1]
            Received rewards.
        stai : array-like with a single float in [0,1]
            Anxiety score. Higher values increase eligibility trace, perseveration, and transition-outcome bias,
            and reduce reliance on model-based planning.
        model_parameters : iterable of 5 floats
            - eta in [0,1]: learning rate for MF action values (stage 2) and eligibility at stage 1.
            - b1 in [0,10]: inverse temperature at stage 1.
            - b2 in [0,10]: inverse temperature at stage 2.
            - pi_base in [0,1]: baseline perseveration strength at stage 1.
            - chi_base in [0,1]: baseline weight for transition-outcome interaction (stay/switch bias).
    shared_mechanism_1()
            Negative log-likelihood of observed choices.
        eta, b1, b2, pi_base, chi_base = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        T = np.array([[0.7, 0.3],  # A -> (X,Y)
                      [0.3, 0.7]]) # U -> (X,Y)
        q2 = np.zeros((2, 2))       # stage-2 MF values per state and action
        q1_mf = np.zeros(2)         # stage-1 MF cached values
        lam = np.clip(0.4 + 0.5 * stai, 0.0, 1.0)                 # eligibility trace
        pi = np.clip(pi_base * (1.0 + 0.8 * stai), 0.0, 5.0)      # perseveration bias added to previous action logit
        chi = np.clip(chi_base * (0.5 + 0.5 * stai), 0.0, 5.0)    # transition-outcome interaction weight
        w_mb = np.clip(0.7 - 0.4 * stai, 0.0, 1.0)                # anxiety reduces MB reliance
    shared_mechanism_0()
        prev_a1 = None
        prev_sigma = 0.0  # transition-outcome signal from previous trial
        for t in range(n_trials):
            max_q2 = np.max(q2, axis=1)         # value per state
            q1_mb = T @ max_q2                  # plan with transition structure
            q1 = (1.0 - w_mb) * q1_mf + w_mb * q1_mb
            logits1 = b1 * q1
            if prev_a1 is not None:
                logits1[prev_a1] += pi
                logits1[prev_a1] += chi * prev_sigma
            maxl1 = np.max(logits1)
            probs1 = np.exp(logits1 - maxl1)
            probs1 = probs1 / np.sum(probs1)
            a1 = action_1[t]
            p_choice_1[t] = probs1[a1]
            s = state[t]
            logits2 = b2 * q2[s]
            maxl2 = np.max(logits2)
            probs2 = np.exp(logits2 - maxl2)
            probs2 = probs2 / np.sum(probs2)
            a2 = action_2[t]
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            delta2 = r - q2[s, a2]
            q2[s, a2] += eta * delta2
            q1_mf[a1] += eta * lam * delta2
            common = int((a1 == 0 and s == 0) or (a1 == 1 and s == 1))
            sigma = 1 if (common and r > 0.5) or ((1 - common) and r <= 0.5) else -1
            prev_sigma = float(sigma)
            prev_a1 = a1
    shared_mechanism_3()

models[16] = participant_16

def participant_17(action_1, state, action_2, reward, stai, model_parameters):
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices (spaceship A=0, U=1) for each trial.
        state : array-like of int (0 or 1)
            Second-stage state (planet X=0, Y=1) reached on each trial.
        action_2 : array-like of int (0 or 1)
            Second-stage choices (alien indices W/P=0, S/H=1) for each trial.
        reward : array-like of float
            Reward (coins) on each trial, typically in {0,1}.
        stai : array-like of float in [0,1]
            Anxiety score (single value array). Used to modulate MB weighting, forgetfulness, and perseveration.
        model_parameters : tuple/list
            Parameters (all in [0,1] except beta in [0,10]):
            - alpha: learning rate for value updates in [0,1]
            - beta: inverse temperature for softmax in [0,10]
            - w_slope: base MB mixing weight; anxiety shifts toward (1 - w_slope) in [0,1]
                       Effective w_mb = (1 - stai)*w_slope + stai*(1 - w_slope)
            - rho_forget: baseline forgetting/decay strength in [0,1]
                          Effective decay factor each trial: decay = 1 - rho_forget * stai
            - tau_stay: state-conditional perseveration strength in [0,1]
                        Bias to repeat previous second-stage action in same state scales with stai
    shared_mechanism_1()
            Negative log-likelihood of observed choices.
        alpha, beta, w_slope, rho_forget, tau_stay = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        T = np.array([[0.7, 0.3],
                      [0.3, 0.7]])
        w_mb = (1.0 - stai) * w_slope + stai * (1.0 - w_slope)
        w_mb = 0.0 if w_mb < 0.0 else (1.0 if w_mb > 1.0 else w_mb)
    shared_mechanism_0()
        q1_mf = np.zeros(2)        # stage-1 model-free
        q2 = np.zeros((2, 2))      # stage-2 action values
        prev_a2 = None
        prev_s = None
        decay = 1.0 - rho_forget * stai
        if decay < 0.0: decay = 0.0
        if decay > 1.0: decay = 1.0
        for t in range(n_trials):
            max_q2 = np.max(q2, axis=1)           # best action per state
            q1_mb = T @ max_q2                    # expected value via transition model
            q1 = (1.0 - w_mb) * q1_mf + w_mb * q1_mb
            z1 = q1 - np.max(q1)
            exp1 = np.exp(beta * z1)
            probs1 = exp1 / np.sum(exp1)
            a1 = action_1[t]
            p_choice_1[t] = probs1[a1]
            s = state[t]
            q2_s = q2[s].copy()
            bias2 = np.zeros(2)
            if prev_a2 is not None and prev_s == s:
                bias2[prev_a2] += tau_stay * stai  # stronger stickiness under higher anxiety
            z2 = (q2_s + bias2) - np.max(q2_s + bias2)
            exp2 = np.exp(beta * z2)
            probs2 = exp2 / np.sum(exp2)
            a2 = action_2[t]
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            q1_mf *= decay
            q2 *= decay
            delta2 = r - q2[s, a2]
            q2[s, a2] += alpha * delta2
            delta1 = q2[s, a2] - q1_mf[a1]
            q1_mf[a1] += alpha * delta1
            prev_a2 = a2
            prev_s = s
        eps = 1e-10
        nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[17] = participant_17

def participant_18(action_1, state, action_2, reward, stai, model_parameters):
        Description:
        - Anxiety reduces choice precision via an adaptive inverse temperature:
          beta_eff = beta * (1 - k_temp * stai), bounded below by a small floor.
        - Stage-1 includes a perseveration bias that decays over trials; the bias strength is
          reduced by anxiety.
        - Stage-2 is MF; both stage-2 values and perseveration traces are subject to mild forgetting
          toward zero (recency), controlled by tau_forget.
        Parameters (model_parameters):
        - alpha: reward learning rate in [0,1]
        - beta: base inverse temperature in [0,10]
        - k_temp: anxiety sensitivity of temperature in [0,1] (higher -> more reduction in beta)
        - tau_forget: forgetting/decay parameter in [0,1] applied to Q2 and perseveration traces
        - pers: baseline perseveration strength in [0,1]
        Inputs:
        - action_1: array-like of ints in {0,1}; chosen spaceship (0=A, 1=U)
        - state: array-like of ints in {0,1}; observed planet (0=X, 1=Y)
        - action_2: array-like of ints in {0,1}; chosen alien on the observed planet
        - reward: array-like of floats in [0,1]; received coins
        - stai: array-like with one float in [0,1]; anxiety score
        - model_parameters: tuple/list (alpha, beta, k_temp, tau_forget, pers)
        Returns:
        - Negative log-likelihood of observed choices across both stages.
        alpha, beta, k_temp, tau_forget, pers = model_parameters
        n_trials = len(action_1)
        st = float(stai[0])
        T = np.array([[0.7, 0.3],
                      [0.3, 0.7]], dtype=float)
        Q1_mf = np.zeros(2)
        Q2 = np.zeros((2, 2))
        trace1 = np.zeros(2)
        beta_eff = max(1e-3, beta * (1.0 - k_temp * st))  # higher anxiety -> lower precision
        stick_strength = pers * (1.0 - st)                # higher anxiety -> weaker perseveration
        decay = np.clip(tau_forget, 0.0, 1.0)             # used for Q2 and trace forgetting
    shared_mechanism_0()
        eps = 1e-10
        for t in range(n_trials):
            s = int(state[t])
            a1 = int(action_1[t])
            a2 = int(action_2[t])
            r = float(reward[t])
            max_Q2 = np.max(Q2, axis=1)
            Q1_mb = T @ max_Q2
            pref1 = 0.5 * Q1_mf + 0.5 * Q1_mb + stick_strength * trace1
            centered1 = pref1 - np.max(pref1)
            exp1 = np.exp(beta_eff * centered1)
            probs1 = exp1 / (np.sum(exp1) + eps)
            p_choice_1[t] = probs1[a1]
            pref2 = Q2[s]
            centered2 = pref2 - np.max(pref2)
            exp2 = np.exp(beta_eff * centered2)
            probs2 = exp2 / (np.sum(exp2) + eps)
            p_choice_2[t] = probs2[a2]
            Q2 *= (1.0 - decay)
            delta2 = r - Q2[s, a2]
            Q2[s, a2] += alpha * delta2
            delta1 = Q2[s, a2] - Q1_mf[a1]
            Q1_mf[a1] += alpha * delta1
            trace1 *= (1.0 - decay)
            trace1[a1] += 1.0
        nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[18] = participant_18

def participant_19(action_1, state, action_2, reward, stai, model_parameters):
        Parameters (model_parameters):
          - alpha: learning rate for Q-value updates at both stages, in [0,1]
          - beta: inverse temperature for both stages, in [0,10]
          - lambda_loss: base loss-aversion coefficient, in [0,1] (effective loss-aversion increases with anxiety)
          - kappa_anx: strength of anxiety-driven pessimism in model-based lookahead, in [0,1]
          - phi_forget: forgetting rate for unchosen actions (value decay), in [0,1]
        Inputs:
          - action_1: array of first-stage choices (0=A, 1=U)
          - state: array of second-stage states (0=X, 1=Y)
          - action_2: array of second-stage choices (0 or 1; e.g., alien index on the planet)
          - reward: array of scalar rewards (can be negative or positive)
          - stai: array-like with a single anxiety score in [0,1]
          - model_parameters: list/tuple as described above
        Returns:
          - Negative log-likelihood of the observed first- and second-stage choices.
        Model summary:
          - Second-stage learning is model-free with loss-averse utility u(r).
          - First-stage decision uses a convex combination of:
              (i) model-free Q1 values bootstrapped from second-stage MF values via eligibility,
              (ii) a pessimistic model-based lookahead over the transition matrix.
            Anxiety increases loss aversion and pessimism, and also increases forgetting of unchosen actions.
        alpha, beta, lambda_loss, kappa_anx, phi_forget = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        T = np.array([[0.7, 0.3],
                      [0.3, 0.7]])
        Q1_mf = np.zeros(2)        # model-free values for first-stage actions
        Q2 = np.zeros((2, 2))      # second-stage state-action values
    shared_mechanism_0()
        lambda_eff = lambda_loss * (1.0 + stai)
        xi = np.clip(1.0 - kappa_anx * stai, 0.0, 1.0)
        forget = np.clip(phi_forget * (0.5 + stai), 0.0, 1.0)
        eps = 1e-12
        for t in range(n_trials):
            vmax = np.max(Q2, axis=1)   # per state
            vmin = np.min(Q2, axis=1)   # per state
            V_state = xi * vmax + (1.0 - xi) * vmin  # pessimism-weighted value per state
            Q1_mb_pess = T @ V_state  # model-based action values under pessimistic evaluation
            w_eff = np.clip(0.5 + 0.5 * stai * (1.0 - lambda_loss), 0.0, 1.0)
            Q1 = w_eff * Q1_mb_pess + (1.0 - w_eff) * Q1_mf
            q1c = Q1 - np.max(Q1)
            probs_1 = np.exp(beta * q1c)
            probs_1 = probs_1 / (np.sum(probs_1) + eps)
            a1 = int(action_1[t])
            p_choice_1[t] = probs_1[a1]
            s = int(state[t])
            q2c = Q2[s, :] - np.max(Q2[s, :])
            probs_2 = np.exp(beta * q2c)
            probs_2 = probs_2 / (np.sum(probs_2) + eps)
            a2 = int(action_2[t])
            p_choice_2[t] = probs_2[a2]
            r = float(reward[t])
            if r >= 0:
                u = r
            else:
                u = - (1.0 + lambda_eff) * (-r)
            delta2 = u - Q2[s, a2]
            Q2[s, a2] += alpha * delta2
            other_a2 = 1 - a2
            Q2[s, other_a2] *= (1.0 - forget)
            td1 = Q2[s, a2] - Q1_mf[a1]
            Q1_mf[a1] += alpha * td1
            other_a1 = 1 - a1
            Q1_mf[other_a1] *= (1.0 - forget)
        nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[19] = participant_19

def participant_20(action_1, state, action_2, reward, stai, model_parameters):
    shared_mechanism_2()
        action_1 : 1D array-like of int (0 or 1)
            First-stage choices (spaceship A=0, U=1) per trial.
        state : 1D array-like of int (0 or 1)
            Second-stage state (planet X=0, Y=1) reached on each trial.
        action_2 : 1D array-like of int (0 or 1)
            Second-stage choices (alien index within the reached planet) per trial.
        reward : 1D array-like of float (0 or 1)
            Obtained coins per trial.
        stai : 1D array-like of float in [0,1]
            Participant STAI score; higher means higher anxiety. Used to modulate arbitration and stickiness.
        model_parameters : iterable of floats
            Parameters (bounds):
              - alpha in [0,1]: learning rate for action values
              - beta in [0,10]: inverse temperature for softmax
              - w0 in [0,1]: base model-based weight (anxiety reduces it)
              - lam in [0,1]: eligibility trace mixing second-stage PE into first-stage MF update
              - pers in [0,1]: base first-stage choice stickiness strength (scaled up by anxiety)
    shared_mechanism_1()
            Negative log-likelihood of the observed first- and second-stage choices.
        alpha, beta, w0, lam, pers = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    shared_mechanism_0()
        q_stage1_mf = np.zeros(2)          # model-free first-stage Q
        q_stage2_mf = np.zeros((2, 2))     # model-free second-stage Q per state
        prev_choice1 = None                # for stickiness
        for t in range(n_trials):
            max_q_stage2 = np.max(q_stage2_mf, axis=1)     # best alien per planet
            q_stage1_mb = transition_matrix @ max_q_stage2
            w_eff = np.clip(w0 * (1.0 - 0.6 * stai), 0.0, 1.0)
            kappa = pers * (0.5 + stai)  # stickiness scales up with anxiety
            q1_combined = w_eff * q_stage1_mb + (1.0 - w_eff) * q_stage1_mf
            pref1 = beta * q1_combined
            if prev_choice1 is not None:
                stick = np.zeros(2)
                stick[prev_choice1] = 1.0
                pref1 = pref1 + kappa * stick
            exp_q1 = np.exp(pref1 - np.max(pref1))
            probs_1 = exp_q1 / np.sum(exp_q1)
            a1 = action_1[t]
            p_choice_1[t] = probs_1[a1]
            s = state[t]
            pref2 = beta * q_stage2_mf[s]
            exp_q2 = np.exp(pref2 - np.max(pref2))
            probs_2 = exp_q2 / np.sum(exp_q2)
            a2 = action_2[t]
            p_choice_2[t] = probs_2[a2]
            delta1 = q_stage2_mf[s, a2] - q_stage1_mf[a1]          # bootstrapped MF TD error at stage 1
            delta2 = reward[t] - q_stage2_mf[s, a2]                # reward PE at stage 2
            q_stage2_mf[s, a2] += alpha * delta2
            q_stage1_mf[a1] += alpha * (delta1 + lam * delta2)
            prev_choice1 = a1
    shared_mechanism_3()

models[20] = participant_20

def participant_21(action_1, state, action_2, reward, stai, model_parameters):
        Overview:
        - Stage-2 action values are learned via incremental reward prediction errors.
        - Stage-1 uses a hybrid of model-based (via known transitions) and model-free values.
        - The model-based weight is modulated by anxiety (stai).
        - Perseveration (choice stickiness) acts at both stages and is weaker with higher anxiety.
        Parameters (bounds):
        - model_parameters[0] = alpha (0 to 1): learning rate for stage-2 values and stage-1 bootstrapping
        - model_parameters[1] = beta (0 to 10): inverse temperature for softmax at both stages
        - model_parameters[2] = omega0 (0 to 1): baseline weight on model-based value at stage 1
        - model_parameters[3] = anx_mod (0 to 1): strength of anxiety modulation on model-based weight
           Effective MB weight: omega_eff = clip(omega0 + anx_mod * (0.5 - stai), 0, 1).
           Higher anxiety (stai>0.5) reduces model-based weight.
        - model_parameters[4] = pi (0 to 1): baseline perseveration magnitude; applied as pi*(1 - stai)
        Inputs:
        - action_1: array-like of ints in {0,1}, chosen spaceship (0=A, 1=U) per trial
        - state: array-like of ints in {0,1}, reached planet (0=X, 1=Y) per trial
        - action_2: array-like of ints in {0,1}, chosen alien on reached planet per trial
        - reward: array-like of floats, received coins per trial
        - stai: array-like with one float in [0,1], anxiety score
        - model_parameters: list/array of 5 parameters as specified above
        Returns:
        - Negative log-likelihood of the observed sequence of choices at both stages.
        alpha, beta, omega0, anx_mod, pi = model_parameters
        n_trials = len(action_1)
        stai_val = float(stai[0])
        transition_matrix = np.array([[0.7, 0.3],
                                      [0.3, 0.7]])
        q1_mf = np.zeros(2)        # model-free stage-1 values
        q2 = np.zeros((2, 2))      # stage-2 values: states x actions
    shared_mechanism_0()
        prev_a1 = None
        prev_a2_by_state = [None, None]
        omega_eff = np.clip(omega0 + anx_mod * (0.5 - stai_val), 0.0, 1.0)
        stick_strength = pi * (1.0 - stai_val)  # less stickiness with higher anxiety
        for t in range(n_trials):
            max_q2 = np.max(q2, axis=1)                  # shape (2,)
            q1_mb = transition_matrix @ max_q2           # shape (2,)
            bias1 = np.zeros(2)
            if prev_a1 is not None:
                bias1[prev_a1] += stick_strength
            q1_net = omega_eff * q1_mb + (1.0 - omega_eff) * q1_mf + bias1
            q1c = q1_net - np.max(q1_net)
            probs_1 = np.exp(beta * q1c)
            probs_1 = probs_1 / np.sum(probs_1)
            a1 = action_1[t]
            p_choice_1[t] = probs_1[a1]
            s = state[t]
            bias2 = np.zeros(2)
            if prev_a2_by_state[s] is not None:
                bias2[prev_a2_by_state[s]] += stick_strength
            q2_net = q2[s] + bias2
            q2c = q2_net - np.max(q2_net)
            probs_2 = np.exp(beta * q2c)
            probs_2 = probs_2 / np.sum(probs_2)
            a2 = action_2[t]
            p_choice_2[t] = probs_2[a2]
            r = reward[t]
            delta2 = r - q2[s, a2]
            q2[s, a2] += alpha * delta2
            delta1 = q2[s, a2] - q1_mf[a1]
            q1_mf[a1] += alpha * delta1
            prev_a1 = a1
            prev_a2_by_state[s] = a2
    shared_mechanism_3()

models[21] = participant_21

def participant_22(action_1, state, action_2, reward, stai, model_parameters):
        This model blends model-based (MB) and model-free (MF) values at stage 1.
        The arbitration weight w_t is shaped by:
        - baseline w0,
        - state of second-stage uncertainty (smaller alien value differences => higher MB weight),
        - anxiety (higher anxiety reduces MB reliance).
        Stage-2 uses MF Q-learning. MF values decay for unchosen actions to capture forgetting.
        Parameters (model_parameters):
        - alpha: [0,1] learning rate for MF updates at stage 2 and bootstrapped update to stage 1.
        - beta: [0,10] inverse temperature for both stages.
        - w0: [0,1] baseline MB weight at stage 1.
        - k_unc: [0,1] strength of uncertainty-driven increase in MB weight.
        - k_decay: [0,1] decay/forgetting of unupdated MF action values per trial.
        Inputs:
        - action_1: int array in {0,1}, chosen spaceship per trial.
        - state: int array in {0,1}, reached planet per trial.
        - action_2: int array in {0,1}, chosen alien per trial.
        - reward: float array, coins received per trial.
        - stai: array-like (length 1), anxiety score in [0,1].
        - model_parameters: list/array [alpha, beta, w0, k_unc, k_decay].
        Returns:
        - Negative log-likelihood of observed choices at both stages.
        alpha, beta, w0, k_unc, k_decay = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        T = np.array([[0.7, 0.3],  # action 0 (A): P(X)=0.7, P(Y)=0.3
                      [0.3, 0.7]]) # action 1 (U): P(X)=0.3, P(Y)=0.7
        q1_mf = np.zeros(2)
        q2 = np.zeros((2, 2))  # state x action
        p1 = np.zeros(n_trials)
        p2 = np.zeros(n_trials)
        for t in range(n_trials):
            s = int(state[t])
            a1 = int(action_1[t])
            a2 = int(action_2[t])
            r = reward[t]
            diff_x = abs(q2[0, 0] - q2[0, 1])
            diff_y = abs(q2[1, 0] - q2[1, 1])
            unc = 1.0 - 0.5 * (np.tanh(diff_x) + np.tanh(diff_y))  # in (0,1), saturating
            w = w0 * (1.0 - 0.6 * stai) + k_unc * unc
            w = min(1.0, max(0.0, w))
            max_q2 = np.max(q2, axis=1)  # [X_best, Y_best]
            q1_mb = T @ max_q2
            q1_hybrid = (1.0 - w) * q1_mf + w * q1_mb
            logits1 = beta * q1_hybrid
            logits1 -= np.max(logits1)
            probs1 = np.exp(logits1)
            probs1 /= (np.sum(probs1) + 1e-16)
            p1[t] = probs1[a1]
            logits2 = beta * q2[s, :]
            logits2 -= np.max(logits2)
            probs2 = np.exp(logits2)
            probs2 /= (np.sum(probs2) + 1e-16)
            p2[t] = probs2[a2]
            delta2 = r - q2[s, a2]
            q2[s, a2] += alpha * delta2
            target1 = q2[s, a2]
            delta1 = target1 - q1_mf[a1]
            q1_mf[a1] += alpha * delta1
            other_a1 = 1 - a1
            q1_mf[other_a1] *= (1.0 - k_decay)
            other_a2 = 1 - a2
            q2[s, other_a2] *= (1.0 - k_decay)
            other_s = 1 - s
            q2[other_s, :] *= (1.0 - 0.5 * k_decay)
        eps = 1e-12
        nll = -(np.sum(np.log(p1 + eps)) + np.sum(np.log(p2 + eps)))

models[22] = participant_22

def participant_23(action_1, state, action_2, reward, stai, model_parameters):
        Idea
        ----
        The agent learns the stage-1 transition probabilities from experience via
        simple Dirichlet counts. Stage-2 action selection includes an uncertainty
        bonus (UCB-style) based on visit counts, with anxiety reducing exploratory
        bonus. First-stage action values combine learned model-based values and
        model-free values; anxiety reduces the model-based blend. Q-values decay
        toward 0 each trial (forgetting), which is stronger under higher anxiety.
    shared_mechanism_2()
        action_1 : array-like of int in {0,1}
            First-stage choices.
        state : array-like of int in {0,1}
            Second-stage state per trial.
        action_2 : array-like of int in {0,1}
            Second-stage actions.
        reward : array-like of float
            Rewards per trial.
        stai : array-like of float in [0,1]
            Anxiety score; only stai[0] is used.
        model_parameters : list/tuple of floats
            [alpha, beta, zeta0, phi0, forget0]
            - alpha in [0,1]: learning rate for Q-values at both stages.
            - beta in [0,10]: inverse temperature for both stages.
            - zeta0 in [0,1]: baseline exploration bonus scale (reduced by anxiety).
            - phi0 in [0,1]: baseline model-based weight at stage 1 (reduced by anxiety).
            - forget0 in [0,1]: baseline per-trial forgetting rate (increased by anxiety).
    shared_mechanism_1()
            Negative log-likelihood of observed choices.
        alpha, beta, zeta0, phi0, forget0 = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        zeta = max(0.0, min(1.0, zeta0 * (1.0 - stai)))            # less exploration with anxiety
        phi = max(0.0, min(1.0, phi0 * (1.0 - 0.5 * stai)))        # less MB weighting with anxiety
        forget = max(0.0, min(1.0, forget0 * (0.5 + 0.5 * stai)))  # more forgetting with anxiety
        trans_counts = np.ones((2, 2))  # rows: a1 in {0,1}, cols: state in {0,1}
        q1_mf = np.zeros(2)
        q2_mf = np.zeros((2, 2))
        visit_counts = np.ones((2, 2))
    shared_mechanism_0()
        for t in range(n_trials):
            q1_mf *= (1.0 - forget)
            q2_mf *= (1.0 - forget)
            trans_probs = trans_counts / np.sum(trans_counts, axis=1, keepdims=True)  # shape (2,2)
            max_q2 = np.max(q2_mf, axis=1)           # best per state
            q1_mb = trans_probs @ max_q2
            q1_hybrid = phi * q1_mb + (1.0 - phi) * q1_mf
            logits1 = beta * q1_hybrid
            logits1 -= np.max(logits1)
            probs1 = np.exp(logits1)
            probs1 /= np.sum(probs1)
            a1 = int(action_1[t])
            s = int(state[t])
            a2 = int(action_2[t])
            r = reward[t]
            p_choice_1[t] = probs1[a1]
            bonus = zeta / np.sqrt(visit_counts[s] + 1e-8)
            logits2 = beta * (q2_mf[s] + bonus)
            logits2 -= np.max(logits2)
            probs2 = np.exp(logits2)
            probs2 /= np.sum(probs2)
            p_choice_2[t] = probs2[a2]
            visit_counts[s, a2] += 1.0
            delta2 = r - q2_mf[s, a2]
            q2_mf[s, a2] += alpha * delta2
            backup = q2_mf[s, a2]
            delta1 = backup - q1_mf[a1]
            q1_mf[a1] += alpha * delta1
            trans_counts[a1, s] += 1.0
    shared_mechanism_3()

models[23] = participant_23

def participant_24(action_1, state, action_2, reward, stai, model_parameters):
        This model learns MF values for both stages. Learning rates differ for positive vs. negative
        prediction errors, and anxiety shifts the positivity bias (higher anxiety reduces the difference).
        Choice policies at both stages include a perseveration (stickiness) bias that grows with anxiety.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices (0=A, 1=U).
        state : array-like of int (0 or 1)
            Second-stage states visited (0=X, 1=Y).
        action_2 : array-like of int (0 or 1)
            Second-stage choices within the visited state (X: 0=W,1=S; Y: 0=P,1=H).
        reward : array-like of float
            Received reward on each trial (e.g., 0/1).
        stai : array-like of float
            Trait anxiety score in [0,1]; stai[0] is used.
        model_parameters : array-like
            [alpha_pos, alpha_neg, beta, kappa0, kappa_stai]
            - alpha_pos in [0,1]: base learning rate for positive prediction errors.
            - alpha_neg in [0,1]: base learning rate for negative prediction errors.
            - beta in [0,10]: inverse temperature for softmax at both stages.
            - kappa0 in [0,1]: baseline perseveration bias added to the last chosen action.
            - kappa_stai in [0,1]: how much anxiety increases perseveration (stickiness).
    shared_mechanism_1()
            Negative log-likelihood of the observed first- and second-stage choices.
        alpha_pos, alpha_neg, beta, kappa0, kappa_stai = model_parameters
        n_trials = len(action_1)
        s = float(stai[0])
        q1 = np.zeros(2)            # stage-1 MF values
        q2 = np.zeros((2, 2))       # stage-2 MF values
        last_a1 = None
        last_a2_by_state = {0: None, 1: None}
        kappa = kappa0 + kappa_stai * s
    shared_mechanism_0()
        for t in range(n_trials):
            bias1 = np.zeros(2)
            if last_a1 is not None:
                bias1[last_a1] += kappa
            st = state[t]
            bias2 = np.zeros(2)
            if last_a2_by_state[st] is not None:
                bias2[last_a2_by_state[st]] += kappa
            prefs1 = q1 + bias1
            exp1 = np.exp(beta * (prefs1 - np.max(prefs1)))
            probs1 = exp1 / np.sum(exp1)
            a1 = action_1[t]
            p_choice_1[t] = probs1[a1]
            prefs2 = q2[st] + bias2
            exp2 = np.exp(beta * (prefs2 - np.max(prefs2)))
            probs2 = exp2 / np.sum(exp2)
            a2 = action_2[t]
            p_choice_2[t] = probs2[a2]
            eff_alpha_pos = (1 - s) * alpha_pos + s * alpha_neg
            eff_alpha_neg = (1 - s) * alpha_neg + s * alpha_pos
            r = reward[t]
            pe2 = r - q2[st, a2]
            a2_lr = eff_alpha_pos if pe2 >= 0 else eff_alpha_neg
            q2[st, a2] += a2_lr * pe2
            pe1 = q2[st, a2] - q1[a1]
            a1_lr = eff_alpha_pos if pe1 >= 0 else eff_alpha_neg
            q1[a1] += a1_lr * pe1
            last_a1 = a1
            last_a2_by_state[st] = a2
    shared_mechanism_3()

models[24] = participant_24

def participant_25(action_1, state, action_2, reward, stai, model_parameters):
        This model blends model-based (MB) and model-free (MF) control at stage 1,
        uses a softmax policy at both stages, and learns with TD(λ). Anxiety (stai)
        down-weights model-based control and increases perseveration.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices (0=A, 1=U) on each trial.
        state : array-like of int (0 or 1)
            Second-stage state reached (0=X, 1=Y) on each trial.
        action_2 : array-like of int (0 or 1)
            Second-stage choices (0 or 1; e.g., alien indices) on each trial.
        reward : array-like of float
            Reward received on each trial (e.g., 0.0 or 1.0).
        stai : array-like of float
            Anxiety score(s). Uses the first element as the participant's anxiety.
            Interpretation in this model:
              - Higher stai reduces model-based weight and increases perseveration strength.
        model_parameters : tuple/list of 5 floats
            (alpha, beta, w_mb, lam, kappa)
            - alpha in [0,1]: learning rate.
            - beta in [0,10]: inverse temperature for softmax.
            - w_mb in [0,1]: baseline model-based weight (before anxiety).
            - lam in [0,1]: eligibility trace parameter λ.
            - kappa in [0,1]: baseline perseveration weight.
    shared_mechanism_1()
            Negative log-likelihood of the observed action sequence under the model.
        alpha, beta, w_mb, lam, kappa = model_parameters
        n_trials = len(action_1)
        stai_val = float(stai[0]) if hasattr(stai, "__len__") else float(stai)
        transition_matrix = np.array([[0.7, 0.3],
                                      [0.3, 0.7]])
        q_stage1_mf = np.zeros(2)         # MF values for stage-1 actions
        q_stage2_mf = np.zeros((2, 2))    # MF values for stage-2 actions per state
    shared_mechanism_0()
        prev_a1 = -1
        prev_a2_state0 = -1
        prev_a2_state1 = -1
        w_eff = w_mb * (1.0 - stai_val)
        kappa_eff = kappa * (1.0 + stai_val)
        eps = 1e-10
        for t in range(n_trials):
            s = state[t]
            max_q_stage2 = np.max(q_stage2_mf, axis=1)        # size 2: best value at each state
            q_stage1_mb = transition_matrix @ max_q_stage2     # size 2
            q1_hybrid = w_eff * q_stage1_mb + (1.0 - w_eff) * q_stage1_mf
            bias1 = np.zeros(2)
            if prev_a1 in (0, 1):
                bias1[prev_a1] += kappa_eff
            logits1 = beta * q1_hybrid + bias1
            logits1 -= np.max(logits1)
            probs1 = np.exp(logits1)
            probs1 /= np.sum(probs1)
            a1 = action_1[t]
            p_choice_1[t] = max(probs1[a1], eps)
            bias2 = np.zeros(2)
            prev_a2 = prev_a2_state0 if s == 0 else prev_a2_state1
            if prev_a2 in (0, 1):
                bias2[prev_a2] += kappa_eff
            logits2 = beta * q_stage2_mf[s] + bias2
            logits2 -= np.max(logits2)
            probs2 = np.exp(logits2)
            probs2 /= np.sum(probs2)
            a2 = action_2[t]
            p_choice_2[t] = max(probs2[a2], eps)
            r = reward[t]
            q2_old = q_stage2_mf[s, a2]
            delta2 = r - q2_old
            q_stage2_mf[s, a2] += alpha * delta2
            delta1 = q2_old - q_stage1_mf[a1]
            q_stage1_mf[a1] += alpha * (delta1 + lam * delta2)
            prev_a1 = a1
            if s == 0:
                prev_a2_state0 = a2
            else:
                prev_a2_state1 = a2
        neg_ll = -(np.sum(np.log(p_choice_1)) + np.sum(np.log(p_choice_2)))

models[25] = participant_25

def participant_26(action_1, state, action_2, reward, stai, model_parameters):
        Overview:
        - Learns an internal transition model T_est for each first-stage action and plans model-based values through it.
        - Combines MB and MF values at stage 1 with a simple anxiety-based arbitration: weight on MF increases with anxiety.
        - Stage-2 learning uses valence-asymmetric learning rates that scale with anxiety (more anxious -> stronger learning from non-reward).
        - Includes an anxiety-amplified perseveration bias at stage 1.
        Parameters (model_parameters):
        - alpha: base learning rate in [0,1].
        - beta: inverse temperature for softmax in [0,10].
        - kappa0: base learning rate for updating the transition model in [0,1].
        - phi: perseveration strength in [0,1], bias to repeat previous stage-1 action.
        - a_neg: asymmetry coefficient in [0,1] scaling how anxiety increases learning from negative outcomes.
        Inputs:
        - action_1: array of ints in {0,1}, first-stage choices (0=A, 1=U).
        - state: array of ints in {0,1}, second-stage state reached (0=X, 1=Y).
        - action_2: array of ints in {0,1}, second-stage choices on the observed planet.
        - reward: array of floats (typically 0 or 1).
        - stai: array-like with a single float in [0,1], anxiety score for the participant.
        - model_parameters: list/array [alpha, beta, kappa0, phi, a_neg].
        Returns:
        - Negative log-likelihood of the observed choices at both stages.
        alpha, beta, kappa0, phi, a_neg = model_parameters
        n_trials = len(action_1)
        st = float(stai[0])
        T_est = np.array([[0.7, 0.3],
                          [0.3, 0.7]], dtype=float)
        q_stage1_mf = np.zeros(2)
        q_stage2 = np.zeros((2, 2))
        prev_a1 = None
    shared_mechanism_0()
        eps = 1e-12
        w_mf = st
        w_mb = 1.0 - w_mf
        kappa = kappa0 * (0.25 + 0.75 * st)
        alpha_pos = np.clip(alpha * (1.0 - a_neg * st), 0.0, 1.0)
        alpha_neg = np.clip(alpha * (1.0 + a_neg * st), 0.0, 1.0)
        for t in range(n_trials):
            a1 = int(action_1[t])
            s = int(state[t])
            a2 = int(action_2[t])
            r = reward[t]
            max_q2 = np.max(q_stage2, axis=1)      # best alien on each planet
            q1_mb = T_est @ max_q2
            persev_bias = np.zeros(2)
            if prev_a1 is not None:
                persev_bias[prev_a1] = phi * (0.5 + 0.5 * st)
            q1_total = w_mb * q1_mb + w_mf * q_stage1_mf + persev_bias
            exp_q1 = np.exp(beta * (q1_total - np.max(q1_total)))
            probs_1 = exp_q1 / (np.sum(exp_q1) + eps)
            p_choice_1[t] = probs_1[a1]
            q2_vec = q_stage2[s]
            exp_q2 = np.exp(beta * (q2_vec - np.max(q2_vec)))
            probs_2 = exp_q2 / (np.sum(exp_q2) + eps)
            p_choice_2[t] = probs_2[a2]
            target = np.array([1.0 if i == s else 0.0 for i in range(2)])
            T_est[a1] = (1.0 - kappa) * T_est[a1] + kappa * target
            T_est[a1] = T_est[a1] / (np.sum(T_est[a1]) + eps)
            pe2 = r - q_stage2[s, a2]
            lr2 = alpha_pos if pe2 >= 0.0 else alpha_neg
            q_stage2[s, a2] += lr2 * pe2
            pe1 = q_stage2[s, a2] - q_stage1_mf[a1]
            q_stage1_mf[a1] += lr2 * pe1
            prev_a1 = a1
        log_loss = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[26] = participant_26

def participant_27(action_1, state, action_2, reward, stai, model_parameters):
        This model blends model-free and model-based control with a fixed weight, but:
        - Down-weights model-based control specifically after rare transitions, more so with higher anxiety.
        - Uses anxiety to reduce effective beta (more exploration with higher anxiety).
        - Adds a Pavlovian 'safety' bias toward action 0 at stage 2 that grows with anxiety and uncertainty.
        - Includes value forgetting toward 0.5 to capture drift in unchosen/unstimulated values.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices each trial.
        state : array-like of int (0 or 1)
            Second-stage state each trial.
        action_2 : array-like of int (0 or 1)
            Second-stage choices each trial.
        reward : array-like of float
            Rewards each trial.
        stai : array-like of float
            Single anxiety score in [0,1].
        model_parameters : list or array of floats
            [alpha, beta, mb_weight, bias_safe, forget]
            - alpha in [0,1]: learning rate for Q updates.
            - beta in [0,10]: base inverse temperature.
            - mb_weight in [0,1]: baseline model-based mixing weight at stage 1.
            - bias_safe in [0,1]: strength of Pavlovian bias favoring action 0 under uncertainty.
            - forget in [0,1]: forgetting rate pulling Qs toward 0.5 each trial.
    shared_mechanism_1()
            Negative log-likelihood of observed choices.
        alpha, beta, mb_weight, bias_safe, forget = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        T_fixed = np.array([[0.7, 0.3],
                            [0.3, 0.7]], dtype=float)
        Q1_mf = np.zeros(2)
        Q2 = 0.5 * np.ones((2, 2))
    shared_mechanism_0()
        for t in range(n_trials):
            a1 = int(action_1[t])
            s = int(state[t])
            a2 = int(action_2[t])
            r = reward[t]
            is_common = ((a1 == 0 and s == 0) or (a1 == 1 and s == 1))
            mb_w_eff = mb_weight if is_common else mb_weight * (1.0 - 0.7 * stai)
            mb_w_eff = np.clip(mb_w_eff, 0.0, 1.0)
            max_Q2 = np.max(Q2, axis=1)
            Q1_mb = T_fixed @ max_Q2
            Q1 = mb_w_eff * Q1_mb + (1.0 - mb_w_eff) * Q1_mf
            beta_eff = beta * (1.0 - 0.5 * stai)
            logits1 = beta_eff * Q1
            logits1 -= np.max(logits1)
            probs1 = np.exp(logits1)
            probs1 /= np.sum(probs1)
            p_choice_1[t] = probs1[a1]
            uncertainty = 1.0 - abs(Q2[s, 0] - Q2[s, 1])  # in [0,1]
            bias_vec = np.array([bias_safe * stai * uncertainty, 0.0])
            logits2 = beta_eff * Q2[s] + bias_vec
            logits2 -= np.max(logits2)
            probs2 = np.exp(logits2)
            probs2 /= np.sum(probs2)
            p_choice_2[t] = probs2[a2]
            Q2 = (1.0 - forget) * Q2 + forget * 0.5
            Q1_mf = (1.0 - forget) * Q1_mf + forget * 0.0  # MF baseline around 0
            delta2 = r - Q2[s, a2]
            Q2[s, a2] += alpha * delta2
            delta1 = Q2[s, a2] - Q1_mf[a1]
            Q1_mf[a1] += alpha * (0.5 * delta1 + 0.5 * delta2)
    shared_mechanism_3()

models[27] = participant_27

def participant_28(action_1, state, action_2, reward, stai, model_parameters):
        A model-free learner updates stage-2 values using a nonlinear utility function
        u(r) = r^curv (concave if curv<1). Learning is asymmetric for positive vs negative
        prediction errors, with asymmetry scaled by anxiety. Values undergo forgetting
        toward a neutral prior (0.5). A single inverse temperature governs both stages,
        and its effective strength is reduced for higher anxiety.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices: 0=A, 1=U.
        state : array-like of int (0 or 1)
            Second-stage states: 0=X, 1=Y.
        action_2 : array-like of int (0 or 1)
            Second-stage choices within the reached state.
        reward : array-like of float in [0,1]
            Reward outcome.
        stai : array-like with one float in [0,1]
            Anxiety score; increases learning asymmetry and softens choice.
        model_parameters : list or array-like of 5 floats
            [alpha, beta, c_asym, forget, curv]
            - alpha: base learning rate [0,1]
            - beta: inverse temperature baseline [0,10]
            - c_asym: scales asymmetry via anxiety: 
                      alpha_plus = alpha * (1 + c_asym*stai), 
                      alpha_minus = alpha * (1 - c_asym*stai) [0,1]
            - forget: forgetting rate toward 0.5 for all Qs each trial [0,1]
            - curv: utility curvature for rewards in [0,1], u = r**curv [0,1]
    shared_mechanism_1()
            Negative log-likelihood of the observed choices.
        alpha, beta, c_asym, forget, curv = model_parameters
        n_trials = len(action_1)
        stai_val = float(stai[0])
        beta_eff = beta * (1.0 - 0.5 * (stai_val - 0.5))
        if beta_eff < 1e-6:
            beta_eff = 1e-6
        Q1 = np.zeros(2)        # MF stage-1 values
        Q2 = np.zeros((2, 2))   # stage-2 values
    shared_mechanism_0()
        eps = 1e-10
        alpha_plus = alpha * (1.0 + c_asym * stai_val)
        alpha_minus = alpha * (1.0 - c_asym * stai_val)
        if alpha_plus > 1.0:
            alpha_plus = 1.0
        if alpha_minus < 0.0:
            alpha_minus = 0.0
        for t in range(n_trials):
            Q1 = (1.0 - forget) * Q1 + forget * 0.5
            Q2 = (1.0 - forget) * Q2 + forget * 0.5
            q1c = Q1 - np.max(Q1)
            ps1 = np.exp(beta_eff * q1c)
            ps1 = ps1 / (np.sum(ps1) + eps)
            a1 = int(action_1[t])
            p_choice_1[t] = ps1[a1]
            s = int(state[t])
            q2c = Q2[s] - np.max(Q2[s])
            ps2 = np.exp(beta_eff * q2c)
            ps2 = ps2 / (np.sum(ps2) + eps)
            a2 = int(action_2[t])
            p_choice_2[t] = ps2[a2]
            r = reward[t]
            u = r ** curv
            pe2 = u - Q2[s, a2]
            a2_lr = alpha_plus if pe2 >= 0.0 else alpha_minus
            Q2[s, a2] += a2_lr * pe2
            pe1 = Q2[s, a2] - Q1[a1]
            a1_lr = alpha_plus if pe1 >= 0.0 else alpha_minus
            Q1[a1] += a1_lr * pe1
        neg_log_lik = -(np.sum(np.log(p_choice_1 + 1e-10)) + np.sum(np.log(p_choice_2 + 1e-10)))

models[28] = participant_28

def participant_29(action_1, state, action_2, reward, stai, model_parameters):
        The agent dislikes uncertain second-stage options: both action selection and learning
        penalize options with higher reward uncertainty (estimated from current Q2 via p*(1-p)).
        Anxiety increases uncertainty aversion. Additionally, a simple choice kernel captures
        repetition tendencies at both stages, with strength growing slightly with anxiety.
        Parameters
        - action_1: array-like of ints in {0,1}. First-stage choices (0=A, 1=U).
        - state: array-like of ints in {0,1}. Second-stage state reached (0=X, 1=Y).
        - action_2: array-like of ints in {0,1}. Second-stage choices within the reached state.
        - reward: array-like of floats in [0,1]. Coins received on each trial.
        - stai: array-like with one float in [0,1]. Anxiety score.
        - model_parameters: iterable of 5 floats
            alpha: [0,1] — learning rate for MF values
            beta: [0,10] — inverse temperature for both stages
            kernel_lr: [0,1] — learning/decay rate of the choice kernels
            risk_penalty: [0,1] — base penalty weight for uncertainty
            anx_risk_gain: [0,1] — how strongly anxiety amplifies uncertainty aversion and kernel strength
        Returns
        - Negative log-likelihood of the observed action_1 and action_2 sequences.
        alpha, beta, kernel_lr, risk_penalty, anx_risk_gain = model_parameters
        n_trials = len(action_1)
        stai_val = float(stai[0])
        q1 = np.zeros(2)
        q2 = np.zeros((2, 2))
        k1 = np.zeros(2)
        k2 = np.zeros((2, 2))
    shared_mechanism_0()
        lam = risk_penalty * (1.0 + anx_risk_gain * stai_val)
        kernel_strength = kernel_lr * (1.0 + 0.5 * anx_risk_gain * stai_val)
        eps = 1e-12
        for t in range(n_trials):
            var2 = q2 * (1.0 - q2)
            unc2 = np.sqrt(np.clip(var2, 0.0, 0.25))  # in [0, 0.5]
            logits1 = q1 + kernel_strength * (2.0 * k1 - 1.0)
            logits1 = beta * (logits1 - np.max(logits1))
            probs1 = np.exp(logits1)
            probs1 /= (np.sum(probs1) + eps)
            a1 = action_1[t]
            p_choice_1[t] = probs1[a1]
            s = state[t]
            logits2 = (q2[s] - lam * unc2[s]) + kernel_strength * (2.0 * k2[s] - 1.0)
            logits2 = beta * (logits2 - np.max(logits2))
            probs2 = np.exp(logits2)
            probs2 /= (np.sum(probs2) + eps)
            a2 = action_2[t]
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            var_chosen = q2[s, a2] * (1.0 - q2[s, a2])
            unc_chosen = np.sqrt(max(0.0, min(0.25, var_chosen)))
            r_subj = r - lam * unc_chosen
            pe2 = r_subj - q2[s, a2]
            q2[s, a2] += alpha * pe2
            target1 = q2[s, a2]
            pe1 = target1 - q1[a1]
            q1[a1] += alpha * pe1
            k1 = (1.0 - kernel_lr) * k1
            k1[a1] += kernel_lr
            k2[s] = (1.0 - kernel_lr) * k2[s]
            k2[s, a2] += kernel_lr
        neg_ll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[29] = participant_29

def participant_30(action_1, state, action_2, reward, stai, model_parameters):
        Anxiety use: Higher STAI reduces model-based control weight (w), biasing toward model-free control.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices per trial (spaceship: 0=A, 1=U).
        state : array-like of int (0 or 1)
            Second-stage state per trial (planet: 0=X, 1=Y).
        action_2 : array-like of int (0 or 1)
            Second-stage choices per trial (alien on the planet; index within the planet).
        reward : array-like of float (0 or 1)
            Received reward per trial.
        stai : array-like of float
            Trait anxiety score; stai[0] used here. Interpreted in [0,1].
        model_parameters : array-like of float
            [alpha, beta, w, lambda_, perseveration]
            - alpha in [0,1]: learning rate for Q-value updates.
            - beta in [0,10]: inverse temperature for softmax choice.
            - w in [0,1]: baseline model-based weight; anxiety reduces this.
            - lambda_ in [0,1]: eligibility trace mixing stage-2 PE into stage-1 update.
            - perseveration in [0,1]: tendency to repeat previous action (applied at both stages).
    shared_mechanism_1()
            Negative log-likelihood of observed first- and second-stage choices.
        alpha, beta, w, lambda_, perseveration = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        transition_matrix = np.array([[0.7, 0.3],
                                      [0.3, 0.7]])
    shared_mechanism_0()
        q_stage1_mf = np.zeros(2)           # MF Q for first-stage actions (A,U)
        q_stage2_mf = np.zeros((2, 2))      # MF Q for second-stage actions at each state
        prev_a1 = None
        prev_a2_by_state = [None, None]
        w_eff = np.clip(w * (1.0 - 0.8 * stai), 0.0, 1.0)
        for t in range(n_trials):
            s = int(state[t])
            max_q_stage2 = np.max(q_stage2_mf, axis=1)      # size 2: best alien at X, Y
            q_stage1_mb = transition_matrix @ max_q_stage2  # expected value of A and U
            q1 = (1.0 - w_eff) * q_stage1_mf + w_eff * q_stage1_mb
            bias1 = np.zeros(2)
            if prev_a1 is not None:
                bias1[prev_a1] += perseveration
            logits1 = beta * q1 + bias1
            logits1 -= np.max(logits1)
            probs_1 = np.exp(logits1) / np.sum(np.exp(logits1))
            a1 = int(action_1[t])
            p_choice_1[t] = probs_1[a1]
            q2 = q_stage2_mf[s].copy()
            bias2 = np.zeros(2)
            if prev_a2_by_state[s] is not None:
                bias2[prev_a2_by_state[s]] += perseveration
            logits2 = beta * q2 + bias2
            logits2 -= np.max(logits2)
            probs_2 = np.exp(logits2) / np.sum(np.exp(logits2))
            a2 = int(action_2[t])
            p_choice_2[t] = probs_2[a2]
            r = float(reward[t])
            delta1 = q_stage2_mf[s, a2] - q_stage1_mf[a1]
            delta2 = r - q_stage2_mf[s, a2]
            q_stage2_mf[s, a2] += alpha * delta2
            q_stage1_mf[a1] += alpha * (delta1 + lambda_ * delta2)
            prev_a1 = a1
            prev_a2_by_state[s] = a2
        eps = 1e-10
        nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[30] = participant_30

def participant_31(action_1, state, action_2, reward, stai, model_parameters):
        Idea:
        - Use MF Q-learning at both stages with per-trial forgetting/decay of all Q-values.
        - Anxiety increases the effective decay (forgetting), capturing higher volatility beliefs.
        - First-stage decision includes a heuristic bias that favors the spaceship whose common
          destination currently has higher estimated value (based on max second-stage Q for each planet).
          This uses the known transition structure without full planning.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices (0=A, 1=U).
        state : array-like of int (0 or 1)
            Second-stage states encountered (0=X, 1=Y).
        action_2 : array-like of int (0 or 1)
            Second-stage choices within the encountered state (0/1).
        reward : array-like of float
            Rewards obtained on each trial.
        stai : array-like of float
            Anxiety score; uses stai[0] in [0,1]. Higher anxiety => stronger forgetting.
        model_parameters : array-like of floats, length 5
            [alpha, beta, decay, k_anx_decay, tr_bias]
            - alpha in [0,1]: learning rate.
            - beta in [0,10]: inverse temperature.
            - decay in [0,1]: baseline forgetting rate applied each trial to all Q-values.
            - k_anx_decay in [0,1]: scales how anxiety increases effective decay:
                                    decay_eff = min(1, decay + k_anx_decay * stai).
            - tr_bias in [0,1]: strength of heuristic transition bias added to first-stage logits.
    shared_mechanism_1()
            Negative log-likelihood of observed choices.
        alpha, beta, decay, k_anx_decay, tr_bias = model_parameters
        n_trials = len(action_1)
        stai_val = float(stai[0])
        decay_eff = decay + k_anx_decay * stai_val
        if decay_eff < 0.0:
            decay_eff = 0.0
        if decay_eff > 1.0:
            decay_eff = 1.0
        keep = 1.0 - decay_eff
    shared_mechanism_0()
        q1 = np.zeros(2)
        q2 = np.zeros((2, 2))
        for t in range(n_trials):
            v_planet = np.max(q2, axis=1)  # [vX, vY]
            bias1 = np.array([tr_bias * (v_planet[0] - v_planet[1]),
                              -tr_bias * (v_planet[0] - v_planet[1])])
            logits1 = beta * q1 + bias1
            logits1 -= np.max(logits1)
            p1 = np.exp(logits1)
            p1 = p1 / (np.sum(p1) + 1e-12)
            a1 = int(action_1[t])
            p_choice_1[t] = p1[a1]
            s = int(state[t])
            logits2 = beta * q2[s]
            logits2 -= np.max(logits2)
            p2 = np.exp(logits2)
            p2 = p2 / (np.sum(p2) + 1e-12)
            a2 = int(action_2[t])
            p_choice_2[t] = p2[a2]
            r = reward[t]
            q1 *= keep
            q2 *= keep
            delta2 = r - q2[s, a2]
            q2[s, a2] += alpha * delta2
            delta1 = q2[s, a2] - q1[a1]
            q1[a1] += alpha * delta1
        eps = 1e-10
        nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[31] = participant_31

def participant_32(action_1, state, action_2, reward, stai, model_parameters):
        Asymmetric (win/loss) learning with anxiety-modulated forgetting and choice persistence.
        Core idea:
        - Model-free learner with separate learning rates for positive vs. negative second-stage prediction errors.
        - Anxiety increases global forgetting of Q-values (toward a neutral prior), reflecting reduced confidence/maintenance.
        - Persistence (choice stickiness) at both stages, scaled up by anxiety.
    shared_mechanism_2()
        action_1 : array-like of int {0,1}
            First-stage choices per trial.
        state : array-like of int {0,1}
            Reached second-stage state per trial.
        action_2 : array-like of int {0,1}
            Second-stage choices (alien) per trial.
        reward : array-like of float
            Reward obtained each trial.
        stai : array-like of float
            Anxiety score array; uses stai[0].
        model_parameters : list or array-like of float
            [mu_win, mu_loss, beta, z_forget, psi_persist]
            Bounds:
            - mu_win: [0,1] learning rate used when the second-stage prediction error is positive.
            - mu_loss: [0,1] learning rate used when the second-stage prediction error is negative.
            - beta: [0,10] inverse temperature for both stages.
            - z_forget: [0,1] forgetting strength per trial, scaled by anxiety.
            - psi_persist: [0,1] baseline stickiness (persistence) strength, scaled by anxiety.
    shared_mechanism_1()
            Negative log-likelihood of observed choices.
        mu_win, mu_loss, beta, z_forget, psi_persist = model_parameters
        n_trials = len(action_1)
        stai = stai[0]
        q1_mf = np.zeros(2)        # values for first-stage actions A/U
        q2 = np.zeros((2, 2))      # values for second-stage state x action
    shared_mechanism_0()
        prev_a1 = None
        prev_a2_state = {0: None, 1: None}
        prior_q1 = 0.5
        prior_q2 = 0.5
        for t in range(n_trials):
            stick = psi_persist * (0.5 + 0.5 * stai)
            bias1 = np.zeros(2)
            if prev_a1 is not None:
                bias1[prev_a1] = 1.0
            logits1 = beta * (q1_mf - np.max(q1_mf)) + stick * bias1
            probs1 = np.exp(logits1 - np.max(logits1))
            probs1 = probs1 / np.sum(probs1)
            a1 = action_1[t]
            p_choice_1[t] = probs1[a1]
            s = state[t]
            q2_s = q2[s].copy()
            bias2 = np.zeros(2)
            if prev_a2_state[s] is not None:
                bias2[prev_a2_state[s]] = 1.0
            logits2 = beta * (q2_s - np.max(q2_s)) + stick * bias2
            probs2 = np.exp(logits2 - np.max(logits2))
            probs2 = probs2 / np.sum(probs2)
            a2 = action_2[t]
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            pe2 = r - q2[s, a2]
            lr2 = mu_win if pe2 >= 0.0 else mu_loss
            q2[s, a2] += lr2 * pe2
            pe1 = q2[s, a2] - q1_mf[a1]
            lr1 = mu_win if pe1 >= 0.0 else mu_loss
            q1_mf[a1] += lr1 * pe1
            f = np.clip(z_forget * (0.5 + 0.5 * stai), 0.0, 1.0)
            q1_mf = (1.0 - f) * q1_mf + f * prior_q1
            q2 = (1.0 - f) * q2 + f * prior_q2
            prev_a1 = a1
            prev_a2_state[s] = a2
        eps = 1e-12
        neg_log_lik = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[32] = participant_32

def participant_33(action_1, state, action_2, reward, stai, model_parameters):
    shared_mechanism_2()
        action_1 : array-like of int (0/1)
            First-stage choices (0=A, 1=U) for each trial.
        state : array-like of int (0/1)
            Second-stage state reached on each trial (0=X, 1=Y).
        action_2 : array-like of int (0/1)
            Second-stage choices (0/1) on each trial (e.g., W vs S on X; P vs H on Y).
        reward : array-like of float
            Reward received on each trial (e.g., 0.0 or 1.0).
        stai : array-like of float
            Trait anxiety score; use stai[0]. Higher means higher anxiety.
        model_parameters : tuple/list of floats
            (alpha, beta, w_base, lambda_e, persev)
            - alpha in [0,1]: learning rate for value updates.
            - beta in [0,10]: inverse temperature for softmax at both stages.
            - w_base in [0,1]: baseline weight on model-based control at stage 1.
            - lambda_e in [0,1]: eligibility trace for backpropagating reward to stage 1 MF.
            - persev in [0,1]: perseveration strength added to the previously chosen first-stage action.
    shared_mechanism_1()
            Negative log-likelihood of observed first- and second-stage choices.
        alpha, beta, w_base, lambda_e, persev = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        transition_matrix = np.array([[0.7, 0.3],
                                      [0.3, 0.7]])
        q_stage2 = np.zeros((2, 2))     # Q at second stage: state x action
        q_stage1_mf = np.zeros(2)       # Model-free first-stage values
    shared_mechanism_0()
        prev_a1 = None
        w = w_base * (1.0 - 0.7 * stai)
        w = max(0.0, min(1.0, w))
        for t in range(n_trials):
            s = int(state[t])
            a1 = int(action_1[t])
            a2 = int(action_2[t])
            r = float(reward[t])
            mb_values = transition_matrix @ np.max(q_stage2, axis=1)  # shape (2,)
            persev_eff = persev * (1.0 + 0.5 * stai)
            bias = np.zeros(2)
            if prev_a1 is not None:
                bias[prev_a1] += persev_eff
            q1 = w * mb_values + (1.0 - w) * q_stage1_mf + bias
            q1_shift = q1 - np.max(q1)  # numerical stability
            exp_q1 = np.exp(beta * q1_shift)
            probs_1 = exp_q1 / (np.sum(exp_q1) + 1e-12)
            p_choice_1[t] = probs_1[a1]
            q2 = q_stage2[s, :].copy()
            q2_shift = q2 - np.max(q2)
            exp_q2 = np.exp(beta * q2_shift)
            probs_2 = exp_q2 / (np.sum(exp_q2) + 1e-12)
            p_choice_2[t] = probs_2[a2]
            delta2 = r - q_stage2[s, a2]
            q_stage2[s, a2] += alpha * delta2
            delta1 = q_stage2[s, a2] - q_stage1_mf[a1]
            q_stage1_mf[a1] += alpha * delta1
            q_stage1_mf[a1] += alpha * lambda_e * delta2
            prev_a1 = a1
    shared_mechanism_3()

models[33] = participant_33

def participant_34(action_1, state, action_2, reward, stai, model_parameters):
        The agent learns the first-stage transition probabilities online and uses them
        for model-based evaluation. Anxiety biases transition learning: rare transitions
        (relative to current belief) are learned faster when anxiety is high, while
        common transitions are learned more conservatively. Additionally, first-stage
        choices exhibit perseveration that scales with anxiety.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices (0=A, 1=U).
        state : array-like of int (0 or 1)
            Observed second-stage state (0=X, 1=Y).
        action_2 : array-like of int (0 or 1)
            Second-stage choices (alien 0 or 1).
        reward : array-like of float
            Reward obtained each trial.
        stai : array-like of float
            Anxiety score in [0,1]; higher indicates greater anxiety. Only stai[0] is used.
        model_parameters : list/tuple of 5 floats
            [alphaQ, beta, tau_T, k_anx_trans_bias, psi_perseverate]
            - alphaQ in [0,1]: learning rate for MF values (both stages).
            - beta in [0,10]: inverse temperature for both stages.
            - tau_T in [0,1]: base learning rate for transition probabilities.
            - k_anx_trans_bias in [0,1]: scales anxiety-dependent modulation of transition learning.
              Effective tau: increased for rare transitions, decreased for common ones:
              tau_eff = tau_T * (1 + k_anx_trans_bias * stai) if rare else tau_T * (1 - k_anx_trans_bias * stai).
            - psi_perseverate in [0,1]: base perseveration weight at stage 1; actual bias is psi_perseverate * stai.
    shared_mechanism_1()
            Negative log-likelihood of the observed first- and second-stage choices.
        alphaQ, beta, tau_T, k_anx_trans_bias, psi_perseverate = model_parameters
        n_trials = len(action_1)
        stai_val = float(stai[0])
        T = np.full((2, 2), 0.5)
    shared_mechanism_0()
        q1_mf = np.zeros(2)
        q2 = np.zeros((2, 2))
        prev_a1 = None
        for t in range(n_trials):
            max_q2 = np.max(q2, axis=1)
            q1_mb = T @ max_q2
            stick_vec = np.zeros(2)
            if prev_a1 is not None:
                stick_vec[prev_a1] = 1.0
            perseveration_bias = psi_perseverate * stai_val * stick_vec
            q1_comb = 0.5 * q1_mb + 0.5 * q1_mf
            logits1 = q1_comb + perseveration_bias
            logits1 = logits1 - np.max(logits1)
            probs1 = np.exp(beta * logits1)
            probs1 = probs1 / np.sum(probs1)
            a1 = action_1[t]
            p_choice_1[t] = probs1[a1]
            s = state[t]
            logits2 = q2[s] - np.max(q2[s])
            probs2 = np.exp(beta * logits2)
            probs2 = probs2 / np.sum(probs2)
            a2 = action_2[t]
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            p_obs = T[a1, s]
            is_rare = p_obs < 0.5
            if is_rare:
                tau_eff = tau_T * (1.0 + k_anx_trans_bias * stai_val)
            else:
                tau_eff = tau_T * (1.0 - k_anx_trans_bias * stai_val)
            if tau_eff < 0.0:
                tau_eff = 0.0
            if tau_eff > 1.0:
                tau_eff = 1.0
            T[a1, s] += tau_eff * (1.0 - T[a1, s])
            other = 1 - s
            T[a1, other] = 1.0 - T[a1, s]
            delta2 = r - q2[s, a2]
            q2[s, a2] += alphaQ * delta2
            boot = q2[s, a2]
            delta1 = boot - q1_mf[a1]
            q1_mf[a1] += alphaQ * delta1
            prev_a1 = a1
    shared_mechanism_3()

models[34] = participant_34

def participant_35(action_1, state, action_2, reward, stai, model_parameters):
        This model uses:
        - Risk-sensitive utility at stage 2 with loss aversion that increases with anxiety.
        - A mixture of softmax and win-stay/lose-shift (WSLS) at stage 2; anxiety increases
          the WSLS weight, promoting heuristic repetition/switching.
        - A leak/forgetting term on Q-values to capture drift and reduced confidence.
        - Stage-1 values are MF bootstrapped from Q2; softmax inverse temperature decreases
          with anxiety (noisier choices).
        Parameters
        - action_1: array-like of ints in {0,1}, first-stage actions (0=A, 1=U)
        - state: array-like of ints in {0,1}, reached second-stage state (0=X, 1=Y)
        - action_2: array-like of ints in {0,1}, second-stage actions (0/1)
        - reward: array-like of floats in [0,1], reward outcome
        - stai: array-like length-1, scalar anxiety score in [0,1]
        - model_parameters: tuple/list with 5 parameters:
            alpha_q: base learning rate for Q updates in [0,1]
            beta_base: base inverse temperature in [0,10]
            phi_leak: value leak/forgetting rate in [0,1] (applied each trial)
            zeta_wsls: base WSLS mixture weight in [0,1]
            nu_loss: base loss aversion coefficient in [0,1] (utility for negative outcomes)
        Bounds
        - alpha_q, phi_leak, zeta_wsls, nu_loss in [0,1]
        - beta_base in [0,10]
        Anxiety usage
        - Loss aversion increases with anxiety: nu_eff = nu_loss * (1 + stai)
        - WSLS weight increases with anxiety: w_wsls = clip(zeta_wsls * (0.5 + 0.5*stai), 0, 1)
        - Inverse temperature decreases with anxiety: beta = beta_base * (1 - 0.5*stai)
        Returns
        - Negative log-likelihood of observed choices under the model.
        alpha_q, beta_base, phi_leak, zeta_wsls, nu_loss = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        beta = max(0.0, min(10.0, beta_base * (1.0 - 0.5 * stai)))
        w_wsls = max(0.0, min(1.0, zeta_wsls * (0.5 + 0.5 * stai)))
        nu_eff = max(0.0, min(2.0, nu_loss * (1.0 + stai)))  # cap at 2 to keep utilities reasonable
        leak = max(0.0, min(1.0, phi_leak))
        q1 = np.zeros(2)         # MF first-stage
        q2 = np.zeros((2, 2))    # second-stage
        prev_a2 = np.zeros(2, dtype=int)  # last chosen action for each state
        prev_sign = np.zeros(2)           # last reward sign (+1 / -1) for each state
        has_prev = np.zeros(2, dtype=bool)
    shared_mechanism_0()
        for t in range(n_trials):
            c_q1 = q1 - np.max(q1)
            probs1 = np.exp(beta * c_q1)
            probs1 /= np.sum(probs1)
            a1 = int(action_1[t])
            p_choice_1[t] = probs1[a1]
            s = int(state[t])
            c_q2 = q2[s] - np.max(q2[s])
            probs2_soft = np.exp(beta * c_q2)
            probs2_soft /= np.sum(probs2_soft)
            if has_prev[s]:
                if prev_sign[s] >= 0.0:
                    wsls_probs = np.array([0.0, 0.0])
                    wsls_probs[prev_a2[s]] = 1.0
                else:
                    wsls_probs = np.array([0.0, 0.0])
                    wsls_probs[1 - prev_a2[s]] = 1.0
            else:
                wsls_probs = np.array([0.5, 0.5])
            probs2 = (1.0 - w_wsls) * probs2_soft + w_wsls * wsls_probs
            probs2 /= np.sum(probs2)
            a2 = int(action_2[t])
            p_choice_2[t] = probs2[a2]
            r = float(reward[t])
            util = r if r >= 0.0 else -nu_eff * (-r)
            q2 *= (1.0 - leak)
            q1 *= (1.0 - leak)
            pe2 = util - q2[s, a2]
            q2[s, a2] += alpha_q * pe2
            target1 = q2[s, a2]
            pe1 = target1 - q1[a1]
            q1[a1] += alpha_q * pe1
            prev_a2[s] = a2
            prev_sign[s] = 1.0 if r >= 0.0 else -1.0
            has_prev[s] = True
    shared_mechanism_3()

models[35] = participant_35

def participant_36(action_1, state, action_2, reward, stai, model_parameters):
        Overview
        --------
        The agent learns second-stage action values (Q2) with a standard TD rule and backs up a model-free
        first-stage value (Q1_mf). A fixed transition model (0.7 common, 0.3 rare) supports model-based
        evaluation (Q1_mb). First-stage choice uses a hybrid of MB and MF values.
        Novel mechanisms:
          - Volatility-adaptive inverse temperature: beta is down-regulated as estimated reward volatility
            increases, and further reduced by anxiety (stai).
          - Perseveration bias at both stages to capture choice stickiness.
    shared_mechanism_2()
        action_1 : array-like of int {0,1}
            First-stage choices (0: spaceship A, 1: spaceship U).
        state : array-like of int {0,1}
            Second-stage state (0: planet X, 1: planet Y).
        action_2 : array-like of int {0,1}
            Second-stage choices within the observed state.
        reward : array-like of float
            Rewards (typically 0 or 1).
        stai : array-like of float
            Anxiety score; stai[0] in [0,1]. Higher anxiety reduces effective exploration temperature.
        model_parameters : list or array
            [alpha, beta, k_vol, w_MB, stickiness]
            - alpha in [0,1]: learning rate for Q updates (both stages).
            - beta in [0,10]: base inverse temperature for both stages.
            - k_vol in [0,1]: volatility learning rate controlling sensitivity to reward PE variance.
            - w_MB in [0,1]: weight on model-based values at stage 1 (1=fully MB).
            - stickiness in [0,1]: strength of choice perseveration bias at both stages.
        Returns
        -------
        neg_log_likelihood : float
            Negative log-likelihood of observed first- and second-stage choices.
        alpha, beta_base, k_vol, w_MB, stickiness = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        T = np.array([[0.7, 0.3],
                      [0.3, 0.7]])
        Q2 = np.zeros((2, 2))
        Q1_mf = np.zeros(2)
    shared_mechanism_0()
        v = 0.0
        prev_a1 = 0
        prev_a2 = 0
        beta_base_eff = beta_base * (1.0 - 0.5 * stai)
        for t in range(n_trials):
            a1 = int(action_1[t])
            s = int(state[t])
            a2 = int(action_2[t])
            r = float(reward[t])
            max_Q2 = np.max(Q2, axis=1)
            Q1_mb = T @ max_Q2
            Q1 = w_MB * Q1_mb + (1.0 - w_MB) * Q1_mf
            beta_t = beta_base_eff / (1.0 + v)
            bias1 = np.array([0.0, 0.0])
            bias1[prev_a1] += stickiness
            logits1 = beta_t * (Q1 - np.max(Q1)) + bias1
            probs1 = np.exp(logits1 - np.max(logits1))
            probs1 = probs1 / np.sum(probs1)
            p_choice_1[t] = probs1[a1]
            bias2 = np.array([0.0, 0.0])
            bias2[prev_a2] += stickiness
            logits2 = beta_t * (Q2[s] - np.max(Q2[s])) + bias2
            probs2 = np.exp(logits2 - np.max(logits2))
            probs2 = probs2 / np.sum(probs2)
            p_choice_2[t] = probs2[a2]
            pe2 = r - Q2[s, a2]
            Q2[s, a2] += alpha * pe2
            v = (1.0 - k_vol) * v + k_vol * (pe2 * pe2)
            target1 = Q2[s, a2]
            pe1 = target1 - Q1_mf[a1]
            Q1_mf[a1] += alpha * pe1
            prev_a1 = a1
            prev_a2 = a2
        eps = 1e-12
        neg_log_likelihood = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[36] = participant_36

def participant_37(action_1, state, action_2, reward, stai, model_parameters):
        Participants learn the first-stage transition probabilities over time. Anxiety (stai)
        increases assumed volatility: higher stai -> larger transition learning rate. A
        perseveration (stickiness) bias at stage 1 is also scaled by anxiety. Hybrid MB/MF
        control is combined with an anxiety-determined MB weight (no extra parameter).
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices (0: A, 1: U).
        state : array-like of int (0 or 1)
            Second-stage state reached (0: X, 1: Y).
        action_2 : array-like of int (0 or 1)
            Second-stage action on the reached planet.
        reward : array-like of float
            Coins received each trial.
        stai : array-like of float
            Trait anxiety score in [0,1]; used to modulate transition learning and stickiness.
        model_parameters : iterable of floats
            [alpha, beta, alpha_T0, alpha_Tgain, kappa]
            - alpha in [0,1]: reward learning rate for MF Q values.
            - beta in [0,10]: inverse temperature for both stages.
            - alpha_T0 in [0,1]: baseline transition learning rate.
            - alpha_Tgain in [0,1]: how much anxiety increases transition learning rate.
            - kappa in [0,1]: baseline perseveration strength (adds bias toward last chosen ship).
    shared_mechanism_1()
            Negative log-likelihood of the observed choices.
        alpha, beta, alpha_T0, alpha_Tgain, kappa = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        T = np.full((2, 2), 0.5)  # start uncertain
        q_stage1_mf = np.zeros(2)
        q_stage2_mf = np.zeros((2, 2))
    shared_mechanism_0()
        alpha_T = alpha_T0 + alpha_Tgain * stai
        alpha_T = min(1.0, max(0.0, alpha_T))
        stick = kappa * (1.0 + 0.5 * (stai - 0.51))
        weight_mb = 0.5 + 0.4 * (0.51 - stai)
        weight_mb = min(1.0, max(0.0, weight_mb))
        prev_a1 = -1  # no previous choice initially
        for t in range(n_trials):
            max_q_stage2 = np.max(q_stage2_mf, axis=1)  # [V(X), V(Y)]
            q_stage1_mb = T @ max_q_stage2
            q1 = weight_mb * q_stage1_mb + (1.0 - weight_mb) * q_stage1_mf
            bias = np.zeros(2)
            if prev_a1 >= 0:
                bias[prev_a1] = stick
            logits1 = beta * q1 + bias
            logits1 -= np.max(logits1)
            probs1 = np.exp(logits1)
            probs1 /= np.sum(probs1)
            a1 = action_1[t]
            p_choice_1[t] = probs1[a1]
            s = state[t]
            q2 = q_stage2_mf[s, :]
            logits2 = beta * q2
            logits2 -= np.max(logits2)
            probs2 = np.exp(logits2)
            probs2 /= np.sum(probs2)
            a2 = action_2[t]
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            T[a1, s] += alpha_T * (1.0 - T[a1, s])
            T[a1, 1 - s] = 1.0 - T[a1, s]
            delta2 = r - q_stage2_mf[s, a2]
            q_stage2_mf[s, a2] += alpha * delta2
            lam = stai
            delta1_bootstrap = q_stage2_mf[s, a2] - q_stage1_mf[a1]
            q_stage1_mf[a1] += alpha * (delta1_bootstrap + lam * delta2)
            prev_a1 = a1
    shared_mechanism_3()

models[37] = participant_37

def participant_38(action_1, state, action_2, reward, stai, model_parameters):
        Stage-2 learning uses outcome-valence asymmetric learning rates. Anxiety increases
        sensitivity to negative outcomes and reduces reliance on model-based planning.
        Both stage-2 and stage-1 MF values decay over time (forgetting). Stage-1 choice
        uses a hybrid of MB planning (using the known transition structure) and MF values.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices per trial. 0 = spaceship A, 1 = spaceship U.
        state : array-like of int (0 or 1)
            Second-stage state visited. 0 = planet X, 1 = planet Y.
        action_2 : array-like of int (0 or 1)
            Second-stage choices per trial (two aliens per planet).
        reward : array-like of float
            Coins received on each trial.
        stai : array-like, length 1, float in [0,1]
            Anxiety score; higher values increase negative-learning asymmetry and forgetting,
            and reduce MB arbitration weight.
        model_parameters : iterable of floats
            [alpha, beta, mb_w, asym, decay]
            - alpha: [0,1] base learning rate for TD updates.
            - beta: [0,10] inverse temperature for softmax choices (both stages).
            - mb_w: [0,1] baseline MB weight at stage 1 (reduced by anxiety).
            - asym: [0,1] strength of valence asymmetry in learning rates.
            - decay: [0,1] forgetting rate (toward neutral values) per trial.
        Returns
        -------
        neg_log_likelihood : float
            Negative log-likelihood of the observed choices.
        alpha, beta, mb_w, asym, decay = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    shared_mechanism_0()
        q1_mf = np.zeros(2)       # stage-1 MF Q for A/U (initialized neutral at 0)
        q2 = np.full((2, 2), 0.5) # stage-2 Q initialized neutral at 0.5
        w = mb_w * (1.0 - stai)
        w = np.clip(w, 0.0, 1.0)
        decay_eff_2 = np.clip(decay * (0.5 + 0.5 * stai), 0.0, 1.0)
        decay_eff_1 = np.clip(decay * (0.5 + 0.5 * stai), 0.0, 1.0)
        for t in range(n_trials):
            max_q2 = np.max(q2, axis=1)
            q1_mb = transition_matrix @ max_q2
            q1 = w * q1_mb + (1.0 - w) * q1_mf
            exp_q1 = np.exp(beta * (q1 - np.max(q1)))
            probs1 = exp_q1 / np.sum(exp_q1)
            a1 = action_1[t]
            p_choice_1[t] = probs1[a1]
            s = state[t]
            exp_q2 = np.exp(beta * (q2[s] - np.max(q2[s])))
            probs2 = exp_q2 / np.sum(exp_q2)
            a2 = action_2[t]
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            delta2 = r - q2[s, a2]
            alpha_pos = np.clip(alpha * (1.0 + asym * (1.0 - stai)), 0.0, 1.0)
            alpha_neg = np.clip(alpha * (1.0 + asym * stai), 0.0, 1.0)
            lr = alpha_pos if delta2 >= 0 else alpha_neg
            q2[s, a2] += lr * delta2
            target1 = q2[s, a2]
            q1_mf[a1] += alpha * (target1 - q1_mf[a1])
            q2 = (1.0 - decay_eff_2) * q2 + decay_eff_2 * 0.5
            q1_mf = (1.0 - decay_eff_1) * q1_mf
        eps = 1e-12
        neg_log_likelihood = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[38] = participant_38

def participant_39(action_1, state, action_2, reward, stai, model_parameters):
        The agent learns second-stage values model-free and uses a fixed transition
        model to compute model-based first-stage values. The weight on model-based
        control is dynamically increased by transition surprise (rare transitions),
        with a gain parameter scaled by anxiety. Perseveration (choice stickiness)
        is present at both stages and is modulated by anxiety in opposite directions:
        anxiety increases first-stage perseveration and decreases second-stage
        perseveration.
        Parameters
        - action_1: np.array (n_trials,), first-stage actions (0=spaceship A, 1=spaceship U)
        - state:    np.array (n_trials,), second-stage state (0=planet X, 1=planet Y)
        - action_2: np.array (n_trials,), second-stage actions (0/1; aliens)
        - reward:   np.array (n_trials,), outcomes (0/1 coins)
        - stai:     np.array (1,) or (n_trials,), anxiety trait score in [0,1]
        - model_parameters: iterable of 5 parameters, all used
            alpha: [0,1] learning rate for stage-2 MF values and stage-1 MF bootstrapping
            beta:  [0,10] inverse temperature for softmax at both stages
            rho_surp0: [0,1] gain for surprise-gated increase in planning weight
            kappa_rep0: [0,1] strength of first-stage perseveration (repeat last a1)
            zeta_pers2: [0,1] strength of second-stage perseveration (repeat last a2 in a state)
        Returns
        - Negative log-likelihood of observed first- and second-stage choices.
        alpha, beta, rho_surp0, kappa_rep0, zeta_pers2 = model_parameters
        n_trials = len(action_1)
        stai0 = float(stai[0])
        T = np.array([[0.7, 0.3],
                      [0.3, 0.7]])
        q1_mf = np.zeros(2)         # model-free first-stage values
        q2_mf = np.zeros((2, 2))    # second-stage MF values (state x action)
    shared_mechanism_0()
        w_mb = max(0.0, min(1.0, 1.0 - stai0))
        prev_a1 = None
        prev_a2_by_state = {0: None, 1: None}
        eps = 1e-12
        for t in range(n_trials):
            max_q2 = np.max(q2_mf, axis=1)           # best alien per planet
            q1_mb = T @ max_q2                       # expected value per spaceship
            q1_hybrid = w_mb * q1_mb + (1.0 - w_mb) * q1_mf
            if prev_a1 is not None:
                stick = np.zeros(2)
                stick[prev_a1] = 1.0
                kappa_eff = kappa_rep0 * (1.0 + stai0)
                q1_hybrid = q1_hybrid + kappa_eff * stick
            q1c = q1_hybrid - np.max(q1_hybrid)
            probs_1 = np.exp(beta * q1c)
            probs_1 = probs_1 / (np.sum(probs_1) + eps)
            a1 = action_1[t]
            p_choice_1[t] = probs_1[a1]
            s2 = state[t]
            q2 = q2_mf[s2].copy()
            prev_a2 = prev_a2_by_state[s2]
            if prev_a2 is not None:
                stick2 = np.zeros(2)
                stick2[prev_a2] = 1.0
                zeta_eff = zeta_pers2 * (1.0 - 0.5 * stai0)
                q2 = q2 + zeta_eff * stick2
            q2c = q2 - np.max(q2)
            probs_2 = np.exp(beta * q2c)
            probs_2 = probs_2 / (np.sum(probs_2) + eps)
            a2 = action_2[t]
            p_choice_2[t] = probs_2[a2]
            r = reward[t]
            pe2 = r - q2_mf[s2, a2]
            q2_mf[s2, a2] += alpha * pe2
            td_target1 = q2_mf[s2, a2]
            pe1 = td_target1 - q1_mf[a1]
            q1_mf[a1] += alpha * pe1
            p_trans = T[a1, s2]
            surprise = 1.0 - p_trans
            w_mb = w_mb + rho_surp0 * (1.0 + stai0) * (surprise - (w_mb - (1.0 - stai0)))
            w_mb = max(0.0, min(1.0, w_mb))
            prev_a1 = a1
            prev_a2_by_state[s2] = a2
        nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[39] = participant_39

def participant_40(action_1, state, action_2, reward, stai, model_parameters):
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices each trial: 0 = spaceship A, 1 = spaceship U.
        state : array-like of int (0 or 1)
            Second-stage state each trial: 0 = planet X, 1 = planet Y.
        action_2 : array-like of int (0 or 1)
            Second-stage choices each trial (alien on the visited planet).
        reward : array-like of float
            Obtained reward each trial (gold coins; can be negative, zero, or positive).
        stai : array-like of float
            Trait anxiety score in [0,1]; stai[0] is used.
        model_parameters : tuple/list
            (alpha2, beta, tau, kappa1)
            - alpha2 in [0,1]: learning rate for second-stage Q-values.
            - beta in [0,10]: inverse temperature for softmax (both stages).
            - tau in [0,1]: transition learning rate for updating P(state | action).
            - kappa1 in [0,1]: first-stage stickiness strength (bias to repeat previous spaceship).
    shared_mechanism_1()
            Negative log-likelihood of the observed choices.
        Notes
        -----
        - The agent learns the transition matrix T (rows = actions, cols = states) online.
        - Anxiety reduces transition learning: tau_eff = tau * (1 - stai).
        - Stage-1 policy is fully model-based using the learned T and current stage-2 values.
        - Stage-2 values are learned via simple TD with learning rate alpha2.
        alpha2, beta, tau, kappa1 = model_parameters
        n_trials = len(action_1)
        stai_val = float(stai[0])
        T = np.full((2, 2), 0.5)
    shared_mechanism_0()
        Q2 = np.zeros((2, 2))
        prev_a1 = None
        tau_eff = tau * (1.0 - stai_val)
        for t in range(n_trials):
            max_Q2 = np.max(Q2, axis=1)
            Q1_MB = T @ max_Q2
            bias1 = np.zeros(2)
            if prev_a1 is not None:
                bias1[prev_a1] = kappa1
            a1 = int(action_1[t])
            logits1 = beta * Q1_MB + bias1
            logits1 -= np.max(logits1)
            probs1 = np.exp(logits1)
            probs1 /= np.sum(probs1)
            p_choice_1[t] = probs1[a1]
            s2 = int(state[t])
            a2 = int(action_2[t])
            logits2 = beta * Q2[s2]
            logits2 -= np.max(logits2)
            probs2 = np.exp(logits2)
            probs2 /= np.sum(probs2)
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            if tau_eff > 0.0:
                T[a1, :] = (1.0 - tau_eff) * T[a1, :]
                T[a1, s2] += tau_eff
                row_sum = np.sum(T[a1, :])
                if row_sum > 0:
                    T[a1, :] /= row_sum
            delta2 = r - Q2[s2, a2]
            Q2[s2, a2] += alpha2 * delta2
            prev_a1 = a1
    shared_mechanism_3()

models[40] = participant_40

def participant_41(action_1, state, action_2, reward, stai, model_parameters):
        This model combines model-free (MF) and model-based (MB) values at the first stage,
        with an eligibility trace propagating second-stage outcomes back to first-stage values.
        Anxiety (stai) increases the reliance on perseveration and decreases the MB weight.
    shared_mechanism_2()
        action_1 : array-like of int (0 or 1)
            First-stage choices (0=A, 1=U) for each trial.
        state : array-like of int (0 or 1)
            Observed second-stage state (0=X, 1=Y) for each trial.
        action_2 : array-like of int (0 or 1)
            Second-stage choices (per planet; 0 or 1) for each trial.
        reward : array-like of float
            Obtained reward on each trial (e.g., 0.0 or 1.0).
        stai : array-like of float
            Participant's anxiety score in [0,1]; here, single-element array with stai[0].
            Higher stai reduces model-based control and increases perseveration.
        model_parameters : list or array of floats
            [alpha, lambda_, beta, w_base, pers_base]
            - alpha in [0,1]: learning rate for both stages.
            - lambda_ in [0,1]: eligibility trace; propagates second-stage TD error to first stage.
            - beta in [0,10]: inverse temperature for softmax at both stages.
            - w_base in [0,1]: baseline MB/MF mixing weight; anxiety reduces this weight.
            - pers_base in [0,1]: baseline perseveration magnitude; anxiety increases its effect.
    shared_mechanism_1()
            Negative log-likelihood of the observed first- and second-stage choices.
        alpha, lambda_, beta, w_base, pers_base = model_parameters
        n_trials = len(action_1)
        st = stai[0]
        transition_matrix = np.array([[0.7, 0.3],
                                      [0.3, 0.7]])
        q_stage1_mf = np.zeros(2)       # MF values for A/U
        q_stage2_mf = np.zeros((2, 2))  # MF values for second-stage actions in states X/Y
    shared_mechanism_0()
        prev_a1 = -1  # for perseveration bias
        w_eff_scale = 1.0 - 0.5 * st
        w_eff_base = w_base * w_eff_scale
        pers_eff = pers_base * (0.5 + 0.5 * st)
        for t in range(n_trials):
            max_q_stage2 = np.max(q_stage2_mf, axis=1)  # shape (2,)
            q_stage1_mb = transition_matrix @ max_q_stage2  # shape (2,)
            w_eff = w_eff_base
            q1_combined = w_eff * q_stage1_mb + (1.0 - w_eff) * q_stage1_mf
            bias = np.zeros(2)
            if prev_a1 >= 0:
                bias[prev_a1] = pers_eff
            q1_policy_vals = q1_combined + bias
            exp_q1 = np.exp(beta * (q1_policy_vals - np.max(q1_policy_vals)))
            probs_1 = exp_q1 / np.sum(exp_q1)
            a1 = action_1[t]
            p_choice_1[t] = probs_1[a1]
            s = state[t]
            q2_policy_vals = q_stage2_mf[s, :]
            exp_q2 = np.exp(beta * (q2_policy_vals - np.max(q2_policy_vals)))
            probs_2 = exp_q2 / np.sum(exp_q2)
            a2 = action_2[t]
            p_choice_2[t] = probs_2[a2]
            r = reward[t]
            delta2 = r - q_stage2_mf[s, a2]
            q_stage2_mf[s, a2] += alpha * delta2
            delta1_boot = q_stage2_mf[s, a2] - q_stage1_mf[a1]
            q_stage1_mf[a1] += alpha * delta1_boot
            q_stage1_mf[a1] += alpha * lambda_ * delta2
            prev_a1 = a1
    shared_mechanism_3()

models[41] = participant_41

def participant_42(action_1, state, action_2, reward, stai, model_parameters):
        Purely model-free learning with separate learning rates for gains and losses at the
        second stage. A TD(λ) credit assignment to the first stage uses λ = 1 - stai, so higher
        anxiety shortens the eligibility span. An anxiety-modulated lapse (random choice mixing)
        is applied at both stages.
    shared_mechanism_2()
        action_1 : array-like of int
            First-stage choices per trial (0: spaceship A, 1: spaceship U).
        state : array-like of int
            Reached second-stage state per trial (0: planet X, 1: planet Y).
        action_2 : array-like of int
            Second-stage choices per trial within the reached state (0/1).
        reward : array-like of float
            Obtained reward per trial.
        stai : array-like of float
            Anxiety score in [0,1]; higher means higher anxiety.
        model_parameters : sequence of floats
            [alpha_gain, alpha_loss, beta1, beta2, lapse_base]
            Bounds:
            - alpha_gain in [0,1]: learning rate when second-stage PE is positive.
            - alpha_loss in [0,1]: learning rate when second-stage PE is negative or zero.
            - beta1 in [0,10]: inverse temperature for first-stage softmax.
            - beta2 in [0,10]: inverse temperature for second-stage softmax.
            - lapse_base in [0,1]: base lapse mixture; effective lapse increases with anxiety:
              lapse_eff = min(0.5, lapse_base * stai).
    shared_mechanism_1()
            Negative log-likelihood of the observed choices.
        alpha_g, alpha_l, beta1, beta2, lapse_base = model_parameters
        n_trials = len(action_1)
        stai = float(stai[0])
        lam = 1.0 - stai
        lam = min(max(lam, 0.0), 1.0)
        lapse = min(0.5, max(0.0, lapse_base * stai))
        q1 = np.zeros(2)
        q2 = np.zeros((2, 2))
    shared_mechanism_0()
        for t in range(n_trials):
            pref1 = beta1 * q1
            z1 = np.max(pref1)
            exp1 = np.exp(pref1 - z1)
            soft1 = exp1 / np.sum(exp1)
            probs_1 = (1.0 - lapse) * soft1 + lapse * 0.5
            a1 = action_1[t]
            p_choice_1[t] = probs_1[a1]
            s = state[t]
            pref2 = beta2 * q2[s]
            z2 = np.max(pref2)
            exp2 = np.exp(pref2 - z2)
            soft2 = exp2 / np.sum(exp2)
            probs_2 = (1.0 - lapse) * soft2 + lapse * 0.5
            a2 = action_2[t]
            p_choice_2[t] = probs_2[a2]
            r = reward[t]
            pe2 = r - q2[s, a2]
            alpha2 = alpha_g if pe2 > 0.0 else alpha_l
            q2[s, a2] += alpha2 * pe2
            pe1 = q2[s, a2] - q1[a1]
            q1[a1] += alpha2 * (pe1 + lam * pe2)
    shared_mechanism_3()

models[42] = participant_42

def participant_43(action_1, state, action_2, reward, stai, model_parameters):
        Risk-sensitive model-free learner with anxiety-modulated risk aversion and value decay.
        Idea:
        - Stage-2 maintains exponentially weighted estimates of both mean and second moment for each alien.
          Utility is mean - rho * std, where std is derived from the EW second moment.
        - Stage-1 is purely model-free, bootstrapping toward the realized stage-2 utility.
        - STAI modulates risk aversion rho:
            rho = clip(rho_base + k_anx_rho * stai, 0, 1)
          Higher STAI can increase (or decrease) risk aversion depending on k_anx_rho.
        - A decay parameter shrinks unchosen values toward zero, capturing forgetting/instability.
        Parameters (all in [0,1] except beta in [0,10]):
        - model_parameters[0]: alpha (0..1), EW update rate for mean and second moment
        - model_parameters[1]: beta (0..10), inverse temperature at both stages
        - model_parameters[2]: rho_base (0..1), baseline risk aversion
        - model_parameters[3]: k_anx_rho (0..1), slope of STAI effect on risk aversion
        - model_parameters[4]: decay (0..1), per-trial decay applied to unchosen values
        Inputs:
        - action_1: array of ints in {0,1}
        - state: array of ints in {0,1}
        - action_2: array of ints in {0,1}
        - reward: array of floats
        - stai: array-like with one float in [0,1]
        - model_parameters: list/array of 5 parameters as above
        Returns:
        - Negative log-likelihood of observed choices.
        alpha, beta, rho_base, k_anx_rho, decay = model_parameters
        n_trials = len(action_1)
        stai_val = float(stai[0])
        rho = rho_base + k_anx_rho * stai_val
        if rho < 0.0:
            rho = 0.0
        if rho > 1.0:
            rho = 1.0
        m = np.zeros((2, 2))     # EW mean reward
        m2 = np.zeros((2, 2))    # EW mean of squared reward
        q1 = np.zeros(2)
    shared_mechanism_0()
        eps = 1e-10
        for t in range(n_trials):
            var = np.maximum(m2 - m**2, 0.0)
            std = np.sqrt(var + 1e-12)
            u = m - rho * std  # utility
            logits1 = beta * q1
            logits1 -= np.max(logits1)
            soft1 = np.exp(logits1)
            probs1 = soft1 / (np.sum(soft1) + eps)
            a1 = int(action_1[t])
            p_choice_1[t] = probs1[a1]
            s = int(state[t])
            logits2 = beta * u[s]
            logits2 -= np.max(logits2)
            soft2 = np.exp(logits2)
            probs2 = soft2 / (np.sum(soft2) + eps)
            a2 = int(action_2[t])
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            other_a2 = 1 - a2
            m[s, other_a2] = (1.0 - decay) * m[s, other_a2]
            m2[s, other_a2] = (1.0 - decay) * m2[s, other_a2]
            other_state = 1 - s
            m[other_state, 0] = (1.0 - decay) * m[other_state, 0]
            m[other_state, 1] = (1.0 - decay) * m[other_state, 1]
            m2[other_state, 0] = (1.0 - decay) * m2[other_state, 0]
            m2[other_state, 1] = (1.0 - decay) * m2[other_state, 1]
            m[s, a2] = m[s, a2] + alpha * (r - m[s, a2])
            m2[s, a2] = m2[s, a2] + alpha * (r*r - m2[s, a2])
            var_sa = max(m2[s, a2] - m[s, a2]**2, 0.0)
            u_sa = m[s, a2] - rho * np.sqrt(var_sa + 1e-12)
            q1[1 - a1] = (1.0 - decay) * q1[1 - a1]
            delta1 = u_sa - q1[a1]
            q1[a1] = q1[a1] + alpha * delta1
        nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[43] = participant_43

def participant_44(action_1, state, action_2, reward, stai, model_parameters):
        Idea:
        - The agent learns the transition matrix online and uses it for model-based (MB) evaluation.
        - Anxiety increases transition learning rate (hypervigilance) and increases forgetting of Q values.
        - Anxiety decreases inverse temperature (more exploration).
        - First-stage policy arbitrates between MB and model-free (MF) values based on current transition uncertainty and anxiety.
        - MF credit assignment from stage 2 to stage 1 uses an eligibility trace equal to the participant's anxiety (no extra parameter).
        Parameters (with bounds):
        - alpha in [0,1]: learning rate for second-stage Q updates; also used for first-stage MF TD.
        - beta in [0,10]: base inverse temperature.
        - phi_trans in [0,1]: base transition learning rate; anxiety scales it upward.
        - eta_forget in [0,1]: base forgetting rate; anxiety scales it upward (more forgetting under anxiety).
        - xi_anxTemp in [0,1]: strength of anxiety-driven temperature reduction.
        Inputs:
        - action_1: array-like ints in {0,1} for first-stage choices (0=A, 1=U).
        - state: array-like ints in {0,1} for reached planet (0=X, 1=Y).
        - action_2: array-like ints in {0,1} for second-stage choices (0/1 for the two aliens on that planet).
        - reward: array-like floats (typically 0/1).
        - stai: array-like with a single float in [0,1], anxiety score.
        - model_parameters: [alpha, beta, phi_trans, eta_forget, xi_anxTemp].
        Returns:
        - Negative log-likelihood of the observed sequence of choices under the model.
        alpha, beta, phi_trans, eta_forget, xi_anxTemp = model_parameters
        n_trials = len(action_1)
        stai = stai[0]
        T = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)
        beta_eff = max(1e-3, beta * (1.0 - xi_anxTemp * stai))  # higher anxiety -> lower beta
        phi_eff = np.clip(phi_trans * (1.0 + 0.5 * stai), 0.0, 1.0)  # higher anxiety -> faster transition learning
        forget_eff = np.clip(eta_forget * stai, 0.0, 1.0)  # higher anxiety -> more forgetting
        lambda_anx = np.clip(stai, 0.0, 1.0)  # eligibility trace equals anxiety
        q1_mf = np.zeros(2)          # MF Q for A/U
        q2 = np.zeros((2, 2))        # Q for aliens on each planet
    shared_mechanism_0()
        eps = 1e-12
        for t in range(n_trials):
            max_q2 = np.max(q2, axis=1)  # best value on each planet
            q1_mb = T @ max_q2
            row_ent = []
            for r in range(2):
                p = T[r]
                h = 0.0
                for x in p:
                    if x > 0:
                        h -= x * (np.log(x) / np.log(2))
                row_ent.append(h)  # in [0,1] for binary
            ent_mean = 0.5 * (row_ent[0] + row_ent[1])  # 0=certainty, 1=max uncertainty
            w_mb = np.clip((1.0 - stai) * (1.0 - ent_mean), 0.0, 1.0)
            q1 = (1.0 - w_mb) * q1_mf + w_mb * q1_mb
            logits1 = beta_eff * q1
            logits1 -= np.max(logits1)
            probs1 = np.exp(logits1)
            probs1 /= np.sum(probs1)
            a1 = action_1[t]
            p_choice_1[t] = probs1[a1]
            s = state[t]
            logits2 = beta_eff * q2[s, :]
            logits2 -= np.max(logits2)
            probs2 = np.exp(logits2)
            probs2 /= np.sum(probs2)
            a2 = action_2[t]
            p_choice_2[t] = probs2[a2]
            r = reward[t]
            q1_mf = (1.0 - forget_eff) * q1_mf
            q2 = (1.0 - forget_eff) * q2
            td1 = q2[s, a2] - q1_mf[a1]
            q1_mf[a1] += alpha * td1
            td2 = r - q2[s, a2]
            q2[s, a2] += alpha * td2
            q1_mf[a1] += alpha * lambda_anx * td2
            obs = np.array([1.0 if i == s else 0.0 for i in range(2)])
            T[a1, :] = (1.0 - phi_eff) * T[a1, :] + phi_eff * obs
            T[a1, :] = np.clip(T[a1, :], 1e-6, 1.0)
            T[a1, :] /= np.sum(T[a1, :])
        neg_log_lik = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

models[44] = participant_44

