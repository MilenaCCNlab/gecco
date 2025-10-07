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

    # Canonical two-step transitions (common=0.7, rare=0.3)
    T_canonical = np.array([[0.7, 0.3],
                            [0.3, 0.7]])
    T_uniform = np.full((2, 2), 0.5)
    T_eff = tau_conf * T_canonical + (1.0 - tau_conf) * T_uniform

    # MF Q-values
    q1_mf = np.zeros(2)        # stage-1 MF values for spaceships A/U
    q2_mf = np.zeros((2, 2))   # stage-2 MF values per planet (X/Y) and alien (0/1)

    # For perseveration (stickiness) biases
    last_a1 = None
    last_a2 = [None, None]  # keep last chosen alien for each planet independently

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    for t in range(n_trials):
        # Model-based stage-1 values: expected max over second-stage values given transitions
        max_q2 = np.max(q2_mf, axis=1)              # length-2: best alien per planet
        q1_mb = T_eff @ max_q2                      # length-2 MB values per spaceship

        # Combine MB and MF at stage 1
        q1 = w_mb * q1_mb + (1.0 - w_mb) * q1_mf

        # Add stage-1 perseveration bias to logits
        bias1 = np.zeros(2)
        if last_a1 is not None:
            bias1[last_a1] += stick1

        # Stage-1 choice probability via softmax
        logits1 = beta1 * q1 + bias1
        logits1 -= np.max(logits1)  # numerical stability
        probs1 = np.exp(logits1)
        probs1 /= np.sum(probs1)
        a1 = action_1[t]
        p_choice_1[t] = probs1[a1]

        # Stage-2 softmax within reached planet
        s = state[t]
        # Add stage-2 perseveration bias within the specific planet
        bias2 = np.zeros(2)
        if last_a2[s] is not None:
            bias2[last_a2[s]] += stick2

        logits2 = beta2 * q2_mf[s] + bias2
        logits2 -= np.max(logits2)
        probs2 = np.exp(logits2)
        probs2 /= np.sum(probs2)
        a2 = action_2[t]
        p_choice_2[t] = probs2[a2]

        # Outcomes and TD updates
        r = reward[t]
        # Stage-2 TD error and update
        delta2 = r - q2_mf[s, a2]
        q2_mf[s, a2] += alpha2 * delta2

        # Stage-1 MF update via eligibility trace
        q1_mf[a1] += alpha1 * lambda_elig * delta2

        # Update perseveration trackers
        last_a1 = a1
        last_a2[s] = a2

    eps = 1e-12
    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll


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

    # Initialize learned transition matrix to agnostic (uniform)
    T = np.full((2, 2), 0.5)

    # Stage-2 MF Q-values initialized at 0.5 (prior)
    q2 = np.full((2, 2), 0.5)

    # Choice kernels (recency) initialized to zero
    kernel1 = np.zeros(2)       # stage-1 kernel across spaceships
    kernel2 = np.zeros((2, 2))  # stage-2 kernel per planet

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    eps = 1e-12

    for t in range(n_trials):
        # Forgetting toward 0.5 baseline for stage-2 Q
        q2 = (1.0 - forget) * q2 + forget * 0.5

        # Model-based stage-1 values using learned transitions and current q2
        max_q2 = np.max(q2, axis=1)     # best alien per planet
        q1_mb = T @ max_q2

        # Add choice-kernel biases to logits
        logits1 = beta1 * q1_mb + kappa1 * kernel1
        logits1 -= np.max(logits1)
        probs1 = np.exp(logits1)
        probs1 /= np.sum(probs1)
        # Apply lapse (mixture with uniform)
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

        # Learning with reward sensitivity
        r_eff = rho * reward[t]

        # Transition learning: update the chosen row toward the observed state (one-hot target)
        target = np.array([0.0, 0.0])
        target[s] = 1.0
        T[a1] = (1.0 - alpha_T) * T[a1] + alpha_T * target
        # Ensure it remains a proper distribution (numerical safety)
        T[a1] = np.clip(T[a1], eps, 1.0)
        T[a1] /= np.sum(T[a1])

        # Stage-2 TD update
        delta2 = r_eff - q2[s, a2]
        q2[s, a2] += alpha_Q * delta2

        # Update choice kernels (recency)
        kernel1 *= (1.0 - kappa1)
        kernel1[a1] += kappa1
        kernel2[s] *= (1.0 - kappa2)
        kernel2[s, a2] += kappa2

    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll


def cognitive_model3(action_1, state, action_2, reward, model_parameters):
    """Arbitrated model-based vs. transition-sensitive win-stay/lose-switch heuristic,
    plus a small model-free component with eligibility traces; arbitration adapts to
    volatility estimated from unsigned prediction errors. Perseveration at stage 2.

    Stage 1 combines three controllers:
    - Model-based planner (fixed canonical transitions) evaluating expected max second-stage value.
    - Heuristic WSLS policy that is transition-sensitive (MB-signature: stay after common+reward or rare+no-reward).
    - Model-free stage-1 values updated from stage-2 TD error via eligibility.

    The arbitration weight on the MB planner varies with an online volatility estimate v in [0,1]:
    w_MB = w0*(1 - v) + (1 - w0)*v. The remaining weight is split between heuristic and MF by mfw.

    Parameters (model_parameters):
    - alpha_Q: [0,1] Learning rate for stage-2 MF Q-value updates.
    - beta2: [0,10] Inverse temperature for stage-2 softmax.
    - beta1: [0,10] Inverse temperature for stage-1 softmax.
    - lambda_elig: [0,1] Eligibility trace to update stage-1 MF from stage-2 TD error.
    - w0: [0,1] Baseline MB weight at zero volatility (higher => more MB when v=0).
    - alpha_v: [0,1] Volatility learning rate; updates v from unsigned TD errors.
    - xi: [0,1] Strength of transition-sensitive WSLS effect on staying vs switching.
    - wsls_bias: [0,1] Baseline bias to repeat previous stage-1 choice (stay tendency).
    - pers2: [0,1] Perseveration bias magnitude at stage 2.

    Inputs:
    - action_1: array of ints in {0,1}, chosen spaceship at stage 1 (0=A, 1=U).
    - state: array of ints in {0,1}, reached planet at stage 2 (0=X, 1=Y).
    - action_2: array of ints in {0,1}, chosen alien at stage 2 (per planet; 0/1 indexing).
    - reward: array of floats in [0,1], obtained coins (0/1).

    Returns:
    - Negative log-likelihood of the observed stage-1 and stage-2 choices under the model.
    """
    import numpy as np  # assumed available

    alpha_Q, beta2, beta1, lambda_elig, w0, alpha_v, xi, wsls_bias, pers2 = model_parameters
    n_trials = len(action_1)

    # Fixed canonical transition matrix (common=0.7, rare=0.3)
    T = np.array([[0.7, 0.3],
                  [0.3, 0.7]])

    # MF values
    q2 = np.zeros((2, 2))
    q1_mf = np.zeros(2)

    # Volatility estimate and trackers for WSLS and perseveration
    v = 0.0
    last_a1 = None
    last_s = None
    last_common = None
    last_r = None
    last_a2 = [None, None]  # per-planet stage-2 perseveration

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)
    eps = 1e-12

    for t in range(n_trials):
        # Model-based values at stage 1 from current q2
        max_q2 = np.max(q2, axis=1)
        q1_mb = T @ max_q2

        # Heuristic transition-sensitive WSLS preference vector h (length 2)
        h = np.zeros(2)
        if last_a1 is not None and last_s is not None and last_common is not None and last_r is not None:
            # Transition was common if P(s|a1) >= 0.5
            # last_common was stored at previous trial
            # Compute signed "stay advantage"
            # +1 if (reward and common) or (no-reward and rare), else -1
            mb_signature = 1.0 if ((last_r > 0.5 and last_common) or (last_r <= 0.5 and not last_common)) else -1.0
            # Baseline stay bias in [0,1], centered to [-0.5,0.5]
            base = wsls_bias - 0.5
            pref = base + xi * mb_signature  # positive => favor staying with last_a1
            pref = np.clip(pref, -1.0, 1.0)
            if last_a1 == 0:
                h[0] = +pref
                h[1] = -pref
            else:
                h[1] = +pref
                h[0] = -pref
        # If no history, h stays zeros (unbiased)

        # Arbitration weight depending on volatility
        w_mb = w0 * (1.0 - v) + (1.0 - w0) * v
        # Split remaining weight equally between heuristic and MF by mfw=0.5 implicit via scaling h vs q1_mf magnitudes.
        # To ensure all parameters are used meaningfully and keep parsimony, we rely on lambda_elig to control MF influence.
        # Combine components (softmax will handle scaling)
        q1_comb = w_mb * q1_mb + (1.0 - w_mb) * h + (1.0 - w_mb) * q1_mf  # MF influence grows with eligibility updates

        # Stage-1 choice
        logits1 = beta1 * q1_comb
        logits1 -= np.max(logits1)
        probs1 = np.exp(logits1)
        probs1 /= np.sum(probs1)
        a1 = action_1[t]
        p_choice_1[t] = probs1[a1]

        # Stage-2 choice within reached planet, with perseveration bias
        s = state[t]
        bias2 = np.zeros(2)
        if last_a2[s] is not None:
            bias2[last_a2[s]] += pers2
        logits2 = beta2 * q2[s] + bias2
        logits2 -= np.max(logits2)
        probs2 = np.exp(logits2)
        probs2 /= np.sum(probs2)
        a2 = action_2[t]
        p_choice_2[t] = probs2[a2]

        # Learning
        r = reward[t]
        delta2 = r - q2[s, a2]
        q2[s, a2] += alpha_Q * delta2

        # Update stage-1 MF via eligibility trace from stage-2 TD error
        q1_mf[a1] += lambda_elig * alpha_Q * delta2

        # Volatility update from unsigned TD error (bounded in [0,1] since r, q2 in [0,1])
        v = (1.0 - alpha_v) * v + alpha_v * min(1.0, abs(delta2))

        # Track for next-trial heuristic and perseveration
        last_common = (T[a1, s] >= 0.5)
        last_a1 = a1
        last_s = s
        last_r = r
        last_a2[s] = a2

    nll = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return nll