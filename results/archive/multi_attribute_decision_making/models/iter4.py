def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Reliability-weighted congruence accumulator with asymmetric gain, prior bias, and lapses.
    
    Mechanism:
    - Each cue provides signed evidence favoring option B (+1), A (-1), or neither (0).
    - Cue weights are a power transform of expert validities, allowing under/over-reliance on expertise.
    - Evidence is asymmetric: positive (pro-B) and negative (pro-A) signs can be amplified differently.
    - A congruence gain scales evidence magnitude when cues are more/less mutually consistent.
    - A prior bias toward B shifts the decision variable.
    - A logistic choice with inverse temperature maps evidence to choice probability; a lapse rate mixes with random choice.
    
    Parameters (all used):
    - kappa: [0,1] reliability shaping exponent; 0 = equal weights, 1 = veridical weighting by validity
    - pos_gain: [0,1] asymmetric amplification of pro-B vs pro-A evidence. Positive signs get (1+pos_gain), negative get (1-pos_gain)
    - congruency: [0,1] scales the magnitude of evidence by the degree of cue agreement (from down- to up-weighting)
    - bias0: [0,1] prior bias toward B, mapped to centered bias in [-0.5, +0.5] and added to the evidence
    - epsilon: [0,1] lapse probability mixing with uniform choice
    - temperature: [0,10] inverse temperature scaling of the decision variable
    
    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B, 0 = chose A)
    - A_feature1..4, B_feature1..4: arrays of 0/1 cue values for options A and B
    
    Returns:
    - negative log-likelihood of the observed decisions under the model
    """
    kappa, pos_gain, congruency, bias0, epsilon, temperature = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    # Power-shaped reliability weights (kappa=0 -> equal; kappa=1 -> validities)
    rel = validities ** np.clip(kappa, 0.0, 1.0)
    rel = rel / np.sum(rel)

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)

    # Signed cues: +1 favors B, -1 favors A, 0 ties
    S = np.sign(B - A)

    decisions = np.asarray(decisions).astype(float)
    n_trials = len(decisions)
    nll = 0.0

    # Prior bias centered around 0; mapped from [0,1] to [-0.5, +0.5]
    prior_bias = (bias0 - 0.5)

    for t in range(n_trials):
        s = S[t]

        # Asymmetric gain: amplify positive vs negative signed cues differently
        gain_vec = np.where(s > 0, 1.0 + pos_gain, np.where(s < 0, 1.0 - pos_gain, 0.0))
        weighted = rel * s * gain_vec

        # Base evidence (compensatory sum over cues)
        ev_base = np.sum(weighted)

        # Congruence factor: proportion of nonzero cues that agree with the majority sign
        nz = (s != 0)
        if np.any(nz):
            maj_sign = np.sign(np.sum(s))
            agree_prop = np.mean((s[nz] == maj_sign).astype(float))
            # Map agree_prop in [0.5,1] (or [0,1] if few) to a scaling in [1-congruency, 1+congruency]
            # Center around 0.5 agreement -> scale ~1
            scale = 1.0 + congruency * (2.0 * agree_prop - 1.0)
        else:
            scale = 1.0  # no information

        # Final decision variable with prior bias
        dv = temperature * (scale * ev_base + prior_bias)

        # Logistic choice prob for B with lapse
        pB = 1.0 / (1.0 + np.exp(-dv))
        pB = (1.0 - epsilon) * pB + 0.5 * epsilon

        # Likelihood contribution
        p_obs = decisions[t] * pB + (1.0 - decisions[t]) * (1.0 - pB)
        p_obs = np.clip(p_obs, 1e-12, 1.0)
        nll += -np.log(p_obs)

    return nll


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Noisy quasi-Bayesian integration with miscalibrated reliability, interaction boost, indecision threshold, and lapses.
    
    Mechanism:
    - Cue reliabilities are blended between equal weighting and stated validities.
    - Each cue can be misread (sign flipped) with some probability.
    - Agreement among the two most valid cues adds an interaction boost to the evidence.
    - If the absolute evidence is below a threshold, the model is indecisive and falls back to a baseline preference for B.
    - Otherwise, a logistic with inverse temperature maps the evidence to choice probability; lapse mixes with uniform.
    
    Parameters (all used):
    - reliance: [0,1] reliance on expert validities vs equal weighting (0 = equal, 1 = full validity)
    - interaction: [0,1] additional weight added when the two highest-validity cues agree (scaled by their average weight)
    - flip: [0,1] probability of misreading a cue (sign flip)
    - threshold: [0,1] indecision threshold on absolute evidence (post-noise, pre-logistic)
    - prefB: [0,1] baseline probability of choosing B when indecisive
    - temperature: [0,10] inverse temperature for mapping evidence to choice probability
    
    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B)
    - A_feature1..4, B_feature1..4: arrays of 0/1 features
    
    Returns:
    - negative log-likelihood under the model
    """
    reliance, interaction, flip, threshold, prefB, temperature = parameters
    val = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    eq = np.ones_like(val) / len(val)
    # Miscalibrated reliability blend
    w = reliance * val + (1.0 - reliance) * eq
    w = w / np.sum(w)

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    S = np.sign(B - A)

    decisions = np.asarray(decisions).astype(float)
    n_trials = len(decisions)
    nll = 0.0

    # Precompute indices of the two most valid cues (0 and 1 in validity order)
    idx_sorted = np.argsort(-val)  # descending validity
    top1, top2 = idx_sorted[0], idx_sorted[1]

    # Effective sign attenuation due to flips: E[sign_after] = sign * (1 - 2 flip)
    att = (1.0 - 2.0 * flip)

    for t in range(n_trials):
        s = S[t]

        # Expected noisy evidence
        ev = np.sum(w * s * att)

        # Interaction boost if top two agree in sign (both nonzero and equal)
        if (s[top1] != 0) and (s[top2] != 0) and (np.sign(s[top1]) == np.sign(s[top2])):
            # Boost direction follows their sign; magnitude scaled by average of their weights
            boost = interaction * 0.5 * (w[top1] + w[top2]) * np.sign(s[top1])
        else:
            boost = 0.0

        dv_raw = ev + boost

        # Indecision rule before temperature: if |evidence| small, use prefB; else logistic on scaled dv
        if abs(dv_raw) < threshold:
            pB = prefB
        else:
            dv = temperature * dv_raw
            pB = 1.0 / (1.0 + np.exp(-dv))

        # Lapse is implicitly captured if prefB near 0.5; but to use all params strictly as specified,
        # we do not include an extra lapse here beyond the indecision mechanism.

        p_obs = decisions[t] * pB + (1.0 - decisions[t]) * (1.0 - pB)
        p_obs = np.clip(p_obs, 1e-12, 1.0)
        nll += -np.log(p_obs)

    return nll


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Stochastic elimination-by-aspects with dynamic attention persistence, saliency, noisy cue reading, and stickiness.
    
    Mechanism:
    - The model allocates attention over cues each trial, starting from a base (validity-proportional) distribution.
    - Attention persists across trials toward the most attended cue from the previous trial.
    - Current-trial saliency increases attention to cues that differ between A and B.
    - A cue is sampled according to attention; if it discriminates, its (possibly flipped) sign is used and the process halts with probability 'halt'; otherwise the model continues sampling among remaining cues.
    - The expected signed evidence from the first effective discriminating cue drives choice via a logistic with inverse temperature.
    - Response stickiness pulls the probability toward the previous observed choice.
    
    Parameters (all used):
    - persistence: [0,1] carryover of attention toward last attended cue (0 = no carryover, 1 = full persistence)
    - saliency: [0,1] boosts attention to currently discriminating cues in proportion to their absolute difference
    - halt: [0,1] probability of stopping upon encountering a discriminating cue (1 = strict take-the-first)
    - flip_p: [0,1] probability of misreading a discriminating cue (sign flip)
    - stick: [0,1] response stickiness toward previous observed choice
    - temperature: [0,10] inverse temperature scaling of the expected evidence
    
    Inputs:
    - decisions: array-like of 0/1 choices (1 = chose B)
    - A_feature1..4, B_feature1..4: arrays of 0/1 features
    
    Returns:
    - negative log-likelihood under the model
    """
    persistence, saliency, halt, flip_p, stick, temperature = parameters

    val = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    base_attn = val / np.sum(val)

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)
    S = np.sign(B - A)  # +1 favors B, -1 favors A, 0 tie
    D = np.abs(B - A)   # saliency: 1 if discriminating, 0 if tie

    decisions = np.asarray(decisions).astype(float)
    n_trials = len(decisions)
    nll = 0.0

    # Initialize previous attention focus as base (no single cue focus)
    prev_focus = None
    prev_choice = 0.5  # neutral initial stickiness anchor

    for t in range(n_trials):
        s = S[t]
        d = D[t]

        # Construct prior attention distribution with persistence toward previous focus
        if prev_focus is None:
            prior_attn = base_attn.copy()
        else:
            one_hot = np.zeros(4, dtype=float)
            one_hot[prev_focus] = 1.0
            prior_attn = (1.0 - persistence) * base_attn + persistence * one_hot

        # Apply current-trial saliency modulation and renormalize
        attn = prior_attn * (1.0 + saliency * d)
        if np.sum(attn) > 0:
            attn = attn / np.sum(attn)
        else:
            attn = prior_attn.copy()  # fallback if all zero (shouldn't happen)

        # Expected signed evidence from the first effective discriminating cue with probabilistic stopping
        remaining = np.ones(4, dtype=bool)
        expected_ev = 0.0
        mass_continue = 1.0
        attn_curr = attn.copy()
        # Effective sign attenuation from misread
        sign_factor = (1.0 - 2.0 * flip_p)

        # We'll remember the most attended cue at the start of the trial for persistence update
        focus_idx = int(np.argmax(attn))

        for step in range(4):
            idxs = np.where(remaining)[0]
            if len(idxs) == 0:
                break

            # Selection probabilities restricted to remaining cues
            sel_probs = attn_curr[idxs]
            if np.sum(sel_probs) > 0:
                sel_probs = sel_probs / np.sum(sel_probs)
            else:
                sel_probs = np.ones_like(sel_probs) / len(sel_probs)

            signs = s[idxs]

            # Expected evidence contribution at this step: only from discriminating cues, with stop prob
            step_ev = np.sum(sel_probs * (signs != 0).astype(float) * halt * signs * sign_factor)
            expected_ev += mass_continue * step_ev

            # Probability of continuing to next step
            cont_probs = sel_probs * ( (signs == 0).astype(float) + (signs != 0).astype(float) * (1.0 - halt) )
            cont_prob = np.sum(cont_probs)
            mass_continue *= cont_prob

            # Remove the most likely inspected cue from remaining for the next step
            chosen_local = int(np.argmax(sel_probs))
            chosen_global = idxs[chosen_local]
            remaining[chosen_global] = False

            # Zero out its attention and renormalize for next iteration
            attn_curr = attn_curr.copy()
            attn_curr[chosen_global] = 0.0
            if np.any(remaining):
                attn_curr = attn_curr / np.sum(attn_curr[remaining])
                attn_curr[~remaining] = 0.0

        # Map expected evidence to choice probability with stickiness and temperature
        dv = temperature * expected_ev
        pB = 1.0 / (1.0 + np.exp(-dv))
        pB = (1.0 - stick) * pB + stick * prev_choice

        p_obs = decisions[t] * pB + (1.0 - decisions[t]) * (1.0 - pB)
        p_obs = np.clip(p_obs, 1e-12, 1.0)
        nll += -np.log(p_obs)

        # Update persistence focus and previous choice
        prev_focus = focus_idx
        prev_choice = decisions[t]

    return nll