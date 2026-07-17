def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Weighted-cue integration with bias and lapse.

    Description:
    - Computes a value difference as a weighted sum of expert cues (validities 0.9, 0.8, 0.7, 0.6),
      where each cue's contribution is scaled by a subjective weight parameter.
    - Adds a response bias toward option B (as a log-odds bias).
    - Transforms the value difference through a logistic with inverse temperature.
    - Mixes with a lapse rate for random responding.

    Inputs:
    - decisions: array-like of 0/1 choices; 1 indicates choosing option B, 0 indicates choosing option A.
    - A_feature1..A_feature4: arrays of 0/1 expert ratings for option A, ordered by validity 0.9, 0.8, 0.7, 0.6.
    - B_feature1..B_feature4: arrays of 0/1 expert ratings for option B, ordered by validity 0.9, 0.8, 0.7, 0.6.
    - parameters: list or array of 7 parameters:
        w1: [0,1] subjective weight for cue with validity 0.9
        w2: [0,1] subjective weight for cue with validity 0.8
        w3: [0,1] subjective weight for cue with validity 0.7
        w4: [0,1] subjective weight for cue with validity 0.6
        bias_B: [0,1] response bias toward B (converted to log-odds; 0.5 = no bias)
        lapse: [0,1] lapse rate; mixture with uniform choice
        temperature: [0,10] inverse temperature; higher = more deterministic

    Returns:
    - Negative log-likelihood of observed decisions under the model.
    """
    w1, w2, w3, w4, bias_B, lapse, temperature = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    weights = np.array([w1, w2, w3, w4], dtype=float) * validities

    A = np.vstack([A_feature1, A_feature2, A_feature3, A_feature4]).T.astype(float)
    B = np.vstack([B_feature1, B_feature2, B_feature3, B_feature4]).T.astype(float)

    # Value difference: positive means evidence for B
    dv = (B - A) @ weights

    # Convert bias in [0,1] to log-odds bias; 0.5 => 0
    eps = 1e-9
    bias_logodds = np.log((bias_B + eps) / (1.0 - bias_B + eps))

    # Choice probability for B
    logits = temperature * dv + bias_logodds
    p_B = 1.0 / (1.0 + np.exp(-logits))

    # Lapse mixture
    p_B = lapse * 0.5 + (1.0 - lapse) * p_B

    # Clamp to avoid log(0)
    p_B = np.clip(p_B, 1e-12, 1.0 - 1e-12)

    decisions = np.asarray(decisions).astype(float)
    log_likelihood = np.sum(decisions * np.log(p_B) + (1.0 - decisions) * np.log(1.0 - p_B))
    return -float(log_likelihood)


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Stochastic Take-The-Best with sequential adherence, fallback integration, bias, and lapse.

    Description:
    - Processes cues in descending validity (0.9, 0.8, 0.7, 0.6).
    - If a cue discriminates (A_i != B_i), with probability rho_i the decision is made based on that cue
      (choosing the option with the positive rating); otherwise continue to the next cue.
    - If no decision has been made after all cues, a fallback weighted-integration is used
      combining validity weights and equal weights controlled by omega, then passed through a logistic
      with inverse temperature and bias.
    - A lapse rate mixes in random choice.

    Inputs:
    - decisions: array-like of 0/1 choices; 1 indicates choosing option B.
    - A_feature1..A_feature4: arrays of 0/1 expert ratings for option A, ordered by validity 0.9..0.6.
    - B_feature1..B_feature4: arrays of 0/1 expert ratings for option B, ordered by validity 0.9..0.6.
    - parameters: list or array of 8 parameters:
        rho1: [0,1] probability to decide on the most valid discriminating cue (0.9) when it discriminates
        rho2: [0,1] probability to decide on the second cue (0.8) when it discriminates and no decision yet
        rho3: [0,1] probability to decide on the third cue (0.7)
        rho4: [0,1] probability to decide on the fourth cue (0.6)
        omega: [0,1] mixing between validity weights and equal weights in fallback (0=equal, 1=validity)
        bias_B: [0,1] response bias toward B in fallback stage (converted to log-odds)
        lapse: [0,1] lapse rate; mixture with uniform choice
        temperature: [0,10] inverse temperature in fallback logistic

    Returns:
    - Negative log-likelihood of observed decisions under the model.
    """
    rho1, rho2, rho3, rho4, omega, bias_B, lapse, temperature = parameters
    rho = np.array([rho1, rho2, rho3, rho4], dtype=float)

    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    equal_w = np.ones(4, dtype=float) / 4.0
    fallback_w = omega * validities + (1.0 - omega) * equal_w

    A = np.vstack([A_feature1, A_feature2, A_feature3, A_feature4]).T.astype(float)
    B = np.vstack([B_feature1, B_feature2, B_feature3, B_feature4]).T.astype(float)

    n = len(decisions)
    p_B = np.zeros(n, dtype=float)

    eps = 1e-9
    bias_logodds = np.log((bias_B + eps) / (1.0 - bias_B + eps))

    for t in range(n):
        a = A[t]
        b = B[t]
        disc = (a != b).astype(int)  # 1 if discriminates
        # Identify which option is favored by each discriminating cue: +1 => favors B, -1 => favors A
        favors = np.where(b > a, 1, -1)

        # Probability mass that reaches each cue without having decided earlier
        mass_reaching = 1.0
        p_B_t = 0.0

        for i in range(4):
            if disc[i] == 1:
                # Decide at cue i with probability rho[i]
                decide_prob = mass_reaching * rho[i]
                if favors[i] == 1:
                    p_B_t += decide_prob
                # If favors A, no addition to p_B_t (implicitly adds to P(A))
                # Survive to next cue if either non-discriminating or decide not to choose at this cue
                mass_reaching *= (1.0 - rho[i])
            else:
                # Non-discriminating cue: pass through unchanged
                mass_reaching *= 1.0

        # Fallback if still undecided after all cues
        if mass_reaching > 0.0:
            dv_fb = np.dot((b - a), fallback_w)  # positive => evidence for B
            logit_fb = temperature * dv_fb + bias_logodds
            p_B_fb = 1.0 / (1.0 + np.exp(-logit_fb))
            p_B_t += mass_reaching * p_B_fb

        p_B[t] = p_B_t

    # Lapse mixture
    p_B = lapse * 0.5 + (1.0 - lapse) * p_B
    p_B = np.clip(p_B, 1e-12, 1.0 - 1e-12)

    decisions = np.asarray(decisions).astype(float)
    log_likelihood = np.sum(decisions * np.log(p_B) + (1.0 - decisions) * np.log(1.0 - p_B))
    return -float(log_likelihood)


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Redundancy-sensitive reliability transform with nonlinearity, bias, and lapse.

    Description:
    - Builds subjective cue weights from stated validities (0.9, 0.8, 0.7, 0.6) via:
        w_i = s * (validity_i ** exp) + (1 - s) * rank_norm_i,
      where exp = 0.5 + 1.5 * gamma, s in [0,1], and rank_norm are normalized ranks (descending).
    - Applies redundancy discount: when multiple cues agree on the same option, each agreeing cue past the first
      is down-weighted by (1 - delta) per additional agreeing cue.
    - Computes a value difference as the redundancy-discounted weighted sum of (B_i - A_i).
    - Adds a response bias toward B in log-odds space and transforms via a logistic with inverse temperature.
    - Mixes with a lapse rate for random responding.

    Inputs:
    - decisions: array-like of 0/1 choices; 1 indicates choosing option B.
    - A_feature1..A_feature4: arrays of 0/1 expert ratings for option A, ordered by validity 0.9..0.6.
    - B_feature1..B_feature4: arrays of 0/1 expert ratings for option B, ordered by validity 0.9..0.6.
    - parameters: list or array of 6 parameters:
        s: [0,1] mixing of transformed validities vs rank-based weights (0=rank only, 1=validity only)
        gamma: [0,1] controls nonlinearity exponent on validities: exp = 0.5 + 1.5*gamma
        delta: [0,1] redundancy discount per additional agreeing cue (0=no discount, 1=only first cue counts)
        bias_B: [0,1] response bias toward B (converted to log-odds)
        lapse: [0,1] lapse rate; mixture with uniform choice
        temperature: [0,10] inverse temperature; higher = more deterministic

    Returns:
    - Negative log-likelihood of observed decisions under the model.
    """
    s, gamma, delta, bias_B, lapse, temperature = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    # Rank-based normalized weights (descending validity ranks)
    ranks = np.array([4, 3, 2, 1], dtype=float)
    rank_norm = ranks / np.sum(ranks)

    expv = 0.5 + 1.5 * gamma
    trans_validities = validities ** expv
    base_w = s * trans_validities + (1.0 - s) * rank_norm

    A = np.vstack([A_feature1, A_feature2, A_feature3, A_feature4]).T.astype(float)
    B = np.vstack([B_feature1, B_feature2, B_feature3, B_feature4]).T.astype(float)

    n = len(decisions)
    dv = np.zeros(n, dtype=float)

    for t in range(n):
        a = A[t]
        b = B[t]

        # Identify agreeing cues for each option
        agree_B = (b == 1) & (a == 0)
        agree_A = (a == 1) & (b == 0)

        n_agree_B = int(np.sum(agree_B))
        n_agree_A = int(np.sum(agree_A))

        # Compute redundancy multipliers for each agreeing cue:
        # The first agreeing cue gets factor 1.0, the k-th gets (1 - delta)^(k-1).
        # We assign in order of decreasing base weight (more valid cues treated as earlier).
        idx_sorted = np.argsort(-base_w)  # indices in descending weight order

        # Initialize per-cue multipliers
        mult = np.ones(4, dtype=float)

        # Apply discounts among agreeing-for-B cues
        count = 0
        for i in idx_sorted:
            if agree_B[i]:
                mult[i] = (1.0 - delta) ** count
                count += 1

        # Apply discounts among agreeing-for-A cues
        count = 0
        for i in idx_sorted:
            if agree_A[i]:
                mult[i] = (1.0 - delta) ** count
                count += 1

        # Value difference with redundancy discounts
        dv[t] = np.sum((b - a) * (base_w * mult))

    eps = 1e-9
    bias_logodds = np.log((bias_B + eps) / (1.0 - bias_B + eps))

    logits = temperature * dv + bias_logodds
    p_B = 1.0 / (1.0 + np.exp(-logits))

    # Lapse mixture
    p_B = lapse * 0.5 + (1.0 - lapse) * p_B
    p_B = np.clip(p_B, 1e-12, 1.0 - 1e-12)

    decisions = np.asarray(decisions).astype(float)
    log_likelihood = np.sum(decisions * np.log(p_B) + (1.0 - decisions) * np.log(1.0 - p_B))
    return -float(log_likelihood)