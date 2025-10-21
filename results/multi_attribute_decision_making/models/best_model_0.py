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

        favors = np.where(b > a, 1, -1)

        mass_reaching = 1.0
        p_B_t = 0.0

        for i in range(4):
            if disc[i] == 1:

                decide_prob = mass_reaching * rho[i]
                if favors[i] == 1:
                    p_B_t += decide_prob


                mass_reaching *= (1.0 - rho[i])
            else:

                mass_reaching *= 1.0

        if mass_reaching > 0.0:
            dv_fb = np.dot((b - a), fallback_w)  # positive => evidence for B
            logit_fb = temperature * dv_fb + bias_logodds
            p_B_fb = 1.0 / (1.0 + np.exp(-logit_fb))
            p_B_t += mass_reaching * p_B_fb

        p_B[t] = p_B_t

    p_B = lapse * 0.5 + (1.0 - lapse) * p_B
    p_B = np.clip(p_B, 1e-12, 1.0 - 1e-12)

    decisions = np.asarray(decisions).astype(float)
    log_likelihood = np.sum(decisions * np.log(p_B) + (1.0 - decisions) * np.log(1.0 - p_B))
    return -float(log_likelihood)