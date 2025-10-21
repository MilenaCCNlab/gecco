def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Choice-driven adaptive attention (online cue-weight learning with decay and asymmetry).
    
    Idea:
    - The decision maker maintains internal attention/weight on each cue and updates it trial-by-trial based on
      the chosen option (choice-driven reinforcement of supportive cues).
    - Cues that support the chosen option are reinforced; cues that contradict the chosen option are penalized
      with an asymmetric strength.
    - A small decay implements forgetting. A softmax/logit transforms the weighted evidence into a choice
      probability with an inverse temperature.
    
    Decision rule per trial t:
    - Evidence S_t = sum_i w_{t,i} * (B_i - A_i).
    - P(B)_t = sigmoid(beta * S_t).
    - Update after observing decision d_t in {0,1}:
        pe_t = d_t - P(B)_t
        w_{t+1,i} = (1 - decay) * w_{t,i} + alpha * pe_t * g_i(d_t, A_i, B_i)
      where g_i scales the update by cue direction and asymmetry:
        dir_i = sign(B_i - A_i) in {-1,0,1}
        agree_with_choice = 1 if dir_i == +1 when d_t=1 or dir_i == -1 when d_t=0, else 0
        disagree_with_choice = 1 - agree_with_choice (for |dir_i|==1; 0 when dir_i==0)
        g_i = dir_i * [agree_with_choice + lambda_loss * disagree_with_choice]
      No update if the cue does not discriminate (dir_i=0).
    
    Initialization of w_0:
    - Validity vector v = [0.9, 0.8, 0.7, 0.6].
    - Prior blending: w_0 = prior_bias * v + (1 - prior_bias) * 0.5 for each cue.
      This sets a prior attention near 0.5 and pulls toward known validities as prior_bias increases.
    
    Parameters (all are used):
    - alpha: [0,1] learning rate for attention updates.
    - decay: [0,1] exponential forgetting of weights each trial.
    - prior_bias: [0,1] blend between uniform 0.5 and validity prior; higher = more aligned with validities.
    - lambda_loss: [0,1] asymmetry factor for penalizing cues that contradict the chosen option
                   (0 = ignore contradictory cues in updates; 1 = symmetric reinforcement/penalty).
    - beta: [0,10] inverse temperature scaling of the decision evidence.
    
    Returns:
    - Negative log-likelihood of the observed choices under the model.
    """
    alpha, decay, prior_bias, lambda_loss, beta = parameters

    v = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    # Initialize weights with a blend of validity prior and uniform 0.5
    w = prior_bias * v + (1.0 - prior_bias) * 0.5

    # Prepare arrays
    A1 = np.asarray(A_feature1, dtype=float)
    A2 = np.asarray(A_feature2, dtype=float)
    A3 = np.asarray(A_feature3, dtype=float)
    A4 = np.asarray(A_feature4, dtype=float)
    B1 = np.asarray(B_feature1, dtype=float)
    B2 = np.asarray(B_feature2, dtype=float)
    B3 = np.asarray(B_feature3, dtype=float)
    B4 = np.asarray(B_feature4, dtype=float)
    D = np.asarray(decisions, dtype=float)

    n = len(D)
    nll = 0.0
    eps = 1e-12

    for t in range(n):
        d_vec = np.array([
            B1[t] - A1[t],
            B2[t] - A2[t],
            B3[t] - A3[t],
            B4[t] - A4[t]
        ], dtype=float)

        S = float(np.dot(w, d_vec))
        pB = 1.0 / (1.0 + np.exp(-beta * S))
        pB = np.clip(pB, eps, 1.0 - eps)

        # Accumulate NLL
        if D[t] == 1.0:
            nll -= np.log(pB)
        else:
            nll -= np.log(1.0 - pB)

        # Compute direction and agreement for update
        dir_vec = np.sign(d_vec)  # {-1,0,1}
        agree_choice = ((D[t] == 1.0) & (dir_vec == 1)) | ((D[t] == 0.0) & (dir_vec == -1))
        disagree_choice = ((D[t] == 1.0) & (dir_vec == -1)) | ((D[t] == 0.0) & (dir_vec == 1))

        agree_choice = agree_choice.astype(float)
        disagree_choice = disagree_choice.astype(float)

        g = dir_vec * (agree_choice + lambda_loss * disagree_choice)

        pe = D[t] - pB  # signed prediction error
        # Decay + learning update (no change for nondiscriminating cues where dir=0 -> g=0)
        w = (1.0 - decay) * w + alpha * pe * g

    return float(nll)


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Perceptual-noise tally with asymmetric valuation and redundancy discounting.
    
    Idea:
    - Features are perceived with cue-specific misread probabilities (encoding noise).
      A flip probability epsilon_i corrupts each cue before integration.
    - The expected internal evidence from each cue becomes (1 - 2*epsilon_i) * (B_i - A_i).
    - Negative evidence (favoring A) can be weighted differently from positive evidence (favoring B).
    - A redundancy penalty discounts the impact of multiple agreeing cues on the same trial.
    - Weights are anchored to cue validities with a shrinkage parameter.
    - A logistic choice rule with inverse temperature maps evidence to P(B).
    
    Evidence per trial:
      d_i = B_i - A_i in {-1,0,1}
      e_i = (1 - 2*epsilon_i) * d_i
      Split into positive and negative parts with asymmetry kappa:
        e_i_asym = max(e_i, 0) - kappa * max(-e_i, 0)
      Base weights from validities with shrinkage omega:
        w_i = omega * v_i + (1 - omega) * mean(v)
      Redundancy discount:
        When summing across cues in descending validity order, if a cue's signed evidence
        has the same sign as the cumulative evidence so far, its contribution is divided by
        (1 + delta * n_agree_before), where n_agree_before counts prior same-sign contributions.
    
    Parameters (all are used):
    - omega: [0,1] shrinkage toward validity vector (1=full validity, 0=flat mean weight).
    - epsilon_base: [0,1] baseline misread probability for the most valid cue.
    - epsilon_slope: [0,1] linear increase in misread probability across cues (less valid cues noisier).
                     epsilon_i = clip(epsilon_base + epsilon_slope * rank_i, 0, 0.5), with rank_i ∈ {0,1,2,3}.
                     We cap at 0.5 to avoid negative effective information (complete randomization at 0.5).
    - kappa: [0,1] asymmetry for evidence against B (favoring A); 1 = symmetric, 0 = ignore negative evidence.
    - beta: [0,10] inverse temperature for the logistic choice rule.
    
    Returns:
    - Negative log-likelihood of observed choices.
    """
    omega, epsilon_base, epsilon_slope, kappa, beta = parameters

    v = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    v_mean = float(np.mean(v))
    w = omega * v + (1.0 - omega) * v_mean

    # Cue order by descending validity
    order = np.argsort(-v)

    # Build arrays
    A = np.stack([
        np.asarray(A_feature1, dtype=float),
        np.asarray(A_feature2, dtype=float),
        np.asarray(A_feature3, dtype=float),
        np.asarray(A_feature4, dtype=float),
    ], axis=0)

    B = np.stack([
        np.asarray(B_feature1, dtype=float),
        np.asarray(B_feature2, dtype=float),
        np.asarray(B_feature3, dtype=float),
        np.asarray(B_feature4, dtype=float),
    ], axis=0)

    D = np.asarray(decisions, dtype=float)
    n = D.shape[0]
    nll = 0.0
    eps = 1e-12

    # Per-cue misread probabilities (capped at 0.5 for identifiability)
    ranks = np.array([0, 1, 2, 3], dtype=float)
    epsilons = epsilon_base + epsilon_slope * ranks
    epsilons = np.clip(epsilons, 0.0, 0.5)

    for t in range(n):
        # Compute expected internal evidence per cue with perceptual noise and asymmetry
        d = B[:, t] - A[:, t]  # shape (4,)
        e = (1.0 - 2.0 * epsilons) * d
        # Asymmetric valuation: downweight negative contributions by kappa
        e_asym = np.maximum(e, 0.0) - kappa * np.maximum(-e, 0.0)

        # Redundancy discount while summing in validity order
        S = 0.0
        n_agree = 0  # count of prior contributions sharing the current cumulative sign
        for idx in order:
            contrib = w[idx] * e_asym[idx]
            if contrib != 0.0:
                same_sign = (S > 0 and contrib > 0) or (S < 0 and contrib < 0)
                if same_sign:
                    # delta derived from epsilon_slope to ensure all parameters affect the model distinctly:
                    # more slope (noisier low-validity cues) increases redundancy discount pressure.
                    delta = epsilon_slope  # in [0,1]
                    n_agree += 1
                    contrib = contrib / (1.0 + delta * n_agree)
            S += contrib

        pB = 1.0 / (1.0 + np.exp(-beta * S))
        pB = np.clip(pB, eps, 1.0 - eps)

        if D[t] == 1.0:
            nll -= np.log(pB)
        else:
            nll -= np.log(1.0 - pB)

    return float(nll)


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Difficulty-modulated precision and lapse with side bias.
    
    Idea:
    - The effective decision precision increases with the absolute strength of evidence, and the lapse rate
      increases with difficulty (weak/conflicting evidence).
    - A side bias shifts the internal evidence toward choosing B when evidence is neutral.
    - Base evidence uses the given expert validities as weights.
    
    Evidence per trial:
      v = [0.9, 0.8, 0.7, 0.6]
      E = sum_i v_i * (B_i - A_i)
      |E|_norm = |E| / sum(v)  (sum(v)=3.0)
      precision(E) = beta0 + beta1 * |E|_norm
      lapse(E) = l0 + l1 * (1 - |E|_norm)  (higher when evidence is weak)
      bias shift mu = 2 * (bias - 0.5) in [-1,1], added to E before scaling
      P(B) = (1 - lapse) * sigmoid( precision * (E + mu) ) + 0.5 * lapse
    
    Parameters (all are used):
    - beta0: [0,10] baseline inverse temperature (precision) at zero evidence.
    - beta1: [0,1] gain in precision with evidence strength.
    - l0: [0,1] baseline lapse rate (unmodeled random choice).
    - l1: [0,1] increase in lapse with difficulty (maximal when evidence is zero).
    - bias: [0,1] side bias toward B (bias>0.5) or toward A (bias<0.5); mapped to shift mu in [-1,1].
    
    Returns:
    - Negative log-likelihood of observed choices.
    """
    beta0, beta1, l0, l1, bias = parameters

    v = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    v_sum = float(np.sum(v))

    # Arrays
    A1 = np.asarray(A_feature1, dtype=float)
    A2 = np.asarray(A_feature2, dtype=float)
    A3 = np.asarray(A_feature3, dtype=float)
    A4 = np.asarray(A_feature4, dtype=float)
    B1 = np.asarray(B_feature1, dtype=float)
    B2 = np.asarray(B_feature2, dtype=float)
    B3 = np.asarray(B_feature3, dtype=float)
    B4 = np.asarray(B_feature4, dtype=float)
    D = np.asarray(decisions, dtype=float)

    n = len(D)
    nll = 0.0
    eps = 1e-12

    mu = 2.0 * (bias - 0.5)  # in [-1, 1]

    for t in range(n):
        d_vec = np.array([
            B1[t] - A1[t],
            B2[t] - A2[t],
            B3[t] - A3[t],
            B4[t] - A4[t]
        ], dtype=float)

        E = float(np.dot(v, d_vec))
        E_abs_norm = min(abs(E) / v_sum, 1.0)

        precision = beta0 + beta1 * E_abs_norm
        lapse = l0 + l1 * (1.0 - E_abs_norm)
        lapse = np.clip(lapse, 0.0, 1.0)  # ensure valid probability

        z = precision * (E + mu)
        p_core = 1.0 / (1.0 + np.exp(-z))
        pB = (1.0 - lapse) * p_core + 0.5 * lapse

        pB = np.clip(pB, eps, 1.0 - eps)

        if D[t] == 1.0:
            nll -= np.log(pB)
        else:
            nll -= np.log(1.0 - pB)

    return float(nll)