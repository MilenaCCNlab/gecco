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

    w = prior_bias * v + (1.0 - prior_bias) * 0.5

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

        if D[t] == 1.0:
            nll -= np.log(pB)
        else:
            nll -= np.log(1.0 - pB)

        dir_vec = np.sign(d_vec)  # {-1,0,1}
        agree_choice = ((D[t] == 1.0) & (dir_vec == 1)) | ((D[t] == 0.0) & (dir_vec == -1))
        disagree_choice = ((D[t] == 1.0) & (dir_vec == -1)) | ((D[t] == 0.0) & (dir_vec == 1))

        agree_choice = agree_choice.astype(float)
        disagree_choice = disagree_choice.astype(float)

        g = dir_vec * (agree_choice + lambda_loss * disagree_choice)

        pe = D[t] - pB  # signed prediction error

        w = (1.0 - decay) * w + alpha * pe * g

    return float(nll)