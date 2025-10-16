def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Probabilistic Take-The-Best–inspired priority integration with cue misread, leakage, bias, and lapse.

    Cognitive idea:
    - Cues (experts) are prioritized via a softmax over their validities; gamma controls sharpness of priority.
    - The comparison signal per cue is the (possibly misread) signed difference (B - A), scaled by priority and a leakage
      factor that downweights lower-priority cues.
    - A global "stop" factor scales reliance on prioritized cues (low values approximate more compensatory use; high values
      approximate lexicographic emphasis).
    - A side bias toward B shifts choices irrespective of attributes, and a lapse parameter mixes in random choice.

    Parameters (in order):
    - gamma: [0,10] Priority sharpness for cue ordering; higher focuses weight on the most valid cue.
    - stop: [0,1] Lexicographic emphasis; higher increases reliance on prioritized cues (0 ≈ compensatory).
    - leak: [0,1] Leakage across priority ranks; larger leak reduces influence of lower-priority cues via (1 - leak)^rank.
    - flip: [0,1] Probability of misreading a cue comparison (flips the sign of B-A with this probability in expectation).
    - biasB: [0,1] Baseline bias toward choosing B (0.5 = no bias), internally centered to [-1,1].
    - lapse: [0,1] Probability of random choice (uniform 0.5), mixed with model probability.
    - beta: [0,10] Inverse temperature controlling choice determinism for the final choice.

    Inputs:
    - decisions: array-like of 0/1 where 1 indicates choosing B, 0 indicates choosing A.
    - A_feature1..4, B_feature1..4: arrays of 0/1 expert ratings per trial for options A and B.
    - parameters: iterable of length 7 as described above.

    Returns:
    - Negative log-likelihood of observed choices under the model.
    """
    gamma, stop, leak, flip, biasB, lapse, beta = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    # Softmax over validities to get priority weights
    logits = gamma * validities  # shape (4,)
    logits = logits - np.max(logits)  # stabilize
    priority = np.exp(logits)
    priority = priority / np.sum(priority)

    # Rank indices: 0 for highest priority, 3 for lowest
    order = np.argsort(-validities)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(4)

    # Stack features
    A = np.vstack([A_feature1, A_feature2, A_feature3, A_feature4]).T.astype(float)
    B = np.vstack([B_feature1, B_feature2, B_feature3, B_feature4]).T.astype(float)
    d = B - A  # in {-1,0,1}
    # Expected sign after potential flip (flip chance flips the sign, 0 stays 0)
    d_eff = d * (1.0 - 2.0 * flip)

    # Leakage downweights lower-priority cues
    decay = (1.0 - leak) ** ranks  # shape (4,)
    # Stop factor scales reliance on prioritized cues
    cue_weights = stop * priority * decay  # shape (4,)

    # Add a small compensatory component so model remains sensitive when stop≈0
    comp_weights = (1.0 - stop) * (validities / np.sum(validities))

    total_weights = cue_weights + comp_weights  # convex combination; sums to ≤ 1 typically
    # Normalize total weights to sum to 1 so beta scale is interpretable
    total_weights = total_weights / (np.sum(total_weights) + 1e-12)

    # Bias term
    bias_term = (biasB - 0.5) * 2.0

    # Decision variable per trial
    dv = (d_eff @ total_weights) + bias_term  # shape (n_trials,)

    pB_model = 1.0 / (1.0 + np.exp(-beta * dv))
    pB = lapse * 0.5 + (1.0 - lapse) * pB_model
    pB = np.clip(pB, 1e-9, 1.0 - 1e-9)

    decisions = np.asarray(decisions).astype(int)
    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1.0 - pB)
    return -np.sum(ll)


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Bayesian cue integration with grouped attentional gains, nonlinear LLR compression, prior bias, and lapse.

    Cognitive idea:
    - Each cue contributes a log-likelihood ratio (LLR) based on its validity v: LLR = log(v/(1-v)) supporting the option
      indicated by the cue (B vs A).
    - Attention is allocated differently to high-validity (top-2) and low-validity (bottom-2) cues via two gains.
    - The integrated LLR is compressed/expanded by a nonlinear exponent (lambda) to capture diminishing sensitivity.
    - A prior bias toward B is incorporated as a log prior odds; integration noise attenuates the overall evidence.
    - Choices follow a softmax with inverse temperature; lapse mixes in random responding.

    Parameters (in order):
    - gain_hi: [0,1] Attentional gain applied to the two highest-validity cues (0.9, 0.8).
    - gain_lo: [0,1] Attentional gain applied to the two lower-validity cues (0.7, 0.6).
    - priorB: [0,1] Prior probability for B being superior (0.5 = neutral). Converted to log-odds internally.
    - lambda_nl: [0,1] Nonlinear compression/expansion exponent on absolute LLR (1=no compression, 0=all-or-none).
    - atten: [0,1] Integration attenuation (effective 1 - noise); smaller values reduce overall evidence impact.
    - lapse: [0,1] Probability of random choice (uniform 0.5).
    - beta: [0,10] Inverse temperature controlling choice determinism.

    Inputs:
    - decisions: array-like of 0/1 where 1 indicates choosing B.
    - A_feature1..4, B_feature1..4: arrays of 0/1 expert ratings per trial for options A and B.
    - parameters: iterable of length 7 as described above.

    Returns:
    - Negative log-likelihood of observed choices under the model.
    """
    gain_hi, gain_lo, priorB, lambda_nl, atten, lapse, beta = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    # Precompute cue LLR magnitudes from validities
    llr_mag = np.log(validities / (1.0 - validities))  # positive magnitudes

    # Gains: top-2 (0.9,0.8) get gain_hi; bottom-2 (0.7,0.6) get gain_lo
    gains = np.array([gain_hi, gain_hi, gain_lo, gain_lo], dtype=float)

    # Stack features and compute signed cue support: +1 if cue favors B, -1 if favors A, 0 if tie
    A = np.vstack([A_feature1, A_feature2, A_feature3, A_feature4]).T.astype(float)
    B = np.vstack([B_feature1, B_feature2, B_feature3, B_feature4]).T.astype(float)
    d = B - A  # in {-1,0,1}

    # Signed LLR per trial and cue
    signed_llr = d * llr_mag  # sign applies, zero when tie

    # Apply nonlinear compression on absolute evidence with sign preserved
    abs_llr = np.abs(signed_llr)
    # Avoid 0**0 by adding tiny epsilon
    abs_llr_nl = abs_llr ** np.maximum(lambda_nl, 1e-9)
    signed_llr_nl = np.sign(signed_llr) * abs_llr_nl

    # Apply attentional gains and attenuation
    weighted_llr = signed_llr_nl * gains
    total_llr = atten * np.sum(weighted_llr, axis=1)

    # Add prior log-odds for B
    priorB = np.clip(priorB, 1e-6, 1.0 - 1e-6)
    logit_prior = np.log(priorB / (1.0 - priorB))
    dv = total_llr + logit_prior

    # Map DV to choice probability
    pB_model = 1.0 / (1.0 + np.exp(-beta * dv))
    pB = lapse * 0.5 + (1.0 - lapse) * pB_model
    pB = np.clip(pB, 1e-9, 1.0 - 1e-9)

    decisions = np.asarray(decisions).astype(int)
    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1.0 - pB)
    return -np.sum(ll)


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Selective sequential integration with leader-driven suppression/amplification, memory leak, bias, and lapse.

    Cognitive idea:
    - Cues are processed sequentially in descending validity (0.9→0.6).
    - A running decision variable (DV) accumulates signed cue evidence with memory leak.
    - When a provisional leader emerges (|DV| exceeds a threshold), the next cue is selectively modulated:
      cues supporting the leader are amplified, and opposing cues are suppressed (selective integration).
    - A side bias toward B and lapse are included; final DV is mapped to choice via a softmax.

    Parameters (in order):
    - amp_up: [0,1] Amplification factor for cues aligned with the current leader once above threshold (>=1 means no amp; we bound [0,1] so we map internally).
    - sup_down: [0,1] Suppression factor for cues opposing the leader once above threshold (0=full suppression, 1=no suppression).
    - thresh: [0,1] Leader threshold on |DV| at which selective modulation begins (in DV units).
    - mem_leak: [0,1] Memory leak between cues; higher values mean more leak (less carryover), implemented as DV *= (1 - mem_leak).
    - biasB: [0,1] Baseline bias toward B (0.5 = no bias), internally centered to [-1,1].
    - lapse: [0,1] Probability of random choice (uniform 0.5).
    - beta: [0,10] Inverse temperature controlling choice determinism.

    Notes on amp_up:
    - amp_up is remapped to [1, 2] via 1 + amp_up to yield amplification >=1, while still respecting the [0,1] bound on parameters.

    Inputs:
    - decisions: array-like of 0/1 where 1 indicates choosing B.
    - A_feature1..4, B_feature1..4: arrays of 0/1 expert ratings per trial for options A and B.
    - parameters: iterable of length 7 as described above.

    Returns:
    - Negative log-likelihood of observed choices under the model.
    """
    amp_up, sup_down, thresh, mem_leak, biasB, lapse, beta = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    # Process cues in descending validity order: indices [0,1,2,3]
    idx_order = np.array([0, 1, 2, 3], dtype=int)

    # Pre-weight each cue's raw contribution by its validity (base evidence strength)
    base_w = validities

    # Stack features and compute signed cue support: +1 if cue favors B, -1 if favors A, 0 if tie
    A = np.vstack([A_feature1, A_feature2, A_feature3, A_feature4]).T.astype(float)
    B = np.vstack([B_feature1, B_feature2, B_feature3, B_feature4]).T.astype(float)
    d = B - A  # in {-1,0,1}

    # Initialize DV per trial
    n_trials = d.shape[0]
    DV = np.zeros(n_trials, dtype=float)

    # Remap amplification to >=1
    amp_up_eff = 1.0 + amp_up  # in [1,2]

    # Sequentially integrate
    for k in idx_order:
        # Leak memory before adding new evidence
        DV *= (1.0 - mem_leak)

        cue_signal = d[:, k] * base_w[k]  # signed contribution for this cue
        # Determine whether selective modulation applies based on current |DV|
        leader_active = (np.abs(DV) >= thresh).astype(float)
        # Determine alignment of this cue with current leader
        leader_sign = np.sign(DV)  # 1 for B-leading, -1 for A-leading, 0 for none
        cue_align = np.sign(cue_signal)  # 1 supports B, -1 supports A, 0 neutral
        same_side = (leader_sign * cue_align) > 0  # boolean array

        # Compute modulation factors per trial
        mod = np.ones(n_trials, dtype=float)
        # When leader_active:
        # - if same_side: amplify by amp_up_eff
        # - else if opposing: suppress by sup_down
        mod = np.where(leader_active * same_side, amp_up_eff, mod)
        mod = np.where(leader_active * (~same_side), sup_down, mod)

        DV += mod * cue_signal

    # Add side bias
    bias_term = (biasB - 0.5) * 2.0
    DV = DV + bias_term

    # Map to choice probability with lapse
    pB_model = 1.0 / (1.0 + np.exp(-beta * DV))
    pB = lapse * 0.5 + (1.0 - lapse) * pB_model
    pB = np.clip(pB, 1e-9, 1.0 - 1e-9)

    decisions = np.asarray(decisions).astype(int)
    ll = decisions * np.log(pB) + (1 - decisions) * np.log(1.0 - pB)
    return -np.sum(ll)