def cognitive_model1(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Validity-weighted additive integration with asymmetry, trust blending, and lapses.
    The model computes option values as a weighted sum of expert ratings, where weights are a
    participant-specific blend between true expert validities and uniform weights. Negative
    ratings (0) are down/up-weighted via an asymmetry parameter. Choices are generated via
    a logistic choice rule with inverse temperature and a lapse rate.

    Parameters (all in [0,1] except temperature in [0,10]):
    - reliance: [0,1] Degree of reliance on true expert validities vs uniform weights.
                0 => uniform weights; 1 => true validities.
    - asymmetry: [0,1] Penalty magnitude for negative ratings (0). Feature coding is:
                  +1 for positive (1) and -asymmetry for negative (0). Higher => stronger
                  aversion to negatives.
    - trust_shift: [0,1] Smoothly pushes the effective weights toward equality:
                   w_eff <- (1 - trust_shift) * w_eff + trust_shift * mean(w_eff).
                   0 => no shift; 1 => fully equalized.
    - lapse: [0,1] Lapse probability mixing decisions with random choice (0.5).
    - temperature: [0,10] Inverse temperature scaling the value difference in the logistic.

    Returns:
    - Negative log-likelihood of observed choices (0 = chose A, 1 = chose B).
    """
    reliance, asymmetry, trust_shift, lapse, temperature = parameters
    validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    # Blend validities with uniform weights according to reliance
    uniform = np.ones_like(validities) / len(validities)
    weights = reliance * validities + (1.0 - reliance) * uniform

    # Optional trust equalization toward mean (uses trust_shift)
    weights = (1.0 - trust_shift) * weights + trust_shift * np.mean(weights)

    # Normalize weights to sum to 1 for identifiability
    weights = weights / np.sum(weights)

    # Prepare features per trial
    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(float)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(float)

    # Asymmetric feature coding: 1 -> +1, 0 -> -asymmetry
    def recode(X):
        return X * 1.0 + (1.0 - X) * (-asymmetry)

    A_coded = recode(A)
    B_coded = recode(B)

    # Compute values
    vA = A_coded.dot(weights)
    vB = B_coded.dot(weights)

    # Choice probability for choosing B
    dv = vB - vA
    pB_core = 1.0 / (1.0 + np.exp(-temperature * dv))
    pB = (1.0 - lapse) * pB_core + 0.5 * lapse
    pB = np.clip(pB, 1e-12, 1.0 - 1e-12)

    decisions = np.asarray(decisions).astype(int)
    loglik = decisions * np.log(pB) + (1 - decisions) * np.log(1.0 - pB)
    return -np.sum(loglik)


def cognitive_model2(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Probabilistic take-the-best with personalized cue salience, noisy ordering, and lapses.
    The model approximates lexicographic decision making: it searches cues in order of a
    personalized 'effective validity' and stops at the first discriminating cue. The indicated
    option is chosen via a logistic choice rule. If no cue discriminates, it falls back to a
    simple additive rule. Lapses mix in random responding.

    Parameters (all in [0,1] except temperature in [0,10]):
    - sal1, sal2, sal3, sal4: [0,1] Participant-specific salience for each cue, blending with
                              true validities to form effective ordering.
    - blend: [0,1] Blending between true validities and salience: eff = blend*validity + (1-blend)*salience.
    - fallback: [0,1] Weight of fallback additive rule when no discriminating cue is found; also
                        softly included even when a discriminating cue exists.
    - lapse: [0,1] Lapse probability mixing with random choice (0.5).
    - temperature: [0,10] Inverse temperature scaling the (signed) discriminating evidence.

    Returns:
    - Negative log-likelihood of observed choices (0 = A, 1 = B).
    """
    sal1, sal2, sal3, sal4, blend, fallback, lapse, temperature = parameters
    base_validities = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    sal = np.array([sal1, sal2, sal3, sal4], dtype=float)

    # Effective cue strengths used to define search order
    eff_strength = blend * base_validities + (1.0 - blend) * sal

    # Prepare features
    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(int)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(int)

    nT = len(decisions)
    # Precompute additive fallback scores (uniform weights for simplicity)
    add_w = np.ones(4, dtype=float) / 4.0
    vA_add = A.dot(add_w)
    vB_add = B.dot(add_w)
    dv_add = vB_add - vA_add  # in [-1,1]

    # For lexicographic component: find first discriminating cue in eff_strength order
    order = np.argsort(-eff_strength)  # descending
    # Compute signed evidence from first discriminating cue per trial
    signed_evidence = np.zeros(nT, dtype=float)
    has_disc = np.zeros(nT, dtype=bool)

    for idx in order:
        disc = A[:, idx] != B[:, idx]
        # For trials not yet discriminated, set evidence from this cue
        need = ~has_disc & disc
        if np.any(need):
            # If B has 1 and A has 0 => evidence +eff_strength[idx]; else negative
            sign = (B[need, idx] - A[need, idx]).astype(float)  # +1 or -1
            signed_evidence[need] = sign * eff_strength[idx]
            has_disc[need] = True

    # Combine lexicographic and additive components:
    # If discriminating cue found: core evidence from lexicographic;
    # Else: fall back to additive only.
    lex_dv = signed_evidence  # positive favors B
    # Mix with additive difference using 'fallback' both as soft mix and as hard fallback when no cue.
    dv = np.where(has_disc, (1.0 - fallback) * lex_dv + fallback * dv_add,
                  dv_add)

    # Map decision variable to probability of choosing B
    pB_core = 1.0 / (1.0 + np.exp(-temperature * dv))
    pB = (1.0 - lapse) * pB_core + 0.5 * lapse
    pB = np.clip(pB, 1e-12, 1.0 - 1e-12)

    decisions = np.asarray(decisions).astype(int)
    loglik = decisions * np.log(pB) + (1 - decisions) * np.log(1.0 - pB)
    return -np.sum(loglik)


def cognitive_model3(decisions, A_feature1, A_feature2, A_feature3, A_feature4,
                     B_feature1, B_feature2, B_feature3, B_feature4, parameters):
    """Bayesian cue integration with personalized trust calibration and prior bias.
    Each expert is treated as a noisy informant with validity v_k. The participant
    personalizes trust per cue and globally calibrates validities toward 0.5. For each option,
    the log-odds of being a high-quality product given its ratings are computed under
    independence, combined with a prior bias for 'goodness'. Choices depend on the log-odds
    difference between B and A via a logistic rule with lapse.

    Parameters (all in [0,1] except temperature in [0,10]):
    - trust1, trust2, trust3, trust4: [0,1] Cue-specific trust scaling (0=no trust, 1=full trust).
    - calibrate: [0,1] Global calibration toward 0.5: v_eff = 0.5 + calibrate*(v-0.5).
    - prior_good: [0,1] Prior probability that any given option is a high-quality product.
    - lapse: [0,1] Lapse probability mixing with random choice (0.5).
    - temperature: [0,10] Inverse temperature scaling the log-odds difference.

    Returns:
    - Negative log-likelihood of observed choices (0 = A, 1 = B).
    """
    trust1, trust2, trust3, trust4, calibrate, prior_good, lapse, temperature = parameters
    base_v = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)
    trust = np.array([trust1, trust2, trust3, trust4], dtype=float)

    # Personalized effective validities: shrink toward 0.5 by calibrate, then scale by trust toward 0.5
    # Two-stage: first calibrate global, then cue-wise trust
    v_cal = 0.5 + calibrate * (base_v - 0.5)
    v_eff = 0.5 + trust * (v_cal - 0.5)
    v_eff = np.clip(v_eff, 1e-6, 1.0 - 1e-6)  # numerical safety

    # Prior log-odds for an option being good
    prior_good = np.clip(prior_good, 1e-6, 1.0 - 1e-6)
    logit_prior = np.log(prior_good / (1.0 - prior_good))

    # Features
    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(int)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(int)

    # For each option X, compute log-odds that X is good given its ratings:
    # logit P(Good|ratings) = logit(prior) + sum_k [ r_k*log(v/(1-v)) + (1-r_k)*log((1-v)/v) ]
    # which simplifies to: logit(prior) + sum_k (2*r_k - 1)*log(v/(1-v))
    logit_coef = np.log(v_eff / (1.0 - v_eff))  # per cue

    def option_logit(X):
        s = (2 * X - 1)  # +1 if rating=1, -1 if rating=0
        # broadcast multiply and sum across cues
        return logit_prior + s.dot(logit_coef)

    logit_A = option_logit(A)
    logit_B = option_logit(B)

    # Decision variable: difference in goodness log-odds between B and A
    dv = logit_B - logit_A

    # Choice probability for B
    pB_core = 1.0 / (1.0 + np.exp(-temperature * dv))
    pB = (1.0 - lapse) * pB_core + 0.5 * lapse
    pB = np.clip(pB, 1e-12, 1.0 - 1e-12)

    decisions = np.asarray(decisions).astype(int)
    loglik = decisions * np.log(pB) + (1 - decisions) * np.log(1.0 - pB)
    return -np.sum(loglik)