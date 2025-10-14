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

    eff_strength = blend * base_validities + (1.0 - blend) * sal

    A = np.column_stack([A_feature1, A_feature2, A_feature3, A_feature4]).astype(int)
    B = np.column_stack([B_feature1, B_feature2, B_feature3, B_feature4]).astype(int)

    nT = len(decisions)

    add_w = np.ones(4, dtype=float) / 4.0
    vA_add = A.dot(add_w)
    vB_add = B.dot(add_w)
    dv_add = vB_add - vA_add  # in [-1,1]

    order = np.argsort(-eff_strength)  # descending

    signed_evidence = np.zeros(nT, dtype=float)
    has_disc = np.zeros(nT, dtype=bool)

    for idx in order:
        disc = A[:, idx] != B[:, idx]

        need = ~has_disc & disc
        if np.any(need):

            sign = (B[need, idx] - A[need, idx]).astype(float)  # +1 or -1
            signed_evidence[need] = sign * eff_strength[idx]
            has_disc[need] = True



    lex_dv = signed_evidence  # positive favors B

    dv = np.where(has_disc, (1.0 - fallback) * lex_dv + fallback * dv_add,
                  dv_add)

    pB_core = 1.0 / (1.0 + np.exp(-temperature * dv))
    pB = (1.0 - lapse) * pB_core + 0.5 * lapse
    pB = np.clip(pB, 1e-12, 1.0 - 1e-12)

    decisions = np.asarray(decisions).astype(int)
    loglik = decisions * np.log(pB) + (1 - decisions) * np.log(1.0 - pB)
    return -np.sum(loglik)