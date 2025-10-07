def cognitive_model3(action_1, state, action_2, reward, model_parameters):
    """
    Risk-sensitive hybrid with reference adaptation and uncertainty-gated model-based control.
    Concept:
      - Stage 2 uses a risk-sensitive utility with a dynamic reference point; utility guides learning.
      - Stage 1 mixes MF and MB values with a state-action-specific MB weight that decreases under transition uncertainty.
      - Transition model learned with pseudo-count prior; uncertainty measured via entropy of learned transitions.
      - Habit accumulation and stickiness biases (both stages).
      - Loss-contingent beta boost for stage-1 choices, bias toward spaceship A.
      - Global forgetting on Q-values.

    Parameters (in order):
      alpha1: [0,1] Stage-1 MF learning rate (also used for reference adaptation)
      alpha2: [0,1] Stage-2 learning rate on utility prediction errors
      beta1: [0,10] Stage-1 inverse temperature
      beta2: [0,10] Stage-2 inverse temperature
      rho: [0,1] Utility curvature (concavity/convexity) for deviations from reference
      kappa_loss: [0,1] Loss aversion multiplier for negative utility (0=no extra loss weight, 1=full extra weight)
      tau_trans: [0,1] Scale of uncertainty gating: MB weight w_a = 1 - tau_trans * (entropy(T_est[a]) / log(2))
      habit_gain: [0,1] Habit increment for chosen stage-1 action
      habit_decay: [0,1] Habit decay per trial
      forget: [0,1] Forgetting applied to Q-values each trial
      biasA: [0,1] Additive bias toward spaceship A, b = (biasA - 0.5)*2
      dirichlet_prior: [0,1] Strength of prior pseudo-counts for transitions (scaled internally)
      alphaT: [0,1] Transition learning step (count update EWMA)
      stickiness: [0,1] Stickiness bias added to repeating previous choices (both stages)
      beta_loss_boost: [0,1] Multiplicative boost to beta1 after prior loss

    Returns:
      Negative log-likelihood of the observed choices.
    """
    import numpy as np  # assumed already imported

    (alpha1, alpha2, beta1, beta2, rho, kappa_loss, tau_trans,
     habit_gain, habit_decay, forget, biasA, dirichlet_prior, alphaT,
     stickiness, beta_loss_boost) = model_parameters

    n_trials = len(action_1)

    q1_mf = np.zeros(2)
    q2 = np.zeros((2, 2))
    habit = np.zeros(2)

    ref = 0.5  # initial expected reward


    prior_scale = 10.0 * dirichlet_prior + 0.1
    counts = np.full((2, 2), prior_scale)  # counts[action, state]
    T_est = counts / np.sum(counts, axis=1, keepdims=True)

    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    bias_add = (biasA - 0.5) * 2.0
    prev_a1 = None
    prev_a2 = [None, None]  # store last action per state for stage-2 stickiness
    prev_reward = 1.0
    eps = 1e-12