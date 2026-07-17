"""Cognitive Library - Interpretable Primitives

Verified primitives for two-step task cognitive models.
Each function represents a distinct cognitive mechanism.

Categories:
  - Initialization: Q-values, transition matrices, trial arrays
  - Model-Based: Planning via transition model
  - MF-MB Integration: Hybrid decision-making
  - Action Selection: Softmax, value centering
  - TD Learning: Q-value updates
  - Likelihood: NLL computation
"""

import numpy as np


# ============================================================
# INITIALIZATION
# ============================================================

def init_q_and_T():
    """Initialize Q-values and transition matrix with priors."""
    Q2 = np.zeros((2, 2))
    T = np.array([[0.7, 0.3], [0.3, 0.7]])
    return Q2, T


def init_trial_arrays_basic(a1_seq):
    """Initialize trial probability arrays only."""
    num_trials = len(a1_seq)
    p1_seq = np.zeros(num_trials)
    p2_seq = np.zeros(num_trials)
    return num_trials, p1_seq, p2_seq


def init_trial_arrays(a1_seq):
    """Initialize trial arrays with stage-1 Q-values."""
    num_trials = len(a1_seq)
    p1_seq = np.zeros(num_trials)
    p2_seq = np.zeros(num_trials)
    q1 = np.zeros(2)
    return num_trials, p1_seq, p2_seq, q1


def init_prob_arrays(num_trials):
    """Initialize probability arrays from trial count."""
    p1_seq = np.zeros(num_trials)
    p2_seq = np.zeros(num_trials)
    return p1_seq, p2_seq


def init_with_last_action(a1_seq):
    """Initialize with last-action tracking."""
    last_a1 = None
    num_trials = len(a1_seq)
    p1_seq = np.zeros(num_trials)
    p2_seq = np.zeros(num_trials)
    return last_a1, num_trials, p1_seq, p2_seq


def get_trial_actions(a1_seq, a2_seq, t):
    """Extract actions for trial t."""
    a1 = int(a1_seq[t])
    a2 = int(a2_seq[t])
    return a1, a2


# ============================================================
# MODEL-BASED VALUATION
# ============================================================

def max_q2_per_state(Q2):
    """Maximum Q-value for each state."""
    return np.max(Q2, axis=1)


def compute_mb_values(Q2, T):
    """Model-based planning: expected value via transition model."""
    max_q2 = np.max(Q2, axis=1)
    q1_mb = T @ max_q2
    return max_q2, q1_mb


# ============================================================
# MF-MB INTEGRATION
# ============================================================

def integrate_mf_mb(q1_mf, q1_mb, w_mb):
    """Hybrid MF-MB: weighted combination of strategies."""
    return (1.0 - w_mb) * q1_mf + w_mb * q1_mb


# ============================================================
# ACTION SELECTION
# ============================================================

def center_values(q):
    """Center values for numerical stability: q - max(q)."""
    return q - np.max(q)


def logsumexp_max(logits):
    """Max for log-sum-exp trick."""
    return np.max(logits)


def exp_logits(logits):
    """Exponentiate logits."""
    return np.exp(logits)


def softmax_exp(beta, q_centered):
    """Softmax numerator: exp(beta * q_centered)."""
    return np.exp(beta * q_centered)


def normalize_probs(logits):
    """Convert logits to normalized probabilities."""
    exp_vals = np.exp(logits)
    return exp_vals / np.sum(exp_vals)


def record_stage1(a1, p1_seq, probs1, s_seq, t):
    """Record stage-1 choice probability and return state."""
    p1_seq[t] = probs1[a1]
    s = s_seq[t]
    return s


def record_stage2(a2, p2_seq, probs2, r_seq, t):
    """Record stage-2 choice probability and return reward."""
    p2_seq[t] = probs2[a2]
    r = r_seq[t]
    return r


# ============================================================
# TD LEARNING
# ============================================================

def td_step(alpha, delta):
    """Single TD update: alpha * delta."""
    return alpha * delta


def update_q2(Q2, a2, alpha, r, s):
    """Update stage-2 Q-value, return TD error."""
    delta2 = r - Q2[s, a2]
    Q2[s, a2] += alpha * delta2
    return delta2


def update_q1(Q2, a1, a2, alpha, q1, s):
    """Update stage-1 Q from stage-2 bootstrap, return delta only."""
    target1 = Q2[s, a2]
    delta1 = target1 - q1[a1]
    q1[a1] += alpha * delta1
    return delta1


def update_q1_full(Q2, a1, a2, alpha, q1, s):
    """Update stage-1 Q, return delta and target."""
    target1 = Q2[s, a2]
    delta1 = target1 - q1[a1]
    q1[a1] += alpha * delta1
    return delta1, target1


# ============================================================
# MEMORY & FORGETTING
# ============================================================

def memory_retention(decay):
    """Memory retention factor: 1 - decay."""
    return 1.0 - decay


# ============================================================
# STAI/ANXIETY MODULATION
# ============================================================

def stai_scale(stai):
    """STAI linear scaling: 0.5 * stai."""
    return 0.5 * stai


def stai_modulate(stai):
    """STAI affine modulation: 0.5 * stai + 0.5."""
    return 0.5 * stai + 0.5


# ============================================================
# LIKELIHOOD
# ============================================================

def log_likelihood(epsilon, p1_seq, p2_seq):
    """Sum of log probabilities."""
    return np.sum(np.log(epsilon + p1_seq)) + np.sum(np.log(epsilon + p2_seq))


def compute_nll(p1_seq, p2_seq):
    """Compute epsilon and negative log-likelihood."""
    epsilon = 1e-12
    nll = -(np.sum(np.log(epsilon + p1_seq)) + np.sum(np.log(epsilon + p2_seq)))
    return epsilon, nll
