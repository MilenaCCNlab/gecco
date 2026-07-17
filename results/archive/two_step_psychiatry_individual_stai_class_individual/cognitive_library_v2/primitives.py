"""
Cognitive Primitives Library
============================

Atomic building blocks for cognitive models of the two-step task.
Each primitive implements a well-understood cognitive mechanism.

Categories:
  - helper: Core computational utilities (softmax, MB values)
  - modulation: STAI modulation functions (how anxiety affects parameters)
  - policy: Action selection components (perseveration, MB/MF mixing)
  - value_update: Learning rule components (TD, asymmetric)
  - decay: Memory/value decay components
"""

import numpy as np
from typing import Optional, Tuple


# ============================================================
# HELPER
# ============================================================

# Numerically stable softmax for action selection
# Used by: 27 participants
def softmax(values: np.ndarray, beta: float) -> np.ndarray:
    """Softmax action selection with inverse temperature beta."""
    centered = values - np.max(values)
    exp_vals = np.exp(beta * centered)
    return exp_vals / np.sum(exp_vals)

# Compute model-based Q-values using transition matrix
# Used by: 3 participants
def compute_mb_values(q_stage2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Model-based values: Q_MB = T @ max(Q2 per state)."""
    v_stage2 = np.max(q_stage2, axis=1)
    return T @ v_stage2


# ============================================================
# MODULATION
# ============================================================

# Anxiety amplifies parameter: param * stai
# Used by: 22 participants
def stai_multiplicative(base_param: float, stai: float) -> float:
    """Higher anxiety increases the effect."""
    return base_param * stai

# Anxiety adds to parameter: base + slope * stai
# Used by: 0 participants
def stai_additive(base: float, slope: float, stai: float) -> float:
    """Linear combination: base + slope * stai."""
    return base + slope * stai

# Anxiety reduces parameter: param * (1 - stai)
# Used by: 3 participants
def stai_inverse_linear(param: float, stai: float) -> float:
    """Higher anxiety decreases the effect."""
    return param * (1.0 - stai)

# Anxiety reduces parameter: param / (1 + stai)
# Used by: 1 participants
def stai_inverse_division(param: float, stai: float) -> float:
    """Higher anxiety dampens the effect."""
    return param / (1.0 + stai)


# ============================================================
# POLICY
# ============================================================

# Add bonus to previously chosen action (stickiness)
# Used by: 22 participants
def add_perseveration_bonus(q_values: np.ndarray, last_action: Optional[int], 
                            bonus: float) -> np.ndarray:
    """Add perseveration bonus to the last chosen action."""
    q_modified = q_values.copy()
    if last_action is not None:
        q_modified[last_action] += bonus
    return q_modified

# Add bonus if last trial was rewarded (win-stay strategy)
# Used by: 3 participants
def add_win_stay_bonus(q_values: np.ndarray, last_action: Optional[int],
                       last_reward: float, bonus: float) -> np.ndarray:
    """Add bonus to repeat action after reward (win-stay)."""
    q_modified = q_values.copy()
    if last_action is not None and last_reward == 1.0:
        q_modified[last_action] += bonus
    return q_modified

# Mix model-based and model-free values
# Used by: 3 participants
def mb_mf_mixture(q_mf: np.ndarray, q_mb: np.ndarray, w: float) -> np.ndarray:
    """Combine model-based and model-free Q-values.
    
    Q_net = w * Q_MB + (1 - w) * Q_MF
    """
    w = np.clip(w, 0, 1)
    return w * q_mb + (1.0 - w) * q_mf


# ============================================================
# VALUE_UPDATE
# ============================================================

# TD update for stage 2 Q-values
# Used by: 26 participants
def td_update_stage2(q_stage2: np.ndarray, state: int, action: int,
                     reward: float, alpha: float) -> np.ndarray:
    """Standard TD update: Q(s,a) += alpha * (reward - Q(s,a))."""
    q_new = q_stage2.copy()
    delta = reward - q_stage2[state, action]
    q_new[state, action] += alpha * delta
    return q_new, delta

# TD update for stage 1 Q-values (propagated from stage 2)
# Used by: 26 participants
def td_update_stage1(q_stage1: np.ndarray, action: int, 
                     target: float, alpha: float) -> np.ndarray:
    """TD update: Q(a) += alpha * (target - Q(a))."""
    q_new = q_stage1.copy()
    delta = target - q_stage1[action]
    q_new[action] += alpha * delta
    return q_new

# Asymmetric learning: different rates for pos/neg PE
# Used by: 1 participants
def asymmetric_td(q_value: float, target: float, 
                  alpha_pos: float, alpha_neg: float) -> float:
    """Use alpha_pos for positive PE, alpha_neg for negative PE."""
    delta = target - q_value
    alpha = alpha_pos if delta >= 0 else alpha_neg
    return q_value + alpha * delta


# ============================================================
# DECAY
# ============================================================

# Decay unchosen options toward zero (forgetting)
# Used by: 2 participants
def apply_memory_decay(q_values: np.ndarray, chosen_idx: int,
                       decay_rate: float) -> np.ndarray:
    """Decay unchosen options: Q(unchosen) *= (1 - decay_rate)."""
    q_new = q_values.copy()
    for i in range(len(q_values)):
        if i != chosen_idx:
            q_new[i] *= (1.0 - decay_rate)
    return q_new

# Eligibility trace for multi-step credit assignment
# Used by: 1 participants
def eligibility_update(trace: np.ndarray, action: int, 
                       decay_lambda: float) -> np.ndarray:
    """Update eligibility trace: decay old, increment current."""
    trace_new = trace * decay_lambda
    trace_new[action] = 1.0
    return trace_new
