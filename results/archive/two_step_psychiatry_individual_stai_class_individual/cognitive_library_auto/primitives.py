import numpy as np
from typing import Optional, Tuple, Union

# =============================================================================
# 1. Helper Functions
# =============================================================================

def softmax(values: np.ndarray, beta: float) -> np.ndarray:
    """
    Computes the softmax probability distribution over values.
    
    Args:
        values: Array of action values (e.g., Q-values).
        beta: Inverse temperature parameter (higher = more deterministic).
        
    Returns:
        Probability distribution summing to 1.
    """
    # Subtract max for numerical stability
    exponentiated = np.exp(beta * (values - np.max(values)))
    return exponentiated / np.sum(exponentiated)

def compute_mb_values(transition_matrix: np.ndarray, q_stage2: np.ndarray) -> np.ndarray:
    """
    Computes Model-Based Q-values for Stage 1 based on transition probabilities 
    and Stage 2 values.
    
    Q_MB(a) = Sum_s' [ T(s'|a) * max_a'(Q_stage2(s', a')) ]
    
    Args:
        transition_matrix: Shape (n_actions_stage1, n_states_stage2).
        q_stage2: Shape (n_states_stage2, n_actions_stage2).
        
    Returns:
        Array of Model-Based values for Stage 1 choices.
    """
    # Value of each second-stage state is the max Q-value in that state
    state_values = np.max(q_stage2, axis=1)
    # Expected value for Stage 1 actions
    return transition_matrix @ state_values

# =============================================================================
# 2. STAI Modulation
# =============================================================================

def stai_multiplicative(param: float, stai: float) -> float:
    """
    Modulates a parameter proportional to STAI.
    Used for anxiety-driven stickiness scaling (e.g., p21, p26, p27).
    
    Returns: param * stai
    """
    return param * stai

def stai_additive(base: float, slope: float, stai: float) -> float:
    """
    Linearly modulates a parameter based on STAI.
    Used for effective stickiness or perseveration (e.g., p19, p22, p36).
    
    Returns: base + (slope * stai)
    """
    return base + (slope * stai)

def stai_inverse_linear(param: float, stai: float) -> float:
    """
    Reduces a parameter linearly as STAI increases.
    Used for Model-Based weights (e.g., p24, p37, p43).
    
    Returns: param * (1 - stai)
    """
    return param * (1.0 - stai)

def stai_inverse_division(param: float, stai: float) -> float:
    """
    Reduces a parameter using inverse division as STAI increases.
    Used for effective stickiness in p18.
    
    Returns: param * (1 / (1 + stai))
    """
    return param * (1.0 / (1.0 + stai))

def stai_affine_amplification(base: float, bias_factor: float, stai: float) -> float:
    """
    Amplifies a base parameter by a factor of (1 + bias * stai).
    Used for learning rate biases (e.g., p23, p25).
    
    Returns: base * (1 + bias_factor * stai)
    """
    return base * (1.0 + bias_factor * stai)

# =============================================================================
# 3. Policy Components
# =============================================================================

def add_perseveration_bonus(
    q_values: np.ndarray, 
    last_action: Optional[int], 
    bonus: float
) -> np.ndarray:
    """
    Adds a stickiness/perseveration bonus to the previously chosen action.
    
    Args:
        q_values: Original Q-values.
        last_action: Index of the last action taken (or None).
        bonus: Magnitude of the bonus to add.
        
    Returns:
        Modified Q-values (copy).
    """
    q_new = q_values.copy()
    if last_action is not None:
        q_new[int(last_action)] += bonus
    return q_new

def add_win_stay_bonus(
    q_values: np.ndarray, 
    last_action: Optional[int], 
    last_reward: float, 
    bonus: float
) -> np.ndarray:
    """
    Adds a bonus to the last action only if it was rewarded (Win-Stay).
    Used in p28, p35.
    
    Args:
        q_values: Original Q-values.
        last_action: Index of the last action taken.
        last_reward: Reward received in the last trial (typically 0 or 1).
        bonus: Magnitude of the bonus.
        
    Returns:
        Modified Q-values (copy).
    """
    q_new = q_values.copy()
    if last_action is not None and last_reward > 0.5:
        q_new[int(last_action)] += bonus
    return q_new

def mb_mf_mixture(q_mb: np.ndarray, q_mf: np.ndarray, w: float) -> np.ndarray:
    """
    Combines Model-Based and Model-Free values using a mixing weight.
    Used in p24, p37, p43.
    
    Args:
        q_mb: Model-Based Q-values.
        q_mf: Model-Free Q-values.
        w: Weight of the Model-Based component [0, 1].
        
    Returns:
        w * Q_MB + (1 - w) * Q_MF
    """
    # Ensure w is clipped to valid range [0, 1] usually handled by bounds, 
    # but here for safety in calculation
    w_clipped = np.clip(w, 0.0, 1.0)
    return (w_clipped * q_mb) + ((1.0 - w_clipped) * q_mf)

def add_habit_influence(
    q_values: np.ndarray,
    habit_values: np.ndarray,
    weight: float
) -> np.ndarray:
    """
    Adds a separate habit trace to the Q-values (p44).
    
    Returns: q_values + weight * habit_values
    """
    return q_values + (weight * habit_values)

# =============================================================================
# 4. Value Updates
# =============================================================================

def td_update_stage1(
    q_stage1: np.ndarray,
    action: int,
    q_next_value: float,
    alpha: float
) -> np.ndarray:
    """
    Performs a standard Temporal Difference (TD) update for Stage 1.
    
    Args:
        q_stage1: Current Stage 1 Q-values.
        action: The action taken.
        q_next_value: The value of the subsequent state/action (e.g., Q(s', a') or max Q(s')).
        alpha: Learning rate.
        
    Returns:
        Updated Stage 1 Q-values (copy).
    """
    q_new = q_stage1.copy()
    prediction_error = q_next_value - q_new[action]
    q_new[action] += alpha * prediction_error
    return q_new

def td_update_stage2(
    q_stage2: np.ndarray,
    state: int,
    action: int,
    reward: float,
    alpha: float
) -> np.ndarray:
    """
    Performs a standard Temporal Difference (TD) update for Stage 2.
    
    Args:
        q_stage2: Current Stage 2 Q-values (shape: [n_states, n_actions]).
        state: The state visited.
        action: The action taken in that state.
        reward: The observed reward.
        alpha: Learning rate.
        
    Returns:
        Updated Stage 2 Q-values (copy).
    """
    q_new = q_stage2.copy()
    prediction_error = reward - q_new[state, action]
    q_new[state, action] += alpha * prediction_error
    return q_new

def td_update_asymmetric(
    current_q: float,
    target: float,
    alpha_pos: float,
    alpha_neg: float
) -> float:
    """
    Updates a single Q-value using asymmetric learning rates based on 
    whether the prediction error (or outcome) is positive/negative.
    Used in p23.
    
    Args:
        current_q: The current estimate.
        target: The target value (reward or next Q).
        alpha_pos: Learning rate for positive outcomes/errors.
        alpha_neg: Learning rate for negative outcomes/errors.
        
    Returns:
        The updated Q-value.
    """
    # Note: p23 uses reward magnitude (0 vs 1) to decide alpha.
    # Other models might use prediction error sign. 
    # Based on p23 logic:
    if target > 0.5: # Assuming binary reward 1
        return current_q + alpha_pos * (target - current_q)
    else:            # Assuming binary reward 0
        return current_q + alpha_neg * (target - current_q)

def update_habit_trace(
    habit_values: np.ndarray,
    action: int,
    alpha: float
) -> np.ndarray:
    """
    Updates habit strength (p44). Strengthens chosen, weakens unchosen.
    
    H(chosen) += alpha * (1 - H(chosen))
    H(unchosen) += alpha * (0 - H(unchosen))
    
    Returns:
        Updated habit values (copy).
    """
    h_new = habit_values.copy()
    # Strengthen chosen
    h_new[action] += alpha * (1.0 - h_new[action])
    # Weaken unchosen (assuming binary choice for simplicity of this primitive)
    unchosen = 1 - action
    if 0 <= unchosen < len(h_new):
        h_new[unchosen] += alpha * (0.0 - h_new[unchosen])
    return h_new

# =============================================================================
# 5. Decay / Memory Mechanisms
# =============================================================================

def apply_memory_decay(
    q_values: np.ndarray, 
    decay_rate: float,
    chosen_action: Optional[int] = None
) -> np.ndarray:
    """
    Decays Q-values towards zero.
    Used in p42.
    
    Args:
        q_values: Current Q-values.
        decay_rate: Rate of decay [0, 1].
        chosen_action: If provided, only UNCHOSEN actions decay. 
                       If None, ALL actions decay.
                       
    Returns:
        Decayed Q-values (copy).
    """
    q_new = q_values.copy()
    
    if chosen_action is not None:
        # Decay only unchosen
        mask = np.ones(q_new.shape, dtype=bool)
        if 0 <= chosen_action < len(q_new):
            mask[chosen_action] = False
        q_new[mask] *= (1.0 - decay_rate)
    else:
        # Decay all
        q_new *= (1.0 - decay_rate)
        
    return q_new

# ============ Auto-generated fixes ============

# Fix for p37
def td_update_stage1_direct_reward(q_stage1: np.ndarray, action: int, reward: float, alpha: float) -> np.ndarray:
    """
    Updates Stage 1 Q-values directly based on the final reward (TD(1)-like).
    Ignores Stage 2 values.
    """
    q_new = q_stage1.copy()
    prediction_error = reward - q_new[action]
    q_new[action] += alpha * prediction_error
    return q_new


# ============ Auto-generated fixes ============

# Fix for p19
def add_context_dependent_perseveration(
    q_values: np.ndarray,
    last_action: Optional[int],
    bonus: float,
    current_state: Optional[int] = None,
    last_state: Optional[int] = None
) -> np.ndarray:
    """
    Adds stickiness bonus to last_action. 
    If state context is provided (e.g. Stage 2), only applies if current_state == last_state.
    If state context is None (e.g. Stage 1), applies unconditionally.
    """
    q_new = q_values.copy()
    should_apply = False
    if last_action is not None:
        if current_state is not None and last_state is not None:
            if current_state == last_state:
                should_apply = True
        else:
            should_apply = True
            
    if should_apply:
        q_new[int(last_action)] += bonus
    return q_new

# Fix for p20
def add_state_dependent_perseveration_bonus(
    q_values: np.ndarray, 
    last_action: Optional[int], 
    bonus: float,
    last_state: Optional[int] = None,
    current_state: Optional[int] = None
) -> np.ndarray:
    """
    Adds stickiness bonus to last_action. 
    If state context is provided (Stage 2), only adds if current_state matches last_state.
    """
    q_new = q_values.copy()
    
    # Check state dependency if states are provided
    if last_state is not None and current_state is not None:
        if last_state != current_state:
            return q_new

    if last_action is not None:
        q_new[int(last_action)] += bonus
        
    return q_new

# Fix for p23
def td_update_reward_based_asymmetric(current_q: float, target: float, reward: float, alpha_pos: float, alpha_neg: float) -> float:
    """
    Updates Q-value using an alpha determined by the reward outcome.
    Matches p23: if reward > 0.5 use alpha_pos, else alpha_neg.
    """
    if reward > 0.5:
        alpha = alpha_pos
    else:
        alpha = alpha_neg
    
    # Ensure alpha stays within bounds (Original model clips alpha_neg at init)
    alpha = max(0.0, min(1.0, alpha))
    
    return current_q + alpha * (target - current_q)

# Fix for p25
def stai_amplification_unit_bias(param: float, stai: float) -> float:
    """
    Amplifies a base parameter by a factor of (1 + 1.0 * stai).
    Used for fixed-bias anxiety modulation (e.g. p25).
    
    Returns: param * (1.0 + stai)
    """
    return param * (1.0 + stai)

# Fix for p29
def stage1_q_learning_update(q_stage1: np.ndarray, action: int, q_stage2: np.ndarray, state: int, alpha: float) -> np.ndarray:
    """
    Updates Stage 1 Q-values using Q-learning (Max over Stage 2).
    Target = max(Q_stage2[state])
    """
    q_new = q_stage1.copy()
    target = np.max(q_stage2[state])
    prediction_error = target - q_new[action]
    q_new[action] += alpha * prediction_error
    return q_new

# Fix for p33
def td_update_stage1_sarsa(q_stage1: np.ndarray, action: int, q_stage2: np.ndarray, state: int, action_2: int, alpha: float) -> np.ndarray:
    """
    Performs a SARSA update for Stage 1 (using the value of the action actually taken in Stage 2).
    """
    q_new = q_stage1.copy()
    target = q_stage2[state, action_2]
    prediction_error = target - q_new[action]
    q_new[action] += alpha * prediction_error
    return q_new

# Fix for p34
def update_sequential_sarsa(q_stage1: np.ndarray, q_stage2: np.ndarray, action_1: int, state_2: int, action_2: int, reward: float, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates Q-values for both stages sequentially.
    Stage 2 is updated first. Stage 1 is updated using the NEW Stage 2 value of the chosen action (SARSA).
    """
    # Update Stage 2
    q2_new = q_stage2.copy()
    delta_2 = reward - q2_new[state_2, action_2]
    q2_new[state_2, action_2] += alpha * delta_2
    
    # Update Stage 1 using UPDATED Stage 2 value of chosen action
    q1_new = q_stage1.copy()
    delta_1 = q2_new[state_2, action_2] - q1_new[action_1]
    q1_new[action_1] += alpha * delta_1
    
    return q1_new, q2_new

# Fix for p36
def add_anxiety_perseveration_bonus(q_values: np.ndarray, last_action: Optional[int], p_base: float, p_slope: float, stai: float) -> np.ndarray:
    """
    Adds a perseveration bonus modulated linearly by STAI.
    Hypothesis: Anxiety increases choice perseveration.
    Bonus = p_base + (stai * p_slope)
    """
    q_new = q_values.copy()
    if last_action is not None:
        bonus = p_base + (stai * p_slope)
        q_new[int(last_action)] += bonus
    return q_new

# Fix for p38
def td_update_stage1_sarsa(q_stage1: np.ndarray, action: int, q_stage2: np.ndarray, state: int, action_2: int, alpha: float) -> np.ndarray:
    """
    Updates Stage 1 Q-values towards the value of the chosen action in Stage 2 (SARSA).
    """
    q_new = q_stage1.copy()
    target = q_stage2[state, action_2]
    prediction_error = target - q_new[action]
    q_new[action] += alpha * prediction_error
    return q_new

# Fix for p42
def decay_stage1_p42(q_stage1: np.ndarray, action: int, decay_rate: float) -> np.ndarray:
    """
    Decays unchosen Stage 1 options with rate clipping (p42).
    """
    decay_rate = np.clip(decay_rate, 0.0, 1.0)
    q_new = q_stage1.copy()
    unchosen = 1 - action
    q_new[unchosen] *= (1.0 - decay_rate)
    return q_new

def decay_stage2_p42(q_stage2: np.ndarray, state: int, action: int, decay_rate: float) -> np.ndarray:
    """
    Decays unchosen Stage 2 options in visited state AND all options in unvisited state (p42).
    """
    decay_rate = np.clip(decay_rate, 0.0, 1.0)
    q_new = q_stage2.copy()
    
    # Decay unchosen in visited state
    unchosen_visited = 1 - action
    q_new[state, unchosen_visited] *= (1.0 - decay_rate)
    
    # Decay all in unvisited state
    unvisited_state = 1 - state
    q_new[unvisited_state, :] *= (1.0 - decay_rate)
    
    return q_new
