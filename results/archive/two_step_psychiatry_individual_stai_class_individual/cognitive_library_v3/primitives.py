import numpy as np

# -----------------------------------------------------------------------------
# 1. Helper Functions
# -----------------------------------------------------------------------------

def softmax(values: np.ndarray, beta: float) -> np.ndarray:
    """
    Computes the softmax probability distribution over values.
    
    Args:
        values (np.ndarray): Array of value estimates (e.g., Q-values).
        beta (float): Inverse temperature parameter. Higher beta -> more greedy.
        
    Returns:
        np.ndarray: Probability distribution summing to 1.
    """
    # Subtract max for numerical stability to avoid overflow
    scaled_values = beta * (values - np.max(values))
    exp_values = np.exp(scaled_values)
    return exp_values / np.sum(exp_values)

# -----------------------------------------------------------------------------
# 2. STAI Modulation (Parameter Transformation)
# -----------------------------------------------------------------------------

def stai_linear_additive(base: float, slope: float, stai: float) -> float:
    """
    Linearly modulates a parameter based on STAI.
    Used in p19, p22, p36, p38.
    
    Formula: param = base + (slope * stai)
    """
    return base + (slope * stai)

def stai_multiplicative(factor: float, stai: float) -> float:
    """
    Calculates a parameter strictly proportional to STAI.
    Used in p21, p26, p27, p29, p30, p31, p32, p33, p34, p39, p40, p41.
    
    Formula: param = factor * stai
    """
    return factor * stai

def stai_inverse_linear(max_val: float, stai: float) -> float:
    """
    Reduces a parameter linearly as STAI increases.
    Used for Model-Based weights (p24, p37, p43).
    
    Formula: param = max_val * (1 - stai)
    """
    return max_val * (1.0 - stai)

def stai_inverse_division(base: float, stai: float) -> float:
    """
    Reduces a parameter non-linearly using division.
    Used in p18.
    
    Formula: param = base * (1 / (1 + stai))
    """
    return base * (1.0 / (1.0 + stai))

def stai_scaling(base: float, scale_factor: float, stai: float) -> float:
    """
    Scales a base parameter by a factor dependent on STAI.
    Used in p23 (learning rate bias) and p25.
    
    Formula: param = base * (1 + scale_factor * stai)
    """
    return base * (1.0 + (scale_factor * stai))

# -----------------------------------------------------------------------------
# 3. Policy Components (Action Selection)
# -----------------------------------------------------------------------------

def add_perseveration_bonus(q_values: np.ndarray, 
                            last_action: int, 
                            bonus: float) -> np.ndarray:
    """
    Adds a stickiness/perseveration bonus to the Q-value of the previously chosen action.
    This creates a tendency to repeat choices.
    
    Args:
        q_values (np.ndarray): Current Q-values.
        last_action (int or None): The index of the action taken in the previous trial.
        bonus (float): The magnitude of the bonus to add.
        
    Returns:
        np.ndarray: Modified Q-values (copy).
    """
    q_modified = q_values.copy()
    if last_action is not None:
        q_modified[int(last_action)] += bonus
    return q_modified

def add_win_stay_bonus(q_values: np.ndarray, 
                       last_action: int, 
                       last_reward: float, 
                       bonus: float) -> np.ndarray:
    """
    Adds a bonus to the previous action ONLY if it resulted in a reward.
    Implements the "Win-Stay" heuristic (p28, p35).
    
    Args:
        q_values (np.ndarray): Current Q-values.
        last_action (int or None): The index of the action taken in the previous trial.
        last_reward (float): The reward received in the previous trial (0 or 1).
        bonus (float): The magnitude of the bonus to add.
        
    Returns:
        np.ndarray: Modified Q-values (copy).
    """
    q_modified = q_values.copy()
    # Assuming binary reward where 1.0 is a "Win"
    if last_action is not None and last_reward == 1.0:
        q_modified[int(last_action)] += bonus
    return q_modified

def add_habit_influence(q_values: np.ndarray, 
                        habit_trace: np.ndarray, 
                        weight: float) -> np.ndarray:
    """
    Combines goal-directed values with habit traces (p44).
    
    Args:
        q_values (np.ndarray): Current Q-values.
        habit_trace (np.ndarray): Current habit strength for each action.
        weight (float): How strongly habits influence the decision.
        
    Returns:
        np.ndarray: Combined values.
    """
    return q_values + (weight * habit_trace)

def calculate_mb_values(transition_matrix: np.ndarray, 
                        q_stage2: np.ndarray) -> np.ndarray:
    """
    Calculates Model-Based values for Stage 1 based on transitions and Stage 2 values.
    
    Args:
        transition_matrix (np.ndarray): Shape (n_actions, n_states). Probabilities T(s|a).
        q_stage2 (np.ndarray): Shape (n_states, n_actions_stage2). Q-values for second stage.
        
    Returns:
        np.ndarray: Model-Based Q-values for Stage 1.
    """
    # V(s') = max_a' Q(s', a')
    state_values = np.max(q_stage2, axis=1)
    
    # Q_MB(a) = sum_s' T(s'|a) * V(s')
    # transition_matrix assumed to be [n_actions, n_states] based on p24, p37
    q_mb = transition_matrix @ state_values
    return q_mb

def mb_mf_mixture(q_mb: np.ndarray, q_mf: np.ndarray, w: float) -> np.ndarray:
    """
    Combines Model-Based and Model-Free values using a mixing weight.
    
    Args:
        q_mb (np.ndarray): Model-Based values.
        q_mf (np.ndarray): Model-Free values.
        w (float): Weight for Model-Based system (0 to 1).
        
    Returns:
        np.ndarray: Weighted sum of values.
    """
    # Clip w to ensure valid convex combination
    w = np.clip(w, 0.0, 1.0)
    return (w * q_mb) + ((1.0 - w) * q_mf)

# -----------------------------------------------------------------------------
# 4. Value Updates (Learning)
# -----------------------------------------------------------------------------

def update_q_td(current_q: float, 
                target: float, 
                alpha: float) -> float:
    """
    Performs a standard Temporal Difference (TD) or Rescorla-Wagner update.
    
    Args:
        current_q (float): The current Q-value estimate.
        target (float): The target value (Reward or Next State Value).
        alpha (float): Learning rate [0, 1].
        
    Returns:
        float: Updated Q-value.
    """
    prediction_error = target - current_q
    return current_q + (alpha * prediction_error)

def select_learning_rate(alpha_pos: float, 
                         alpha_neg: float, 
                         reward: float) -> float:
    """
    Selects between positive and negative learning rates based on outcome.
    Used in p23.
    
    Args:
        alpha_pos (float): Learning rate for rewards (Win).
        alpha_neg (float): Learning rate for omissions (Loss).
        reward (float): Outcome value.
        
    Returns:
        float: The selected alpha.
    """
    return alpha_pos if reward > 0.5 else alpha_neg

def update_habit_trace(habit_trace: np.ndarray, 
                       chosen_action: int, 
                       alpha: float) -> np.ndarray:
    """
    Updates habit traces (p44). Chosen action strengthens towards 1, 
    unchosen weakens towards 0.
    
    Args:
        habit_trace (np.ndarray): Current habit strengths.
        chosen_action (int): Index of action taken.
        alpha (float): Learning rate for habit formation.
        
    Returns:
        np.ndarray: Updated habit trace (copy).
    """
    new_habit = habit_trace.copy()
    # Strengthen chosen
    new_habit[chosen_action] += alpha * (1.0 - new_habit[chosen_action])
    # Weaken all others (assuming binary choice, index 1-chosen works for 2 options)
    unchosen = 1 - chosen_action
    new_habit[unchosen] += alpha * (0.0 - new_habit[unchosen])
    return new_habit

# -----------------------------------------------------------------------------
# 5. Decay / Memory
# -----------------------------------------------------------------------------

def apply_memory_decay(values: np.ndarray, 
                       indices_to_decay: list, 
                       decay_rate: float) -> np.ndarray:
    """
    Applies forgetting/decay to specific value entries (p42).
    
    Args:
        values (np.ndarray): The array of values (1D or 2D).
        indices_to_decay (list): List of indices (tuples if 2D) to decay.
        decay_rate (float): Rate of decay [0, 1].
        
    Returns:
        np.ndarray: Values with decay applied (copy).
    """
    decayed_values = values.copy()
    retention = 1.0 - decay_rate
    
    for idx in indices_to_decay:
        decayed_values[idx] *= retention
        
    return decayed_values

# ============ Auto-generated fixes ============

# Fix for p25
def stai_scaling_unitary(base: float, stai: float) -> float:
    """
    Scales a base parameter by (1 + STAI).
    Used in p25.
    Formula: param = base * (1.0 + stai)
    """
    return base * (1.0 + stai)

# Fix for p42
def apply_anxiety_decay_p42(q_stage1: np.ndarray, q_stage2: np.ndarray, action_1: int, state: int, action_2: int, decay_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies anxiety-modulated decay specifically for ParticipantModel2 (p42).
    Decays unchosen options in active contexts and ALL options in unvisited contexts.
    """
    # Clip decay rate to [0, 1] as per original model logic
    effective_decay = np.clip(decay_rate, 0.0, 1.0)
    retention = 1.0 - effective_decay
    
    new_q1 = q_stage1.copy()
    new_q2 = q_stage2.copy()
    
    # Decay unchosen Stage 1 option
    unchosen_1 = 1 - action_1
    new_q1[unchosen_1] *= retention
    
    # Decay unchosen Stage 2 option in the visited state
    unchosen_2 = 1 - action_2
    new_q2[state, unchosen_2] *= retention
    
    # Decay the unvisited state's values (both options)
    unvisited_state = 1 - state
    new_q2[unvisited_state, :] *= retention
    
    return new_q1, new_q2
