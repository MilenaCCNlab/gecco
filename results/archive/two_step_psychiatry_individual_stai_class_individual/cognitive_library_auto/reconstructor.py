import numpy as np
from typing import Optional, Tuple, Union, Any, Dict, List
from abc import ABC, abstractmethod
import ast

# =============================================================================
# 1. Primitives (Embedded from prompt)
# =============================================================================

def softmax(values: np.ndarray, beta: float) -> np.ndarray:
    exponentiated = np.exp(beta * (values - np.max(values)))
    return exponentiated / np.sum(exponentiated)

def compute_mb_values(transition_matrix: np.ndarray, q_stage2: np.ndarray) -> np.ndarray:
    state_values = np.max(q_stage2, axis=1)
    return transition_matrix @ state_values

def stai_multiplicative(param: float, stai: float) -> float:
    return param * stai

def stai_additive(base: float, slope: float, stai: float) -> float:
    return base + (slope * stai)

def stai_inverse_linear(param: float, stai: float) -> float:
    return param * (1.0 - stai)

def stai_inverse_division(param: float, stai: float) -> float:
    return param * (1.0 / (1.0 + stai))

def stai_affine_amplification(base: float, bias_factor: float, stai: float) -> float:
    return base * (1.0 + bias_factor * stai)

def add_perseveration_bonus(q_values: np.ndarray, last_action: Optional[int], bonus: float) -> np.ndarray:
    q_new = q_values.copy()
    if last_action is not None:
        q_new[int(last_action)] += bonus
    return q_new

def add_win_stay_bonus(q_values: np.ndarray, last_action: Optional[int], last_reward: float, bonus: float) -> np.ndarray:
    q_new = q_values.copy()
    if last_action is not None and last_reward > 0.5:
        q_new[int(last_action)] += bonus
    return q_new

def mb_mf_mixture(q_mb: np.ndarray, q_mf: np.ndarray, w: float) -> np.ndarray:
    w_clipped = np.clip(w, 0.0, 1.0)
    return (w_clipped * q_mb) + ((1.0 - w_clipped) * q_mf)

def add_habit_influence(q_values: np.ndarray, habit_values: np.ndarray, weight: float) -> np.ndarray:
    return q_values + (weight * habit_values)

def td_update_stage1(q_stage1: np.ndarray, action: int, q_next_value: float, alpha: float) -> np.ndarray:
    q_new = q_stage1.copy()
    prediction_error = q_next_value - q_new[action]
    q_new[action] += alpha * prediction_error
    return q_new

def td_update_stage2(q_stage2: np.ndarray, state: int, action: int, reward: float, alpha: float) -> np.ndarray:
    q_new = q_stage2.copy()
    prediction_error = reward - q_new[state, action]
    q_new[state, action] += alpha * prediction_error
    return q_new

def td_update_asymmetric(current_q: float, target: float, alpha_pos: float, alpha_neg: float) -> float:
    if target > 0.5:
        return current_q + alpha_pos * (target - current_q)
    else:
        return current_q + alpha_neg * (target - current_q)

def update_habit_trace(habit_values: np.ndarray, action: int, alpha: float) -> np.ndarray:
    h_new = habit_values.copy()
    h_new[action] += alpha * (1.0 - h_new[action])
    unchosen = 1 - action
    if 0 <= unchosen < len(h_new):
        h_new[unchosen] += alpha * (0.0 - h_new[unchosen])
    return h_new

def apply_memory_decay(q_values: np.ndarray, decay_rate: float, chosen_action: Optional[int] = None) -> np.ndarray:
    q_new = q_values.copy()
    if chosen_action is not None:
        mask = np.ones(q_new.shape, dtype=bool)
        if 0 <= chosen_action < len(q_new):
            mask[chosen_action] = False
        q_new[mask] *= (1.0 - decay_rate)
    else:
        q_new *= (1.0 - decay_rate)
    return q_new

def td_update_stage1_direct_reward(q_stage1: np.ndarray, action: int, reward: float, alpha: float) -> np.ndarray:
    q_new = q_stage1.copy()
    prediction_error = reward - q_new[action]
    q_new[action] += alpha * prediction_error
    return q_new

def add_context_dependent_perseveration(q_values: np.ndarray, last_action: Optional[int], bonus: float, current_state: Optional[int] = None, last_state: Optional[int] = None) -> np.ndarray:
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

def add_state_dependent_perseveration_bonus(q_values: np.ndarray, last_action: Optional[int], bonus: float, last_state: Optional[int] = None, current_state: Optional[int] = None) -> np.ndarray:
    q_new = q_values.copy()
    if last_state is not None and current_state is not None:
        if last_state != current_state:
            return q_new
    if last_action is not None:
        q_new[int(last_action)] += bonus
    return q_new

def td_update_reward_based_asymmetric(current_q: float, target: float, reward: float, alpha_pos: float, alpha_neg: float) -> float:
    if reward > 0.5:
        alpha = alpha_pos
    else:
        alpha = alpha_neg
    alpha = max(0.0, min(1.0, alpha))
    return current_q + alpha * (target - current_q)

def stai_amplification_unit_bias(param: float, stai: float) -> float:
    return param * (1.0 + stai)

def stage1_q_learning_update(q_stage1: np.ndarray, action: int, q_stage2: np.ndarray, state: int, alpha: float) -> np.ndarray:
    q_new = q_stage1.copy()
    target = np.max(q_stage2[state])
    prediction_error = target - q_new[action]
    q_new[action] += alpha * prediction_error
    return q_new

def td_update_stage1_sarsa(q_stage1: np.ndarray, action: int, q_stage2: np.ndarray, state: int, action_2: int, alpha: float) -> np.ndarray:
    q_new = q_stage1.copy()
    target = q_stage2[state, action_2]
    prediction_error = target - q_new[action]
    q_new[action] += alpha * prediction_error
    return q_new

def update_sequential_sarsa(q_stage1: np.ndarray, q_stage2: np.ndarray, action_1: int, state_2: int, action_2: int, reward: float, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    q2_new = q_stage2.copy()
    delta_2 = reward - q2_new[state_2, action_2]
    q2_new[state_2, action_2] += alpha * delta_2
    q1_new = q_stage1.copy()
    delta_1 = q2_new[state_2, action_2] - q1_new[action_1]
    q1_new[action_1] += alpha * delta_1
    return q1_new, q2_new

def add_anxiety_perseveration_bonus(q_values: np.ndarray, last_action: Optional[int], p_base: float, p_slope: float, stai: float) -> np.ndarray:
    q_new = q_values.copy()
    if last_action is not None:
        bonus = p_base + (stai * p_slope)
        q_new[int(last_action)] += bonus
    return q_new

def decay_stage1_p42(q_stage1: np.ndarray, action: int, decay_rate: float) -> np.ndarray:
    decay_rate = np.clip(decay_rate, 0.0, 1.0)
    q_new = q_stage1.copy()
    unchosen = 1 - action
    q_new[unchosen] *= (1.0 - decay_rate)
    return q_new

def decay_stage2_p42(q_stage2: np.ndarray, state: int, action: int, decay_rate: float) -> np.ndarray:
    decay_rate = np.clip(decay_rate, 0.0, 1.0)
    q_new = q_stage2.copy()
    unchosen_visited = 1 - action
    q_new[state, unchosen_visited] *= (1.0 - decay_rate)
    unvisited_state = 1 - state
    q_new[unvisited_state, :] *= (1.0 - decay_rate)
    return q_new

# =============================================================================
# 2. Specifications
# =============================================================================

PARTICIPANT_SPECS = {
    "p18": {"class": "ParticipantModel2", "primitives": ['helper::softmax', 'stai_modulation::stai_inverse_division', 'policy::add_perseveration_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'phi'], "stai_modulation": "{'phi': 'inverse_division'}", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'phi': [0, 5]}, "bic": 305.38341476794284},
    "p19": {"class": "ParticipantModel1", "primitives": ['helper::softmax', 'stai_modulation::stai_additive', 'policy::add_context_dependent_perseveration', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'stick_base', 'stick_stai_slope'], "stai_modulation": "additive", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stick_base': [0, 5], 'stick_stai_slope': [-5, 5]}, "bic": 0.0},
    "p20": {"class": "ParticipantModel3", "primitives": ['helper::softmax', 'stai_modulation::stai_additive', 'policy::add_state_dependent_perseveration_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'stickiness_base', 'anxiety_stick'], "stai_modulation": "additive", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stickiness_base': [0, 5], 'anxiety_stick': [0, 5]}, "bic": 381.9787884185716},
    "p21": {"class": "ParticipantModel2", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'stickiness'], "stai_modulation": "multiplicative", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stickiness': [0, 5]}, "bic": None},
    "p22": {"class": "ParticipantModel3", "primitives": ['helper::softmax', 'stai_modulation::stai_additive', 'policy::add_perseveration_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'stick_base', 'stick_stai'], "stai_modulation": "additive", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stick_base': [-2, 2], 'stick_stai': [0, 5]}, "bic": 0.0},
    "p23": {"class": "ParticipantModel2", "primitives": ['helper::softmax', 'stai_modulation::stai_affine_amplification', 'value_update::td_update_reward_based_asymmetric'], "parameters": ['alpha_base', 'beta', 'bias_factor'], "stai_modulation": "affine_amplification", "bounds": {'alpha_base': [0, 1], 'beta': [0, 10], 'bias_factor': [-1, 2]}, "bic": 353.6022018424507},
    "p24": {"class": "ParticipantModel1", "primitives": ['helper::compute_mb_values', 'value_update::td_update_stage1', 'value_update::td_update_stage2', 'stai_modulation::stai_inverse_linear', 'policy::mb_mf_mixture', 'policy::add_perseveration_bonus', 'helper::softmax'], "parameters": ['alpha', 'beta', 'w_max', 'perseveration'], "stai_modulation": "{'w_max': 'inverse_linear'}", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'w_max': [0, 1], 'perseveration': [0, 5]}, "bic": 0.0},
    "p25": {"class": "ParticipantModel2", "primitives": ['helper::softmax', 'stai_modulation::stai_amplification_unit_bias', 'policy::add_perseveration_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'k_stick'], "stai_modulation": "unit_amplification", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'k_stick': [0, 5]}, "bic": 119.8063597714025},
    "p26": {"class": "ParticipantModel2", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'stickiness_factor'], "stai_modulation": "multiplicative", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stickiness_factor': [0, 5]}, "bic": 0.0},
    "p27": {"class": "ParticipantModel3", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'pers_k'], "stai_modulation": "{'pers_k': 'multiplicative'}", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'pers_k': [0, 5]}, "bic": 0.0},
    "p28": {"class": "ParticipantModel1", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_win_stay_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'win_bonus'], "stai_modulation": "multiplicative", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'win_bonus': [0, 5]}, "bic": 0.0},
    "p29": {"class": "ParticipantModel3", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::stage1_q_learning_update', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'stickiness'], "stai_modulation": "multiplicative", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stickiness': [0, 5]}, "bic": 256.5459180268013},
    "p30": {"class": "ParticipantModel2", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'stick_sensitivity'], "stai_modulation": "multiplicative", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stick_sensitivity': [0, 5]}, "bic": 477.67521678131635},
    "p31": {"class": "ParticipantModel2", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'stickiness'], "stai_modulation": "multiplicative", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stickiness': [0, 5]}, "bic": 373.64927906442557},
    "p32": {"class": "ParticipantModel3", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'k'], "stai_modulation": "multiplicative", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'k': [0, 5]}, "bic": 0},
    "p33": {"class": "ParticipantModel2", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::td_update_stage2', 'value_update::td_update_stage1_sarsa'], "parameters": ['alpha', 'beta', 'stickiness'], "stai_modulation": "multiplicative", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stickiness': [0, 5]}, "bic": 386.37827393527584},
    "p34": {"class": "ParticipantModel3", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::update_sequential_sarsa'], "parameters": ['alpha', 'beta', 'rho'], "stai_modulation": "multiplicative", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'rho': [0, 5]}, "bic": 373.8582871089093},
    "p35": {"class": "ParticipantModel3", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_win_stay_bonus'], "parameters": ['alpha', 'beta', 'win_stay_bonus'], "stai_modulation": "multiplicative", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'win_stay_bonus': [0, 5]}, "bic": 283.01581577064576},
    "p36": {"class": "ParticipantModel2", "primitives": ['helper::softmax', 'policy::add_anxiety_perseveration_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'p_base', 'p_slope'], "stai_modulation": "none", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'p_base': [0, 5], 'p_slope': [0, 5]}, "bic": 0.0},
    "p37": {"class": "ParticipantModel1", "primitives": ['helper::softmax', 'helper::compute_mb_values', 'stai_modulation::stai_inverse_linear', 'policy::mb_mf_mixture', 'value_update::td_update_stage2', 'value_update::td_update_stage1_direct_reward'], "parameters": ['alpha', 'beta', 'w_base'], "stai_modulation": "{'w_base': 'inverse_linear'}", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'w_base': [0, 1]}, "bic": 0},
    "p38": {"class": "ParticipantModel2", "primitives": ['helper::softmax', 'stai_modulation::stai_additive', 'policy::add_perseveration_bonus', 'value_update::td_update_stage1_sarsa', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'pers_base', 'stai_pers'], "stai_modulation": "additive", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'pers_base': [0, 5], 'stai_pers': [0, 5]}, "bic": 384.9927139795815},
    "p39": {"class": "ParticipantModel2", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'persev_w'], "stai_modulation": "{'persev_w': 'multiplicative'}", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'persev_w': [0, 5]}, "bic": 327.36430783923794},
    "p40": {"class": "ParticipantModel3", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'stickiness'], "stai_modulation": "multiplicative", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stickiness': [0, 5]}, "bic": 0.0},
    "p41": {"class": "ParticipantModel2", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::td_update_stage1', 'value_update::td_update_stage2'], "parameters": ['alpha', 'beta', 'stick_factor'], "stai_modulation": "multiplicative", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stick_factor': [0, 5]}, "bic": 271.3039688021288},
    "p42": {"class": "ParticipantModel2", "primitives": ['helper::softmax', 'value_update::td_update_stage1', 'value_update::td_update_stage2', 'stai_modulation::stai_additive', 'decay::decay_stage1_p42', 'decay::decay_stage2_p42'], "parameters": ['alpha', 'beta', 'decay_base', 'decay_stai'], "stai_modulation": "additive", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'decay_base': [0, 1], 'decay_stai': [0, 1]}, "bic": 0.0},
    "p43": {"class": "ParticipantModel1", "primitives": ['helper::softmax', 'helper::compute_mb_values', 'stai_modulation::stai_inverse_linear', 'policy::mb_mf_mixture', 'value_update::td_update_stage2', 'value_update::td_update_stage1'], "parameters": ['alpha', 'beta', 'w_base'], "stai_modulation": "inverse_linear", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'w_base': [0, 1]}, "bic": 0},
    "p44": {"class": "ParticipantModel1", "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_habit_influence', 'value_update::td_update_stage1', 'value_update::td_update_stage2', 'value_update::update_habit_trace'], "parameters": ['alpha', 'beta', 'habit_weight'], "stai_modulation": "multiplicative", "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'habit_weight': [0, 5]}, "bic": 402.74022725825887},
}

# =============================================================================
# 3. Base Class
# =============================================================================

class CognitiveModelBase(ABC):
    def __init__(self, n_trials: int, stai: float, model_parameters: tuple):
        self.n_trials = n_trials
        self.n_choices = 2
        self.n_states = 2
        self.stai = stai
        self.T = np.array([[0.7, 0.3], [0.3, 0.7]])
        self.p_choice_1 = np.zeros(n_trials)
        self.p_choice_2 = np.zeros(n_trials)
        self.q_stage1 = np.zeros(self.n_choices)
        self.q_stage2 = np.zeros((self.n_states, self.n_choices))
        self.trial = 0
        self.last_action1 = None
        self.last_action2 = None
        self.last_state = None
        self.last_reward = None
        self.unpack_parameters(model_parameters)
        self.init_model()

    @abstractmethod
    def unpack_parameters(self, model_parameters: tuple) -> None:
        pass

    def init_model(self) -> None:
        pass

    def policy_stage1(self) -> np.ndarray:
        return self.softmax(self.q_stage1, self.beta)

    def policy_stage2(self, state: int) -> np.ndarray:
        return self.softmax(self.q_stage2[state], self.beta)

    def value_update(self, action_1: int, state: int, action_2: int, reward: float) -> None:
        delta_2 = reward - self.q_stage2[state, action_2]
        self.q_stage2[state, action_2] += self.alpha * delta_2
        delta_1 = self.q_stage2[state, action_2] - self.q_stage1[action_1]
        self.q_stage1[action_1] += self.alpha * delta_1

    def pre_trial(self) -> None:
        pass

    def post_trial(self, action_1: int, state: int, action_2: int, reward: float) -> None:
        self.last_action1 = action_1
        self.last_action2 = action_2
        self.last_state = state
        self.last_reward = reward

    def run_model(self, action_1, state, action_2, reward) -> float:
        for self.trial in range(self.n_trials):
            a1, s = int(action_1[self.trial]), int(state[self.trial])
            a2, r = int(action_2[self.trial]), float(reward[self.trial])
            
            self.pre_trial()
            self.p_choice_1[self.trial] = self.policy_stage1()[a1]
            self.p_choice_2[self.trial] = self.policy_stage2(s)[a2]
            self.value_update(a1, s, a2, r)
            self.post_trial(a1, s, a2, r)
        
        return self.compute_nll()
    
    def compute_nll(self) -> float:
        eps = 1e-12
        return -(np.sum(np.log(self.p_choice_1 + eps)) + np.sum(np.log(self.p_choice_2 + eps)))
    
    def softmax(self, values: np.ndarray, beta: float) -> np.ndarray:
        centered = values - np.max(values)
        exp_vals = np.exp(beta * centered)
        return exp_vals / np.sum(exp_vals)

def make_cognitive_model(ModelClass):
    def cognitive_model(action_1, state, action_2, reward, stai, model_parameters):
        n_trials = len(action_1)
        stai_val = float(stai[0]) if hasattr(stai, '__len__') else float(stai)
        model = ModelClass(n_trials, stai_val, model_parameters)
        return model.run_model(action_1, state, action_2, reward)
    return cognitive_model

# =============================================================================
# 4. Reconstructor
# =============================================================================

def reconstruct_model(participant_id: str):
    spec = PARTICIPANT_SPECS.get(participant_id)
    if not spec:
        raise ValueError(f"No spec for {participant_id}")

    class DynamicModel(CognitiveModelBase):
        def unpack_parameters(self, model_parameters: tuple) -> None:
            params = spec['parameters']
            for i, p_name in enumerate(params):
                setattr(self, p_name, model_parameters[i])

        def init_model(self) -> None:
            # 1. Initialization logic
            if 'helper::compute_mb_values' in spec['primitives']:
                self.q_mf = np.zeros(self.n_choices)
                self.q_mb = np.zeros(self.n_choices)
            
            if 'value_update::update_habit_trace' in spec['primitives']:
                self.habit = np.zeros(self.n_choices)

            # 2. STAI Modulation
            mod_spec = spec.get('stai_modulation')
            
            if isinstance(mod_spec, str):
                if mod_spec == 'multiplicative':
                    target = next((p for p in spec['parameters'] if p not in ['alpha', 'beta']), None)
                    # Specific overrides for non-standard parameter names
                    if participant_id == 'p28': target = 'win_bonus'
                    if participant_id == 'p26': target = 'stickiness_factor'
                    if participant_id == 'p32': target = 'k'
                    if participant_id == 'p41': target = 'stick_factor'
                    
                    if target:
                        val = getattr(self, target)
                        setattr(self, f"{target}_eff", val * self.stai)

                elif mod_spec == 'additive':
                    base = next((p for p in spec['parameters'] if 'base' in p), None)
                    slope = next((p for p in spec['parameters'] if 'stai' in p or 'slope' in p or 'anxiety' in p), None)
                    if base and slope:
                        b_val = getattr(self, base)
                        s_val = getattr(self, slope)
                        setattr(self, 'effective_stickiness', b_val + (s_val * self.stai))
                        
                elif mod_spec == 'affine_amplification':
                    if 'bias_factor' in spec['parameters']:
                        self.alpha_pos = getattr(self, 'alpha_base', 0.1)
                        self.alpha_neg = np.clip(self.alpha_pos * (1.0 + self.stai * getattr(self, 'bias_factor')), 0.0, 1.0)
                
                elif mod_spec == 'unit_amplification':
                    if 'k_stick' in spec['parameters']:
                        self.k_stick_eff = getattr(self, 'k_stick') * (1.0 + self.stai)

                elif mod_spec == 'inverse_linear':
                    target = next((p for p in spec['parameters'] if 'w_' in p), None)
                    if target:
                        val = getattr(self, target)
                        setattr(self, 'w_eff', val * (1.0 - self.stai))

            elif mod_spec and isinstance(mod_spec, str) and mod_spec.startswith("{"):
                d = ast.literal_eval(mod_spec)
                for param, func_type in d.items():
                    val = getattr(self, param)
                    if func_type == 'inverse_division':
                        res = val * (1.0 / (1.0 + self.stai))
                    elif func_type == 'inverse_linear':
                        res = val * (1.0 - self.stai)
                    elif func_type == 'multiplicative':
                        res = val * self.stai
                    setattr(self, f"{param}_eff", res)

        def policy_stage1(self) -> np.ndarray:
            prims = spec['primitives']
            q_vals = self.q_stage1.copy()

            if hasattr(self, 'q_mf'):
                # MB/MF Mixture
                v_stage2 = np.max(self.q_stage2, axis=1)
                q_mb = self.T @ v_stage2
                w_val = getattr(self, 'w_eff', getattr(self, 'w_max_eff', getattr(self, 'w_base_eff', getattr(self, 'w_base', 0.5))))
                q_vals = mb_mf_mixture(q_mb, self.q_mf, w_val)

            # Perseveration
            if 'policy::add_perseveration_bonus' in prims or \
               'policy::add_context_dependent_perseveration' in prims or \
               'policy::add_anxiety_perseveration_bonus' in prims:
                
                bonus = 0.0
                eff_candidates = [k for k in self.__dict__ if '_eff' in k or 'effective' in k]
                stick_cands = [k for k in eff_candidates if 'stick' in k or 'phi' in k or 'pers' in k or 'k' in k or 'rho' in k]
                
                if stick_cands:
                    bonus = getattr(self, stick_cands[0])
                elif 'p36' in participant_id:
                     bonus = getattr(self, 'p_base', 0) + (self.stai * getattr(self, 'p_slope', 0))
                else:
                    raw_cands = [p for p in spec['parameters'] if 'stick' in p or 'phi' in p or 'pers' in p or 'rho' in p or 'k' in p]
                    if raw_cands:
                        bonus = getattr(self, raw_cands[0])

                if 'policy::add_context_dependent_perseveration' in prims:
                    q_vals = add_context_dependent_perseveration(q_vals, self.last_action1, bonus, None, None)
                elif 'policy::add_anxiety_perseveration_bonus' in prims:
                    q_vals = add_anxiety_perseveration_bonus(q_vals, self.last_action1, getattr(self, 'p_base', 0), getattr(self, 'p_slope', 0), self.stai)
                else:
                    q_vals = add_perseveration_bonus(q_vals, self.last_action1, bonus)
            
            # Win-Stay
            if 'policy::add_win_stay_bonus' in prims:
                bonus = 0.0
                if hasattr(self, 'win_bonus_eff'): bonus = self.win_bonus_eff
                elif hasattr(self, 'cling_factor_eff'): bonus = self.cling_factor_eff
                elif hasattr(self, 'win_stay_bonus_eff'): bonus = self.win_stay_bonus_eff
                else:
                    if 'cling_factor' in spec['parameters']:
                        bonus = getattr(self, 'cling_factor') * self.stai
                    elif 'win_bonus' in spec['parameters']:
                        bonus = getattr(self, 'win_bonus', 0)
                q_vals = add_win_stay_bonus(q_vals, self.last_action1, self.last_reward, bonus)

            # Habit
            if 'policy::add_habit_influence' in prims:
                w = getattr(self, 'habit_weight', 0) * self.stai
                q_vals = add_habit_influence(q_vals, self.habit, w)
                
            return self.softmax(q_vals, self.beta)

        def policy_stage2(self, state: int) -> np.ndarray:
            q_vals = self.q_stage2[state].copy()
            prims = spec['primitives']
            
            if 'policy::add_context_dependent_perseveration' in prims:
                bonus = getattr(self, 'effective_stickiness', 0.0)
                q_vals = add_context_dependent_perseveration(q_vals, self.last_action2, bonus, current_state=state, last_state=self.last_state)
            
            elif 'policy::add_state_dependent_perseveration_bonus' in prims:
                bonus = getattr(self, 'effective_stickiness', 0.0)
                q_vals = add_state_dependent_perseveration_bonus(q_vals, self.last_action2, bonus, last_state=self.last_state, current_state=state)
            
            return self.softmax(q_vals, self.beta)

        def value_update(self, action_1: int, state: int, action_2: int, reward: float) -> None:
            prims = spec['primitives']
            
            if 'value_update::update_sequential_sarsa' in prims:
                self.q_stage1, self.q_stage2 = update_sequential_sarsa(self.q_stage1, self.q_stage2, action_1, state, action_2, reward, self.alpha)
                return

            # Stage 2
            if 'value_update::td_update_stage2' in prims:
                if 'value_update::td_update_reward_based_asymmetric' in prims:
                    # p23 logic
                    q_old = self.q_stage2[state, action_2]
                    q_new = td_update_reward_based_asymmetric(q_old, reward, reward, self.alpha_pos, self.alpha_neg)
                    self.q_stage2[state, action_2] = q_new
                    
                    target = self.q_stage2[state, action_2]
                    q1_old = self.q_stage1[action_1]
                    self.q_stage1[action_1] = td_update_reward_based_asymmetric(q1_old, target, reward, self.alpha_pos, self.alpha_neg)
                    return 
                else:
                    self.q_stage2 = td_update_stage2(self.q_stage2, state, action_2, reward, self.alpha)

            # Stage 1
            if 'value_update::td_update_stage1' in prims:
                if hasattr(self, 'q_mf'):
                    if 'value_update::td_update_stage1_direct_reward' in prims:
                        self.q_mf = td_update_stage1_direct_reward(self.q_mf, action_1, reward, self.alpha)
                    else:
                        target = self.q_stage2[state, action_2]
                        self.q_mf = td_update_stage1(self.q_mf, action_1, target, self.alpha)
                else:
                    target = self.q_stage2[state, action_2]
                    self.q_stage1 = td_update_stage1(self.q_stage1, action_1, target, self.alpha)

            elif 'value_update::stage1_q_learning_update' in prims:
                self.q_stage1 = stage1_q_learning_update(self.q_stage1, action_1, self.q_stage2, state, self.alpha)

            elif 'value_update::td_update_stage1_sarsa' in prims:
                self.q_stage1 = td_update_stage1_sarsa(self.q_stage1, action_1, self.q_stage2, state, action_2, self.alpha)


        def post_trial(self, action_1, state, action_2, reward):
            super().post_trial(action_1, state, action_2, reward)
            prims = spec['primitives']
            
            if 'value_update::update_habit_trace' in prims:
                self.habit = update_habit_trace(self.habit, action_1, self.alpha)
                
            if 'decay::decay_stage1_p42' in prims:
                self.q_stage1 = decay_stage1_p42(self.q_stage1, action_1, self.decay_rate)
                self.q_stage2 = decay_stage2_p42(self.q_stage2, state, action_2, self.decay_rate)

    return DynamicModel

def reconstruct_model_func(participant_id: str):
    ModelClass = reconstruct_model(participant_id)
    return make_cognitive_model(ModelClass)