import numpy as np
import results.archive.two_step_psychiatry_individual_stai_class_individual.cognitive_library_v3.primitives as P
from abc import ABC, abstractmethod

# -----------------------------------------------------------------------------
# Participant Specifications
# -----------------------------------------------------------------------------

PARTICIPANT_SPECS = {
    "p18": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'stai_modulation::stai_inverse_division', 'policy::add_perseveration_bonus', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'phi'],
        "stai_modulation": "inverse_division",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'phi': [0, 5]},
        "bic": 305.38341476794284,
    },
    "p19": {
        "class": "ParticipantModel1",
        "primitives": ['helper::softmax', 'stai_modulation::stai_linear_additive', 'policy::add_perseveration_bonus'],
        "parameters": ['alpha', 'beta', 'stick_base', 'stick_stai_slope'],
        "stai_modulation": "additive",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stick_base': [0, 5], 'stick_stai_slope': [-5, 5]},
        "bic": 216.05809567881084,
    },
    "p20": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'stai_modulation::stai_linear_additive', 'policy::add_perseveration_bonus', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'stickiness_base', 'anxiety_stick'],
        "stai_modulation": "additive",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stickiness_base': [0, 5], 'anxiety_stick': [0, 5]},
        "bic": 381.9787884185716,
    },
    "p21": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'phi'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'phi': [0, 5]},
        "bic": 341.3765093808048,
    },
    "p22": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'stai_modulation::stai_linear_additive', 'policy::add_perseveration_bonus'],
        "parameters": ['alpha', 'beta', 'stick_base', 'stick_stai'],
        "stai_modulation": "additive",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stick_base': [-2, 2], 'stick_stai': [0, 5]},
        "bic": 331.7044420273894,
    },
    "p23": {
        "class": "ParticipantModel2",
        "primitives": ['stai_modulation::stai_scaling', 'value_update::select_learning_rate', 'value_update::update_q_td'],
        "parameters": ['alpha_base', 'beta', 'bias_factor'],
        "stai_modulation": "scaling",
        "bounds": {'alpha_base': [0, 1], 'beta': [0, 10], 'bias_factor': [-1, 2]},
        "bic": 353.6022018424507,
    },
    "p24": {
        "class": "ParticipantModel1",
        "primitives": ['helper::softmax', 'stai_modulation::stai_inverse_linear', 'policy::add_perseveration_bonus', 'policy::calculate_mb_values', 'policy::mb_mf_mixture', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'w_max', 'perseveration'],
        "stai_modulation": "inverse_linear",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'w_max': [0, 1], 'perseveration': [0, 5]},
        "bic": 0,
    },
    "p25": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'stai_modulation::stai_scaling_unitary', 'policy::add_perseveration_bonus', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'k_stick'],
        "stai_modulation": "stai_scaling_unitary",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'k_stick': [0, 5]},
        "bic": 119.8063597714025,
    },
    "p26": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'stickiness_factor'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stickiness_factor': [0, 5]},
        "bic": 242.96155239953993,
    },
    "p27": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'pers_k'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'pers_k': [0, 5]},
        "bic": 396.3592954960134,
    },
    "p28": {
        "class": "ParticipantModel1",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_win_stay_bonus'],
        "parameters": ['alpha', 'beta', 'win_bonus'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'win_bonus': [0, 5]},
        "bic": 361.42875904364337,
    },
    "p29": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::update_q_learning'],
        "parameters": ['alpha', 'beta', 'stickiness'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stickiness': [0, 5]},
        "bic": 0.0,
    },
    "p30": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus'],
        "parameters": ['alpha', 'beta', 'stick_sensitivity'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stick_sensitivity': [0, 5]},
        "bic": 477.67521678131635,
    },
    "p31": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus'],
        "parameters": ['alpha', 'beta', 'k_anx'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'k_anx': [0, 5]},
        "bic": 0.0,
    },
    "p32": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'k'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'k': [0, 5]},
        "bic": 0.0,
    },
    "p33": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'stickiness'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stickiness': [0, 5]},
        "bic": 0.0,
    },
    "p34": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'rho'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'rho': [0, 5]},
        "bic": 373.8582871089093,
    },
    "p35": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_win_stay_bonus'],
        "parameters": ['alpha', 'beta', 'cling_factor'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'cling_factor': [0, 5]},
        "bic": 0.0,
    },
    "p36": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'stai_modulation::stai_linear_additive', 'policy::add_perseveration_bonus', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'p_base', 'p_slope'],
        "stai_modulation": "additive",
        "bounds": {'alpha': [0.0, 1.0], 'beta': [0.0, 10.0], 'p_base': [0.0, 5.0], 'p_slope': [0.0, 5.0]},
        "bic": 458.62585197173024,
    },
    "p37": {
        "class": "ParticipantModel1",
        "primitives": ['helper::softmax', 'stai_modulation::stai_inverse_linear', 'policy::calculate_mb_values', 'policy::mb_mf_mixture', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'w_base'],
        "stai_modulation": "inverse_linear",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'w_base': [0, 1]},
        "bic": 457.0242580808181,
    },
    "p38": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'stai_modulation::stai_linear_additive', 'policy::add_perseveration_bonus', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'pers_base', 'stai_pers'],
        "stai_modulation": "stai_linear_additive",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'pers_base': [0, 5], 'stai_pers': [0, 5]},
        "bic": 0.0,
    },
    "p39": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus'],
        "parameters": ['alpha', 'beta', 'persev_w'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'persev_w': [0, 5]},
        "bic": None,
    },
    "p40": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'p_scale'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'p_scale': [0, 5]},
        "bic": 437.2013879032032,
    },
    "p41": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_perseveration_bonus', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'stick_factor'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'stick_factor': [0, 5]},
        "bic": 271.3039688021288,
    },
    "p42": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'value_update::update_q_td', 'stai_modulation::stai_linear_additive', 'decay::apply_anxiety_decay_p42'],
        "parameters": ['alpha', 'beta', 'decay_base', 'decay_stai'],
        "stai_modulation": "additive",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'decay_base': [0, 1], 'decay_stai': [0, 1]},
        "bic": 310.6393892059592,
    },
    "p43": {
        "class": "ParticipantModel1",
        "primitives": ['helper::softmax', 'stai_modulation::stai_inverse_linear', 'policy::calculate_mb_values', 'policy::mb_mf_mixture', 'value_update::update_q_td'],
        "parameters": ['alpha', 'beta', 'w_base'],
        "stai_modulation": "inverse_linear",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'w_base': [0, 1]},
        "bic": 523.5970917606509,
    },
    "p44": {
        "class": "ParticipantModel1",
        "primitives": ['helper::softmax', 'stai_modulation::stai_multiplicative', 'policy::add_habit_influence', 'value_update::update_q_td', 'value_update::update_habit_trace'],
        "parameters": ['alpha', 'beta', 'habit_weight'],
        "stai_modulation": "multiplicative",
        "bounds": {'alpha': [0, 1], 'beta': [0, 10], 'habit_weight': [0, 5]},
        "bic": 0.0,
    },
}

# -----------------------------------------------------------------------------
# CognitiveModelBase
# -----------------------------------------------------------------------------

class CognitiveModelBase(ABC):
    def __init__(self, n_trials: int, stai: float, model_parameters: tuple):
        # Task structure
        self.n_trials = n_trials
        self.n_choices = 2
        self.n_states = 2
        self.stai = stai
        
        # Transition matrix
        self.T = np.array([[0.7, 0.3], [0.3, 0.7]])
        
        # Choice probability sequences
        self.p_choice_1 = np.zeros(n_trials)
        self.p_choice_2 = np.zeros(n_trials)
        
        # Value representations
        self.q_stage1 = np.zeros(self.n_choices)
        self.q_stage2 = np.zeros((self.n_states, self.n_choices))
        
        # Trial tracking
        self.trial = 0
        self.last_action1 = None
        self.last_action2 = None
        self.last_state = None
        self.last_reward = None
        
        # Initialize
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
        # NOTE: Takes data arrays as arguments, NOT from self
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

# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------

def make_cognitive_model(ModelClass):
    def cognitive_model(action_1, state, action_2, reward, stai, model_parameters):
        n_trials = len(action_1)
        stai_val = float(stai[0]) if hasattr(stai, '__len__') else float(stai)
        model = ModelClass(n_trials, stai_val, model_parameters)
        return model.run_model(action_1, state, action_2, reward)
    return cognitive_model

# -----------------------------------------------------------------------------
# Model Reconstructor
# -----------------------------------------------------------------------------

def reconstruct_model(participant_id):
    spec = PARTICIPANT_SPECS[participant_id]
    
    class GeneratedModel(CognitiveModelBase):
        def unpack_parameters(self, model_parameters: tuple) -> None:
            for i, param_name in enumerate(spec['parameters']):
                setattr(self, param_name, model_parameters[i])

        # ----------------------------------------------------------------
        # GROUP 1: Standard Stickiness (Multiplicative)
        # ----------------------------------------------------------------
        if participant_id in ['p21', 'p26', 'p27', 'p30', 'p31', 'p32', 'p33', 'p34', 'p39', 'p40', 'p41']:
            def policy_stage1(self) -> np.ndarray:
                # Find stickiness param (not alpha or beta)
                params = [p for p in spec['parameters'] if p not in ['alpha', 'beta']]
                stick_param = getattr(self, params[0])
                
                # Bonus = param * stai
                bonus = stick_param * self.stai
                
                # Add perseveration bonus
                q_mod = P.add_perseveration_bonus(self.q_stage1, self.last_action1, bonus)
                return self.softmax(q_mod, self.beta)

        # ----------------------------------------------------------------
        # GROUP 2: Standard Stickiness (Additive)
        # ----------------------------------------------------------------
        elif participant_id in ['p19', 'p20', 'p22', 'p36', 'p38']:
            def init_model(self) -> None:
                # Identify base and slope parameters
                params = [p for p in spec['parameters'] if p not in ['alpha', 'beta']]
                p_base = next(p for p in params if 'base' in p)
                p_slope = next(p for p in params if p != p_base)
                
                base_val = getattr(self, p_base)
                slope_val = getattr(self, p_slope)
                
                self.stickiness = P.stai_linear_additive(base_val, slope_val, self.stai)

            def policy_stage1(self) -> np.ndarray:
                q_mod = P.add_perseveration_bonus(self.q_stage1, self.last_action1, self.stickiness)
                return self.softmax(q_mod, self.beta)
            
            # P19 and P20 also apply stickiness in Stage 2 based on state/context
            if participant_id in ['p19', 'p20']:
                def policy_stage2(self, state: int) -> np.ndarray:
                    q_mod = self.q_stage2[state].copy()
                    if self.last_state == state and self.last_action2 is not None:
                        q_mod[self.last_action2] += self.stickiness
                    return self.softmax(q_mod, self.beta)

        # ----------------------------------------------------------------
        # GROUP 3: Win-Stay (Multiplicative)
        # ----------------------------------------------------------------
        elif participant_id in ['p28', 'p35']:
            def policy_stage1(self) -> np.ndarray:
                params = [p for p in spec['parameters'] if p not in ['alpha', 'beta']]
                k = getattr(self, params[0])
                
                bonus = k * self.stai
                q_mod = P.add_win_stay_bonus(self.q_stage1, self.last_action1, self.last_reward, bonus)
                return self.softmax(q_mod, self.beta)

        # ----------------------------------------------------------------
        # GROUP 4: MB/MF Mixture (Inverse Linear)
        # ----------------------------------------------------------------
        elif participant_id in ['p24', 'p37', 'p43']:
            def init_model(self) -> None:
                # Initialize MB specific structures
                if participant_id != 'p24':
                    self.q_mf = np.zeros(self.n_choices)
                
                # Calculate mixing weight
                w_param = 'w_max' if 'w_max' in spec['parameters'] else 'w_base'
                w_val = getattr(self, w_param)
                self.w_eff = P.stai_inverse_linear(w_val, self.stai)
                # Clip implied by some models but primitive returns simple math.
                # Usually w is clipped in mixing.
                
            def policy_stage1(self) -> np.ndarray:
                # 1. Calc MB
                q_mb = P.calculate_mb_values(self.T, self.q_stage2)
                
                # 2. Mix
                if participant_id == 'p24':
                    # P24 mixes with q_stage1 (MF) and adds perseveration
                    q_net = P.mb_mf_mixture(q_mb, self.q_stage1, self.w_eff)
                    if self.last_action1 is not None:
                        q_net[self.last_action1] += self.perseveration
                else:
                    # P37/43 mix with separate q_mf
                    q_net = P.mb_mf_mixture(q_mb, self.q_mf, self.w_eff)
                
                return self.softmax(q_net, self.beta)

            # Special update rules for P37/P43
            if participant_id in ['p37', 'p43']:
                def value_update(self, action_1: int, state: int, action_2: int, reward: float) -> None:
                    # Update Stage 2
                    delta_2 = reward - self.q_stage2[state, action_2]
                    self.q_stage2[state, action_2] += self.alpha * delta_2
                    
                    # Update Stage 1 MF
                    if participant_id == 'p37':
                        # P37 Sample logic: Direct reward update
                        delta_1 = reward - self.q_mf[action_1]
                    else:
                        # P43 Sample logic: Stage 2 value update
                        delta_1 = self.q_stage2[state, action_2] - self.q_mf[action_1]
                        
                    self.q_mf[action_1] += self.alpha * delta_1

        # ----------------------------------------------------------------
        # SPECIAL: P18 (Inverse Division)
        # ----------------------------------------------------------------
        elif participant_id == 'p18':
            def init_model(self) -> None:
                self.phi_eff = P.stai_inverse_division(self.phi, self.stai)
            
            def policy_stage1(self) -> np.ndarray:
                q_mod = P.add_perseveration_bonus(self.q_stage1, self.last_action1, self.phi_eff)
                return self.softmax(q_mod, self.beta)

        # ----------------------------------------------------------------
        # SPECIAL: P23 (Learning Rate Bias)
        # ----------------------------------------------------------------
        elif participant_id == 'p23':
            def init_model(self) -> None:
                self.alpha_pos = self.alpha_base
                # alpha_neg = alpha_base * (1 + bias_factor * stai)
                self.alpha_neg = P.stai_scaling(self.alpha_base, self.bias_factor, self.stai)
                self.alpha_neg = np.clip(self.alpha_neg, 0.0, 1.0)
                
            def value_update(self, action_1: int, state: int, action_2: int, reward: float) -> None:
                alpha = P.select_learning_rate(self.alpha_pos, self.alpha_neg, reward)
                
                delta_2 = reward - self.q_stage2[state, action_2]
                self.q_stage2[state, action_2] += alpha * delta_2
                
                delta_1 = self.q_stage2[state, action_2] - self.q_stage1[action_1]
                self.q_stage1[action_1] += alpha * delta_1

        # ----------------------------------------------------------------
        # SPECIAL: P25 (Unitary Scaling)
        # ----------------------------------------------------------------
        elif participant_id == 'p25':
            def policy_stage1(self) -> np.ndarray:
                # k_eff = k * (1 + stai)
                k_eff = P.stai_scaling_unitary(self.k_stick, self.stai)
                q_mod = P.add_perseveration_bonus(self.q_stage1, self.last_action1, k_eff)
                return self.softmax(q_mod, self.beta)

        # ----------------------------------------------------------------
        # SPECIAL: P29 (Q-Learning Update)
        # ----------------------------------------------------------------
        elif participant_id == 'p29':
            def policy_stage1(self) -> np.ndarray:
                bonus = self.stickiness * self.stai
                q_mod = P.add_perseveration_bonus(self.q_stage1, self.last_action1, bonus)
                return self.softmax(q_mod, self.beta)

            def value_update(self, action_1: int, state: int, action_2: int, reward: float) -> None:
                # Stage 2
                delta_2 = reward - self.q_stage2[state, action_2]
                self.q_stage2[state, action_2] += self.alpha * delta_2
                
                # Stage 1 (Q-Learning: Target is Max of Stage 2)
                target = np.max(self.q_stage2[state])
                delta_1 = target - self.q_stage1[action_1]
                self.q_stage1[action_1] += self.alpha * delta_1

        # ----------------------------------------------------------------
        # SPECIAL: P42 (Anxiety Decay)
        # ----------------------------------------------------------------
        elif participant_id == 'p42':
            def init_model(self) -> None:
                self.decay_rate = P.stai_linear_additive(self.decay_base, self.decay_stai, self.stai)
                self.decay_rate = np.clip(self.decay_rate, 0.0, 1.0)

            def post_trial(self, action_1: int, state: int, action_2: int, reward: float) -> None:
                super().post_trial(action_1, state, action_2, reward)
                self.q_stage1, self.q_stage2 = P.apply_anxiety_decay_p42(
                    self.q_stage1, self.q_stage2, action_1, state, action_2, self.decay_rate
                )

        # ----------------------------------------------------------------
        # SPECIAL: P44 (Habit)
        # ----------------------------------------------------------------
        elif participant_id == 'p44':
            def init_model(self) -> None:
                self.habit = np.zeros(self.n_choices)
                
            def policy_stage1(self) -> np.ndarray:
                # Influence = habit_weight * stai * habit
                weight = self.habit_weight * self.stai
                net = P.add_habit_influence(self.q_stage1, self.habit, weight)
                return self.softmax(net, self.beta)
                
            def post_trial(self, action_1: int, state: int, action_2: int, reward: float) -> None:
                super().post_trial(action_1, state, action_2, reward)
                self.habit = P.update_habit_trace(self.habit, action_1, self.alpha)

    return GeneratedModel

def reconstruct_model_func(participant_id):
    return make_cognitive_model(reconstruct_model(participant_id))