"""
Participant Model Specifications
=================================

Compressed representation of each participant's cognitive model.
Each spec defines:
  - Which primitives are used
  - Parameter names and their roles
  - STAI modulation pattern

Models can be reconstructed using primitives + this specification.
"""

PARTICIPANT_SPECS = {
    "p18": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'modulation::inverse_division', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'phi'],
        "stai_modulation": "inverse_division",
        "bic": 305.38,
    },
    "p19": {
        "class": "ParticipantModel1",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'stick_base', 'stick_stai_slope'],
        "stai_modulation": "multiplicative",
        "bic": 216.06,
    },
    "p20": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'stickiness_base', 'anxiety_stick'],
        "stai_modulation": "multiplicative",
        "bic": 381.98,
    },
    "p21": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'phi'],
        "stai_modulation": "multiplicative",
        "bic": 341.38,
    },
    "p22": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'stick_base', 'stick_stai'],
        "stai_modulation": "multiplicative",
        "bic": 331.70,
    },
    "p23": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'value_update::asymmetric_td'],
        "parameters": ['alpha_base', 'beta', 'bias_factor'],
        "stai_modulation": "multiplicative",
        "bic": 353.60,
    },
    "p24": {
        "class": "ParticipantModel1",
        "primitives": ['helper::compute_mb_values', 'helper::softmax', 'modulation::inverse_linear', 'policy::mb_mf_mixture', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'w_max', 'perseveration'],
        "stai_modulation": "inverse_linear",
        "bic": 444.80,
    },
    "p25": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'modulation::custom', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'k_stick'],
        "stai_modulation": "custom",
        "bic": 119.81,
    },
    "p26": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'stickiness_factor'],
        "stai_modulation": "multiplicative",
        "bic": 242.96,
    },
    "p27": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'pers_k'],
        "stai_modulation": "multiplicative",
        "bic": 396.36,
    },
    "p28": {
        "class": "ParticipantModel1",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'policy::win_stay_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'win_bonus'],
        "stai_modulation": "multiplicative",
        "bic": 361.43,
    },
    "p29": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'stickiness'],
        "stai_modulation": "multiplicative",
        "bic": 256.55,
    },
    "p30": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'stick_sensitivity'],
        "stai_modulation": "multiplicative",
        "bic": 477.68,
    },
    "p31": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'k_anx'],
        "stai_modulation": "multiplicative",
        "bic": 373.65,
    },
    "p32": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'k'],
        "stai_modulation": "multiplicative",
        "bic": 325.98,
    },
    "p33": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'stickiness'],
        "stai_modulation": "multiplicative",
        "bic": 386.38,
    },
    "p34": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'rho'],
        "stai_modulation": "multiplicative",
        "bic": 373.86,
    },
    "p35": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::win_stay_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'cling_factor'],
        "stai_modulation": "multiplicative",
        "bic": 283.02,
    },
    "p36": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'p_base', 'p_slope'],
        "stai_modulation": "multiplicative",
        "bic": 458.63,
    },
    "p37": {
        "class": "ParticipantModel1",
        "primitives": ['helper::compute_mb_values', 'helper::softmax', 'modulation::inverse_linear', 'policy::mb_mf_mixture', 'value_update::separate_mf_td1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'w_base'],
        "stai_modulation": "inverse_linear",
        "bic": 457.02,
    },
    "p38": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'pers_base', 'stai_pers'],
        "stai_modulation": "multiplicative",
        "bic": 384.99,
    },
    "p39": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'persev_w'],
        "stai_modulation": "multiplicative",
        "bic": 327.36,
    },
    "p40": {
        "class": "ParticipantModel3",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'policy::win_stay_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'p_scale'],
        "stai_modulation": "multiplicative",
        "bic": 437.20,
    },
    "p41": {
        "class": "ParticipantModel2",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'stick_factor'],
        "stai_modulation": "multiplicative",
        "bic": 271.30,
    },
    "p42": {
        "class": "ParticipantModel2",
        "primitives": ['decay::memory_decay', 'helper::softmax', 'modulation::multiplicative', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'decay_base', 'decay_stai'],
        "stai_modulation": "multiplicative",
        "bic": 310.64,
    },
    "p43": {
        "class": "ParticipantModel1",
        "primitives": ['helper::compute_mb_values', 'helper::softmax', 'modulation::inverse_linear', 'policy::mb_mf_mixture', 'value_update::td_stage1', 'value_update::td_stage2'],
        "parameters": ['alpha', 'beta', 'w_base'],
        "stai_modulation": "inverse_linear",
        "bic": 523.60,
    },
    "p44": {
        "class": "ParticipantModel1",
        "primitives": ['helper::softmax', 'modulation::multiplicative', 'value_update::td_stage1', 'value_update::td_stage2', 'habit::habit_trace'],
        "parameters": ['alpha', 'beta', 'habit_weight'],
        "stai_modulation": "multiplicative",
        "bic": 402.74,
    },
}

# ============================================================
# SUMMARY STATISTICS
# ============================================================

TOTAL_PARTICIPANTS = 27

PRIMITIVE_USAGE = {
    "helper::softmax": 27,  # 100%
    "value_update::td_stage1": 26,  # 96%
    "value_update::td_stage2": 26,  # 96%
    "policy::perseveration_bonus": 22,  # 81%
    "modulation::multiplicative": 22,  # 81%
    "helper::compute_mb_values": 3,  # 11%
    "modulation::inverse_linear": 3,  # 11%
    "policy::mb_mf_mixture": 3,  # 11%
    "policy::win_stay_bonus": 3,  # 11%
    "decay::memory_decay": 2,  # 7%
    "modulation::inverse_division": 1,  # 4%
    "value_update::asymmetric_td": 1,  # 4%
    "modulation::custom": 1,  # 4%
    "decay::eligibility_trace": 1,  # 4%
}