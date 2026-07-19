# Cognitive module library

Extracted by gemini-3.1-pro-preview (temperature 0, logged in llm_log/) from the individual best programs of participants [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49].

## MB/MF Mixture Weight (`mb_mf_mixture`)

Stage 1 values are a weighted combination of model-based and model-free Q-values.

- params: w [0.0, 1.0]
- provenance: participants [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
- excludes: ['separate_mb_mf_betas']

## Eligibility Trace (`eligibility_trace`)

Stage 1 model-free values are updated by the stage 2 reward prediction error scaled by an eligibility trace.

- params: lam [0.0, 1.0]
- provenance: participants [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 46, 47, 48, 49]
- excludes: none

## Stage 1 Choice Stickiness (`stage1_stickiness`)

A bonus is added to the stage 1 logits for repeating the previous trial's stage 1 action.

- params: stickiness_1 [-5.0, 5.0]
- provenance: participants [0, 1, 2, 3, 5, 8, 11, 12, 13, 14, 15, 16, 18, 19, 22, 24, 25, 27, 28, 29, 31, 32, 33, 34, 36, 38, 39, 40, 41, 42, 43, 44, 48, 49]
- excludes: ['reward_dependent_stickiness_stage1', 'transition_dependent_perseveration', 'shared_stickiness']

## Stage 2 Choice Stickiness (`stage2_stickiness`)

A bonus is added to the stage 2 logits for repeating the previous trial's stage 2 action in the same state.

- params: stickiness_2 [-5.0, 5.0]
- provenance: participants [0, 2, 3, 12, 15, 18, 32, 41, 42, 43, 47]
- excludes: ['shared_stickiness']

## Stage 2 Global Choice Perseveration (`stage2_global_stickiness`)

Adds a bonus to the stage-2 logits for repeating the most recent stage-2 action, regardless of the state.

- params: pers_2_global [-5.0, 5.0]
- provenance: participants [2, 44]
- excludes: none

## Value Decay to Zero (`value_decay_to_zero`)

Unchosen Q-values in both stages decay towards 0.

- params: decay_rate_zero [0.0, 1.0]
- provenance: participants [8, 10, 11, 13, 14, 16, 21, 23, 26, 27, 35, 37, 40, 41, 45]
- excludes: ['value_decay_to_half', 'value_decay_and_init_center', 'dynamic_value_decay']

## Value Decay to 0.5 (`value_decay_to_half`)

Unchosen Q-values in both stages decay towards 0.5.

- params: decay_rate_half [0.0, 1.0]
- provenance: participants [0, 1, 5, 12, 15, 19, 20, 38, 42, 43, 46, 48]
- excludes: ['value_decay_to_zero', 'value_decay_and_init_center', 'dynamic_value_decay']

## Value Decay and Init to Center (`value_decay_and_init_center`)

Initializes Q-values to a center parameter and decays unchosen values towards it.

- params: decay_rate_center [0.0, 1.0], center [0.0, 1.0]
- provenance: participants [4, 18, 36]
- excludes: ['value_decay_to_zero', 'value_decay_to_half', 'dynamic_value_decay', 'q_init_05']

## Stage 2 Value Decay to Zero (`stage2_value_decay_to_zero`)

Unchosen Q-values in stage 2 decay towards 0.

- params: decay_rate_s2_zero [0.0, 1.0]
- provenance: participants [31]
- excludes: ['value_decay_to_zero', 'value_decay_to_half', 'stage2_value_decay_to_half']

## Stage 2 Value Decay to 0.5 (`stage2_value_decay_to_half`)

Unchosen Q-values in stage 2 decay towards 0.5.

- params: decay_rate_s2_half [0.0, 1.0]
- provenance: participants [32]
- excludes: ['value_decay_to_zero', 'value_decay_to_half', 'stage2_value_decay_to_zero']

## Q-Value Initialization to 0.5 (`q_init_05`)

Model-free Q-values for both stages are initialized to 0.5.

- params: none
- provenance: participants [0, 12, 15, 17, 19, 20, 30, 42]
- excludes: ['value_decay_and_init_center']

## Asymmetric Learning Rates (`asymmetric_learning_rates`)

Separate learning rates are used for positive and negative reward prediction errors.

- params: alpha_pos [0.0, 1.0], alpha_neg [0.0, 1.0]
- provenance: participants [0, 5, 13, 31, 34, 36, 37, 45, 46, 47]
- excludes: ['separate_learning_rates', 'direct_mf_update', 'asymmetric_direct_mf_update']

## Separate Learning Rates (`separate_learning_rates`)

Uses distinct learning rates for updating Stage 1 and Stage 2 model-free values.

- params: alpha_1 [0.0, 1.0], alpha_2 [0.0, 1.0]
- provenance: participants [7, 22]
- excludes: ['asymmetric_learning_rates', 'direct_mf_update', 'asymmetric_direct_mf_update']

## Reward-Dependent Stickiness Stage 1 (`reward_dependent_stickiness_stage1`)

Adds a bonus or penalty to the previously chosen stage 1 action's logits depending on whether the previous trial was rewarded.

- params: p_win_1 [-5.0, 5.0], p_lose_1 [-5.0, 5.0]
- provenance: participants [4, 6, 7, 9, 10, 30, 44]
- excludes: ['stage1_stickiness', 'transition_dependent_perseveration', 'shared_stickiness']

## Reward-Dependent Global Stickiness Stage 2 (`reward_dependent_global_stickiness_stage2`)

Applies a bonus to the stage 2 action chosen on the previous trial, with the magnitude depending on whether the previous trial was rewarded.

- params: p_win_2_global [-5.0, 5.0], p_lose_2_global [-5.0, 5.0]
- provenance: participants [6]
- excludes: ['stage2_stickiness', 'reward_dependent_state_stickiness_stage2', 'shared_stickiness']

## Reward-Dependent State-Specific Stickiness Stage 2 (`reward_dependent_state_stickiness_stage2`)

Adds a bonus to the Stage 2 action value of the previously chosen action if the same state is visited, depending on reward.

- params: p_win_2_state [-5.0, 5.0], p_lose_2_state [-5.0, 5.0]
- provenance: participants [30]
- excludes: ['stage2_stickiness', 'reward_dependent_global_stickiness_stage2', 'shared_stickiness']

## Separate MB and MF Inverse Temperatures (`separate_mb_mf_betas`)

Uses distinct inverse temperature parameters for model-based and model-free values at stage 1.

- params: beta_mb [0.0, 10.0], beta_mf1 [0.0, 10.0]
- provenance: participants [10, 31]
- excludes: ['mb_mf_mixture']

## Separate Stage 2 Inverse Temperature (`separate_stage2_beta`)

Uses a distinct inverse temperature parameter for stage 2 choices.

- params: beta_2 [0.0, 10.0]
- provenance: participants [10, 20, 25, 34, 42, 49]
- excludes: none

## Motor Chunking Bias (`motor_chunking_bias`)

Adds a bias to the stage-2 logit corresponding to the same motor action chosen in stage 1 of the current trial.

- params: chunking_bias [-5.0, 5.0]
- provenance: participants [11, 35]
- excludes: none

## Direct Model-Free Update (`direct_mf_update`)

Stage 1 model-free values are updated directly by the final reward prediction error, bypassing the stage 2 value.

- params: none
- provenance: participants [13]
- excludes: ['asymmetric_learning_rates', 'separate_learning_rates', 'asymmetric_direct_mf_update']

## Asymmetric Direct MF Update (`asymmetric_direct_mf_update`)

Stage 1 model-free values are updated directly by the final reward prediction error, using asymmetric learning rates.

- params: alpha_pos_dir [0.0, 1.0], alpha_neg_dir [0.0, 1.0]
- provenance: participants [13]
- excludes: ['asymmetric_learning_rates', 'direct_mf_update', 'separate_learning_rates']

## Dynamic Value Decay (`dynamic_value_decay`)

Decays unchosen Q-values towards a dynamic target that tracks average reward.

- params: decay_rate_dyn [0.0, 1.0], lr_avg_reward [0.0, 1.0]
- provenance: participants [17]
- excludes: ['value_decay_to_zero', 'value_decay_to_half', 'value_decay_and_init_center']

## Counterfactual Updating (`counterfactual_updating`)

Updates the value of the unchosen stage-2 action in the visited state assuming it would have yielded the opposite reward.

- params: alpha_cf [0.0, 1.0]
- provenance: participants [23]
- excludes: none

## Learned Transition Matrix (`learned_transition_matrix`)

The agent dynamically updates its belief about the transition probabilities from experience.

- params: alpha_t [0.0, 1.0]
- provenance: participants [27]
- excludes: none

## Stage 1 Action Bias (`action_bias_stage1`)

Applies an intrinsic bias to the logit of choosing action 0 at stage 1.

- params: bias_1 [-5.0, 5.0]
- provenance: participants [34, 40, 43, 46]
- excludes: none

## Transition-Dependent Perseveration (`transition_dependent_perseveration`)

A perseveration bonus is applied to the previously chosen stage 1 action, with the magnitude depending on whether the previous trial's transition was common or rare.

- params: pi_common [-5.0, 5.0], pi_rare [-5.0, 5.0]
- provenance: participants [45, 47]
- excludes: ['stage1_stickiness', 'reward_dependent_stickiness_stage1', 'shared_stickiness']

## Shared Choice Stickiness (`shared_stickiness`)

Adds a bonus to the Q-value of the action chosen on the previous trial at the same stage, using a single parameter.

- params: rho [-5.0, 5.0]
- provenance: participants [26]
- excludes: ['stage1_stickiness', 'stage2_stickiness', 'reward_dependent_stickiness_stage1', 'reward_dependent_global_stickiness_stage2', 'reward_dependent_state_stickiness_stage2', 'transition_dependent_perseveration']
