# Cognitive module library

Extracted by gemini-3.1-pro-preview (temperature 0, logged in llm_log/) from the individual best programs of participants [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13].

## Independent MB/MF Betas (`independent_betas`)

Uses separate inverse temperature parameters to independently scale model-based and model-free values, replacing the traditional shared beta and mixing weight.

- params: beta_mf [0.0, 10.0]
- provenance: participants [1]
- excludes: ['mb_mf_mixture', 'outcome_dependent_temperature', 'pure_model_free', 'separate_stage_betas']

## Choice Stickiness (`choice_stickiness`)

Adds a scalar bonus to the stage 1 logit of the action chosen on the previous trial to model perseveration.

- params: stickiness [0.0, 5.0]
- provenance: participants [1, 2, 4, 6, 7, 8, 9, 10, 12]
- excludes: ['reward_dependent_stickiness', 'signed_choice_stickiness']

## Signed Choice Stickiness (`signed_choice_stickiness`)

Adds a signed scalar bonus to the stage 1 logit of the action chosen on the previous trial to model perseveration or alternation.

- params: perseveration [-3.0, 3.0]
- provenance: participants [13]
- excludes: ['choice_stickiness', 'reward_dependent_stickiness']

## Direct Stage 1 Reward Update (`direct_stage1_reward_update`)

Updates the stage 1 model-free Q-value directly using the final reward, equivalent to an eligibility trace (lambda) of 1.

- params: none
- provenance: participants [1, 2]
- excludes: ['eligibility_trace', 'stage2_first_update']

## Unchosen Value Decay (`unchosen_value_decay`)

The values of all unchosen actions in both Stage 1 and Stage 2 decay towards zero at a constant rate.

- params: decay_rate [0.0, 1.0]
- provenance: participants [2, 5, 6, 8, 9]
- excludes: ['unchosen_value_decay_center', 'stage2_value_decay_center', 'stage2_unchosen_value_decay']

## Neutral Q-value Initialization (`q_init_05`)

Stage 1 and Stage 2 model-free Q-values are initialized to a neutral value of 0.5 instead of 0.

- params: none
- provenance: participants [4, 13]
- excludes: none

## Stage 2 Value Decay to Center (`stage2_value_decay_center`)

Stage 2 Q-values decay towards a neutral center of 0.5 on every trial, representing forgetting or uncertainty.

- params: decay_rate_center [0.0, 1.0]
- provenance: participants [4]
- excludes: ['unchosen_value_decay', 'unchosen_value_decay_center', 'stage2_unchosen_value_decay']

## MB/MF Mixture (`mb_mf_mixture`)

Stage 1 action values are computed as a weighted mixture of model-based and model-free Q-values.

- params: w [0.0, 1.0]
- provenance: participants [4, 7, 8, 9, 12]
- excludes: ['independent_betas', 'pure_model_free']

## Counterfactual Updating (`counterfactual_updating`)

The Q-value of the unchosen Stage 2 action is updated using an assumed opposite reward.

- params: cf_weight [0.0, 1.0]
- provenance: participants [4, 7, 12]
- excludes: none

## Eligibility Trace (`eligibility_trace`)

Stage-1 action values are updated by the stage-2 reward prediction error scaled by an eligibility trace parameter.

- params: lambda_elig [0.0, 1.0]
- provenance: participants [5, 7, 8, 9, 10, 11, 12]
- excludes: ['direct_stage1_reward_update', 'stage2_first_update']

## Separate Stage Learning Rates (`separate_learning_rates`)

Uses distinct learning rates for updating stage-1 and stage-2 Q-values rather than a single global learning rate.

- params: alpha_2 [0.0, 1.0]
- provenance: participants [6]
- excludes: none

## Unchosen Value Decay to Center (`unchosen_value_decay_center`)

Decays the Q-values of unchosen Stage 1 actions and unvisited Stage 2 states towards the neutral value of 0.5.

- params: decay_rate_center_unchosen [0.0, 1.0]
- provenance: participants [7]
- excludes: ['unchosen_value_decay', 'stage2_value_decay_center', 'stage2_unchosen_value_decay']

## Reward-Dependent Stickiness (`reward_dependent_stickiness`)

Applies a stickiness bonus to the previously chosen actions in both stages, where the bonus magnitude depends on whether the previous trial resulted in a reward.

- params: stick_win [0.0, 5.0], stick_loss [0.0, 5.0]
- provenance: participants [11]
- excludes: ['choice_stickiness', 'signed_choice_stickiness']

## Outcome-Dependent Inverse Temperature (`outcome_dependent_temperature`)

The inverse temperature applied to the Q-values changes depending on whether the previous trial was rewarded or unrewarded.

- params: beta_loss [0.0, 10.0]
- provenance: participants [13]
- excludes: ['independent_betas', 'separate_stage_betas']

## Pure Model-Free (`pure_model_free`)

Uses only model-free Q-values for Stage 1 decisions, ignoring model-based calculations.

- params: none
- provenance: participants [2, 5, 10, 11, 13]
- excludes: ['mb_mf_mixture', 'independent_betas']

## Separate Stage Betas (`separate_stage_betas`)

Uses distinct inverse temperature parameters for Stage 1 and Stage 2.

- params: beta_2 [0.0, 10.0]
- provenance: participants [5, 11]
- excludes: ['outcome_dependent_temperature', 'independent_betas']

## Stage 2 Unchosen Value Decay (`stage2_unchosen_value_decay`)

Only the values of unchosen actions in Stage 2 decay towards zero.

- params: decay_rate_stage2 [0.0, 1.0]
- provenance: participants [10]
- excludes: ['unchosen_value_decay', 'unchosen_value_decay_center', 'stage2_value_decay_center']

## Stage 2 First Update (`stage2_first_update`)

Updates the Stage 2 Q-value before computing the Stage 1 TD error, so the Stage 1 update uses the newly updated Stage 2 value.

- params: none
- provenance: participants [13]
- excludes: ['direct_stage1_reward_update', 'eligibility_trace']
