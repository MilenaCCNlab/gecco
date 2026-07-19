# Cognitive module library

Extracted by gemini-3.1-pro-preview (temperature 0, logged in llm_log/) from the individual best programs of participants [1, 2, 3, 10, 11, 12, 13, 14].

## Eligibility Trace (`eligibility_trace`)

Maintains a decaying eligibility trace to update previously visited state-action pairs in the RL system.

- params: eligibility_trace_lambda_trace [0.0, 1.0]
- provenance: participants [1]
- excludes: ['asymmetric_rl_learning_rates']

## Action Stickiness (`action_stickiness`)

Adds a bias to the RL value of the previously chosen action to model choice perseveration.

- params: action_stickiness_kappa [-10.0, 10.0]
- provenance: participants [1]
- excludes: ['choice_perseveration']

## Load-Driven WM Noise (`load_driven_wm_noise`)

Corrupts the working memory policy by blending it with a uniform distribution, with the noise level increasing exponentially with set size.

- params: load_driven_wm_noise_wm_noise [0.0, 10.0]
- provenance: participants [1]
- excludes: ['wm_set_size_retrieval_noise', 'set_size_wm_noise', 'set_size_dependent_wm_lapse']

## Asymmetric WM Update (P1) (`wm_asymmetric_update_p1`)

Working memory perfectly encodes the chosen action upon reward, but relaxes toward a uniform distribution upon non-reward.

- params: wm_asymmetric_update_p1_decay [0.0, 1.0]
- provenance: participants [1]
- excludes: ['wm_asymmetric_update_p2', 'unified_wm_update_decay', 'wm_perfect_encoding_on_reward', 'asymmetric_wm_update', 'load_dependent_wm_learning', 'asymmetric_fixed_wm_update', 'arbitration_scaled_wm_update']

## Asymmetric RL Learning Rates (`asymmetric_rl_learning_rates`)

RL uses separate learning rates for positive and negative prediction errors.

- params: asymmetric_rl_learning_rates_lr_pos [0.0, 1.0], asymmetric_rl_learning_rates_lr_neg [0.0, 1.0]
- provenance: participants [2, 10, 13]
- excludes: ['eligibility_trace']

## Set-Size Dependent WM Retrieval Noise (`wm_set_size_retrieval_noise`)

WM retrieval is corrupted by noise that increases as a sigmoid function of set size, pulling retrieved values toward uniform.

- params: wm_set_size_retrieval_noise_noise_gain [-10.0, 10.0]
- provenance: participants [2]
- excludes: ['load_driven_wm_noise', 'set_size_wm_noise', 'set_size_dependent_wm_lapse']

## Asymmetric WM Update (P2) (`wm_asymmetric_update_p2`)

WM values are updated toward the chosen action upon reward, and decay toward uniform upon non-reward.

- params: wm_asymmetric_update_p2_lr_pos [0.0, 1.0], wm_asymmetric_update_p2_lr_neg [0.0, 1.0]
- provenance: participants [2]
- excludes: ['wm_asymmetric_update_p1', 'unified_wm_update_decay', 'wm_perfect_encoding_on_reward', 'asymmetric_wm_update', 'load_dependent_wm_learning', 'asymmetric_fixed_wm_update', 'arbitration_scaled_wm_update']

## Outcome-Gated WM Reliance (`outcome_gated_wm_reliance`)

A latent gate tracks recent reward history with asymmetric win/loss sensitivities and modulates the baseline WM weight via a logistic transform.

- params: outcome_gated_wm_reliance_win_boost [0.0, 10.0], outcome_gated_wm_reliance_loss_suppression [0.0, 10.0], outcome_gated_wm_reliance_kappa [0.0, 1.0]
- provenance: participants [3]
- excludes: ['load_scaled_wm_weight', 'load_dependent_wm_arbitration']

## Unified WM Update and Decay (`unified_wm_update_decay`)

Working memory for the current state updates toward a one-hot action vector on reward, or decays toward uniform on no-reward, using a single shared rate parameter.

- params: unified_wm_update_decay_kappa [0.0, 1.0]
- provenance: participants [3]
- excludes: ['wm_asymmetric_update_p1', 'wm_asymmetric_update_p2', 'wm_perfect_encoding_on_reward', 'asymmetric_wm_update', 'load_dependent_wm_learning', 'asymmetric_fixed_wm_update', 'arbitration_scaled_wm_update']

## Set-Size Dependent WM Decay (`set_size_wm_decay`)

Working memory values decay toward a uniform distribution at a rate that exponentially approaches 1 as set size increases.

- params: set_size_wm_decay_gamma [0.0, 10.0]
- provenance: participants [10]
- excludes: ['load_dependent_wm_decay_global']

## Set-Size Dependent WM Noise (`set_size_wm_noise`)

The working memory policy becomes noisier (lower inverse temperature) as the set-size dependent interference increases.

- params: set_size_wm_noise_base_temp [1.0, 100.0], set_size_wm_noise_gamma [0.0, 10.0]
- provenance: participants [10]
- excludes: ['load_driven_wm_noise', 'wm_set_size_retrieval_noise', 'set_size_dependent_wm_lapse']

## Perfect WM Encoding on Reward (`wm_perfect_encoding_on_reward`)

Working memory perfectly encodes the chosen action as a sharp distribution when a reward is received, overriding previous values.

- params: none
- provenance: participants [10]
- excludes: ['wm_asymmetric_update_p1', 'wm_asymmetric_update_p2', 'unified_wm_update_decay', 'asymmetric_wm_update', 'load_dependent_wm_learning', 'asymmetric_fixed_wm_update', 'arbitration_scaled_wm_update']

## Load-Scaled WM Weight (`load_scaled_wm_weight`)

The baseline working memory mixture weight is scaled down by a power law of the set size relative to a baseline set size of 3.

- params: load_scaled_wm_weight_load_beta [0.0, 10.0]
- provenance: participants [11]
- excludes: ['outcome_gated_wm_reliance', 'load_dependent_wm_arbitration']

## Asymmetric WM Update (`asymmetric_wm_update`)

Working memory weights are updated differently for rewarded versus unrewarded trials, with errors causing a reduction in the chosen action's weight that is evenly distributed to unchosen actions.

- params: asymmetric_wm_update_wm_alpha_pos [0.0, 1.0], asymmetric_wm_update_wm_alpha_neg [0.0, 1.0]
- provenance: participants [11]
- excludes: ['wm_asymmetric_update_p1', 'wm_asymmetric_update_p2', 'unified_wm_update_decay', 'wm_perfect_encoding_on_reward', 'load_dependent_wm_learning', 'asymmetric_fixed_wm_update', 'arbitration_scaled_wm_update']

## WM Normalization (`wm_normalization`)

Working memory weights are explicitly normalized to sum to 1 after each update, falling back to a uniform distribution if the sum is zero.

- params: none
- provenance: participants [11]
- excludes: none

## Choice perseveration (`choice_perseveration`)

Adds an additive bias to the RL Q-value of the last action taken in the current state.

- params: choice_perseveration_stay_bias [-10.0, 10.0]
- provenance: participants [12]
- excludes: ['action_stickiness']

## Load-dependent WM learning (`load_dependent_wm_learning`)

Working memory updates its values via a delta rule toward a target distribution that incorporates load-dependent confusion on reward, and forgets toward uniform on error.

- params: load_dependent_wm_learning_wm_lr_pos [0.0, 1.0], load_dependent_wm_learning_wm_lr_neg [0.0, 1.0], load_dependent_wm_learning_confusion_rate [0.0, 1.0]
- provenance: participants [12]
- excludes: ['wm_asymmetric_update_p1', 'wm_asymmetric_update_p2', 'unified_wm_update_decay', 'wm_perfect_encoding_on_reward', 'asymmetric_wm_update', 'asymmetric_fixed_wm_update', 'arbitration_scaled_wm_update']

## Set-Size-Dependent WM Lapse (`set_size_dependent_wm_lapse`)

Working memory reliability decreases at higher set sizes, modeled as a lapse rate that scales linearly from 0 at set size 3 to a maximum at set size 6, blending the WM policy with a uniform distribution. Also includes arbitration between RL and WM based on the magnitude of the RL prediction error.

- params: set_size_dependent_wm_lapse_lapse0 [0.0, 1.0]
- provenance: participants [13]
- excludes: ['load_driven_wm_noise', 'wm_set_size_retrieval_noise', 'set_size_wm_noise']

## Asymmetric Fixed-Rate WM Updating (`asymmetric_fixed_wm_update`)

Working memory updates perfectly after positive feedback but only partially decays the chosen action's weight and renormalizes after negative feedback, using fixed learning rates.

- params: none
- provenance: participants [13]
- excludes: ['wm_asymmetric_update_p1', 'wm_asymmetric_update_p2', 'unified_wm_update_decay', 'wm_perfect_encoding_on_reward', 'asymmetric_wm_update', 'load_dependent_wm_learning', 'arbitration_scaled_wm_update']

## Load-Dependent WM Decay (`load_dependent_wm_decay_global`)

Working memory weights decay toward a uniform distribution on every trial, with the decay rate increasing as a function of set size for set sizes greater than 3.

- params: load_dependent_wm_decay_global_load_scale [0.0, 10.0]
- provenance: participants [14]
- excludes: ['set_size_wm_decay']

## Load-Dependent WM Arbitration (`load_dependent_wm_arbitration`)

The probability of relying on working memory increases with reward prediction error (surprise) and decreases as the set size (load) increases beyond a threshold of 3.

- params: load_dependent_wm_arbitration_wm_base [-10.0, 10.0], load_dependent_wm_arbitration_surprise_gain [0.0, 10.0], load_dependent_wm_arbitration_load_scale [0.0, 10.0]
- provenance: participants [14]
- excludes: ['outcome_gated_wm_reliance', 'load_scaled_wm_weight']

## Arbitration-Scaled WM Encoding (`arbitration_scaled_wm_update`)

The learning rate for working memory on rewarded trials is scaled by the current arbitration weight for working memory, and diffuses toward uniform on negative feedback.

- params: arbitration_scaled_wm_update_wm_learn [0.0, 1.0]
- provenance: participants [14]
- excludes: ['wm_asymmetric_update_p1', 'wm_asymmetric_update_p2', 'unified_wm_update_decay', 'wm_perfect_encoding_on_reward', 'asymmetric_wm_update', 'load_dependent_wm_learning', 'asymmetric_fixed_wm_update']
