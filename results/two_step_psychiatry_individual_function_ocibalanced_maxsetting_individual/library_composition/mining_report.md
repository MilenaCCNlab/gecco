# Shared-fragment mining report

Fragments (normalized AST windows of 1-3 statements) shared by >=2 of 12 models,
ranked by participant coverage then size. Evidence base for cognitive_library.py primitives.

## 12/12 participants (1 stmt, 14 nodes, 24 occurrences)

```python
probs_1 = exp_q1 / np.sum(exp_q1)
```
participants: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

## 12/12 participants (1 stmt, 9 nodes, 40 occurrences)

```python
a1 = action_1[trial]
```
participants: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

## 12/12 participants (1 stmt, 8 nodes, 12 occurrences)

```python
n_trials = len(action_1)
```
participants: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

## 10/12 participants (1 stmt, 10 nodes, 10 occurrences)

```python
logits_stage1[last_action_1] += stickiness
```
participants: [1, 2, 4, 6, 8, 9, 10, 11, 12, 13]

## 10/12 participants (1 stmt, 9 nodes, 10 occurrences)

```python
q_mf_stage1 = np.zeros(2)
```
participants: [1, 2, 4, 5, 6, 8, 9, 10, 11, 12]

## 10/12 participants (1 stmt, 6 nodes, 11 occurrences)

```python
last_action_1 = -1
```
participants: [1, 2, 4, 6, 8, 9, 10, 11, 12, 13]

## 9/12 participants (2 stmt, 31 nodes, 16 occurrences)

```python
probs_1 = exp_q1 / np.sum(exp_q1)
p_choice_1[trial] = probs_1[action_1[trial]]
```
participants: [4, 5, 6, 7, 8, 9, 10, 11, 13]

## 9/12 participants (2 stmt, 21 nodes, 9 occurrences)

```python
q_mf_stage1 = np.zeros(2)
q_mf_stage2 = np.zeros((2, 2))
```
participants: [1, 2, 5, 6, 8, 9, 10, 11, 12]

## 9/12 participants (2 stmt, 20 nodes, 9 occurrences)

```python
p_choice_1 = np.zeros(n_trials)
p_choice_2 = np.zeros(n_trials)
```
participants: [4, 5, 6, 7, 8, 9, 10, 11, 13]

## 9/12 participants (1 stmt, 18 nodes, 9 occurrences)

```python
if last_action_1 != -1:
    logits_stage1[last_action_1] += stickiness
```
participants: [1, 2, 4, 6, 8, 9, 10, 12, 13]

## 9/12 participants (1 stmt, 17 nodes, 16 occurrences)

```python
p_choice_1[trial] = probs_1[action_1[trial]]
```
participants: [4, 5, 6, 7, 8, 9, 10, 11, 13]

## 9/12 participants (1 stmt, 12 nodes, 9 occurrences)

```python
q_mf_stage2 = np.zeros((2, 2))
```
participants: [1, 2, 5, 6, 8, 9, 10, 11, 12]

## 9/12 participants (1 stmt, 10 nodes, 18 occurrences)

```python
p_choice_1 = np.zeros(n_trials)
```
participants: [4, 5, 6, 7, 8, 9, 10, 11, 13]

## 8/12 participants (2 stmt, 18 nodes, 17 occurrences)

```python
a1 = action_1[trial]
s_idx = state[trial]
```
participants: [1, 2, 4, 5, 6, 8, 11, 13]

## 6/12 participants (3 stmt, 46 nodes, 6 occurrences)

```python
eps = 1e-10
log_loss = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
return log_loss
```
participants: [4, 6, 7, 8, 11, 13]

## 6/12 participants (2 stmt, 43 nodes, 6 occurrences)

```python
eps = 1e-10
log_loss = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
```
participants: [4, 6, 7, 8, 11, 13]

## 6/12 participants (2 stmt, 42 nodes, 6 occurrences)

```python
log_loss = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
return log_loss
```
participants: [4, 6, 7, 8, 11, 13]

## 6/12 participants (1 stmt, 39 nodes, 6 occurrences)

```python
log_loss = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
```
participants: [4, 6, 7, 8, 11, 13]

## 6/12 participants (3 stmt, 29 nodes, 6 occurrences)

```python
p_choice_1 = np.zeros(n_trials)
p_choice_2 = np.zeros(n_trials)
q_stage1_mf = np.zeros(2)
```
participants: [4, 5, 6, 9, 10, 11]

## 6/12 participants (3 stmt, 27 nodes, 6 occurrences)

```python
q_mf_stage1 = np.zeros(2)
q_mf_stage2 = np.zeros((2, 2))
last_action_1 = -1
```
participants: [1, 6, 9, 10, 11, 12]

## 6/12 participants (2 stmt, 26 nodes, 6 occurrences)

```python
n_trials = len(action_1)
transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
```
participants: [1, 4, 7, 8, 9, 12]

## 6/12 participants (2 stmt, 21 nodes, 6 occurrences)

```python
max_q_stage2 = np.max(q_mf_stage2, axis=1)
q_mb_stage1 = transition_matrix @ max_q_stage2
```
participants: [1, 4, 7, 8, 9, 12]

## 6/12 participants (2 stmt, 19 nodes, 6 occurrences)

```python
p_choice_2 = np.zeros(n_trials)
q_stage1_mf = np.zeros(2)
```
participants: [4, 5, 6, 9, 10, 11]

## 6/12 participants (1 stmt, 18 nodes, 6 occurrences)

```python
transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
```
participants: [1, 4, 7, 8, 9, 12]

## 6/12 participants (2 stmt, 18 nodes, 6 occurrences)

```python
q_mf_stage2 = np.zeros((2, 2))
last_action_1 = -1
```
participants: [1, 6, 9, 10, 11, 12]

## 6/12 participants (1 stmt, 17 nodes, 6 occurrences)

```python
delta_2 = r - q_mf_stage2[s_idx, a2]
```
participants: [1, 4, 5, 7, 12, 13]

## 6/12 participants (1 stmt, 12 nodes, 6 occurrences)

```python
max_q_stage2 = np.max(q_mf_stage2, axis=1)
```
participants: [1, 4, 7, 8, 9, 12]

## 6/12 participants (1 stmt, 9 nodes, 6 occurrences)

```python
q_mb_stage1 = transition_matrix @ max_q_stage2
```
participants: [1, 4, 7, 8, 9, 12]

## 5/12 participants (3 stmt, 60 nodes, 5 occurrences)

```python
probs_2 = exp_q2 / np.sum(exp_q2)
p_choice_2[trial] = probs_2[action_2[trial]]
delta_stage1 = q_stage2_mf[state_idx, action_2[trial]] - q_stage1_mf[action_1[trial]]
```
participants: [6, 7, 9, 10, 11]

## 5/12 participants (2 stmt, 46 nodes, 5 occurrences)

```python
p_choice_2[trial] = probs_2[action_2[trial]]
delta_stage1 = q_stage2_mf[state_idx, action_2[trial]] - q_stage1_mf[action_1[trial]]
```
participants: [6, 7, 9, 10, 11]

## 5/12 participants (3 stmt, 41 nodes, 5 occurrences)

```python
max_q_stage2 = np.max(q_stage2_mf, axis=1)
q_stage1_mb = transition_matrix @ max_q_stage2
q_net_1 = w * q_stage1_mb + (1 - w) * q_stage1_mf
```
participants: [4, 7, 8, 9, 12]

## 5/12 participants (3 stmt, 40 nodes, 5 occurrences)

```python
probs_1 = exp_q1 / np.sum(exp_q1)
p_choice_1[trial] = probs_1[action_1[trial]]
state_idx = state[trial]
```
participants: [5, 7, 8, 10, 11]

## 5/12 participants (3 stmt, 31 nodes, 5 occurrences)

```python
p_choice_2 = np.zeros(n_trials)
q_stage1_mf = np.zeros(2)
q_stage2_mf = np.zeros((2, 2))
```
participants: [5, 6, 9, 10, 11]

## 5/12 participants (2 stmt, 29 nodes, 5 occurrences)

```python
q_stage1_mb = transition_matrix @ max_q_stage2
q_net_1 = w * q_stage1_mb + (1 - w) * q_stage1_mf
```
participants: [4, 7, 8, 9, 12]

## 5/12 participants (1 stmt, 29 nodes, 5 occurrences)

```python
delta_stage1 = q_stage2_mf[state_idx, action_2[trial]] - q_stage1_mf[action_1[trial]]
```
participants: [6, 7, 9, 10, 11]

## 5/12 participants (2 stmt, 28 nodes, 5 occurrences)

```python
exp_q1 = np.exp(beta * q_eff_1)
probs_1 = exp_q1 / np.sum(exp_q1)
```
participants: [2, 5, 8, 9, 10]

## 5/12 participants (2 stmt, 26 nodes, 5 occurrences)

```python
p_choice_1[trial] = probs_1[action_1[trial]]
state_idx = state[trial]
```
participants: [5, 7, 8, 10, 11]

## 5/12 participants (1 stmt, 25 nodes, 5 occurrences)

```python
delta_stage2 = reward[trial] - q_stage2_mf[state_idx, action_2[trial]]
```
participants: [6, 7, 9, 10, 11]

## 5/12 participants (2 stmt, 25 nodes, 5 occurrences)

```python
(alpha_1, alpha_2, decay_rate, beta_1, beta_2, stickiness) = model_parameters
n_trials = len(action_1)
```
participants: [6, 8, 9, 11, 12]

## 5/12 participants (1 stmt, 22 nodes, 5 occurrences)

```python
q_stage2_mf[state_idx, action_2[trial]] += alpha_2 * delta_stage2
```
participants: [6, 7, 9, 10, 11]
