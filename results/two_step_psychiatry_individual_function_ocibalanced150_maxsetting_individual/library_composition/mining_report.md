# Shared-fragment mining report

Fragments (normalized AST windows of 1-3 statements) shared by >=2 of 50 models,
ranked by participant coverage then size. Evidence base for cognitive_library.py primitives.

## 50/50 participants (3 stmt, 46 nodes, 50 occurrences)

```python
eps = 1e-10
log_loss = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
return log_loss
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (2 stmt, 43 nodes, 50 occurrences)

```python
eps = 1e-10
log_loss = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (2 stmt, 42 nodes, 50 occurrences)

```python
log_loss = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
return log_loss
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (1 stmt, 39 nodes, 50 occurrences)

```python
log_loss = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (3 stmt, 38 nodes, 50 occurrences)

```python
transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
p_choice_1 = np.zeros(n_trials)
p_choice_2 = np.zeros(n_trials)
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (3 stmt, 36 nodes, 50 occurrences)

```python
n_trials = len(action_1)
transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
p_choice_1 = np.zeros(n_trials)
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (2 stmt, 28 nodes, 50 occurrences)

```python
transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
p_choice_1 = np.zeros(n_trials)
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (2 stmt, 26 nodes, 50 occurrences)

```python
n_trials = len(action_1)
transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (2 stmt, 21 nodes, 50 occurrences)

```python
max_q_stage2 = np.max(q_stage2_mf, axis=1)
q_stage1_mb = transition_matrix @ max_q_stage2
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (2 stmt, 20 nodes, 50 occurrences)

```python
p_choice_1 = np.zeros(n_trials)
p_choice_2 = np.zeros(n_trials)
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (1 stmt, 18 nodes, 50 occurrences)

```python
transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (1 stmt, 12 nodes, 50 occurrences)

```python
max_q_stage2 = np.max(q_stage2_mf, axis=1)
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (1 stmt, 10 nodes, 100 occurrences)

```python
p_choice_1 = np.zeros(n_trials)
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (1 stmt, 9 nodes, 115 occurrences)

```python
a1 = action_1[trial]
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (1 stmt, 9 nodes, 50 occurrences)

```python
q_stage1_mb = transition_matrix @ max_q_stage2
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 50/50 participants (1 stmt, 8 nodes, 50 occurrences)

```python
n_trials = len(action_1)
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 49/50 participants (1 stmt, 14 nodes, 98 occurrences)

```python
probs_1 = exp_q1 / np.sum(exp_q1)
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

## 46/50 participants (3 stmt, 41 nodes, 46 occurrences)

```python
max_q_stage2 = np.max(q_stage2_mf, axis=1)
q_stage1_mb = transition_matrix @ max_q_stage2
q_net = w * q_stage1_mb + (1 - w) * q_stage1_mf
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49]

## 46/50 participants (2 stmt, 29 nodes, 46 occurrences)

```python
q_stage1_mb = transition_matrix @ max_q_stage2
q_net = w * q_stage1_mb + (1 - w) * q_stage1_mf
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49]

## 46/50 participants (1 stmt, 20 nodes, 46 occurrences)

```python
q_net = w * q_stage1_mb + (1 - w) * q_stage1_mf
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49]

## 44/50 participants (1 stmt, 9 nodes, 81 occurrences)

```python
rep_1 = np.zeros(2)
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 47, 49]

## 42/50 participants (1 stmt, 17 nodes, 83 occurrences)

```python
p_choice_1[trial] = probs_1[action_1[trial]]
```
participants: [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49]

## 41/50 participants (2 stmt, 31 nodes, 81 occurrences)

```python
probs_1 = exp_q1 / np.sum(exp_q1)
p_choice_1[trial] = probs_1[action_1[trial]]
```
participants: [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49]

## 40/50 participants (1 stmt, 29 nodes, 40 occurrences)

```python
delta_stage1 = q_stage2_mf[state_idx, action_2[trial]] - q_stage1_mf[action_1[trial]]
```
participants: [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49]

## 40/50 participants (1 stmt, 22 nodes, 40 occurrences)

```python
q_stage2_mf[state_idx, action_2[trial]] += alpha * delta_stage2
```
participants: [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49]

## 39/50 participants (2 stmt, 26 nodes, 40 occurrences)

```python
p_choice_1[trial] = probs_1[action_1[trial]]
state_idx = state[trial]
```
participants: [2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 46, 47, 48, 49]

## 39/50 participants (1 stmt, 25 nodes, 39 occurrences)

```python
delta_stage2 = reward[trial] - q_stage2_mf[state_idx, action_2[trial]]
```
participants: [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49]

## 39/50 participants (1 stmt, 6 nodes, 60 occurrences)

```python
last_action_1 = -1
```
participants: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 18, 19, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49]

## 38/50 participants (3 stmt, 40 nodes, 39 occurrences)

```python
probs_1 = exp_q1 / np.sum(exp_q1)
p_choice_1[trial] = probs_1[action_1[trial]]
state_idx = state[trial]
```
participants: [2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 43, 44, 46, 47, 48, 49]

## 37/50 participants (1 stmt, 18 nodes, 37 occurrences)

```python
q_stage1_mf[action_1[trial]] += alpha * delta_stage1
```
participants: [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48]

## 35/50 participants (3 stmt, 29 nodes, 35 occurrences)

```python
p_choice_1 = np.zeros(n_trials)
p_choice_2 = np.zeros(n_trials)
q_stage1_mf = np.zeros(2)
```
participants: [2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 39, 40, 41, 43, 44, 45, 47, 49]

## 35/50 participants (2 stmt, 19 nodes, 35 occurrences)

```python
p_choice_2 = np.zeros(n_trials)
q_stage1_mf = np.zeros(2)
```
participants: [2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 39, 40, 41, 43, 44, 45, 47, 49]

## 34/50 participants (2 stmt, 43 nodes, 34 occurrences)

```python
q_stage1_mf[action_1[trial]] += alpha * delta_stage1
delta_stage2 = reward[trial] - q_stage2_mf[state_idx, action_2[trial]]
```
participants: [1, 2, 5, 7, 8, 9, 11, 12, 13, 15, 17, 18, 21, 22, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48]

## 34/50 participants (3 stmt, 31 nodes, 34 occurrences)

```python
p_choice_2 = np.zeros(n_trials)
q_stage1_mf = np.zeros(2)
q_stage2_mf = np.zeros((2, 2))
```
participants: [2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 39, 40, 41, 44, 45, 47, 49]

## 34/50 participants (2 stmt, 21 nodes, 34 occurrences)

```python
q_stage1_mf = np.zeros(2)
q_stage2_mf = np.zeros((2, 2))
```
participants: [2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 39, 40, 41, 44, 45, 47, 49]

## 34/50 participants (1 stmt, 12 nodes, 34 occurrences)

```python
q_stage2_mf = np.zeros((2, 2))
```
participants: [2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 39, 40, 41, 44, 45, 47, 49]

## 33/50 participants (3 stmt, 60 nodes, 33 occurrences)

```python
probs_2 = exp_q2 / np.sum(exp_q2)
p_choice_2[trial] = probs_2[action_2[trial]]
delta_stage1 = q_stage2_mf[state_idx, action_2[trial]] - q_stage1_mf[action_1[trial]]
```
participants: [1, 2, 4, 5, 7, 8, 9, 11, 13, 14, 15, 16, 17, 21, 22, 23, 25, 26, 28, 29, 31, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49]

## 33/50 participants (2 stmt, 46 nodes, 33 occurrences)

```python
p_choice_2[trial] = probs_2[action_2[trial]]
delta_stage1 = q_stage2_mf[state_idx, action_2[trial]] - q_stage1_mf[action_1[trial]]
```
participants: [1, 2, 4, 5, 7, 8, 9, 11, 13, 14, 15, 16, 17, 21, 22, 23, 25, 26, 28, 29, 31, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49]

## 32/50 participants (3 stmt, 38 nodes, 32 occurrences)

```python
q_stage1_mb = transition_matrix @ max_q_stage2
q_net = w * q_stage1_mb + (1 - w) * q_stage1_mf
rep_1 = np.zeros(2)
```
participants: [0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 13, 14, 15, 16, 18, 19, 22, 24, 26, 27, 28, 29, 32, 36, 38, 39, 40, 43, 44, 45, 47, 49]

## 32/50 participants (2 stmt, 29 nodes, 32 occurrences)

```python
q_net = w * q_stage1_mb + (1 - w) * q_stage1_mf
rep_1 = np.zeros(2)
```
participants: [0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 13, 14, 15, 16, 18, 19, 22, 24, 26, 27, 28, 29, 32, 36, 38, 39, 40, 43, 44, 45, 47, 49]
