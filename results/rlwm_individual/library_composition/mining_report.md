# Shared-fragment mining report

Fragments (normalized AST windows of 1-3 statements) shared by >=2 of 8 models,
ranked by participant coverage then size. Evidence base for cognitive_library.py primitives.

## 8/8 participants (2 stmt, 15 nodes, 8 occurrences)

```python
nA = 3
nS = int(block_set_sizes[0])
```
participants: [1, 2, 3, 10, 11, 12, 13, 14]

## 8/8 participants (1 stmt, 11 nodes, 8 occurrences)

```python
nS = int(block_set_sizes[0])
```
participants: [1, 2, 3, 10, 11, 12, 13, 14]

## 8/8 participants (1 stmt, 6 nodes, 8 occurrences)

```python
blocks_log_p += log_p
```
participants: [1, 2, 3, 10, 11, 12, 13, 14]

## 8/8 participants (1 stmt, 5 nodes, 8 occurrences)

```python
return -blocks_log_p
```
participants: [1, 2, 3, 10, 11, 12, 13, 14]

## 7/8 participants (3 stmt, 36 nodes, 7 occurrences)

```python
a = int(block_actions[t])
s = int(block_states[t])
r = float(block_rewards[t])
```
participants: [1, 2, 3, 10, 11, 12, 13]

## 7/8 participants (2 stmt, 24 nodes, 7 occurrences)

```python
s = int(block_states[t])
r = float(block_rewards[t])
```
participants: [1, 2, 3, 10, 11, 12, 13]

## 7/8 participants (2 stmt, 24 nodes, 7 occurrences)

```python
a = int(block_actions[t])
s = int(block_states[t])
```
participants: [1, 2, 3, 10, 11, 12, 13]

## 7/8 participants (1 stmt, 12 nodes, 7 occurrences)

```python
r = float(block_rewards[t])
```
participants: [1, 2, 3, 10, 11, 12, 13]

## 7/8 participants (1 stmt, 12 nodes, 14 occurrences)

```python
a = int(block_actions[t])
```
participants: [1, 2, 3, 10, 11, 12, 13]

## 7/8 participants (1 stmt, 12 nodes, 12 occurrences)

```python
W_s = w[s, :]
```
participants: [1, 2, 3, 10, 11, 12, 13]

## 7/8 participants (1 stmt, 9 nodes, 26 occurrences)

```python
block_actions = actions[block_mask]
```
participants: [1, 2, 3, 10, 11, 13, 14]

## 6/8 participants (3 stmt, 63 nodes, 6 occurrences)

```python
q = 1.0 / nA * np.ones((nS, nA))
w = 1.0 / nA * np.ones((nS, nA))
w_0 = 1.0 / nA * np.ones((nS, nA))
```
participants: [1, 2, 10, 11, 12, 13]

## 6/8 participants (3 stmt, 53 nodes, 6 occurrences)

```python
nS = int(block_set_sizes[0])
q = 1.0 / nA * np.ones((nS, nA))
w = 1.0 / nA * np.ones((nS, nA))
```
participants: [1, 2, 10, 11, 12, 13]

## 6/8 participants (2 stmt, 42 nodes, 12 occurrences)

```python
q = 1.0 / nA * np.ones((nS, nA))
w = 1.0 / nA * np.ones((nS, nA))
```
participants: [1, 2, 10, 11, 12, 13]

## 6/8 participants (3 stmt, 36 nodes, 6 occurrences)

```python
nA = 3
nS = int(block_set_sizes[0])
q = 1.0 / nA * np.ones((nS, nA))
```
participants: [1, 2, 10, 11, 12, 13]

## 6/8 participants (2 stmt, 32 nodes, 6 occurrences)

```python
nS = int(block_set_sizes[0])
q = 1.0 / nA * np.ones((nS, nA))
```
participants: [1, 2, 10, 11, 12, 13]

## 6/8 participants (3 stmt, 27 nodes, 11 occurrences)

```python
block_actions = actions[block_mask]
block_rewards = rewards[block_mask]
block_states = states[block_mask]
```
participants: [1, 2, 10, 11, 13, 14]

## 6/8 participants (1 stmt, 21 nodes, 18 occurrences)

```python
q = 1.0 / nA * np.ones((nS, nA))
```
participants: [1, 2, 10, 11, 12, 13]

## 6/8 participants (1 stmt, 20 nodes, 6 occurrences)

```python
p_total = wm_weight * p_wm + (1.0 - wm_weight) * p_rl
```
participants: [2, 10, 11, 12, 13, 14]

## 6/8 participants (2 stmt, 18 nodes, 17 occurrences)

```python
block_actions = actions[block_mask]
block_rewards = rewards[block_mask]
```
participants: [1, 2, 10, 11, 13, 14]

## 6/8 participants (1 stmt, 11 nodes, 6 occurrences)

```python
log_p += np.log(p_total)
```
participants: [1, 2, 3, 10, 11, 13]

## 6/8 participants (1 stmt, 9 nodes, 6 occurrences)

```python
block_mask = blocks == b
```
participants: [1, 2, 10, 11, 12, 13]

## 6/8 participants (2 stmt, 9 nodes, 6 occurrences)

```python
softmax_beta *= 10.0
softmax_beta_wm = 50.0
```
participants: [1, 2, 10, 11, 12, 13]

## 6/8 participants (1 stmt, 5 nodes, 6 occurrences)

```python
softmax_beta *= 10.0
```
participants: [1, 2, 10, 11, 12, 13]

## 5/8 participants (3 stmt, 36 nodes, 5 occurrences)

```python
s = int(block_states[t])
r = float(block_rewards[t])
Q_s = q[s, :]
```
participants: [2, 3, 10, 11, 13]

## 5/8 participants (3 stmt, 27 nodes, 5 occurrences)

```python
block_mask = blocks == b
block_actions = actions[block_mask]
block_rewards = rewards[block_mask]
```
participants: [1, 2, 10, 11, 13]

## 5/8 participants (3 stmt, 24 nodes, 5 occurrences)

```python
block_set_sizes = set_sizes[block_mask]
nA = 3
nS = int(block_set_sizes[0])
```
participants: [1, 2, 10, 11, 13]

## 5/8 participants (2 stmt, 24 nodes, 5 occurrences)

```python
r = float(block_rewards[t])
Q_s = q[s, :]
```
participants: [2, 3, 10, 11, 13]

## 5/8 participants (3 stmt, 22 nodes, 5 occurrences)

```python
block_states = states[block_mask]
block_set_sizes = set_sizes[block_mask]
nA = 3
```
participants: [1, 2, 10, 11, 13]

## 5/8 participants (2 stmt, 18 nodes, 5 occurrences)

```python
block_mask = blocks == b
block_actions = actions[block_mask]
```
participants: [1, 2, 10, 11, 13]

## 5/8 participants (1 stmt, 17 nodes, 5 occurrences)

```python
(lr, lambda_trace, wm_weight_base, softmax_beta, kappa, wm_noise) = model_parameters
```
participants: [1, 3, 11, 12, 14]

## 5/8 participants (1 stmt, 13 nodes, 5 occurrences)

```python
delta = r - Q_s[a]
```
participants: [2, 3, 10, 11, 13]

## 5/8 participants (2 stmt, 13 nodes, 5 occurrences)

```python
block_set_sizes = set_sizes[block_mask]
nA = 3
```
participants: [1, 2, 10, 11, 13]

## 4/8 participants (1 stmt, 41 nodes, 4 occurrences)

```python
w[s, :] = (1.0 - relax) * w[s, :] + relax * w_0[s, :]
```
participants: [1, 3, 10, 14]

## 4/8 participants (3 stmt, 36 nodes, 4 occurrences)

```python
r = float(block_rewards[t])
Q_s = q[s, :]
W_s = w[s, :]
```
participants: [3, 10, 11, 13]

## 4/8 participants (2 stmt, 24 nodes, 4 occurrences)

```python
Q_s = q[s, :]
W_s = w[s, :]
```
participants: [3, 10, 11, 13]

## 3/8 participants (3 stmt, 41 nodes, 3 occurrences)

```python
p_total = wm_weight * p_wm + (1.0 - wm_weight) * p_rl
p_total = max(p_total, eps)
log_p += np.log(p_total)
```
participants: [2, 11, 13]

## 3/8 participants (1 stmt, 34 nodes, 3 occurrences)

```python
w[s, :] = (1.0 - kappa) * w[s, :] + kappa * one_hot
```
participants: [3, 12, 14]

## 3/8 participants (2 stmt, 30 nodes, 3 occurrences)

```python
p_total = wm_weight * p_wm + (1.0 - wm_weight) * p_rl
p_total = max(p_total, eps)
```
participants: [2, 11, 13]

## 3/8 participants (1 stmt, 30 nodes, 6 occurrences)

```python
p_rl = 1.0 / np.sum(np.exp(softmax_beta * (Q_s - Q_s[a])))
```
participants: [1, 11, 13]
