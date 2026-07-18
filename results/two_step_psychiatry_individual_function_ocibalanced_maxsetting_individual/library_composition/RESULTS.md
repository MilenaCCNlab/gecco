# Library composition results

Winner modules: `choice_stickiness+q_init_05+unchosen_value_decay_center` (4 params)

## Mean BIC on final test (21 participants)

| model | mean BIC |
|---|---|
| individual | 379.65 |
| hybrid | 404.29 |
| group | 410.89 |
| composed | 414.42 |

composed vs group: mean dBIC 3.53, W/T/L 8/3/10, wilcoxon p=0.7854
composed vs hybrid: mean dBIC 10.13, wilcoxon p=0.2290

## Cross-check warnings

- group refit BIC differs from stored for p16: 464.07 vs 438.44
- group refit BIC differs from stored for p33: 532.22 vs 496.04
- hybrid refit BIC differs from baseline_bic for p14: 504.47 vs 516.92
- hybrid refit BIC differs from baseline_bic for p20: 286.25 vs 362.19
- hybrid refit BIC differs from baseline_bic for p27: 334.60 vs 369.78
- hybrid refit BIC differs from baseline_bic for p33: 484.54 vs 495.58
- hybrid refit BIC differs from baseline_bic for p37: 296.23 vs 322.93
- hybrid refit BIC differs from baseline_bic for p38: 406.68 vs 419.61
- hybrid refit BIC differs from baseline_bic for p39: 389.85 vs 397.45
- hybrid refit BIC differs from baseline_bic for p41: 402.62 vs 409.34
