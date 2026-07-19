# Library composition results

Winner modules: `eligibility_trace+mb_mf_mixture+separate_learning_rates+separate_stage2_beta+stage1_stickiness+value_decay_to_half` (9 params)

## Mean BIC on final test (50 participants)

| model | mean BIC |
|---|---|
| individual | 340.33 |
| group | 357.39 |
| composed | 360.13 |
| hybrid | 371.42 |

composed vs group: mean dBIC 2.74, W/T/L 20/3/27, wilcoxon p=0.2491
composed vs hybrid: mean dBIC -11.29, wilcoxon p=0.0023

## Cross-check warnings

- hybrid refit BIC differs from baseline_bic for p100: 378.65 vs 391.81
- hybrid refit BIC differs from baseline_bic for p101: 416.54 vs 422.41
- hybrid refit BIC differs from baseline_bic for p102: 264.15 vs 323.03
- hybrid refit BIC differs from baseline_bic for p105: 577.85 vs 589.15
- hybrid refit BIC differs from baseline_bic for p107: 406.31 vs 419.15
- hybrid refit BIC differs from baseline_bic for p108: 576.00 vs 583.39
- hybrid refit BIC differs from baseline_bic for p113: 280.01 vs 291.00
- hybrid refit BIC differs from baseline_bic for p114: 359.55 vs 365.33
- hybrid refit BIC differs from baseline_bic for p119: 334.60 vs 369.78
- hybrid refit BIC differs from baseline_bic for p120: 487.00 vs 492.73
- hybrid refit BIC differs from baseline_bic for p122: 296.65 vs 308.25
- hybrid refit BIC differs from baseline_bic for p129: 423.02 vs 489.71
- hybrid refit BIC differs from baseline_bic for p132: 385.26 vs 391.87
- hybrid refit BIC differs from baseline_bic for p137: 524.64 vs 591.61
- hybrid refit BIC differs from baseline_bic for p139: 327.03 vs 338.68
- hybrid refit BIC differs from baseline_bic for p140: 315.51 vs 363.17
- hybrid refit BIC differs from baseline_bic for p141: 404.80 vs 434.42
- hybrid refit BIC differs from baseline_bic for p146: 385.69 vs 393.24
- hybrid refit BIC differs from baseline_bic for p149: 339.67 vs 350.15
