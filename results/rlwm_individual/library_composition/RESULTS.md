# RLWM library composition results

Winner modules: `unified_wm_update_decay` (4 params)

## Mean BIC on final test (15 participants)

| model | mean BIC |
|---|---|
| composed | 502.80 |
| canonical | 519.81 |
| individual | 535.48 |
| group | 536.60 |

composed vs group: mean dBIC -33.79, W/T/L 14/0/1, wilcoxon p=0.0001
composed vs canonical: mean dBIC -17.01, wilcoxon p=0.0103

## Cross-check warnings

- canonical refit BIC differs from baseline_bic for p4: 314.78 vs 203.19 (the column's producing variant is unknown)
- canonical refit BIC differs from baseline_bic for p6: 488.83 vs 268.38 (the column's producing variant is unknown)
- canonical refit BIC differs from baseline_bic for p7: 328.44 vs 184.97 (the column's producing variant is unknown)
- canonical refit BIC differs from baseline_bic for p8: 628.36 vs 399.94 (the column's producing variant is unknown)
- canonical refit BIC differs from baseline_bic for p9: 374.91 vs 221.97 (the column's producing variant is unknown)
- canonical refit BIC differs from baseline_bic for p36: 416.56 vs 261.54 (the column's producing variant is unknown)
- canonical refit BIC differs from baseline_bic for p38: 616.30 vs 379.80 (the column's producing variant is unknown)
- canonical refit BIC differs from baseline_bic for p39: 640.76 vs 391.51 (the column's producing variant is unknown)
- canonical refit BIC differs from baseline_bic for p41: 630.27 vs 408.89 (the column's producing variant is unknown)
- canonical refit BIC differs from baseline_bic for p42: 591.44 vs 330.53 (the column's producing variant is unknown)
- canonical refit BIC differs from baseline_bic for p43: 407.55 vs 261.82 (the column's producing variant is unknown)
- canonical refit BIC differs from baseline_bic for p44: 660.31 vs 404.28 (the column's producing variant is unknown)
- canonical refit BIC differs from baseline_bic for p47: 464.96 vs 311.89 (the column's producing variant is unknown)
- canonical refit BIC differs from baseline_bic for p48: 591.30 vs 353.31 (the column's producing variant is unknown)
- canonical refit BIC differs from baseline_bic for p50: 642.35 vs 383.23 (the column's producing variant is unknown)

## Library reconstruction (leak-free, 7 held-out participants)

Per-participant best composition beats the participant's own individual-gecco
refit 7/7 (mean 458.4 vs 515.0) and the group program 7/7 (497.2). Caveat: the
per-participant composition selects on that participant's own data, as does
individual gecco — information-matched, but not a held-out claim.

## Disclosures

- Library seeds: 8 of the 13 group-seen participants (eval pids 15-19 have no
  individual gecco fits). All seeds young (18-36); 20 of 22 test/reconstruction
  participants are older adults (46-85).
- Held-out pool reclaims fitted pids 0, 4-9 (never consumed by group gecco's
  prompt or eval) in addition to test-split pids 36-50; config eval/test
  overlap pids 14-19 are treated as group-seen and excluded.
- Three extracted modules condition the current choice's likelihood on the
  current trial's reward (faithful to their source programs; reward is
  deterministic given choice, so this is within-trial outcome leakage). They
  are excluded from the composed-program and reconstruction arms
  (module_inventory_leakfree.json); the unconstrained arm is archived under
  unconstrained_search/ and reconstruction_results_fulllib.json. Individual
  gecco refits retain their original mechanisms, leaks included — where a
  source program leaks, its "ceiling" BIC is inflated in the composed
  program's disfavor.
- Stored BICs (best_bic_0_participant*.json and the data's baseline_bic
  column) are not comparable under this pipeline's protocol (different data
  span); all comparisons use refits under one seeded protocol. The
  reconstruction fidelity gate references the original-program refit.
- The composed backbone skips missed trials (actions == -2, ~3%) in
  likelihood and updates; several original programs index them unguarded.
- Samples are small: 8 seeds, 10 validation, 7 reconstruction, 15 test.
  Search was greedy (exhaustive ~18 h; DECISIONS.md #7).
