# Manual edits to LLM output

Base inventory: `call_026_merge_repair6.json` (Gemini merge + 6 logged repair
rounds; rounds 4-6 driven by `repair_driver.py` after the in-pipeline limit
of 3). Every edit below is mechanical (no module semantics changed by hand);
all semantic repairs were made by Gemini itself in logged calls.

1. `eligibility_trace`: moved the `"rl_update"` entry from `"slots"` to
   `"overrides"` — it is an override slot; Gemini kept returning it under
   `"slots"` across repair rounds 1-3.
2. All modules: renamed `n_states` → `nS` and `n_actions` → `nA`
   (word-boundary regex) in slot/override code. Gemini introduced these
   non-whitelist names in repair round 6.
3. `eligibility_trace`, `action_stickiness`, `choice_perseveration`:
   dropped the `init` snippet that duplicated `block_init` verbatim
   (nS-sized state cannot exist at function level; prompt rule 5).

Applied by `scratchpad/mechanical_fix.py` / `targeted_repair.py` (same edits
re-applied deterministically when rebuilding from call_026).

Semantic fidelity repairs (Gemini, logged): after the reconstruction gate
flagged seeds 3, 13, 14, targeted repair calls with the original program +
composed reconstruction + diagnosis in-prompt
(`call_027..call_030`, tags `targeted_repair_i{n}_p{pid}`) replaced:
- p3: `outcome_gated_wm_reliance`
- p13: `set_size_dependent_wm_lapse`, `asymmetric_fixed_wm_update`
- p14: `load_dependent_wm_decay_global`, `load_dependent_wm_arbitration`,
  `arbitration_scaled_wm_update` (two iterations)
Final gate: 8/8 seeds within +15 BIC of their original-program refit.
