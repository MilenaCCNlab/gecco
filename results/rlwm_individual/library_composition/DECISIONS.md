# Autonomous-run decision log — RLWM library composition

User directive (2026-07-18, before bed): "make the most reasonable assumption,
log it and continue"; deliverable is the full HTML report. Spec:
`docs/superpowers/specs/2026-07-18-rlwm-library-composition-design.md`.

1. **Execution mode: inline** (not subagent-driven). The run stages are
   long-lived compute needing direct monitoring and timing-based decisions;
   the code tasks were already fully specified in the plan.
2. **Branch/worktree:** stayed on `gecco-individual-differences` in the main
   checkout — same branch and layout as the completed two-step run, and the
   run artifacts land in `results/` here.
3. **Pre-existing test failures:** `tests/test_ocibalanced150_dataset.py`
   (2 failures) fail identically without any of this work's changes —
   they reference an ocibalanced150 dataset from a different work line.
   Left untouched; all 84 other tests pass, including every two-step
   compose test (the "two-step unmodified" gate).
4. **Preflight timing:** one backbone candidate fit ≈ 2.3 s/participant
   (10 L-BFGS-B starts, 324 trials) → ≈ 23 s per candidate on the
   10-participant validation set. Exhaustive-vs-greedy decided after
   compose-count with the plan's rule: exhaustive iff estimate ≤ 4 h.
5. **Fidelity-gate reference changed to original-program refit.** The stored
   `best_bic_0_participant*.json` values are not comparable under this
   pipeline's protocol: refitting the ORIGINAL seed programs on the full
   324 trials (same seeded L-BFGS-B protocol used everywhere here) lands
   +100..+220 BIC above stored, uniformly across seeds (e.g. p1 438.67 vs
   stored 222.22) — the gecco RLWM run recorded BICs on a different data
   span. Meanwhile recompositions match the original refits closely
   (p1: 439.17 vs 438.67). The gate now compares recon vs original-refit
   (tolerance unchanged, 15 BIC); stored values remain in
   reconstruction_report.json for disclosure. Two-step is unaffected
   (its stored BICs matched refits).
6. **Repair strategy after round 6:** Gemini's repair rounds oscillated on
   mechanical issues (misplaced rl_update key; whitelist renames
   n_states/n_actions; nS-sized state in function-level init). Fixed those
   three classes deterministically in code (scratchpad/mechanical_fix.py),
   every edit listed in llm_log/MANUAL_EDITS.md; no module semantics
   changed. Code-level repairs in rounds 4-6 remain Gemini's
   (call_024..026).
7. **Search mode: greedy.** compose-count: 22 modules → 918 cap-respecting
   candidates. Measured cost of one 6-param candidate on the 10 validation
   participants: 71.9 s → exhaustive ≈ 18.3 h, far over both the plan's 4 h
   rule and the report-by-morning deadline. Greedy forward selection
   (~90-130 candidate fits ≈ 1.5-2.5 h) mirrors the two-step precedent
   (exhaustive ≈ 21 h there → greedy, pre-authorized).
8. **Within-trial outcome leakage found and quarantined from the shared-
   program arm.** The first greedy winner was dominated by
   `outcome_gated_wm_reliance` (validation BIC 482→299), which updates a
   gate with the CURRENT trial's reward before the current choice's
   likelihood. In RLWM reward is deterministic given the choice, so this is
   within-trial outcome leakage — and it is FAITHFUL to the source: seed 3's
   original GPT-5-evolved program does the same (lines 66-76), while the
   group program does not. Audit of all 22 modules found 3 leaky
   (`outcome_gated_wm_reliance`, `set_size_dependent_wm_lapse`,
   `load_dependent_wm_arbitration` — the latter two via `abs(r - q[s,a])`
   in mix_weight). Resolution:
   - Shared-program arm (headline, vs group/canonical): re-searched over the
     19 leak-free modules (`module_inventory_leakfree.json`); fair rivals.
   - The unconstrained (leak-permitting) search artifacts moved to
     `unconstrained_search/` and reported only as a disclosed confound.
   - Reconstruction arm: the full-22-module run confirmed total leak
     dominance (all 7 pids picked `outcome_gated_wm_reliance` and "beat"
     their individual ceiling by 130-290 BIC — spurious, since not every
     individual program leaks). REVISED: the leak-free library is the
     primary reconstruction arm too; the full-library run is archived as
     `reconstruction_results_fulllib.json` / `reconstruction_fulllib/` and
     reported only as a leak-dominance demonstration.
   - The individual-gecco "ceiling" BICs are themselves inflated wherever
     source programs leak; flagged in the report.
9. **Extraction run 1 failed on a harness bug, not Gemini** (this run's
   analog of the two-step int/str lesson): the renderer substituted
   `rl_update`/`wm_update` overrides inline, so multi-line if/else WM
   updates — which Gemini produced for most WM modules — lost their
   indentation and every affected module failed the syntax smoke check;
   all 3 repair rounds were burned on our bug. Fixed by block-indenting
   statement overrides (commit "fix: block-indent rl_update/wm_update"),
   regression test added, extraction re-run from scratch (fresh annotate +
   merge calls; run-1 calls remain in llm_log/ for the record; annotations
   at temperature 0 are expected to match). No hand edits to inventory
   content.
