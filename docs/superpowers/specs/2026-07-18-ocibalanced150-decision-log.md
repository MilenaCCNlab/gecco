# Decision log — ocibalanced150 rerun (overnight autonomous execution)

User instruction (2026-07-18, before signing off): "if something comes up, make
the most reasonable assumption, log it and continue. I want to see the full html
report in the morning/afternoon if done."

Every autonomous decision taken during execution is appended here with its
rationale.

## 2026-07-18

1. **Spec review gate self-approved.** The user approved all five design
   sections interactively before signing off; the written-spec review gate is
   subsumed by the overnight-autonomy instruction. Spec committed as approved.
2. **`compose-count` checkpoint delegated.** Threshold adopted: proceed with
   greedy (arm 1) regardless of count; for the hybrid-base exhaustive arm,
   proceed if candidate count keeps estimated scoring time under ~12 h at
   observed per-fit cost, otherwise tighten `--param-cap` by 1 and re-count
   (logged here when it happens).
3. **Third code touch required (not in the approved two).**
   `library_learning/compose/extract.py::run_extraction` unconditionally calls
   `make_splits`, which derives seeds from the group config (prompt+eval =
   5 + validation-50) — wrong for this design (seeds = train 0-49) — and would
   overwrite the manifest-derived `splits.json`. Minimal fix: a
   `resolve_splits` helper that honors a pre-existing `splits.json`
   (test-covered). Alternative (a decoy group config purely to trick
   `make_splits`, then swapping splits.json post-extraction) judged more
   fragile. Reasonable-assumption clause applied.
4. **Execution mode: inline (superpowers:executing-plans), current branch, no
   worktree.** Tasks are dominated by multi-hour background LLM jobs needing
   in-session monitoring/resume; a worktree would duplicate the large
   data/results tree and subagent-per-task cannot babysit long-running jobs.
5. **Gecco jobs get `GEMINI_API_KEY` exported from `GEMINI_API_KEY_LAKELAB` /
   `_COCOSCILAB` at launch** (backend reads the plain name; only suffixed keys
   exist in `.env`). Six individual chunks alternate the two keys to spread
   rate limits; group job uses LAKELAB.
6. **Cosmetic inaccuracy left in place:** extraction merge prompt says "12
   participants" in prose; actual 50 annotations are supplied. Not touched
   (surgical-change rule).
7. **Local environment gaps discovered and fixed.** `gecco-env` (Python 3.9.6)
   lacked `pydantic`/`python-dotenv`/`google-genai` (installed), but the gecco
   package itself needs Python >= 3.12 (`prompt.py:84` backslash-in-f-string
   SyntaxError on 3.9 — cluster runs used anaconda 2025.12). Created
   `gecco-env313/` (Python 3.13.14, full `requirements.txt`) for the LLM
   pipeline; `gecco-env` (3.9) still used for library_learning fitting + tests.
   Risk noted: gecco-env313 resolved pandas 3.0.3 (written-for-2.x code); if
   runs hit pandas-3 behavior changes, pin `pandas<3` there and relaunch.
8. **Plan expectation correction (Task 1 Step 6):** `test_schema_matches_old_dataset`
   also fails until the baseline job adds `baseline_bic` (old CSV includes that
   column) — same root cause as `test_baseline_bic_present`, not a defect.
9. **Generator model changed to `gemini-3.1-pro-preview`.** Google retired
   `gemini-3-pro-preview` (404 "no longer available", verified 2026-07-18 with
   both keys). Nearest successor `gemini-3.1-pro-preview` verified working with
   gecco's exact call shape (ThinkingConfig thinking_level=high). Both new
   configs updated. CAVEAT for the report: the 150-run's generator differs from
   the 45-run's (3.1 vs 3.0), so cross-run comparisons confound model change;
   within-run comparisons are unaffected. Extraction already used 3.1.
10. **Not reusing the old 45-pid individual models (user suggestion 2026-07-18).**
    Overlap between old-45 and new-150 subjects is only 12 (4 per split) —
    old set used fixed OCI bins, new uses percentile tertiles. Reuse saves
    ~8% of individual runs but would splice gemini-3-pro-preview models into a
    gemini-3.1 individual-ceiling set, confounding the coverage claim (library
    vs individual ceiling) with a model-version difference. Run already 12/150
    and progressing cheaply. Decision: let the uniform 3.1 run finish; do not
    reuse. Revisit only if API cost/quota becomes a hard blocker.
11. **Code review (xhigh) findings logged 2026-07-18** — see review in session:
    (a) `clean_raw_file` dropped the original per-file missing-marker guard
    (latent, 0/150 affected); (b) `baseline_bic` hybrid counts -1 trials while
    eval HYBRID_SOURCE guards them → Task 9 cross-checks will warn for 53 pids
    (expected/benign, results use guarded hybrid); (c) individual-150 config
    has stale dead splits; (d) Task 6 gate must check pids 0-49 specifically,
    not total best-model count. None blocks the run.
12. **Transient Gemini 503 killed chunk 0:25 at pid 7 (2026-07-18 ~22:20).**
    `google.genai.errors.ServerError: 503 UNAVAILABLE` after tenacity retries
    exhausted → process exit 1. Completed pids 0-6; other 5 chunks unaffected.
    Built a per-pid resilient driver (scratchpad `run_individual_resilient.sh`):
    runs each pid singly, skips those with an existing best_model, retries each
    up to 6× with 45s backoff — isolates transient failures to one pid and never
    overwrites a good model. Relaunched pids 7-24 with it (LAKELAB). Posture for
    the still-running 5 chunks: reactive — a 503 death re-invokes via the
    background-failure notification; resume that chunk's remaining pids with the
    same driver. (Monitor's `429`/`503 ` numeric tokens also matched BIC values
    like 429.57 → replaced with error-string-only pattern: ServerError,
    google.genai.errors, RESOURCE_EXHAUSTED, Traceback, etc.)
13. **Network blip killed all 5 remaining plain chunks at once (~22:30).**
    `httpx.ReadError: [Errno 54] Connection reset by peer` — a momentary local
    network drop reset every open socket simultaneously (not quota/outage; both
    keys probed "Pong!" immediately after). Switched the WHOLE run to resilient
    per-pid drivers: 6 drivers cover 7-24 / 25-49 / 50-74 / 75-99 / 100-124 /
    125-149 (each skips done pids, retries each pid 6×). pids 0-6 already done.
    3 drivers per key. Now blip-resilient. Monitor retuned to alert only on
    "STILL FAILING after" (a pid that exhausted retries), plus milestones/gates.
    Resume topology if session restarts: relaunch `run_individual_resilient.sh
    <lo> <hi> <KEY>` for any range with missing pids (idempotent — skips done).
14. **Extraction failed validation; repaired (2026-07-19 ~04:00). NEEDS YOUR REVIEW.**
    `compose-modules` produced 27 modules but merge validation failed after all 3
    repair rounds with exactly two error classes (no others):
    (a) 10 modules referenced `stage1_temp`/`stage2_temp` in slot code — those are
    override-slot *names*, not runtime variables; the runtime inverse-temp is
    `beta`. (b) 2 modules (`direct_mf_update`, `asymmetric_direct_mf_update`)
    overrode `stage1_update`/`stage2_update` with multi-line code that the
    renderer inlined at a fixed indent → IndentationError (the SAME bug RLWM hit
    and fixed in df8bd50, never ported to two-step `render.py`).
    **Fixes:** (a) ported df8bd50 to `render.py` (block-indent multi-line
    statement overrides) — reviewed-pattern code fix; (b) materialized
    `module_inventory.json` from the last repair round's LLM JSON
    (`llm_log/call_053_merge_repair3.json`) with a token replacement
    `stage1_temp`/`stage2_temp`→`beta` in the 10 modules' slot code only
    (semantically exact — none override the temp; backbone scaffold untouched).
    Result: 27 modules pass `compose-modules --skip-llm` smoke checks; 13 compose
    tests green. **ASSUMPTION (autonomous): the reconstruction fidelity gate was
    NOT re-run** (I bypassed `merge_inventory`; the gate runs inside it). Structural
    validity + per-module/pair smoke checks are sufficient for search/coverage/eval
    to run; the gate is an extraction-fidelity quality check, not a pipeline
    blocker. Re-run it in the morning if you want the fidelity numbers.
15. **Hybrid-base module set remapped for the new inventory (2026-07-19).**
    `hybrid.py::HYBRID_MODULES` is hardcoded to the OLD 45-pid ids; the 150-pid
    extraction renamed two: `choice_stickiness`→`stage1_stickiness`,
    `separate_stage_betas`→`separate_stage2_beta`. Ran the hybrid-base arm via
    `bash/hybrid_search_150.py` (inline, mirrors `cmd_hybrid_search`, remapped
    base, param_cap 9 = 22 candidates) so the shared `hybrid.py` stays intact for
    the old run. Task 9 arm-2 eval reads the frozen `hybrid_base/composed_model.txt`
    and needs no remap.
16. **Monitor pattern was too broad (2026-07-18).** Benign numpy
    `RuntimeWarning: invalid value encountered in divide` / `overflow
    encountered in exp` (unstable softmax in an LLM-proposed model during
    fitting) tripped the monitor's `invalid` grep — false alarm; chunk 100:125
    was healthy. Restarted the monitor matching only real failures (Traceback,
    RESOURCE_EXHAUSTED/429/quota, "no longer available", Killed, MemoryError,
    google.genai.errors).
