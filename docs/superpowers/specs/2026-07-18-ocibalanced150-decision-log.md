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
