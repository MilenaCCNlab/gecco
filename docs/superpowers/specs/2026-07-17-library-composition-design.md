# Library Composition: Cognitive-Module Library → Single Composed Program → Generalization Test

**Date:** 2026-07-17
**Status:** Approved design (brainstorming session with Akshay)
**Task pair:** two-step psychiatry, OCI-balanced, max setting
- Group run: `results/two_step_psychiatry_group_function_ocibalanced_maxsetting/`
- Individual run: `results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual/`
- Data: `data/two_step_gillan_2016_ocibalanced.csv` (45 participants, ids 0–44)

## Context & motivation

Group gecco evolved a single program (`models/best_model_0.txt`, hybrid MB/MF +
win/lose stickiness) using prompt participants **1, 2** (in LLM context) and eval
participants **4–13** (mean-BIC scoring each iteration); participants **14–44**
were held-out test (0 and 3 unused — splits `prompt [1:3]`, `eval [4:14]`,
`test [14:]` are index slices over sorted ids, see
`gecco/prepare_data/io.py:parse_split`). Individual gecco separately evolved one
best program per participant for all 45.

**Question:** can we do better than group gecco's single evolved program by
(1) building a library of cognitive modules from the *individual* best programs
of exactly the participants group gecco saw (1, 2, 4–13 — same information
budget), (2) composing library modules into a single program via deterministic
search, and (3) testing generalization on unseen participants?

## Splits (no double-dipping)

| Set | Participants | Used for |
|---|---|---|
| Library seed | 1, 2, 4–13 (12 subjects) | Module extraction only; never re-scored |
| Validation | 10 of 14–44, OCI-stratified, seeded | Composition selection (mean BIC) |
| Final test | remaining 21 of 14–44 | Frozen winner + baselines only |

Validation/test assignment is written once to `splits.json` (seeded,
OCI-stratified: sort test participants by `oci`, alternate/seeded-sample within
strata to get 10 validation + 21 test) and reused everywhere. Report OCI
mean/std per half as a balance check.

## Stage 1 — Module extraction (LLM-curated, miner-grounded)

- Run the existing `library_learning` AST fragment miner (`library_learning/mining.py`)
  over the 12 seed programs (`models/best_model_0_participant{P}.txt`).
- Claude reads the 12 programs + mining report and curates `module_inventory.py`:
  - **Backbone**: shared scaffold — softmax choice rules, TD updates, MB lookahead
    with `T=[[.7,.3],[.3,.7]]`, standard NLL tail.
  - **Mechanism modules**: every distinct mechanism found in the seeds —
    **singletons included; deduplication only** (identical mechanisms in two
    seeds contribute one module). Expected kinds: MB/MF mixture weight,
    stickiness/perseveration variants (incl. outcome-dependent), value
    decay/forgetting, eligibility trace, separate stage-1/stage-2 learning
    rates, idiosyncratic inits, softmax variants (raw vs. stable stay distinct —
    float-exactness rules from the earlier library-learning work carry over).
  - Each module declares: injected code per injection point (init, stage-1 logit
    assembly, stage-2 logit assembly, value update, post-trial bookkeeping),
    parameters + bounds, provenance (which seed participants), and
    incompatibilities/dependencies (e.g., two stage-1 stickiness variants are
    mutually exclusive).
- Human-readable `MODULES.md` documents every module with provenance.

## Stage 2 — Compose & select

- **Candidate rendering:** template engine assembles backbone + chosen modules
  into one flat standalone
  `cognitive_model(action_1, state, action_2, reward, model_parameters)` with a
  docstring carrying parameter bounds — same format as gecco output, so
  `gecco/offline_evaluation/utils.py:build_model_spec` fits it unchanged.
  No imports, numpy-only.
- **Parameter cap: 8** (same guardrail group gecco ran under).
- **CHECKPOINT (user-confirmed):** after extraction, enumerate all
  cap-respecting, compatibility-respecting combinations and **report the count
  to Akshay before any fitting**. He decides:
  - exhaustive search (fit every candidate), or
  - greedy forward selection from the backbone (add best-gain module until no
    validation mean-BIC improvement; O(M²) fits).
- **Fitting protocol** (mirrors `run_fit`): per-participant L-BFGS-B, uniform
  random starts within bounds, n_starts = 10, **seeded RNG** (seed = f(participant
  id, candidate hash)) for full reproducibility. Objective = mean BIC across the
  10 validation participants. Ties (< 1 BIC point) → fewer parameters wins.
- Every candidate logged in `search_log.json` (module set, param count,
  per-participant BICs, mean BIC).

## Stage 3 — Generalization test

Winner frozen as `composed_model.txt`, then all models fit per-participant on
the 21 final-test participants under the identical seeded protocol:

| Model | Role |
|---|---|
| Composed program | The claim |
| Group gecco `best_model_0.txt` | Headline rival (same information budget) |
| Hybrid (Daw) baseline | Field standard; cross-check vs. `baseline_bic` column |
| Individual gecco per-participant programs | Ceiling reference (not a fair rival) |

**Stats:** mean BIC per model; per-participant ΔBIC vs. group gecco;
win/tie/loss counts; paired Wilcoxon (composed vs. group gecco, composed vs.
hybrid). Validation-set tables reported separately, labeled as selection data.

## Outputs

New dir `results/two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual/library_composition/`:
- `module_inventory.py`, `MODULES.md`
- `splits.json`
- `candidates/` (rendered candidate programs), `search_log.json`
- `composed_model.txt`, `winner_params_validation.csv`
- `test_results.json`, `test_results.csv` (per-participant BICs, all models)
- `RESULTS.md` + comparison figure, paper style (teal = hybrid reference,
  blue = composed winner, gray = others; PNG + PDF)

## Code placement

New subcommands in the existing repo-level `library_learning/` package
(reuse `config.py` resolution, `loading.py` loaders, `mining.py`):
- `compose-modules` — mine + emit inventory skeleton (curation stays manual/LLM)
- `compose-search` — enumerate (count-only mode first for the checkpoint), fit,
  select
- `compose-eval` — final-test fits + stats + figure

Python 3.9 (`gecco-env/` venv at repo root: numpy/pandas/scipy/pyyaml).

## Verification

1. Every rendered candidate is exec'd and smoke-fit on one participant before
   the search starts (catches render bugs early).
2. Backbone-only candidate behaves hybrid-like (sanity).
3. Stored group-gecco test BICs (`bics/best_bic_on_test_run0.json`) cross-checked
   against our refits — WARN on mismatch, investigate before reporting.
4. Hybrid refits cross-checked against the data's `baseline_bic` column.
5. `splits.json` OCI balance reported (means/stds per half).
6. End-to-end: `compose-search` exits nonzero if any candidate fails to fit;
   `compose-eval` refuses to run if `composed_model.txt` is missing or
   validation artifacts are newer than it (freeze discipline).

## Non-goals

- No re-running of gecco/LLM evolution; all programs already exist.
- No per-participant model *assignment* (mixture/BMC across library programs) —
  single composed program only. (Natural follow-up, out of scope.)
- No refactor of the existing `library_learning` compression/verify pipeline.
