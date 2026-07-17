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

## Splits (revised 2026-07-17 after pitfall review)

| Set | Participants | Used for |
|---|---|---|
| Library seed | 1, 2, 4–13 (12 subjects) | Module extraction (+ seed reconstruction gate) |
| Composition validation | 4–13 (group gecco's eval set) | Selecting the single composed program (mean BIC) |
| Library reconstruction | 10 of 14–44, OCI-stratified, deterministic | Per-participant best composition — does the library span unseen individuals? |
| Final test | remaining 21 of 14–44 | The ONLY set results are claimed on |

Rationale: selecting the composition on 4–13 matches group gecco's
information budget exactly (it scored candidates on those same 10 each
iteration). The library overlap on those subjects is deliberate and
disclosed; every reported claim lives on the untouched 21. The
reconstruction 10 are assigned deterministically (sort held-out pids by
`(oci, pid)`, indices `i % 3 == 1`), written once to `splits.json` with OCI
mean/std per subset as a balance check.

## Stage 1 — Module extraction (Gemini-curated, miner-grounded, fully logged)

**Reproducibility requirement (user-mandated):** the LLM abstraction step runs
through the **Gemini API** — not in-session — with **every prompt and response
logged verbatim** to disk.

- **Model:** `gemini-3.1-pro-preview` for ALL calls (annotation, merge,
  repair — user decision after pitfall review; listed as available
  2026-07-17), temperature 0. **Key:** `GEMINI_API_KEY_LAKELAB` from `.env`
  (verified working; plain `GEMINI_API_KEY` is invalid — do not use).
- **Logging:** every call writes `llm_log/call_{NNN}.json` containing
  `{model, modelVersion (from response), generationConfig, full prompt,
  full raw response, timestamp}`. The logged artifacts freeze the actual run;
  re-running replays from prompts.
- **LLM step structure:** (1) one call per seed program → structured mechanism
  annotation; (2) one merge call over all 12 annotations + mining report →
  deduplicated module inventory (JSON: name, injection-point code, params +
  bounds, provenance, incompatibilities). Deterministic Python validates and
  renders the inventory into `module_inventory.py`; malformed LLM output fails
  loudly, and any manual repair is recorded in `llm_log/MANUAL_EDITS.md`.

**Extraction quality gates (added after pitfall review, all user-approved):**

1. **Seed reconstruction gate** — for each seed participant, render the
   candidate composed of exactly their annotated modules, fit it to their own
   data (seeded protocol), and compare BIC to their stored individual-gecco
   BIC. Written to `reconstruction_report.json`; a seed whose reconstructed
   BIC exceeds stored by > 15 fails the gate → repair loop / manual fix.
   Catches extraction infidelity (mechanism distorted in translation).
2. **Slot-expressibility escape hatch** — the annotation prompt lets Gemini
   mark a mechanism `"expressible_in_slots": false` with a reason instead of
   shoehorning it; the validator reports these per seed so slot-contract
   extensions are deliberate decisions, not silent distortions.
3. **Coverage audit** — the merge reply must map every annotated mechanism to
   a module id or an explicit logged decision; the validator fails if any
   annotated mechanism is unaccounted for.
4. **Bounds cross-check** — module parameter bounds are checked against the
   provenance participants' source docstring bounds (`parse_bounds`).
5. **Module state isolation** — module-internal state variables must be
   prefixed with the module id (prompt rule + AST validator check), and all
   module PAIRS are smoke-tested at inventory time, not just singletons.
- `google-genai` gets pip-installed into `gecco-env/` (or plain REST via
  urllib — decide at implementation; REST avoids a new dependency).

Pipeline details:

- Run the existing `library_learning` AST fragment miner (`library_learning/mining.py`)
  over the 12 seed programs (`models/best_model_0_participant{P}.txt`).
- Gemini reads the 12 programs + mining report and curates the inventory behind
  `module_inventory.py`:
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
  composition-validation participants (4–13). Ties (< 1 BIC point) → fewer
  parameters wins.
- Every candidate logged in `search_log.jsonl` (module set, param count,
  per-participant BICs, mean BIC).
- **Selection stability report** (`selection_report.json`): top-10 candidates
  by validation mean BIC, plus leave-one-participant-out rank stability of
  the top candidates — reported alongside the winner to expose winner's-curse
  noise on a 10-subject validation set.

## Stage 2b — Library reconstruction on unseen participants

For each of the 10 reconstruction participants (OCI-stratified from 14–44):
search the library for that individual's best module combination (same cap,
same seeded fitting, same search mode as Stage 2) and compare the resulting
BIC against (a) their individual-gecco program refit (ceiling) and (b) the
group gecco model refit. This measures whether the library spans unseen
individuals — a library-quality metric reported separately
(`reconstruction_results.json` + RESULTS.md section) and never mixed into
the single-program claim.

## Stage 3 — Generalization test

Winner frozen as `composed_model.txt`, then all models fit per-participant on
the **21 final-test participants** (the only set results are claimed on)
under the identical seeded protocol:

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
- `llm_log/` (every Gemini prompt/response, verbatim, numbered)
- `splits.json`
- `search_log.jsonl`, `selection_report.json` (top-10 + LOO rank stability)
- `reconstruction_report.json` (seed fidelity gate),
  `reconstruction_results.json` (unseen-participant library coverage)
- `composed_model.txt`, `winner.json`
- `test_results.json`, `test_results.csv` (per-participant BICs, all models)
- `RESULTS.md` + comparison figure, paper style (teal = hybrid reference,
  blue = composed winner, gray = others; PNG + PDF)

## Code placement

New subcommands in the existing repo-level `library_learning/` package
(reuse `config.py` resolution, `loading.py` loaders, `mining.py`):
- `compose-modules` — mine + run the logged Gemini extraction calls + validate/render inventory
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
