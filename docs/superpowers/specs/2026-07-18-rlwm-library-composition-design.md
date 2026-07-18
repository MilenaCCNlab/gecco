# RLWM Library Composition: Young-Seeded Module Library → Composed Program → Aging Generalization Test

**Date:** 2026-07-18
**Status:** Approved design (brainstorming session with Akshay)
**Task pair:** RLWM (Collins-style reinforcement learning / working memory, aging dataset)
- Group run: `results/rlwm/`
- Individual run: `results/rlwm_individual/`
- Data: `data/rlwm.csv` (78 participants; pids 0–35 young 18–36, pids 36–77 old 46–85;
  324 trials each across 8 blocks; set sizes 3/6; 3 actions; missed trials coded
  `actions == -2`, ~3% of trials; per-participant `age` and `baseline_bic` columns)

## Context & motivation

This is the RLWM replication of the two-step library-composition run
(`docs/superpowers/specs/2026-07-17-library-composition-design.md`). Group gecco
evolved a single RL+WM program (`results/rlwm/models/best_model_0.txt`, 5 params)
from prompt participants **1–3** and eval participants **10–19** (splits `prompt
[1:4]`, `eval [10:20]`, `test [14:]` over sorted ids in `config/rlwm.yaml` —
note the config's eval window **overlaps** its test split on pids 14–19; those
pids are treated as group-seen and excluded from our test pool). Individual
gecco programs exist for **30 of 78** participants only: pids **0–14** (young)
and **36–50** (old).

**LLM usage:** the original gecco programs were generated with GPT-5 (data
provenance only). This pipeline makes **no OpenAI calls** — every LLM call is
`gemini-3.1-pro-preview` (user decision).

**Question:** same as two-step — can a library of cognitive modules extracted
from the individual programs of exactly the participants group gecco saw beat
the group program on held-out participants, and does the library span unseen
individuals? The aging twist: the seeds are all **young adults**, and most of
the held-out evaluation pool is **older adults** — so the generalization test
doubles as *does a young-derived module library span older adults?*

## Splits (age replaces OCI as the stratification variable)

| Set | Participants | Used for |
|---|---|---|
| Library seed | {1,2,3} ∪ ({10–19} ∩ fitted) = **{1,2,3,10–14}** (8 subjects, all young) | Module extraction (+ seed reconstruction gate) |
| Composition validation | 10–19 (group gecco's eval set; fitting needs data only) | Selecting the single composed program (mean BIC) |
| Library reconstruction | 7 of the held-out fitted pool, age-stratified, deterministic | Per-participant best composition — does the library span unseen individuals? |
| Final test | remaining 15 of the held-out fitted pool | The ONLY set results are claimed on |

- **Group-seen pids** = prompt ∪ eval = {1,2,3} ∪ {10–19}. Eval pids
  **15–19 have no individual fits** and therefore cannot seed the library —
  disclosed in `splits.json` (`seed_pids_excluded_unfitted`). Pids 14–19 sit
  in both the config's eval and test windows; they are treated as group-seen
  and excluded from the test pool.
- **Held-out fitted pool** (user decision: reclaim unused young) = fitted
  pids never seen by group gecco = **{0, 4–9} ∪ {36–50}** (22 subjects:
  7 young, 15 old). Pids 0 and 4–9 are outside the config's declared test
  split but were consumed by neither prompt nor eval — leak-free, disclosed
  deviation from the two-step "held-out ⊆ config test" convention. The 48
  unfitted held-out pids are unused (no individual-gecco ceiling); evaluating
  composed vs group on them is a disclosed follow-up, not part of this run.
- **Reconstruction assignment** is deterministic: sort the pool by
  `(age, pid)`, take indices `i % 3 == 1` → 7 reconstruction pids, 15 test
  pids (~1:2 ratio as in two-step). Written once to `splits.json` with age
  mean/std per subset as a balance check.
- **Disclosed limitation:** 8 seeds and 7/15 recon/test subjects is smaller
  than two-step's 12 seeds and 10/21; stated in RESULTS.md.

## Stage 1 — Module extraction (Gemini-curated, miner-grounded, fully logged)

Identical protocol to two-step, RLWM-specific content:

- **Model:** `gemini-3.1-pro-preview` for ALL calls, temperature 0.
  **Key:** `GEMINI_API_KEY_LAKELAB` from `.env` (fallback
  `GEMINI_API_KEY_COCOSCILAB`; plain `GEMINI_API_KEY` is invalid — do not use).
- **Logging:** every call → `llm_log/call_{NNN}.json` (model, modelVersion,
  generationConfig, full prompt, full raw response, timestamp). Manual repairs
  recorded in `llm_log/MANUAL_EDITS.md`.
- **LLM step structure:** (1) one annotation call per seed program (8 calls);
  (2) one merge call over all annotations + mining report → deduplicated module
  inventory. Deterministic Python validates and renders; ≤3 repair rounds.
- **Pid coercion at the boundary:** LLM-returned JSON pids arrive as strings —
  coerce to int on parse (two-step lesson, baked in from day one).

**RLWM backbone** (what the gecco fill-in template forced every seed model to
share): outer loop over blocks with per-block reset (`nS` from the block's
set size, `nA = 3`, `q`, `w`, `w_0` all initialized to `1/nA`), RL softmax
policy, near-deterministic WM softmax, probability-level mixture, delta-rule
Q update, NLL accumulation. Backbone parameters: `learning_rate` [0, 1],
`beta` [0, 10], `wm_weight` [0, 1] (3 params, vs two-step's 2).

**Slot contract:**
- Append slots: `init` (function level), `block_init` (after per-block
  q/w/w_0 reset), `pre_choice`, `rl_logits_extra`, `wm_logits_extra`,
  `probs_extra` (after the mixture — lapse etc.), `update_extra` (inside the
  valid-trial guard, after RL+WM updates), `post_trial` (every trial,
  including missed — decay lives here).
- Override slots: `q_init` (default `(1.0 / nA) * np.ones((nS, nA))`),
  `w_init` (same default), `rl_values` (default `q[s]`), `wm_values`
  (default `w[s]`), `rl_temp` (default `beta`), `wm_temp` (default `50.0`),
  `mix_weight` (default `wm_weight`), `rl_update` (default
  `q[s, a] += learning_rate * delta`), `wm_update` (default one-shot
  `w[s, a] = r`).
- **Missed trials:** backbone guards `0 <= a < nA` — skips likelihood and
  value updates but still runs `post_trial`. This matches the group model.
  Some seed models index `-2` unguarded; skipping only *lowers* the
  reconstructed NLL, so the fidelity gate direction is safe. Noted in
  RESULTS.md as a protocol difference between refits and original fits.
- The exact backbone text is finalized during implementation against the 8
  seed programs and `mining.py`'s shared-fragment report **before any Gemini
  call**; the annotation prompt's "backbone machinery — do not list as
  mechanisms" list must name exactly what the frozen backbone contains.

**Extraction quality gates** (unchanged from two-step): seed reconstruction
gate (recomposed seed BIC ≤ stored individual BIC + 15, all 8 seeds,
`reconstruction_report.json`), slot-expressibility escape hatch, coverage
audit, bounds cross-check against source docstrings, module state isolation
(`{id}_` prefix rule + AST check, all module pairs smoke-tested).

Expected mechanism kinds (annotation prompt examples, from eyeballing seed
programs): WM decay/forgetting variants, capacity/set-size scaling of the WM
weight or learning rate, lapse/uniform mixture, chunking/interference across
states, load-dependent drift, negative-feedback asymmetry, perseveration.

## Stage 2 — Compose & select

- **Candidate rendering:** flat standalone
  `cognitive_model(stimulus, actions, rewards, blocks, set_sizes, model_parameters)`
  with docstring-carried bounds — same format gecco's `run_fit` regexes parse.
  No imports, numpy-only.
- **Parameter cap: 6** — the guardrail RLWM group gecco ran under ("total
  number of parameters … shouldn't be over 6"). Backbone uses 3, so modules
  add ≤3 params per candidate.
- **CHECKPOINT (user-confirmed):** after extraction, `compose-count` reports
  the cap- and compatibility-respecting candidate count to Akshay before any
  fitting. He picks exhaustive vs greedy (same rule as two-step).
- **Fitting protocol:** per-participant L-BFGS-B, uniform random starts,
  n_starts = 10, seeded RNG (`fitting.py` reused unchanged). Objective = mean
  BIC on composition-validation pids 4–13. Ties (< 1 BIC) → fewer params.
- `search_log.jsonl` + `selection_report.json` (top-10 + leave-one-pid-out
  rank stability) as before.

## Stage 2b — Library reconstruction on unseen participants

For each of the 5 reconstruction pids: search the library for that
individual's best module combination (same cap, seeding, and search mode) and
compare against (a) their individual-gecco refit (ceiling) and (b) the group
program refit. Reported separately (`reconstruction_results.json`), never
mixed into the single-program claim.

## Stage 3 — Generalization test

Winner frozen as `composed_model.txt`; all models fit per-participant on the
**11 final-test pids** under the identical seeded protocol:

| Model | Role |
|---|---|
| Composed program | The claim |
| Group gecco `best_model_0.txt` | Headline rival (same information budget) |
| Canonical RLWM (Collins & Frank 2012) baseline | Field standard; cross-check vs. `baseline_bic` column |
| Individual gecco per-participant programs | Ceiling reference (not a fair rival) |

The canonical baseline (delta-rule RL + one-shot WM with decay toward uniform
and set-size-scaled mixture weight; exact source frozen in
`canonical_rlwm.py`) mirrors two-step's Daw-hybrid role exactly, including
the cross-check of its refits against the data's stored `baseline_bic`
column (WARN on mismatch, investigate before reporting). Only the
**hybrid-base composition search** (`compose-hybrid-search` analog: canonical
model as fixed base + exhaustive search over missing library modules) is
deferred (user decision: main arms first).

**Stats:** mean BIC per model; per-participant ΔBIC vs group; win/tie/loss;
paired Wilcoxon (composed vs group, composed vs canonical). Validation
tables reported separately, labeled as selection data.

## Outputs

New dir `results/rlwm_individual/library_composition/`:
- `module_inventory.json`, `MODULES.md`, `mining_report.md`
- `llm_log/` (every Gemini call, verbatim, numbered)
- `splits.json` (with age balance stats)
- `search_log.jsonl`, `selection_report.json`
- `reconstruction_report.json` (seed fidelity), `reconstruction_results.json`
  (unseen-participant coverage)
- `composed_model.txt`, `winner.json`
- `test_results.json`, `test_results.csv`, `RESULTS.md`
- Comparison figure, paper style (teal = canonical RLWM reference,
  blue = composed winner, gray = others; PNG + PDF)

## Code placement — parallel RLWM modules (user decision)

Two-step modules stay byte-identical; RLWM gets parallel siblings in
`library_learning/compose/`:

| New file | Contents | Reuses via import (unmodified) |
|---|---|---|
| `render_rlwm.py` | RLWM backbone template, slot rendering, smoke data (2 blocks, set sizes 3+6, `-2` missed trials), `smoke_check` | `loading.exec_model`, bounds parsing |
| `inventory_rlwm.py` | RLWM `APPEND_SLOTS`/`OVERRIDE_SLOTS`/`BACKBONE_PARAM_NAMES` + `parse_inventory` | `inventory.Module/Param/Inventory/compatible/InventoryError` |
| `extract_rlwm.py` | RLWM annotation + merge prompts, RLWM `CANONICAL_NAMES`, orchestration | `extract._parse_json_reply`, `_target_names`, `gemini.GeminiClient`, `mining.py` |
| `splits_rlwm.py` | age-stratified partition, held-out ∩ fitted intersection | `splits.parse_split`, `splits.group_split_pids` |
| `search_rlwm.py` | enumeration/greedy with 3 backbone params, cap 6 | `fitting.py` |
| `reconstruct_rlwm.py` | per-pid library coverage | `fitting.py` |
| `canonical_rlwm.py` | Collins & Frank RLWM source + bounds (analog of `hybrid.py`) | — |
| `evaluate_rlwm.py` | composed/group/canonical/individual fits, stats, RESULTS.md | `fitting.py`, `loading.py` |

- `__main__.py` gains dispatch **by the target's resolved `task.name`**
  (`rlwm` → rlwm modules; anything else → existing two-step path). Existing
  subcommand defaults and two-step behavior unchanged; the only edit to an
  existing file is this additive wiring.
- `figure.py` reused if its labels prove task-agnostic, else a small
  `figure_rlwm.py`.
- Python 3.9, `gecco-env/` venv (numpy/pandas/scipy/pyyaml, `google-genai`
  already present from the two-step run).

## Verification

1. RLWM backbone renders, execs, and smoke-fits (empty candidate) on pid 1
   before any Gemini call; smoke data includes `-2` missed trials.
2. Backbone-only candidate behaves like a plain RL+WM mixture (sanity).
3. Seed reconstruction gate 8/8 before search.
4. Every rendered candidate smoke-checked before fitting; `compose-search`
   exits nonzero on any fit failure.
5. `splits.json` age balance reported (mean/std per subset).
6. Canonical RLWM refits cross-checked against the data's `baseline_bic`
   column (WARN on mismatch — the column's producing variant is unknown;
   investigate before reporting), mirroring two-step's hybrid cross-check.
7. Freeze discipline: `compose-eval` refuses to run without
   `composed_model.txt`.
8. Existing test suite passes unchanged (confirms two-step untouched);
   two-step `compose-*` subcommands still resolve their defaults.
9. `set -o pipefail` on any piped shell invocations (two-step lesson).

## Non-goals

- No hybrid-base composition search (`compose-hybrid-search` analog:
  canonical model as fixed base + missing-module search) — designated
  follow-up. The canonical baseline itself IS fit in Stage 3.
- No individual-gecco fitting of the 48 unfitted held-out pids; no
  evaluation on them this run.
- No task-adapter refactor of the two-step compose modules (explicitly
  declined in favor of parallel modules).
- No use of the `age` column as a model input (the `rlwm_individual_age.yaml`
  arm is a separate line of work).
- No per-participant model assignment / mixture across library programs.
