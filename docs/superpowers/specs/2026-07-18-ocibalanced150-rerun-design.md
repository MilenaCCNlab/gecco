# OCI-Balanced 150-Participant Rerun: Group + Individual GeCCo + Library Composition

**Date:** 2026-07-18
**Status:** Approved design, pending implementation plan
**Branch:** gecco-individual-differences
**Execution mode:** user delegated overnight autonomy (2026-07-18): reasonable
assumptions are made, logged in `docs/superpowers/specs/2026-07-18-ocibalanced150-decision-log.md`,
and execution continues without blocking; the `compose-count` checkpoint is
self-approved with logged thresholds. Final deliverable: full HTML report.

## Goal

Redo the complete two-step psychiatry analysis (group gecco, individual gecco,
library composition with bare-bone and hybrid-base arms) on a new 150-participant
OCI-stratified dataset built from the full Gillan 2016 study-1 data, replacing the
45-participant `two_step_gillan_2016_ocibalanced.csv` run. Reuse existing reviewed
code throughout; only configuration changes plus two approved code touches
(dataset script, participant-range argument on the individual runner).

## Key decisions (settled during brainstorming)

1. **Stratification: percentile tertiles, not the old fixed bins.** The old fixed
   OCI ranges (0-20 / 21-40 / 41-60) hold 396 / 125 / **19** of the 548 study-1
   participants — 50 per bin is impossible. Instead use the 33rd/66th percentile
   cuts (OCI 9 and 18, as in `data/ocd/select_participants_percentile.py`), giving
   tertiles of 201 / 176 / 171. Sample 50 per tertile, `random_state=42`.
2. **Split assignment: OCI-stratified random 50/50/50** (train / validation /
   test), not purely random. Within each tertile, shuffle (seeded) and deal with a
   rotated 17/17/16 pattern so each split totals exactly 50 with 16-17 per tertile.
3. **Participant indices are shuffled** — contiguous blocks (train 0-49,
   validation 50-99, test 100-149) with random order inside each block, so index
   order carries no OCI information (the old dataset's indices ran low-to-high OCI
   bin, confounding slice-based splits).
4. **Coverage (reconstruction) runs on the same 50 test pids** as the
   shared-program comparison. No leakage into the shared-program claim (winners are
   frozen on validation before test data is touched); trade-off accepted: the test
   set is no longer untouched by adaptive per-person search.
5. **Group gecco prompt: 5 OCI-stratified in-context participants** from train
   (2 low / 2 mid / 1 high, seeded random), listed explicitly in the config.
6. **Parallel artifacts:** new dataset file, configs, task names, and results
   directories alongside the completed 45-participant run; nothing overwritten.
7. **Individual gecco config base:**
   `two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting.yaml`
   (the variant with `individual_difference.individual_feature: None` — no OCI
   information in the prompt).
8. **Run independent lanes in parallel** wherever artifacts don't overlap.

## Section 1 — Dataset creation

**New script** `data/ocd/preprocess_data_150.py`, assembled from reviewed logic in
`select_participants_percentile.py` (sampling) and `preprocess_data.py` (cleaning,
combining, score merge, remapping, baseline hybrid fit). One new step: split and
index assignment.

Pipeline:

1. **Sample.** Load `self_report_study1.csv` (548 participants; all verified to
   have raw task files in `twostep_data_study1/`). Tertile cuts at
   `oci_total.quantile(0.33)` = 9 and `.quantile(0.66)` = 18. Sample 50 per
   tertile with `random_state=42` → 150 participants labeled Low/Medium/High.
2. **Assign splits (stratified random, seeded).** All random steps in this
   script (sampling, dealing, within-block shuffles, prompt-5 pick) use a single
   `numpy.random.default_rng(42)` / `random_state=42` lineage so the dataset is
   fully reproducible from the script alone. Rotated dealing:

   | tertile | train | validation | test |
   | ------- | ----- | ---------- | ---- |
   | Low     | 17    | 17         | 16   |
   | Medium  | 17    | 16         | 17   |
   | High    | 16    | 17         | 17   |

3. **Assign participant indices.** Shuffle order within each split block
   (seeded), then number train = 0-49, validation = 50-99, test = 100-149.
   Contiguous blocks keep gecco configs as simple slices; within-block shuffle plus
   stratified dealing removes any index↔OCI ordering.
4. **Pick the 5 prompt participants** from train: 2 Low, 2 Mid, 1 High (seeded
   random). Their indices go into the group config as an explicit list
   (`parse_split` accepts lists).
5. **Build the CSV** reusing existing stages unchanged: copy raw files, clean
   (`twostep_instruct_9` marker), combine with participant indices, merge
   `stai/sds/oci` scores, remap left/right → 0/1 and states 2/3 → 0/1, 0-base
   trials, normalize OCI by 60.
6. **Baseline.** Fit the Daw hybrid per participant (10 restarts, as before) →
   `baseline_bic` column. Column schema identical to the old CSV so all existing
   loaders work untouched.

Outputs:

- `data/two_step_gillan_2016_ocibalanced150.csv`
- `data/ocd/ocibalanced150_manifest.json` — subject_id ↔ participant index,
  tertile, split, prompt-5 (provenance; source for the library `splits.json`)

Verification: 150 participants; 50/50/50 per tertile; each split exactly 50 with
the 16-17 pattern; correlation(participant index, OCI) ≈ 0; `baseline_bic`
non-null for all; column set identical to the old dataset.

## Section 2 — GeCCo configs

Two new YAMLs, copied from the reviewed maxsetting configs with only these fields
changed:

**`config/two_step_psychiatry_group_ocd_maxsetting_150.yaml`**
(base: `two_step_psychiatry_group_ocd_maxsetting.yaml`)

- `task.name: two_step_psychiatry_group_function_ocibalanced150_maxsetting`
- `data.path: data/two_step_gillan_2016_ocibalanced150.csv`
- `data.splits.prompt:` explicit 5-element list from the manifest
- `data.splits.eval: "[50:100]"` (validation 50)
- `data.splits.test: "[100:]"` (held-out 50)
- Everything else byte-identical (gemini-3-pro-preview, temp 0.7, high reasoning,
  10 iterations, 3 models/iteration, guardrails, templates,
  `max_prompt_trials: 50`).

**`config/two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting_150.yaml`**
(base: `two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting.yaml`
— no OCI info in prompt)

- `task.name: two_step_psychiatry_individual_function_ocibalanced150_maxsetting`
- `data.path:` the new CSV
- Everything else identical, including `max_prompt_trials: 200` (full session
  in prompt; each participant fit on their own full data, same-data model
  discovery as in the previous run).

## Section 3 — Running gecco

**Group gecco** — no code changes:
`python scripts/two_step_psychiatry_group.py --config two_step_psychiatry_group_ocd_maxsetting_150.yaml`.
Prompt holds the 5 stratified in-context participants; search scores candidates on
the 50 validation participants; test 50 never touched during search. Results in
`results/two_step_psychiatry_group_function_ocibalanced150_maxsetting/`.

**Individual gecco** — all 150 participants, each fit independently. Approved code
touch: `scripts/two_step_individual_function.py` line 35 hardcodes
`df.participant.unique()[:14]`; replace with a `--participants START:END` argument
(default = all). Enables chunked parallel runs (e.g. `0:50`, `50:100`, `100:150`)
via `bash/run_gecco_individual.sh`.

**Scale:** individual arm ≈ 3.3× the 45-participant run (150 × 10 iterations ×
3 models ≈ 4,500 model evaluations + 1 simulation call each, gemini-3-pro-preview).

**Ordering / parallelism:** group gecco and individual chunks run concurrently.
Library extraction needs only the train-50 individual programs; reconstruction and
eval need the test-50 programs and the frozen group model.

## Section 4 — Library learning

All via the existing CLI, no code changes:
`python -m library_learning <cmd> --results-dir results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual --group-dir results/two_step_psychiatry_group_function_ocibalanced150_maxsetting`

**Splits file.** `make_splits` is NOT used (it derives seeds from the group
config's prompt+eval, which now spans two splits). Instead `splits.json` is
generated from the dataset manifest (final stage of the dataset script):

- `seed_pids`: 0-49 (train; extraction sources)
- `composition_validation_pids`: 50-99
- `reconstruction_pids`: 100-149 (coverage-on-test decision)
- `test_pids`: 100-149

`load_splits` reads this file; the downstream pipeline follows it.

Stages in dependency order:

1. `compose-modules` — Gemini extraction from the 50 train individual programs
   (12 seeds → 18 modules last time; expect a larger inventory), including the
   seed-reconstruction fidelity gate on all 50 seeds. Keys
   `GEMINI_API_KEY_LAKELAB`/`_COCOSCILAB`, model gemini-3.1-pro-preview, calls
   logged to `llm_log/`. *Needs: individual gecco done for train pids.*
2. `compose-count` — candidate-count checkpoint; user approves before scoring.
3. **Arm 1 (bare-bone):** `compose-search --mode greedy` — winner by mean BIC on
   the 50 validation pids, frozen to `composed_model.txt`.
4. **Arm 2 (hybrid base):** `compose-hybrid-search --param-cap 9` — exhaustive
   over modules missing from the Daw hybrid, same validation pids, frozen under
   `hybrid_base/`. Prints candidate count before scoring; if the larger library
   makes it explode, stop and decide (e.g. tighter param cap) rather than burn
   compute. Arms 1-2 independent → parallel.
5. `compose-reconstruct --mode greedy` — per-person coverage on the 50 test pids.
   *Needs: individual gecco for test pids (ceilings) + frozen group model
   (floor).* Independent of arms 1-2 → parallel with them.
6. `compose-eval --sets validation,test` — both frozen winners + group gecco +
   Daw hybrid on validation and test; cross-checks, `RESULTS.md`, figures.
   *Needs: everything above.*

**Scale:** validation scoring is per-candidate × 50 pids (was 10);
reconstruction is 50 per-person greedy searches (was 10). Pure fitting compute,
no LLM; parallelizable across pids/candidates at the process level.

## Section 5 — Deliverables & success criteria

New artifacts (parallel to the old run; nothing overwritten):

| artifact | path |
| -------- | ---- |
| dataset | `data/two_step_gillan_2016_ocibalanced150.csv` |
| provenance manifest | `data/ocd/ocibalanced150_manifest.json` |
| dataset script | `data/ocd/preprocess_data_150.py` |
| group config | `config/two_step_psychiatry_group_ocd_maxsetting_150.yaml` |
| individual config | `config/two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting_150.yaml` |
| group results | `results/two_step_psychiatry_group_function_ocibalanced150_maxsetting/` |
| individual results | `results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual/` |
| library artifacts | `<individual results>/library_composition/` |

Code touches (the only two): new dataset script; `--participants START:END` on
`scripts/two_step_individual_function.py`.

Reported (all on the 50 test participants):

- Shared-program comparison: bare-bone composed winner and hybrid-base composed
  winner vs group gecco vs Daw hybrid — mean BIC, paired tests, win/loss counts
  (same statistics as the previous `RESULTS.md`).
- Coverage: per-person best library composition vs that person's individual-gecco
  ceiling vs group gecco, 50/50 pids.
- Validation numbers reported alongside, labeled as selection data.

Success criteria (each stage verifiable before the next starts):

1. Dataset passes Section 1 verification.
2. Group gecco: 10 iterations complete, `best_model_0.txt` frozen.
3. Individual gecco: 150/150 participants with a best program (train-50
   completion unblocks extraction early).
4. Extraction: inventory validates; fidelity gate passes on all 50 seeds.
5. Both arms: winner frozen with selection report; candidate counts approved at
   the `compose-count` checkpoint.
6. Reconstruction: 50/50 test pids with library/individual/group BIC triplets.
7. `compose-eval` completes with cross-checks clean; `RESULTS.md` + figures
   generated.

## Out of scope

- Any refactor of gecco or library_learning internals (beyond the two approved
  touches).
- The paused compression pipeline (scan/verify wiring in
  `library_learning/__main__.py`).
- Changes to the completed 45-participant run or its artifacts.
