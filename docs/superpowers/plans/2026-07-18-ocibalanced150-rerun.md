# OCI-Balanced 150 Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 150-participant OCI-tertile-stratified Gillan-2016 dataset and rerun the full pipeline on it: group gecco, individual gecco (all 150), library extraction from the 50 train participants, bare-bone + hybrid-base composition arms selected on 50 validation participants, coverage + shared-program evaluation on 50 test participants, ending in a self-contained HTML report.

**Architecture:** Existing reviewed pipeline (gecco scripts + `library_learning` CLI) driven by new configs and a new dataset. Three approved code touches only: (1) new dataset script `data/ocd/preprocess_data_150.py`, (2) `--participants START:END` on `scripts/two_step_individual_function.py`, (3) `run_extraction` honors a pre-existing `splits.json` instead of unconditionally regenerating it (without this, extraction derives the wrong seed set from the group config and clobbers our hand-provisioned splits).

**Tech Stack:** Python (gecco-env venv), pandas/numpy/scipy, Gemini API (gemini-3-pro-preview for gecco, gemini-3.1-pro-preview for extraction), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-18-ocibalanced150-rerun-design.md`. Decision log (append every autonomous call): `docs/superpowers/specs/2026-07-18-ocibalanced150-decision-log.md`.
- Only the three code touches above; everything else config/data/artifacts.
- All randomness in the dataset script from `numpy.random.default_rng(42)` / `random_state=42`.
- Never modify `data/two_step_gillan_2016_ocibalanced.csv`, old configs, or old results dirs.
- Python: `gecco-env/bin/python` from repo root. Repo root: `/Users/akshay/projects/gecco`.
- API keys: `.env` has `GEMINI_API_KEY_LAKELAB`, `GEMINI_API_KEY_COCOSCILAB`. gecco jobs need `GEMINI_API_KEY` exported from one of these at launch. Extraction (`GeminiClient`) reads `GEMINI_API_KEY_LAKELAB` itself.
- Long jobs run in background with logs under `logs/` (create dir if missing); on failure (429s, crashes), resume by re-launching the narrowed `--participants` range — completed participants' outputs are on disk and are NOT redone only if the relaunched range excludes them (the script has no skip logic).
- New task names: group `two_step_psychiatry_group_function_ocibalanced150_maxsetting`, individual `two_step_psychiatry_individual_function_ocibalanced150_maxsetting` (results dir gains `_individual`).
- IND=`results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual`, GRP=`results/two_step_psychiatry_group_function_ocibalanced150_maxsetting` (shell vars used throughout).

---

### Task 1: Dataset script + structural verification test

**Files:**
- Create: `data/ocd/preprocess_data_150.py`
- Test: `tests/test_ocibalanced150_dataset.py`

**Interfaces:**
- Produces: `data/two_step_gillan_2016_ocibalanced150.csv` (columns identical to `data/two_step_gillan_2016_ocibalanced.csv`), `data/ocd/ocibalanced150_manifest.json` with schema `{"seed": 42, "cuts": {"low": <q33>, "high": <q66>}, "participants": [{"participant": int 0-149, "subject_id": str, "oci_total": int, "tertile": "Low|Medium|High", "split": "train|validation|test", "prompt": bool}]}`.
- Train = participants 0–49, validation = 50–99, test = 100–149 (by construction).

- [ ] **Step 1: Write the failing structural test**

```python
# tests/test_ocibalanced150_dataset.py
import json
from pathlib import Path

import pandas as pd
import pytest
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data" / "two_step_gillan_2016_ocibalanced150.csv"
OLD_CSV = REPO / "data" / "two_step_gillan_2016_ocibalanced.csv"
MANIFEST = REPO / "data" / "ocd" / "ocibalanced150_manifest.json"

pytestmark = pytest.mark.skipif(not CSV.exists(), reason="dataset not built yet")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


def test_participants_0_to_149(df):
    pids = sorted(df["participant"].unique())
    assert pids == list(range(150))


def test_schema_matches_old_dataset(df):
    old_cols = list(pd.read_csv(OLD_CSV, nrows=1).columns)
    assert list(df.columns) == old_cols


def test_manifest_tertiles_and_splits(manifest):
    rows = manifest["participants"]
    assert len(rows) == 150
    tert = pd.Series([r["tertile"] for r in rows]).value_counts()
    assert tert.to_dict() == {"Low": 50, "Medium": 50, "High": 50}
    split = pd.Series([r["split"] for r in rows]).value_counts()
    assert split.to_dict() == {"train": 50, "validation": 50, "test": 50}
    # stratified dealing: every (split, tertile) cell is 16 or 17
    cells = pd.DataFrame(rows).groupby(["split", "tertile"]).size()
    assert set(cells.tolist()) <= {16, 17}
    # contiguous index blocks
    by_split = {s: sorted(r["participant"] for r in rows if r["split"] == s)
                for s in ["train", "validation", "test"]}
    assert by_split["train"] == list(range(0, 50))
    assert by_split["validation"] == list(range(50, 100))
    assert by_split["test"] == list(range(100, 150))


def test_no_serial_oci_confound(manifest):
    rows = sorted(manifest["participants"], key=lambda r: r["participant"])
    rho, _ = spearmanr([r["participant"] for r in rows],
                       [r["oci_total"] for r in rows])
    assert abs(rho) < 0.15


def test_prompt_five_stratified(manifest):
    prompt = [r for r in manifest["participants"] if r["prompt"]]
    assert len(prompt) == 5
    assert all(r["split"] == "train" for r in prompt)
    tert = pd.Series([r["tertile"] for r in prompt]).value_counts().to_dict()
    assert tert == {"Low": 2, "Medium": 2, "High": 1}


def test_oci_normalized_and_trials_zero_based(df):
    assert df["oci"].between(0, 1).all()
    assert (df.groupby("participant")["trial"].min() == 0).all()


def test_baseline_bic_present(df):
    assert "baseline_bic" in df.columns
    assert df["baseline_bic"].notna().all()
```

- [ ] **Step 2: Run test — expect all skipped (dataset absent)**

Run: `gecco-env/bin/python -m pytest tests/test_ocibalanced150_dataset.py -v`
Expected: all tests SKIPPED ("dataset not built yet").

- [ ] **Step 3: Write the dataset script**

```python
# data/ocd/preprocess_data_150.py
"""Build data/two_step_gillan_2016_ocibalanced150.csv.

150 participants from Gillan 2016 study 1, OCI-tertile-stratified (33rd/66th
percentile cuts, 50 per tertile, sampling logic from
select_participants_percentile.py), with OCI-stratified random assignment to
train/validation/test blocks (participant indices 0-49/50-99/100-149, shuffled
within block) and per-participant Daw-hybrid baseline BICs (logic from
preprocess_data.py). Spec: docs/superpowers/specs/2026-07-18-ocibalanced150-rerun-design.md

Usage:
  python preprocess_data_150.py                 # sample + build (no baseline)
  python preprocess_data_150.py --baseline-only # fit baselines, atomic-replace CSV
"""
import argparse
import json
import os
from io import StringIO

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SELF_REPORT = os.path.join(HERE, "self_report_study1.csv")
RAW_DIR = os.path.join(HERE, "twostep_data_study1")
MANIFEST = os.path.join(HERE, "ocibalanced150_manifest.json")
OUT_CSV = os.path.join(REPO, "data", "two_step_gillan_2016_ocibalanced150.csv")

N_PER_TERTILE = 50
# rotated 17/17/16 dealing so each split totals 50 with 16-17 per tertile
DEAL = {"Low": (17, 17, 16), "Medium": (17, 16, 17), "High": (16, 17, 17)}
PROMPT_PER_TERTILE = {"Low": 2, "Medium": 2, "High": 1}

# raw-file column names (from preprocess_data.py)
COLUMN_NAMES = [
    "trial_num", "drift_1", "drift_2", "drift_3", "drift_4",
    "stage_1_response", "stage_1_selected_stimulus", "stage_1_rt",
    "transition", "stage_2_response", "stage_2_selected_stimulus",
    "stage_2_state", "stage_2_rt", "reward", "redundant",
]


def sample_and_assign():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(SELF_REPORT)
    low_cut = df["oci_total"].quantile(0.33)
    high_cut = df["oci_total"].quantile(0.66)
    groups = {
        "Low": df[df["oci_total"] <= low_cut],
        "Medium": df[(df["oci_total"] > low_cut) & (df["oci_total"] <= high_cut)],
        "High": df[df["oci_total"] > high_cut],
    }
    sampled = {}
    for tert, g in groups.items():
        assert len(g) >= N_PER_TERTILE, "tertile %s has %d < %d" % (tert, len(g), N_PER_TERTILE)
        sampled[tert] = g.sample(n=N_PER_TERTILE, random_state=SEED)

    splits = {"train": [], "validation": [], "test": []}
    for tert in ["Low", "Medium", "High"]:
        rows = sampled[tert][["subj.x", "oci_total"]].values.tolist()
        rng.shuffle(rows)
        n_tr, n_va, n_te = DEAL[tert]
        for row in rows[:n_tr]:
            splits["train"].append((row[0], int(row[1]), tert))
        for row in rows[n_tr:n_tr + n_va]:
            splits["validation"].append((row[0], int(row[1]), tert))
        for row in rows[n_tr + n_va:]:
            splits["test"].append((row[0], int(row[1]), tert))

    participants = []
    idx = 0
    for split in ["train", "validation", "test"]:
        block = splits[split][:]
        rng.shuffle(block)
        for subj, oci, tert in block:
            participants.append({"participant": idx, "subject_id": str(subj),
                                 "oci_total": oci, "tertile": tert,
                                 "split": split, "prompt": False})
            idx += 1

    # prompt 5 from train: 2 Low / 2 Medium / 1 High
    train = [p for p in participants if p["split"] == "train"]
    for tert, k in PROMPT_PER_TERTILE.items():
        pool = [p for p in train if p["tertile"] == tert]
        for p in rng.choice(len(pool), size=k, replace=False):
            pool[p]["prompt"] = True

    manifest = {"seed": SEED, "cuts": {"low": float(low_cut), "high": float(high_cut)},
                "participants": participants}
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print("manifest -> %s (cuts %.1f / %.1f)" % (MANIFEST, low_cut, high_cut))
    return manifest


def clean_raw_file(subj_id):
    """Marker-based cleaning from preprocess_data.py, in memory."""
    path = os.path.join(RAW_DIR, "%s.csv" % subj_id)
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, line in enumerate(lines) if "twostep_instruct_9" in line)
    return pd.read_csv(StringIO("".join(lines[start + 1:])), header=None,
                       names=COLUMN_NAMES)


def build(manifest):
    survey = pd.read_csv(SELF_REPORT)[["subj.x", "stai_total", "sds_total", "oci_total"]]
    survey = survey.rename(columns={"subj.x": "subject_id"})
    survey["subject_id"] = survey["subject_id"].astype(str)

    frames = []
    for p in manifest["participants"]:
        d = clean_raw_file(p["subject_id"])
        d["subject_id"] = p["subject_id"]
        d["participant"] = p["participant"]
        cols = ["participant", "subject_id"] + [c for c in d.columns
                                                if c not in ("participant", "subject_id")]
        frames.append(d[cols])
    df = pd.concat(frames, ignore_index=True)
    df = pd.merge(df, survey, on="subject_id", how="left")
    assert df["oci_total"].notna().all(), "score merge produced NaNs"

    # remap + rename (from preprocess_data.py)
    df["trial_num"] = df["trial_num"] - 1
    df["stage_1_response"] = df["stage_1_response"].replace({"left": 0, "right": 1})
    df["stage_2_response"] = df["stage_2_response"].replace({"left": 0, "right": 1})
    df["stage_2_state"] = df["stage_2_state"].replace({2: 0, 3: 1})
    df = df.rename(columns={"stage_1_response": "choice_1",
                            "stage_2_response": "choice_2",
                            "stage_2_state": "state", "stai_total": "stai",
                            "sds_total": "sds", "oci_total": "oci",
                            "trial_num": "trial"})
    df["oci"] = df["oci"] / 60.0
    df.to_csv(OUT_CSV, index=False)
    print("dataset -> %s (%d rows, %d participants)"
          % (OUT_CSV, len(df), df["participant"].nunique()))


def hybrid_model(action_1, state, action_2, reward, model_parameters):
    """Verbatim from preprocess_data.py (Daw hybrid with eligibility traces +
    perseveration)."""
    learning_rate, learning_rate_2, beta, beta_2, w, lambd, perseveration = model_parameters
    n_trials = len(action_1)
    transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    prev_action_indicator = np.zeros(2)
    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)
    q_stage1_mf = np.zeros(2)
    q_stage2_mf = np.zeros((2, 2))
    for trial in range(n_trials):
        max_q_stage2 = np.max(q_stage2_mf, axis=1)
        q_stage1_mb = transition_matrix @ max_q_stage2
        q_stage1_combined = w * q_stage1_mb + (1 - w) * q_stage1_mf
        q_stage1_with_pers = q_stage1_combined + perseveration * prev_action_indicator
        exp_q1 = np.exp(beta * q_stage1_with_pers)
        probs_1 = exp_q1 / np.sum(exp_q1)
        p_choice_1[trial] = probs_1[action_1[trial]]
        state_idx = state[trial]
        exp_q2 = np.exp(beta_2 * q_stage2_mf[state_idx])
        probs_2 = exp_q2 / np.sum(exp_q2)
        p_choice_2[trial] = probs_2[action_2[trial]]
        delta_stage1 = q_stage2_mf[state_idx, action_2[trial]] - q_stage1_mf[action_1[trial]]
        q_stage1_mf[action_1[trial]] += learning_rate * delta_stage1
        delta_stage2 = reward[trial] - q_stage2_mf[state_idx, action_2[trial]]
        q_stage2_mf[state_idx, action_2[trial]] += learning_rate_2 * delta_stage2
        q_stage1_mf[action_1[trial]] += lambd * learning_rate * delta_stage2
        prev_action_indicator.fill(0)
        prev_action_indicator[action_1[trial]] = 1
    eps = 1e-10
    return -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))


def fit_hybrid(choice1, state, choice2, reward, n_trials):
    """Verbatim fitting protocol from preprocess_data.py (10 restarts, L-BFGS-B)."""
    nreps = 10
    best = np.inf
    bounds = [[0, 1], [0, 1], [0.1, 10], [0.1, 10], [0, 1], [0, 1], [0, 1]]
    for _ in range(nreps):
        x0 = [np.random.uniform(lo, hi) for lo, hi in bounds]
        res = minimize(lambda p: hybrid_model(choice1, state, choice2, reward, p),
                       x0, method="L-BFGS-B", bounds=bounds)
        best = min(best, res.fun)
    bic = 2 * best + len(bounds) * np.log(n_trials)
    if np.isinf(bic) or np.isnan(bic):
        bic = -4 * np.log(0.5) * len(choice1)
    return bic


def add_baselines():
    df = pd.read_csv(OUT_CSV)
    if "baseline_bic" in df.columns and df["baseline_bic"].notna().all():
        print("baseline_bic already complete; nothing to do")
        return
    np.random.seed(SEED)
    for pid, d in df.groupby("participant"):
        bic = fit_hybrid(d["choice_1"].to_numpy(), d["state"].to_numpy(),
                         d["choice_2"].to_numpy(), d["reward"].to_numpy(),
                         len(d))
        df.loc[df["participant"] == pid, "baseline_bic"] = bic
        print("participant %d baseline BIC %.2f" % (pid, bic))
    tmp = OUT_CSV + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, OUT_CSV)   # atomic: concurrent readers see old or new file
    print("baseline_bic added -> %s" % OUT_CSV)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()
    if args.baseline_only:
        add_baselines()
    else:
        manifest = sample_and_assign()
        build(manifest)
        print("now run with --baseline-only (background) to add baseline_bic")
```

- [ ] **Step 4: Run sample+build stage**

Run: `cd /Users/akshay/projects/gecco/data/ocd && ../../gecco-env/bin/python preprocess_data_150.py`
Expected: `manifest -> .../ocibalanced150_manifest.json (cuts 9.0 / 18.0)`, then `dataset -> .../two_step_gillan_2016_ocibalanced150.csv (~30000 rows, 150 participants)`.

- [ ] **Step 5: Launch baseline fitting in background** (do NOT wait; needed only by Task 9's compose-eval)

Run (background): `cd /Users/akshay/projects/gecco/data/ocd && ../../gecco-env/bin/python preprocess_data_150.py --baseline-only > ../../logs/baseline150.log 2>&1`
Expected: per-participant lines in the log; atomic replace at the end.

- [ ] **Step 6: Run structural tests (baseline test may still fail — rerun that one after Step 5 completes)**

Run: `gecco-env/bin/python -m pytest tests/test_ocibalanced150_dataset.py -v`
Expected: all PASS except `test_baseline_bic_present` (FAILS until the background baseline job finishes; re-run it then).

- [ ] **Step 7: Commit**

```bash
git add data/ocd/preprocess_data_150.py tests/test_ocibalanced150_dataset.py data/ocd/ocibalanced150_manifest.json data/two_step_gillan_2016_ocibalanced150.csv
git commit -m "data: ocibalanced150 dataset (150 pids, OCI-tertile stratified, shuffled split blocks)"
```

---

### Task 2: New gecco configs

**Files:**
- Create: `config/two_step_psychiatry_group_ocd_maxsetting_150.yaml` (copy of `config/two_step_psychiatry_group_ocd_maxsetting.yaml`)
- Create: `config/two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting_150.yaml` (copy of `config/two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting.yaml`)

**Interfaces:**
- Consumes: manifest prompt pids from Task 1.
- Produces: task names `two_step_psychiatry_group_function_ocibalanced150_maxsetting` / `two_step_psychiatry_individual_function_ocibalanced150_maxsetting` — the results dirs all later tasks reference.

- [ ] **Step 1: Get the prompt pid list from the manifest**

Run:
```bash
gecco-env/bin/python -c "
import json
m = json.load(open('data/ocd/ocibalanced150_manifest.json'))['participants']
print(sorted(r['participant'] for r in m if r['prompt']))"
```
Expected: a 5-element list of ints in 0–49 (e.g. `[7, 13, 22, 38, 41]`; use the actual values below).

- [ ] **Step 2: Create both configs**

Copy each base file, then change ONLY these fields (all other lines byte-identical):

Group config (`two_step_psychiatry_group_ocd_maxsetting_150.yaml`):
```yaml
task:
  name: "two_step_psychiatry_group_function_ocibalanced150_maxsetting"
data:
  path: "data/two_step_gillan_2016_ocibalanced150.csv"
  splits:
    prompt: [7, 13, 22, 38, 41]   # ACTUAL manifest prompt pids from Step 1
    eval: "[50:100]"
    test: "[100:]"
```

Individual config (`two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting_150.yaml`):
```yaml
task:
  name: "two_step_psychiatry_individual_function_ocibalanced150_maxsetting"
data:
  path: "data/two_step_gillan_2016_ocibalanced150.csv"
```

- [ ] **Step 3: Verify configs parse and splits resolve**

Run:
```bash
gecco-env/bin/python -c "
import sys; sys.path.insert(0, '.')
from config.schema import load_config
from gecco.prepare_data.io import load_data, split_by_participant
for name in ['two_step_psychiatry_group_ocd_maxsetting_150.yaml',
             'two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting_150.yaml']:
    cfg = load_config('config/' + name)
    print(name, '->', cfg.task.name)
df = load_data('data/two_step_gillan_2016_ocibalanced150.csv', ['choice_1','state','choice_2','reward'])
cfg = load_config('config/two_step_psychiatry_group_ocd_maxsetting_150.yaml')
s = split_by_participant(df, 'participant', cfg.data.splits)
print('prompt', sorted(s['prompt'].participant.unique()))
print('eval n =', s['eval'].participant.nunique(), 'test n =', s['test'].participant.nunique())"
```
Expected: both task names print; prompt = the 5 manifest pids; eval n = 50; test n = 50.

- [ ] **Step 4: Commit**

```bash
git add config/two_step_psychiatry_group_ocd_maxsetting_150.yaml config/two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting_150.yaml
git commit -m "config: ocibalanced150 group + individual maxsetting configs"
```

---

### Task 3: `--participants START:END` on the individual runner

**Files:**
- Modify: `scripts/two_step_individual_function.py:16-17` (argparse) and `:35` (loop)

**Interfaces:**
- Produces: `gecco-env/bin/python scripts/two_step_individual_function.py --config <yaml> --participants 0:25` — positional slice over `df.participant.unique()` (appearance order = pid order 0–149). Default (flag omitted) = ALL participants.

- [ ] **Step 1: Apply the edit**

Replace (line 16-17 area):
```python
    parser.add_argument('--config', type=str, default='', help='*REQUIRED* config.yaml')
    args = parser.parse_args()
```
with:
```python
    parser.add_argument('--config', type=str, default='', help='*REQUIRED* config.yaml')
    parser.add_argument('--participants', type=str, default=None,
                        help='START:END positional slice over participants (default: all)')
    args = parser.parse_args()
```

Replace (line 35):
```python
        for participant in df.participant.unique()[:14]:
```
with:
```python
        participants = df.participant.unique()
        if args.participants:
            start_str, end_str = args.participants.split(":")
            participants = participants[int(start_str) if start_str else None:
                                        int(end_str) if end_str else None]
        for participant in participants:
```

- [ ] **Step 2: Verify with a no-op range (loads config+data, executes zero participants, exits cleanly)**

Run: `gecco-env/bin/python scripts/two_step_individual_function.py --config two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting_150.yaml --participants 0:0`
Expected: exits 0 quickly, no LLM calls, no output files.

- [ ] **Step 3: Commit**

```bash
git add scripts/two_step_individual_function.py
git commit -m "feat(scripts): --participants START:END range on individual runner (was hardcoded [:14])"
```

---

### Task 4: `run_extraction` honors pre-existing splits.json

**Files:**
- Modify: `library_learning/compose/extract.py:317-321`
- Test: `tests/test_resolve_splits.py`

**Interfaces:**
- Produces: `resolve_splits(target, group_dir, out_dir) -> dict` in `library_learning/compose/extract.py` — returns parsed `out_dir/splits.json` if present, else falls through to `make_splits`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resolve_splits.py
import json

import pytest

from library_learning.compose import extract


def test_existing_splits_json_wins(tmp_path, monkeypatch):
    splits = {"seed_pids": [0, 1], "composition_validation_pids": [2],
              "reconstruction_pids": [3], "test_pids": [3]}
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    monkeypatch.setattr(extract, "make_splits",
                        lambda *a, **k: pytest.fail("make_splits must not run"))
    assert extract.resolve_splits(None, None, tmp_path) == splits


def test_falls_back_to_make_splits(tmp_path, monkeypatch):
    sentinel = {"seed_pids": [9]}
    monkeypatch.setattr(extract, "make_splits", lambda *a, **k: sentinel)
    assert extract.resolve_splits(None, None, tmp_path) is sentinel
```

- [ ] **Step 2: Run test to verify it fails**

Run: `gecco-env/bin/python -m pytest tests/test_resolve_splits.py -v`
Expected: FAIL — `extract` has no attribute `resolve_splits`.

- [ ] **Step 3: Implement**

In `library_learning/compose/extract.py`, add above `run_extraction`:
```python
def resolve_splits(target, group_dir, out_dir):
    """Honor a pre-provisioned splits.json (e.g. manifest-derived splits that
    make_splits cannot express); otherwise derive from the group config."""
    path = Path(out_dir) / "splits.json"
    if path.exists():
        return json.loads(path.read_text())
    return make_splits(target, group_dir, out_dir=out_dir)
```
and inside `run_extraction` replace:
```python
    splits = make_splits(target, group_dir, out_dir=out_dir)
```
with:
```python
    splits = resolve_splits(target, group_dir, out_dir)
```

- [ ] **Step 4: Run tests (new + existing compose suite)**

Run: `gecco-env/bin/python -m pytest tests/test_resolve_splits.py tests/test_compose_splits.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add library_learning/compose/extract.py tests/test_resolve_splits.py
git commit -m "fix(compose): run_extraction honors pre-existing splits.json"
```

---

### Task 5: Launch group gecco + individual gecco (all 150) in background

**Files:** none (operational). Logs: `logs/group150.log`, `logs/ind150_<range>.log`.

**Interfaces:**
- Consumes: Task 2 configs, Task 3 flag.
- Produces: `$GRP/models/best_model_0.txt` (+ bics incl. `best_bic_on_test_run0.json`); `$IND/models/best_model_0_participant{0..149}.txt` (+ bics/parameters per pid).

- [ ] **Step 1: Create logs dir, load keys**

```bash
mkdir -p logs
set -a; source .env; set +a
```

- [ ] **Step 2: Launch group gecco (background)**

```bash
GEMINI_API_KEY="$GEMINI_API_KEY_LAKELAB" gecco-env/bin/python scripts/two_step_psychiatry_group.py \
  --config two_step_psychiatry_group_ocd_maxsetting_150.yaml > logs/group150.log 2>&1
```

- [ ] **Step 3: Launch six individual chunks (background), keys alternating**

```bash
for spec in "0:25 LAKELAB" "25:50 COCOSCILAB" "50:75 LAKELAB" "75:100 COCOSCILAB" "100:125 LAKELAB" "125:150 COCOSCILAB"; do
  range=${spec% *}; key=${spec#* }
  keyval=$(eval echo "\$GEMINI_API_KEY_$key")
  GEMINI_API_KEY="$keyval" gecco-env/bin/python scripts/two_step_individual_function.py \
    --config two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting_150.yaml \
    --participants "$range" > "logs/ind150_${range/:/_}.log" 2>&1 &
done
```
(Executor note: launch each as a separate tracked background job rather than a shell `&` loop, so exits are observable.)

- [ ] **Step 4: Monitor loop (repeat until complete; also drives Tasks 6–9 gating)**

Progress: `ls $IND/models 2>/dev/null | grep -c "best_model_0_participant"` (target 150; train-gate = pids 0–49 all present, test-gate = 100–149).
Health: `tail -3 logs/ind150_*.log logs/group150.log`; a dead chunk (log stalled >30 min or process gone) is relaunched with its remaining range, e.g. finished through pid 37 in chunk 25:50 → relaunch `--participants 38:50`. Log every restart in the decision log.

---

### Task 6: Provision library splits.json + run extraction (gated on train 0–49 complete)

**Files:** creates `$IND/library_composition/splits.json`, then extraction artifacts (`module_inventory.json`, `annotations.json`, `mining_report.md`, `MODULES.md`, `llm_log/`).

**Interfaces:**
- Consumes: manifest (Task 1); `$IND/models/best_model_0_participant{0..49}.txt` (Task 5); `resolve_splits` (Task 4).
- Produces: validated inventory for Tasks 7–9; frozen splits for all compose stages.

- [ ] **Step 1: Write splits.json from the manifest**

```bash
gecco-env/bin/python -c "
import json, pathlib
m = json.load(open('data/ocd/ocibalanced150_manifest.json'))['participants']
by = lambda s: sorted(r['participant'] for r in m if r['split'] == s)
out = pathlib.Path('results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual/library_composition')
out.mkdir(parents=True, exist_ok=True)
splits = {'seed_pids': by('train'),
          'composition_validation_pids': by('validation'),
          'reconstruction_pids': by('test'),
          'test_pids': by('test'),
          'method': 'ocibalanced150 manifest: train=seeds, validation=composition, test=coverage+report (spec 2026-07-18)'}
(out / 'splits.json').write_text(json.dumps(splits, indent=2))
print('seed', len(splits['seed_pids']), 'val', len(splits['composition_validation_pids']), 'test', len(splits['test_pids']))"
```
Expected: `seed 50 val 50 test 50`.

- [ ] **Step 2: Run extraction (LLM; logged)**

```bash
gecco-env/bin/python -m library_learning compose-modules \
  --results-dir $IND --group-dir $GRP > logs/compose_modules150.log 2>&1
```
Expected: `extracted N modules -> .../library_composition` (N likely > 18). Fidelity gate runs over all 50 seeds inside extraction; failures raise.

- [ ] **Step 3: Verify splits.json untouched and inventory valid**

```bash
gecco-env/bin/python -c "
import json
s = json.load(open('$IND/library_composition/splits.json'))
assert s['seed_pids'] == list(range(50)) and s['test_pids'] == list(range(100, 150)), s"
gecco-env/bin/python -m library_learning compose-modules --results-dir $IND --group-dir $GRP --skip-llm
```
Expected: assertion silent; `inventory valid: N modules`.

- [ ] **Step 4: Commit extraction artifacts**

```bash
git add $IND/library_composition
git commit -m "results: ocibalanced150 module extraction (50 seeds)"
```

---

### Task 7: Composition arms (bare-bone greedy + hybrid-base exhaustive), parallel

**Files:** creates `$IND/library_composition/{winner.json,composed_model.txt,selection_report.json,search_log.jsonl}` and `hybrid_base/` equivalents.

**Interfaces:**
- Consumes: inventory + splits (Task 6).
- Produces: two frozen winners consumed by Task 9.

- [ ] **Step 1: Candidate-count checkpoint (self-approved per decision log #2)**

```bash
gecco-env/bin/python -m library_learning compose-count --results-dir $IND --group-dir $GRP
```
Record the counts in the decision log. Greedy arm proceeds regardless. For the hybrid arm: estimate time = candidates × 50 pids × observed per-fit seconds (measure one candidate fit first if in doubt); if estimate > 12 h, rerun `compose-hybrid-search` with `--param-cap 8` instead, and log the decision.

- [ ] **Step 2: Launch both arms as parallel background jobs**

```bash
gecco-env/bin/python -m library_learning compose-search --mode greedy \
  --results-dir $IND --group-dir $GRP > logs/compose_search150.log 2>&1
gecco-env/bin/python -m library_learning compose-hybrid-search --param-cap 9 \
  --results-dir $IND --group-dir $GRP > logs/compose_hybrid150.log 2>&1
```
Expected (each log): `WINNER <candidate_id> mean validation BIC <x> -> ... frozen`.

- [ ] **Step 3: Commit both frozen winners**

```bash
git add $IND/library_composition
git commit -m "results: ocibalanced150 composition winners (bare-bone greedy + hybrid-base)"
```

---

### Task 8: Coverage (reconstruction) on the 50 test pids (gated on test 100–149 individual programs + group model)

**Files:** creates `$IND/library_composition/reconstruction_results.json` + per-pid dirs under `reconstruction/`.

**Interfaces:**
- Consumes: inventory, splits (`reconstruction_pids` = 100–149), `$GRP/models/best_model_0.txt`, `$IND/models/best_model_0_participant{100..149}.txt`.
- Produces: per-pid `{library_bic, individual_bic, group_bic}` triplets for the report.

- [ ] **Step 1: Run (background; independent of Task 7 — may run concurrently)**

```bash
gecco-env/bin/python -m library_learning compose-reconstruct --mode greedy \
  --results-dir $IND --group-dir $GRP > logs/compose_reconstruct150.log 2>&1
```
Expected: 50 `[reconstruct] p<pid> library <x> vs individual <y> vs group <z>` lines.

- [ ] **Step 2: Verify count and commit**

```bash
gecco-env/bin/python -c "
import json
r = json.load(open('$IND/library_composition/reconstruction_results.json'))
assert len(r) == 50, len(r); print('coverage rows:', len(r))"
git add $IND/library_composition
git commit -m "results: ocibalanced150 per-participant coverage on 50 test pids"
```

---

### Task 9: Final evaluation — both arms on validation + test (gated on: all 150 individual programs, group model, both winners, baseline_bic column complete)

**Files:** creates `test_results.{json,csv}`, `RESULTS.md`, `comparison.png/pdf` in `$IND/library_composition/` and in `hybrid_base/`.

**Interfaces:**
- Consumes: everything above. Verify baseline first: `gecco-env/bin/python -m pytest tests/test_ocibalanced150_dataset.py::test_baseline_bic_present -v` must PASS (else wait for/rerun the Task 1 Step 5 job).
- Produces: the stats consumed by the report (Task 10).

- [ ] **Step 1: Bare-bone arm eval (built-in CLI)**

```bash
gecco-env/bin/python -m library_learning compose-eval --sets validation,test \
  --results-dir $IND --group-dir $GRP > logs/compose_eval150.log 2>&1
```
Expected: `results -> .../RESULTS.md (warnings: <n>)`; investigate any warning lines in RESULTS.md (tolerance 5 BIC on group/hybrid refits).

- [ ] **Step 2: Hybrid-base arm eval (same functions, hybrid_base out_dir — mirrors previous run)**

```bash
gecco-env/bin/python - <<'EOF' > logs/compose_eval150_hybrid.log 2>&1
import json
from pathlib import Path
from library_learning.config import resolve_target
from library_learning.compose.evaluate import cross_checks, evaluate_models, summarize
from library_learning.compose.figure import plot_comparison
from library_learning.compose.splits import load_splits

IND = "results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual"
GRP = "results/two_step_psychiatry_group_function_ocibalanced150_maxsetting"
target = resolve_target(IND)
hb = Path(IND) / "library_composition" / "hybrid_base"
s = load_splits(target)
rv = evaluate_models(target, GRP, hb, s["composition_validation_pids"], "validation")
rt = evaluate_models(target, GRP, hb, s["test_pids"], "test")
heldout = sorted(set(s["reconstruction_pids"]) | set(s["test_pids"]))
warn = cross_checks(rt, target, GRP, s["test_pids"], heldout_pids=heldout)
summarize(rv, rt, warn, hb, target=target)
plot_comparison(hb / "test_results.csv", hb)
print("hybrid-base eval done, warnings:", len(warn))
EOF
```
Expected: `hybrid-base eval done, warnings: <n>`; `hybrid_base/RESULTS.md` + figures exist.

- [ ] **Step 3: Commit**

```bash
git add $IND/library_composition data/two_step_gillan_2016_ocibalanced150.csv
git commit -m "results: ocibalanced150 final evaluation (both arms, validation+test)"
```
(The CSV is re-added because the baseline job rewrote it with `baseline_bic`.)

---

### Task 10: HTML report + memory + wrap-up

**Files:**
- Create: `$IND/library_composition/report.html` (+ `report_fig*.png`)
- Update: decision log; memory file `library-composition-run.md` (or a new `ocibalanced150-run.md`).

**Interfaces:**
- Consumes: `test_results.json` (both arms), `selection_report.json`(s), `reconstruction_results.json`, `splits.json`, manifest, group/individual run stats.

- [ ] **Step 1: Generate report figures** — follow the previous run's four-figure pattern (test-BIC bars per model; coverage scatter library vs individual BIC; per-pid ΔBIC (composed − group) waterfall; hybrid-base ΔBIC waterfall), rendered with matplotlib from `test_results.csv` + `reconstruction_results.json` into `report_fig1..4.png`. Use the paper figure style memory (teal reference / blue winner / gray others) and consult the dataviz skill before writing the plotting code.

- [ ] **Step 2: Author `report.html`** — self-contained page mirroring the previous run's `report.html` structure (224-line single file, relative-path `<img>` tags), with sections: (1) dataset construction (tertile cuts, stratified splits, shuffle — from manifest), (2) pipeline summary + splits, (3) group gecco result, (4) individual gecco summary stats (mean/median best BIC across 150), (5) bare-bone arm: winner modules, validation BIC, test comparison + Wilcoxon stats, (6) hybrid-base arm: same, (7) coverage: 50-pid triplets, wins vs group, matches vs individual ceiling, (8) cross-check warnings, (9) decision log summary. All numbers read from the artifacts — no hand-typed values.

- [ ] **Step 3: Run the full test suite once more**

Run: `gecco-env/bin/python -m pytest tests/ -v`
Expected: all PASS (including baseline column test).

- [ ] **Step 4: Commit + update memory**

```bash
git add $IND/library_composition docs/superpowers/specs/2026-07-18-ocibalanced150-decision-log.md
git commit -m "results: ocibalanced150 self-contained HTML report"
```
Update memory: new `ocibalanced150-run.md` memory file with headline numbers + artifact locations; add MEMORY.md index line; cross-link from `library-composition-run.md`.

---

## Execution ordering / parallel lanes

```
Task 1 (dataset, ~min) ──> Task 2 (configs) ──> Task 5 (launch group + 6 individual chunks)
        └─ baseline job (bg, hours, atomic)                     │
Task 3 (--participants) before Task 5.   Task 4 (resolve_splits) anytime before Task 6.
train 0-49 done  ──> Task 6 (extraction) ──> Task 7 (two arms, parallel)
test 100-149 done + group done ──> Task 8 (coverage)            │
all 150 + winners + baseline ──> Task 9 (eval) ──> Task 10 (report)
```

## Self-review notes

- Spec coverage: dataset (T1), configs (T2), individual-runner range (T3), splits override (T4 — required deviation, logged), runs (T5), extraction+splits provisioning (T6), both arms (T7), coverage-on-test (T8), eval both arms on validation+test (T9), report+deliverables (T10). Spec's "two code touches" grew to three (T4) — unavoidable, recorded in the decision log and spec status remains accurate via log entry.
- The `PROMPT_MERGE` extraction prompt hardcodes "12 participants" in its text; with 50 seeds the count in prose is wrong but Gemini receives the actual 50 annotations. Cosmetic; intentionally NOT touched (surgical-change rule). Logged.
- Type consistency: `resolve_splits(target, group_dir, out_dir)` matches its call site; splits.json keys match `load_splits` consumers (`seed_pids`, `composition_validation_pids`, `reconstruction_pids`, `test_pids`).
