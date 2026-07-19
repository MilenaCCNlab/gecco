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
        print("participant %d baseline BIC %.2f" % (pid, bic), flush=True)
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
