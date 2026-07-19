"""Exploratory two-step PPC: stay-probability by previous reward x transition,
for four models (Humans, Hybrid, GeCCo group, GeCCo individual), each split by
OCI group (Low vs High). 2x4 grid: rows = OCI group, columns = model.

Test participants only (50; Low/High tertiles used, Medium dropped). Models are
simulated with each participant's own drift schedule and best-fit parameters:
  Humans           - real stage-1 choices
  Hybrid           - Daw hybrid, refit per pid, simulated
  GeCCo (group)    - group run's simulation_model_run0 + best_params_on_test
  GeCCo (individual) - per-pid simulation_model_participant{pid} + per-pid params

Run: PYTHONPATH=. gecco-env/bin/python analysis/two_step_task/plot_ppc_oci_split.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gecco.utils import extract_full_function
from library_learning.compose.hybrid import HYBRID_SOURCE, HYBRID_BOUNDS
from library_learning.compose.fitting import fit_model_on_pids
from library_learning.config import resolve_target

ROOT = Path(__file__).resolve().parents[2]
IND = ROOT / "results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual"
GRP = ROOT / "results/two_step_psychiatry_group_function_ocibalanced150_maxsetting"
DATA = ROOT / "data/two_step_gillan_2016_ocibalanced150.csv"
FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)
N_REPS = 20

BLACK, TEAL, GRAY, BLUE, INK = "#1a1a1a", "#008181", "#708190", "#40baec", "#0b0b0b"
LIGHT = {BLACK: "#b3b3b3", TEAL: "#9dcece", GRAY: "#bfc7ce", BLUE: "#a9e0f6"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


def stay_probs(choice_1, state, reward):
    c = np.asarray(choice_1); s = np.asarray(state); r = np.asarray(reward)
    stay = (c[1:] == c[:-1])
    common = (((c == 0) & (s == 0)) | ((c == 1) & (s == 1)))[:-1]
    rare = ~common
    rew = (r[:-1] == 1); nrew = (r[:-1] == 0)  # ignore -1 missed as prev outcome

    def sp(cond):
        return np.mean(stay[cond]) if np.any(cond) else np.nan
    # order: common/r, rare/r, common/nr, rare/nr
    return [sp(common & rew), sp(rare & rew), sp(common & nrew), sp(rare & nrew)]


def sim_stay(simulate, params, drift, n_trials, seed0):
    reps = []
    for k in range(N_REPS):
        np.random.seed(seed0 + k)
        s1, s2, a2, rew = simulate(n_trials, list(params), *drift)
        reps.append(stay_probs(s1, s2, rew))
    return np.nanmean(reps, axis=0)


def load_simulate(path, name="simulate_model"):
    src = extract_full_function(Path(path).read_text(), name)
    ns = {"np": np}
    exec(src, ns)
    return ns[name]


def main():
    df = pd.read_csv(DATA)
    manifest = {p["participant"]: p for p in
                json.loads((ROOT / "data/ocd/ocibalanced150_manifest.json").read_text())["participants"]}
    test = [p for p in sorted(manifest) if manifest[p]["split"] == "test"]
    groups = {"Low": [p for p in test if manifest[p]["tertile"] == "Low"],
              "High": [p for p in test if manifest[p]["tertile"] == "High"]}
    target = resolve_target(str(IND))

    drift = {p: [df[df.participant == p]["drift_%d" % i].to_numpy() for i in (1, 2, 3, 4)]
             for p in test}
    ntr = {p: int((df.participant == p).sum()) for p in test}

    # ---- humans ----
    human = {}
    for p in test:
        d = df[df.participant == p]
        human[p] = stay_probs(d.choice_1.to_numpy(), d.state.to_numpy(), d.reward.to_numpy())

    # ---- hybrid: fit per pid, simulate with the Daw hybrid simulator ----
    hybrid = {}
    hyb_fit = {p: fit_model_on_pids(HYBRID_SOURCE, target, [p], HYBRID_BOUNDS,
                                    tag="ppc:hybrid")[p]["params"] for p in test}
    for p in test:
        hybrid[p] = sim_stay(_daw_simulate, hyb_fit[p], drift[p], ntr[p], 7000 + p)

    # ---- group gecco ----
    g_sim = load_simulate(GRP / "simulation" / "simulation_model_run0.txt")
    g_par = pd.read_csv(GRP / "parameters" / "best_params_on_test_run0.csv")
    group = {}
    for idx, p in enumerate(test):  # best_params_on_test rows are in test-pid order
        params = g_par.iloc[idx].to_numpy()
        group[p] = sim_stay(g_sim, params, drift[p], ntr[p], 8000 + p)

    # ---- individual gecco ----
    indiv = {}
    for p in test:
        sim_path = IND / "simulation" / ("simulation_model_participant%d.txt" % p)
        par_path = IND / "parameters" / ("best_params_run0_participant%d.csv" % p)
        if not (sim_path.exists() and par_path.exists()):
            continue
        sim = load_simulate(sim_path)
        params = pd.read_csv(par_path).iloc[0].to_numpy()
        indiv[p] = sim_stay(sim, params, drift[p], ntr[p], 9000 + p)

    models = [("Humans", human, BLACK), ("Hybrid", hybrid, TEAL),
              ("GeCCo (group)", group, GRAY), ("GeCCo (individual)", indiv, BLUE)]

    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.0), sharey=True)
    for row, g in enumerate(["Low", "High"]):
        for col, (name, data, color) in enumerate(models):
            ax = axes[row, col]
            vals = np.array([data[p] for p in groups[g] if p in data and not np.all(np.isnan(data[p]))])
            m = np.nanmean(vals, axis=0)
            e = np.nanstd(vals, axis=0, ddof=1) / np.sqrt(len(vals))
            _draw(ax, m, e, color)
            if row == 0:
                ax.set_title(name, fontsize=12)
            if col == 0:
                ax.set_ylabel("%s OCI\n(n=%d)\nP(stay)" % (g, len(groups[g])), fontsize=10)
    fig.suptitle("Two-step stay probability: models vs humans, split by OCI (test participants)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "ppc_stay_oci_split.png")
    fig.savefig(FIG / "ppc_stay_oci_split.pdf")
    plt.close(fig)
    print("saved figures/ppc_stay_oci_split.{png,pdf}")


def _draw(ax, means, errs, color):
    """means order: common/r, rare/r, common/nr, rare/nr. x = rewarded / not,
    dark = common, light = rare."""
    light = LIGHT[color]
    pos = [0.0, 0.5, 1.6, 2.1]
    colors = [color, light, color, light]
    ax.bar(pos, means, width=0.48, color=colors, zorder=2)
    ax.errorbar(pos, means, yerr=errs, fmt="none", ecolor="black", elinewidth=1.2, capsize=0, zorder=3)
    ax.axhline(0.5, color="0.6", lw=0.7, ls=":", zorder=1)
    ax.set_ylim(0.4, 1.0); ax.set_yticks(np.arange(0.4, 1.01, 0.2))
    ax.set_xticks([0.25, 1.85]); ax.set_xticklabels(["rewarded", "unrewarded"], fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.text(0.0, 0.42, "common", rotation=90, ha="center", va="bottom", fontsize=6.5, color="white")
    ax.text(0.5, 0.42, "rare", rotation=90, ha="center", va="bottom", fontsize=6.5, color=INK)


def _daw_simulate(n_trials, parameters, drift1, drift2, drift3, drift4):
    """Daw hybrid generative model (matches HYBRID_SOURCE param order:
    learning_rate, learning_rate_2, beta, beta_2, w, lambd, perseveration)."""
    lr, lr2, beta, beta2, w, lam, persev = parameters
    T = np.array([[0.7, 0.3], [0.3, 0.7]])
    q1 = np.zeros(2); q2 = np.zeros((2, 2)); pers = np.zeros(2)
    s1o = np.zeros(n_trials, dtype=int); s2o = np.zeros(n_trials, dtype=int)
    a2o = np.zeros(n_trials, dtype=int); ro = np.zeros(n_trials, dtype=int)
    for t in range(n_trials):
        rp = [[drift1[t], drift2[t]], [drift3[t], drift4[t]]]
        mb = T @ np.max(q2, axis=1)
        net = w * mb + (1 - w) * q1 + persev * pers
        p1 = np.exp(beta * net); p1 = p1 / p1.sum()
        a1 = np.random.choice([0, 1], p=p1)
        s2 = np.random.choice([0, 1], p=T[a1])
        pq = np.exp(beta2 * q2[s2]); pq = pq / pq.sum()
        a2 = np.random.choice([0, 1], p=pq)
        r = int(np.random.random() < rp[s2][a2])
        d1 = q2[s2, a2] - q1[a1]; q1[a1] += lr * d1
        d2 = r - q2[s2, a2]; q2[s2, a2] += lr2 * d2
        q1[a1] += lam * lr * d2
        pers[:] = 0; pers[a1] = 1
        s1o[t] = a1; s2o[t] = s2; a2o[t] = a2; ro[t] = r
    return s1o, s2o, a2o, ro


if __name__ == "__main__":
    main()
