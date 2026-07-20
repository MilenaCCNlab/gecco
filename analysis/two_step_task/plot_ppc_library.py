"""Two-step PPC (stay probability) for four models: Humans, Hybrid, GeCCo
(group), and Library (individual). The composed library programs are likelihood
functions, so this builds a generative twin of render_candidate's backbone
(samples a1, s2, a2, and reward from the participant's drift schedule, applying
the same slot/override updates). Each test participant's best composition
(reconstruction_results.json) is refit for parameters, then simulated N times.

  Humans            - real stage-1 choices
  Hybrid            - Daw hybrid, refit per pid, simulated
  GeCCo (group)     - group simulation_model_run0 + best_params_on_test
  Library (individual) - per-pid best composition, generative twin

Pooled over the 50 test participants. Output: figures/ppc_humans_vs_gecco_library.

Run: PYTHONPATH=. gecco-env/bin/python analysis/two_step_task/plot_ppc_library.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from library_learning.compose.inventory import APPEND_SLOTS, load_inventory
from library_learning.compose.render import (DEFAULT_OVERRIDES, EMPTY_SLOT_SENTINEL,
                                             candidate_params, render_candidate, _indent)
from library_learning.compose.fitting import fit_model_on_pids
from library_learning.compose.hybrid import HYBRID_SOURCE, HYBRID_BOUNDS
from library_learning.config import resolve_target
from library_learning.loading import exec_model
# reuse helpers from the OCI-split PPC script
from analysis.two_step_task.plot_ppc_oci_split import (
    stay_probs, sim_stay, load_simulate, _daw_simulate, _draw)

ROOT = Path(__file__).resolve().parents[2]
IND = ROOT / "results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual"
GRP = ROOT / "results/two_step_psychiatry_group_function_ocibalanced150_maxsetting"
LC = IND / "library_composition"
DATA = ROOT / "data/two_step_gillan_2016_ocibalanced150.csv"
FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)

BLACK, TEAL, GRAY, BLUE, INK = "#1a1a1a", "#008181", "#708190", "#40baec", "#0b0b0b"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


def render_simulate(inventory, module_ids):
    """Generative twin of render_candidate: samples a1, s2 (via the 0.7/0.3
    transition), a2, and reward (from the drift schedule), applying the same
    slot/override updates. Same variable names as the backbone (a1, s_idx, a2, r)
    so module snippets run unchanged."""
    overrides = dict(DEFAULT_OVERRIDES)
    mods = [inventory.module(m) for m in sorted(module_ids)]
    for m in mods:
        overrides.update(m.overrides)
    appends = {slot: [] for slot in APPEND_SLOTS}
    for m in mods:
        for slot, code in m.slots.items():
            appends[slot].append(code)
    params = candidate_params(inventory, module_ids)
    unpack = ", ".join(p.name for p in params)
    if len(params) == 1:
        unpack += ","

    def block(slot, level):
        parts = [_indent(c, level) for c in appends[slot]]
        return "\n".join(parts) + "\n" if parts else _indent(EMPTY_SLOT_SENTINEL, level)

    src = '''def simulate_model(n_trials, parameters, drift1, drift2, drift3, drift4):
    {unpack} = parameters
    transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    q_stage1_mf = np.zeros(2)
    q_stage2_mf = {q2_init}
    eps = 1e-10
    s1o = np.zeros(n_trials, dtype=int); s2o = np.zeros(n_trials, dtype=int)
    a2o = np.zeros(n_trials, dtype=int); ro = np.zeros(n_trials, dtype=int)
{init}
    for trial in range(n_trials):
        max_q_stage2 = np.max(q_stage2_mf, axis=1)
        q_stage1_mb = transition_matrix @ max_q_stage2
{pre_stage1}
        stage1_values = {stage1_values}
        logits_1 = ({stage1_temp}) * stage1_values
{stage1_logits_extra}
        exp_q1 = np.exp(logits_1 - np.max(logits_1))
        probs_1 = exp_q1 / np.sum(exp_q1)
        a1 = int(np.random.choice(2, p=probs_1))
        s_idx = int(np.random.choice(2, p=transition_matrix[a1]))
        stage2_values = {stage2_values}
        logits_2 = ({stage2_temp}) * stage2_values
{stage2_logits_extra}
        exp_q2 = np.exp(logits_2 - np.max(logits_2))
        probs_2 = exp_q2 / np.sum(exp_q2)
        a2 = int(np.random.choice(2, p=probs_2))
        reward_probs = [[drift1[trial], drift2[trial]], [drift3[trial], drift4[trial]]]
        r = float(np.random.random() < reward_probs[s_idx][a2])
        delta_stage1 = q_stage2_mf[s_idx, a2] - q_stage1_mf[a1]
{stage1_update}
        delta_stage2 = r - q_stage2_mf[s_idx, a2]
{stage2_update}
{update_extra}
{post_trial}
        s1o[trial] = a1; s2o[trial] = s_idx; a2o[trial] = a2; ro[trial] = int(r)
    return s1o, s2o, a2o, ro
'''.format(
        unpack=unpack, q2_init=overrides["q2_init"],
        stage1_values=overrides["stage1_values"], stage1_temp=overrides["stage1_temp"],
        stage2_values=overrides["stage2_values"], stage2_temp=overrides["stage2_temp"],
        stage1_update=_indent(overrides["stage1_update"], 2),
        stage2_update=_indent(overrides["stage2_update"], 2),
        init=block("init", 1), pre_stage1=block("pre_stage1", 2),
        stage1_logits_extra=block("stage1_logits_extra", 2),
        stage2_logits_extra=block("stage2_logits_extra", 2),
        update_extra=block("update_extra", 2), post_trial=block("post_trial", 2))
    return "\n".join(l for l in src.splitlines() if l.strip() != EMPTY_SLOT_SENTINEL) + "\n"


def main():
    df = pd.read_csv(DATA)
    manifest = {p["participant"]: p for p in
                json.loads((ROOT / "data/ocd/ocibalanced150_manifest.json").read_text())["participants"]}
    test = [p for p in sorted(manifest) if manifest[p]["split"] == "test"]
    target = resolve_target(str(IND))
    inv = load_inventory(LC / "module_inventory.json")
    recon = {int(r["pid"]): r["best_module_ids"] for r in
             json.loads((LC / "reconstruction_results.json").read_text())}

    drift = {p: [df[df.participant == p]["drift_%d" % i].to_numpy() for i in (1, 2, 3, 4)] for p in test}
    ntr = {p: int((df.participant == p).sum()) for p in test}

    human, hybrid, group, library = {}, {}, {}, {}
    # humans
    for p in test:
        d = df[df.participant == p]
        human[p] = stay_probs(d.choice_1.to_numpy(), d.state.to_numpy(), d.reward.to_numpy())
    # hybrid
    for p in test:
        par = fit_model_on_pids(HYBRID_SOURCE, target, [p], HYBRID_BOUNDS, tag="ppc:hybrid")[p]["params"]
        hybrid[p] = sim_stay(_daw_simulate, par, drift[p], ntr[p], 7000 + p)
    # group gecco
    g_sim = load_simulate(GRP / "simulation" / "simulation_model_run0.txt")
    g_par = pd.read_csv(GRP / "parameters" / "best_params_on_test_run0.csv")
    for idx, p in enumerate(test):
        group[p] = sim_stay(g_sim, g_par.iloc[idx].to_numpy(), drift[p], ntr[p], 8000 + p)
    # library individual: refit each pid's best composition, simulate its twin
    for p in test:
        mods = recon[p]
        like = render_candidate(inv, mods)
        bounds = [pp.bounds for pp in candidate_params(inv, mods)]
        par = fit_model_on_pids(like, target, [p], bounds, tag="ppc:lib:%s" % "+".join(mods))[p]["params"]
        sim = exec_model(render_simulate(inv, mods), "simulate_model")
        library[p] = sim_stay(sim, par, drift[p], ntr[p], 9500 + p)

    models = [("Humans", human, BLACK), ("Hybrid", hybrid, TEAL),
              ("GeCCo (group)", group, GRAY), ("Library (individual)", library, BLUE)]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2), sharey=True)
    for ax, (name, data, color) in zip(axes, models):
        vals = np.array([data[p] for p in test if not np.all(np.isnan(data[p]))])
        m = np.nanmean(vals, axis=0)
        e = np.nanstd(vals, axis=0, ddof=1) / np.sqrt(len(vals))
        _draw(ax, m, e, color)
        ax.set_title(name, fontsize=12)
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.suptitle("Two-step stay probability: models vs humans (test participants)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "ppc_humans_vs_gecco_library.png")
    fig.savefig(FIG / "ppc_humans_vs_gecco_library.pdf")
    plt.close(fig)
    print("saved figures/ppc_humans_vs_gecco_library.{png,pdf}")
    for name, data, _ in models:
        vals = np.array([data[p] for p in test if not np.all(np.isnan(data[p]))])
        print("  %-22s mean stay [c/r,r/r,c/nr,r/nr] = %s"
              % (name, np.round(np.nanmean(vals, axis=0), 3)))


if __name__ == "__main__":
    main()
