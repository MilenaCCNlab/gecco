"""PPC learning curves for the library-composed models (plot-span protocol).

The composed programs are likelihood functions, so this builds a generative
simulator for the backbone + module slots: each trial samples an action from
the model's choice probabilities, sets reward = 1 iff it matches the state's
correct action, and applies the same RL/WM updates with the sampled outcome.
Each participant's composition is refit (per-pid, plot-span data) to get
parameters, then simulated N times; p(correct) by stimulus iteration and set
size is averaged and plotted next to the human curves.

  Library (group)      = the one shared frozen program (composed_model.txt)
  Library (individual) = each participant's best composition
                         (perpid_plotspan_results.json)

Run from repo root:
  PYTHONPATH=. gecco-env/bin/python analysis/rlwm/plot_ppc_library.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from library_learning.compose.inventory_rlwm import load_inventory, parse_inventory
from library_learning.compose.render_rlwm import (
    DEFAULT_OVERRIDES, EMPTY_SLOT_SENTINEL, candidate_params)
from library_learning.compose.render import _indent
from library_learning.compose.inventory_rlwm import APPEND_SLOTS
from library_learning.compose.fitting import fit_participant, seed_for
from library_learning.loading import bounds_for_code, exec_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
FIG_DIR = ANALYSIS_DIR / "figures"
LC = PROJECT_ROOT / "results" / "rlwm_individual" / "library_composition"
DATA_CSV = PROJECT_ROOT / "data" / "rlwm.csv"
COLS = ["stimulus", "actions", "rewards", "blocks", "set_sizes"]
PIDS = list(range(15)) + list(range(36, 51))
N_SIM_REPS = 20
N_ITERS = 9

BLACK, TEAL, GRAY, BLUE = "#1a1a1a", "#008181", "#708190", "#40baec"
INK, GRID = "#0b0b0b", "#e1e0d9"
LIGHT = {BLACK: "#b3b3b3", TEAL: "#9dcece", GRAY: "#bfc7ce", BLUE: "#a9e0f6"}
LIT_SIM_CSV = ANALYSIS_DIR / "rlwm_literature_model_simulated.csv"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


def render_simulate(inventory, module_ids):
    """Generative twin of render_candidate: samples actions, computes reward
    from correct_answer, applies the same slot updates with the sampled a/r."""
    mods = [inventory.module(m) for m in sorted(module_ids)]
    params = candidate_params(inventory, module_ids)
    ov = dict(DEFAULT_OVERRIDES)
    for m in mods:
        ov.update(m.overrides)
    appends = {s: [] for s in APPEND_SLOTS}
    for m in mods:
        for slot, code in m.slots.items():
            appends[slot].append(code)
    unpack = ", ".join(p.name for p in params)
    if len(params) == 1:
        unpack += ","

    def block(slot, level):
        parts = [_indent(c, level) for c in appends[slot]]
        return "\n".join(parts) + "\n" if parts else _indent(EMPTY_SLOT_SENTINEL, level)

    src = '''def simulate_model(stimulus, blocks, set_sizes, correct_answer, parameters):
    {unpack} = parameters
    nA = 3
    eps = 1e-10
    n_trials = len(stimulus)
    simulated_actions = np.zeros(n_trials, dtype=int)
    simulated_rewards = np.zeros(n_trials, dtype=int)
    tr = 0
{init}
    for b in np.unique(blocks):
        block_mask = blocks == b
        block_states = stimulus[block_mask]
        block_correct = correct_answer[block_mask]
        nS = int(set_sizes[block_mask][0])
        correct = np.zeros(nS, dtype=int)
        for st in range(nS):
            vals = block_correct[block_states == st]
            correct[st] = int(vals[0]) if len(vals) else 0
        q = {q_init}
        w = {w_init}
        w_0 = (1.0 / nA) * np.ones((nS, nA))
{block_init}
        for trial in range(len(block_states)):
            s = int(block_states[trial])
{pre_choice}
            rl_values = {rl_values}
            logits_rl = ({rl_temp}) * rl_values
{rl_logits_extra}
            exp_rl = np.exp(logits_rl - np.max(logits_rl))
            probs_rl = exp_rl / np.sum(exp_rl)
            wm_values = {wm_values}
            logits_wm = ({wm_temp}) * wm_values
{wm_logits_extra}
            exp_wm = np.exp(logits_wm - np.max(logits_wm))
            probs_wm = exp_wm / np.sum(exp_wm)
            mix = {mix_weight}
            probs = mix * probs_wm + (1.0 - mix) * probs_rl
{probs_extra}
            probs = np.clip(probs, 1e-12, None)
            probs = probs / np.sum(probs)
            a = int(np.random.choice(nA, p=probs))
            r = 1.0 if a == correct[s] else 0.0
            simulated_actions[tr] = a
            simulated_rewards[tr] = int(r)
            delta = r - q[s, a]
{rl_update}
{wm_update}
{update_extra}
{post_trial}
            tr += 1
    return simulated_actions, simulated_rewards
'''.format(
        unpack=unpack,
        q_init=ov["q_init"], w_init=ov["w_init"], rl_values=ov["rl_values"],
        rl_temp=ov["rl_temp"], wm_values=ov["wm_values"], wm_temp=ov["wm_temp"],
        mix_weight=ov["mix_weight"],
        rl_update=_indent(ov["rl_update"], 3), wm_update=_indent(ov["wm_update"], 3),
        init=block("init", 1), block_init=block("block_init", 2),
        pre_choice=block("pre_choice", 3), rl_logits_extra=block("rl_logits_extra", 3),
        wm_logits_extra=block("wm_logits_extra", 3), probs_extra=block("probs_extra", 3),
        update_extra=block("update_extra", 3), post_trial=block("post_trial", 3))
    return "\n".join(l for l in src.splitlines()
                     if l.strip() != EMPTY_SLOT_SENTINEL) + "\n"


def learning_curves(p_df):
    by = {3: [], 6: []}
    for b in p_df.blocks.unique():
        blk = p_df[p_df.blocks == b]
        ns = int(blk.set_sizes.iloc[0])
        stim = blk.stimulus.to_numpy(); rew = blk.rewards.to_numpy()
        it = np.stack([np.cumsum(stim == s) for s in np.unique(stim)], axis=1)
        col = it[np.arange(len(stim)), stim] - 1
        by[ns].append([np.mean(rew[col == i]) if np.any(col == i) else np.nan
                       for i in range(N_ITERS)])
    return np.nanmean(by[3], axis=0), np.nanmean(by[6], axis=0)


def human_curves(df):
    ns3, ns6 = [], []
    for p in PIDS:
        d = df[(df.participant == p) & (df.blocks < 5) & (df.rewards >= 0)]
        c3, c6 = learning_curves(d)
        ns3.append(c3); ns6.append(c6)
    return np.array(ns3), np.array(ns6)


def library_curves(df, inv, per_pid_modules):
    """per_pid_modules: {pid: [module ids]}. Refit each per pid, simulate."""
    rng = np.random.RandomState(0)
    ns3, ns6 = [], []
    for p in PIDS:
        mods = per_pid_modules[p]
        like = None
        from library_learning.compose.render_rlwm import render_candidate
        like_src = render_candidate(inv, mods)
        func = exec_model(like_src, "cognitive_model")
        bounds = bounds_for_code(like_src)
        d = df[(df.participant == p) & (df.blocks < 5) & (df.rewards >= 0)]
        inputs = [d[c].to_numpy() for c in COLS]
        fit = fit_participant(func, inputs, bounds, seed_for("ppc:%s" % "+".join(mods), p))
        params = fit["params"]
        sim_src = render_simulate(inv, mods)
        simulate = exec_model(sim_src, "simulate_model")
        stim = d.stimulus.to_numpy(); blk = d.blocks.to_numpy()
        ss = d.set_sizes.to_numpy(); corr = d.correct_answer.to_numpy()
        reps3, reps6 = [], []
        for _ in range(N_SIM_REPS):
            np.random.seed(rng.randint(1 << 30))
            _, rew = simulate(stim, blk, ss, corr, params)
            sim_df = pd.DataFrame({"stimulus": stim, "blocks": blk,
                                   "set_sizes": ss, "rewards": rew})
            c3, c6 = learning_curves(sim_df)
            reps3.append(c3); reps6.append(c6)
        ns3.append(np.nanmean(reps3, axis=0)); ns6.append(np.nanmean(reps6, axis=0))
    return np.array(ns3), np.array(ns6)


def baseline_curves(lit):
    """RLWM literature-model simulated learning curves (same 30 pids)."""
    ns3, ns6 = [], []
    for p in PIDS:
        d = lit[lit.participant == p]
        if "blocks" in d.columns:
            d = d[d.blocks < 5]
        d = d[d.rewards >= 0]
        c3, c6 = learning_curves(d)
        ns3.append(c3); ns6.append(c6)
    return np.array(ns3), np.array(ns6)


def group_gecco_curves():
    """Pooled young+old GeCCo-group PPC curves saved by the main pipeline."""
    ns3 = pd.concat([pd.read_csv(ANALYSIS_DIR / "ns3_young_group_age.csv", index_col=0),
                     pd.read_csv(ANALYSIS_DIR / "ns3_old_group_age.csv", index_col=0)]).to_numpy()
    ns6 = pd.concat([pd.read_csv(ANALYSIS_DIR / "ns6_young_group_age.csv", index_col=0),
                     pd.read_csv(ANALYSIS_DIR / "ns6_old_group_age.csv", index_col=0)]).to_numpy()
    return ns3, ns6


def sem(v):
    v = np.asarray(v, float)
    return np.nanstd(v, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(v), axis=0))


def draw(ax, ns3, ns6, color, ylabel=False, xlabel=False, annotate=False):
    dark, light = color, LIGHT[color]
    x = np.arange(1, ns3.shape[1] + 1)
    for curves, shade in ((ns3, dark), (ns6, light)):
        mean = np.nanmean(curves, axis=0); err = sem(curves)
        ax.fill_between(x, mean - err, mean + err, color=shade, alpha=0.25, lw=0)
        ax.plot(x, mean, color=shade, lw=2)
    if annotate:
        ax.text(x[-1], np.nanmean(ns3, axis=0)[-1] + 0.06, "set size 3",
                ha="right", fontsize=10, color=INK)
        ax.text(x[-1], np.nanmean(ns6, axis=0)[-1] - 0.12, "set size 6",
                ha="right", fontsize=10, color=INK)
    ax.set_xlim(0.8, ns3.shape[1] + 0.2); ax.set_xticks([1, 3, 5, 7, 9])
    ax.set_ylim(0, 1.0); ax.set_yticks(np.arange(0, 1.01, 0.25))
    if ylabel:
        ax.set_ylabel("p(correct)")
    if xlabel:
        ax.set_xlabel("stimulus iteration")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)


def main():
    df = pd.read_csv(DATA_CSV)
    inv_full = load_inventory(LC / "module_inventory_leakfree.json")

    perpid = json.load(open(LC / "perpid_plotspan_results.json"))
    indiv_mods = {r["pid"]: r["modules"] for r in perpid}

    print("simulating human / RLWM / gecco(group) / library(individual) curves...")
    h3, h6 = human_curves(df)
    b3, b6 = baseline_curves(pd.read_csv(LIT_SIM_CSV))
    gg3, gg6 = group_gecco_curves()
    i3, i6 = library_curves(df, inv_full, indiv_mods)
    for lbl, c3, c6 in [("human", h3, h6), ("RLWM", b3, b6),
                        ("gecco-group", gg3, gg6), ("lib-indiv", i3, i6)]:
        print("  %-12s ss3 final %.2f | ss6 final %.2f"
              % (lbl, np.nanmean(c3, 0)[-1], np.nanmean(c6, 0)[-1]))

    panels = [("Humans", h3, h6, BLACK), ("RLWM", b3, b6, TEAL),
              ("GeCCo\n(group)", gg3, gg6, GRAY),
              ("Library\n(individual)", i3, i6, BLUE)]
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.9), sharey=True)
    for i, (ax, (title, ns3, ns6, color)) in enumerate(zip(axes, panels)):
        draw(ax, ns3, ns6, color, ylabel=(i == 0), xlabel=(i == 0), annotate=(i == 0))
        ax.set_title(title)
    fig.subplots_adjust(wspace=0.15)
    fig.savefig(FIG_DIR / "ppc_learning_curves_library.png")
    fig.savefig(FIG_DIR / "ppc_learning_curves_library.pdf")
    plt.close(fig)
    print("saved figures/ppc_learning_curves_library.{png,pdf}")


if __name__ == "__main__":
    main()
