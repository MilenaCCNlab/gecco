"""Publication-quality versions of the main RLWM figures.

Same style as analysis/two_step_task/plot_clean_figures.py (palette from
refs/planning.pdf): baseline model in teal, GeCCo group in slate gray,
GeCCo individual (winning) in light blue, humans in black. Age split is
ignored — young and old participants are pooled.

Regenerates, from existing result files (no model re-fitting):
  1. figures/bic_comparison_baseline_vs_gecco.{png,pdf}
       Baseline (literature RLWM) vs GeCCo group / individual mean BIC.
  2. figures/ppc_learning_curves.{png,pdf}
       Learning curves p(correct) by stimulus iteration and set size:
       humans, baseline, GeCCo group, GeCCo individual.

Inputs: baseline_bics.csv / group_bics.csv / individual_bics.csv,
ns*_group_age.csv (pooled), rlwm_literature_model_simulated.csv, and the
per-participant simulation models + fitted parameters under
results/rlwm_individual/ (individual PPC is re-simulated, which is fast).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
FIG_DIR = ANALYSIS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

DATA_CSV = PROJECT_ROOT / "data" / "rlwm.csv"
LIT_SIM_CSV = ANALYSIS_DIR / "rlwm_literature_model_simulated.csv"
INDIV_PARAM_DIR = PROJECT_ROOT / "results" / "rlwm_individual" / "parameters"
INDIV_SIM_DIR = PROJECT_ROOT / "results" / "rlwm_individual" / "simulation"

N_SIM_REPS = 10  # stochastic simulations per participant (individual PPC)

# Palette extracted from refs/planning.pdf & refs/Multi_DM.pdf
TEAL = "#008181"    # reference model (baseline)
BLUE = "#40baec"    # winning model (GeCCo individual)
GRAY = "#708190"    # other models (GeCCo group)
BLACK = "#1a1a1a"   # humans

LIGHT_SHADES = {
    TEAL: "#9dcece",
    BLUE: "#a9e0f6",
    BLACK: "#b3b3b3",
    GRAY: "#bfc7ce",
}

INK = "#0b0b0b"
GRID = "#e1e0d9"


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })


def save(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.png")
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def sem(values, axis=0):
    values = np.asarray(values, dtype=float)
    return (np.nanstd(values, axis=axis, ddof=1)
            / np.sqrt(np.sum(~np.isnan(values), axis=axis)))


def analysis_participants(df):
    """Young/old selection used throughout the RLWM analyses (pooled here)."""
    young = list(df[df.age < 45].participant.unique()[:15])
    old = list(df[df.age > 45].participant.unique()[:15])
    return young + old


# ------------------------------------------------------------------
# BIC comparison
# ------------------------------------------------------------------

def plot_bic_comparison():
    models = [
        ("Baseline", pd.read_csv(ANALYSIS_DIR / "baseline_bics.csv")["bic"],
         TEAL),
        ("GeCCo\n(group)", pd.read_csv(ANALYSIS_DIR / "group_bics.csv")["bic"],
         GRAY),
        ("GeCCo\n(individual)",
         pd.read_csv(ANALYSIS_DIR / "individual_bics.csv")["bic"], BLUE),
    ]
    print("BIC n per model:", [len(v) for _, v, _ in models])

    fig, ax = plt.subplots(figsize=(2.6, 3.2))
    x = np.arange(len(models))
    means = [np.mean(v) for _, v, _ in models]
    errs = [sem(v) for _, v, _ in models]

    ax.bar(x, means, width=0.75, color=[c for _, _, c in models], zorder=2)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.errorbar(x, means, yerr=errs, fmt="o", markersize=5, color="black",
                ecolor="black", elinewidth=1.6, capsize=0, zorder=3)

    lo = np.floor((min(means) - max(errs)) / 20) * 20 - 20
    hi = np.ceil((max(means) + max(errs)) / 20) * 20
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models], fontsize=10.5)
    ax.set_ylabel("BIC")
    ax.set_ylim(lo, hi)
    ax.tick_params(axis="x", length=0)

    save(fig, "bic_comparison_baseline_vs_gecco")


# ------------------------------------------------------------------
# Learning curves (PPC)
# ------------------------------------------------------------------

def learning_curves(p_df, n_iters=9):
    """Per-participant mean p(correct) by stimulus iteration and set size."""
    by_set_size = {3: [], 6: []}
    for b in p_df.blocks.unique():
        block = p_df[p_df.blocks == b]
        ns = int(block.set_sizes.iloc[0])
        stimulus = block.stimulus.to_numpy()
        rewards = block.rewards.to_numpy()
        iteration = np.stack([np.cumsum(stimulus == s)
                              for s in np.unique(stimulus)], axis=1)
        col_iter = iteration[np.arange(len(stimulus)), stimulus] - 1
        curve = [np.mean(rewards[col_iter == i]) if np.any(col_iter == i)
                 else np.nan for i in range(n_iters)]
        by_set_size[ns].append(curve)
    return (np.nanmean(by_set_size[3], axis=0),
            np.nanmean(by_set_size[6], axis=0))


def curves_from_trials(df, participants):
    """Learning curves for each participant from trial-level data."""
    ns3, ns6 = [], []
    for p in participants:
        p_df = df[df.participant == p]
        p_df = p_df[p_df.rewards >= 0]
        c3, c6 = learning_curves(p_df)
        ns3.append(c3)
        ns6.append(c6)
    return np.array(ns3), np.array(ns6)


def simulate_individual_curves(df, participants):
    """Simulate each participant's fitted GeCCo model and compute curves."""
    np.random.seed(0)
    ns3, ns6 = [], []
    for p in participants:
        params_file = INDIV_PARAM_DIR / f"best_params_run0_participant{p}.csv"
        model_file = INDIV_SIM_DIR / f"simulation_model_participant{p}.txt"
        if not (params_file.exists() and model_file.exists()):
            print(f"skipping participant {p}: missing params or model")
            continue
        best_parameters = pd.read_csv(params_file)
        parameters = [best_parameters[c][0] for c in best_parameters.columns]

        namespace = {"np": np}
        exec(model_file.read_text(), namespace)
        simulate = namespace["simulate_model"]

        p_df = df[df.participant == p].reset_index(drop=True)
        p_df = p_df[p_df.rewards >= 0].reset_index(drop=True)
        stimulus = p_df.stimulus.to_numpy()
        blocks = p_df.blocks.to_numpy()
        set_sizes = p_df.set_sizes.to_numpy()
        correct_answer = p_df.correct_answer.to_numpy()

        reps3, reps6 = [], []
        for _ in range(N_SIM_REPS):
            actions, rewards = simulate(stimulus, blocks, set_sizes,
                                        correct_answer, parameters)
            sim_df = pd.DataFrame({"stimulus": stimulus, "blocks": blocks,
                                   "set_sizes": set_sizes,
                                   "rewards": rewards})
            c3, c6 = learning_curves(sim_df)
            reps3.append(c3)
            reps6.append(c6)
        ns3.append(np.nanmean(reps3, axis=0))
        ns6.append(np.nanmean(reps6, axis=0))
    return np.array(ns3), np.array(ns6)


def pooled_group_curves():
    """Pool the saved young/old group-model curves (age ignored)."""
    ns3 = pd.concat([
        pd.read_csv(ANALYSIS_DIR / "ns3_young_group_age.csv", index_col=0),
        pd.read_csv(ANALYSIS_DIR / "ns3_old_group_age.csv", index_col=0),
    ]).to_numpy()
    ns6 = pd.concat([
        pd.read_csv(ANALYSIS_DIR / "ns6_young_group_age.csv", index_col=0),
        pd.read_csv(ANALYSIS_DIR / "ns6_old_group_age.csv", index_col=0),
    ]).to_numpy()
    return ns3, ns6


def draw_learning_curves(ax, ns3, ns6, color, show_ylabel=True,
                         show_xlabel=True, annotate=False):
    """Dark shade = set size 3, light shade = set size 6."""
    dark, light = color, LIGHT_SHADES[color]
    x = np.arange(1, ns3.shape[1] + 1)
    for curves, shade in ((ns3, dark), (ns6, light)):
        mean = np.nanmean(curves, axis=0)
        err = sem(curves)
        ax.fill_between(x, mean - err, mean + err, color=shade, alpha=0.25,
                        linewidth=0)
        ax.plot(x, mean, color=shade, linewidth=2)

    if annotate:
        ax.text(x[-1], np.nanmean(ns3, axis=0)[-1] + 0.07, "set size 3",
                ha="right", fontsize=10.5, color=INK)
        ax.text(x[-1], np.nanmean(ns6, axis=0)[-1] - 0.13, "set size 6",
                ha="right", fontsize=10.5, color=INK)

    ax.set_xlim(0.8, ns3.shape[1] + 0.2)
    ax.set_xticks([1, 3, 5, 7, 9])
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    if show_ylabel:
        ax.set_ylabel("p(correct)")
    if show_xlabel:
        ax.set_xlabel("stimulus iteration")
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def plot_ppc_learning_curves():
    df = pd.read_csv(DATA_CSV)
    participants = analysis_participants(df)
    print(f"PPC over n={len(participants)} participants")

    human3, human6 = curves_from_trials(df, participants)
    lit = pd.read_csv(LIT_SIM_CSV)
    base3, base6 = curves_from_trials(lit, participants)
    group3, group6 = pooled_group_curves()
    indiv3, indiv6 = simulate_individual_curves(df, participants)
    print(f"individual PPC simulated for n={len(indiv3)} participants")

    panels = [
        ("Humans", human3, human6, BLACK),
        ("Baseline", base3, base6, TEAL),
        ("GeCCo\n(group)", group3, group6, GRAY),
        ("GeCCo\n(individual)", indiv3, indiv6, BLUE),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(8.2, 2.9), sharey=True)
    for i, (ax, (title, ns3, ns6, color)) in enumerate(zip(axes, panels)):
        draw_learning_curves(ax, ns3, ns6, color, show_ylabel=(i == 0),
                             show_xlabel=(i == 0), annotate=(i == 0))
        ax.set_title(title)
    fig.subplots_adjust(wspace=0.15)

    save(fig, "ppc_learning_curves")


if __name__ == "__main__":
    set_style()
    plot_bic_comparison()
    plot_ppc_learning_curves()
    print(f"Figures written to {FIG_DIR}")
