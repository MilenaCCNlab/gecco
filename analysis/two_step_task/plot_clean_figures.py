"""Publication-quality versions of the main two-step-task figures.

Styling follows the reference paper figures in refs/planning.pdf and
refs/Multi_DM.pdf: reference model in teal, winning model in light blue,
other models in slate gray; PPC panels tinted per model with a dark shade
for common transitions and a light shade for rare transitions.

Regenerates, from existing result files (no model re-fitting):
  1. figures/bic_comparison_hybrid_vs_gecco.{png,pdf}
       Hybrid baseline vs GeCCo group / individual (w/o OCI,
       gemini-3-pro max-setting runs on the OCI-balanced dataset).
  2. figures/ppc_humans_vs_gecco.{png,pdf}
       Stay-probability PPCs: humans vs GeCCo group & individual.
  3. figures/ppc_gecco_individual.{png,pdf}
  4. figures/ppc_gecco_group.{png,pdf}
"""

import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
FIG_DIR = ANALYSIS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

DATA_CSV = PROJECT_ROOT / "data" / "two_step_gillan_2016_ocibalanced.csv"
GROUP_BICS = (PROJECT_ROOT / "results"
              / "two_step_psychiatry_group_function_ocibalanced_maxsetting"
              / "bics" / "best_bic_on_test_run0.json")
INDIV_BICS_DIR = (PROJECT_ROOT / "results"
                  / "two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual"
                  / "bics")
# best_bic_on_test_run0.json stores individual_BIC positionally for these ids
GROUP_PARTICIPANTS = list(range(14, 45))

PPC_HUMANS = ANALYSIS_DIR / "ppcs_humans.csv"
PPC_GROUP = ANALYSIS_DIR / ("ppcs_two_step_psychiatry_group_function_"
                            "ocibalanced_maxsetting_group_oci.csv")
PPC_INDIV = ANALYSIS_DIR / ("ppcs_two_step_psychiatry_individual_function_"
                            "ocibalanced_maxsetting_individual_oci.csv")


def lighten(hex_color, factor=0.55):
    """Mix a color with white; factor is the share of white."""
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(
        round(c + (255 - c) * factor) for c in rgb)


# Palette extracted from refs/planning.pdf & refs/Multi_DM.pdf
TEAL = "#008181"    # reference model (Hybrid)
BLUE = "#40baec"    # winning model (GeCCo individual)
GRAY = "#708190"    # other models (GeCCo group)
BLACK = "#1a1a1a"   # humans

MODEL_COLORS = {
    "hybrid": TEAL,
    "group": GRAY,
    "individual": BLUE,
    "humans": BLACK,
}

# Rare-transition (light) shades; teal/blue/black taken from the reference PDFs
LIGHT_SHADES = {
    TEAL: "#9dcece",
    BLUE: "#a9e0f6",
    BLACK: "#b3b3b3",
}

INK = "#0b0b0b"
INK2 = "#52514e"
AXIS = "#c3c2b7"
GRID = "#e1e0d9"

PPC_COLS = [
    "prob_stay_common_rewarded",
    "prob_stay_rare_rewarded",
    "prob_stay_common_not_rewarded",
    "prob_stay_rare_not_rewarded",
]


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
        "pdf.fonttype": 42,  # embed TrueType so text stays editable
    })


def save(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.png")
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return values.std(ddof=1) / np.sqrt(len(values))


# ------------------------------------------------------------------
# BIC data
# ------------------------------------------------------------------

def load_baseline_bics():
    df = pd.read_csv(DATA_CSV)
    return df.groupby("participant")["baseline_bic"].first().to_dict()


def load_group_bics():
    with open(GROUP_BICS) as fp:
        bics = json.load(fp)["individual_BIC"]
    return dict(zip(GROUP_PARTICIPANTS, bics))


def load_individual_bics():
    bics = {}
    for path in glob.glob(str(INDIV_BICS_DIR / "iter*_participant*.json")):
        pid = int(re.search(r"participant(\d+)\.json$", path).group(1))
        with open(path) as fp:
            results = json.load(fp)
        best = min((r.get("metric_value", np.inf) for r in results),
                   default=np.inf)
        bics[pid] = min(best, bics.get(pid, np.inf))
    return {pid: bic for pid, bic in bics.items() if np.isfinite(bic)}


def plot_bic_comparison():
    baseline = load_baseline_bics()
    group = load_group_bics()
    individual = load_individual_bics()

    pids = sorted(set(baseline) & set(group) & set(individual))
    print(f"BIC comparison over n={len(pids)} participants")

    models = [
        ("Hybrid", [baseline[p] for p in pids], MODEL_COLORS["hybrid"]),
        ("GeCCo\n(group)", [group[p] for p in pids], MODEL_COLORS["group"]),
        ("GeCCo\n(individual)", [individual[p] for p in pids],
         MODEL_COLORS["individual"]),
    ]

    fig, ax = plt.subplots(figsize=(3.2, 3.4))
    x = np.arange(len(models))
    means = [np.mean(v) for _, v, _ in models]
    errs = [sem(v) for _, v, _ in models]

    ax.bar(x, means, width=0.75, color=[c for _, _, c in models], zorder=2)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.errorbar(x, means, yerr=errs, fmt="o", markersize=5, color="black",
                ecolor="black", elinewidth=1.6, capsize=0, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models], fontsize=11)
    ax.set_ylabel("BIC")
    ax.set_ylim(360, 450)
    ax.set_yticks(np.arange(360, 451, 20))
    ax.tick_params(axis="x", length=0)

    save(fig, "bic_comparison_hybrid_vs_gecco")


# ------------------------------------------------------------------
# PPC stay-probability plots
# ------------------------------------------------------------------

def draw_stay_probability(ax, df, color, show_ylabel=True, show_xlabel=True,
                          reference_means=None, annotate_shades=False):
    """Grouped bars: previous outcome on x, transition type as shade.

    Dark shade = common transition, light shade = rare transition,
    following the reference paper figures.
    """
    dark, light = color, LIGHT_SHADES.get(color, lighten(color))
    means = [np.nanmean(df[c]) for c in PPC_COLS]
    errs = [sem(df[c]) for c in PPC_COLS]
    # PPC_COLS order: common/r, rare/r, common/nr, rare/nr
    positions = [-0.2, 0.2, 0.8, 1.2]
    colors = [dark, light] * 2

    ax.bar(positions, means, width=0.38, color=colors, zorder=2)
    ax.errorbar(positions, means, yerr=errs, fmt="none",
                ecolor="#444444", elinewidth=1.5, capsize=0, zorder=3)

    if annotate_shades:
        ax.text(positions[0], 0.05, "common", rotation=90, ha="center",
                va="bottom", fontsize=11, color="white", zorder=4)
        ax.text(positions[1], 0.05, "rare", rotation=90, ha="center",
                va="bottom", fontsize=11, color=INK, zorder=4)

    if reference_means is not None:
        ax.hlines(reference_means, np.array(positions) - 0.19,
                  np.array(positions) + 0.19, color="black",
                  linestyle=(0, (3, 2)), linewidth=1.4, zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["yes", "no"])
    if show_xlabel:
        ax.set_xlabel("rewarded")
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    if show_ylabel:
        ax.set_ylabel("Stay probability")
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return means


def plot_single_ppc(csv_path, title, color, stem):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(2.3, 3.2))
    draw_stay_probability(ax, df, color, annotate_shades=True)
    ax.set_title(title)
    save(fig, stem)


def plot_ppc_comparison():
    humans = pd.read_csv(PPC_HUMANS)
    group = pd.read_csv(PPC_GROUP)
    indiv = pd.read_csv(PPC_INDIV)

    fig, axes = plt.subplots(1, 3, figsize=(6.4, 3.2), sharey=True)
    draw_stay_probability(axes[0], humans, MODEL_COLORS["humans"],
                          annotate_shades=True)
    draw_stay_probability(axes[1], group, MODEL_COLORS["group"],
                          show_ylabel=False, show_xlabel=False)
    draw_stay_probability(axes[2], indiv, MODEL_COLORS["individual"],
                          show_ylabel=False, show_xlabel=False)

    axes[0].set_title("Humans")
    axes[1].set_title("GeCCo\n(group)")
    axes[2].set_title("GeCCo\n(individual)")
    fig.subplots_adjust(wspace=0.15)

    save(fig, "ppc_humans_vs_gecco")


if __name__ == "__main__":
    set_style()
    plot_bic_comparison()
    plot_single_ppc(PPC_INDIV, "GeCCo (individual)",
                    MODEL_COLORS["individual"], "ppc_gecco_individual")
    plot_single_ppc(PPC_GROUP, "GeCCo (group)",
                    MODEL_COLORS["group"], "ppc_gecco_group")
    plot_ppc_comparison()
    print(f"Figures written to {FIG_DIR}")
