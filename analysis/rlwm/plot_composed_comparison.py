"""bic_comparison_baseline_vs_gecco + the library-composed program.

Extends figures/bic_comparison_baseline_vs_gecco with a fourth bar: the
composed program from the 2026-07-18 library-composition run
(results/rlwm_individual/library_composition/composed_model.txt), fit under
the IDENTICAL protocol behind the stored bars — blocks < 5 only, missed
trials dropped (rewards >= 0), same 30 participants (pids 0-14 + 36-50),
per-participant parameter fits. Stored baseline/group/individual CSVs are
used as-is. Note: full-trial held-out results live in
results/rlwm_individual/library_composition/report.html and are NOT
comparable to these numbers (different data span).

Run from the repo root:
  PYTHONPATH=. gecco-env/bin/python analysis/rlwm/plot_composed_comparison.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from library_learning.compose.fitting import bic, fit_participant, seed_for
from library_learning.loading import bounds_for_code, exec_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
FIG_DIR = ANALYSIS_DIR / "figures"
OUT = PROJECT_ROOT / "results" / "rlwm_individual" / "library_composition"
CSV = OUT / "composed_plotspan_bics.csv"
COLS = ["stimulus", "actions", "rewards", "blocks", "set_sizes"]
PIDS = list(range(15)) + list(range(36, 51))

TEAL, BLUE, GRAY, DBLUE = "#008181", "#40baec", "#708190", "#2b6cb8"
INK = "#0b0b0b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})


def fit_composed_plotspan():
    src = (OUT / "composed_model.txt").read_text()
    func = exec_model(src, "cognitive_model")
    bounds = bounds_for_code(src)
    df = pd.read_csv(PROJECT_ROOT / "data" / "rlwm.csv")
    df = df[df.blocks < 5]
    rows = []
    for pid in PIDS:
        d = df[df.participant == pid]
        d = d[d.rewards >= 0]
        inputs = [d[c].to_numpy() for c in COLS]
        res = fit_participant(func, inputs, bounds,
                              seed_for("plotspan:composed", pid))
        rows.append({"participant": pid,
                     "bic": bic(res["nll"], len(bounds), len(d)),
                     "nll": res["nll"], "n_trials": len(d)})
        print("p%-3d composed plot-span BIC %7.2f" % (pid, rows[-1]["bic"]))
    pd.DataFrame(rows).to_csv(CSV, index=False)


def sem(v):
    v = np.asarray(v, dtype=float)
    return np.nanstd(v, ddof=1) / np.sqrt(np.sum(~np.isnan(v)))


def main():
    if not CSV.exists():
        fit_composed_plotspan()
    models = [
        ("Baseline", pd.read_csv(ANALYSIS_DIR / "baseline_bics.csv")["bic"], TEAL),
        ("GeCCo\n(group)", pd.read_csv(ANALYSIS_DIR / "group_bics.csv")["bic"], GRAY),
        ("Composed\n(library)", pd.read_csv(CSV)["bic"], DBLUE),
        ("GeCCo\n(individual)",
         pd.read_csv(ANALYSIS_DIR / "individual_bics.csv")["bic"], BLUE),
    ]
    print("BIC n per model:", [len(v) for _, v, _ in models])
    print({n.replace("\n", " "): round(float(np.mean(v)), 2)
           for n, v, _ in models})

    fig, ax = plt.subplots(figsize=(3.3, 3.2))
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
    fig.savefig(FIG_DIR / "bic_comparison_baseline_vs_gecco_composed.png")
    fig.savefig(FIG_DIR / "bic_comparison_baseline_vs_gecco_composed.pdf")
    plt.close(fig)
    print("saved figures/bic_comparison_baseline_vs_gecco_composed.{png,pdf}")


def main_with_individual_composed():
    """Variant: add the per-participant ('individual') library composition as a
    bar, and show GeCCo-individual as a horizontal reference line."""
    import json
    if not CSV.exists():
        fit_composed_plotspan()
    perpid = json.load(open(OUT / "perpid_plotspan_results.json"))
    pp = np.array([r["library_bic"] for r in perpid])  # pid order 0-14,36-50
    indiv = pd.read_csv(ANALYSIS_DIR / "individual_bics.csv")["bic"].to_numpy()

    models = [
        ("RLWM", pd.read_csv(ANALYSIS_DIR / "baseline_bics.csv")["bic"].to_numpy(), TEAL),
        ("GeCCo\n(group)", pd.read_csv(ANALYSIS_DIR / "group_bics.csv")["bic"].to_numpy(), GRAY),
        ("Composed\nlibrary\n(shared)", pd.read_csv(CSV)["bic"].to_numpy(), DBLUE),
        ("Composed\nlibrary\n(individual)", pp, BLUE),
    ]
    print("variant means:", {n.replace(chr(10), " "): round(float(np.mean(v)), 2)
                             for n, v, _ in models},
          "| individual line:", round(float(indiv.mean()), 2))

    fig, ax = plt.subplots(figsize=(4.0, 3.3))
    x = np.arange(len(models))
    means = [np.mean(v) for _, v, _ in models]
    errs = [sem(v) for _, v, _ in models]
    ax.bar(x, means, width=0.72, color=[c for _, _, c in models], zorder=2)
    ax.errorbar(x, means, yerr=errs, fmt="o", markersize=5, color="black",
                ecolor="black", elinewidth=1.6, capsize=0, zorder=3)
    # GeCCo individual as a reference line
    im = float(indiv.mean())
    ax.axhline(im, color=INK, lw=1.4, ls=(0, (5, 3)), zorder=4)
    ax.text(-0.5, im, "GeCCo individual", va="bottom", ha="left",
            fontsize=9, color=INK, style="italic")
    ax.set_xlim(-0.6, len(models) - 0.4)
    lo = np.floor((min(means + [im]) - max(errs)) / 20) * 20 - 20
    hi = np.ceil((max(means + [im]) + max(errs)) / 20) * 20
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models], fontsize=9.5)
    ax.set_ylabel("BIC")
    ax.set_ylim(lo, hi)
    ax.tick_params(axis="x", length=0)
    fig.savefig(FIG_DIR / "bic_comparison_baseline_vs_gecco_composed_individual.png")
    fig.savefig(FIG_DIR / "bic_comparison_baseline_vs_gecco_composed_individual.pdf")
    plt.close(fig)
    print("saved figures/bic_comparison_baseline_vs_gecco_composed_individual.{png,pdf}")


if __name__ == "__main__":
    main()
    main_with_individual_composed()
