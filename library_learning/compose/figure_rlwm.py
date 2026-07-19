# library_learning/compose/figure_rlwm.py
"""RLWM comparison figure, paper style (teal reference / blue winner / gray
others). Parallel sibling of figure.py with the canonical RLWM baseline in
the reference slot."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"composed": "#2b6cb8", "canonical": "#1b9e91",
          "group": "#9a9a9a", "individual": "#c4c4c4"}
ORDER = ["canonical", "group", "composed", "individual"]
LABELS = {"canonical": "RLWM\n(Collins & Frank)", "group": "Group GeCCo",
          "composed": "Composed (library)", "individual": "Individual GeCCo\n(ceiling)"}


def plot_comparison(test_results_csv, out_dir, rng_seed=7):
    out_dir = Path(out_dir)
    df = pd.read_csv(test_results_csv)
    df = df[df["set"] == "test"]
    rng = np.random.default_rng(rng_seed)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for i, model in enumerate(ORDER):
        vals = df[df.model == model]["bic"].to_numpy()
        if len(vals) == 0:
            raise ValueError("no test rows for model '%s'" % model)
        ax.bar(i, vals.mean(), width=0.62, color=COLORS[model],
               edgecolor="none", zorder=2)
        x = i + rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(x, vals, s=9, color="0.25", alpha=0.55, lw=0, zorder=3)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABELS[m] for m in ORDER], fontsize=8)
    ax.set_ylabel("BIC (test participants)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=300)
    fig.savefig(out_dir / "comparison.pdf")
    plt.close(fig)
