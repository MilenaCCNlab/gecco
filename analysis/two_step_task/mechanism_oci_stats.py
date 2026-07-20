"""Which cognitive mechanisms differ with OCI, discovered via the library.

For every library module we have a per-participant importance = BIC gain of that
single module over the bare backbone (module_gains_oci.csv, all 150 pids). We
test each mechanism against OCI two ways:
  - continuous: Spearman(gain, oci) across all 150 (primary; most power),
    Benjamini-Hochberg FDR across the 27 modules.
  - low vs high tertile: mean gain difference (High - Low) + Mann-Whitney U.

Outputs a stats table (mechanism_oci_stats.csv) and a figure
(mechanism_oci_differences.{png,pdf}) — per-module High-Low gain difference,
sorted, FDR-significant mechanisms marked.

Run: PYTHONPATH=. gecco-env/bin/python analysis/two_step_task/mechanism_oci_stats.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

ADIR = Path(__file__).resolve().parent
FIG = ADIR / "figures"; FIG.mkdir(exist_ok=True)
CSV = ADIR / "module_gains_oci.csv"

TEAL, BLUE, GRAY, INK = "#008181", "#40baec", "#708190", "#1a1a1a"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q


def main():
    df = pd.read_csv(CSV)
    recs = []
    for mod, d in df.groupby("module"):
        r, p = spearmanr(d["gain"], d["oci"])
        low = d[d["tertile"] == "Low"]["gain"].to_numpy()
        high = d[d["tertile"] == "High"]["gain"].to_numpy()
        try:
            u_p = mannwhitneyu(high, low, alternative="two-sided").pvalue
        except ValueError:
            u_p = 1.0
        recs.append({"module": mod, "spearman_r": r, "spearman_p": p,
                     "low_mean_gain": low.mean(), "high_mean_gain": high.mean(),
                     "high_minus_low": high.mean() - low.mean(), "mwu_p": u_p,
                     "n": len(d)})
    res = pd.DataFrame(recs)
    res["spearman_q"] = bh_fdr(res["spearman_p"].to_numpy())
    res["mwu_q"] = bh_fdr(res["mwu_p"].to_numpy())
    res = res.sort_values("spearman_p").reset_index(drop=True)
    res.to_csv(ADIR / "mechanism_oci_stats.csv", index=False)

    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(res[["module", "spearman_r", "spearman_p", "spearman_q",
               "high_minus_low", "mwu_p", "mwu_q"]].to_string(index=False,
              float_format=lambda x: "%.3f" % x))
    sig = res[res["spearman_q"] < 0.05]
    print("\nFDR-significant (q<0.05) mechanisms vs OCI:",
          list(sig["module"]) if len(sig) else "NONE")

    # ---- figure: High-Low gain difference per module, sorted, sig marked ----
    plot = res.sort_values("high_minus_low")
    y = np.arange(len(plot))
    colors = [BLUE if v > 0 else TEAL for v in plot["high_minus_low"]]
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(plot) + 1.2))
    ax.barh(y, plot["high_minus_low"], color=colors, edgecolor="white", zorder=2)
    for yi, (_, row) in zip(y, plot.iterrows()):
        star = "*" if row["spearman_q"] < 0.05 else ("†" if row["spearman_p"] < 0.05 else "")
        if star:
            v = row["high_minus_low"]
            ax.text(v + (0.6 if v >= 0 else -0.6), yi, star, va="center",
                    ha="left" if v >= 0 else "right", fontsize=12, color=INK)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels([m.replace("_", " ") for m in plot["module"]], fontsize=8.5)
    ax.set_xlabel("Mechanism importance:  High − Low OCI  (ΔBIC gain over backbone)", fontsize=10)
    ax.set_title("Cognitive mechanisms that differ with OCI (library-discovered, n=150)\n"
                 "* FDR q<0.05   † uncorrected p<0.05", fontsize=10)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(FIG / "mechanism_oci_differences.png")
    fig.savefig(FIG / "mechanism_oci_differences.pdf")
    plt.close(fig)
    print("\nsaved mechanism_oci_stats.csv + mechanism_oci_differences.{png,pdf}")


if __name__ == "__main__":
    main()
