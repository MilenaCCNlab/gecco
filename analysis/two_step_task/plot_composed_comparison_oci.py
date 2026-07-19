"""BIC comparison for the ocibalanced150 run, matching the RLWM figure
analysis/rlwm/plot_composed_comparison.py::main_with_individual_composed
(bic_comparison_baseline_vs_gecco_composed_individual): four bars
  Hybrid, GeCCo (group), Library (group), Library (individual)
plus GeCCo-individual as a dashed reference line. Pooled + split by OCI
(Low vs High). All from already-frozen eval + coverage outputs, no new fitting.

  Hybrid              <- test_results.json  model 'hybrid'  (Daw hybrid baseline)
  GeCCo (group)       <- test_results.json  model 'group'
  Library (group)     <- test_results.json  model 'composed' (bare-bone shared)
  Library (individual)<- reconstruction_results.json 'library_bic' (per-pid best)
  GeCCo individual    <- test_results.json  model 'individual'  (dashed line)

Run: PYTHONPATH=. gecco-env/bin/python analysis/two_step_task/plot_composed_comparison_oci.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LC = ROOT / "results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual/library_composition"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

# RLWM palette (plot_composed_comparison.py)
TEAL, BLUE, GRAY, DBLUE, INK = "#008181", "#40baec", "#708190", "#2b6cb8", "#0b0b0b"
MODELS = [  # bars, in RLWM order/colors
    ("Hybrid", TEAL, ("tr", "hybrid")),
    ("GeCCo\n(group)", GRAY, ("tr", "group")),
    ("Library\n(group)", DBLUE, ("tr", "composed")),
    ("Library\n(individual)", BLUE, ("rc", "library_bic")),
]
INDIV = ("tr", "individual")  # GeCCo individual -> dashed reference line
TERTILES = ["Low", "High"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 12,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


def sem(v):
    v = np.asarray(v, float)
    return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def load():
    tr = json.loads((LC / "test_results.json").read_text())["rows"]
    tr_by = {}
    for r in tr:
        if r["set"] == "test":
            tr_by.setdefault(r["model"], {})[int(r["participant"])] = r["bic"]
    rc = {int(r["pid"]): r for r in json.loads((LC / "reconstruction_results.json").read_text())}
    manifest = json.loads((ROOT / "data/ocd/ocibalanced150_manifest.json").read_text())["participants"]
    tert = {p["participant"]: p["tertile"] for p in manifest if p["split"] == "test"}
    pids = sorted(tert)

    def series(src):
        kind, key = src
        return {p: (tr_by[key][p] if kind == "tr" else rc[p][key]) for p in pids}

    data = {lab: series(src) for lab, _, src in MODELS}
    data["_indiv"] = series(INDIV)
    return data, tert, pids


def panel(ax, means, errs, colors, labels, indiv_mean, title=None, ylim=None):
    x = np.arange(len(means))
    ax.bar(x, means, width=0.72, color=colors, zorder=2)
    ax.errorbar(x, means, yerr=errs, fmt="o", markersize=5, color="black",
                ecolor="black", elinewidth=1.6, capsize=0, zorder=3)
    ax.axhline(indiv_mean, color=INK, lw=1.4, ls=(0, (5, 3)), zorder=4)
    ax.text(-0.5, indiv_mean, "GeCCo individual", va="bottom", ha="left",
            fontsize=8.5, color=INK, style="italic")
    ax.set_xlim(-0.6, len(means) - 0.4)
    if ylim is None:
        ylim = (np.floor((min(means + [indiv_mean]) - max(errs)) / 20) * 20 - 15,
                np.ceil((max(means + [indiv_mean]) + max(errs)) / 20) * 20)
    ax.set_ylim(*ylim)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("BIC")
    if title:
        ax.set_title(title, fontsize=12)


def main():
    data, tert, pids = load()
    labels = [m[0] for m in MODELS]
    colors = [m[1] for m in MODELS]

    def stat(grp):
        m = [np.mean([data[l][p] for p in grp]) for l in labels]
        e = [sem([data[l][p] for p in grp]) for l in labels]
        return m, e, float(np.mean([data["_indiv"][p] for p in grp]))

    # ---- pooled ----
    m, e, im = stat(pids)
    fig, ax = plt.subplots(figsize=(3.9, 3.3))
    panel(ax, m, e, colors, labels, im)
    fig.savefig(FIG / "bic_comparison_baseline_vs_gecco_composed.png")
    fig.savefig(FIG / "bic_comparison_baseline_vs_gecco_composed.pdf")
    plt.close(fig)

    # ---- by OCI (Low vs High), shared y ----
    per = {t: stat([p for p in pids if tert[p] == t]) for t in TERTILES}
    all_m = [v for t in TERTILES for v in per[t][0] + [per[t][2]]]
    all_e = [v for t in TERTILES for v in per[t][1]]
    ylim = (np.floor((min(all_m) - max(all_e)) / 20) * 20 - 15,
            np.ceil((max(all_m) + max(all_e)) / 20) * 20)
    fig, axes = plt.subplots(1, len(TERTILES), figsize=(4.0 * len(TERTILES), 3.3), sharey=True)
    for ax, t in zip(axes, TERTILES):
        mm, ee, ii = per[t]
        n = sum(1 for p in pids if tert[p] == t)
        panel(ax, mm, ee, colors, labels, ii, title="%s OCI (n=%d)" % (t, n), ylim=ylim)
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG / "bic_comparison_baseline_vs_gecco_composed_by_oci.png")
    fig.savefig(FIG / "bic_comparison_baseline_vs_gecco_composed_by_oci.pdf")
    plt.close(fig)

    print("pooled test mean BIC:")
    for l, mn in zip(labels, m):
        print("  %-22s %.1f" % (l.replace("\n", " "), mn))
    print("  %-22s %.1f (dashed line)" % ("GeCCo individual", im))
    for t in TERTILES:
        mm, ee, ii = per[t]
        print("%s: %s  | indiv %.1f" % (t,
              {l.replace(chr(10), " "): round(v, 1) for l, v in zip(labels, mm)}, ii))
    print("saved bic_comparison_baseline_vs_gecco_composed{,_by_oci}.{png,pdf}")


if __name__ == "__main__":
    main()
