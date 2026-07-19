"""BIC comparison for the ocibalanced150 run: five models on the 50 held-out
test participants, pooled and split by OCI tertile (low / medium / high) — the
two-step OCI analogue of analysis/rlwm/plot_composed_comparison.py (which pooled
and had no age split here).

Five models (all from already-frozen eval + coverage outputs, no new fitting):
  Baseline (Daw hybrid)   <- test_results.json  model 'hybrid'
  GeCCo (group)           <- test_results.json  model 'group'
  Composed (group)        <- test_results.json  model 'composed'  (bare-bone shared)
  GeCCo (individual)      <- test_results.json  model 'individual'
  Composed (individual)   <- reconstruction_results.json  'library_bic' (per-pid best library)

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

TEAL, BLUE, GRAY, DBLUE, INK = "#008181", "#40baec", "#708190", "#2b6cb8", "#1a1a1a"
MODELS = [  # (label, color, source-key)
    ("Baseline", TEAL, ("tr", "hybrid")),
    ("GeCCo\n(group)", GRAY, ("tr", "group")),
    ("Composed\n(group)", BLUE, ("tr", "composed")),
    ("GeCCo\n(individual)", INK, ("tr", "individual")),
    ("Composed\n(individual)", DBLUE, ("rc", "library_bic")),
]
TERTILES = ["Low", "High"]  # medium dropped for a clean low-vs-high contrast

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK,
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
        if kind == "tr":
            return {p: tr_by[key][p] for p in pids}
        return {p: rc[p][key] for p in pids}

    data = {lab: series(src) for lab, _, src in MODELS}
    return data, tert, pids


def bar_panel(ax, means, errs, colors, labels, title=None, ylim=None):
    x = np.arange(len(means))
    ax.bar(x, means, width=0.75, color=colors, zorder=2)
    ax.errorbar(x, means, yerr=errs, fmt="o", markersize=4, color="black",
                ecolor="black", elinewidth=1.4, capsize=0, zorder=3)
    if ylim is None:
        ylim = (np.floor((min(means) - max(errs)) / 20) * 20 - 10,
                np.ceil((max(means) + max(errs)) / 20) * 20)
    ax.set_ylim(*ylim)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Test BIC")
    if title:
        ax.set_title(title, fontsize=11)


def main():
    data, tert, pids = load()
    labels = [m[0] for m in MODELS]
    colors = [m[1] for m in MODELS]

    # ---- pooled (all 50) ----
    means = [np.mean(list(data[l].values())) for l in labels]
    errs = [sem(list(data[l].values())) for l in labels]
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    bar_panel(ax, means, errs, colors, [l.replace("\n", " ") if len(l) < 12 else l for l in labels])
    fig.savefig(FIG / "bic_comparison_baseline_vs_gecco_composed.png")
    fig.savefig(FIG / "bic_comparison_baseline_vs_gecco_composed.pdf")
    plt.close(fig)

    # ---- by OCI tertile ----
    # precompute one shared y-range so sharey panels never clip a low bar
    per_t = {}
    for t in TERTILES:
        grp = [p for p in pids if tert[p] == t]
        per_t[t] = ([np.mean([data[l][p] for p in grp]) for l in labels],
                    [sem([data[l][p] for p in grp]) for l in labels], len(grp))
    all_m = [v for m, e, _ in per_t.values() for v in m]
    all_e = [v for m, e, _ in per_t.values() for v in e]
    shared_ylim = (np.floor((min(all_m) - max(all_e)) / 20) * 20 - 10,
                   np.ceil((max(all_m) + max(all_e)) / 20) * 20)
    fig, axes = plt.subplots(1, len(TERTILES), figsize=(3.4 * len(TERTILES), 3.3), sharey=True)
    for ax, t in zip(axes, TERTILES):
        m, e, n = per_t[t]
        bar_panel(ax, m, e, colors, ["" for _ in labels],
                  title="%s OCI (n=%d)" % (t, n), ylim=shared_ylim)
        if ax is not axes[0]:
            ax.set_ylabel("")
    # shared legend
    from matplotlib.patches import Patch
    axes[-1].legend(handles=[Patch(facecolor=c, label=l.replace("\n", " ")) for l, c, _ in MODELS],
                    loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "bic_comparison_baseline_vs_gecco_composed_by_oci.png")
    fig.savefig(FIG / "bic_comparison_baseline_vs_gecco_composed_by_oci.pdf")
    plt.close(fig)

    print("pooled test mean BIC:")
    for l, mn in zip(labels, means):
        print("  %-22s %.1f" % (l.replace("\n", " "), mn))
    print("by OCI tertile:")
    for t in TERTILES:
        grp = [p for p in pids if tert[p] == t]
        print("  %s (n=%d): %s" % (t, len(grp),
              {l.replace(chr(10), " "): round(np.mean([data[l][p] for p in grp]), 1) for l in labels}))
    print("saved bic_comparison_baseline_vs_gecco_composed{,_by_oci}.{png,pdf}")


if __name__ == "__main__":
    main()
