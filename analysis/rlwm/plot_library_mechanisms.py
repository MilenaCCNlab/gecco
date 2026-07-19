"""Library-mechanism summary figures (plot-span, 30 matched participants).

Fig 1 (library_mechanisms_impact_frequency): top-5 mechanisms by marginal
BIC impact (single-module gain over backbone, from the per-participant
greedy search logs) and top mechanisms by selection frequency in the final
compositions — colored by whether the mechanism is absent from the
literature baseline and/or group GeCCo.

Fig 2 (library_mechanisms_age): the same two quantities split by age group
(15 young, 15 old).

Inputs: per-participant search logs + perpid_plotspan_results.json produced
by the 2026-07-19 plot-span run (paths below).
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent / "figures"
RESULTS = PROJECT_ROOT / ("results/rlwm_individual/library_composition/"
                          "perpid_plotspan_results.json")
SEARCH_LOGS = Path("/private/tmp/claude-501/-Users-akshay-projects-gecco/"
                   "420634c3-f29d-483d-ab87-6f2cf48ae8c1/scratchpad/"
                   "perpid_plotspan")

BLUE, TEAL, GRAY, INK = "#2b6cb8", "#1b9e91", "#bfc7ce", "#0b0b0b"
# absence status vs the two reference models ("both" = in neither model;
# "baseline" = missing from baseline only — group GeCCo renormalizes WM on
# reward trials; "none" = present in both; set_size_wm_decay is 'partial':
# both models decay, but at a load-INdependent rate).
MISSING = {
    "action_stickiness": "both", "choice_perseveration": "both",
    "unified_wm_update_decay": "both", "arbitration_scaled_wm_update": "both",
    "wm_asymmetric_update_p1": "both", "wm_asymmetric_update_p2": "both",
    "asymmetric_fixed_wm_update": "both", "eligibility_trace": "both",
    "load_dependent_wm_learning": "both", "load_dependent_wm_decay_global": "both",
    "wm_normalization": "baseline",
    "set_size_wm_decay": "partial",
    "wm_perfect_encoding_on_reward": "none", "load_scaled_wm_weight": "none",
}
COLOR = {"both": BLUE, "baseline": TEAL, "partial": GRAY, "none": GRAY}
HATCH = {"partial": "//"}
SHORT = {
    "action_stickiness": "action\nstickiness",
    "choice_perseveration": "choice\nperseveration",
    "unified_wm_update_decay": "unified WM\nupdate/decay",
    "arbitration_scaled_wm_update": "arbitration-scaled\nWM update",
    "wm_asymmetric_update_p1": "asymmetric\nWM update (a)",
    "wm_asymmetric_update_p2": "asymmetric\nWM update (b)",
    "asymmetric_fixed_wm_update": "asymmetric fixed\nWM update",
    "wm_normalization": "WM\nnormalization",
    "set_size_wm_decay": "load-scaled\nWM decay",
    "wm_perfect_encoding_on_reward": "one-shot WM\nencoding",
    "load_scaled_wm_weight": "load-scaled\nWM weight",
    "eligibility_trace": "eligibility\ntrace",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.labelsize": 12,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})


def load_gains():
    """Per-pid single-module gain over backbone from the search logs."""
    gains = defaultdict(dict)
    for pdir in sorted(SEARCH_LOGS.iterdir()):
        logf = pdir / "search_log.jsonl"
        if not logf.exists():
            continue
        pid = int(pdir.name[1:])
        recs = [json.loads(l) for l in logf.read_text().splitlines()]
        bb = next(r["mean_bic"] for r in recs if r["candidate_id"] == "backbone")
        for r in recs:
            if len(r["module_ids"]) == 1:
                gains[r["module_ids"][0]][pid] = bb - r["mean_bic"]
    return gains


gains = load_gains()
rows = json.load(open(RESULTS))
freq = Counter(m for r in rows for m in r["modules"])
young_pids = {r["pid"] for r in rows if r["pid"] < 36}


def bars(ax, names, values, title, xlabel):
    y = np.arange(len(names))[::-1]
    for yi, n, v in zip(y, names, values):
        st = MISSING[n]
        ax.barh(yi, v, height=0.72, color=COLOR[st],
                hatch=HATCH.get(st, ""), edgecolor="white", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[n] for n in names], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="y", length=0)


# ---------------- Fig 1: impact + frequency ----------------
imp_rank = sorted(gains, key=lambda m: -np.mean(list(gains[m].values())))[:5]
frq_rank = [m for m, _ in freq.most_common(5)]

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
bars(axes[0], imp_rank,
     [np.mean(list(gains[m].values())) for m in imp_rank],
     "Top 5 by impact", "mean BIC improvement over backbone")
bars(axes[1], frq_rank, [freq[m] for m in frq_rank],
     "Top 5 by selection frequency", "participants using it (of 30)")
legend = [Patch(facecolor=BLUE, label="absent from baseline AND group GeCCo"),
          Patch(facecolor=TEAL, label="absent from baseline only"),
          Patch(facecolor=GRAY, hatch="//", edgecolor="white",
                label="simplified (load-independent) version present"),
          Patch(facecolor=GRAY, label="present in both")]
fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.14))
fig.tight_layout()
fig.savefig(FIG_DIR / "library_mechanisms_impact_frequency.png")
fig.savefig(FIG_DIR / "library_mechanisms_impact_frequency.pdf")
plt.close(fig)

# ---------------- Fig 2: split by age ----------------
imp6 = sorted(gains, key=lambda m: -np.mean(list(gains[m].values())))[:6]
frq6 = [m for m, _ in freq.most_common(6)]


def age_pair(ax, names, val_fn, title, xlabel):
    y = np.arange(len(names))[::-1]
    h = 0.36
    for yi, n in zip(y, names):
        vy, vo = val_fn(n)
        ax.barh(yi + h / 2, vy, height=h, color="#40baec", zorder=2)
        ax.barh(yi - h / 2, vo, height=h, color="#708190", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[n] for n in names], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="y", length=0)


def gain_by_age(m):
    g = gains[m]
    return (np.mean([v for p, v in g.items() if p in young_pids]),
            np.mean([v for p, v in g.items() if p not in young_pids]))


def freq_by_age(m):
    cy = sum(1 for r in rows if r["pid"] in young_pids and m in r["modules"])
    co = sum(1 for r in rows if r["pid"] not in young_pids and m in r["modules"])
    return cy, co


fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))
age_pair(axes[0], imp6, gain_by_age, "Impact by age",
         "mean BIC improvement over backbone")
age_pair(axes[1], frq6, freq_by_age, "Selection frequency by age",
         "participants using it (of 15)")
legend = [Patch(facecolor="#40baec", label="young (18–36, n=15)"),
          Patch(facecolor="#708190", label="old (46–85, n=15)")]
fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
fig.savefig(FIG_DIR / "library_mechanisms_age.png")
fig.savefig(FIG_DIR / "library_mechanisms_age.pdf")
plt.close(fig)

print("saved library_mechanisms_impact_frequency + library_mechanisms_age")
