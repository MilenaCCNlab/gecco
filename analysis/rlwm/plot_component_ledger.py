"""Component ledger vs the canonical baseline (plot-span, 30 participants).

Every component that separates the library-composed model from the canonical
RLWM baseline, with its signed BIC contribution per age group (+ = improves
fit / lowers BIC for that group):

  REMOVED from baseline (leave-one-out ablation, full - ablation):
    capacity scaling, uniform lapse, load-independent WM decay
  ADDED by the library (single-module gain over the backbone):
    graded/asymmetric WM update, choice habits, etc.

Inputs: baseline_ablation_bics.json (removals) + per-participant search logs
(additions). Removals and additions are on the same BIC scale but come from
different references (full baseline vs backbone); the divider marks that.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent / "figures"
LC = ROOT / "results/rlwm_individual/library_composition"
SEARCH = Path("/private/tmp/claude-501/-Users-akshay-projects-gecco/"
              "420634c3-f29d-483d-ab87-6f2cf48ae8c1/scratchpad/perpid_plotspan")

YOUNG_C, OLD_C, INK = "#40baec", "#708190", "#0b0b0b"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

PIDS = list(range(15)) + list(range(36, 51))
young = [p for p in PIDS if p < 36]
old = [p for p in PIDS if p >= 36]

# ---- removals (leave-one-out on baseline) ----
abl = json.load(open(LC / "baseline_ablation_bics.json"))
def rem(name, grp):
    return float(np.mean([abl["full"][str(p)] - abl[name][str(p)] for p in grp]))

def rem_n(name, grp):  # # of group helped by >1 BIC by REMOVING the component
    return sum((abl["full"][str(p)] - abl[name][str(p)]) > 1 for p in grp)

# ---- additions (single-module gain over backbone) ----
gains = defaultdict(dict)
for pdir in sorted(SEARCH.iterdir()):
    if not (pdir / "search_log.jsonl").exists():
        continue
    pid = int(pdir.name[1:])
    recs = [json.loads(l) for l in (pdir / "search_log.jsonl").read_text().splitlines()]
    bb = next(r["mean_bic"] for r in recs if r["candidate_id"] == "backbone")
    for r in recs:
        if len(r["module_ids"]) == 1:
            gains[r["module_ids"][0]][pid] = bb - r["mean_bic"]
def add(m, grp):
    return float(np.mean([gains[m][p] for p in grp if p in gains[m]]))

def add_n(m, grp):  # # of group helped by >1 BIC by ADDING the module (over backbone)
    return sum(gains[m].get(p, 0) > 1 for p in grp)

# ledger: (label, kind, young, old, young_n, old_n) — n = of 15 helped by >1 BIC
rows = [
    ("REMOVE capacity scaling", "rm", rem("no_capacity", young), rem("no_capacity", old), rem_n("no_capacity", young), rem_n("no_capacity", old)),
    ("REMOVE uniform lapse", "rm", rem("no_lapse", young), rem("no_lapse", old), rem_n("no_lapse", young), rem_n("no_lapse", old)),
    ("REMOVE load-indep. WM decay", "rm", rem("no_decay", young), rem("no_decay", old), rem_n("no_decay", young), rem_n("no_decay", old)),
    ("ADD choice stickiness/persev.", "add", add("action_stickiness", young), add("action_stickiness", old), add_n("action_stickiness", young), add_n("action_stickiness", old)),
    ("ADD graded WM update/decay", "add", add("unified_wm_update_decay", young), add("unified_wm_update_decay", old), add_n("unified_wm_update_decay", young), add_n("unified_wm_update_decay", old)),
    ("ADD arbitration-scaled WM upd.", "add", add("arbitration_scaled_wm_update", young), add("arbitration_scaled_wm_update", old), add_n("arbitration_scaled_wm_update", young), add_n("arbitration_scaled_wm_update", old)),
    ("ADD asymmetric WM update", "add", add("wm_asymmetric_update_p1", young), add("wm_asymmetric_update_p1", old), add_n("wm_asymmetric_update_p1", young), add_n("wm_asymmetric_update_p1", old)),
]

labels = [r[0] for r in rows]
y = np.arange(len(rows))[::-1]
h = 0.38
fig, ax = plt.subplots(figsize=(8.4, 4.6))
for yi, (_, kind, gy, go, ny, no) in zip(y, rows):
    ax.barh(yi + h/2, gy, height=h, color=YOUNG_C,
            edgecolor="white", zorder=2)
    ax.barh(yi - h/2, go, height=h, color=OLD_C, edgecolor="white", zorder=2)
    for val, off, cnt in [(gy, h/2, ny), (go, -h/2, no)]:
        sgn = "+" if val >= 0 else "−"
        ax.text(val + (0.3 if val >= 0 else -0.3), yi + off,
                "%s%.1f  (%d/15)" % (sgn, abs(val), cnt), va="center",
                ha="left" if val >= 0 else "right", fontsize=7.5, color=INK)
ax.axvline(0, color=INK, lw=1.0)
ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 4)  # room for (x/15) labels
# divider between removals (top 3) and additions
ax.axhline(y[3] + 0.5, color="0.6", lw=0.8, ls=":")
ax.text(ax.get_xlim()[1], y[0] + 0.55, "removed from baseline (+ = removal lowers BIC)",
        ha="right", fontsize=8, color="#4a5560", style="italic")
ax.text(ax.get_xlim()[1], y[3] - 0.45, "added by library (+ = addition lowers BIC vs backbone)",
        ha="right", fontsize=8, color="#4a5560", style="italic")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("BIC improvement contributed  (+ better fit)", fontsize=10)
ax.set_title("Component ledger vs canonical baseline, by age group", fontsize=11.5)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(facecolor=YOUNG_C, label="young (18–36, n=15)"),
                   Patch(facecolor=OLD_C, label="old (46–85, n=15)")],
          loc="lower right", fontsize=8.5, frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "library_component_ledger.png")
fig.savefig(FIG_DIR / "library_component_ledger.pdf")
plt.close(fig)

# ---- overall (all 30) single-series companion ----
ALL = PIDS
rows_all = [
    ("REMOVE capacity scaling", "rm", rem("no_capacity", ALL), rem_n("no_capacity", ALL)),
    ("REMOVE uniform lapse", "rm", rem("no_lapse", ALL), rem_n("no_lapse", ALL)),
    ("REMOVE load-indep. WM decay", "rm", rem("no_decay", ALL), rem_n("no_decay", ALL)),
    ("ADD choice stickiness/persev.", "add", add("action_stickiness", ALL), add_n("action_stickiness", ALL)),
    ("ADD graded WM update/decay", "add", add("unified_wm_update_decay", ALL), add_n("unified_wm_update_decay", ALL)),
    ("ADD arbitration-scaled WM upd.", "add", add("arbitration_scaled_wm_update", ALL), add_n("arbitration_scaled_wm_update", ALL)),
    ("ADD asymmetric WM update", "add", add("wm_asymmetric_update_p1", ALL), add_n("wm_asymmetric_update_p1", ALL)),
]
ya = np.arange(len(rows_all))[::-1]
fig, ax = plt.subplots(figsize=(8.0, 4.4))
for yi, (_, kind, g, ncnt) in zip(ya, rows_all):
    ax.barh(yi, g, height=0.6, color="#2b6cb8" if kind == "add" else "#1b9e91",
            edgecolor="white", zorder=2)
    sgn = "+" if g >= 0 else "−"
    ax.text(g + (0.3 if g >= 0 else -0.3), yi, "%s%.1f  (%d/30)" % (sgn, abs(g), ncnt),
            va="center", ha="left" if g >= 0 else "right", fontsize=8.5, color=INK)
ax.axvline(0, color=INK, lw=1.0)
ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 3)  # room for (x/30) labels
ax.axhline(ya[3] + 0.5, color="0.6", lw=0.8, ls=":")
ax.set_yticks(ya)
ax.set_yticklabels([r[0] for r in rows_all], fontsize=9)
ax.set_xlabel("BIC improvement contributed  (+ better fit)", fontsize=10)
ax.set_title("Component ledger vs canonical baseline (all 30 participants)", fontsize=11.5)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(facecolor="#1b9e91", label="removed from baseline"),
                   Patch(facecolor="#2b6cb8", label="added by library")],
          loc="lower right", fontsize=8.5, frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "library_component_ledger_overall.png")
fig.savefig(FIG_DIR / "library_component_ledger_overall.pdf")
plt.close(fig)
print("overall (all 30):")
for lbl, kind, g, ncnt in rows_all:
    print("  %-30s %+.1f (%d/30)" % (lbl, g, ncnt))

for kind_lbl, kk in [("removals (full-ablation, +=removal helps)", "rm"),
                     ("additions (gain over backbone)", "add")]:
    print(kind_lbl + ":")
    for lbl, kind, gy, go, ny, no in rows:
        if kind == kk:
            print("  %-30s young %+.1f (%d/15)  old %+.1f (%d/15)"
                  % (lbl, gy, ny, go, no))
print("saved library_component_ledger")

# ---------------- poster-ready age-split ledger ----------------
CLEAN = {  # drop the REMOVE/ADD prefix (section bands carry that), tidy names
    "REMOVE capacity scaling": "Capacity scaling",
    "REMOVE uniform lapse": "Uniform lapse",
    "REMOVE load-indep. WM decay": "Load-independent WM decay",
    "ADD choice stickiness/persev.": "Choice stickiness / perseveration",
    "ADD graded WM update/decay": "Graded WM update / decay",
    "ADD arbitration-scaled WM upd.": "Arbitration-scaled WM update",
    "ADD asymmetric WM update": "Asymmetric WM update",
}
# light / dark shades of the composed-library blue (#2b6cb8)
Y_C, O_C = "#86bce6", "#17456f"           # young = lighter, older = darker
BAND_RM, BAND_ADD = "#f3f5f4", "#eef2f7"  # faint neutral section bands
GRID = "#e1e0d9"

fig, ax = plt.subplots(figsize=(11.0, 5.6))
yp = np.arange(len(rows))[::-1]
hh = 0.36
div = yp[3] + 0.5                      # between removals (top 3) and additions
top, bot = yp[0] + 0.6, yp[-1] - 0.6
ax.axhspan(div, top, color=BAND_RM, zorder=0)
ax.axhspan(bot, div, color=BAND_ADD, zorder=0)
xmin, xmax = -5.0, 28.0
for yi, (lbl, kind, gy, go, ny, no) in zip(yp, rows):
    ax.barh(yi + hh/2, gy, height=hh, color=Y_C, edgecolor="white", lw=1.2, zorder=3)
    ax.barh(yi - hh/2, go, height=hh, color=O_C, edgecolor="white", lw=1.2, zorder=3)
    for val, off, cnt in [(gy, hh/2, ny), (go, -hh/2, no)]:
        sgn = "+" if val >= 0 else "-"
        txt = r"$\mathbf{%s%.1f}$  (%d/15)" % (sgn, abs(val), cnt)
        ax.annotate(txt, (val, yi + off), xytext=(5 if val >= 0 else -5, 0),
                    textcoords="offset points", va="center",
                    ha="left" if val >= 0 else "right", fontsize=11.5,
                    color=INK, zorder=4)
ax.axvline(0, color=INK, lw=1.2, zorder=2)
ax.axhline(div, color="0.72", lw=0.9, ls=(0, (4, 3)), zorder=1)
for gx in range(5, 26, 5):
    ax.axvline(gx, color=GRID, lw=0.8, zorder=0)
ax.set_yticks(yp)
ax.set_yticklabels([CLEAN[r[0]] for r in rows], fontsize=12.5)
ax.set_xlim(xmin, xmax)
ax.set_ylim(bot, top)
ax.set_xticks(range(0, 26, 5))
ax.set_xlabel("BIC improvement contributed   ( +  better fit )", fontsize=13.5)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", labelsize=11.5)
# vertical section labels in the left margin (no collision with bars/labels)
ax.text(xmin + 0.5, (div + top) / 2, "REMOVED\nFROM BASELINE", rotation=90,
        va="center", ha="center", fontsize=10, fontweight="bold",
        color="#008181", linespacing=0.95, zorder=4)
ax.text(xmin + 0.5, (bot + div) / 2, "ADDED\nBY LIBRARY", rotation=90,
        va="center", ha="center", fontsize=10, fontweight="bold",
        color="#2b6cb8", linespacing=0.95, zorder=4)
ax.legend(handles=[Patch(facecolor=Y_C, label="Young (18–36, n = 15)"),
                   Patch(facecolor=O_C, label="Older (46–85, n = 15)")],
          loc="upper right", fontsize=11.5, frameon=False,
          bbox_to_anchor=(1.0, 0.99))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "library_component_ledger_poster.png", dpi=400)
fig.savefig(FIG_DIR / "library_component_ledger_poster.pdf")
plt.close(fig)
print("saved library_component_ledger_poster")
