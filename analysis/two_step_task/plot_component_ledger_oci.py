"""Library component ledger for the ocibalanced150 run, split by OCI tertile
(low / medium / high) — two-step analogue of analysis/rlwm/plot_component_ledger.py
(which split by age).

  REMOVED from the Daw-hybrid base (leave-one-out, full - ablation; + = removing
    that hybrid component lowers BIC): from baseline_ablation_bics_oci.json.
  ADDED by the library (single-module gain over backbone; + = adding lowers BIC):
    from the coverage per-pid search logs (reconstruction/p*/search_log.jsonl).

Both on a BIC scale but different references (full hybrid base vs bare backbone);
the divider marks that. Test participants only (50), grouped by OCI tertile.

Run: PYTHONPATH=. gecco-env/bin/python analysis/two_step_task/plot_component_ledger_oci.py
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
ADIR = Path(__file__).resolve().parent
FIG = ADIR / "figures"; FIG.mkdir(exist_ok=True)
LC = ROOT / "results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual/library_composition"
ABL = ADIR / "baseline_ablation_bics_oci.json"

# low / high OCI — sequential teal->blue (medium dropped for a clean contrast)
C_LOW, C_HIGH, INK = "#9dcece", "#2b6cb8", "#0b0b0b"
GROUPS = ["Low", "High"]
TCOL = {"Low": C_LOW, "High": C_HIGH}
N_ADD = 7  # top library modules to show

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

BASE = ["stage1_stickiness", "eligibility_trace", "mb_mf_mixture",
        "separate_learning_rates", "separate_stage2_beta"]
RM_LABEL = {"stage1_stickiness": "REMOVE stage-1 stickiness",
            "eligibility_trace": "REMOVE eligibility trace",
            "mb_mf_mixture": "REMOVE MB/MF mixture",
            "separate_learning_rates": "REMOVE separate learning rates",
            "separate_stage2_beta": "REMOVE separate stage-2 beta"}


def tertiles():
    m = json.loads((ROOT / "data/ocd/ocibalanced150_manifest.json").read_text())["participants"]
    return {p["participant"]: p["tertile"] for p in m if p["split"] == "test"}


def load_additions():
    """gain[module][pid] = backbone_bic - single_module_bic (from coverage logs)."""
    gains = defaultdict(dict)
    for pdir in sorted((LC / "reconstruction").iterdir()):
        log = pdir / "search_log.jsonl"
        if not log.exists():
            continue
        pid = int(pdir.name[1:])
        recs = [json.loads(l) for l in log.read_text().splitlines()]
        bb = next((r["mean_bic"] for r in recs if r["candidate_id"] == "backbone"), None)
        if bb is None:  # no backbone baseline logged for this pid — skip it
            continue
        for r in recs:
            if len(r["module_ids"]) == 1:
                gains[r["module_ids"][0]][pid] = bb - r["mean_bic"]  # + = adding helps
    return gains


def load_removals():
    abl = json.loads(ABL.read_text())
    # rem[module][pid] = full - ablation ; + = removing that component lowers BIC
    rem = {}
    for m in BASE:
        key = "no_%s" % m
        rem[m] = {int(p): abl["full"][p] - abl[key][p] for p in abl["full"]}
    return rem


def grp_mean(d, pids):
    vals = [d[p] for p in pids if p in d]
    return float(np.mean(vals)) if vals else 0.0


def module_pretty(m):
    return "ADD " + m.replace("_", " ")


def main():
    tert = tertiles()
    pids = sorted(tert)
    by_t = {t: [p for p in pids if tert[p] == t] for t in GROUPS}
    gains = load_additions()
    have_rem = ABL.exists()
    rem = load_removals() if have_rem else {}

    # pick top-N added modules by pooled mean gain
    pooled_gain = {m: grp_mean(g, pids) for m, g in gains.items()}
    top_add = sorted(pooled_gain, key=pooled_gain.get, reverse=True)[:N_ADD]

    # v2 selection: modules where Low vs High OCI differ most, with a floor so
    # at least one group shows a reasonable improvement (else a big % difference
    # between two tiny gains would dominate).
    FLOOR = 8.0
    add_low = {m: grp_mean(gains[m], by_t["Low"]) for m in gains}
    add_high = {m: grp_mean(gains[m], by_t["High"]) for m in gains}
    reasonable = [m for m in gains if max(add_low[m], add_high[m]) >= FLOOR]
    diff_add = sorted(reasonable, key=lambda m: abs(add_high[m] - add_low[m]),
                      reverse=True)[:N_ADD]

    # de-duplicated selection: collapse each excludes-substitute family to its
    # single best-gaining representative (+ keep singletons), so each DISTINCT
    # mechanism idea appears once.
    inv_path = LC / "module_inventory.json"
    dedup_add, dedup_labels = [], {}
    if inv_path.exists():
        for fam in excludes_families(inv_path):
            fam_g = [m for m in fam if m in gains]
            if not fam_g:
                continue
            rep = max(fam_g, key=lambda m: pooled_gain.get(m, -1e9))
            if pooled_gain.get(rep, 0) <= 0:
                continue
            dedup_add.append(rep)
            lbl = family_label(fam)
            if lbl:
                dedup_labels[rep] = "ADD " + lbl
        dedup_add = sorted(dedup_add, key=lambda m: pooled_gain[m], reverse=True)[:N_ADD]

    # removals: keep only components whose removal IMPROVES fit (pooled full-abl > 0)
    rm_modules = [m for m in BASE if have_rem and grp_mean(rem[m], pids) > 0]
    n_rm = len(rm_modules)

    # ledger rows: kept removals then additions (top_add)
    rows = []
    for m in rm_modules:
        rows.append((RM_LABEL[m], "rm", {t: grp_mean(rem[m], by_t[t]) for t in by_t}))
    for m in top_add:
        rows.append((module_pretty(m), "add", {t: grp_mean(gains[m], by_t[t]) for t in by_t}))

    # ---- grouped-by-OCI ledger (Low vs High) ----
    y = np.arange(len(rows))[::-1]
    h = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 0.55 * len(rows) + 1.4))
    offs = {"Low": h / 2, "High": -h / 2}
    for yi, (_, kind, vals) in zip(y, rows):
        for t in GROUPS:
            ax.barh(yi + offs[t], vals[t], height=h, color=TCOL[t], edgecolor="white", zorder=2)
    ax.axvline(0, color=INK, lw=1.0)
    if n_rm and n_rm < len(rows):
        ax.axhline(y[n_rm - 1] - 0.5, color="0.6", lw=0.8, ls=":")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("BIC improvement contributed  (+ better fit)", fontsize=10)
    ax.set_title("Library component ledger, by OCI group (test participants)", fontsize=11.5)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(facecolor=TCOL[t], label="%s OCI (n=%d)" % (t, len(by_t[t])))
                       for t in GROUPS],
              loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "library_component_ledger.png")
    fig.savefig(FIG / "library_component_ledger.pdf")
    plt.close(fig)

    # ---- overall (pooled) ----
    rows_all = []
    for m in rm_modules:
        rows_all.append((RM_LABEL[m], "rm", grp_mean(rem[m], pids)))
    for m in top_add:
        rows_all.append((module_pretty(m), "add", grp_mean(gains[m], pids)))
    ya = np.arange(len(rows_all))[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 0.5 * len(rows_all) + 1.2))
    for yi, (_, kind, g) in zip(ya, rows_all):
        ax.barh(yi, g, height=0.6, color=("#2b6cb8" if kind == "add" else "#1b9e91"),
                edgecolor="white", zorder=2)
        sgn = "+" if g >= 0 else "−"
        ax.text(g + (0.3 if g >= 0 else -0.3), yi, "%s%.1f" % (sgn, abs(g)),
                va="center", ha="left" if g >= 0 else "right", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK, lw=1.0)
    if n_rm and n_rm < len(rows_all):
        ax.axhline(ya[n_rm - 1] - 0.5, color="0.6", lw=0.8, ls=":")
    ax.set_yticks(ya); ax.set_yticklabels([r[0] for r in rows_all], fontsize=9)
    ax.set_xlabel("BIC improvement contributed  (+ better fit)", fontsize=10)
    ax.set_title("Library component ledger (all 50 test participants)", fontsize=11.5)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor="#2b6cb8", label="added by library (vs backbone)")]
    if have_rem:
        handles.insert(0, Patch(facecolor="#1b9e91", label="removed from Daw-hybrid base"))
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "library_component_ledger_overall.png")
    fig.savefig(FIG / "library_component_ledger_overall.pdf")
    plt.close(fig)

    # ---- poster-ready Low-vs-High OCI ledgers (two selections) ----
    _poster(rm_modules, top_add, rem, gains, by_t,
            "library_component_ledger_poster",
            "Library component ledger, by OCI group (test participants)")
    _poster(rm_modules, diff_add, rem, gains, by_t,
            "library_component_ledger_poster_ocidiff",
            "Mechanisms most different between Low and High OCI")
    if dedup_add:
        _poster(rm_modules, dedup_add, rem, gains, by_t,
                "library_component_ledger_dedup",
                "Distinct mechanisms (substitute families collapsed to best representative)",
                add_label=dedup_labels)

    print("removals present:", have_rem)
    print("top added modules (pooled gain over backbone):")
    for m in top_add:
        print("  %-34s %+.1f" % (m, pooled_gain[m]))
    print("\nv2: modules where Low/High OCI differ most (floor %.0f):" % FLOOR)
    for m in diff_add:
        print("  %-34s low %+.1f  high %+.1f  |diff| %.1f"
              % (m, add_low[m], add_high[m], abs(add_high[m] - add_low[m])))
    print("saved library_component_ledger{,_overall,_poster,_poster_ocidiff}.{png,pdf}")


def _clean(label):
    return label.replace("REMOVE ", "").replace("ADD ", "").replace("_", " ")


# human names for the excludes-substitute families (keyed by a signature member)
FAMILY_SIG = {
    "mb_mf_mixture": "MB/MF control",
    "stage1_stickiness": "Choice stickiness / perseveration",
    "value_decay_to_zero": "Value decay / forgetting",
    "direct_mf_update": "MF update / learning rates",
}


def excludes_families(inv_path):
    """Connected components of the symmetrized module `excludes` graph =
    substitute families (alternative formulations of one mechanism)."""
    inv = json.loads(Path(inv_path).read_text())
    ids = {m["id"] for m in inv["modules"]}
    adj = {i: set() for i in ids}
    for m in inv["modules"]:
        for e in m.get("excludes", []):
            if e in ids:
                adj[m["id"]].add(e); adj[e].add(m["id"])
    seen, comps = set(), []
    for i in ids:
        if i in seen:
            continue
        stack, comp = [i], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.add(x)
            stack.extend(adj[x] - seen)
        comps.append(comp)
    return comps


def family_label(fam):
    for sig, name in FAMILY_SIG.items():
        if sig in fam:
            return name
    return None  # singleton / unnamed -> caller uses the representative's name


def _npos(d, grp):  # count of participants in grp with positive contribution
    return sum(1 for p in grp if p in d and d[p] > 0)


def _poster(rm_modules, add_modules, rem, gains, by_t, fname, title, add_label=None):
    LOW_C, HIGH_C = "#86bce6", "#17456f"          # low = lighter, high = darker
    BAND_RM, BAND_ADD, GRID = "#f3f5f4", "#eef2f7", "#e1e0d9"
    add_label = add_label or {}
    n_rm = len(rm_modules)
    nlow, nhigh = len(by_t["Low"]), len(by_t["High"])
    raw = {**{m: rem[m] for m in rm_modules}, **{m: gains[m] for m in add_modules}}
    order = rm_modules + add_modules  # top-to-bottom
    rows = ([(RM_LABEL[m], "rm") for m in rm_modules]
            + [(add_label.get(m, module_pretty(m)), "add") for m in add_modules])
    yp = np.arange(len(rows))[::-1]
    hh = 0.36
    fig, ax = plt.subplots(figsize=(11.0, 0.62 * len(rows) + 1.6))
    top, bot = yp[0] + 0.6, yp[-1] - 0.6
    if n_rm and n_rm < len(rows):
        div = yp[n_rm - 1] - 0.5
        ax.axhspan(div, top, color=BAND_RM, zorder=0)
        ax.axhspan(bot, div, color=BAND_ADD, zorder=0)
        ax.axhline(div, color="0.72", lw=0.9, ls=(0, (4, 3)), zorder=1)
    all_vals = [grp_mean(raw[m], by_t[t]) for m in order for t in GROUPS]
    xmax = max(all_vals) * 1.18
    xmin = min(min(all_vals), 0) - 0.04 * xmax - 6
    for i, (m, (lbl, kind)) in enumerate(zip(order, rows)):
        yi = yp[i]
        gl, gh = grp_mean(raw[m], by_t["Low"]), grp_mean(raw[m], by_t["High"])
        cl, ch = _npos(raw[m], by_t["Low"]), _npos(raw[m], by_t["High"])
        ax.barh(yi + hh / 2, gl, height=hh, color=LOW_C, edgecolor="white", lw=1.2, zorder=3)
        ax.barh(yi - hh / 2, gh, height=hh, color=HIGH_C, edgecolor="white", lw=1.2, zorder=3)
        for val, off, cnt, tot in [(gl, hh / 2, cl, nlow), (gh, -hh / 2, ch, nhigh)]:
            sgn = "+" if val >= 0 else "−"
            ax.annotate("%s%.1f  (%d/%d)" % (sgn, abs(val), cnt, tot), (val, yi + off),
                        xytext=(5 if val >= 0 else -5, 0), textcoords="offset points",
                        va="center", ha="left" if val >= 0 else "right",
                        fontsize=10.5, color=INK, zorder=4)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    step = 20 if xmax > 60 else 10
    for gx in range(step, int(xmax) + 1, step):
        ax.axvline(gx, color=GRID, lw=0.8, zorder=0)
    ax.set_yticks(yp); ax.set_yticklabels([_clean(r[0]) for r in rows], fontsize=12.5)
    ax.set_xlim(xmin, xmax); ax.set_ylim(bot, top)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("BIC improvement contributed   ( +  better fit )", fontsize=13.5)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=11.5)
    if n_rm and n_rm < len(rows):
        lx = xmin / 2.0  # centered in the gap between the left axis and the 0 line
        ax.text(lx, (div + top) / 2, "REMOVED\nFROM BASELINE",
                rotation=90, va="center", ha="center", fontsize=10, color="#008181", linespacing=0.95, zorder=4)
        ax.text(lx, (bot + div) / 2, "ADDED\nBY LIBRARY",
                rotation=90, va="center", ha="center", fontsize=10, color="#2b6cb8", linespacing=0.95, zorder=4)
    ax.legend(handles=[Patch(facecolor=LOW_C, label="Low OCI (n = %d)" % nlow),
                       Patch(facecolor=HIGH_C, label="High OCI (n = %d)" % nhigh)],
              loc="upper right", fontsize=11.5, frameon=False, bbox_to_anchor=(1.0, 0.99))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / (fname + ".png"), dpi=400)
    fig.savefig(FIG / (fname + ".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
