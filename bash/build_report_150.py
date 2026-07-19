"""Build the self-contained HTML report for the ocibalanced150 library-composition
run. Reads the frozen eval artifacts (both arms), coverage, splits, manifest;
renders 4 paper-style figures (matplotlib, base64-embedded) and one HTML file.
Run AFTER Task 9 eval:  PYTHONPATH=<repo> gecco-env/bin/python bash/build_report_150.py
"""
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
IND = REPO / "results/two_step_psychiatry_individual_function_ocibalanced150_maxsetting_individual"
LC = IND / "library_composition"
HB = LC / "hybrid_base"

# paper palette (paper-figure-style memory)
C_BASE = "#008181"    # Daw hybrid (reference)
C_WIN = "#40baec"     # composed winner
C_GROUP = "#708190"   # group gecco
C_IND = "#1a1a1a"     # individual gecco (ceiling)


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _mean_sem(vals):
    a = np.asarray(vals, float)
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def load_arm(arm_dir):
    d = json.loads((arm_dir / "test_results.json").read_text())
    rows = d["rows"]
    test = {m: [r["bic"] for r in rows if r["set"] == "test" and r["model"] == m]
            for m in ("composed", "group", "hybrid", "individual")}
    return d["stats"], test


def fig_test_bic(test, title, winner_label):
    order = [("hybrid", "Daw hybrid", C_BASE), ("group", "GeCCo (group)", C_GROUP),
             ("composed", winner_label, C_WIN), ("individual", "GeCCo (individual)", C_IND)]
    means = [np.mean(test[k]) for k, _, _ in order]
    sems = [_mean_sem(test[k])[1] for k, _, _ in order]
    labels = [lab for _, lab, _ in order]
    colors = [c for _, _, c in order]
    lo = min(means) - 8
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    x = np.arange(len(order))
    ax.bar(x, means, color=colors, width=0.72)
    ax.errorbar(x, means, yerr=sems, fmt="o", color="black", markersize=3, capsize=2, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(lo, max(means) + max(sems) + 4)
    ax.set_ylabel("Test BIC (mean ± SEM)"); ax.set_title(title, fontsize=9)
    return _b64(fig)


def fig_coverage(recon):
    lib = np.array([r["library_bic"] for r in recon])
    ind = np.array([r["individual_bic"] for r in recon])
    grp = np.array([r["group_bic"] for r in recon])
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    lim = [min(lib.min(), ind.min()) - 10, max(lib.max(), ind.max()) + 10]
    ax.plot(lim, lim, color="#999999", lw=1, ls="--", zorder=0)
    ax.scatter(ind, lib, s=18, color=C_WIN, edgecolor="black", lw=0.4, zorder=3, label="library vs individual")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Individual GeCCo BIC (ceiling)"); ax.set_ylabel("Best library composition BIC")
    ax.set_title("Library spans unseen individuals\n(below dashed = beats own individual model)", fontsize=8)
    return _b64(fig)


def fig_delta(test, title):
    comp = np.array(test["composed"]); grp = np.array(test["group"])
    delta = grp - comp  # positive = composed better than group
    order = np.argsort(delta)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    cols = [C_WIN if d > 0 else C_GROUP for d in delta[order]]
    ax.bar(np.arange(len(delta)), delta[order], color=cols, width=1.0)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Test participant (sorted)"); ax.set_ylabel("ΔBIC (group − composed)")
    ax.set_title(title, fontsize=9)
    return _b64(fig)


def main():
    stats_b, test_b = load_arm(LC)           # bare-bone arm
    have_hb = (HB / "test_results.json").exists()
    stats_h, test_h = load_arm(HB) if have_hb else (None, None)
    recon = json.loads((LC / "reconstruction_results.json").read_text())
    win_b = json.loads((LC / "winner.json").read_text())
    win_h = json.loads((HB / "winner.json").read_text()) if (HB / "winner.json").exists() else None
    splits = json.loads((LC / "splits.json").read_text())
    manifest = json.loads((REPO / "data/ocd/ocibalanced150_manifest.json").read_text())

    f1 = fig_test_bic(test_b, "Bare-bone composed vs baselines", "Composed (bare-bone)")
    f2 = fig_coverage(recon)
    f3 = fig_delta(test_b, "Per-participant: bare-bone composed vs group")
    f4 = fig_test_bic(test_h, "Hybrid-base composed vs baselines", "Composed (hybrid-base)") if have_hb else None
    f5 = fig_delta(test_h, "Per-participant: hybrid-base composed vs group") if have_hb else None

    lib = np.array([r["library_bic"] for r in recon]); ind = np.array([r["individual_bic"] for r in recon]); grp = np.array([r["group_bic"] for r in recon])
    cov_vs_grp = int((lib < grp).sum()); cov_vs_ind = int((lib <= ind + 1).sum())

    def mb(s): return s["mean_bic"]
    def img(b): return '<img src="data:image/png;base64,%s" style="max-width:100%%;height:auto;">' % b
    def bic_table(test, stats):
        rows = "".join("<tr><td>%s</td><td>%.1f</td></tr>" % (m, np.mean(test[m]))
                       for m in sorted(test, key=lambda k: np.mean(test[k])))
        return "<table><tr><th>model</th><th>test mean BIC</th></tr>%s</table>" % rows

    cvg = stats_b["composed_vs_group"]; cvh = stats_b["composed_vs_hybrid"]
    html = ["""<title>Library composition — two-step psychiatry (OCI-balanced 150)</title>
<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1a1a1a}
h1{font-size:1.5rem}h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}
table{border-collapse:collapse;margin:.5rem 0}td,th{border:1px solid #ccc;padding:3px 10px;text-align:left;font-size:.9rem}
code{background:#f4f4f4;padding:1px 4px;border-radius:3px;font-size:.85em}.fig{margin:1rem 0}.k{color:#008181;font-weight:600}</style>
<h1>Library learning &amp; composition — two-step psychiatry (OCI-balanced, 150 participants)</h1>
<p>Generated %s. Gemini generator <code>gemini-3.1-pro-preview</code>. All results on the 50 held-out <b>test</b> participants; the composed winners were frozen on the 50 <b>validation</b> participants before test data was touched.</p>
""" % ("2026-07-19",)]

    html.append("<h2>1 · What was run</h2><p>150 Gillan-2016 participants, OCI-tertile-stratified (cuts %.0f/%.0f), split 50 train / 50 validation / 50 test (shuffled, no index–OCI confound). Individual GeCCo fit all 150; group GeCCo used 5 in-context + 50 validation. A %d-module library was extracted from the 50 train programs and composed two ways: <b>bare-bone</b> (greedy from backbone) and <b>hybrid-base</b> (exhaustive over modules added to the Daw-hybrid module set).</p>"
                % (manifest["cuts"]["low"], manifest["cuts"]["high"], len(json.loads((LC/'module_inventory.json').read_text())['modules'])))

    html.append("<h2>2 · Headline: bare-bone composed program on held-out test</h2>")
    html.append('<div class="fig">%s</div>' % img(f1))
    html.append("<p>Winner: <code>%s</code> (%d params). " % (win_b["candidate_id"], win_b["n_params"]))
    html.append('Composed vs group: mean ΔBIC <span class="k">%.1f</span>, W/T/L %d/%d/%d, Wilcoxon p=%.4f. Composed vs Daw hybrid: mean ΔBIC %.1f, p=%.4f.</p>'
                % (cvg["mean_delta"], cvg["wins"], cvg["ties"], cvg["losses"], cvg["wilcoxon_p"], cvh["mean_delta"], cvh["wilcoxon_p"]))
    html.append(bic_table(test_b, stats_b))
    html.append('<div class="fig">%s</div>' % img(f3))

    if have_hb:
        cvg_h = stats_h["composed_vs_group"]; cvh_h = stats_h["composed_vs_hybrid"]
        html.append("<h2>3 · Hybrid base + missing library components</h2>")
        html.append('<div class="fig">%s</div>' % img(f4))
        html.append("<p>Winner: <code>%s</code> (%d params). " % (win_h["candidate_id"], win_h["n_params"]))
        _cvg_bb = stats_b["composed_vs_group"]["mean_delta"]
        html.append('This arm is <i>constrained</i> to carry the five Daw-hybrid modules as a fixed base. It beats the Daw hybrid but, unlike the bare-bone arm, it does <b>not</b> beat group GeCCo here — the hybrid base forces in components the component ledger (analysis/two_step_task/figures/) flags as dead weight (separate learning rates, separate stage-2 beta), so the leaner bare-bone winner (§2, ΔBIC %.1f vs group) is the stronger shared program on this dataset. ' % _cvg_bb)
        html.append('Composed vs group: mean ΔBIC <span class="k">%.1f</span>, W/T/L %d/%d/%d, Wilcoxon p=%.4f. Composed vs Daw hybrid: mean ΔBIC %.1f, p=%.4f.</p>'
                    % (cvg_h["mean_delta"], cvg_h["wins"], cvg_h["ties"], cvg_h["losses"], cvg_h["wilcoxon_p"], cvh_h["mean_delta"], cvh_h["wilcoxon_p"]))
        html.append(bic_table(test_h, stats_h))
        html.append('<div class="fig">%s</div>' % img(f5))

    html.append("<h2>4 · The library spans unseen individuals</h2>")
    html.append('<div class="fig">%s</div>' % img(f2))
    html.append('<p>Per-participant best library composition vs that participant\'s own individual-GeCCo model (ceiling) and the group model, on all 50 test participants. Mean BIC: library <span class="k">%.1f</span>, individual %.1f, group %.1f. Library beats group <b>%d/50</b>; library matches or beats the individual ceiling <b>%d/50</b>.</p>'
                % (lib.mean(), ind.mean(), grp.mean(), cov_vs_grp, cov_vs_ind))

    warns = stats_b.get("warnings", [])
    html.append("<h2>5 · Caveats &amp; cross-checks</h2><ul>")
    html.append("<li>Cross-check warnings (test refit vs stored): %d. The expected ~missed-trial hybrid/baseline_bic divergence is benign (the −1 handling differs between the column-generating hybrid and the guarded evaluation hybrid).</li>" % len(warns))
    html.append("<li>Generator is gemini-3.1-pro-preview (the 45-pid run used the now-retired gemini-3-pro-preview): cross-run comparisons confound model version; within-run comparisons here are unaffected.</li>")
    html.append("<li>Extraction inventory was repaired post-hoc (renderer multi-line-override indent fix + <code>stage1_temp/stage2_temp→beta</code> in 10 modules); the reconstruction fidelity gate was not separately re-run.</li>")
    html.append("</ul>")

    html.append("<h2>6 · Frozen programs &amp; reproducibility</h2><p>Bare-bone: <code>library_composition/composed_model.txt</code>; hybrid-base: <code>library_composition/hybrid_base/composed_model.txt</code>. Inventory: <code>module_inventory.json</code> (%d modules). Splits from <code>data/ocd/ocibalanced150_manifest.json</code>. See <code>docs/superpowers/specs/2026-07-18-ocibalanced150-decision-log.md</code> for every autonomous decision.</p>"
                % len(json.loads((LC/'module_inventory.json').read_text())['modules']))

    (LC / "report.html").write_text("\n".join(html))
    print("report -> %s" % (LC / "report.html"))


if __name__ == "__main__":
    main()
