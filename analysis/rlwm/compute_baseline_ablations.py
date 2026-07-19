"""Leave-one-out ablation of the canonical baseline: remove each of its three
'extra' components (capacity scaling, lapse, uniform WM decay) one at a time,
refit per participant under the plot-span protocol (blocks<5, rewards>=0,
30 pids), and measure BIC improvement over the full 6-param baseline. Tells
which removal contributes most to the parsimony gain, by age group."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from library_learning.compose.fitting import bic, fit_participant, seed_for
from library_learning.loading import bounds_for_code, exec_model

COLS = ["stimulus", "actions", "rewards", "blocks", "set_sizes"]
PIDS = list(range(15)) + list(range(36, 51))
OUT = Path("results/rlwm_individual/library_composition")

# full baseline body pieces; each ablation removes exactly one component.
def make(params_line, mix_expr, lapse_line, decay_line, doc):
    return '''def cognitive_model(stimulus, actions, rewards, blocks, set_sizes, model_parameters):
    """
    %s
    """
    %s
    nA = 3
    log_loss = 0.0
    eps = 1e-10
    for b in np.unique(blocks):
        m = blocks == b
        bs, ba, br = stimulus[m], actions[m], rewards[m]
        nS = int(set_sizes[m][0])
        q = (1.0/nA)*np.ones((nS, nA)); w = (1.0/nA)*np.ones((nS, nA)); w_0 = (1.0/nA)*np.ones((nS, nA))
        for t in range(len(bs)):
            s, a, r = int(bs[t]), int(ba[t]), float(br[t])
            if 0 <= s < nS and 0 <= a < nA:
                erl = np.exp(beta*(q[s]-np.max(q[s]))); prl = erl/np.sum(erl)
                ewm = np.exp(50.0*(w[s]-np.max(w[s]))); pwm = ewm/np.sum(ewm)
                mix = %s
                probs = mix*pwm + (1.0-mix)*prl
                %s
                log_loss -= np.log(probs[a]+eps)
                q[s,a] += learning_rate*(r-q[s,a]); w[s,a] = r
            %s
    return log_loss
''' % (doc, params_line, mix_expr, lapse_line, decay_line)

FULL = make("learning_rate, beta, wm_weight, wm_decay, capacity, lapse = model_parameters",
            "wm_weight*min(1.0, capacity/float(nS))",
            "probs = (1.0-lapse)*probs + lapse/nA",
            "w += wm_decay*(w_0-w)",
            "b1[0,1] beta[0,10] wm_weight[0,1] wm_decay[0,1] capacity[1,6] lapse[0,1]")
NO_CAP = make("learning_rate, beta, wm_weight, wm_decay, lapse = model_parameters",
              "wm_weight",
              "probs = (1.0-lapse)*probs + lapse/nA",
              "w += wm_decay*(w_0-w)",
              "b1[0,1] beta[0,10] wm_weight[0,1] wm_decay[0,1] lapse[0,1]")
NO_LAPSE = make("learning_rate, beta, wm_weight, wm_decay, capacity = model_parameters",
                "wm_weight*min(1.0, capacity/float(nS))",
                "probs = probs",
                "w += wm_decay*(w_0-w)",
                "b1[0,1] beta[0,10] wm_weight[0,1] wm_decay[0,1] capacity[1,6]")
NO_DECAY = make("learning_rate, beta, wm_weight, capacity, lapse = model_parameters",
                "wm_weight*min(1.0, capacity/float(nS))",
                "probs = (1.0-lapse)*probs + lapse/nA",
                "w = w",
                "b1[0,1] beta[0,10] wm_weight[0,1] capacity[1,6] lapse[0,1]")

df = pd.read_csv("data/rlwm.csv"); df = df[(df.blocks < 5) & (df.rewards >= 0)]
variants = {"full": FULL, "no_capacity": NO_CAP, "no_lapse": NO_LAPSE, "no_decay": NO_DECAY}
res = {}
for name, src in variants.items():
    func = exec_model(src, "cognitive_model"); bnds = bounds_for_code(src)
    res[name] = {}
    for pid in PIDS:
        d = df[df.participant == pid]
        inp = [d[c].to_numpy() for c in COLS]
        r = fit_participant(func, inp, bnds, seed_for("abl:%s" % name, pid))
        res[name][pid] = bic(r["nll"], len(bnds), len(d))
    print("%s mean %.2f" % (name, np.mean(list(res[name].values()))))

young = [p for p in PIDS if p < 36]; old = [p for p in PIDS if p >= 36]
print("\nBIC improvement from removing each component (full - ablation; + = removal helps):")
for name in ["no_capacity", "no_lapse", "no_decay"]:
    dy = np.mean([res["full"][p]-res[name][p] for p in young])
    do = np.mean([res["full"][p]-res[name][p] for p in old])
    dall = np.mean([res["full"][p]-res[name][p] for p in PIDS])
    print("  remove %-10s all %+5.1f | young %+5.1f | old %+5.1f" % (name.replace("no_",""), dall, dy, do))
json.dump({k: {str(p): v for p, v in d.items()} for k, d in res.items()},
          open(OUT / "baseline_ablation_bics.json", "w"), indent=2)
