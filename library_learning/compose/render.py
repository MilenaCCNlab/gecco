# library_learning/compose/render.py
"""Assemble backbone + module slots into one standalone cognitive_model.

The backbone is the completed gecco template model (MB stage-1 policy, MF
stage-2, single learning_rate/beta TD updates), hardened for -1 missed
trials. Modules append statements to APPEND_SLOTS and/or replace
OVERRIDE_SLOTS expressions. Everything renders to flat numpy-only source
whose docstring bounds and unpack line stay parseable by gecco's regexes.
"""
import textwrap

import numpy as np

from .inventory import APPEND_SLOTS, OVERRIDE_SLOTS, Param
from ..loading import exec_model

BACKBONE_PARAMS = [Param("learning_rate", (0.0, 1.0)), Param("beta", (0.0, 10.0))]

DEFAULT_OVERRIDES = {
    "q2_init": "np.zeros((2, 2))",
    "stage1_values": "q_stage1_mb",
    "stage2_values": "q_stage2_mf[s_idx]",
    "stage1_temp": "beta",
    "stage2_temp": "beta",
    "stage1_update": "q_stage1_mf[a1] += learning_rate * delta_stage1",
    "stage2_update": "q_stage2_mf[s_idx, a2] += learning_rate * delta_stage2",
}

BACKBONE_PARAM_DOCS = {
    "learning_rate": "TD learning rate.",
    "beta": "Softmax inverse temperature.",
}


def candidate_id(module_ids):
    return "+".join(sorted(module_ids)) if module_ids else "backbone"


def candidate_params(inventory, module_ids):
    params = list(BACKBONE_PARAMS)
    for mid in sorted(module_ids):
        params.extend(inventory.module(mid).params)
    return params


def _indent(code, level):
    return textwrap.indent(textwrap.dedent(code).strip(), "    " * level)


def render_candidate(inventory, module_ids):
    mods = [inventory.module(mid) for mid in sorted(module_ids)]
    params = candidate_params(inventory, module_ids)

    overrides = dict(DEFAULT_OVERRIDES)
    for m in mods:
        overrides.update(m.overrides)
    appends = {slot: [] for slot in APPEND_SLOTS}
    for m in mods:
        for slot, code in m.slots.items():
            appends[slot].append(code)

    doc_lines = ["Composed cognitive model: backbone"]
    if mods:
        doc_lines[0] += " + " + ", ".join(m.name for m in mods)
    doc_lines += ["", "Parameters:"]
    for p in params:
        desc = BACKBONE_PARAM_DOCS.get(p.name, "module parameter")
        lo = "%g" % p.bounds[0]
        hi = "%g" % p.bounds[1]
        doc_lines.append("%s: [%s, %s] - %s" % (p.name, lo, hi, desc))
    docstring = "\n    ".join(doc_lines)
    unpack = ", ".join(p.name for p in params)
    if len(params) == 1:
        unpack += ","

    def block(slot, level):
        parts = [_indent(c, level) for c in appends[slot]]
        return ("\n".join(parts) + "\n") if parts else ""

    src = '''def cognitive_model(action_1, state, action_2, reward, model_parameters):
    """
    {docstring}
    """
    {unpack} = model_parameters
    n_trials = len(action_1)
    transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    q_stage1_mf = np.zeros(2)
    q_stage2_mf = {q2_init}
    log_loss = 0.0
    eps = 1e-10
{init}
    for trial in range(n_trials):
        a1 = int(action_1[trial])
        s_idx = int(state[trial])
        a2 = int(action_2[trial])
        r = float(reward[trial])

        max_q_stage2 = np.max(q_stage2_mf, axis=1)
        q_stage1_mb = transition_matrix @ max_q_stage2
{pre_stage1}
        stage1_values = {stage1_values}
        logits_1 = ({stage1_temp}) * stage1_values
{stage1_logits_extra}
        exp_q1 = np.exp(logits_1)
        probs_1 = exp_q1 / np.sum(exp_q1)
        if a1 != -1:
            log_loss -= np.log(probs_1[a1] + eps)

        if s_idx != -1 and a2 != -1:
            stage2_values = {stage2_values}
            logits_2 = ({stage2_temp}) * stage2_values
{stage2_logits_extra}
            exp_q2 = np.exp(logits_2)
            probs_2 = exp_q2 / np.sum(exp_q2)
            log_loss -= np.log(probs_2[a2] + eps)

        if a1 != -1 and s_idx != -1 and a2 != -1:
            delta_stage1 = q_stage2_mf[s_idx, a2] - q_stage1_mf[a1]
            {stage1_update}
            delta_stage2 = r - q_stage2_mf[s_idx, a2]
            {stage2_update}
{update_extra}
{post_trial}
    return log_loss
'''.format(
        docstring=docstring,
        unpack=unpack,
        q2_init=overrides["q2_init"],
        stage1_values=overrides["stage1_values"],
        stage1_temp=overrides["stage1_temp"],
        stage2_values=overrides["stage2_values"],
        stage2_temp=overrides["stage2_temp"],
        stage1_update=overrides["stage1_update"],
        stage2_update=overrides["stage2_update"],
        init=block("init", 1),
        pre_stage1=block("pre_stage1", 2),
        stage1_logits_extra=block("stage1_logits_extra", 2),
        stage2_logits_extra=block("stage2_logits_extra", 3),
        update_extra=block("update_extra", 3),
        post_trial=block("post_trial", 2),
    )
    # drop blank slot lines so the source stays tidy
    src = "\n".join(line for line in src.splitlines() if line.strip() != "") + "\n"
    return src


SMOKE_DATA = {
    "action_1": np.array([0, 1, -1, 0, 1, 0, -1, 1]),
    "state":    np.array([0, 1, -1, 1, 0, 0, -1, 1]),
    "action_2": np.array([1, 0, -1, -1, 1, 0, -1, 0]),
    "reward":   np.array([1, 0, -1, -1, 1, 0, -1, 1]),
}


def smoke_check(source, inventory=None, module_ids=None, params=None):
    """Exec + run on dummy data with -1 missed trials; return NLL or raise."""
    func = exec_model(source, "cognitive_model")
    if params is None:
        import re
        bounds = re.findall(r"\[\s*([\-\d.eE+]+)\s*,\s*([\-\d.eE+]+)\s*\]",
                            source.split('"""')[1])
        params = [(float(lo) + float(hi)) / 2.0 for lo, hi in bounds]
    nll = float(func(SMOKE_DATA["action_1"], SMOKE_DATA["state"],
                     SMOKE_DATA["action_2"], SMOKE_DATA["reward"], params))
    if not np.isfinite(nll):
        raise ValueError("smoke_check: non-finite NLL %r" % nll)
    return nll
