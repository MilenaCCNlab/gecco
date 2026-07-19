# library_learning/compose/render_rlwm.py
"""Assemble RLWM backbone + module slots into one standalone cognitive_model.

The backbone is the shared skeleton the gecco RLWM fill-in template forced on
every seed model: block loop with per-block q/w/w_0 reset to 1/nA, RL softmax
(beta), near-deterministic WM softmax (temp 50), probability-level RL/WM
mixture (wm_weight), delta-rule Q update — hardened for -2 missed trials
(likelihood and updates skipped; post_trial still runs). Parallel sibling of
render.py (two-step); that file stays untouched.
"""
import ast
import textwrap

import numpy as np

from .inventory import Param
from .inventory_rlwm import APPEND_SLOTS
from .render import _indent, candidate_id  # noqa: F401 (candidate_id re-exported)
from ..loading import exec_model, extract_unpack_names, parse_bounds

EMPTY_SLOT_SENTINEL = "###EMPTY_SLOT###"

BACKBONE_PARAMS = [Param("learning_rate", (0.0, 1.0)),
                   Param("beta", (0.0, 10.0)),
                   Param("wm_weight", (0.0, 1.0))]

DEFAULT_OVERRIDES = {
    "q_init": "(1.0 / nA) * np.ones((nS, nA))",
    "w_init": "(1.0 / nA) * np.ones((nS, nA))",
    "rl_values": "q[s]",
    "wm_values": "w[s]",
    "rl_temp": "beta",
    "wm_temp": "50.0",
    "mix_weight": "wm_weight",
    "rl_update": "q[s, a] += learning_rate * delta",
    "wm_update": "w[s, a] = r",
}

BACKBONE_PARAM_DOCS = {
    "learning_rate": "RL delta-rule learning rate.",
    "beta": "RL softmax inverse temperature.",
    "wm_weight": "WM weight in the RL/WM policy mixture.",
}


def candidate_params(inventory, module_ids):
    params = list(BACKBONE_PARAMS)
    for mid in sorted(module_ids):
        params.extend(inventory.module(mid).params)
    return params


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

    for m in mods:
        for slot, code in m.slots.items():
            tree = ast.parse(textwrap.dedent(code))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and "\n" in node.value):
                    raise ValueError(
                        "module snippet contains a multi-line string constant, "
                        "unsupported: %s/%s" % (m.id, slot))

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
        if not parts:
            return _indent(EMPTY_SLOT_SENTINEL, level)
        return "\n".join(parts) + "\n"

    src = '''def cognitive_model(stimulus, actions, rewards, blocks, set_sizes, model_parameters):
    """
    {docstring}
    """
    {unpack} = model_parameters
    nA = 3
    log_loss = 0.0
    eps = 1e-10
{init}
    for b in np.unique(blocks):
        block_mask = blocks == b
        block_states = stimulus[block_mask]
        block_actions = actions[block_mask]
        block_rewards = rewards[block_mask]
        nS = int(set_sizes[block_mask][0])
        q = {q_init}
        w = {w_init}
        w_0 = (1.0 / nA) * np.ones((nS, nA))
{block_init}
        for trial in range(len(block_states)):
            s = int(block_states[trial])
            a = int(block_actions[trial])
            r = float(block_rewards[trial])
            if 0 <= s < nS and 0 <= a < nA:
{pre_choice}
                rl_values = {rl_values}
                logits_rl = ({rl_temp}) * rl_values
{rl_logits_extra}
                exp_rl = np.exp(logits_rl - np.max(logits_rl))
                probs_rl = exp_rl / np.sum(exp_rl)
                wm_values = {wm_values}
                logits_wm = ({wm_temp}) * wm_values
{wm_logits_extra}
                exp_wm = np.exp(logits_wm - np.max(logits_wm))
                probs_wm = exp_wm / np.sum(exp_wm)
                mix = {mix_weight}
                probs = mix * probs_wm + (1.0 - mix) * probs_rl
{probs_extra}
                log_loss -= np.log(probs[a] + eps)
                delta = r - q[s, a]
{rl_update}
{wm_update}
{update_extra}
{post_trial}
    return log_loss
'''.format(
        docstring=docstring,
        unpack=unpack,
        q_init=overrides["q_init"],
        w_init=overrides["w_init"],
        rl_values=overrides["rl_values"],
        rl_temp=overrides["rl_temp"],
        wm_values=overrides["wm_values"],
        wm_temp=overrides["wm_temp"],
        mix_weight=overrides["mix_weight"],
        # statement overrides are block-indented (dedent + reindent) so
        # multi-line statements (e.g. if/else WM updates) stay syntactic
        rl_update=_indent(overrides["rl_update"], 4),
        wm_update=_indent(overrides["wm_update"], 4),
        init=block("init", 1),
        block_init=block("block_init", 2),
        pre_choice=block("pre_choice", 4),
        rl_logits_extra=block("rl_logits_extra", 4),
        wm_logits_extra=block("wm_logits_extra", 4),
        probs_extra=block("probs_extra", 4),
        update_extra=block("update_extra", 4),
        post_trial=block("post_trial", 3),
    )
    # drop only the sentinel lines left by empty append slots; every other
    # line (including blank lines inside the docstring or a snippet) is kept
    # exactly as rendered.
    src = "\n".join(
        line for line in src.splitlines() if line.strip() != EMPTY_SLOT_SENTINEL
    ) + "\n"
    return src


# Two blocks (set sizes 3 and 6), -2-coded missed trials in both.
SMOKE_DATA = {
    "stimulus":  np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 3]),
    "actions":   np.array([0, 2, -2, 1, 0, 2, 1, -2, 0, 2, 1, 0, 2, 1]),
    "rewards":   np.array([1, 0, -2, 1, 1, 0, 0, -2, 1, 0, 1, 1, 0, 1]),
    "blocks":    np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
    "set_sizes": np.array([3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6]),
}


def smoke_check(source, params=None):
    """Exec + run on dummy data with -2 missed trials; return NLL or raise."""
    func = exec_model(source, "cognitive_model")
    if params is None:
        names = extract_unpack_names(source)
        bounds_map = parse_bounds(source, names)
        params = [(bounds_map[n][0] + bounds_map[n][1]) / 2.0 for n in names]
    nll = float(func(SMOKE_DATA["stimulus"], SMOKE_DATA["actions"],
                     SMOKE_DATA["rewards"], SMOKE_DATA["blocks"],
                     SMOKE_DATA["set_sizes"], params))
    if not np.isfinite(nll):
        raise ValueError("smoke_check: non-finite NLL %r" % nll)
    return nll
