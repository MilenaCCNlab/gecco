"""Participant splits: library seeds from the group config; deterministic
OCI-stratified validation/test partition of the held-out participants."""
import json
from pathlib import Path

import pandas as pd
import yaml

from ..config import REPO_ROOT, Target, resolve_target


def parse_split(value, unique_ids):
    """Mirror of gecco/prepare_data/io.py::parse_split (index slice over sorted ids)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        start_str, end_str = value[1:-1].split(":")
        start = int(start_str) if start_str else None
        end = int(end_str) if end_str else None
        return unique_ids[start:end]
    raise ValueError("unsupported split spec: %r" % (value,))


def _group_config(group_dir, config_dir=None):
    config_dir = Path(config_dir) if config_dir else REPO_ROOT / "config"
    task_name = Path(group_dir).name
    for yaml_path in sorted(config_dir.glob("*.yaml")):
        try:
            cfg = yaml.safe_load(yaml_path.read_text())
        except Exception:
            continue
        if isinstance(cfg, dict) and cfg.get("task", {}).get("name") == task_name:
            return cfg
    raise FileNotFoundError("no config with task.name == %r" % task_name)


def group_split_pids(group_dir, config_dir=None, data_path=None):
    cfg = _group_config(group_dir, config_dir)
    data_sec = cfg["data"]
    path = Path(data_path) if data_path else REPO_ROOT / data_sec["path"]
    df = pd.read_csv(path)
    unique_ids = sorted(df[data_sec.get("id_column", "participant")].unique().tolist())
    splits = data_sec["splits"]
    prompt = sorted(parse_split(splits["prompt"], unique_ids))
    ev = sorted(parse_split(splits["eval"], unique_ids))
    heldout = sorted(parse_split(splits["test"], unique_ids))
    return {"prompt": prompt, "eval": ev, "heldout": heldout,
            "seed": sorted(prompt + ev)}


def make_splits(target, group_dir, out_dir=None, oci_column="oci"):
    """Write splits.json.

    - seed_pids: prompt+eval participants (library extraction)
    - composition_validation_pids: the group config's eval participants
      (composition selection — matches group gecco's information budget)
    - reconstruction_pids: 10 of the held-out, OCI-stratified deterministic
      (held-out pids sorted by (oci, pid); index i % 3 == 1)
    - test_pids: remaining 21 held-out — the only set results are claimed on
    """
    out_dir = Path(out_dir) if out_dir else target.results_dir / "library_composition"
    out_dir.mkdir(parents=True, exist_ok=True)
    pids = group_split_pids(group_dir, data_path=target.data_path)
    df = pd.read_csv(target.data_path)
    oci = df.groupby(target.id_column)[oci_column].first()
    ranked = sorted(pids["heldout"], key=lambda p: (float(oci[p]), p))
    reconstruction = [p for i, p in enumerate(ranked) if i % 3 == 1]
    test = [p for p in pids["heldout"] if p not in reconstruction]
    result = {
        "seed_pids": pids["seed"],
        "composition_validation_pids": pids["eval"],
        "reconstruction_pids": sorted(reconstruction),
        "test_pids": sorted(test),
        "method": "oci-sorted alternation i%3==1",
        "oci_stats": {
            "reconstruction_mean": float(oci[reconstruction].mean()),
            "reconstruction_std": float(oci[reconstruction].std()),
            "test_mean": float(oci[test].mean()),
            "test_std": float(oci[test].std()),
        },
    }
    (out_dir / "splits.json").write_text(json.dumps(result, indent=2))
    return result


def load_splits(target):
    path = target.results_dir / "library_composition" / "splits.json"
    return json.loads(path.read_text())
