"""Target resolution: derive paths and data schema for a results directory.

A "target" is one individual-fit results directory
(e.g. ``results/two_step_psychiatry_individual_function_gemini-3-pro_individual``).
The producing config is located by matching ``task.name`` in ``config/*.yaml``
against the results-dir basename (minus the ``_individual`` suffix that
``gecco/run_gecco.py`` appends for individual fits). From it we take the data
path and ``input_columns`` — which also fixes the model-function arity, so
4-arg (plain) and 5-arg (OCI-covariate) variants are handled uniformly.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_COLUMNS = ["choice_1", "state", "choice_2", "reward"]
DEFAULT_DATA_PATH = "data/two_step_gillan_2016.csv"
DEFAULT_ID_COLUMN = "participant"

# Data columns holding binary event sequences; anything else (e.g. oci, stai)
# is treated as a per-participant covariate when generating synthetic data.
BINARY_SEQUENCE_COLUMNS = {"choice_1", "state", "choice_2", "reward"}


@dataclass
class Target:
    results_dir: Path
    data_path: Path
    input_columns: List[str]
    id_column: str = DEFAULT_ID_COLUMN
    config_path: Optional[Path] = None

    @property
    def models_dir(self) -> Path:
        return self.results_dir / "models"

    @property
    def bics_dir(self) -> Path:
        return self.results_dir / "bics"

    @property
    def params_dir(self) -> Path:
        return self.results_dir / "parameters"

    @property
    def library_dir(self) -> Path:
        return self.results_dir / "cognitive_library"

    @property
    def n_model_args(self) -> int:
        """Expected positional-arg count of cognitive_model (inputs + model_parameters)."""
        return len(self.input_columns) + 1


def _task_name_for(results_dir: Path) -> str:
    name = results_dir.name
    suffix = "_individual"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _find_config(task_name: str, config_dir: Path):
    for yaml_path in sorted(config_dir.glob("*.yaml")):
        try:
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
        except Exception:
            continue
        if isinstance(cfg, dict) and cfg.get("task", {}).get("name") == task_name:
            return yaml_path, cfg
    return None, None


def resolve_target(results_dir, data=None, config_dir=None) -> Target:
    results_dir = Path(results_dir).resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results dir not found: {results_dir}")

    config_dir = Path(config_dir) if config_dir else REPO_ROOT / "config"
    task_name = _task_name_for(results_dir)
    config_path, cfg = _find_config(task_name, config_dir)

    input_columns = None
    id_column = DEFAULT_ID_COLUMN
    data_path = Path(data).resolve() if data else None

    if cfg is not None:
        data_sec = cfg.get("data", {})
        input_columns = data_sec.get("input_columns")
        id_column = data_sec.get("id_column", DEFAULT_ID_COLUMN)
        if data_path is None and data_sec.get("path"):
            data_path = (REPO_ROOT / data_sec["path"]).resolve()
    else:
        warnings.warn(
            f"no config with task.name == '{task_name}' under {config_dir}; "
            "falling back to two-step defaults"
        )

    if input_columns is None:
        input_columns = list(DEFAULT_INPUT_COLUMNS)
    if data_path is None:
        data_path = (REPO_ROOT / DEFAULT_DATA_PATH).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"data file not found: {data_path}")

    return Target(
        results_dir=results_dir,
        data_path=data_path,
        input_columns=list(input_columns),
        id_column=id_column,
        config_path=config_path,
    )
