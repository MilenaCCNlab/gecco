import sys
from pathlib import Path
sys.path.append('/home/aj9225/gecco-1')
from gecco.prepare_data.io import load_data
from config.schema import load_config

project_root = Path('/home/aj9225/gecco-1')
cfg = load_config(project_root / "config" / "two_step_psychiatry.yaml")
data_df = load_data(cfg.data.path)
print(data_df.columns)
