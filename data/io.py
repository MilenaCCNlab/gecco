import pandas as pd
from typing import List

def load_data(path: str, input_columns: List[str]) -> pd.DataFrame:
    """Load data and extract only relevant columns."""
    df = pd.read_csv(path)
    available = [col for col in input_columns if col in df.columns]
    return df[available]
