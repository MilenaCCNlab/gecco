import pandas as pd
from typing import Optional
import string



def narrative(
        df: pd.DataFrame,
        template: str,
        id_col: str = "participant",
        trial_col: str = "trial",
        max_trials: int | None = None,
):
    """
    Generate narrative text from behavioral data using a configurable template.

    Args:
        df (pd.DataFrame): The dataset (must include id_col and trial_col).
        template (str): Template string with placeholders like {choice_1}, {reward}.
        id_col (str): Participant identifier column.
        trial_col (str): Trial index column.
        max_trials (int | None): Maximum number of trials to include per participant.
    Returns:
        str: Human-readable narrative text.
    """
    placeholders = [f[1] for f in string.Formatter().parse(template) if f[1] is not None]
    missing = [p for p in placeholders if p not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    text_blocks = []
    for pid, group in df.groupby(id_col):
        if max_trials:
            group = group.head(max_trials)

        participant_text = [f"Here is data from participant {pid}:"]
        for _, row in group.iterrows():
            row_dict = {key: row[key] for key in placeholders}
            trial = int(row[trial_col]) if trial_col in row else _
            trial_text = template.format(**row_dict)
            participant_text.append(f"\n    Trial {trial}:\n    {trial_text}")
        text_blocks.append("\n".join(participant_text))

    return "\n\n".join(text_blocks)


def get_data2text_function(name: str):
    """Return the data→text conversion function specified by name."""
    mapping = {
        "narrative": narrative,  # now points to the dynamic version
    }
    # Default to "narrative" if user passes something unknown
    return mapping.get(name, narrative)
