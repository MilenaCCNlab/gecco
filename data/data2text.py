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


def get_data2text_function(name):
    if name == "narrative":
        def data2text(df, id_col, template, max_trials=None, value_mappings=None):
            """
            Convert participant trial data into a narrative string.
            Generic across tasks — assumes only participant ID and trial rows.

            Args:
                df: pandas.DataFrame (rows = trials)
                id_col: column name identifying participants
                template: str, user-specified text format with placeholders (e.g., {choice}, {reward})
                max_trials: optional int, limit number of trials per participant
                value_mappings: optional dict[col -> mapping_dict], applied only to existing cols
            """
            narratives = []

            for pid in df[id_col].unique():
                sub = df[df[id_col] == pid].head(max_trials or len(df))
                trial_lines = []

                for _, row in sub.iterrows():
                    vals = dict(row)

                    # Apply user-defined mappings if provided
                    if value_mappings:
                        # Convert from SimpleNamespace (if needed)
                        if not isinstance(value_mappings, dict):
                            value_mappings = vars(value_mappings)

                        for col, mapping in value_mappings.items():
                            if not isinstance(mapping, dict):
                                mapping = vars(mapping)
                            if col in vals:
                                key = str(int(vals[col])) if isinstance(vals[col], (int, float)) else str(vals[col])
                                vals[col] = mapping.get(key, vals[col])

                    try:
                        line = template.format(**vals)
                        trial_lines.append(line)
                    except KeyError as e:
                        print(f"[⚠️ GeCCo] Missing column {e} in template for participant {pid}")
                        continue

                # Join participant's trials
                participant_text = "\n".join(trial_lines)
                narratives.append(f"Participant {pid}:\n{participant_text}\n")

            return "\n".join(narratives)
        return data2text
