import pandas as pd
from typing import Optional

def narrative_from_dataframe(
    df: pd.DataFrame,
    id_column: str = "participant_id",
    template: Optional[str] = None
) -> str:
    """Convert trial-level data into a natural language narrative per participant."""
    if template is None:
        template = (
            "The participant chose spaceship {spaceship} and ended up on the {planet} Planet.\n"
            "The participant asked the alien {alien} and received {reward} coins."
        )

    blocks = []
    for pid, df_sub in df.groupby(id_column):
        text = [f"Here is data from participant {pid}:\n"]
        for t, row in df_sub.iterrows():
            text.append(f"    Trial {t}:\n    " + template.format(**row.to_dict()))
            text.append("")  # spacing
        blocks.append("\n".join(text))
    return "\n\n".join(blocks)


def get_data2text_function(name: str):
    mapping = {"narrative": narrative_from_dataframe}
    return mapping.get(name, narrative_from_dataframe)
