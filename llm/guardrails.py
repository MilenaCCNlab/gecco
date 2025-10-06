import re

REQUIRED_SNIPPETS = [
    r"def\s+\w+\s*\(",                 # at least one def
]

def check_basic_guardrails(text: str) -> None:
    """
    Raise ValueError if guardrails fail.
    Customize with stricter rules as needed (e.g., name 'cognitive_model').
    """
    code_block = _first_code_block(text=text)
    for pat in REQUIRED_SNIPPETS:
        if not re.search(pat, code_block):
            raise ValueError("Generated output does not contain a valid function.")
    # optional: forbid imports, file I/O, etc.
    if re.search(r"\bimport\s+(os|sys|subprocess|pathlib)\b", code_block):
        raise ValueError("Disallowed imports detected in generated code.")

def _first_code_block(text: str) -> str:
    m = re.search(r"```python(.*?)```", text, flags=re.S)
    if not m:
        m = re.search(r"```(.*?)```", text, flags=re.S)
    return (m.group(1) if m else text)
