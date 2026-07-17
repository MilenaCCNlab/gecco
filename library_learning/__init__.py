"""Library learning for GECCO per-participant cognitive models.

Compresses the shared mechanisms of individually generated cognitive models
into a per-experiment ``cognitive_library.py`` and rewrites each participant
as ``participant_<id>.py`` (library imports + idiosyncratic local code),
with behavioral-equivalence verification against the originals.

Usage:
    python -m library_learning scan   --results-dir results/<task>_individual
    python -m library_learning verify --results-dir results/<task>_individual [--baseline]
    python -m library_learning report --results-dir results/<task>_individual
"""

__version__ = "0.1.0"
