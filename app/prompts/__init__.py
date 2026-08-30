"""Prompt library.

Every prompt follows the same shape::

    REVIEWER2_TASK: <task_name>
    <instructions>
    OUTPUT: <json schema>
    INPUT:
    { ...json payload... }

The machine-readable task marker lets the offline heuristic provider answer
the same contracts as a real model, and keeps prompt logs greppable.
"""

from .templates import (
    SYSTEM_PROMPT,
    claim_extraction_prompt,
    critique_prompt,
    devil_advocate_prompt,
    internal_contradiction_prompt,
    omission_prompt,
)

__all__ = [
    "SYSTEM_PROMPT",
    "claim_extraction_prompt",
    "critique_prompt",
    "omission_prompt",
    "internal_contradiction_prompt",
    "devil_advocate_prompt",
]
