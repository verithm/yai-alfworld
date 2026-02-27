"""
Shared ALFWorld task-type detection utilities.

Extracted here to break the circular import that would otherwise arise when
ReActAgent inherits FewShotAgent while FewShotAgent imports _detect_task_type
from react_agent.  Both agents now import from this module instead.
"""

import re

def _detect_task_type(observation: str) -> str:
    """
    Infer the ALFWorld task type from the initial observation.
    The initial obs always contains 'Your task is to: <description>'.
    """
    task_line = observation.lower()
    if "your task is to:" in task_line:
        task_line = task_line.split("your task is to:")[1].strip()[:300]

    if "examine" in task_line or "look at" in task_line:
        return "look_at_obj"
    if re.search(r"\btwo\b", task_line):
        return "pick_two_obj"
    if "clean" in task_line:
        return "pick_clean_then_place"
    if re.search(r"\bheat\b|\bhot\b|\bwarm\b", task_line):
        return "pick_heat_then_place"
    if re.search(r"\bcool\b|\bcold\b|\bchill", task_line):
        return "pick_cool_then_place"
    return "pick_and_place"
