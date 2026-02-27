"""
Few-shot agent — ALFWorld implementation.

Extends ZeroShotAgent by overriding _get_system_prompt() to append a
task-type-specific action-only demonstration.  All act() logic (5-turn
history window, user content format, chat call) is inherited unchanged.

Unlike ReAct, examples show pure Action/Observation pairs with no Thought
lines.  Examples are sourced from ACT_EXAMPLES in examples.py; ReActAgent
uses the same trajectories with Thought lines added (REACT_EXAMPLES),
ensuring experimental consistency between the two agents.

SFT compatibility: the example block is appended to the SFT system prompt,
not injected into user/assistant turns.  The user message format
(Observation: / Admissible actions:) — the PRIMARY SFT conditioning signal —
is unchanged.
"""

from typing import List, Optional, Tuple

from .zero_shot_agent import ZeroShotAgent
from .examples import ACT_EXAMPLES
from .task_utils import _detect_task_type


class FewShotAgent(ZeroShotAgent):
    """
    Few-shot agent: task-type-specific action-only demonstrations.

    Extends ZeroShotAgent:
      - _get_system_prompt() appends a task-specific ACT_EXAMPLES block to
        the SFT system message, giving the model one successful trajectory
        as context before it acts.
      - act() detects task type on the first step; _get_system_prompt() uses
        the cached _task_type to inject the correct example.
      - All other logic (5-turn history, user content, match_command) inherited.
    """

    def __init__(self, model_name: str, ollama_url: str):
        super().__init__(model_name, ollama_url)
        self._task_type: Optional[str] = None

    def reset(self):
        super().reset()
        self._task_type = None

    def _get_system_prompt(self) -> str:
        base = super()._get_system_prompt()  # SFT_SYSTEM_PROMPT
        if self._task_type and self._task_type in ACT_EXAMPLES:
            return (
                base
                + "\n\nHere is an example of a successful trajectory:\n\n"
                + ACT_EXAMPLES[self._task_type]
            )
        return base

    def act(
        self,
        observation: str,
        admissible_commands: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
        # Detect task type on first step so _get_system_prompt() can inject
        # the matching example for all subsequent calls this episode.
        if not self._task_type:
            self._task_type = _detect_task_type(observation)
        return super().act(observation, admissible_commands, history)
