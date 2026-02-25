"""
Few-shot agent — ALFWorld implementation.

Provides one action-only demonstration per ALFWorld task type (6 total).
Unlike ReAct, there are no explicit Thoughts — only Observation/Action pairs.
The correct example is selected based on the detected task type.

Examples are sourced from examples.py, which also drives ReActAgent.
Both agents solve identical tasks with identical action sequences; the only
difference is that ReActAgent includes explicit Thought lines while this
agent shows pure Observation → Action pairs.
"""

from typing import List, Optional, Tuple

from .base_agent import BaseAgent
from .react_agent import _detect_task_type
from .examples import ACT_EXAMPLES

_SYSTEM_PREFIX = """\
You are an agent solving text-based household tasks.
Study the example below, then choose the best action for the current situation.
Reply with EXACTLY ONE action from the valid actions list — nothing else.

--- EXAMPLE ---
{example}
--- END EXAMPLE ---"""


class FewShotAgent(BaseAgent):
    """
    Few-shot agent: prepends one task-type-specific demonstration per game.
    No explicit reasoning steps (unlike ReAct) — action-only examples.
    The example is drawn from the same canonical trajectories as ReActAgent
    (examples.py), ensuring experimental consistency between the two agents.
    """

    def __init__(self, model_name: str, ollama_url: str):
        super().__init__(model_name, ollama_url)
        self._task_type: Optional[str] = None

    def reset(self):
        self._task_type = None

    def act(
        self,
        observation: str,
        admissible_commands: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
        if not self._task_type:
            self._task_type = _detect_task_type(observation)

        example = ACT_EXAMPLES.get(self._task_type, ACT_EXAMPLES["pick_and_place"])
        system_prompt = _SYSTEM_PREFIX.format(example=example)

        context_lines = []
        for action, obs in history[-5:]:
            context_lines.append(f"Action: {action}\nObservation: {obs}")
        context = "\n\n".join(context_lines)

        commands_str = "\n".join(f"  - {cmd}" for cmd in admissible_commands)

        user_content = ""
        if context:
            user_content += f"Recent history:\n{context}\n\n"
        user_content += (
            f"Current observation:\n{observation}\n\n"
            f"Valid actions:\n{commands_str}\n\n"
            "Choose ONE action from the list above and reply with it exactly."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

        response = self.chat(messages)
        return self.match_command(response, admissible_commands)
