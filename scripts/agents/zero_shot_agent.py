from typing import List, Optional, Tuple
from .base_agent import BaseAgent

SFT_SYSTEM_PROMPT = (
    "You are a household robot completing tasks in a text-based environment.\n"
    "At each step you are given the current observation and a list of admissible actions.\n"
    "Reply with ONLY the exact action string from the admissible list. Do not explain."
)
SYSTEM_PROMPT = SFT_SYSTEM_PROMPT


class ZeroShotAgent(BaseAgent):
    """Zero-shot agent: no examples, just instruction + current state."""

    def _get_system_prompt(self) -> str:
        """Return the system prompt. Overridable by subclasses to inject examples."""
        return SYSTEM_PROMPT

    def act(
        self,
        observation: str,
        admissible_commands: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
        def _user_turn(obs: str, commands: Optional[List[str]] = None) -> str:
            content = f"Observation: {obs}"
            if commands:
                cmds = "\n".join(f"  - {c}" for c in commands)
                content += f"\n\nAdmissible actions:\n{cmds}"
            return content

        messages: List[dict] = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        # Prior turns as proper multi-turn conversation (last 5 pairs)
        for action, obs in history[-5:]:
            messages.append({"role": "user",      "content": _user_turn(obs)})
            messages.append({"role": "assistant",  "content": action})
        # Current turn
        messages.append({"role": "user", "content": _user_turn(observation, admissible_commands)})

        response = self.chat(messages, max_tokens=32)
        return self.match_command(response, admissible_commands)
