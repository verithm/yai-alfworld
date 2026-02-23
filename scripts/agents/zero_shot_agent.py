from typing import List, Tuple
from .base_agent import BaseAgent

SYSTEM_PROMPT = (
    "You are an agent playing a text-based household task game.\n"
    "Your goal is to complete the assigned task by issuing actions.\n"
    "You will be given your current observation and a list of valid actions.\n"
    "Reply with EXACTLY ONE action from the list — nothing else."
)


class ZeroShotAgent(BaseAgent):
    """Zero-shot agent: no examples, just instruction + current state."""

    def act(
        self,
        observation: str,
        admissible_commands: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
        # Include up to 5 recent (action, observation) pairs for context
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = self.chat(messages)
        return self.match_command(response, admissible_commands)
