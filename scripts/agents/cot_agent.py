from typing import List, Tuple
from .base_agent import BaseAgent

SYSTEM_PROMPT = """\
You are an agent playing a text-based household task game.
At each step, reason step-by-step about what to do, then state your action.

You MUST output exactly two blocks:
  Reasoning: <2-3 sentences of step-by-step thinking about the task and next move>
  Action: <one action chosen exactly from the valid actions list>

The Action line must match one of the valid actions exactly.
Do NOT output anything else after the two blocks.
"""


class CotAgent(BaseAgent):
    """
    Chain-of-Thought agent.

    Generates an explicit reasoning trace before each action.  Unlike ReAct,
    the reasoning is stateless (not accumulated in a scratchpad) — each step
    reasons from scratch using the recent history window.
    """

    def act(
        self,
        observation: str,
        admissible_commands: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
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
            "Now output your Reasoning and Action:"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        raw = self.chat(messages, max_tokens=200)

        # Parse Action: line; fall back to full response if missing
        action_str = ""
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("action:"):
                action_str = stripped[len("action:"):].strip()
                break

        return self.match_command(action_str or raw, admissible_commands)
