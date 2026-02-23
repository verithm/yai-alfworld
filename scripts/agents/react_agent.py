from typing import List, Tuple
from .base_agent import BaseAgent

SYSTEM_PROMPT = """\
You are an agent playing a text-based household task game using the ReAct framework.
At each step you MUST output exactly two lines:
  Thought: <one sentence of reasoning about what to do next>
  Action: <one action chosen from the valid actions list>

Rules:
- The Action line must match one of the valid actions exactly.
- Do NOT output anything else — no extra lines, no explanations after the two lines.
"""

REACT_EXAMPLE = """\
Example of correct ReAct output:
---
Thought: I need to find a clean cup and place it on the counter. Let me first look around.
Action: look
---
Thought: I can see a cup on the shelf. I should go there and pick it up.
Action: go to shelf 1
---
"""


class ReActAgent(BaseAgent):
    """
    ReAct agent (Yao et al., 2022).

    At each step the model generates a Thought (free-form reasoning) and then
    an Action (one of the admissible commands).  The full Thought+Action history
    is maintained in the prompt so the model can reason about its own trajectory.
    """

    def __init__(self, model_name: str, ollama_url: str):
        super().__init__(model_name, ollama_url)
        self._react_history: List[str] = []   # "Thought: … / Action: … / Obs: …"

    def reset(self):
        self._react_history = []

    def act(
        self,
        observation: str,
        admissible_commands: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
        commands_str = "\n".join(f"  - {cmd}" for cmd in admissible_commands)

        # Build the running ReAct scratchpad (last 6 turns to control length)
        scratchpad = "\n".join(self._react_history[-6:])

        user_content = (
            REACT_EXAMPLE
            + (f"\nScratchpad so far:\n{scratchpad}\n" if scratchpad else "")
            + f"\nObservation: {observation}\n\n"
            f"Valid actions:\n{commands_str}\n\n"
            "Now output your Thought and Action:"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw = self.chat(messages, max_tokens=200)

        # Parse Thought and Action lines
        thought, action_str = "", ""
        for line in raw.splitlines():
            line = line.strip()
            if line.lower().startswith("thought:"):
                thought = line[len("thought:"):].strip()
            elif line.lower().startswith("action:"):
                action_str = line[len("action:"):].strip()

        # Format enforcement: if no Action line found, retry once with correction
        if not action_str:
            correction = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    "Your response was missing the required format. "
                    "Reply with EXACTLY these two lines and nothing else:\n"
                    "Thought: <one sentence>\n"
                    f"Action: <one action from: {', '.join(admissible_commands[:5])}...>"
                )},
            ]
            raw = self.chat(correction, max_tokens=100)
            for line in raw.splitlines():
                line = line.strip()
                if line.lower().startswith("thought:"):
                    thought = line[len("thought:"):].strip()
                elif line.lower().startswith("action:"):
                    action_str = line[len("action:"):].strip()

        action = self.match_command(action_str or raw, admissible_commands)

        # Append to ReAct scratchpad for next turn
        self._react_history.append(
            f"Thought: {thought}\nAction: {action}\nObservation: {observation}"
        )

        return action
