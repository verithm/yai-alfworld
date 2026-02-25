"""
ReAct Agent (Yao et al., 2022) — ALFWorld implementation.

Key design decisions matching the paper:
  - Task-type-specific few-shot example (one per 6 ALFWorld task types), sourced
    from examples.py which also drives FewShotAgent — guaranteeing that both
    agents use identical task/action/observation content.
  - Thoughts follow the paper's canonical language:
      Step 1: goal decomposition + location reasoning
              "A X is more likely to appear in ... I can check one by one, starting with ..."
      Step 2+: subgoal tracking
              "Now I find/take/clean/heat/cool X (N). Next, I need to ..."
  - Scratchpad keeps last 12 Thought/Action/Observation turns (paper uses full
    context; 12 is a practical compromise for 8B context limits).
  - _get_system_prompt() is overridable so ReflexionAgent can inject its
    reflection memory without duplicating the ReAct logic.
"""

import re
from typing import List, Optional, Tuple

from .base_agent import BaseAgent
from .examples import REACT_EXAMPLES

# ── Task-type detection ────────────────────────────────────────────────────────

_TASK_TYPES = (
    "look_at_obj",
    "pick_two_obj",
    "pick_clean_then_place",
    "pick_heat_then_place",
    "pick_cool_then_place",
    "pick_and_place",          # default / catch-all
)


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


# ── System prompt ──────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """\
You are an agent solving a text-based household task using the ReAct framework.

At each step output EXACTLY two lines:
  Thought: <reasoning>
  Action: <one action from the valid actions list, copied exactly>

Thought guidelines — your thought should do one or more of:
  1. Decompose the goal and reason about locations (first step):
     "To solve the task, I need to find X, then do Y, then place it in Z.
      First I need to find X. A X is more likely to appear in cabinet (1-6),
      countertop (1-3), shelf (1-3). I can check one by one, starting with cabinet 1."
  2. Track subgoal completion:
     "Now I find a mug (1). Next, I need to take it."
     "Now I take a mug (1). Next, I need to go to sinkbasin (1) and clean it."
     "Now I clean a mug (1). Next, I need to put it in/on countertop 1."

Rules:
  - The Action line must match one of the valid actions EXACTLY.
  - Do NOT output anything beyond the two lines.
  - If you cannot see the target object, reason about where it is likely to be
    and go search those locations systematically.
"""

# ── Agent ──────────────────────────────────────────────────────────────────────

SCRATCHPAD_WINDOW = 12   # keep last N Thought/Action/Observation turns


class ReActAgent(BaseAgent):
    """
    ReAct agent (Yao et al., 2022).

    Generates a Thought (reasoning) and an Action at every step.
    The full Thought+Action+Observation scratchpad is maintained so the model
    can track which locations have been searched and what subgoals remain.

    _get_system_prompt() is intentionally overridable: ReflexionAgent
    subclasses this to prepend its verbal reflection memory.
    """

    def __init__(self, model_name: str, ollama_url: str):
        super().__init__(model_name, ollama_url)
        self._react_history: List[str] = []   # "Thought:…\nAction:…\nObs:…"
        self._task_type: Optional[str] = None

    def reset(self):
        self._react_history = []
        self._task_type = None

    def _get_system_prompt(self) -> str:
        """Build the system prompt: instructions + task-type example."""
        example = REACT_EXAMPLES.get(
            self._task_type or "pick_and_place",
            REACT_EXAMPLES["pick_and_place"],
        )
        return (
            REACT_SYSTEM_PROMPT
            + "\nHere is an example trajectory for this type of task:\n\n"
            + example
            + "\n\nNow solve the task below using the same reasoning style."
        )

    def act(
        self,
        observation: str,
        admissible_commands: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
        # Detect task type once from the initial observation
        if not self._task_type:
            self._task_type = _detect_task_type(observation)

        commands_str = "\n".join(f"  - {cmd}" for cmd in admissible_commands)
        scratchpad = "\n".join(self._react_history[-SCRATCHPAD_WINDOW:])

        user_content = (
            (f"Scratchpad (recent steps):\n{scratchpad}\n\n" if scratchpad else "")
            + f"Current observation: {observation}\n\n"
            f"Valid actions:\n{commands_str}\n\n"
            "Output your Thought and Action:"
        )

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user",   "content": user_content},
        ]

        raw = self.chat(messages, max_tokens=200)

        thought, action_str = self._parse_react(raw)

        # One retry if Action line is missing
        if not action_str:
            retry_msgs = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    "Your response was missing the Action line. "
                    "Reply with EXACTLY these two lines:\n"
                    "Thought: <one sentence>\n"
                    f"Action: <choose from: {', '.join(admissible_commands[:5])}…>"
                )},
            ]
            raw = self.chat(retry_msgs, max_tokens=100)
            thought, action_str = self._parse_react(raw)

        action = self.match_command(action_str or raw, admissible_commands)

        # Append to scratchpad for next turn
        self._react_history.append(
            f"Thought: {thought}\nAction: {action}\nObservation: {observation}"
        )

        return action

    @staticmethod
    def _parse_react(raw: str) -> Tuple[str, str]:
        thought, action_str = "", ""
        for line in raw.splitlines():
            s = line.strip()
            if s.lower().startswith("thought:"):
                thought = s[len("thought:"):].strip()
            elif s.lower().startswith("action:"):
                action_str = s[len("action:"):].strip()
        return thought, action_str
