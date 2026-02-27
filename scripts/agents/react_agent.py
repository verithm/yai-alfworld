"""
ReAct Agent (Yao et al., 2022) — ALFWorld implementation.

Key design decisions matching the paper:
  - Thoughts follow the paper's canonical language:
      Step 1: goal decomposition + location reasoning
              "A X is more likely to appear in ... I can check one by one, starting with ..."
      Step 2+: subgoal tracking
              "Now I find/take/clean/heat/cool X (N). Next, I need to ..."
  - Scratchpad keeps last 12 steps.  Critically, the FULL raw model response
    (Thought+Action text) is stored per step and used as the assistant content
    in multi-turn messages — so prior reasoning is always visible in context.
  - _get_system_prompt() overrides FewShotAgent to use REACT_SYSTEM_PROMPT,
    which explicitly requests Thought+Action format.
  - ReflexionAgent subclasses this to prepend its verbal reflection memory
    without duplicating the ReAct logic.

Inherits from FewShotAgent:
  - _task_type state and detection (act() sets it on first step)
  - reset() chain clears both _task_type and _react_history

SFT compatibility:
  - User message format (Observation: / Admissible actions:) is unchanged —
    the PRIMARY SFT conditioning signal is preserved.
  - System prompt and assistant history content are SECONDARY; the Thought+Action
    format in assistant turns acts as in-context demonstrations.
  - match_command() handles bare-action fallback if the model omits Thought.
"""

from typing import List, Optional, Tuple

from .few_shot_agent import FewShotAgent
from .task_utils import _detect_task_type


# ── System prompt ──────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = (
    "You are a household robot completing tasks in a text-based environment.\n"
    "At each step you are given the current observation and a list of admissible actions.\n"
    "Think step by step. Write 'Thought: <your reasoning>' then 'Action: <exact action from the list>'."
)

# ── Agent ──────────────────────────────────────────────────────────────────────

SCRATCHPAD_WINDOW = 12   # keep last N Thought/Action/Observation turns


class ReActAgent(FewShotAgent):
    """
    ReAct agent (Yao et al., 2022).

    Generates a Thought (reasoning) and an Action at every step.
    Prior reasoning is kept in context by storing the full raw model response
    (Thought+Action text) as assistant content in the multi-turn scratchpad,
    so the model can track which locations were searched and what remains.

    Extends FewShotAgent:
      - _task_type and detection logic inherited; no duplication.
      - _get_system_prompt() overrides FewShot's to use REACT_SYSTEM_PROMPT,
        which requests Thought+Action format.
      - act() builds multi-turn messages from _react_history (full responses)
        rather than the bare-action external history.

    _get_system_prompt() is intentionally overridable: ReflexionAgent
    subclasses this to prepend its verbal reflection memory.
    """

    def __init__(self, model_name: str, ollama_url: str):
        super().__init__(model_name, ollama_url)   # sets self._task_type = None
        self._react_history: List[Tuple[str, str]] = []  # (raw_response, observation)

    def reset(self):
        super().reset()          # clears _task_type via FewShot → ZeroShot → BaseAgent
        self._react_history = []

    def _get_system_prompt(self) -> str:
        return REACT_SYSTEM_PROMPT

    def act(
        self,
        observation: str,
        admissible_commands: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
        if not self._task_type:
            self._task_type = _detect_task_type(observation)

        def _user_turn(obs: str, commands: Optional[List[str]] = None) -> str:
            content = f"Observation: {obs}"
            if commands:
                cmds = "\n".join(f"  - {c}" for c in commands)
                content += f"\n\nAdmissible actions:\n{cmds}"
            return content

        # Build multi-turn from _react_history so the model sees its prior
        # Thought+Action text as assistant content — not bare actions.
        messages: List[dict] = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        for raw_response, obs in self._react_history[-SCRATCHPAD_WINDOW:]:
            messages.append({"role": "user",      "content": _user_turn(obs)})
            messages.append({"role": "assistant",  "content": raw_response})
        messages.append({"role": "user", "content": _user_turn(observation, admissible_commands)})

        raw = self.chat(messages, max_tokens=150)

        # Parse Thought/Action; fall back to bare action if model omits Thought
        thought, action_str = self._parse_react(raw)
        action = self.match_command(action_str or raw, admissible_commands)

        # Store full response + observation for context in future steps
        self._react_history.append((raw, observation))
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
