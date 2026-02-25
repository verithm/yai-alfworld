"""
Hierarchical Agent for ALFWorld
================================
Two-pass architecture:
  Pass 1 (planning): given the task description + initial observation,
                     produce an ordered JSON list of subgoals.
  Pass 2 (execution): for each active subgoal, choose one action using
                      compact state + last 3 (action, obs) turns.

This decouples high-level task decomposition from low-level action
selection, addressing the core failure of 8B models on multi-step
manipulation tasks (0% on pick_clean/heat/cool with flat prompting).
"""

import json
import re
from typing import List, Tuple, Optional

from .base_agent import BaseAgent

# ── Prompts ───────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """\
You are a task planner for a household robot. Given a task goal and an
initial room description, output a JSON array of short, ordered subgoals
that will complete the task.

Rules:
- Each subgoal is a single imperative phrase (≤8 words).
- Include ALL required intermediate steps (e.g. clean before placing).
- Output ONLY valid JSON, nothing else. Example:
  ["find mug 1", "pick up mug 1", "go to sink", "clean mug 1", "go to counter 1", "place mug 1 on counter 1"]
"""

EXECUTOR_SYSTEM = """\
You are a household robot choosing one action per step.

Current subgoal: {subgoal}

Recent history (action → result):
{history}

Current state:
{state}

Admissible actions:
{actions}

Reply with ONLY the exact action string from the admissible list above.
Do not explain. Do not add punctuation.
"""

# ── Observation compactor ──────────────────────────────────────────────────────

_ROOM_PREFIX_RE = re.compile(
    r"You are in .*?(?=\.\s*(?:On|In|You see|$))",
    re.IGNORECASE | re.DOTALL,
)
_HOLDING_RE = re.compile(r"you have ([^\.]+) in your inventory", re.IGNORECASE)
_NOTHING_RE = re.compile(r"you are not holding", re.IGNORECASE)


def _compact_obs(obs: str) -> str:
    """Reduce a verbose ALFWorld observation to a compact state string."""
    obs = obs.strip()
    # Extract held item
    m = _HOLDING_RE.search(obs)
    held = m.group(1).strip() if m else ("nothing" if _NOTHING_RE.search(obs) else "unknown")
    # Strip the long room prefix, keep the remainder (visible objects etc.)
    remainder = _ROOM_PREFIX_RE.sub("", obs).strip().lstrip(". \n")
    # Truncate to first 200 chars to keep context tight
    remainder = remainder[:200] + ("…" if len(remainder) > 200 else "")
    return f"[holding: {held}] {remainder}"


# ── Subgoal completion heuristics ─────────────────────────────────────────────

_COMPLETION_PATTERNS: List[Tuple[re.Pattern, re.Pattern]] = [
    # subgoal keyword          → observation keyword that signals completion
    (re.compile(r"\bpick\b|\btake\b|\bget\b", re.I),  re.compile(r"you pick up|you take", re.I)),
    (re.compile(r"\bplace\b|\bput\b",          re.I),  re.compile(r"you put|you place",    re.I)),
    (re.compile(r"\bclean\b|\bwash\b",         re.I),  re.compile(r"clean|rinsed",          re.I)),
    (re.compile(r"\bheat\b|\bmicrowav\b",      re.I),  re.compile(r"heated|warm",            re.I)),
    (re.compile(r"\bcool\b|\bfridg\b",         re.I),  re.compile(r"cool|cold",              re.I)),
    (re.compile(r"\bexamine\b|\blook at\b",    re.I),  re.compile(r"you examine|you look",   re.I)),
]


def _subgoal_completed(subgoal: str, last_obs: str) -> bool:
    """Return True if any completion heuristic fires for this subgoal."""
    for sg_pat, obs_pat in _COMPLETION_PATTERNS:
        if sg_pat.search(subgoal) and obs_pat.search(last_obs):
            return True
    return False


# ── Agent ─────────────────────────────────────────────────────────────────────

class HierarchicalAgent(BaseAgent):
    """
    Two-pass hierarchical agent.

    Episode flow:
      step 0  → _plan() → populate self._subgoals
      step 1+ → _execute() using current subgoal + compact history
    """

    def __init__(self, model_name: str, ollama_url: str):
        super().__init__(model_name, ollama_url)
        self._subgoals: List[str] = []
        self._subgoal_idx: int = 0
        self._initial_obs: str = ""
        self._last_subgoals: List[str] = []   # saved for trajectory logging

    # ── Public interface ───────────────────────────────────────────────────────

    def reset(self):
        self._subgoals = []
        self._subgoal_idx = 0
        self._initial_obs = ""
        self._last_subgoals = []

    def act(
        self,
        observation: str,
        admissible_commands: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
        # First step: generate subgoal plan
        if not history:
            self._initial_obs = observation
            self._subgoals = self._plan(observation)
            self._last_subgoals = list(self._subgoals)

        # Advance subgoal index based on last observation
        if history:
            last_obs = history[-1][1]
            current = self._current_subgoal()
            if current and _subgoal_completed(current, last_obs):
                self._subgoal_idx = min(self._subgoal_idx + 1, len(self._subgoals) - 1)

        return self._execute(observation, admissible_commands, history)

    # ── Pass 1: planning ───────────────────────────────────────────────────────

    def _plan(self, initial_obs: str) -> List[str]:
        """Call the LLM once to produce an ordered subgoal list."""
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user",   "content": f"Task and initial observation:\n{initial_obs}"},
        ]
        raw = self.chat(messages, max_tokens=400)
        return self._parse_subgoals(raw)

    def _parse_subgoals(self, raw: str) -> List[str]:
        """Extract a JSON list of strings from the planner output."""
        # Try to find a JSON array anywhere in the response
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            try:
                goals = json.loads(m.group(0))
                if isinstance(goals, list) and goals:
                    return [str(g).strip() for g in goals if str(g).strip()]
            except json.JSONDecodeError:
                pass
        # Fallback: treat each non-empty line as a subgoal
        lines = [l.strip(" -•\"'") for l in raw.splitlines() if l.strip()]
        return lines if lines else ["complete the task"]

    # ── Pass 2: execution ──────────────────────────────────────────────────────

    def _execute(
        self,
        observation: str,
        admissible_commands: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
        subgoal = self._current_subgoal()
        compact_state = _compact_obs(observation)

        # Last 3 (action, obs) turns in compact form.
        # HierarchicalAgent uses 3 turns (vs 5 in flat agents) because each entry
        # already uses the compact observation (~220 chars), so the token budget
        # per entry is ~4× smaller. 3 compact entries ≈ 5 full-obs entries in tokens.
        recent = history[-3:] if len(history) >= 3 else history
        history_str = "\n".join(
            f"  > {act}\n  {_compact_obs(obs)}" for act, obs in recent
        ) or "  (start of episode)"

        prompt = EXECUTOR_SYSTEM.format(
            subgoal=subgoal,
            history=history_str,
            state=compact_state,
            actions="\n".join(f"  - {c}" for c in admissible_commands),
        )
        messages = [{"role": "user", "content": prompt}]
        raw = self.chat(messages, max_tokens=128)
        return self.match_command(raw, admissible_commands)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _current_subgoal(self) -> str:
        if not self._subgoals:
            return "complete the task"
        idx = min(self._subgoal_idx, len(self._subgoals) - 1)
        return self._subgoals[idx]
