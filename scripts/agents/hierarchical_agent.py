"""
Hierarchical Agent for ALFWorld
================================
Two-pass architecture with Reflexion self-correction.

Extends ReflexionAgent (Reflexion + Planning):
  - Inherits: multi-trial loop, reflection storage, add_reflection(), reset()
  - Adds:     Pass 1 (LLM planner → ordered JSON subgoal list per trial)
              Pass 2 (executor picks one action per step guided by current subgoal)
              Structured JSON reflection (overrides ReflexionAgent.reflect()) that
              critiques the PLAN rather than step-level execution.

On failed trials, a verbal reflection critiques the PLAN (wrong subgoal
sequence, missing steps, wrong location prediction).  All reflections are
prepended to the Planner prompt for subsequent trials so the model re-plans
with explicit memory of its own mistakes.

This decouples high-level task decomposition from low-level action
selection, addressing the core failure of 8B models on multi-step
manipulation tasks.

Grounded in:
  - Shinn et al. (2023) Reflexion: Language Agents with Verbal Reinforcement
  - AdaPlanner "Hierarchical + Feedback" insight (Sun et al., NeurIPS 2023)

Trial lifecycle (managed by evaluate_reflexion_agent in run_baseline.py):
    agent.reset()                # between games: clears reflections + plan
    for trial in range(max_trials):
        agent.reset_trial()      # clears per-trial plan; keeps reflections
        result = run_episode(env, agent, max_steps, reflexion_heuristic=True)
        if result["success"]: break
        reflection = agent.reflect(result["initial_obs"], result["trajectory"])
        agent.add_reflection(reflection)
"""

import json
import re
from typing import List, Tuple, Optional

from .reflexion_agent import ReflexionAgent

# ── Prompts ───────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """\
You are a task planner for a household robot in a text-based game.
Given a task goal and an initial room description, output a JSON array
of short, ordered subgoals that will complete the task.

GROUNDING RULES (apply BEFORE generating subgoals):
1. Read "Your task is to: …" to identify the TARGET OBJECT (e.g. "cloth", "mug", "apple").
   The target object is ALWAYS the one named in the task description — never a different
   visible object. Then find the target object in the initial observation and use its
   exact in-game name (e.g. "cloth 1", "mug 2") in every subgoal.
2. If the target object is already visible in the initial observation, skip searching for it.
3. If an object might be inside a container (fridge, cabinet, drawer, box), add
   "open <container>" BEFORE the "take <object> from <container>" subgoal.
4. For examine-in-light tasks: go to the desklamp first, turn it on, then examine the object.

SUBGOAL VERBS — use ONLY these exact verb forms (the game rejects all others):
  go to <location>              ← navigation
  take <object> from <location> ← pick up (NEVER write "pick up", "grab", "get")
  put <object> in/on <location> ← place   (NEVER write "place", "insert", "deposit")
  open <container>
  close <container>
  clean <object> with sinkbasin <N>   ← cleaning (NEVER "rinse", "wash", "use water")
  heat <object> with microwave <N>    ← heating  (NEVER "use microwave", "put in microwave")
  cool <object> with fridge <N>       ← cooling  (NEVER "use fridge", "put in fridge")
  use <object>
  examine <object>

CRITICAL — ALFWorld appliance rules (do NOT deviate):
- To HEAT an object  → subgoal: "heat <object> with microwave <N>"  (never stove or burner)
- To COOL an object  → subgoal: "cool <object> with fridge <N>"     (never freezer)
- To CLEAN an object → subgoal: "clean <object> with sinkbasin <N>" (never rinse/wash)
- To LIGHT/EXAMINE   → subgoal: "use desklamp <N>", then "examine <object>"

Rules:
- Each subgoal is a single imperative phrase (≤8 words).
- Navigation subgoals use the form: "go to <location name>".
- NO conditional subgoals (no "if", "or", "else", "unless").
- Include ALL required intermediate steps (e.g. open container before taking, clean before placing).
- If the task goal mentions "light", "lamp", or "examine … under", add
  "use desklamp N" as a subgoal BEFORE the examine subgoal.
- Output ONLY valid JSON, nothing else. Example:
  ["go to countertop 1", "take mug 1 from countertop 1", "go to microwave 1", "heat mug 1 with microwave 1", "go to countertop 2", "put mug 1 in/on countertop 2"]
"""

# Split into static system instructions and dynamic per-step user context.
# The system prompt goes in role="system"; dynamic context in role="user".
EXECUTOR_SYSTEM_PROMPT = (
    "You are a household robot completing tasks in a text-based environment.\n"
    "At each step you are given the current observation and a list of admissible actions.\n"
    "Reply with ONLY the exact action string from the admissible list. Do not explain."
)

HIER_REFLECT_SYSTEM = """\
You are reviewing a failed text-based household task to improve the next plan.

Write your reflection as JSON with these exact fields:
{
  "task_goal": "brief description of what was required",
  "failed_subgoal": "the subgoal that went wrong or where the agent got stuck",
  "failure_reason": "one of: wrong_location | missing_step | wrong_object | stuck_in_loop | premature_completion",
  "missing_steps": ["exact ALFWorld action strings that were absent from the plan"],
  "revised_strategy": "specific instruction for the next plan"
}

CRITICAL — missing_steps must be valid ALFWorld action strings using ONLY these verbs:
  go to <X>  |  take <X> from <Y>  |  put <X> in/on <Y>  |  open <X>  |  close <X>
  clean <X> with sinkbasin <N>  |  heat <X> with microwave <N>  |  cool <X> with fridge <N>
  use <X>  |  examine <X>

NEVER write "pick up", "place", "insert", "grab", "move", or natural-language phrases.
Each entry in missing_steps must be a single action string the robot can execute directly.

Be concrete. Generic statements like "I failed" or "try harder" are useless.
Example:
{
  "task_goal": "find and heat a mug, put it on countertop",
  "failed_subgoal": "take mug 1 from countertop 1",
  "failure_reason": "wrong_location",
  "missing_steps": ["go to cabinet 1", "open cabinet 1", "take mug 1 from cabinet 1"],
  "revised_strategy": "Mug was in cabinet 1, not on countertop. Go to cabinet 1, open it, then take the mug."
}
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
    (re.compile(r"\bpick\b|\btake\b|\bget\b",          re.I), re.compile(r"you pick up|you take",                  re.I)),
    (re.compile(r"\bplace\b|\bput\b",                  re.I), re.compile(r"you put|you place",                    re.I)),
    (re.compile(r"\bclean\b|\bwash\b",                 re.I), re.compile(r"clean|rinsed",                         re.I)),
    (re.compile(r"\bheat\b|\bmicrowav\b",              re.I), re.compile(r"heated|warm",                          re.I)),
    (re.compile(r"\bcool\b|\bfridg\b",                 re.I), re.compile(r"cool|cold",                            re.I)),
    # Fix: use desklamp/lamp had no completion pattern → subgoal never advanced after successful use
    (re.compile(r"\buse\b",                            re.I), re.compile(r"you turn on",                          re.I)),
    # Fix: require non-empty content — "you see nothing" must NOT advance the subgoal
    (re.compile(r"\bexamine\b|\blook at\b|\blook\b",   re.I), re.compile(r"you examine .+\S|you look .+\S|on the .+you see(?! nothing)|there(?:'s| is) .+\S", re.I)),
    # Fix: navigation subgoals previously had no completion pattern → subgoal index stuck at 0
    (re.compile(r"\bgo to\b|\bnavigate\b|\bmove to\b|\bfind\b", re.I), re.compile(r"you arrive at|you go to|on the \w.{0,60}you see", re.I)),
]


def _subgoal_completed(subgoal: str, last_obs: str) -> bool:
    """Return True if any completion heuristic fires for this subgoal."""
    # Never mark examine/look complete when the observation reports "nothing"
    if re.search(r"\bexamine\b|\blook at\b|\blook\b", subgoal, re.I):
        if re.search(r"\bnothing\b", last_obs, re.I):
            return False
    for sg_pat, obs_pat in _COMPLETION_PATTERNS:
        if sg_pat.search(subgoal) and obs_pat.search(last_obs):
            return True
    return False


# ── Agent ─────────────────────────────────────────────────────────────────────

class HierarchicalAgent(ReflexionAgent):
    """
    Hierarchical agent — Reflexion + Planning.

    Inherits ReflexionAgent's multi-trial reflection loop and adds a planning layer:

      Pass 1 (planning)   — LLM produces an ordered JSON subgoal list once per trial.
      Pass 2 (execution)  — executor picks one action per step guided by current subgoal.

    Reflection targets the planner (not the executor): after each failed trial,
    a structured JSON reflection critiques the plan (wrong subgoal sequence,
    missing steps, wrong location).  All reflections are injected into the
    Planner prompt on the next trial so the model re-plans with explicit memory
    of its own mistakes (up to max_trials iterations).

    Episode flow:
      step 0  → _plan() → populate self._subgoals
      step 1+ → _execute() using current subgoal + compact history
    """

    def __init__(self, model_name: str, ollama_url: str, max_trials: int = 3):
        super().__init__(model_name, ollama_url, max_trials)
        # Hierarchical-specific state; reflections + max_trials come from ReflexionAgent
        self._subgoals: List[str] = []
        self._subgoal_idx: int = 0
        self._subgoal_steps: int = 0       # consecutive steps spent on current subgoal
        self._initial_obs: str = ""
        self._last_subgoals: List[str] = []   # saved for trajectory logging

    # ── State management ────────────────────────────────────────────────────────

    def reset(self):
        """Between games: clear reflections (via super) and all plan state."""
        super().reset()   # clears self.reflections, _react_history, _task_type
        self._subgoals = []
        self._subgoal_idx = 0
        self._subgoal_steps = 0
        self._initial_obs = ""
        self._last_subgoals = []

    def reset_trial(self):
        """Between trials of the same game: clear per-trial plan, keep reflections."""
        super().reset_trial()   # clears _react_history; self.reflections preserved
        self._subgoals = []
        self._subgoal_idx = 0
        self._subgoal_steps = 0
        self._initial_obs = ""
        self._last_subgoals = []

    # ── Public interface ────────────────────────────────────────────────────────

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
            self._subgoal_steps = 0

        # Advance subgoal based on completion heuristic or loop-break triggers
        if history:
            last_obs = history[-1][1]
            current = self._current_subgoal()
            prev_idx = self._subgoal_idx

            if current and _subgoal_completed(current, last_obs):
                self._subgoal_idx = min(self._subgoal_idx + 1, len(self._subgoals) - 1)
                if self._subgoal_idx != prev_idx:
                    self._subgoal_steps = 0
            else:
                self._subgoal_steps += 1
                # Force-advance after 8 steps stuck on the same subgoal
                if self._subgoal_steps >= 8:
                    self._subgoal_idx = min(self._subgoal_idx + 1, len(self._subgoals) - 1)
                    self._subgoal_steps = 0

            # ABAB oscillation detection: two actions alternating in last 4 steps
            if len(history) >= 4:
                acts4 = [h[0] for h in history[-4:]]
                if (acts4[0] == acts4[2] and acts4[1] == acts4[3]
                        and acts4[0] != acts4[1]):
                    self._subgoal_idx = min(self._subgoal_idx + 1, len(self._subgoals) - 1)
                    self._subgoal_steps = 0

        return self._execute(observation, admissible_commands, history, self._initial_obs)

    # ── Pass 1: planning ────────────────────────────────────────────────────────

    def _plan(self, initial_obs: str) -> List[str]:
        """
        Generate subgoal list.  On retry trials, past reflections are prepended
        to the Planner system prompt so the model produces a revised plan.
        """
        system = PLANNER_SYSTEM
        if self.reflections:
            reflection_block = "\n".join(
                f"[Attempt {i}]: {r}" for i, r in enumerate(self.reflections, 1)
            )
            system = (
                "Lessons from your previous failed attempts on this task"
                " (treat as hard constraints — do NOT repeat these mistakes):\n"
                + reflection_block
                + "\n\nEach reflection is JSON. Read the 'missing_steps' array and add"
                  " every listed step to your plan.\n"
                + "Read 'revised_strategy' and follow it exactly when generating subgoals.\n\n"
                + PLANNER_SYSTEM
            )

        messages = [
            {"role": "system", "content": system},
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

    # ── Pass 2: execution ───────────────────────────────────────────────────────

    def _execute(
        self,
        observation: str,
        admissible_commands: List[str],
        history: List[Tuple[str, str]],
        initial_obs: str = "",
    ) -> str:
        subgoal = self._current_subgoal()
        cmds_str = "\n".join(f"  - {c}" for c in admissible_commands)

        # Build current user turn: subgoal context + raw observation + admissible actions
        current_user = (
            f"Current subgoal: {subgoal}\n\n"
            f"Observation: {observation}\n\n"
            f"Admissible actions:\n{cmds_str}"
        )

        # Build multi-turn message list (last 3 history pairs)
        messages: List[dict] = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT}
        ]
        for action, obs in history[-3:]:
            messages.append({"role": "user",      "content": f"Observation: {obs}"})
            messages.append({"role": "assistant",  "content": action})
        messages.append({"role": "user", "content": current_user})

        raw = self.chat(messages, max_tokens=32)
        return self.match_command(raw, admissible_commands)

    # ── Reflection generation ────────────────────────────────────────────────────

    def reflect(self, initial_obs: str, trajectory: List[Tuple[str, str]]) -> str:
        """
        Generate a structured JSON reflection after a failed episode.
        Overrides ReflexionAgent.reflect() to critique the PLAN rather than
        step-level execution, and to emit structured JSON consumed by _plan().

        Args:
            initial_obs:  First observation of the episode (contains goal).
            trajectory:   List of (action, resulting_observation) pairs.
        """
        traj_lines = [
            f"  {i + 1}. {act}: {obs[:120]}"
            for i, (act, obs) in enumerate(trajectory[-15:])
        ]
        traj_str = "\n".join(traj_lines)

        subgoals_str = (
            "\n".join(f"  {i + 1}. {sg}" for i, sg in enumerate(self._last_subgoals))
            or "  (none generated)"
        )

        user_content = (
            f"Task context (initial observation):\n{initial_obs[:400]}\n\n"
            f"Planned subgoals:\n{subgoals_str}\n\n"
            f"Execution trajectory (last {len(traj_lines)} steps):\n{traj_str}\n\n"
            "Write your reflection:"
        )

        messages = [
            {"role": "system", "content": HIER_REFLECT_SYSTEM},
            {"role": "user",   "content": user_content},
        ]
        return self.chat(messages, max_tokens=200)

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _current_subgoal(self) -> str:
        if not self._subgoals:
            return "complete the task"
        idx = min(self._subgoal_idx, len(self._subgoals) - 1)
        return self._subgoals[idx]


# Backward-compatibility alias
HierarchicalReflexionAgent = HierarchicalAgent
