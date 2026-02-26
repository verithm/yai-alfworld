"""
Hierarchical Agent for ALFWorld
================================
Two-pass architecture with Reflexion self-correction:

  Pass 1 (planning): given the task description + initial observation,
                     produce an ordered JSON list of subgoals.
  Pass 2 (execution): for each active subgoal, choose one action using
                      compact state + last 3 (action, obs) turns.

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

from .base_agent import BaseAgent

# ── Prompts ───────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """\
You are a task planner for a household robot in a text-based game.
Given a task goal and an initial room description, output a JSON array
of short, ordered subgoals that will complete the task.

GROUNDING RULES (apply before generating subgoals):
1. Read the initial observation and identify the exact names of visible objects.
   Use those exact names (e.g. "mug 1", "drawer 2") in subgoals — never generic names.
2. If the target object is already visible in the initial observation, skip searching for it.
3. If an object might be inside a container (fridge, cabinet, drawer, box), add
   "open <container>" BEFORE the "pick up <object>" subgoal.
4. For examine-in-light tasks: go to the desklamp first, turn it on, then examine the object.

CRITICAL — ALFWorld appliance rules (do NOT deviate):
- To HEAT an object  → use "microwave" (never stove, oven, or burner)
- To COOL an object  → use "fridge"    (never freezer or coffeemachine)
- To CLEAN an object → use "sinkbasin" (never dishwasher or sink)
- To LIGHT/EXAMINE under light → use "desklamp": go to it, turn it on, THEN examine

Rules:
- Each subgoal is a single imperative phrase (≤8 words).
- Navigation subgoals use the form: "go to <location name>".
- NO conditional subgoals (no "if", "or", "else", "unless").
- Include ALL required intermediate steps (e.g. open container, clean before placing).
- If the task goal mentions "light", "lamp", or "examine … under", add
  "turn on desklamp N" as a subgoal BEFORE the examine subgoal.
- Output ONLY valid JSON, nothing else. Example:
  ["go to countertop 1", "pick up mug 1", "go to microwave 1", "heat mug 1 in microwave 1", "go to countertop 2", "place mug 1 on countertop 2"]
"""

EXECUTOR_SYSTEM = """\
You are a household robot choosing one action per step.

Overall task: {task_context}
Current subgoal: {subgoal}

Recent history (action → result):
{history}

Current state:
{state}

Admissible actions:
{actions}

Notes:
- If the subgoal mentions "turn on", choose the "turn on <object>" action from the list.
- If the subgoal mentions "examine" and a desklamp is in the admissible actions, ensure
  the lamp is already on (check history). If not, turn it on first.

Reply with ONLY the exact action string from the admissible list above.
Do not explain. Do not add punctuation.
"""

HIER_REFLECT_SYSTEM = """\
You are reviewing a failed text-based household task to improve the next plan.

Write your reflection as JSON with these exact fields:
{
  "task_goal": "brief description of what was required",
  "failed_subgoal": "the subgoal that went wrong or where the agent got stuck",
  "failure_reason": "one of: wrong_location | missing_step | wrong_object | stuck_in_loop | premature_completion",
  "missing_steps": ["any step absent from the plan that caused failure"],
  "revised_strategy": "specific instruction for the next plan (e.g. 'add open fridge 1 before pick up; search shelf 1 first')"
}

Be concrete. Generic statements like "I failed" or "try harder" are useless.
Example:
{
  "task_goal": "find and heat a mug",
  "failed_subgoal": "pick up mug 1",
  "failure_reason": "wrong_location",
  "missing_steps": ["open cabinet 1"],
  "revised_strategy": "Mug was in cabinet 1, not on counter. Add 'open cabinet 1' before 'pick up mug 1'."
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

class HierarchicalAgent(BaseAgent):
    """
    Hierarchical agent with Reflexion self-correction.

    Two-pass architecture:
      Pass 1 (planning)   — LLM produces an ordered JSON subgoal list once per trial.
      Pass 2 (execution)  — executor picks one action per step guided by current subgoal.

    After each failed trial a verbal reflection critiques the plan.  All
    reflections are injected into the Planner prompt so the model re-plans
    with explicit memory of its own mistakes (up to max_trials iterations).

    Episode flow:
      step 0  → _plan() → populate self._subgoals
      step 1+ → _execute() using current subgoal + compact history
    """

    def __init__(self, model_name: str, ollama_url: str, max_trials: int = 3):
        super().__init__(model_name, ollama_url)
        self.max_trials = max_trials
        self.reflections: List[str] = []   # persists across trials of one game
        self._subgoals: List[str] = []
        self._subgoal_idx: int = 0
        self._subgoal_steps: int = 0       # consecutive steps spent on current subgoal
        self._initial_obs: str = ""
        self._last_subgoals: List[str] = []   # saved for trajectory logging

    # ── State management ────────────────────────────────────────────────────────

    def reset(self):
        """Between games: clear reflections and all plan state."""
        self.reflections = []
        self._subgoals = []
        self._subgoal_idx = 0
        self._subgoal_steps = 0
        self._initial_obs = ""
        self._last_subgoals = []

    def reset_trial(self):
        """Between trials of the same game: clear per-trial plan, keep reflections."""
        self._subgoals = []
        self._subgoal_idx = 0
        self._subgoal_steps = 0
        self._initial_obs = ""
        self._last_subgoals = []
        # self.reflections is preserved so the next trial's planner sees them

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
                + "\n\nFor each reflection that mentions a missing step, add it to your plan.\n"
                + "For each reflection that mentions a wrong location, revise the navigation order.\n\n"
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
            task_context=initial_obs[:300] if initial_obs else "complete the household task",
            subgoal=subgoal,
            history=history_str,
            state=compact_state,
            actions="\n".join(f"  - {c}" for c in admissible_commands),
        )
        messages = [{"role": "user", "content": prompt}]
        raw = self.chat(messages, max_tokens=128)
        return self.match_command(raw, admissible_commands)

    # ── Reflection generation ────────────────────────────────────────────────────

    def reflect(self, initial_obs: str, trajectory: List[Tuple[str, str]]) -> str:
        """
        Generate a verbal reflection after a failed episode.

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

    def add_reflection(self, reflection: str):
        """Add a reflection, keeping at most max_trials entries."""
        self.reflections.append(reflection)
        if len(self.reflections) > self.max_trials:
            self.reflections = self.reflections[-self.max_trials:]

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _current_subgoal(self) -> str:
        if not self._subgoals:
            return "complete the task"
        idx = min(self._subgoal_idx, len(self._subgoals) - 1)
        return self._subgoals[idx]
