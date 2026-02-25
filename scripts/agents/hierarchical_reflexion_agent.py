"""
Hierarchical Reflexion Agent — ALFWorld implementation.

Combines two orthogonal improvements for Llama-3.1-8B on ALFWorld:
  1. HierarchicalAgent's two-pass design:
       Pass 1 — Planner generates an ordered JSON subgoal list (once per trial).
       Pass 2 — Executor picks one action per step, guided by the current subgoal.
     This decouples long-horizon planning from low-level action selection, directly
     addressing the core failure of 8B models on multi-step tasks.
  2. Reflexion's verbal self-correction across trials:
       After each failed trial, a reflection is generated that critiques the PLAN
       (wrong subgoal sequence, missing intermediate step, wrong predicted location).
       Reflections are injected into the Planner prompt so the model re-plans
       with explicit memory of its own mistakes.

Why this outperforms either component alone:
  - Pure Hierarchical: plan is generated once and never revised, so a bad plan
    causes irreversible failure.
  - Pure Reflexion (ReAct actor): 8B models still struggle with long-horizon
    reasoning in a flat scratchpad even after reflection.
  - Combined: planning errors are corrected across trials at the source (the plan),
    while each trial's execution remains focused via subgoal guidance.

Grounded in:
  - Shinn et al. (2023) Reflexion: Language Agents with Verbal Reinforcement Learning
  - The "Hierarchical + Feedback" insight from AdaPlanner (Sun et al., NeurIPS 2023)

Trial lifecycle (managed by evaluate_reflexion_agent in run_baseline.py):
    agent.reset()                # between games: clears reflections + plan state
    for trial in range(max_trials):
        agent.reset_trial()      # clears per-trial plan; keeps reflections
        result = run_episode(env, agent, max_steps, reflexion_heuristic=True)
        if result["success"]: break
        reflection = agent.reflect(result["initial_obs"], result["trajectory"])
        agent.add_reflection(reflection)
"""

from typing import List, Tuple

from .hierarchical_agent import HierarchicalAgent, PLANNER_SYSTEM

MAX_TRIALS = 3   # Reflexion paper cap: Ω = 1–3 reflections per game

HIER_REFLECT_SYSTEM = """\
You are reviewing a failed text-based household task to improve the next plan.

Write a concise reflection (3-5 sentences) covering:
  1. What the task required (goal decomposition: find X, do Y, place in Z).
  2. Where the subgoal plan went wrong — be specific:
     wrong location predicted, missing intermediate step (open container, clean/heat/cool),
     wrong object taken, subgoal marked complete too early, agent stuck in a loop.
  3. A revised planning strategy for the next attempt.
     Example: "Next time, add 'open fridge' before 'take item'. Also search
     shelf 1 first — the mug was not on the counter as initially assumed."

Generic observations like "I failed" or "I should try harder" are not useful.
"""


class HierarchicalReflexionAgent(HierarchicalAgent):
    """
    HierarchicalReflexion: subgoal planning + verbal self-correction.

    Each game is attempted up to MAX_TRIALS times.  After each failed trial a
    verbal reflection is generated that specifically critiques the PLAN (subgoal
    sequence and location predictions), and stored.  All prior reflections are
    prepended to the Planner system prompt for the next trial, so the model
    re-plans with explicit knowledge of its own past mistakes.

    This is the key distinction from ReflexionAgent (which applies feedback to
    ReAct execution): here, feedback targets the planning stage, where errors
    are most consequential for small models.
    """

    def __init__(self, model_name: str, ollama_url: str, max_trials: int = MAX_TRIALS):
        super().__init__(model_name, ollama_url)
        self.max_trials = max_trials
        self.reflections: List[str] = []   # persists across trials of one game

    # ── State management ────────────────────────────────────────────────────────

    def reset(self):
        """Between games: clear reflections and all plan state."""
        super().reset()
        self.reflections = []

    def reset_trial(self):
        """Between trials of the same game: clear per-trial plan, keep reflections."""
        self._subgoals = []
        self._subgoal_idx = 0
        self._initial_obs = ""
        self._last_subgoals = []
        # self.reflections is preserved so the next trial's planner sees them

    # ── Planning with injected reflections ──────────────────────────────────────

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
                "Reflections from your previous failed attempts on this task:\n"
                + reflection_block
                + "\n\nUse these to produce a better subgoal plan this time.\n\n"
                + PLANNER_SYSTEM
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Task and initial observation:\n{initial_obs}"},
        ]
        raw = self.chat(messages, max_tokens=400)
        return self._parse_subgoals(raw)

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
