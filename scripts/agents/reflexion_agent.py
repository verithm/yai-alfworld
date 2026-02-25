"""
Reflexion Agent (Shinn et al., 2023) — ALFWorld implementation.

Key design decisions matching the paper:
  - Actor IS ReAct (inherits ReActAgent), not a plain action-selector
  - Verbal self-reflection generated after each failed trial and stored
  - Reflections prepended to the ReAct system prompt for the next trial,
    giving the model explicit memory of its own mistakes
  - Memory capped at MAX_TRIALS reflections (paper cap = 3)
  - reset_trial() clears per-trial scratchpad (_react_history) only;
    cross-trial state (reflections + _task_type) is preserved
  - Early trial termination via reflexion_heuristic in run_baseline.py:
    same (action, obs) pair repeated 3+ times OR step >= 30 → end trial
"""

from typing import List, Tuple

from .react_agent import ReActAgent

MAX_TRIALS = 3

REFLECT_SYSTEM_PROMPT = """\
You are a self-reflecting agent reviewing a failed text-based household task.

Write a concise reflection (3-5 sentences) covering:
  1. What the task required (goal decomposition: find X, do Y, place in Z).
  2. Where exactly the attempt went wrong — be specific:
     wrong location searched, skipped opening a container, stuck in a loop,
     wrong object taken, forgot a required step (clean/heat/cool).
  3. A concrete revised plan for the next attempt.
     Example: "Next time I will check shelf 1 first, then cabinet 1-3,
     before trying desk. I must open the fridge before taking the item."

Generic observations like "I failed" or "I should try harder" are not useful.
"""


class ReflexionAgent(ReActAgent):
    """
    Reflexion agent (Shinn et al., 2023) — official ALFWorld variant.

    Each game is attempted up to MAX_TRIALS times.  After each failed trial a
    verbal self-reflection is generated and stored.  All prior reflections are
    prepended to the ReAct system prompt for subsequent trials, giving the model
    explicit memory of its own mistakes.

    Trial lifecycle (managed by the runner in run_baseline.py):
        agent.reset()               # between games: clears reflections + all ReAct state
        for trial in range(max_trials):
            agent.reset_trial()     # clears scratchpad only; keeps reflections + task_type
            result = run_episode(env, agent, max_steps, reflexion_heuristic=True)
            if result["success"]: break
            reflection = agent.reflect(result["initial_obs"], result["trajectory"])
            agent.add_reflection(reflection)
    """

    def __init__(self, model_name: str, ollama_url: str, max_trials: int = MAX_TRIALS):
        super().__init__(model_name, ollama_url)
        self.max_trials = max_trials
        self.reflections: List[str] = []   # persists across trials of one game

    # ── State management ──────────────────────────────────────────────────────

    def reset(self):
        """Call between games — clears reflections and all ReAct state."""
        super().reset()          # clears _react_history + _task_type
        self.reflections = []

    def reset_trial(self):
        """Call between trials of the same game — clears scratchpad only."""
        self._react_history = []   # per-trial scratchpad
        # Preserve self.reflections and self._task_type across trials

    # ── System prompt with injected reflections ───────────────────────────────

    def _get_system_prompt(self) -> str:
        """ReAct system prompt, optionally prefixed with past reflections."""
        react_prompt = super()._get_system_prompt()

        if not self.reflections:
            return react_prompt

        reflection_block = "\n".join(
            f"[Attempt {i}]: {r}" for i, r in enumerate(self.reflections, 1)
        )
        return (
            "Reflections from your previous failed attempts on this task:\n"
            + reflection_block
            + "\n\nUse these reflections to avoid repeating past mistakes.\n\n"
            + react_prompt
        )

    # ── Reflection generation ─────────────────────────────────────────────────

    def reflect(self, initial_obs: str, trajectory: List[Tuple[str, str]]) -> str:
        """
        Generate a verbal reflection after a failed episode.

        Args:
            initial_obs:  The first observation of the episode (contains goal).
            trajectory:   List of (action, resulting_observation) pairs.
        """
        traj_lines = [
            f"  {i + 1}. {act}: {obs[:120]}"
            for i, (act, obs) in enumerate(trajectory[-15:])
        ]
        traj_str = "\n".join(traj_lines)

        user_content = (
            f"Task context (initial observation):\n{initial_obs[:400]}\n\n"
            f"Your trajectory (last {len(traj_lines)} steps):\n{traj_str}\n\n"
            "Write your reflection:"
        )

        messages = [
            {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        return self.chat(messages, max_tokens=200)

    def add_reflection(self, reflection: str):
        """Add a reflection, keeping at most max_trials entries."""
        self.reflections.append(reflection)
        if len(self.reflections) > self.max_trials:
            self.reflections = self.reflections[-self.max_trials:]
