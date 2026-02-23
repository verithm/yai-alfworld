from typing import List, Tuple
from .base_agent import BaseAgent

MAX_TRIALS = 3

ACTOR_SYSTEM_PROMPT = """\
You are an agent playing a text-based household task game.
Your goal is to complete the assigned task by issuing actions.
You will be given your current observation and a list of valid actions.
Reply with EXACTLY ONE action from the list — nothing else.
"""

REFLECT_SYSTEM_PROMPT = """\
You are a self-reflecting agent. You just failed a text-based household task.
Analyze what went wrong and write a concise reflection (2-4 sentences) that will
help you do better on the next attempt. Focus on:
- What the task required you to do
- What mistake you made (wrong object, wrong sequence, got stuck in a loop)
- One concrete thing you will do differently next time
"""


class ReflexionAgent(BaseAgent):
    """
    Reflexion agent (Shinn et al., 2023) — official ALFWorld variant.

    Each game is attempted up to MAX_TRIALS times.  After each failed trial a
    verbal self-reflection is generated and stored.  All prior reflections are
    prepended to the actor system prompt for subsequent trials, giving the model
    explicit memory of its own mistakes.

    Trial lifecycle (managed by the runner in run_baseline.py):
        for trial in range(max_trials):
            agent.reset_trial()
            run_episode(env, agent, max_steps)   # calls act() at each step
            if success: break
            reflection = agent.reflect(trajectory)
            agent.add_reflection(reflection)
        agent.reset()  # between games
    """

    def __init__(self, model_name: str, ollama_url: str, max_trials: int = MAX_TRIALS):
        super().__init__(model_name, ollama_url)
        self.max_trials = max_trials
        self.reflections: List[str] = []   # persists across trials of one game

    def reset(self):
        """Call between games — clears reflections and history."""
        self.reflections = []

    def reset_trial(self):
        """Call between trials of the same game — keeps reflections."""
        pass  # actor is stateless per-step; nothing to reset here

    def _actor_system_prompt(self) -> str:
        prompt = ACTOR_SYSTEM_PROMPT
        if self.reflections:
            prompt += "\n\nYour reflections from previous failed attempts on this task:\n"
            for i, r in enumerate(self.reflections, 1):
                prompt += f"  [Attempt {i}]: {r}\n"
        return prompt

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
            "Choose ONE action from the list above and reply with it exactly."
        )

        messages = [
            {"role": "system", "content": self._actor_system_prompt()},
            {"role": "user",   "content": user_content},
        ]

        response = self.chat(messages)
        return self.match_command(response, admissible_commands)

    def reflect(self, initial_obs: str, trajectory: List[Tuple[str, str]]) -> str:
        """
        Generate a verbal reflection after a failed episode.

        Args:
            initial_obs:  The first observation of the episode (contains goal).
            trajectory:   List of (action, resulting_observation) pairs.
        """
        traj_lines = [
            f"  {i + 1}. {act}: {obs[:120]}"
            for i, (act, obs) in enumerate(trajectory[-10:])
        ]
        traj_str = "\n".join(traj_lines)

        user_content = (
            f"Task context (initial observation):\n{initial_obs[:300]}\n\n"
            f"Your trajectory (last {len(traj_lines)} steps):\n{traj_str}\n\n"
            "Write your reflection:"
        )

        messages = [
            {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        return self.chat(messages, max_tokens=150)

    def add_reflection(self, reflection: str):
        self.reflections.append(reflection)
