from abc import ABC, abstractmethod
from typing import List, Tuple
import requests


class BaseAgent(ABC):
    """Base class for ALFWorld agents."""

    def __init__(self, model_name: str, ollama_url: str):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")

    def chat(self, messages: List[dict], max_tokens: int = 128) -> str:
        """Send messages to Ollama and return the response text.

        Inference hyperparameters (shared across ALL agents for experimental consistency):
          temperature=0.1  — Near-deterministic but with slight stochasticity to break
                             action loops; pure 0.0 causes identical repetitive outputs.
                             Value is low enough that results are reproducible across runs.
          num_predict      — Per-agent ceiling on output tokens; passed by caller
                             (128 for action-only outputs, 200 for reasoning, 400 for JSON plans).
          num_thread=12    — CPU threads for Ollama tokenisation on the VESSL GPU instance
                             (16 vCPUs assigned, 12 reserved for LLM preprocessing).
                             Has no effect on GPU matrix multiply, only on CPU-side work.
          timeout=180      — 3-minute network timeout; generous for 8B model inference
                             on an RTX 3090 (~0.5–2 s/step typical, worst-case ~30 s).
        """
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": max_tokens,
                    "num_thread": 12,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def match_command(self, response: str, admissible_commands: List[str]) -> str:
        """Fuzzy-match model output to the closest admissible command."""
        response_lower = response.lower().strip()

        # 1. Exact match
        for cmd in admissible_commands:
            if response_lower == cmd.lower():
                return cmd

        # 2. Response contains the command
        for cmd in admissible_commands:
            if cmd.lower() in response_lower:
                return cmd

        # 3. Command contains the response
        for cmd in admissible_commands:
            if response_lower in cmd.lower():
                return cmd

        # 4. Fallback: first admissible command (avoids invalid actions)
        return admissible_commands[0] if admissible_commands else "look"

    def reset(self):
        """Reset all state between games. Override in subclasses if needed."""
        pass

    def reset_trial(self):
        """Reset per-trial state within a game. Default: full reset.
        Override in agents that must preserve cross-trial state (e.g. Reflexion)."""
        self.reset()

    @abstractmethod
    def act(
        self,
        observation: str,
        admissible_commands: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
        """Return the action to take given the current observation."""
        ...
