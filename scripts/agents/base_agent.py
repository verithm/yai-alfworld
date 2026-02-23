from abc import ABC, abstractmethod
from typing import List, Tuple
import requests


class BaseAgent(ABC):
    """Base class for ALFWorld agents."""

    def __init__(self, model_name: str, ollama_url: str):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")

    def chat(self, messages: List[dict], max_tokens: int = 128) -> str:
        """Send messages to Ollama and return the response text."""
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.1,   # slight stochasticity to break loops
                    "num_predict": max_tokens,
                    "num_thread": 12,     # match CPU thread allocation
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
