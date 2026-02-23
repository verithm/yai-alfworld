from typing import List, Tuple
from .base_agent import BaseAgent

# ---------------------------------------------------------------------------
# Few-shot demonstrations drawn from the ALFWorld task taxonomy.
# Each example covers a different task type so the model sees breadth.
# Format mirrors what the agent will actually observe at runtime.
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    # Task type 1 — pick_and_place_simple
    {
        "task": "Put a pen in a drawer.",
        "trajectory": [
            ("look", "You are in the middle of a room. Looking quickly around you, you see a desk 1, a drawer 1, a shelf 1, and a bed 1."),
            ("go to desk 1", "On the desk 1, you see a pen 1, a book 1, and a laptop 1."),
            ("take pen 1 from desk 1", "You pick up the pen 1 from the desk 1."),
            ("go to drawer 1", "You arrive at drawer 1. The drawer 1 is closed."),
            ("open drawer 1", "You open the drawer 1. The drawer 1 is open. In it, you see nothing."),
            ("put pen 1 in/on drawer 1", "You put the pen 1 in/on the drawer 1."),
        ],
    },
    # Task type 5 — pick_cool_then_place_in_recep
    {
        "task": "Put a cool apple in a garbage can.",
        "trajectory": [
            ("look", "You are in the middle of a room. Looking quickly around you, you see a fridge 1, a garbage can 1, and a counter 1."),
            ("go to fridge 1", "You arrive at fridge 1. The fridge 1 is closed."),
            ("open fridge 1", "You open the fridge 1. In it, you see an apple 1 and a bottle 1."),
            ("take apple 1 from fridge 1", "You pick up the apple 1 from the fridge 1."),
            ("go to garbage can 1", "You arrive at garbage can 1. On the garbage can 1, you see nothing."),
            ("put apple 1 in/on garbage can 1", "You put the apple 1 in/on the garbage can 1."),
        ],
    },
    # Task type 3 — pick_clean_then_place_in_recep
    {
        "task": "Clean a mug and put it on a coffee table.",
        "trajectory": [
            ("look", "You are in the middle of a room. Looking quickly around you, you see a sink 1, a coffee table 1, and a shelf 1."),
            ("go to shelf 1", "On the shelf 1, you see a mug 1 and a plate 1."),
            ("take mug 1 from shelf 1", "You pick up the mug 1 from the shelf 1."),
            ("go to sink 1", "You arrive at sink 1. The sink 1 is not turned on."),
            ("clean mug 1 with sink 1", "You clean the mug 1 with the sink 1."),
            ("go to coffee table 1", "You arrive at coffee table 1."),
            ("put mug 1 in/on coffee table 1", "You put the mug 1 in/on the coffee table 1."),
        ],
    },
]


def _format_example(example: dict) -> str:
    lines = [f"Task: {example['task']}"]
    for i, (action, obs) in enumerate(example["trajectory"], 1):
        lines.append(f"  > Action {i}: {action}")
        lines.append(f"    Obs {i}: {obs}")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are an agent playing a text-based household task game.\n"
    "Study the examples below, then choose the best action for the current situation.\n"
    "Reply with EXACTLY ONE action from the valid actions list — nothing else.\n\n"
    "--- EXAMPLES ---\n"
    + "\n\n".join(_format_example(ex) for ex in FEW_SHOT_EXAMPLES)
    + "\n--- END EXAMPLES ---"
)


class FewShotAgent(BaseAgent):
    """Few-shot agent: prepends curated demonstrations before the current state."""

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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = self.chat(messages)
        return self.match_command(response, admissible_commands)
