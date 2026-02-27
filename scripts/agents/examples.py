"""
Canonical few-shot examples shared by ReActAgent and FewShotAgent.

Each trajectory is defined once as a list of (thought, action, observation) steps:
  - thought : reasoning shown before the action (None = no thought for this step)
  - action  : the exact ALFWorld action string
  - observation : the environment response

Two renderers produce the format each agent needs:
  REACT_EXAMPLES[task_type] — Thought/Action/Observation (for ReActAgent)
  ACT_EXAMPLES[task_type]   — Action/Observation only   (for FewShotAgent)

This guarantees experimental consistency: both agents solve identical tasks
with identical action sequences; the only difference is the presence of
explicit reasoning steps.

Sources:
  pick_clean_then_place — taken verbatim from the ReAct paper (Yao et al., 2022),
                          Appendix Table prompt:alfworld_im.
  look_at_obj           — derived from the Reflexion paper (Shinn et al., 2023),
                          Trial #2 success trajectory (Figure in Appendix).
  Remaining four types  — follow the same room style and thought patterns.
"""

from typing import Dict, List, Optional, Tuple

# (thought, action, observation)
Step = Tuple[Optional[str], str, str]

# ── Canonical trajectory data ──────────────────────────────────────────────────

_EXAMPLES: Dict[str, Dict] = {

    # ── 1. pick_and_place_simple ───────────────────────────────────────────────
    "pick_and_place": dict(
        task="put a pen in a drawer.",
        initial_obs=(
            "You are in the middle of a room. Looking quickly around you, you see "
            "a bed 1, a desk 2, a desk 1, a drawer 2, a drawer 1, a garbagecan 1, "
            "a laundryhamper 1, a safe 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, "
            "a shelf 2, and a shelf 1."
        ),
        steps=[
            (
                "To solve the task, I need to find and take a pen, then put it in a drawer. "
                "First I need to find a pen. A pen is more likely to appear in desk (1-2), "
                "drawer (1-2), shelf (1-6), bed (1). I can check one by one, starting with desk 1.",
                "go to desk 1",
                "On the desk 1, you see a alarmclock 1, a cd 1, a creditcard 1, a pencil 2, and a pen 1.",
            ),
            (
                "Now I find a pen (1). Next, I need to take it.",
                "take pen 1 from desk 1",
                "You pick up the pen 1 from the desk 1.",
            ),
            (
                "Now I take a pen (1). Next, I need to put it in a drawer. I'll try drawer 1.",
                "go to drawer 1",
                "The drawer 1 is closed.",
            ),
            (
                None,
                "open drawer 1",
                "You open the drawer 1. The drawer 1 is open. In it, you see nothing.",
            ),
            (
                None,
                "put pen 1 in/on drawer 1",
                "You put the pen 1 in/on the drawer 1.",
            ),
        ],
    ),

    # ── 2. pick_clean_then_place_in_recep ─────────────────────────────────────
    # Taken verbatim from the ReAct paper (alfworld_im prompt, Appendix).
    "pick_clean_then_place": dict(
        task="put a clean lettuce in diningtable.",
        initial_obs=(
            "You are in the middle of a room. Looking quickly around you, you see "
            "a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, "
            "a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, "
            "a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, "
            "a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, "
            "a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, "
            "a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1."
        ),
        steps=[
            (
                "To solve the task, I need to find and take a lettuce, then clean it with "
                "sinkbasin, then put it in diningtable. "
                "First I need to find a lettuce. A lettuce is more likely to appear in "
                "fridge (1), diningtable (1), sinkbasin (1), stoveburner (1-3), cabinet (1-13). "
                "I can check one by one, starting with fridge 1.",
                "go to fridge 1",
                "The fridge 1 is closed.",
            ),
            (
                None,
                "open fridge 1",
                "You open the fridge 1. The fridge 1 is open. "
                "In it, you see a cup 3, a egg 2, a potato 3, and a potato 2.",
            ),
            (
                None,
                "go to diningtable 1",
                "On the diningtable 1, you see a apple 1, a bread 1, a butterknife 2, "
                "a cup 2, a fork 2, a knife 2, a knife 1, a ladle 1, a lettuce 1, "
                "a mug 2, a mug 1, a pan 2, a peppershaker 1, a spatula 3, "
                "a tomato 2, and a tomato 1.",
            ),
            (
                "Now I find a lettuce (1). Next, I need to take it.",
                "take lettuce 1 from diningtable 1",
                "You pick up the lettuce 1 from the diningtable 1.",
            ),
            (
                "Now I take a lettuce (1). Next, I need to go to sinkbasin (1) and clean it.",
                "go to sinkbasin 1",
                "On the sinkbasin 1, you see a apple 2, a ladle 2, a spoon 1, and a tomato 3.",
            ),
            (
                None,
                "clean lettuce 1 with sinkbasin 1",
                "You clean the lettuce 1 using the sinkbasin 1.",
            ),
            (
                "Now I clean a lettuce (1). Next, I need to put it in/on diningtable 1.",
                "go to diningtable 1",
                "On the diningtable 1, you see a apple 1, a bread 1, a butterknife 2, "
                "a cup 2, a fork 2, a knife 2, a knife 1, a ladle 1, a mug 2, a mug 1, "
                "a pan 2, a peppershaker 1, a spatula 3, a tomato 2, and a tomato 1.",
            ),
            (
                None,
                "put lettuce 1 in/on diningtable 1",
                "You put the lettuce 1 in/on the diningtable 1.",
            ),
        ],
    ),

    # ── 3. pick_heat_then_place_in_recep ──────────────────────────────────────
    "pick_heat_then_place": dict(
        task="put a hot apple in countertop.",
        initial_obs=(
            "You are in the middle of a room. Looking quickly around you, you see "
            "a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, "
            "a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, "
            "a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, "
            "a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, "
            "a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, "
            "and a toaster 1."
        ),
        steps=[
            (
                "To solve the task, I need to find and take an apple, then heat it with "
                "microwave, then put it in countertop. "
                "First I need to find an apple. An apple is more likely to appear in "
                "fridge (1), countertop (1-3), cabinet (1-6), shelf (1-3). "
                "I can check one by one, starting with fridge 1.",
                "go to fridge 1",
                "The fridge 1 is closed.",
            ),
            (
                None,
                "open fridge 1",
                "You open the fridge 1. The fridge 1 is open. "
                "In it, you see a apple 1, a bottle 3, and a egg 1.",
            ),
            (
                "Now I find an apple (1). Next, I need to take it.",
                "take apple 1 from fridge 1",
                "You pick up the apple 1 from the fridge 1.",
            ),
            (
                "Now I take an apple (1). Next, I need to go to microwave (1) and heat it.",
                "go to microwave 1",
                "The microwave 1 is closed.",
            ),
            (
                None,
                "open microwave 1",
                "You open the microwave 1. The microwave 1 is open. In it, you see nothing.",
            ),
            (
                None,
                "heat apple 1 with microwave 1",
                "You heat the apple 1 using the microwave 1.",
            ),
            (
                "Now I heat an apple (1). Next, I need to put it in/on countertop 1.",
                "go to countertop 1",
                "On the countertop 1, you see a butterknife 1, a cup 2, and a plate 1.",
            ),
            (
                None,
                "put apple 1 in/on countertop 1",
                "You put the apple 1 in/on the countertop 1.",
            ),
        ],
    ),

    # ── 4. pick_cool_then_place_in_recep ──────────────────────────────────────
    "pick_cool_then_place": dict(
        task="put a cool tomato in countertop.",
        initial_obs=(
            "You are in the middle of a room. Looking quickly around you, you see "
            "a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, "
            "a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, "
            "a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, "
            "a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, "
            "a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, "
            "and a toaster 1."
        ),
        steps=[
            (
                "To solve the task, I need to find and take a tomato, then cool it with "
                "fridge, then put it in countertop. "
                "First I need to find a tomato. A tomato is more likely to appear in "
                "diningtable (1), countertop (1-3), fridge (1), cabinet (1-6), garbagecan (1). "
                "I can check one by one, starting with diningtable 1.",
                "go to diningtable 1",
                "On the diningtable 1, you see a apple 2, a bread 1, a fork 2, a knife 2, "
                "a ladle 1, a peppershaker 2, a spatula 2, a tomato 1, and a tomato 2.",
            ),
            (
                "Now I find a tomato (1). Next, I need to take it.",
                "take tomato 1 from diningtable 1",
                "You pick up the tomato 1 from the diningtable 1.",
            ),
            (
                "Now I take a tomato (1). Next, I need to go to fridge (1) and cool it.",
                "go to fridge 1",
                "The fridge 1 is closed.",
            ),
            (
                None,
                "open fridge 1",
                "You open the fridge 1. The fridge 1 is open. In it, you see a cup 1 and a egg 2.",
            ),
            (
                None,
                "cool tomato 1 with fridge 1",
                "You cool the tomato 1 using the fridge 1.",
            ),
            (
                "Now I cool a tomato (1). Next, I need to put it in/on countertop 1.",
                "go to countertop 1",
                "On the countertop 1, you see a dishsponge 1 and a mug 2.",
            ),
            (
                None,
                "put tomato 1 in/on countertop 1",
                "You put the tomato 1 in/on the countertop 1.",
            ),
        ],
    ),

    # ── 5. pick_two_obj_and_place ──────────────────────────────────────────────
    "pick_two_obj": dict(
        task="put two mugs in shelf.",
        initial_obs=(
            "You are in the middle of a room. Looking quickly around you, you see "
            "a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, "
            "a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, "
            "a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, "
            "a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, "
            "a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, "
            "and a toaster 1."
        ),
        steps=[
            (
                "To solve the task, I need to find and place two mugs on a shelf, one at a time. "
                "First I need to find a mug. A mug is more likely to appear in cabinet (1-6), "
                "countertop (1-3), shelf (1-3), coffeemachine (1), drawer (1-3). "
                "I can check one by one, starting with cabinet 1.",
                "go to cabinet 1",
                "On the cabinet 1, you see a mug 1 and a plate 1.",
            ),
            (
                "Now I find a mug (1). Next, I need to take it and place it on shelf 1.",
                "take mug 1 from cabinet 1",
                "You pick up the mug 1 from the cabinet 1.",
            ),
            (
                None,
                "go to shelf 1",
                "On the shelf 1, you see a bowl 1.",
            ),
            (
                None,
                "put mug 1 in/on shelf 1",
                "You put the mug 1 in/on the shelf 1.",
            ),
            (
                "I placed the first mug. Now I need to find a second mug. I'll check countertop 1.",
                "go to countertop 1",
                "On the countertop 1, you see a mug 2 and a spoon 1.",
            ),
            (
                "Now I find a second mug (2). Next, I need to take it.",
                "take mug 2 from countertop 1",
                "You pick up the mug 2 from the countertop 1.",
            ),
            (
                "Now I take a second mug (2). Next, I need to place it on shelf 1 as well.",
                "go to shelf 1",
                "On the shelf 1, you see a bowl 1 and a mug 1.",
            ),
            (
                None,
                "put mug 2 in/on shelf 1",
                "You put the mug 2 in/on the shelf 1.",
            ),
        ],
    ),

    # ── 6. look_at_obj_in_light ───────────────────────────────────────────────
    # Derived from Reflexion paper (Shinn et al., 2023), Trial #2 success.
    # Key insight: find desklamp first; take mug and use desklamp from the same
    # desk — moving away before using desklamp causes "Nothing happens."
    "look_at_obj": dict(
        task="examine the mug with the desklamp.",
        initial_obs=(
            "You are in the middle of a room. Looking quickly around you, you see "
            "a bed 1, a desk 2, a desk 1, a drawer 6, a drawer 5, a drawer 4, "
            "a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, "
            "a safe 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, "
            "and a shelf 1."
        ),
        steps=[
            (
                "To solve the task, I need to find a desklamp and a mug, then use the "
                "desklamp to examine the mug. I must use the desklamp from the same "
                "location where it sits — moving away first causes 'Nothing happens.' "
                "First I need to find a desklamp. A desklamp is more likely to appear in "
                "desk (1-2), shelf (1-6), drawer (1-6). "
                "I can check one by one, starting with desk 1.",
                "go to desk 1",
                "On the desk 1, you see a creditcard 3, a desklamp 1, a laptop 2, "
                "a mug 1, a pen 1, and a pencil 1.",
            ),
            (
                "Now I find a desklamp (1) and a mug (1) on the same desk. "
                "I will take the mug and use the desklamp here without moving away.",
                "take mug 1 from desk 1",
                "You pick up the mug 1 from the desk 1.",
            ),
            (
                "Now I take a mug (1). The desklamp (1) is on desk 1 where I am. "
                "I can use it now to examine the mug.",
                "use desklamp 1",
                "You turn on the desklamp 1.",
            ),
            (
                "Now the desklamp is on. Next, I need to examine the mug with it.",
                "examine mug 1",
                "You examine the mug 1.",
            ),
        ],
    ),
}


# ── Rendering functions ────────────────────────────────────────────────────────

def _format_react(ex: dict) -> str:
    """Render a trajectory with Thought/Action/Observation for ReActAgent."""
    lines = [
        f"Task: {ex['task']}",
        f"Observation: {ex['initial_obs']}",
    ]
    for thought, action, observation in ex["steps"]:
        if thought:
            lines.append(f"Thought: {thought}")
        lines.append(f"Action: {action}")
        lines.append(f"Observation: {observation}")
    return "\n".join(lines)


def _format_act(ex: dict) -> str:
    """Render a trajectory with Action/Observation only for FewShotAgent."""
    lines = [
        f"Task: {ex['task']}",
        f"Observation: {ex['initial_obs']}",
    ]
    for _thought, action, observation in ex["steps"]:
        lines.append(f"Action: {action}")
        lines.append(f"Observation: {observation}")
    return "\n".join(lines)


# Pre-rendered for fast lookup at runtime
REACT_EXAMPLES: Dict[str, str] = {k: _format_react(v) for k, v in _EXAMPLES.items()}
ACT_EXAMPLES:   Dict[str, str] = {k: _format_act(v)   for k, v in _EXAMPLES.items()}
