#!/usr/bin/env python3
"""
Oracle Trajectory Collection for SFT
=====================================
Steps through ALFWorld training games using the built-in handcoded oracle
and saves each successful game as a multi-turn chat JSONL record for LoRA
fine-tuning.

Usage:
    export ALFWORLD_DATA=~/data/alfworld
    python scripts/collect_oracle_trajectories.py \\
      --split train \\
      --output results/oracle/trajectories.jsonl

Notes:
  - Requires ALFWORLD_DATA env var pointing to the downloaded game files.
  - configs/alfworld_config.yaml must have expert_type: handcoded (default).
  - The on-disk traj_data.json files use ALFRED vision-format actions and
    are NOT used here.  The text oracle is generated at runtime by the env.
  - Output is saved to local workspace disk — no VESSL volumes required.
  - Only successful oracle trajectories (score > 0) are written.
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import alfworld.agents.environment as environment
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# ── System prompt used for every SFT example ──────────────────────────────────

SYSTEM_PROMPT = """\
You are a household robot completing tasks in a text-based environment.
At each step you are given the current observation and a list of admissible actions.
Reply with ONLY the exact action string from the admissible list. Do not explain.\
"""

# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str, data_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    base = data_path.rstrip("/")
    ds = config.setdefault("dataset", {})
    ds["data_path"]          = f"{base}/json_2.1.1/train"
    ds["eval_id_data_path"]  = f"{base}/json_2.1.1/valid_seen"
    ds["eval_ood_data_path"] = f"{base}/json_2.1.1/valid_unseen"
    lg = config.setdefault("logic", {})
    lg["domain"]  = f"{base}/logic/alfred.pddl"
    lg["grammar"] = f"{base}/logic/alfred.twl2"
    return config


# ── Oracle trajectory collector ────────────────────────────────────────────────

def _collect_trajectory(
    initial_obs: str,
    initial_infos: dict,
    expert_plan: List[str],
    env,
    gamefile: str,
    max_steps: int,
) -> Optional[Dict[str, Any]]:
    """
    Follow the expert_plan action-by-action and build a multi-turn SFT record.
    Returns None if the episode did not end in success.
    """
    obs   = initial_obs
    infos = initial_infos

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    success = False

    for step, oracle_action in enumerate(expert_plan):
        if step >= max_steps:
            break

        admissible: List[str] = (infos.get("admissible_commands") or [[]])[0]
        if not admissible:
            break

        # Skip this game if the oracle action is not in the admissible set
        # (can happen with version mismatches or partial game files)
        if oracle_action not in admissible:
            return None

        user_turn = (
            f"Observation: {obs}\n\n"
            "Admissible actions:\n"
            + "\n".join(f"  - {a}" for a in admissible)
        )
        messages.append({"role": "user",      "content": user_turn})
        messages.append({"role": "assistant", "content": oracle_action})

        next_obs, scores, dones, infos = env.step([oracle_action])
        obs   = next_obs[0]
        done  = bool(dones[0])
        score = float(scores[0])

        if done:
            success = score > 0
            break

    if not success:
        return None

    # Number of (user, assistant) turn pairs
    n_steps = (len(messages) - 1) // 2

    return {
        "messages": messages,
        "gamefile": gamefile,
        "steps":    n_steps,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect ALFWorld oracle trajectories for SFT"
    )
    parser.add_argument(
        "--config", default="configs/alfworld_config.yaml",
        help="Path to ALFWorld YAML config",
    )
    parser.add_argument(
        "--split",  default="train",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
        help="Dataset split to collect from (default: train)",
    )
    parser.add_argument(
        "--output", default="results/oracle/trajectories.jsonl",
        help="Output JSONL file (default: results/oracle/trajectories.jsonl)",
    )
    parser.add_argument(
        "--max-collected", type=int, default=0,
        help="Stop after this many successful trajectories (0 = collect all)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Maximum steps per episode (default: 50)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    console.print("\n[bold blue]══════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]  ALFWorld Oracle Trajectory Collection  [/bold blue]")
    console.print("[bold blue]══════════════════════════════════════════════════[/bold blue]")
    console.print(f"  Split         : {args.split}")
    console.print(f"  Output        : {args.output}")
    console.print(
        f"  Max collected : {'all' if args.max_collected == 0 else args.max_collected}"
    )
    console.print()

    data_path = os.getenv("ALFWORLD_DATA", "/data/alfworld")
    config    = load_config(args.config, data_path)

    # Verify data path exists
    train_path = Path(config["dataset"]["data_path"])
    if not train_path.exists():
        console.print(f"[red]ERROR: data path not found: {train_path}[/red]")
        console.print("  Set ALFWORLD_DATA and run alfworld-download first.")
        sys.exit(1)

    env_class = environment.get_environment(config["env"]["type"])
    env       = env_class(config, train_eval=args.split).init_env(batch_size=1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    collected = 0
    skipped   = 0
    attempted = 0

    with open(out_path, "w") as fout:
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]oracle collection[/cyan]"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.description}"),
            console=console,
        ) as progress:
            prog_task = progress.add_task("", total=None)

            while True:
                if args.max_collected > 0 and collected >= args.max_collected:
                    break

                # Advance the env iterator to the next game
                try:
                    obs, infos = env.reset()
                except StopIteration:
                    break

                attempted += 1
                obs_str   = obs[0]
                gamefile  = str((infos.get("extra.gamefile") or [""])[0])

                # The handcoded oracle plan is provided per-game in infos
                expert_plan: List[str] = (
                    infos.get("extra.expert_plan") or [[]]
                )[0]

                if not expert_plan:
                    skipped += 1
                    progress.update(
                        prog_task,
                        description=f"collected={collected} skipped={skipped} (no oracle plan)",
                    )
                    continue

                record = _collect_trajectory(
                    obs_str, infos, expert_plan, env, gamefile, args.max_steps
                )

                if record is None:
                    skipped += 1
                    progress.update(
                        prog_task,
                        description=f"collected={collected} skipped={skipped}",
                    )
                    continue

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                collected += 1
                progress.update(
                    prog_task,
                    description=(
                        f"collected={collected} skipped={skipped} "
                        f"steps={record['steps']}"
                    ),
                )

    console.print(f"\n  Attempted  : {attempted} games")
    console.print(f"  Collected  : [green]{collected}[/green] trajectories")
    console.print(
        f"  Skipped    : {skipped} "
        f"({'no oracle / failed / non-success'})"
    )
    console.print(f"  Output     : {out_path}")

    if collected == 0:
        console.print(
            "\n[yellow]Warning: 0 trajectories collected.[/yellow]\n"
            "  Check that expert_type: handcoded is set in the config\n"
            "  and that ALFWORLD_DATA points to the correct directory."
        )


if __name__ == "__main__":
    main()
