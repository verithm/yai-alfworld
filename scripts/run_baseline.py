#!/usr/bin/env python3
"""
ALFWorld Baseline Evaluation
=============================
Evaluates Zero-shot, Few-shot, and ReAct agents on ALFWorld text-only tasks
using Llama-3.1-8B served via Ollama (CPU inference).

Usage (inside Docker):
    python scripts/run_baseline.py [OPTIONS]

Options:
    --config        Path to alfworld YAML config  (default: configs/alfworld_config.yaml)
    --ollama-url    Ollama server URL              (default: $OLLAMA_URL or http://localhost:11434)
    --model         Model name in Ollama           (default: $MODEL_NAME or llama3.1:8b)
    --num-games     Games to evaluate per agent    (default: $NUM_GAMES or 10)
    --max-steps     Max actions per episode        (default: $MAX_STEPS or 50)
    --agents        Agents to run (space-sep)      (default: zero_shot few_shot react)
    --output        JSON results file path         (default: results/baseline_results.json)
    --split         Dataset split to evaluate      (default: eval_out_of_distribution)
"""

import os
import re
import sys
import json
import yaml
import random
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import alfworld.agents.environment as environment
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Make sure the scripts/ directory is importable
sys.path.insert(0, str(Path(__file__).parent))
from agents import ZeroShotAgent, FewShotAgent, CotAgent, ReActAgent, ReflexionAgent, HierarchicalAgent

console = Console()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save_checkpoint(path: str, data: Dict) -> None:
    """Atomically write checkpoint — safe against mid-write interruption."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)   # POSIX atomic rename


def _load_checkpoint(path: str) -> Dict:
    """Return existing checkpoint dict, or an empty skeleton."""
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {}


LOOP_WINDOW = 3  # consecutive identical actions before forcing escape

TASK_TYPE_RE = re.compile(
    r"(pick_and_place_simple"
    r"|pick_clean_then_place_in_recep"
    r"|pick_heat_then_place_in_recep"
    r"|pick_cool_then_place_in_recep"
    r"|pick_two_obj_and_place"
    r"|look_at_obj_in_light)"
)


def _task_type(gamefile: str) -> str:
    m = TASK_TYPE_RE.search(gamefile)
    return m.group(1) if m else "unknown"


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str, data_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Override all data/logic paths so the same config works in Docker and Colab
    base = data_path.rstrip("/")   # e.g. /data/alfworld  or  /content/drive/.../alfworld
    ds = config.setdefault("dataset", {})
    ds["data_path"]         = f"{base}/json_2.1.1/train"
    ds["eval_id_data_path"] = f"{base}/json_2.1.1/valid_seen"
    ds["eval_ood_data_path"]= f"{base}/json_2.1.1/valid_unseen"
    lg = config.setdefault("logic", {})
    lg["domain"]  = f"{base}/logic/alfred.pddl"
    lg["grammar"] = f"{base}/logic/alfred.twl2"
    return config


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(env, agent, max_steps: int) -> Dict[str, Any]:
    """
    Run one episode.
    Returns dict with keys: success, steps, score, actions.
    """
    try:
        obs, infos = env.reset()
    except StopIteration:
        # No more games in the split
        return None

    obs = obs[0]          # unwrap batch dim
    initial_obs = obs     # capture before any action (used by Reflexion reflect())
    gamefile = str((infos.get("extra.gamefile") or [""])[0])
    task_type = _task_type(gamefile)
    agent.reset_trial()   # use reset_trial() so Reflexion preserves cross-trial reflections
    history: List[Tuple[str, str]] = []
    actions_taken: List[str] = []
    loops_broken = 0

    for step in range(max_steps):
        admissible = infos.get("admissible_commands", [[]])[0]
        if not admissible:
            break

        try:
            action = agent.act(obs, admissible, history)
        except Exception as e:
            console.print(f"    [red]Agent error at step {step}: {e}[/red]")
            action = admissible[0]   # fall back to first valid action

        # Loop detection: escape if last LOOP_WINDOW actions are identical
        if (len(actions_taken) >= LOOP_WINDOW
                and len(set(actions_taken[-LOOP_WINDOW:])) == 1
                and actions_taken[-1] == action):
            alternatives = [cmd for cmd in admissible if cmd != action]
            if alternatives:
                action = random.choice(alternatives)
                loops_broken += 1

        next_obs, scores, dones, infos = env.step([action])
        next_obs = next_obs[0]
        score = float(scores[0])
        done = bool(dones[0])

        history.append((action, next_obs))
        actions_taken.append(action)
        obs = next_obs

        if done:
            return {
                "success": score > 0,
                "steps": step + 1,
                "score": score,
                "actions": actions_taken,
                "trajectory": history,
                "initial_obs": initial_obs,
                "task_type": task_type,
                "loops_broken": loops_broken,
                "subgoals": list(getattr(agent, "_last_subgoals", [])),
            }

    return {
        "success": False,
        "steps": max_steps,
        "score": 0.0,
        "actions": actions_taken,
        "trajectory": history,
        "initial_obs": initial_obs,
        "task_type": task_type,
        "loops_broken": loops_broken,
        "subgoals": list(getattr(agent, "_last_subgoals", [])),
    }


# ── Env factory with 0-game guard ─────────────────────────────────────────────

_SPLIT_KEY = {
    "train":                    "data_path",
    "eval_in_distribution":     "eval_id_data_path",
    "eval_out_of_distribution": "eval_ood_data_path",
}

def _make_env(env_class, config: dict, split: str):
    """Instantiate and initialise AlfredTWEnv; raise immediately if 0 games found."""
    env = env_class(config, train_eval=split)
    game_files = getattr(env, "game_files", None)
    if game_files is not None and len(game_files) == 0:
        key  = _SPLIT_KEY.get(split, "data_path")
        path = config["dataset"].get(key, "unknown")
        raise RuntimeError(
            f"0 games found for split='{split}'.\n"
            f"  Path : {path}\n"
            f"  Exists: {Path(path).exists()}\n"
            f"  Fix  : verify ALFWORLD_DATA and re-run alfworld-download."
        )
    return env.init_env(batch_size=1)


# ── Reflexion trial runner ─────────────────────────────────────────────────────

def run_reflexion_trials(
    config: dict,
    agent,          # ReflexionAgent
    num_games: int,
    max_steps: int,
    split: str,
) -> List[Dict]:
    """
    Evaluate a ReflexionAgent over num_games games, each with up to
    agent.max_trials attempts.

    Strategy: for each game index k, a fresh AlfredTWEnv is created and
    env.reset() is called (k+1) times so the iterator lands on game k.
    This guarantees the same game is replayed across all trials without
    requiring low-level env surgery.
    """
    env_class = environment.get_environment(config["env"]["type"])
    all_results: List[Dict] = []

    for game_idx in range(num_games):
        agent.reset()          # clear reflections between games
        game_final: Dict = {}

        for trial_num in range(agent.max_trials):
            # Fresh env + skip to game_idx
            trial_env = _make_env(env_class, config, split)
            skip_ok = True
            for _ in range(game_idx):
                try:
                    trial_env.reset()
                except StopIteration:
                    skip_ok = False
                    break
            if not skip_ok:
                break

            agent.reset_trial()
            result = run_episode(trial_env, agent, max_steps)
            if result is None:
                break

            result["trial"] = trial_num + 1
            result["reflections"] = list(agent.reflections)
            game_final = result

            if result["success"]:
                break

            # Generate reflection for the next trial
            reflection = agent.reflect(
                initial_obs=result.get("initial_obs", ""),
                trajectory=result.get("trajectory", []),
            )
            agent.add_reflection(reflection)

        if game_final:
            all_results.append(game_final)

    return all_results


def evaluate_reflexion_agent(
    config: dict,
    agent,
    agent_name: str,
    num_games: int,
    max_steps: int,
    split: str,
    start_game: int = 0,
    existing_games: List[Dict] = None,
    on_game_done: Any = None,
) -> Dict[str, Any]:
    """Evaluate a ReflexionAgent — wraps run_reflexion_trials with stats."""
    console.print(f"  Max trials per game: {agent.max_trials}")
    if start_game > 0:
        console.print(f"  [dim]Resuming from game {start_game + 1}/{num_games}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]{agent_name}[/cyan]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("", total=num_games)
        if start_game > 0:
            progress.update(task, advance=start_game,
                            description=f"(resumed, {start_game} games loaded)")
        game_results: List[Dict] = list(existing_games or [])

        env_class = environment.get_environment(config["env"]["type"])
        for game_idx in range(start_game, num_games):
            agent.reset()
            game_final: Dict = {}

            for trial_num in range(agent.max_trials):
                trial_env = _make_env(env_class, config, split)
                skip_ok = True
                for _ in range(game_idx):
                    try:
                        trial_env.reset()
                    except StopIteration:
                        skip_ok = False
                        break
                if not skip_ok:
                    break

                agent.reset_trial()
                result = run_episode(trial_env, agent, max_steps)
                if result is None:
                    break

                result["trial"] = trial_num + 1
                result["reflections"] = list(agent.reflections)
                game_final = result

                if result["success"]:
                    break

                reflection = agent.reflect(
                    initial_obs=result.get("initial_obs", ""),
                    trajectory=result.get("trajectory", []),
                )
                agent.add_reflection(reflection)

            if game_final:
                game_final["game_idx"] = game_idx
                game_results.append(game_final)
                if on_game_done:
                    on_game_done(game_final)
                status = "✓" if game_final["success"] else "✗"
                trials = game_final.get("trial", 1)
                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"game {game_idx+1}/{num_games} {status} "
                        f"(trial {trials}, {game_final['steps']} steps)"
                    ),
                )

    successes = sum(1 for r in game_results if r["success"])
    total = len(game_results)
    success_rate = successes / total if total else 0.0
    avg_steps = sum(r["steps"] for r in game_results) / total if total else 0.0
    total_loops = sum(r.get("loops_broken", 0) for r in game_results)
    avg_trials = sum(r.get("trial", 1) for r in game_results) / total if total else 0.0

    by_type: Dict[str, Dict] = {}
    for r in game_results:
        tt = r.get("task_type", "unknown")
        if tt not in by_type:
            by_type[tt] = {"successes": 0, "total": 0}
        by_type[tt]["total"] += 1
        if r["success"]:
            by_type[tt]["successes"] += 1
    by_type_stats = {
        tt: {
            "successes": v["successes"],
            "total": v["total"],
            "success_rate": round(v["successes"] / v["total"], 3) if v["total"] else 0.0,
        }
        for tt, v in sorted(by_type.items())
    }

    return {
        "agent": agent_name,
        "success_rate": success_rate,
        "successes": successes,
        "total_games": total,
        "avg_steps": round(avg_steps, 1),
        "avg_trials": round(avg_trials, 2),
        "loops_broken": total_loops,
        "errors": 0,
        "by_task_type": by_type_stats,
        "games": game_results,
    }


# ── Agent evaluator ────────────────────────────────────────────────────────────

def evaluate_agent(
    config: dict,
    agent,
    agent_name: str,
    num_games: int,
    max_steps: int,
    split: str,
    start_game: int = 0,
    existing_games: List[Dict] = None,
    on_game_done: Any = None,
) -> Dict[str, Any]:
    """Evaluate one agent across num_games episodes, with optional resume support."""

    env_class = environment.get_environment(config["env"]["type"])
    env = _make_env(env_class, config, split)

    # Skip already-evaluated games so iterator lands on the right position
    for _ in range(start_game):
        try:
            env.reset()
        except StopIteration:
            break

    game_results: List[Dict] = list(existing_games or [])
    errors = 0

    if start_game > 0:
        console.print(f"  [dim]Resuming from game {start_game + 1}/{num_games}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]{agent_name}[/cyan]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("", total=num_games)
        if start_game > 0:
            progress.update(task, advance=start_game,
                            description=f"(resumed, {start_game} games loaded)")

        for i in range(start_game, num_games):
            result = run_episode(env, agent, max_steps)
            if result is None:
                console.print(f"  [yellow]No more games available after {i} episodes.[/yellow]")
                break

            result["game_idx"] = i
            game_results.append(result)
            if on_game_done:
                on_game_done(result)
            status = "✓" if result["success"] else "✗"
            progress.update(
                task,
                advance=1,
                description=f"game {i+1}/{num_games} {status} ({result['steps']} steps)",
            )

    successes = sum(1 for r in game_results if r["success"])
    total = len(game_results)
    success_rate = successes / total if total else 0.0
    avg_steps = sum(r["steps"] for r in game_results) / total if total else 0.0
    total_loops = sum(r.get("loops_broken", 0) for r in game_results)

    # Per-task-type breakdown
    by_type: Dict[str, Dict] = {}
    for r in game_results:
        tt = r.get("task_type", "unknown")
        if tt not in by_type:
            by_type[tt] = {"successes": 0, "total": 0}
        by_type[tt]["total"] += 1
        if r["success"]:
            by_type[tt]["successes"] += 1
    by_type_stats = {
        tt: {
            "successes": v["successes"],
            "total": v["total"],
            "success_rate": round(v["successes"] / v["total"], 3) if v["total"] else 0.0,
        }
        for tt, v in sorted(by_type.items())
    }

    return {
        "agent": agent_name,
        "success_rate": success_rate,
        "successes": successes,
        "total_games": total,
        "avg_steps": round(avg_steps, 1),
        "loops_broken": total_loops,
        "errors": errors,
        "by_task_type": by_type_stats,
        "games": game_results,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ALFWorld Baseline Evaluation")
    parser.add_argument("--config",      default="configs/alfworld_config.yaml")
    parser.add_argument("--ollama-url",  default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--model",       default=os.getenv("MODEL_NAME", "llama3.1:8b"))
    parser.add_argument("--num-games",   type=int, default=int(os.getenv("NUM_GAMES", "10")))
    parser.add_argument("--max-steps",   type=int, default=int(os.getenv("MAX_STEPS", "50")))
    parser.add_argument("--agents",      nargs="+", default=["zero_shot", "few_shot", "cot", "react", "reflexion", "hierarchical"])
    parser.add_argument("--output",      default="results/baseline_results.json")
    parser.add_argument("--checkpoint",  default=None,
                        help="Path for incremental checkpoint (enables resume on restart)")
    parser.add_argument("--split",       default="eval_out_of_distribution",
                        choices=["train", "eval_in_distribution", "eval_out_of_distribution"])
    return parser.parse_args()


def main():
    args = parse_args()

    console.print("\n[bold blue]══════════════════════════════════════[/bold blue]")
    console.print("[bold blue]  ALFWorld Baseline Evaluation  [/bold blue]")
    console.print("[bold blue]══════════════════════════════════════[/bold blue]")
    console.print(f"  Model      : [green]{args.model}[/green]")
    console.print(f"  Ollama URL : {args.ollama_url}")
    console.print(f"  Split      : {args.split}")
    console.print(f"  Games/agent: {args.num_games}")
    console.print(f"  Max steps  : {args.max_steps}")
    console.print(f"  Agents     : {', '.join(args.agents)}\n")

    data_path = os.getenv("ALFWORLD_DATA", "/data/alfworld")
    config = load_config(args.config, data_path)

    # ── Pre-flight: verify data paths and game counts ──────────────────────────
    console.print("  Resolved data paths:")
    _path_labels = [
        ("train",     "data_path"),
        ("eval_seen", "eval_id_data_path"),
        ("eval_ood",  "eval_ood_data_path"),
    ]
    for label, key in _path_labels:
        p = Path(config["dataset"].get(key, ""))
        if p.exists():
            n = len(list(p.rglob("*.tw-pddl")))
            console.print(f"    {label:<12}: {p}  [green]✓ {n} games[/green]")
        else:
            console.print(f"    {label:<12}: {p}  [red]✗ NOT FOUND[/red]")
    console.print()

    agent_registry = {
        "zero_shot":    ZeroShotAgent(args.model, args.ollama_url),
        "few_shot":     FewShotAgent(args.model, args.ollama_url),
        "cot":          CotAgent(args.model, args.ollama_url),
        "react":        ReActAgent(args.model, args.ollama_url),
        "reflexion":    ReflexionAgent(args.model, args.ollama_url),
        "hierarchical": HierarchicalAgent(args.model, args.ollama_url),
    }

    # ── Load or initialise checkpoint ─────────────────────────────────────────
    ckpt = _load_checkpoint(args.checkpoint) if args.checkpoint else {}
    all_results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model":     ckpt.get("model",     args.model),
        "split":     ckpt.get("split",     args.split),
        "num_games": ckpt.get("num_games", args.num_games),
        "max_steps": ckpt.get("max_steps", args.max_steps),
        "agents":    ckpt.get("agents",    {}),
    }
    if ckpt:
        done = [a for a, v in all_results["agents"].items()
                if v.get("status") == "completed"]
        console.print(f"  [dim]Checkpoint loaded — completed: {done or 'none'}[/dim]")

    for agent_name in args.agents:
        if agent_name not in agent_registry:
            console.print(f"[yellow]Unknown agent '{agent_name}', skipping.[/yellow]")
            continue

        # Skip fully completed agents
        existing = all_results["agents"].get(agent_name, {})
        if existing.get("status") == "completed":
            console.print(f"\n[dim]── {agent_name}: already complete (skipping)[/dim]")
            continue

        existing_games = existing.get("games", [])
        start_game     = len(existing_games)

        console.print(f"\n[bold]── {agent_name} agent ──────────────────────────[/bold]")

        # Callback: save partial results to checkpoint after every game
        def _make_callback(name: str) -> Any:
            def _cb(game_result: Dict) -> None:
                if args.checkpoint:
                    all_results["agents"][name]["games"] = (
                        all_results["agents"].get(name, {}).get("games", [])
                        + [game_result]
                    )
                    _save_checkpoint(args.checkpoint, all_results)
            return _cb

        # Seed the in-progress entry so partial saves have the right structure
        all_results["agents"][agent_name] = {
            "status": "in_progress",
            "games":  list(existing_games),
        }

        try:
            evaluator = (
                evaluate_reflexion_agent if agent_name == "reflexion"
                else evaluate_agent
            )
            result = evaluator(
                config,
                agent_registry[agent_name],
                agent_name,
                args.num_games,
                args.max_steps,
                args.split,
                start_game=start_game,
                existing_games=existing_games,
                on_game_done=_make_callback(agent_name),
            )
            result["status"] = "completed"
            all_results["agents"][agent_name] = result
            if args.checkpoint:
                _save_checkpoint(args.checkpoint, all_results)
            console.print(
                f"  Done: [green]{result['successes']}/{result['total_games']}[/green] "
                f"succeeded ({result['success_rate']:.1%}), "
                f"avg {result['avg_steps']} steps"
            )
        except Exception:
            console.print(f"  [red]Agent {agent_name} failed:[/red]")
            traceback.print_exc()

    # ── Save final results ────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────────
    table = Table(title="\nBaseline Results Summary", show_header=True, header_style="bold magenta")
    table.add_column("Agent",        style="cyan",  min_width=12)
    table.add_column("Success Rate", style="green", justify="right")
    table.add_column("Successes",    justify="right")
    table.add_column("Avg Steps",    justify="right")
    table.add_column("Avg Trials",   justify="right")
    table.add_column("Loops Broken", justify="right")

    for agent_name, res in all_results["agents"].items():
        table.add_row(
            agent_name,
            f"{res['success_rate']:.1%}",
            f"{res['successes']}/{res['total_games']}",
            str(res["avg_steps"]),
            str(res.get("avg_trials", "—")),
            str(res.get("loops_broken", 0)),
        )

    console.print(table)

    # ── Per-task-type breakdown ────────────────────────────────────────────────
    all_types: set = set()
    for res in all_results["agents"].values():
        all_types.update(res.get("by_task_type", {}).keys())

    if all_types:
        tt_table = Table(title="Per-Task-Type Success", show_header=True, header_style="bold blue")
        tt_table.add_column("Task Type", style="cyan", min_width=30)
        for agent_name in all_results["agents"]:
            tt_table.add_column(agent_name, justify="right")

        for tt in sorted(all_types):
            row = [tt]
            for agent_name, res in all_results["agents"].items():
                stats = res.get("by_task_type", {}).get(tt)
                if stats:
                    row.append(f"{stats['successes']}/{stats['total']}")
                else:
                    row.append("—")
            tt_table.add_row(*row)

        console.print(tt_table)

    console.print(f"\n[dim]Results saved to: {out_path}[/dim]")


if __name__ == "__main__":
    main()
