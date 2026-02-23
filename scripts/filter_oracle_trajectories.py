#!/usr/bin/env python3
"""
Filter Oracle Trajectories for SFT
=====================================
Reads a run_baseline.py results JSON, filters to successful trajectories,
and writes them as JSONL for LoRA SFT training.

Each output record (one JSONL line) has the format:
{
  "agent":      str,               # agent name
  "task_type":  str,               # ALFWorld task type
  "game_idx":   int,               # game index in the split
  "trial":      int,               # trial number (1-indexed, Reflexion only)
  "subgoals":   [str, ...],        # planned subgoals (HierarchicalAgent only)
  "initial_obs": str,              # initial environment observation (task goal)
  "trajectory": [[action, obs], ...]  # full (action, observation) sequence
}

Usage:
    python scripts/filter_oracle_trajectories.py \
        --input  results/e4_raw_results.json \
        --output results/oracle/trajectories.jsonl \
        --agents hierarchical react \
        --min-steps 3            # skip trivially short episodes
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Filter oracle trajectories for SFT")
    p.add_argument("--input",   required=True,  help="run_baseline results JSON")
    p.add_argument("--output",  required=True,  help="Output JSONL path")
    p.add_argument("--agents",  nargs="+",      default=None,
                   help="Only include these agents (default: all)")
    p.add_argument("--min-steps", type=int,     default=3,
                   help="Skip successful episodes with fewer steps (default: 3)")
    p.add_argument("--min-success-rate", type=float, default=0.0,
                   help="Warn if agent success rate is below this threshold")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.input) as f:
        data = json.load(f)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    agents_data = data.get("agents", {})
    target_agents = set(args.agents) if args.agents else set(agents_data.keys())

    total_written = 0
    stats = {}

    with open(output_path, "w") as out:
        for agent_name, agent_data in agents_data.items():
            if agent_name not in target_agents:
                continue

            sr = agent_data.get("success_rate", 0.0)
            if sr < args.min_success_rate:
                print(f"  WARNING: {agent_name} SR={sr:.1%} < threshold {args.min_success_rate:.1%}",
                      file=sys.stderr)

            games = agent_data.get("games", [])
            written = skipped_fail = skipped_short = skipped_no_traj = 0

            for game in games:
                if not game.get("success", False):
                    skipped_fail += 1
                    continue

                trajectory = game.get("trajectory", [])
                if not trajectory:
                    skipped_no_traj += 1
                    continue

                steps = game.get("steps", len(trajectory))
                if steps < args.min_steps:
                    skipped_short += 1
                    continue

                record = {
                    "agent":       agent_name,
                    "task_type":   game.get("task_type", "unknown"),
                    "game_idx":    game.get("game_idx", -1),
                    "trial":       game.get("trial", 1),
                    "subgoals":    game.get("subgoals", []),   # HierarchicalAgent field
                    "initial_obs": game.get("initial_obs", ""),
                    "trajectory":  trajectory,
                    "steps":       steps,
                    "score":       game.get("score", 1.0),
                }
                out.write(json.dumps(record) + "\n")
                written += 1

            stats[agent_name] = {
                "total_games": len(games),
                "success_rate": sr,
                "written":       written,
                "skipped_fail":  skipped_fail,
                "skipped_short": skipped_short,
                "skipped_no_traj": skipped_no_traj,
            }
            total_written += written

    # Summary
    print(f"\nOracle trajectory extraction complete")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Total records written: {total_written}\n")
    print(f"  {'Agent':<14} {'SR':>6}  {'Written':>8}  {'Skipped(fail)':>14}  {'Skipped(short)':>15}")
    print(f"  {'-'*65}")
    for name, s in stats.items():
        print(f"  {name:<14} {s['success_rate']:>6.1%}  {s['written']:>8}  "
              f"{s['skipped_fail']:>14}  {s['skipped_short']:>15}")

    if total_written == 0:
        print("\n  WARNING: 0 trajectories written — check success rates and --agents filter",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
