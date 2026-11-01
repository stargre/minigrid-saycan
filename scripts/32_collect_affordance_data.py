#!/usr/bin/env python3
"""Collect observation-grounded skill execution outcomes as JSONL."""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401

from affordance.data import AffordanceDatasetWriter
from llm.skill_relevance import KeywordRelevanceScorer
from planners.unified_saycan_planner import UnifiedSayCanPlanner
from skills.observation_runtime import ObservationRuntime


DEFAULT_ENVS = (
    "MiniGrid-Empty-5x5-v0", "MiniGrid-GoToDoor-5x5-v0",
    "MiniGrid-GoToObject-6x6-N2-v0", "MiniGrid-Fetch-5x5-N2-v0",
    "MiniGrid-PutNear-6x6-N2-v0", "MiniGrid-Unlock-v0",
    "MiniGrid-DoorKey-5x5-v0", "MiniGrid-LockedRoom-v0",
    "MiniGrid-RedBlueDoors-6x6-v0", "MiniGrid-ObstructedMaze-1Dl-v0",
    "MiniGrid-ObstructedMaze-2Dl-v0", "MiniGrid-Dynamic-Obstacles-6x6-v0",
    "MiniGrid-FourRooms-v0", "MiniGrid-MultiRoom-N4-S5-v1",
    "MiniGrid-KeyCorridorS4R3-v0", "MiniGrid-SimpleCrossingS9N2-v0",
)


class StepOnlyEnv:
    def __init__(self, env):
        object.__setattr__(self, "_env", env)

    def step(self, action):
        caller = inspect.currentframe().f_back
        if caller is None or caller.f_code.co_name != "execute_action":
            raise AssertionError("Primitive action bypassed ObservationRuntime.")
        return object.__getattribute__(self, "_env").step(action)

    def __getattr__(self, name):
        raise AssertionError(f"Decision-time hidden-state access: env.{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-ids", nargs="+", default=list(DEFAULT_ENVS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--output", default="outputs/affordance/skill_outcomes.jsonl")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--max-actions", type=int, default=1200)
    parser.add_argument("--max-high-level-steps", type=int, default=300)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.append:
        output.unlink()
    records = []
    for env_id in args.env_ids:
        for seed in args.seeds:
            env = gym.make(env_id)
            try:
                obs, _ = env.reset(seed=seed)
                mission = str(obs.get("mission", "")).strip()
                runtime = ObservationRuntime(StepOnlyEnv(env))
                runtime.reset(obs, initial_inventory="empty")
                writer = AffordanceDatasetWriter(
                    output, context={"env_id": env_id, "seed": int(seed)}
                )
                budget = min(args.max_actions, int(getattr(env.unwrapped, "max_steps", args.max_actions)))
                planner = UnifiedSayCanPlanner(
                    runtime,
                    relevance_scorer=KeywordRelevanceScorer(),
                    max_actions=budget,
                    max_skill_actions=budget,
                    transition_recorder=writer,
                )
                result = planner.run(
                    mission, max_high_level_steps=args.max_high_level_steps
                )
                record = {
                    "env_id": env_id, "seed": seed,
                    "success": bool(result.get("success")),
                    "status": result.get("status"),
                    "samples": writer.rows_written,
                    "actions": result.get("low_level_actions"),
                }
            except Exception as exc:
                record = {
                    "env_id": env_id, "seed": seed, "success": False,
                    "error": f"{type(exc).__name__}: {exc}", "samples": 0,
                }
            finally:
                env.close()
            records.append(record)
            marker = "PASS" if record.get("success") else "FAIL"
            print(f"[{marker}] {env_id} seed={seed} samples={record.get('samples')}")
    summary = {
        "episodes": len(records),
        "successful_episodes": sum(bool(item.get("success")) for item in records),
        "samples": sum(int(item.get("samples", 0)) for item in records),
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps({"summary": summary, "episodes": records}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
