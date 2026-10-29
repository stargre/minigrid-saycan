"""End-to-end acceptance tests for three official MiniGrid task mechanisms.

Run from the repository root:
    python scripts/25_official_basic_env_acceptance.py --scorer keyword

The planner receives only observations and a step-only environment proxy.
Ground truth is inspected only after the planner has stopped.
"""
import argparse
import inspect
import json
import time
from pathlib import Path

import gymnasium as gym
from minigrid.core.world_object import Door, Goal

from llm.skill_relevance import KeywordRelevanceScorer, QwenSkillRelevanceScorer
from planners.unified_saycan_planner import UnifiedSayCanPlanner
from skills.observation_runtime import ObservationRuntime


CASES = {
    "MiniGrid-Empty-5x5-v0": "Get to the green goal square.",
    "MiniGrid-GoToDoor-5x5-v0": None,  # Use the generated mission from the observation.
    "MiniGrid-Unlock-v0": "Find the key, unlock the door, and open the door.",
}


class StepOnlyEnv:
    """Expose only step(action) to decision code and reject hidden-state reads."""

    def __init__(self, env):
        object.__setattr__(self, "_StepOnlyEnv__env", env)
        object.__setattr__(self, "step_calls", 0)

    def step(self, action):
        caller = inspect.currentframe().f_back
        if (
            caller is None
            or caller.f_code.co_name != "execute_action"
            or caller.f_globals.get("__name__") != "skills.observation_runtime"
        ):
            raise AssertionError("Primitive action bypassed ObservationRuntime.execute_action.")
        object.__setattr__(self, "step_calls", self.step_calls + 1)
        env = object.__getattribute__(self, "_StepOnlyEnv__env")
        return env.step(action)

    def __getattr__(self, name):
        raise AssertionError(f"Decision code attempted hidden-state access: env.{name}")


def evaluate_after_run(env, env_id, runtime):
    """Evaluation-only oracle; never pass this information back to the planner."""
    base = env.unwrapped
    subgoals = set(runtime.belief.completed_subgoals)
    common = {
        "reward": float(runtime.last_reward),
        "terminated": bool(runtime.terminated),
        "truncated": bool(runtime.truncated),
        "completed_subgoals": sorted(subgoals),
    }

    if "Empty" in env_id:
        goals = [
            (x, y)
            for x in range(base.width)
            for y in range(base.height)
            if isinstance(base.grid.get(x, y), Goal)
        ]
        success = tuple(base.agent_pos) in goals
        return {**common, "oracle_success": success, "agent_on_goal": success}

    if "GoToDoor" in env_id:
        target_pos = tuple(base.target_pos)
        agent_pos = tuple(base.agent_pos)
        adjacent = bool(
            abs(agent_pos[0] - target_pos[0])
            + abs(agent_pos[1] - target_pos[1]) == 1
        )
        success = bool(adjacent and runtime.terminated and runtime.last_reward > 0)
        return {
            **common,
            "oracle_success": success,
            "agent_adjacent_to_target_door": adjacent,
            "target_door_color": str(base.target_color),
            "target_door_position": list(target_pos),
        }

    if "Unlock" in env_id:
        door = base.door
        door_open = bool(door.is_open and not door.is_locked)
        success = bool(door_open and runtime.terminated and runtime.last_reward > 0)
        return {
            **common,
            "oracle_success": success,
            "door_open": door_open,
            "door_locked": bool(door.is_locked),
        }

    raise ValueError(f"No evaluation predicate defined for {env_id}")


def run_case(env_id, seed, scorer_name, max_actions=500, max_high_level_steps=160):
    env = gym.make(env_id, render_mode="rgb_array")
    started = time.perf_counter()
    try:
        obs, _reset_info = env.reset(seed=seed)
        task = CASES[env_id] or str(obs.get("mission", ""))
        guarded = StepOnlyEnv(env)
        runtime = ObservationRuntime(guarded)
        runtime.reset(obs, initial_inventory="empty")

        scorer = QwenSkillRelevanceScorer() if scorer_name == "qwen" else KeywordRelevanceScorer()
        planner = UnifiedSayCanPlanner(
            runtime,
            relevance_scorer=scorer,
            max_actions=max_actions,
            max_skill_actions=300,
        )
        result = planner.run(task, max_high_level_steps=max_high_level_steps)
        elapsed = time.perf_counter() - started

        # Ground-truth inspection occurs only after planning has stopped.
        oracle = evaluate_after_run(env, env_id, runtime)
        passed = bool(result.get("success") and oracle["oracle_success"])
        if guarded.step_calls != runtime.action_count:
            passed = False

        return {
            "env_id": env_id,
            "seed": seed,
            "task": task,
            "scorer": scorer_name,
            "passed": passed,
            "planner_success": bool(result.get("success")),
            "planner_status": result.get("status"),
            "high_level_steps": result.get("high_level_steps"),
            "low_level_actions": result.get("low_level_actions"),
            "wall_time_s": round(elapsed, 4),
            "guarded_step_calls": guarded.step_calls,
            "runtime_action_count": runtime.action_count,
            "oracle": oracle,
            "skill_trace": [
                {
                    "round": record.get("round"),
                    "skill": record.get("call", {}).get("name"),
                    "args": record.get("call", {}).get("args", {}),
                    "status": record.get("result", {}).get("status"),
                    "success": record.get("result", {}).get("success"),
                    "actions": record.get("result", {}).get("low_level_actions"),
                }
                for record in result.get("history", [])
            ],
        }
    except Exception as exc:
        return {
            "env_id": env_id,
            "seed": seed,
            "scorer": scorer_name,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-ids", nargs="+", choices=sorted(CASES), default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--scorer", choices=["keyword", "qwen"], default="keyword")
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--max-high-level-steps", type=int, default=160)
    parser.add_argument("--output", default="outputs/official_basic_envs/results.json")
    args = parser.parse_args()

    records = []
    for env_id in args.env_ids:
        for seed in args.seeds:
            record = run_case(
                env_id, seed, args.scorer,
                max_actions=args.max_actions,
                max_high_level_steps=args.max_high_level_steps,
            )
            records.append(record)
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = [record for record in records if not record.get("passed")]
    print(f"Official basic environment acceptance: {len(records)-len(failures)}/{len(records)} passed.")
    print(f"Results: {output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
