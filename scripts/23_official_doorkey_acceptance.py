"""Acceptance test for the official MiniGrid DoorKey environments.

The planner receives only observations and a step-only environment proxy.
Ground-truth success is checked after planning using the environment's final
agent position, door state, and terminal reward; none of these are passed back
to the planner or skills.

Run from the repository root:
    python scripts/23_official_doorkey_acceptance.py
"""
import argparse
import inspect
import json
from pathlib import Path

import gymnasium as gym
from minigrid.core.world_object import Door, Goal

from llm.skill_relevance import KeywordRelevanceScorer
from planners.unified_saycan_planner import UnifiedSayCanPlanner
from skills.observation_runtime import ObservationRuntime


TASK = (
    "Find the yellow key, unlock the yellow door, open the yellow door, "
    "and reach the green goal."
)
DEFAULT_ENV_IDS = (
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-DoorKey-6x6-v0",
)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


class StepOnlyEnv:
    """Expose only step(action), and reject all privileged environment reads."""

    def __init__(self, inner):
        object.__setattr__(self, "_StepOnlyEnv__inner", inner)
        object.__setattr__(self, "step_calls", 0)

    def step(self, action):
        caller = inspect.currentframe().f_back
        if (
            caller is None
            or caller.f_code.co_name != "execute_action"
            or caller.f_globals.get("__name__") != "skills.observation_runtime"
        ):
            raise AssertionError(
                "A primitive action bypassed ObservationRuntime.execute_action."
            )
        object.__setattr__(self, "step_calls", self.step_calls + 1)
        inner = object.__getattribute__(self, "_StepOnlyEnv__inner")
        return inner.step(action)

    def __getattr__(self, name):
        raise AssertionError(
            f"Decision code attempted hidden-state access: env.{name}; "
            "only step(action) is available."
        )


def inspect_final_state_for_evaluation_only(env, runtime):
    """Read ground truth only after the planner has stopped."""
    base = env.unwrapped
    goal_positions = []
    doors = []
    for x in range(base.width):
        for y in range(base.height):
            obj = base.grid.get(x, y)
            if isinstance(obj, Goal):
                goal_positions.append((x, y))
            elif isinstance(obj, Door):
                doors.append(obj)

    agent_on_goal = tuple(base.agent_pos) in goal_positions
    door_open = any(door.is_open and not door.is_locked for door in doors)
    terminal_success = bool(runtime.terminated and runtime.last_reward > 0)
    subgoals = set(runtime.belief.completed_subgoals)
    return {
        "agent_on_goal": agent_on_goal,
        "door_open": door_open,
        "terminal_success_reward": terminal_success,
        "unlock_observed": "unlock_door" in subgoals,
        "open_observed": "open_door" in subgoals,
        "completed_subgoals": sorted(subgoals),
        "reward": runtime.last_reward,
        "terminated": runtime.terminated,
        "truncated": runtime.truncated,
    }


def run_case(env_id, seed, max_actions=1200, max_high_level_steps=250):
    env = gym.make(env_id)
    try:
        obs, _reset_info = env.reset(seed=seed)
        guarded_env = StepOnlyEnv(env)
        runtime = ObservationRuntime(guarded_env)
        runtime.reset(obs, initial_inventory="empty")
        planner = UnifiedSayCanPlanner(
            runtime,
            relevance_scorer=KeywordRelevanceScorer(),
            max_actions=max_actions,
            max_skill_actions=500,
        )
        result = planner.run(TASK, max_high_level_steps=max_high_level_steps)

        # This is evaluation-only: it is deliberately called after planner.run.
        final_state = inspect_final_state_for_evaluation_only(env, runtime)
        oracle_success = bool(
            final_state["agent_on_goal"]
            and final_state["door_open"]
            and final_state["terminal_success_reward"]
        )
        required_skill_evidence = bool(
            final_state["unlock_observed"] and final_state["open_observed"]
        )
        passed = bool(result.get("success") and oracle_success and required_skill_evidence)
        record = {
            "env_id": env_id,
            "seed": seed,
            "passed": passed,
            "planner_success": bool(result.get("success")),
            "planner_status": result.get("status"),
            "high_level_steps": result.get("high_level_steps"),
            "low_level_actions": result.get("low_level_actions"),
            "guarded_step_calls": guarded_env.step_calls,
            "runtime_action_count": runtime.action_count,
            "oracle_success": oracle_success,
            **final_state,
        }
        if guarded_env.step_calls != runtime.action_count:
            record["passed"] = False
            record["error"] = (
                f"guarded step count {guarded_env.step_calls} != "
                f"runtime action count {runtime.action_count}"
            )
        return record
    except Exception as exc:
        return {
            "env_id": env_id,
            "seed": seed,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-ids", nargs="+", default=list(DEFAULT_ENV_IDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-actions", type=int, default=1200)
    parser.add_argument("--max-high-level-steps", type=int, default=250)
    parser.add_argument("--output", default="outputs/doorkey_eval/results.json")
    args = parser.parse_args()

    records = []
    for env_id in args.env_ids:
        for seed in args.seeds:
            record = run_case(
                env_id,
                seed,
                max_actions=args.max_actions,
                max_high_level_steps=args.max_high_level_steps,
            )
            records.append(record)
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    failures = [r for r in records if not r.get("passed")]
    print(
        f"DoorKey acceptance: {len(records) - len(failures)}/{len(records)} passed. "
        f"Results: {output}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
