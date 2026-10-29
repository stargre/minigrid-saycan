#!/usr/bin/env python3
"""End-to-end acceptance tests for official MiniGrid PutNear environments.

Run from the repository root:
    python scripts/26_official_putnear_acceptance.py --scorer keyword --env-ids MiniGrid-PutNear-6x6-N2-v0 --seeds 0
    python scripts/26_official_putnear_acceptance.py --scorer keyword --seeds 0 1 2 3 4
"""
import argparse
import inspect
import json
import os
import time
from pathlib import Path

import gymnasium as gym

from llm.skill_relevance import KeywordRelevanceScorer, QwenSkillRelevanceScorer
from planners.unified_saycan_planner import UnifiedSayCanPlanner
from skills.observation_runtime import ObservationRuntime


DEFAULT_ENV_IDS = (
    "MiniGrid-PutNear-6x6-N2-v0",
    "MiniGrid-PutNear-8x8-N3-v0",
)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


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
            raise AssertionError(
                "Primitive action bypassed ObservationRuntime.execute_action."
            )
        object.__setattr__(self, "step_calls", self.step_calls + 1)
        env = object.__getattribute__(self, "_StepOnlyEnv__env")
        return env.step(action)

    def __getattr__(self, name):
        raise AssertionError(
            f"Decision code attempted hidden-state access: env.{name}"
        )


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return json_safe(item_method())
        except (ValueError, TypeError):
            pass
    return str(value)


def evaluate_after_run(env, runtime):
    """Use ground truth only after planning has stopped."""
    base = env.unwrapped
    moved_positions, target_positions = [], []

    for x in range(base.width):
        for y in range(base.height):
            obj = base.grid.get(x, y)
            if obj is None:
                continue
            obj_type = getattr(obj, "type", None)
            obj_color = getattr(obj, "color", None)
            if obj_type == base.move_type and obj_color == base.moveColor:
                moved_positions.append((int(x), int(y)))
            if obj_type == base.target_type and obj_color == base.target_color:
                target_positions.append((int(x), int(y)))

    moved_pos = moved_positions[0] if len(moved_positions) == 1 else None
    target_pos = target_positions[0] if len(target_positions) == 1 else None
    near_target = bool(
        moved_pos is not None
        and target_pos is not None
        and moved_pos != target_pos
        and abs(moved_pos[0] - target_pos[0]) <= 1
        and abs(moved_pos[1] - target_pos[1]) <= 1
    )
    gripper_empty = base.carrying is None
    episode_success = bool(
        runtime.terminated
        and not runtime.truncated
        and float(runtime.last_reward) > 0.0
    )
    return {
        "oracle_success": bool(episode_success and near_target and gripper_empty),
        "episode_success_reward": episode_success,
        "move_object": {
            "type": str(base.move_type),
            "color": str(base.moveColor),
            "position": list(moved_pos) if moved_pos is not None else None,
            "matching_objects_in_grid": len(moved_positions),
        },
        "target_object": {
            "type": str(base.target_type),
            "color": str(base.target_color),
            "position": list(target_pos) if target_pos is not None else None,
            "matching_objects_in_grid": len(target_positions),
        },
        "near_target": near_target,
        "gripper_empty": gripper_empty,
        "reward": float(runtime.last_reward),
        "terminated": bool(runtime.terminated),
        "truncated": bool(runtime.truncated),
        "completed_subgoals": sorted(runtime.belief.completed_subgoals),
    }


def run_case(env_id, seed, scorer_name="keyword", max_actions=500,
             max_high_level_steps=160):
    started = time.perf_counter()
    env = None
    try:
        env = gym.make(env_id, render_mode="rgb_array")
        obs, _reset_info = env.reset(seed=seed)
        task = str(obs.get("mission", "")).strip()
        if not task:
            raise RuntimeError("The environment observation did not contain a mission.")

        guarded = StepOnlyEnv(env)
        runtime = ObservationRuntime(guarded)
        runtime.reset(obs, initial_inventory="empty")
        scorer = (
            QwenSkillRelevanceScorer()
            if scorer_name == "qwen"
            else KeywordRelevanceScorer()
        )
        planner = UnifiedSayCanPlanner(
            runtime,
            relevance_scorer=scorer,
            max_actions=max_actions,
            max_skill_actions=300,
        )
        result = planner.run(task, max_high_level_steps=max_high_level_steps)

        # Ground-truth evaluation is intentionally after planner.run().
        oracle = evaluate_after_run(env, runtime)
        history = result.get("history", [])
        api_error_fallbacks = sum(
            "API error:" in str(record.get("relevance_source", ""))
            for record in history
        )
        passed = bool(result.get("success") and oracle["oracle_success"])
        if guarded.step_calls != runtime.action_count:
            passed = False
        if scorer_name == "qwen" and api_error_fallbacks:
            passed = False

        return json_safe({
            "env_id": env_id,
            "seed": int(seed),
            "task": task,
            "scorer": scorer_name,
            "passed": passed,
            "planner_success": bool(result.get("success")),
            "planner_status": result.get("status"),
            "high_level_steps": result.get("high_level_steps"),
            "low_level_actions": result.get("low_level_actions"),
            "wall_time_s": round(time.perf_counter() - started, 4),
            "guarded_step_calls": guarded.step_calls,
            "runtime_action_count": runtime.action_count,
            "qwen_model": getattr(scorer, "model", None),
            "qwen_api_requests": getattr(scorer, "request_count", 0),
            "qwen_api_error_fallbacks": api_error_fallbacks,
            "oracle": oracle,
            "skill_trace": [
                {
                    "round": record.get("round"),
                    "skill": record.get("call", {}).get("name"),
                    "args": record.get("call", {}).get("args", {}),
                    "status": record.get("result", {}).get("status"),
                    "success": record.get("result", {}).get("success"),
                    "actions": record.get("result", {}).get("low_level_actions"),
                    "relevance": record.get("relevance"),
                    "feasibility": record.get("feasibility"),
                    "relevance_source": record.get("relevance_source"),
                    "relevance_error": record.get("relevance_error"),
                    "reason": record.get("result", {}).get("reason"),
                }
                for record in history
            ],
        })
    except Exception as exc:
        return {
            "env_id": env_id,
            "seed": int(seed),
            "scorer": scorer_name,
            "passed": False,
            "wall_time_s": round(time.perf_counter() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-ids", nargs="+", default=list(DEFAULT_ENV_IDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--scorer", choices=("keyword", "qwen"), default="keyword")
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--max-high-level-steps", type=int, default=160)
    parser.add_argument("--output", default="outputs/official_putnear/results.json")
    args = parser.parse_args()

    if args.scorer == "qwen" and not os.getenv("DASHSCOPE_API_KEY"):
        parser.error(
            "--scorer qwen requires DASHSCOPE_API_KEY; the key is never printed."
        )

    records = []
    for env_id in args.env_ids:
        for seed in args.seeds:
            record = run_case(
                env_id=env_id,
                seed=seed,
                scorer_name=args.scorer,
                max_actions=args.max_actions,
                max_high_level_steps=args.max_high_level_steps,
            )
            records.append(record)
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    failures = [record for record in records if not record.get("passed")]
    print(
        f"Official PutNear acceptance: {len(records) - len(failures)}/"
        f"{len(records)} passed."
    )
    print(f"Results: {output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
