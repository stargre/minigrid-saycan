#!/usr/bin/env python3
"""Unified acceptance suite for the MiniGrid SayCan reproduction.

The planner receives the official mission string and a step-only environment
proxy. Simulator ground truth is read only after planning has stopped. This
keeps decision-time perception honest while still providing a strict oracle.

Examples:
    python scripts/30_minigrid_saycan_acceptance.py --profile smoke
    python scripts/30_minigrid_saycan_acceptance.py --profile standard
    python scripts/30_minigrid_saycan_acceptance.py --scorer qwen --profile smoke
"""
from __future__ import annotations

import argparse
import inspect
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401 - registers official environments
from minigrid.core.world_object import Door, Goal

from llm.skill_relevance import KeywordRelevanceScorer, QwenSkillRelevanceScorer
from planners.unified_saycan_planner import UnifiedSayCanPlanner
from skills.observation_runtime import ObservationRuntime


CASES = {
    "empty": "MiniGrid-Empty-5x5-v0",
    "goto_door": "MiniGrid-GoToDoor-5x5-v0",
    "goto_object": "MiniGrid-GoToObject-6x6-N2-v0",
    "fetch": "MiniGrid-Fetch-5x5-N2-v0",
    "put_near": "MiniGrid-PutNear-6x6-N2-v0",
    "unlock": "MiniGrid-Unlock-v0",
    "door_key": "MiniGrid-DoorKey-5x5-v0",
}

PROFILES = {
    "smoke": (0,),
    "standard": tuple(range(5)),
    "extended": tuple(range(20)),
}


class StepOnlyEnv:
    """Expose only step(action) and enforce the primitive-action gateway."""

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
            f"Decision code attempted hidden-state access: env.{name}; "
            "only step(action) is available."
        )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except (TypeError, ValueError):
            pass
    return str(value)


def _grid_positions(base, object_type):
    positions = []
    for x in range(base.width):
        for y in range(base.height):
            obj = base.grid.get(x, y)
            if isinstance(obj, object_type):
                positions.append(((int(x), int(y)), obj))
    return positions


def evaluate_after_run(env, family, runtime):
    """Strict evaluation-only oracle; never feed this state to the planner."""
    base = env.unwrapped
    terminal_success = bool(
        runtime.terminated
        and not runtime.truncated
        and float(runtime.last_reward) > 0.0
    )
    common = {
        "terminal_success_reward": terminal_success,
        "reward": float(runtime.last_reward),
        "terminated": bool(runtime.terminated),
        "truncated": bool(runtime.truncated),
        "completed_subgoals": sorted(runtime.belief.completed_subgoals),
    }

    if family == "empty":
        goal_positions = [pos for pos, _ in _grid_positions(base, Goal)]
        agent_on_goal = tuple(int(v) for v in base.agent_pos) in goal_positions
        return {
            **common,
            "oracle_success": bool(terminal_success and agent_on_goal),
            "agent_on_goal": agent_on_goal,
        }

    if family in {"goto_door", "goto_object"}:
        target_pos = tuple(int(v) for v in base.target_pos)
        agent_pos = tuple(int(v) for v in base.agent_pos)
        adjacent = abs(agent_pos[0] - target_pos[0]) + abs(
            agent_pos[1] - target_pos[1]
        ) == 1
        return {
            **common,
            "oracle_success": bool(terminal_success and adjacent),
            "agent_adjacent_to_target": adjacent,
            "target_position": list(target_pos),
        }

    if family == "fetch":
        carrying = getattr(base, "carrying", None)
        correct_object = bool(
            carrying is not None
            and carrying.type == base.targetType
            and carrying.color == base.targetColor
        )
        return {
            **common,
            "oracle_success": bool(terminal_success and correct_object),
            "correct_object_held": correct_object,
            "target": {"object": base.targetType, "color": base.targetColor},
        }

    if family == "put_near":
        moved, targets = [], []
        for x in range(base.width):
            for y in range(base.height):
                obj = base.grid.get(x, y)
                if obj is None:
                    continue
                if obj.type == base.move_type and obj.color == base.moveColor:
                    moved.append((int(x), int(y)))
                if obj.type == base.target_type and obj.color == base.target_color:
                    targets.append((int(x), int(y)))
        moved_pos = moved[0] if len(moved) == 1 else None
        target_pos = targets[0] if len(targets) == 1 else None
        near = bool(
            moved_pos is not None
            and target_pos is not None
            and moved_pos != target_pos
            and abs(moved_pos[0] - target_pos[0]) <= 1
            and abs(moved_pos[1] - target_pos[1]) <= 1
        )
        released = base.carrying is None
        return {
            **common,
            "oracle_success": bool(terminal_success and near and released),
            "moved_position": list(moved_pos) if moved_pos else None,
            "target_position": list(target_pos) if target_pos else None,
            "near_target": near,
            "gripper_empty": released,
        }

    doors = [obj for _, obj in _grid_positions(base, Door)]
    door_open = any(door.is_open and not door.is_locked for door in doors)
    if family == "unlock":
        return {
            **common,
            "oracle_success": bool(terminal_success and door_open),
            "door_open": door_open,
        }

    if family == "door_key":
        goal_positions = [pos for pos, _ in _grid_positions(base, Goal)]
        agent_on_goal = tuple(int(v) for v in base.agent_pos) in goal_positions
        subgoals = set(runtime.belief.completed_subgoals)
        transition_evidence = {"unlock_door", "open_door"}.issubset(subgoals)
        return {
            **common,
            "oracle_success": bool(
                terminal_success and door_open and agent_on_goal
                and transition_evidence
            ),
            "door_open": door_open,
            "agent_on_goal": agent_on_goal,
            "unlock_and_open_observed": transition_evidence,
        }

    raise ValueError(f"Unknown acceptance family: {family}")


def run_case(family, env_id, seed, scorer_name, max_actions,
             max_high_level_steps, affordance_model=None):
    started = time.perf_counter()
    env = None
    try:
        env = gym.make(env_id)
        obs, _ = env.reset(seed=seed)
        mission = str(obs.get("mission", "")).strip()
        if not mission:
            raise RuntimeError("Official observation did not contain a mission.")

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
            max_skill_actions=max_actions,
            affordance_model=affordance_model,
        )
        result = planner.run(
            mission, max_high_level_steps=max_high_level_steps
        )

        # Ground truth is deliberately inspected only after planner.run().
        oracle = evaluate_after_run(env, family, runtime)
        history = result.get("history", [])
        fallback_records = [
            record for record in history
            if record.get("relevance_error") is not None
        ]
        gateway_consistent = guarded.step_calls == runtime.action_count
        qwen_valid = scorer_name != "qwen" or (
            not fallback_records and getattr(scorer, "request_count", 0) > 0
        )
        passed = bool(
            result.get("success")
            and oracle["oracle_success"]
            and gateway_consistent
            and qwen_valid
        )
        return _json_safe({
            "family": family,
            "env_id": env_id,
            "seed": int(seed),
            "mission": mission,
            "scorer": scorer_name,
            "experiment_kind": (
                "llm_saycan" if scorer_name == "qwen" else "offline_baseline"
            ),
            "affordance_mode": planner.affordance_mode,
            "affordance_model": str(affordance_model) if affordance_model else None,
            "passed": passed,
            "planner_success": bool(result.get("success")),
            "planner_status": result.get("status"),
            "oracle": oracle,
            "gateway_consistent": gateway_consistent,
            "runtime_action_count": runtime.action_count,
            "guarded_step_calls": guarded.step_calls,
            "high_level_steps": result.get("high_level_steps"),
            "low_level_actions": result.get("low_level_actions"),
            "wall_time_s": round(time.perf_counter() - started, 6),
            "qwen_model": getattr(scorer, "model", None),
            "qwen_api_requests": getattr(scorer, "request_count", 0),
            "qwen_api_error_fallbacks": len(fallback_records),
            "qwen_parse_retries": getattr(scorer, "parse_retry_count", 0),
            "skill_sequence": [
                {
                    "round": record.get("round"),
                    "skill": record.get("call", {}).get("name"),
                    "args": record.get("call", {}).get("args", {}),
                    "status": record.get("result", {}).get("status"),
                    "relevance": record.get("relevance"),
                    "feasibility": record.get("feasibility"),
                    "hard_feasibility": record.get("hard_feasibility"),
                    "learned_affordance": record.get("learned_affordance"),
                    "affordance_source": record.get("affordance_source"),
                    "combined_score": record.get("score"),
                    "selection_policy": record.get("selection_policy"),
                    "relevance_source": record.get("relevance_source"),
                    "relevance_error": record.get("relevance_error"),
                }
                for record in history
            ],
        })
    except Exception as exc:
        return {
            "family": family,
            "env_id": env_id,
            "seed": int(seed),
            "scorer": scorer_name,
            "passed": False,
            "wall_time_s": round(time.perf_counter() - started, 6),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if env is not None:
            env.close()


def build_summary(records, scorer, profile, affordance_model=None):
    by_family = {}
    for family in CASES:
        family_records = [r for r in records if r["family"] == family]
        if not family_records:
            continue
        passed = sum(bool(r.get("passed")) for r in family_records)
        by_family[family] = {
            "cases": len(family_records),
            "passed": passed,
            "failed": len(family_records) - passed,
            "success_rate": passed / len(family_records),
        }
    passed = sum(bool(record.get("passed")) for record in records)
    action_counts = [
        int(record["low_level_actions"])
        for record in records
        if record.get("low_level_actions") is not None
    ]
    return {
        "suite": "minigrid_saycan_acceptance_v1",
        "runtime": {
            "python": platform.python_version(),
            "minigrid": importlib.metadata.version("minigrid"),
            "gymnasium": importlib.metadata.version("gymnasium"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "scorer": scorer,
        "experiment_kind": (
            "llm_saycan" if scorer == "qwen" else "offline_baseline"
        ),
        "profile": profile,
        "affordance_mode": (
            "learned_hard_mask" if affordance_model else "deterministic"
        ),
        "affordance_model": str(affordance_model) if affordance_model else None,
        "cases": len(records),
        "passed": passed,
        "failed": len(records) - passed,
        "success_rate": passed / len(records) if records else 0.0,
        "mean_low_level_actions": (
            statistics.mean(action_counts) if action_counts else None
        ),
        "families": by_family,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Strict end-to-end acceptance for the MiniGrid SayCan reproduction."
    )
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument(
        "--families", nargs="+", choices=sorted(CASES),
        default=list(CASES),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        help="Override the seed set defined by --profile.",
    )
    parser.add_argument("--scorer", choices=("keyword", "qwen"), default="keyword")
    parser.add_argument("--max-actions", type=int, default=1200)
    parser.add_argument("--max-high-level-steps", type=int, default=250)
    parser.add_argument(
        "--affordance-model",
        help="Trained JSON model; enables hard-mask x learned-affordance Can scores.",
    )
    parser.add_argument(
        "--output", default="outputs/minigrid_acceptance/report.json"
    )
    args = parser.parse_args()

    if args.scorer == "qwen" and not os.getenv("DASHSCOPE_API_KEY"):
        parser.error(
            "--scorer qwen requires DASHSCOPE_API_KEY; the value is never printed."
        )
    seeds = tuple(args.seeds) if args.seeds is not None else PROFILES[args.profile]

    records = []
    for family in args.families:
        for seed in seeds:
            record = run_case(
                family=family,
                env_id=CASES[family],
                seed=seed,
                scorer_name=args.scorer,
                max_actions=args.max_actions,
                max_high_level_steps=args.max_high_level_steps,
                affordance_model=args.affordance_model,
            )
            records.append(record)
            marker = "PASS" if record.get("passed") else "FAIL"
            print(
                f"[{marker}] {family} seed={seed} "
                f"status={record.get('planner_status', record.get('error'))} "
                f"actions={record.get('low_level_actions')}"
            )

    summary = build_summary(
        records, args.scorer, args.profile, args.affordance_model
    )
    report = {"summary": summary, "records": records}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output}")
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
