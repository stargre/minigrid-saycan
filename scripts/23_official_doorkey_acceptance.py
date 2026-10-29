"""Acceptance test for official MiniGrid DoorKey with keyword or Qwen scoring.

The planner receives only observations and a step-only environment proxy.
Ground-truth success is checked only after planning stops. In Qwen mode, a case
is considered passed only if the physical task succeeds AND no API error caused
the planner to silently fall back to keyword scoring.

Examples:
    python scripts/23_official_doorkey_acceptance.py --scorer keyword
    python scripts/23_official_doorkey_acceptance.py --scorer qwen \\
        --env-ids MiniGrid-DoorKey-5x5-v0 --seeds 0 \\
        --output outputs/doorkey_eval/qwen_smoke_seed0.json
"""
import argparse
import inspect
import json
import os
import statistics
import time
from pathlib import Path

import gymnasium as gym
from minigrid.core.world_object import Door, Goal

from llm.skill_relevance import (
    KeywordRelevanceScorer,
    QwenSkillRelevanceScorer,
)
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


class TimedQwenSkillRelevanceScorer(QwenSkillRelevanceScorer):
    """Collect per-request latency, including requests that raise exceptions."""

    def __init__(self):
        super().__init__()
        self.api_latencies_s = []

    def score(self, task, calls, belief_summary=None):
        requests_before = self.request_count
        started = time.perf_counter()
        try:
            return super().score(task, calls, belief_summary)
        finally:
            # request_count increments only on a cache miss, immediately before
            # the remote call. This records latency for successful and failed
            # remote requests, but not local cache hits.
            if self.request_count > requests_before:
                self.api_latencies_s.append(time.perf_counter() - started)


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


def run_case(
    env_id,
    seed,
    scorer_name="keyword",
    max_actions=1200,
    max_high_level_steps=250,
):
    env = gym.make(env_id)
    started = time.perf_counter()
    try:
        obs, _reset_info = env.reset(seed=seed)
        guarded_env = StepOnlyEnv(env)
        runtime = ObservationRuntime(guarded_env)
        runtime.reset(obs, initial_inventory="empty")

        if scorer_name == "qwen":
            # A fresh scorer per episode prevents cache reuse across seeds from
            # making the Qwen test look faster or more capable than it is.
            scorer = TimedQwenSkillRelevanceScorer()
        else:
            scorer = KeywordRelevanceScorer()

        planner = UnifiedSayCanPlanner(
            runtime,
            relevance_scorer=scorer,
            max_actions=max_actions,
            max_skill_actions=500,
        )
        result = planner.run(TASK, max_high_level_steps=max_high_level_steps)
        wall_time_s = time.perf_counter() - started

        # Evaluation-only oracle: deliberately after planner.run().
        final_state = inspect_final_state_for_evaluation_only(env, runtime)
        oracle_success = bool(
            final_state["agent_on_goal"]
            and final_state["door_open"]
            and final_state["terminal_success_reward"]
        )
        required_skill_evidence = bool(
            final_state["unlock_observed"] and final_state["open_observed"]
        )
        task_passed = bool(
            result.get("success") and oracle_success and required_skill_evidence
        )
        history = result.get("history", [])
        api_error_records = [
            record for record in history
            if "API error:" in str(record.get("relevance_source", ""))
        ]
        api_error_fallbacks = len(api_error_records)
        api_error_details = [
            {
                "round": record.get("round"),
                "skill": record.get("call", {}).get("name"),
                "error": record.get("relevance_error") or "No exception detail recorded.",
            }
            for record in api_error_records
        ]

        # Do not count silent keyword fallbacks as successful Qwen evaluations.
        passed = task_passed and not (
            scorer_name == "qwen" and api_error_fallbacks > 0
        )
        record = {
            "env_id": env_id,
            "seed": seed,
            "scorer": scorer_name,
            "passed": passed,
            "task_passed": task_passed,
            "planner_success": bool(result.get("success")),
            "planner_status": result.get("status"),
            "high_level_steps": result.get("high_level_steps"),
            "low_level_actions": result.get("low_level_actions"),
            "wall_time_s": round(wall_time_s, 4),
            "guarded_step_calls": guarded_env.step_calls,
            "runtime_action_count": runtime.action_count,
            "oracle_success": oracle_success,
            "qwen_model": getattr(scorer, "model", None),
            "qwen_api_requests": getattr(scorer, "request_count", 0),
            "qwen_api_latency_samples_s": [
                round(value, 4)
                for value in getattr(scorer, "api_latencies_s", [])
            ],
            "qwen_avg_api_latency_s": (
                round(statistics.mean(scorer.api_latencies_s), 4)
                if getattr(scorer, "api_latencies_s", [])
                else None
            ),
            "qwen_last_api_latency_s": (
                round(scorer.api_latencies_s[-1], 4)
                if getattr(scorer, "api_latencies_s", [])
                else None
            ),
            "qwen_score_cache_entries": len(getattr(scorer, "_cache", {})),
            "qwen_parse_retries": getattr(scorer, "parse_retry_count", 0),
            "qwen_api_error_fallbacks": api_error_fallbacks,
            "qwen_api_error_details": api_error_details,
            "skill_sequence": " -> ".join(
                item.get("call", {}).get("name", "?") for item in history
            ),
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
            "scorer": scorer_name,
            "passed": False,
            "task_passed": False,
            "wall_time_s": round(time.perf_counter() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", choices=("keyword", "qwen"), default="keyword")
    parser.add_argument("--env-ids", nargs="+", default=list(DEFAULT_ENV_IDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-actions", type=int, default=1200)
    parser.add_argument("--max-high-level-steps", type=int, default=250)
    parser.add_argument("--output", default="outputs/doorkey_eval/results.json")
    args = parser.parse_args()

    if args.scorer == "qwen" and not os.getenv("DASHSCOPE_API_KEY"):
        parser.error(
            "--scorer qwen requires DASHSCOPE_API_KEY; the key value is never printed."
        )

    records = []
    for env_id in args.env_ids:
        for seed in args.seeds:
            record = run_case(
                env_id,
                seed,
                scorer_name=args.scorer,
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
    requests = sum(r.get("qwen_api_requests", 0) for r in records)
    latencies = [
        latency
        for record in records
        for latency in record.get("qwen_api_latency_samples_s", [])
    ]
    summary = {
        "scorer": args.scorer,
        "num_cases": len(records),
        "passed": len(records) - len(failures),
        "failed": len(failures),
        "task_successes": sum(bool(r.get("task_passed")) for r in records),
        "total_qwen_api_requests": requests,
        "mean_qwen_api_latency_s": (
            round(statistics.mean(latencies), 4) if latencies else None
        ),
        "median_qwen_api_latency_s": (
            round(statistics.median(latencies), 4) if latencies else None
        ),
        "total_wall_time_s": round(
            sum(r.get("wall_time_s", 0.0) for r in records), 4
        ),
        "results_path": str(output),
    }
    print("Summary:", json.dumps(summary, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
