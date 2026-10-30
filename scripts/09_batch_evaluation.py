"""Qwen/keyword batch evaluation for compositional MiniGrid tasks T1-T5.

Qwen mode is strict: it requires DASHSCOPE_API_KEY and reports any planner
fallbacks instead of silently treating them as Qwen-controlled decisions.
"""
import argparse
import csv
import inspect
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from minigrid.core.world_object import Ball

from environments.saycan_grid import SayCanGridEnv
from skills.observation_runtime import ObservationRuntime
from planners.unified_saycan_planner import UnifiedSayCanPlanner
from llm.skill_relevance import KeywordRelevanceScorer, QwenSkillRelevanceScorer


ROOT = Path(__file__).resolve().parents[1]

TASKS = [
    ("T1", "Pick up the red ball, open the yellow door, and place it above the green goal."),
    ("T2", "Open the yellow door first, then pick up the red ball, and finally place it above the green goal."),
    ("T3", "Take the red ball, open the yellow door, and drop it above the green goal."),
    ("T4", "Open the yellow door, then place the red ball above the green goal after picking it up."),
    ("T5", "Grab the red ball and put it above the green goal; open the yellow door if needed."),
]




class RecordingStepOnlyEnv:
    """Record frames while exposing only the guarded step(action) interface."""

    def __init__(self, env, frames):
        object.__setattr__(self, "_RecordingStepOnlyEnv__env", env)
        object.__setattr__(self, "_RecordingStepOnlyEnv__frames", frames)
        object.__setattr__(self, "step_calls", 0)

    def capture(self):
        env = object.__getattribute__(self, "_RecordingStepOnlyEnv__env")
        frames = object.__getattribute__(self, "_RecordingStepOnlyEnv__frames")
        frame = env.render()
        if frame is None:
            raise RuntimeError("MiniGrid returned no RGB frame.")
        frames.append(np.asarray(frame, dtype=np.uint8).copy())

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

        env = object.__getattribute__(self, "_RecordingStepOnlyEnv__env")
        result = env.step(int(action))
        object.__setattr__(self, "step_calls", self.step_calls + 1)
        self.capture()
        return result

    def __getattr__(self, name):
        raise AssertionError(
            f"Decision code attempted hidden-state access: env.{name}"
        )


def save_visuals(frames, output_dir, duration_ms=180):
    if not frames:
        raise RuntimeError("No frames were recorded.")
    output_dir.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    gif_path = output_dir / "trajectory.gif"
    png_path = output_dir / "final_frame.png"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(duration_ms),
        loop=0,
        optimize=False,
    )
    images[-1].save(png_path)
    return gif_path, png_path


def _is_conditional_door_instruction(task):
    return bool(re.search(
        r"\b(?:if|when)\s+(?:it\s+is\s+)?(?:needed|necessary)\b",
        str(task).lower(),
    ))


def ground_truth_success(env, task_id, task):
    """Evaluation-only oracle; never used by the planner or skill implementation."""
    base = env.unwrapped
    placed = base.grid.get(*base.target_pos)
    door = base.grid.get(*base.door_pos)

    conditional_door = _is_conditional_door_instruction(task)
    # In this fixed benchmark layout, the agent starts left of the central
    # wall while the green goal is always on the right. Therefore the yellow
    # door is physically required even when the instruction says "if needed";
    # the ball's random side does not change this topological requirement.
    door_instruction = "open" in task.lower() or "unlock" in task.lower()
    door_required = bool(door_instruction)
    door_open = bool(getattr(door, "is_open", False))
    ball_at_target = bool(isinstance(placed, Ball) and placed.color == "red")
    released = getattr(base, "carrying", None) is None
    door_requirement_satisfied = bool(not door_required or door_open)

    return {
        "success": bool(ball_at_target and released and door_requirement_satisfied),
        "ball_at_target": ball_at_target,
        "released": released,
        "door_open": door_open,
        "door_required": bool(door_required),
        "door_requirement_satisfied": door_requirement_satisfied,
        "ball_side": str(getattr(base, "ball_side", "unknown")),
        "unnecessary_door_open": bool(
            conditional_door and not door_required and door_open
        ),
    }


def _successful_skill_sequence(history):
    return [
        record.get("call", {}).get("name")
        for record in history
        if record.get("result", {}).get("success")
        and record.get("result", {}).get("status") not in {
            "action_executed", "searching", "phase_complete",
        }
    ]


def _first_index(sequence, skill):
    try:
        return sequence.index(skill)
    except ValueError:
        return None


def order_constraints(task_id, successful_sequence, door_required):
    """Check hard precedence constraints, not incidental navigation ordering."""
    indices = {name: _first_index(successful_sequence, name) for name in (
        "PickUp", "OpenDoor", "PlaceNear"
    )}
    pickup, open_door, place = (
        indices["PickUp"], indices["OpenDoor"], indices["PlaceNear"]
    )
    checks = {
        "pickup_before_place": (
            pickup is not None and place is not None and pickup < place
        ),
    }
    if task_id in {"T2", "T4"}:
        checks["open_before_pickup"] = (
            open_door is not None and pickup is not None and open_door < pickup
        )
        checks["pickup_before_place"] = (
            pickup is not None and place is not None and pickup < place
        )
    if door_required:
        checks["open_before_place"] = (
            open_door is not None and place is not None and open_door < place
        )
    return {
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
        "successful_skill_sequence": successful_sequence,
    }


def run_one(task_id, task, seed, scorer, output_dir):
    frames = []
    env = SayCanGridEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    recorder = RecordingStepOnlyEnv(env, frames)
    recorder.capture()
    runtime = ObservationRuntime(recorder)
    runtime.reset(obs, initial_inventory="empty")
    planner = UnifiedSayCanPlanner(
        runtime,
        relevance_scorer=scorer,
        max_actions=450,
        max_skill_actions=300,
    )

    api_requests_before = int(getattr(scorer, "request_count", 0))
    parse_retries_before = int(getattr(scorer, "parse_retry_count", 0))
    latency_count_before = len(getattr(scorer, "api_latencies_s", []))
    started = time.perf_counter()
    try:
        result = planner.run(task, max_high_level_steps=100)
        error = ""
    except Exception as exc:
        result = {
            "success": False,
            "status": "exception",
            "history": [],
            "high_level_steps": 0,
            "low_level_actions": runtime.action_count,
        }
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started
    planner_success = bool(result.get("success", False))
    oracle = ground_truth_success(env, task_id, task)
    history = result.get("history", [])
    successful_sequence = _successful_skill_sequence(history)
    case_dir = output_dir / f"{task_id}_seed_{seed}"
    gif_path, png_path = save_visuals(frames, case_dir)
    frame_action_match = len(frames) == runtime.action_count + 1
    gateway_action_match = recorder.step_calls == runtime.action_count
    recording_integrity = bool(frame_action_match and gateway_action_match)

    qwen_scored_rounds = sum(
        record.get("relevance_source") == "QwenSkillRelevanceScorer"
        for record in history
    )
    api_fallback_rounds = sum(
        str(record.get("relevance_source", "")).startswith("KeywordRelevanceScorer")
        for record in history
    )
    deterministic_recovery_rounds = sum(
        record.get("selection_policy") == "deterministic_recovery"
        or "relevance_source" not in record
        for record in history
    )
    zero_score_fallback_rounds = sum(
        bool(record.get("zero_score_fallback")) for record in history
    )
    qwen_api_requests = (
        int(getattr(scorer, "request_count", 0)) - api_requests_before
    )
    qwen_parse_retries = (
        int(getattr(scorer, "parse_retry_count", 0)) - parse_retries_before
    )
    latencies = list(getattr(scorer, "api_latencies_s", []))[latency_count_before:]
    qwen_scored_rounds = int(qwen_scored_rounds)
    qwen_controlled = bool(
        isinstance(scorer, QwenSkillRelevanceScorer)
        and qwen_scored_rounds == len(history)
        and api_fallback_rounds == 0
        and deterministic_recovery_rounds == 0
        and zero_score_fallback_rounds == 0
    )
    order = order_constraints(
        task_id, successful_sequence, oracle["door_required"]
    )
    verified_success = bool(
        planner_success and oracle["success"] and recording_integrity
    )
    qwen_controlled_success = bool(verified_success and qwen_controlled)
    strict_qwen_pass = bool(
        qwen_controlled_success and order["passed"] and recording_integrity
    )

    row = {
        "task_id": task_id,
        "task": task,
        "seed": int(seed),
        "scorer": type(scorer).__name__,
        "planner_success": planner_success,
        "ground_truth_success": oracle["success"],
        "verified_success": verified_success,
        "qwen_controlled": qwen_controlled,
        "qwen_controlled_success": qwen_controlled_success,
        "order_constraints_passed": order["passed"],
        "strict_qwen_pass": strict_qwen_pass,
        "status": result.get("status", "unknown"),
        "high_level_steps": result.get("high_level_steps", len(history)),
        "low_level_actions": result.get("low_level_actions", runtime.action_count),
        "wall_time_s": round(elapsed, 4),
        "skill_sequence": " -> ".join(
            record.get("call", {}).get("name", "unknown") for record in history
        ),
        "successful_skill_sequence": " -> ".join(successful_sequence),
        "qwen_scored_rounds": qwen_scored_rounds,
        "qwen_api_requests": qwen_api_requests,
        "qwen_parse_retries": qwen_parse_retries,
        "qwen_api_fallback_rounds": api_fallback_rounds,
        "deterministic_recovery_rounds": deterministic_recovery_rounds,
        "zero_score_fallback_rounds": zero_score_fallback_rounds,
        "mean_qwen_latency_s": (
            round(sum(latencies) / len(latencies), 4) if latencies else None
        ),
        "recorded_frames": len(frames),
        "runtime_action_count": runtime.action_count,
        "guarded_step_calls": recorder.step_calls,
        "frame_action_match": frame_action_match,
        "gateway_action_match": gateway_action_match,
        "trajectory_gif": str(gif_path),
        "final_frame": str(png_path),
        "oracle": oracle,
        "order_constraints": order,
        "error": error,
    }
    env.close()
    return row, result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen skill selection and composition on MiniGrid T1-T5."
    )
    parser.add_argument(
        "--scorer", choices=("qwen", "keyword", "auto"), default="auto",
        help="Use qwen for strict LLM evaluation; keyword is the offline baseline.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/multiskill_qwen",
        help="Directory for JSON and CSV evaluation results.",
    )
    parser.add_argument(
        "--seed-start", type=int, default=42,
        help="T1 uses this seed; T2-T5 use consecutive seeds.",
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=[task_id for task_id, _ in TASKS],
        default=[task_id for task_id, _ in TASKS],
    )
    args = parser.parse_args()

    if args.scorer == "qwen":
        if not os.getenv("DASHSCOPE_API_KEY"):
            parser.error("--scorer qwen requires DASHSCOPE_API_KEY; no keyword fallback is allowed.")
        scorer = QwenSkillRelevanceScorer()
    elif args.scorer == "keyword":
        scorer = KeywordRelevanceScorer()
    else:
        scorer = (
            QwenSkillRelevanceScorer()
            if os.getenv("DASHSCOPE_API_KEY")
            else KeywordRelevanceScorer()
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_tasks = [item for item in TASKS if item[0] in args.tasks]

    rows, details = [], []
    for index, (task_id, task) in enumerate(selected_tasks):
        seed = args.seed_start + index
        row, result = run_one(task_id, task, seed, scorer, output_dir)
        rows.append(row)
        details.append({"metrics": row, "result": result})
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))

    count = len(rows)
    summary = {
        "requested_scorer": args.scorer,
        "actual_scorer": type(scorer).__name__,
        "num_tasks": count,
        "planner_successes": sum(row["planner_success"] for row in rows),
        "ground_truth_successes": sum(row["ground_truth_success"] for row in rows),
        "verified_successes": sum(row["verified_success"] for row in rows),
        "qwen_controlled_successes": sum(row["qwen_controlled_success"] for row in rows),
        "strict_qwen_passes": sum(row["strict_qwen_pass"] for row in rows),
        "order_constraint_passes": sum(row["order_constraints_passed"] for row in rows),
        "total_qwen_api_requests": sum(row["qwen_api_requests"] for row in rows),
        "total_qwen_parse_retries": sum(row["qwen_parse_retries"] for row in rows),
        "total_api_fallback_rounds": sum(row["qwen_api_fallback_rounds"] for row in rows),
        "total_deterministic_recovery_rounds": sum(
            row["deterministic_recovery_rounds"] for row in rows
        ),
        "total_zero_score_fallback_rounds": sum(
            row["zero_score_fallback_rounds"] for row in rows
        ),
        "avg_high_level_steps": sum(row["high_level_steps"] for row in rows) / count,
        "avg_low_level_actions": sum(row["low_level_actions"] for row in rows) / count,
        "avg_wall_time_s": sum(row["wall_time_s"] for row in rows) / count,
    }

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "results.json"
    json_path.write_text(
        json.dumps({"summary": summary, "tasks": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Summary:", json.dumps(summary, ensure_ascii=False))
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
