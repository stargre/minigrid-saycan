#!/usr/bin/env python3
"""Qwen-API end-to-end demos for official MiniGrid GoToObject environments.

The planner uses SayCan-style selection:
    Qwen semantic relevance x observation-grounded feasibility.
Primitive actions are executed only through ObservationRuntime. A GIF frame is
recorded after every primitive action. Ground truth is inspected only after the
planner has stopped.

Run:
    python scripts/28_official_gotoobject_qwen_e2e.py \
        --env-ids MiniGrid-GoToObject-6x6-N2-v0 --seeds 0
"""
import argparse
import inspect
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from llm.skill_relevance import QwenSkillRelevanceScorer
from planners.unified_saycan_planner import UnifiedSayCanPlanner
from skills.observation_runtime import ObservationRuntime
from skills.target import PICKABLE_OBJECTS, TargetSpec, parse_target_specs


GOTOOBJECT_ENV_IDS = (
    "MiniGrid-GoToObject-6x6-N2-v0",
    "MiniGrid-GoToObject-8x8-N2-v0",
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
    list_method = getattr(value, "tolist", None)
    if callable(list_method):
        try:
            return json_safe(list_method())
        except (ValueError, TypeError):
            pass
    return str(value)


class RecordingStepOnlyEnv:
    """Record frames while exposing only the guarded step(action) interface."""

    def __init__(self, env, frames, mission):
        object.__setattr__(self, "_RecordingStepOnlyEnv__env", env)
        object.__setattr__(self, "_RecordingStepOnlyEnv__frames", frames)
        object.__setattr__(self, "_RecordingStepOnlyEnv__mission", str(mission))
        object.__setattr__(self, "step_calls", 0)

    def capture(self):
        env = object.__getattribute__(self, "_RecordingStepOnlyEnv__env")
        frames = object.__getattribute__(self, "_RecordingStepOnlyEnv__frames")
        frame = env.render()
        if frame is None:
            raise RuntimeError("MiniGrid returned no RGB frame.")
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        canvas = Image.new("RGB", (image.width, image.height + 34), "white")
        canvas.paste(image, (0, 34))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        mission = object.__getattribute__(self, "_RecordingStepOnlyEnv__mission")
        draw.text((5, 4), f"Mission: {mission}", fill=(0, 0, 0), font=font)
        draw.text(
            (5, 18),
            f"Primitive actions: {len(frames)}",
            fill=(0, 0, 0),
            font=font,
        )
        frames.append(np.asarray(canvas, dtype=np.uint8).copy())

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


def evaluate_after_run(env, runtime, history):
    """Read simulator ground truth only after planner.run() has returned."""
    base = env.unwrapped
    target_type = str(base.targetType)
    target_color = str(base.target_color)
    target_pos = tuple(int(value) for value in base.target_pos)
    agent_pos = tuple(int(value) for value in base.agent_pos)
    adjacent = bool(
        abs(agent_pos[0] - target_pos[0])
        + abs(agent_pos[1] - target_pos[1]) == 1
    )
    target_spec = TargetSpec.from_value(
        {"object": target_type, "color": target_color}
    )
    belief_target_observed = any(
        target_spec.matches(cell)
        for cell in runtime.belief.known_map.values()
    )
    carrying = base.carrying
    no_object_picked_up = bool(
        carrying is None and runtime.belief.inventory_status == "empty"
    )
    done_skill_called = any(
        record.get("call", {}).get("name") == "Done"
        for record in history
    )
    pickup_skill_called = any(
        record.get("call", {}).get("name") == "PickUp"
        for record in history
    )
    episode_success = bool(
        runtime.terminated
        and not runtime.truncated
        and float(runtime.last_reward) > 0.0
    )
    oracle_success = bool(
        episode_success
        and adjacent
        and belief_target_observed
        and no_object_picked_up
        and done_skill_called
        and not pickup_skill_called
    )
    return {
        "oracle_success": oracle_success,
        "episode_success": episode_success,
        "target": {"object": target_type, "color": target_color},
        "target_position": list(target_pos),
        "agent_position": list(agent_pos),
        "agent_adjacent_to_target": adjacent,
        "belief_target_observed": belief_target_observed,
        "simulator_carrying": (
            None
            if carrying is None
            else {
                "object": str(getattr(carrying, "type", "")),
                "color": str(getattr(carrying, "color", "")),
            }
        ),
        "no_object_picked_up": no_object_picked_up,
        "done_skill_called": done_skill_called,
        "pickup_skill_called": pickup_skill_called,
        "reward": float(runtime.last_reward),
        "terminated": bool(runtime.terminated),
        "truncated": bool(runtime.truncated),
        "completed_subgoals": sorted(runtime.belief.completed_subgoals),
        "belief_state": runtime.belief.summary(),
    }

def build_skill_trace(history):
    trace = []
    for record in history:
        execution = record.get("result", {})
        trace.append({
            "round": record.get("round"),
            "skill": record.get("call", {}).get("name"),
            "args": record.get("call", {}).get("args", {}),
            "relevance": record.get("relevance"),
            "feasibility": record.get("feasibility"),
            "score": record.get("score"),
            "relevance_source": record.get("relevance_source"),
            "relevance_error": record.get("relevance_error"),
            "relevance_rationale": record.get("relevance_rationale"),
            "feasibility_reason": record.get("feasibility_reason"),
            "execution_status": execution.get("status"),
            "execution_success": execution.get("success"),
            "low_level_actions": execution.get("low_level_actions"),
            "execution_trace": execution.get("execution_trace", []),
        })
    return trace


def run_case(env_id, seed, output_root, max_actions, max_high_level_steps,
             gif_duration_ms):
    started = time.perf_counter()
    env = None
    scorer = None
    frames = []
    recorder = None
    case_dir = output_root / f"{env_id}_seed_{seed}"
    case_dir.mkdir(parents=True, exist_ok=True)

    try:
        env = gym.make(env_id, render_mode="rgb_array")
        obs, _reset_info = env.reset(seed=int(seed))
        task = str(obs.get("mission", "")).strip()
        if not task:
            raise RuntimeError("GoToObject observation did not contain a mission.")

        explicit_targets = [
            spec for spec in parse_target_specs(task)
            if spec.object_name in PICKABLE_OBJECTS and spec.color
        ]
        if len(explicit_targets) != 1:
            raise RuntimeError(
                "Expected exactly one color-and-type GoToObject target in mission; "
                f"parsed={[(x.object_name, x.color) for x in explicit_targets]}"
            )

        recorder = RecordingStepOnlyEnv(env, frames, task)
        recorder.capture()
        runtime = ObservationRuntime(recorder)
        runtime.reset(obs, initial_inventory="empty")

        # This runner is intentionally Qwen-only. It never silently switches
        # to KeywordRelevanceScorer when the API is unavailable or malformed.
        scorer = QwenSkillRelevanceScorer()
        planner = UnifiedSayCanPlanner(
            runtime,
            relevance_scorer=scorer,
            max_actions=max_actions,
            max_skill_actions=300,
        )
        result = planner.run(task, max_high_level_steps=max_high_level_steps)

        # Ground-truth state is inspected only after all planning has stopped.
        history = result.get("history", [])
        oracle = evaluate_after_run(env, runtime, history)
        skill_trace = build_skill_trace(history)

        api_error_fallbacks = sum(
            "API error:" in str(record.get("relevance_source", ""))
            for record in history
        )
        qwen_scored_rounds = sum(
            record.get("relevance_source") == "QwenSkillRelevanceScorer"
            for record in history
        )
        deterministic_recovery_rounds = sum(
            "relevance_source" not in record for record in history
        )
        api_requests = int(getattr(scorer, "request_count", 0))
        api_latencies_ms = [
            round(float(value) * 1000.0, 2)
            for value in getattr(scorer, "api_latencies_s", [])
        ]

        gif_path, png_path = save_visuals(
            frames, case_dir, duration_ms=gif_duration_ms
        )
        frame_action_match = len(frames) == runtime.action_count + 1
        gateway_action_match = recorder.step_calls == runtime.action_count

        passed = bool(
            result.get("success")
            and oracle["oracle_success"]
            and oracle["done_skill_called"]
            and not oracle["pickup_skill_called"]
            and api_requests > 0
            and qwen_scored_rounds > 0
            and api_error_fallbacks == 0
            and frame_action_match
            and gateway_action_match
        )

        record = {
            "env_id": env_id,
            "seed": int(seed),
            "task": task,
            "scorer": "QwenSkillRelevanceScorer",
            "qwen_model": getattr(scorer, "model", None),
            "qwen_api_used": api_requests > 0,
            "qwen_api_requests": api_requests,
            "qwen_scored_rounds": qwen_scored_rounds,
            "qwen_api_error_fallbacks": api_error_fallbacks,
            "deterministic_recovery_rounds": deterministic_recovery_rounds,
            "qwen_parse_retries": int(getattr(scorer, "parse_retry_count", 0)),
            "qwen_api_latencies_ms": api_latencies_ms,
            "planner_success": bool(result.get("success")),
            "planner_status": result.get("status"),
            "oracle": oracle,
            "passed": passed,
            "high_level_steps": result.get("high_level_steps"),
            "low_level_actions": result.get("low_level_actions"),
            "guarded_step_calls": recorder.step_calls,
            "runtime_action_count": runtime.action_count,
            "recorded_frames": len(frames),
            "frame_action_match": frame_action_match,
            "gateway_action_match": gateway_action_match,
            "wall_time_s": round(time.perf_counter() - started, 4),
            "trajectory_gif": str(gif_path),
            "final_frame": str(png_path),
            "summary_json": str(case_dir / "summary.json"),
            "skill_trace_json": str(case_dir / "skill_trace.json"),
            "skill_trace": skill_trace,
        }
        record = json_safe(record)

        (case_dir / "summary.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (case_dir / "skill_trace.json").write_text(
            json.dumps(json_safe(skill_trace), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record

    except Exception as exc:
        gif_path = None
        png_path = None
        if frames:
            try:
                gif_path, png_path = save_visuals(
                    frames, case_dir, duration_ms=gif_duration_ms
                )
            except Exception:
                pass
        failure_record = json_safe({
            "env_id": env_id,
            "seed": int(seed),
            "scorer": "QwenSkillRelevanceScorer",
            "passed": False,
            "wall_time_s": round(time.perf_counter() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
            "qwen_model": getattr(scorer, "model", None),
            "qwen_api_requests": getattr(scorer, "request_count", 0),
            "qwen_api_error_fallbacks": None,
            "recorded_frames": len(frames),
            "trajectory_gif": str(gif_path) if gif_path else None,
            "final_frame": str(png_path) if png_path else None,
            "summary_json": str(case_dir / "summary.json"),
        })
        (case_dir / "summary.json").write_text(
            json.dumps(failure_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (case_dir / "skill_trace.json").write_text("[]", encoding="utf-8")
        return failure_record
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Qwen-API + SayCan-style end-to-end tests on official MiniGrid GoToObject."
    )
    parser.add_argument(
        "--env-ids",
        nargs="+",
        choices=GOTOOBJECT_ENV_IDS,
        default=list(GOTOOBJECT_ENV_IDS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--max-high-level-steps", type=int, default=160)
    parser.add_argument(
        "--output-dir",
        default="outputs/official_gotoobject_qwen",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--gif-duration-ms", type=int, default=180)
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        parser.error(
            "This is a Qwen-API-only test. Configure DASHSCOPE_API_KEY first; "
            "the key is never printed or written to result files."
        )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = (
        Path(args.output)
        if args.output
        else output_root / "results.json"
    )

    records = []
    for env_id in args.env_ids:
        for seed in args.seeds:
            record = run_case(
                env_id=env_id,
                seed=seed,
                output_root=output_root,
                max_actions=args.max_actions,
                max_high_level_steps=args.max_high_level_steps,
                gif_duration_ms=args.gif_duration_ms,
            )
            records.append(record)
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))

    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(
        json.dumps(json_safe(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    failures = [record for record in records if not record.get("passed")]
    total_api_requests = sum(
        int(record.get("qwen_api_requests") or 0) for record in records
    )
    total_fallbacks = sum(
        int(record.get("qwen_api_error_fallbacks") or 0) for record in records
    )
    print(
        f"Official GoToObject Qwen acceptance: "
        f"{len(records) - len(failures)}/{len(records)} passed."
    )
    print(f"Qwen API requests: {total_api_requests}; API fallbacks: {total_fallbacks}")
    print(f"Aggregate results: {aggregate_path}")
    for record in records:
        if record.get("trajectory_gif"):
            print(f"GIF: {record['trajectory_gif']}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
