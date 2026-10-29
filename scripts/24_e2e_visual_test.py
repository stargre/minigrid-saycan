
import argparse
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from minigrid.core.world_object import Ball

from environments.saycan_grid import SayCanGridEnv
from skills.observation_runtime import ObservationRuntime
from planners.unified_saycan_planner import UnifiedSayCanPlanner
from llm.skill_relevance import (
    KeywordRelevanceScorer,
    QwenSkillRelevanceScorer,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TASK = (
    "Pick up the red ball, open the yellow door, "
    "and place it above the green goal."
)


class FrameRecorder:
    """Record RGB frames while keeping the environment hidden from the planner."""

    def __init__(self, env, frames):
        object.__setattr__(self, "_FrameRecorder__env", env)
        object.__setattr__(self, "_FrameRecorder__frames", frames)

    def step(self, action):
        caller = inspect.currentframe().f_back
        if (
            caller is None
            or caller.f_code.co_name != "execute_action"
            or caller.f_globals.get("__name__")
            != "skills.observation_runtime"
        ):
            raise AssertionError(
                "Environment actions must go through "
                "ObservationRuntime.execute_action."
            )

        env = object.__getattribute__(self, "_FrameRecorder__env")
        result = env.step(action)
        self.capture()
        return result

    def capture(self):
        env = object.__getattribute__(self, "_FrameRecorder__env")
        frames = object.__getattribute__(
            self, "_FrameRecorder__frames"
        )
        frame = env.render()

        if frame is not None:
            frames.append(
                np.asarray(frame, dtype=np.uint8).copy()
            )

    def __getattr__(self, name):
        raise AssertionError(
            f"Planner attempted environment access: {name}"
        )


def inspect_final_state(env, runtime):
    """Ground-truth inspection, called only after planning has stopped."""

    placed = env.grid.get(*env.target_pos)
    door = env.grid.get(*env.door_pos)

    ball_at_target = (
        isinstance(placed, Ball)
        and placed.color == "red"
    )

    door_open = bool(
        getattr(door, "is_open", False)
        and not getattr(door, "is_locked", False)
    )

    carrying_empty = env.carrying is None

    red_ball_positions = []
    for x, y in ((x, y) for y in range(env.grid.height) for x in range(env.grid.width)):
        obj = env.grid.get(x, y)
        if isinstance(obj, Ball) and obj.color == "red":
            red_ball_positions.append([x, y])

    return {
        "ground_truth_success": bool(
            ball_at_target and door_open and carrying_empty
        ),
        "checks": {
            "red_ball_at_target": ball_at_target,
            "yellow_door_open": door_open,
            "inventory_empty": carrying_empty,
        },
        "environment_state": {
            "agent_position": list(env.agent_pos),
            "door_position": list(env.door_pos),
            "door_open": door_open,
            "goal_position": list(env.goal_pos),
            "placement_target_position": list(env.target_pos),
            "red_ball_positions": red_ball_positions,
            "carrying": (
                None
                if env.carrying is None
                else {
                    "object": type(env.carrying).__name__,
                    "color": getattr(env.carrying, "color", None),
                }
            ),
            "reward": runtime.last_reward,
            "terminated": runtime.terminated,
            "truncated": runtime.truncated,
        },
        "belief_state": runtime.belief.summary(),
    }


def save_frames(frames, output_dir):
    if not frames:
        raise RuntimeError("No frames were recorded.")

    images = [Image.fromarray(frame) for frame in frames]

    gif_path = output_dir / "trajectory.gif"
    png_path = output_dir / "final_frame.png"

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=180,
        loop=0,
    )
    images[-1].save(png_path)

    return gif_path, png_path


def _json_default(value):
    """Convert NumPy scalar and array values to JSON-compatible types."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scorer",
        choices=["auto", "qwen", "keyword"],
        default="auto",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-actions", type=int, default=450)
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "outputs" / "e2e_visual" / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.scorer == "qwen":
        scorer = QwenSkillRelevanceScorer()
        scorer_name = "Qwen"
    elif args.scorer == "keyword":
        scorer = KeywordRelevanceScorer()
        scorer_name = "Keyword"
    elif os.getenv("DASHSCOPE_API_KEY"):
        scorer = QwenSkillRelevanceScorer()
        scorer_name = "Qwen"
    else:
        scorer = KeywordRelevanceScorer()
        scorer_name = "Keyword"

    env = SayCanGridEnv(render_mode="rgb_array")
    frames = []

    try:
        obs, _ = env.reset(seed=args.seed)

        recorder = FrameRecorder(env, frames)
        recorder.capture()

        runtime = ObservationRuntime(recorder)
        runtime.reset(obs, initial_inventory="empty")

        planner = UnifiedSayCanPlanner(
            runtime,
            relevance_scorer=scorer,
            max_actions=args.max_actions,
            max_skill_actions=300,
        )

        started = time.perf_counter()
        result = planner.run(
            args.task,
            max_high_level_steps=100,
        )
        elapsed = time.perf_counter() - started

        # Inspect the real environment only after planning is complete.
        final = inspect_final_state(env, runtime)

        passed = bool(
            result.get("success")
            and final["ground_truth_success"]
        )

        history = []
        print("\n========== HIGH-LEVEL SKILL TRACE ==========")

        for record in result.get("history", []):
            call = record["call"]
            execution = record["result"]

            item = {
                "round": record["round"],
                "skill": call["name"],
                "args": call.get("args", {}),
                "relevance": record["relevance"],
                "feasibility": record["feasibility"],
                "score": record["score"],
                "relevance_rationale": record["relevance_rationale"],
                "feasibility_reason": record["feasibility_reason"],
                "execution_status": execution.get("status"),
                "execution_success": execution.get("success"),
                "low_level_actions": execution.get("low_level_actions"),
                "execution_trace": execution.get("execution_trace", []),
            }
            history.append(item)

            print(
                f"Round {item['round']:02d} | "
                f"{item['skill']} | "
                f"R={item['relevance']:.3f} | "
                f"F={item['feasibility']:.1f} | "
                f"Score={item['score']:.3f} | "
                f"Status={item['execution_status']} | "
                f"Actions={item['low_level_actions']}"
            )

        gif_path, png_path = save_frames(frames, output_dir)

        summary = {
            "seed": args.seed,
            "task": args.task,
            "scorer": scorer_name,
            "planner_success": bool(result.get("success")),
            "planner_status": result.get("status"),
            "ground_truth_success": final["ground_truth_success"],
            "passed": passed,
            "high_level_steps": result.get("high_level_steps"),
            "low_level_actions": result.get("low_level_actions"),
            "wall_time_s": round(elapsed, 4),
            "final_state": final,
            "trajectory_gif": str(gif_path),
            "final_frame": str(png_path),
        }

        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

        (output_dir / "skill_trace.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n========== FINAL AGENT STATE ==========")
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
        print("\n========== RESULT FILES ==========")
        print("Summary:", output_dir / "summary.json")
        print("Skill trace:", output_dir / "skill_trace.json")
        print("Trajectory:", gif_path)
        print("Final image:", png_path)
        print(
            f"Recorded frames: {len(frames)} "
            f"(actions: {runtime.action_count})"
        )

        if len(frames) != runtime.action_count + 1:
            raise AssertionError(
                "Frame/action count mismatch: expected one initial frame "
                "plus one frame per primitive action."
            )

        if not passed:
            raise SystemExit(1)

    finally:
        env.close()


if __name__ == "__main__":
    main()

