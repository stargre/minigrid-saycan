#!/usr/bin/env python3
"""Run a visual SayCan audit across every registered MiniGrid environment.

This is a capability audit, not a claim that every MiniGrid mechanism is
supported. Each runnable environment is attempted and classified as passed or
failed. Empty-mission environments are unsupported and optional-dependency
failures are unavailable. Every instantiated case gets a GIF, final PNG and
JSON trace; an HTML gallery and manifest summarize the whole catalog.

Examples:
    python scripts/31_all_minigrid_envs_gif_audit.py
    python scripts/31_all_minigrid_envs_gif_audit.py --match "*Door*" "*Unlock*"
    python scripts/31_all_minigrid_envs_gif_audit.py --scorer qwen \
        --env-ids MiniGrid-DoorKey-5x5-v0 MiniGrid-Fetch-5x5-N2-v0
"""
from __future__ import annotations

import argparse
import fnmatch
import html
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401 - registers official environments
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from llm.skill_relevance import KeywordRelevanceScorer, QwenSkillRelevanceScorer
from planners.unified_saycan_planner import UnifiedSayCanPlanner
from skills.observation_runtime import ObservationRuntime


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return json_safe(item_method())
        except (TypeError, ValueError):
            pass
    return str(value)


def registered_minigrid_envs():
    return sorted(
        spec.id for spec in gym.registry.values()
        if spec.id.startswith("MiniGrid-")
    )


def select_envs(all_ids, explicit, patterns):
    if explicit:
        unknown = sorted(set(explicit) - set(all_ids))
        if unknown:
            raise ValueError(f"Unknown MiniGrid environment IDs: {unknown}")
        return list(dict.fromkeys(explicit))
    if patterns:
        return [
            env_id for env_id in all_ids
            if any(fnmatch.fnmatchcase(env_id, pattern) for pattern in patterns)
        ]
    return all_ids


def slugify(env_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", env_id)


class RecordingStepOnlyEnv:
    """Guard primitive actions and retain a bounded set of rendered frames."""

    def __init__(self, env, mission, env_id, frame_stride, max_frames,
                 max_width):
        object.__setattr__(self, "_RecordingStepOnlyEnv__env", env)
        object.__setattr__(self, "_RecordingStepOnlyEnv__mission", str(mission))
        object.__setattr__(self, "_RecordingStepOnlyEnv__env_id", str(env_id))
        object.__setattr__(self, "_RecordingStepOnlyEnv__frame_stride", max(1, int(frame_stride)))
        object.__setattr__(self, "_RecordingStepOnlyEnv__max_frames", max(2, int(max_frames)))
        object.__setattr__(self, "_RecordingStepOnlyEnv__max_width", max(64, int(max_width)))
        object.__setattr__(self, "frames", [])
        object.__setattr__(self, "step_calls", 0)

    def _render_frame(self):
        env = object.__getattribute__(self, "_RecordingStepOnlyEnv__env")
        frame = env.render()
        if frame is None:
            raise RuntimeError("MiniGrid returned no RGB frame.")
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        max_width = object.__getattribute__(self, "_RecordingStepOnlyEnv__max_width")
        if image.width > max_width:
            height = max(1, round(image.height * max_width / image.width))
            image = image.resize((max_width, height), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (image.width, image.height + 48), "white")
        canvas.paste(image, (0, 48))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        env_id = object.__getattribute__(self, "_RecordingStepOnlyEnv__env_id")
        mission = object.__getattribute__(self, "_RecordingStepOnlyEnv__mission")
        draw.text((5, 3), env_id, fill=(0, 0, 0), font=font)
        draw.text((5, 17), f"Mission: {mission[:100]}", fill=(0, 0, 0), font=font)
        draw.text((5, 32), f"Primitive actions: {self.step_calls}", fill=(0, 0, 0), font=font)
        return np.asarray(canvas, dtype=np.uint8).copy()

    def capture(self, force=False):
        frames = self.frames
        max_frames = object.__getattribute__(self, "_RecordingStepOnlyEnv__max_frames")
        stride = object.__getattribute__(self, "_RecordingStepOnlyEnv__frame_stride")
        if not force and self.step_calls % stride:
            return
        frame = self._render_frame()
        if force and frames and np.array_equal(frames[-1], frame):
            return
        if len(frames) < max_frames:
            frames.append(frame)
        elif force:
            frames[-1] = frame

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
            f"Decision code attempted hidden-state access: env.{name}; "
            "only step(action) is available."
        )


def save_visuals(frames, case_dir, duration_ms):
    case_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        return None, None
    images = [Image.fromarray(frame) for frame in frames]
    gif_path = case_dir / "trajectory.gif"
    png_path = case_dir / "final_frame.png"
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


def compact_history(history):
    return [
        {
            "round": record.get("round"),
            "skill": record.get("call", {}).get("name"),
            "args": record.get("call", {}).get("args", {}),
            "status": record.get("result", {}).get("status"),
            "success": record.get("result", {}).get("success"),
            "low_level_actions": record.get("result", {}).get("low_level_actions"),
            "relevance": record.get("relevance"),
            "affordance": record.get("feasibility"),
            "combined_score": record.get("score"),
            "selection_policy": record.get("selection_policy"),
            "relevance_source": record.get("relevance_source"),
            "relevance_error": record.get("relevance_error"),
        }
        for record in history
    ]


def run_case(env_id, seed, scorer_name, output_root, max_actions,
             max_high_level_steps, max_gif_frames, gif_max_width,
             gif_duration_ms):
    started = time.perf_counter()
    case_dir = output_root / slugify(env_id) / f"seed_{seed}"
    env = None
    recorder = None
    record = {
        "env_id": env_id,
        "seed": int(seed),
        "scorer": scorer_name,
        "experiment_kind": "llm_saycan" if scorer_name == "qwen" else "offline_baseline",
    }
    try:
        try:
            env = gym.make(env_id, render_mode="rgb_array")
        except Exception as exc:
            record.update({
                "classification": "unavailable",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
            return record

        obs, _ = env.reset(seed=seed)
        base = env.unwrapped
        mission = str(obs.get("mission", "")).strip()
        env_budget = int(getattr(base, "max_steps", max_actions))
        action_budget = max(1, min(int(max_actions), env_budget))
        stride = max(1, math.ceil(action_budget / max(1, max_gif_frames - 1)))
        recorder = RecordingStepOnlyEnv(
            env=env,
            mission=mission or "<empty mission>",
            env_id=env_id,
            frame_stride=stride,
            max_frames=max_gif_frames,
            max_width=gif_max_width,
        )
        recorder.capture(force=True)
        record.update({
            "environment_class": type(base).__name__,
            "mission": mission,
            "environment_max_steps": env_budget,
            "planner_action_budget": action_budget,
            "gif_frame_stride": stride,
        })

        if not mission:
            recorder.capture(force=True)
            gif_path, png_path = save_visuals(
                recorder.frames, case_dir, gif_duration_ms
            )
            record.update({
                "classification": "unsupported",
                "passed": False,
                "reason": "The environment exposes no mission string.",
                "trajectory_gif": str(gif_path) if gif_path else None,
                "final_frame": str(png_path) if png_path else None,
            })
            return record

        runtime = ObservationRuntime(recorder)
        runtime.reset(obs, initial_inventory="empty")
        scorer = (
            QwenSkillRelevanceScorer()
            if scorer_name == "qwen"
            else KeywordRelevanceScorer()
        )
        planner = UnifiedSayCanPlanner(
            runtime,
            relevance_scorer=scorer,
            max_actions=action_budget,
            max_skill_actions=action_budget,
        )
        result = planner.run(
            mission, max_high_level_steps=max_high_level_steps
        )
        recorder.capture(force=True)
        gif_path, png_path = save_visuals(
            recorder.frames, case_dir, gif_duration_ms
        )

        history = result.get("history", [])
        fallbacks = [
            item for item in history if item.get("relevance_error") is not None
        ]
        environment_success = bool(
            runtime.terminated
            and not runtime.truncated
            and float(runtime.last_reward) > 0.0
        )
        gateway_consistent = recorder.step_calls == runtime.action_count
        qwen_valid = scorer_name != "qwen" or (
            getattr(scorer, "request_count", 0) > 0 and not fallbacks
        )
        passed = bool(
            result.get("success")
            and environment_success
            and gateway_consistent
            and qwen_valid
        )
        record.update({
            "classification": "passed" if passed else "failed",
            "passed": passed,
            "planner_success": bool(result.get("success")),
            "planner_status": result.get("status"),
            "environment_success": environment_success,
            "reward": float(runtime.last_reward),
            "terminated": bool(runtime.terminated),
            "truncated": bool(runtime.truncated),
            "gateway_consistent": gateway_consistent,
            "runtime_action_count": runtime.action_count,
            "guarded_step_calls": recorder.step_calls,
            "high_level_steps": result.get("high_level_steps"),
            "low_level_actions": result.get("low_level_actions"),
            "qwen_model": getattr(scorer, "model", None),
            "qwen_api_requests": getattr(scorer, "request_count", 0),
            "qwen_api_error_fallbacks": len(fallbacks),
            "qwen_parse_retries": getattr(scorer, "parse_retry_count", 0),
            "skill_trace": compact_history(history),
            "trajectory_gif": str(gif_path) if gif_path else None,
            "final_frame": str(png_path) if png_path else None,
            "recorded_frames": len(recorder.frames),
        })
        return record
    except Exception as exc:
        record.update({
            "classification": "error",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        if recorder is not None:
            try:
                recorder.capture(force=True)
                gif_path, png_path = save_visuals(
                    recorder.frames, case_dir, gif_duration_ms
                )
                record["trajectory_gif"] = str(gif_path) if gif_path else None
                record["final_frame"] = str(png_path) if png_path else None
            except Exception as visual_exc:
                record["visual_error"] = f"{type(visual_exc).__name__}: {visual_exc}"
        return record
    finally:
        record["wall_time_s"] = round(time.perf_counter() - started, 6)
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "result.json").write_text(
            json.dumps(json_safe(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if env is not None:
            env.close()


def build_summary(records, scorer, seed):
    counts = {}
    for record in records:
        key = record.get("classification", "error")
        counts[key] = counts.get(key, 0) + 1
    attempted = [
        record for record in records
        if record.get("classification") in {"passed", "failed", "error"}
    ]
    return {
        "suite": "all_minigrid_visual_capability_audit_v1",
        "runtime": {
            "python": platform.python_version(),
            "minigrid": importlib.metadata.version("minigrid"),
            "gymnasium": importlib.metadata.version("gymnasium"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "scorer": scorer,
        "seed": int(seed),
        "catalog_cases": len(records),
        "attempted_cases": len(attempted),
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "unsupported": counts.get("unsupported", 0),
        "unavailable": counts.get("unavailable", 0),
        "errors": counts.get("error", 0),
        "attempted_success_rate": (
            counts.get("passed", 0) / len(attempted) if attempted else 0.0
        ),
    }


def write_gallery(records, summary, output_root):
    cards = []
    for record in records:
        classification = str(record.get("classification", "error"))
        gif_path = record.get("trajectory_gif")
        gif_html = ""
        if gif_path:
            try:
                relative = Path(gif_path).resolve().relative_to(output_root.resolve())
                gif_html = (
                    f'<img loading="lazy" src="{html.escape(relative.as_posix())}" '
                    f'alt="{html.escape(record["env_id"])} trajectory">'
                )
            except ValueError:
                pass
        error = record.get("error") or record.get("reason") or ""
        cards.append(f"""
<article class="card {html.escape(classification)}">
  <h2>{html.escape(record['env_id'])}</h2>
  <div class="badge">{html.escape(classification)}</div>
  {gif_html}
  <p><b>Mission:</b> {html.escape(str(record.get('mission', '')))}</p>
  <p><b>Status:</b> {html.escape(str(record.get('planner_status', '')))}</p>
  <p><b>Actions:</b> {html.escape(str(record.get('low_level_actions', '')))}</p>
  <p class="error">{html.escape(str(error))}</p>
</article>""")
    summary_text = html.escape(json.dumps(summary, ensure_ascii=False))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniGrid SayCan visual capability audit</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f4f6f8;color:#16202a}}
header{{max-width:1200px;margin:auto auto 24px}} code{{white-space:pre-wrap}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;max-width:1600px;margin:auto}}
.card{{background:white;border:2px solid #ccd4dc;border-radius:12px;padding:14px;box-shadow:0 3px 12px #0001}}
.card.passed{{border-color:#27ae60}} .card.failed,.card.error{{border-color:#c0392b}}
.card.unsupported,.card.unavailable{{border-color:#f39c12}} h2{{font-size:16px;word-break:break-word}}
.badge{{display:inline-block;padding:3px 9px;border-radius:999px;background:#e8edf2;font-weight:700}}
img{{display:block;width:100%;height:auto;margin:12px 0;border-radius:6px;background:#111}}
p{{font-size:13px;word-break:break-word}} .error{{color:#9b2c2c}}
</style></head><body><header><h1>MiniGrid SayCan visual capability audit</h1>
<code>{summary_text}</code></header><main class="grid">{''.join(cards)}</main></body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Audit every registered MiniGrid environment and render GIFs."
    )
    parser.add_argument("--env-ids", nargs="+", help="Exact environment IDs.")
    parser.add_argument(
        "--match", nargs="+",
        help='Shell-style filters, e.g. "*Door*" "*Fetch*".',
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scorer", choices=("keyword", "qwen"), default="keyword")
    parser.add_argument("--max-actions", type=int, default=1200)
    parser.add_argument("--max-high-level-steps", type=int, default=300)
    parser.add_argument("--max-gif-frames", type=int, default=160)
    parser.add_argument("--gif-max-width", type=int, default=320)
    parser.add_argument("--gif-duration-ms", type=int, default=160)
    parser.add_argument("--output-dir", default="outputs/all_minigrid_visual_audit")
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero if any attempted environment does not pass.",
    )
    args = parser.parse_args()

    if args.scorer == "qwen" and not os.getenv("DASHSCOPE_API_KEY"):
        parser.error("--scorer qwen requires DASHSCOPE_API_KEY.")
    all_ids = registered_minigrid_envs()
    try:
        env_ids = select_envs(all_ids, args.env_ids, args.match)
    except ValueError as exc:
        parser.error(str(exc))
    if not env_ids:
        parser.error("No environments matched the requested filters.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, env_id in enumerate(env_ids, start=1):
        record = run_case(
            env_id=env_id,
            seed=args.seed,
            scorer_name=args.scorer,
            output_root=output_root,
            max_actions=args.max_actions,
            max_high_level_steps=args.max_high_level_steps,
            max_gif_frames=args.max_gif_frames,
            gif_max_width=args.gif_max_width,
            gif_duration_ms=args.gif_duration_ms,
        )
        records.append(record)
        print(
            f"[{index:02d}/{len(env_ids):02d}] "
            f"{record.get('classification', 'error').upper():11s} "
            f"{env_id} actions={record.get('low_level_actions', '-')}"
        )

    summary = build_summary(records, args.scorer, args.seed)
    manifest = {"summary": summary, "records": records}
    (output_root / "manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_gallery(records, summary, output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Gallery: {output_root / 'index.html'}")
    print(f"Manifest: {output_root / 'manifest.json'}")
    if args.strict and (summary["failed"] or summary["errors"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
