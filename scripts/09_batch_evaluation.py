import csv
import json
import time
from pathlib import Path

import numpy as np

from environments.saycan_grid import SayCanGridEnv
from skills.movement import MovementSkill
from skills.manipulation import PickSkill, PlaceSkill
from skills.interaction import OpenDoorSkill
from planners.saycan_planner import SayCanPlanner
from utils.visualization import FrameRecorder


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "batch_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TASKS = [
    {
        "id": "T1",
        "task": (
            "Pick up the red ball, open the yellow door, "
            "and place it above the green goal."
        ),
    },
    {
        "id": "T2",
        "task": (
            "Open the yellow door first, then pick up the "
            "red ball, and finally place it above the green goal."
        ),
    },
    {
        "id": "T3",
        "task": (
            "Take the red ball, open the door, "
            "and drop it above the goal."
        ),
    },
    {
        "id": "T4",
        "task": (
            "Open the door, then place the red ball above "
            "the goal after picking it up."
        ),
    },
    {
        "id": "T5",
        "task": (
            "Grab the red ball and put it above the green goal; "
            "open the yellow door if needed."
        ),
    },
]


def json_default(obj):
    """处理 NumPy 数值等 JSON 不直接支持的类型。"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def run_one_task(task_id, task_text, seed):
    """运行单个任务，返回指标、详细日志和 GIF 路径。"""

    env = SayCanGridEnv(render_mode="rgb_array")
    env.reset(seed=seed)

    recorder = FrameRecorder(env)
    movement = MovementSkill(recorder)
    picker = PickSkill(recorder, movement)
    opener = OpenDoorSkill(recorder, movement)
    placer = PlaceSkill(recorder, movement)

    planner = SayCanPlanner(
        recorder,
        movement,
        picker,
        opener,
        placer,
    )

    start_time = time.perf_counter()

    try:
        success, history = planner.run(
            task_text,
            max_high_level_steps=10,
        )
    except Exception as exc:
        success = False
        history = []
        error = repr(exc)
    else:
        error = ""

    # 包含规划、动作执行、渲染记录和终端输出时间。
    # 不包含后续 GIF 文件编码时间。
    elapsed = time.perf_counter() - start_time

    gif_path = OUTPUT_DIR / f"{task_id}.gif"

    try:
        recorder.save_gif(gif_path, duration=500)
    finally:
        env.close()

    attempted = len(history)
    failed_calls = sum(
        not item["result"]["success"]
        for item in history
    )

    low_level_actions = movement.action_count

    row = {
        "task_id": task_id,
        "task": task_text,
        "success": bool(success),
        "high_level_steps": attempted,
        "low_level_actions": low_level_actions,
        "failed_skill_calls": failed_calls,
        "invalid_skill_call_rate": (
            failed_calls / attempted if attempted else 0.0
        ),
        "planner_run_wall_time_s": round(elapsed, 4),
        "skill_sequence": " -> ".join(
            item["skill"] for item in history
        ),
        "gif_path": str(gif_path.relative_to(ROOT)),
        "error": error,
    }

    detail = {
        **row,
        "planning_history": history,
    }

    print("\nTask result:", row)

    return row, detail


def main():
    rows = []
    details = []

    for index, item in enumerate(TASKS):
        print("\n" + "#" * 72)
        print(f"Evaluating {item['id']}: {item['task']}")
        print("#" * 72)

        row, detail = run_one_task(
            task_id=item["id"],
            task_text=item["task"],
            seed=42 + index,
        )

        rows.append(row)
        details.append(detail)

    # 汇总指标
    num_tasks = len(rows)
    successful_tasks = sum(row["success"] for row in rows)

    total_calls = sum(row["high_level_steps"] for row in rows)
    total_failed_calls = sum(
        row["failed_skill_calls"] for row in rows
    )

    summary = {
        "num_tasks": num_tasks,
        "successful_tasks": successful_tasks,
        "failed_tasks": num_tasks - successful_tasks,
        "task_success_rate": (
            successful_tasks / num_tasks if num_tasks else 0.0
        ),
        "avg_high_level_steps": (
            sum(row["high_level_steps"] for row in rows)
            / num_tasks if num_tasks else 0.0
        ),
        "avg_low_level_actions": (
            sum(row["low_level_actions"] for row in rows)
            / num_tasks if num_tasks else 0.0
        ),
        "total_skill_calls": total_calls,
        "failed_skill_calls": total_failed_calls,
        "invalid_skill_call_rate": (
            total_failed_calls / total_calls
            if total_calls else 0.0
        ),
        "avg_planner_run_wall_time_s": (
            sum(row["planner_run_wall_time_s"] for row in rows)
            / num_tasks if num_tasks else 0.0
        ),
    }

    # 保存逐任务 CSV
    csv_path = OUTPUT_DIR / "results.csv"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    # 保存汇总与详细规划日志
    json_path = OUTPUT_DIR / "results.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "tasks": details,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )

    print("\n" + "=" * 72)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 72)

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nCSV:", csv_path)
    print("JSON:", json_path)
    print("GIF directory:", OUTPUT_DIR)

    print("\nPer-task results:")
    for row in rows:
        print(
            f"{row['task_id']}: "
            f"success={row['success']}, "
            f"skills={row['skill_sequence']}, "
            f"actions={row['low_level_actions']}"
        )


if __name__ == "__main__":
    main()
