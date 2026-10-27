"""Batch evaluation with planner metrics and independent ground-truth checks."""
import csv
import json
import os
import time
from pathlib import Path

from minigrid.core.world_object import Ball

from environments.saycan_grid import SayCanGridEnv
from skills.observation_runtime import ObservationRuntime
from planners.unified_saycan_planner import UnifiedSayCanPlanner
from llm.skill_relevance import KeywordRelevanceScorer, QwenSkillRelevanceScorer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "batch_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    ("T1", "Pick up the red ball, open the yellow door, and place it above the green goal."),
    ("T2", "Open the yellow door first, then pick up the red ball, and finally place it above the green goal."),
    ("T3", "Take the red ball, open the yellow door, and drop it above the green goal."),
    ("T4", "Open the yellow door, then place the red ball above the green goal after picking it up."),
    ("T5", "Grab the red ball and put it above the green goal; open the yellow door if needed."),
]


def ground_truth_success(env):
    """Evaluation-only oracle; never used by the planner or skill implementation."""
    base = env.unwrapped
    placed = base.grid.get(*base.target_pos)
    door = base.grid.get(*base.door_pos)
    return bool(
        isinstance(placed, Ball)
        and placed.color == "red"
        and getattr(door, "is_open", False)
        and getattr(base, "carrying", None) is None
    )


def run_one(task_id, task, seed, scorer):
    env = SayCanGridEnv(render_mode=None)
    obs, _ = env.reset(seed=seed)
    runtime = ObservationRuntime(env)
    runtime.reset(obs, initial_inventory="empty")
    planner = UnifiedSayCanPlanner(
        runtime,
        relevance_scorer=scorer,
        max_actions=450,
        max_skill_actions=300,
    )

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
        error = repr(exc)

    elapsed = time.perf_counter() - started
    planner_success = bool(result.get("success", False))
    gt_success = ground_truth_success(env)
    verified_success = planner_success and gt_success
    history = result.get("history", [])

    row = {
        "task_id": task_id,
        "task": task,
        "seed": seed,
        "planner_success": planner_success,
        "ground_truth_success": gt_success,
        "success": verified_success,
        "status": result.get("status", "unknown"),
        "high_level_steps": result.get("high_level_steps", len(history)),
        "low_level_actions": result.get("low_level_actions", runtime.action_count),
        "wall_time_s": round(elapsed, 4),
        "skill_sequence": " -> ".join(record["call"]["name"] for record in history),
        "error": error,
    }
    env.close()
    return row, result


def main():
    scorer = (
        QwenSkillRelevanceScorer()
        if os.getenv("DASHSCOPE_API_KEY")
        else KeywordRelevanceScorer()
    )
    rows, details = [], []

    for index, (task_id, task) in enumerate(TASKS):
        row, result = run_one(task_id, task, 42 + index, scorer)
        rows.append(row)
        details.append({"metrics": row, "result": result})
        print(row)

    count = len(rows)
    summary = {
        "num_tasks": count,
        "planner_successes": sum(row["planner_success"] for row in rows),
        "ground_truth_successes": sum(row["ground_truth_success"] for row in rows),
        "verified_successes": sum(row["success"] for row in rows),
        "planner_success_rate": sum(row["planner_success"] for row in rows) / count,
        "ground_truth_success_rate": sum(row["ground_truth_success"] for row in rows) / count,
        "verified_success_rate": sum(row["success"] for row in rows) / count,
        "avg_high_level_steps": sum(row["high_level_steps"] for row in rows) / count,
        "avg_low_level_actions": sum(row["low_level_actions"] for row in rows) / count,
        "avg_wall_time_s": sum(row["wall_time_s"] for row in rows) / count,
    }

    csv_path = OUTPUT_DIR / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUTPUT_DIR / "results.json"
    json_path.write_text(
        json.dumps({"summary": summary, "tasks": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Summary:", summary)
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
