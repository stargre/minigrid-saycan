"""End-to-end SayCan test with independent ground-truth verification."""
import os

from minigrid.core.world_object import Ball

from environments.saycan_grid import SayCanGridEnv
from skills.observation_runtime import ObservationRuntime
from planners.unified_saycan_planner import UnifiedSayCanPlanner
from llm.skill_relevance import KeywordRelevanceScorer, QwenSkillRelevanceScorer


def ground_truth_success(env):
    """Evaluation-only oracle; skills and planner must never use this function."""
    base = env.unwrapped
    placed = base.grid.get(*base.target_pos)
    door = base.grid.get(*base.door_pos)
    return bool(
        isinstance(placed, Ball)
        and placed.color == "red"
        and getattr(door, "is_open", False)
        and getattr(base, "carrying", None) is None
    )


def main():
    seed = int(os.getenv("SEED", "42"))
    env = SayCanGridEnv(render_mode=None)
    obs, _ = env.reset(seed=seed)

    runtime = ObservationRuntime(env)
    runtime.reset(obs, initial_inventory="empty")
    scorer = (
        QwenSkillRelevanceScorer()
        if os.getenv("DASHSCOPE_API_KEY")
        else KeywordRelevanceScorer()
    )
    planner = UnifiedSayCanPlanner(
        runtime,
        relevance_scorer=scorer,
        max_actions=450,
        max_skill_actions=300,
    )
    task = "Pick up the red ball, open the yellow door, and place it above the green goal."
    result = planner.run(task, max_high_level_steps=100)
    gt_success = ground_truth_success(env)

    print("Seed:", seed)
    print("Planner success:", result["success"])
    print("Ground-truth success:", gt_success)
    print("Status:", result["status"])
    print("High-level skill selections:", result["high_level_steps"])
    print("Low-level primitive actions:", result["low_level_actions"])
    print("Belief:", runtime.belief.summary())
    env.close()

    if not result["success"] or not gt_success:
        raise SystemExit(
            f"Task failed: planner_success={result['success']}, ground_truth_success={gt_success}"
        )
    if result["high_level_steps"] > 20:
        raise SystemExit(
            f"Skill calls are still fragmented: {result['high_level_steps']} selections for one task."
        )


if __name__ == "__main__":
    main()
