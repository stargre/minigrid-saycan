"""Integration test for the unified observation-grounded SayCan planner."""
import os
from environments.saycan_grid import SayCanGridEnv
from skills.observation_runtime import ObservationRuntime
from planners.unified_saycan_planner import UnifiedSayCanPlanner
from llm.skill_relevance import KeywordRelevanceScorer, QwenSkillRelevanceScorer


def main():
    seed = int(os.getenv("SEED", "42"))
    env = SayCanGridEnv(render_mode=None)
    obs, _ = env.reset(seed=seed)
    runtime = ObservationRuntime(env)
    runtime.reset(obs, initial_inventory="empty")
    scorer = QwenSkillRelevanceScorer() if os.getenv("DASHSCOPE_API_KEY") else KeywordRelevanceScorer()
    planner = UnifiedSayCanPlanner(runtime, relevance_scorer=scorer, max_actions=450)
    task = "Pick up the red ball, open the yellow door, and place it above the green goal."
    result = planner.run(task, max_high_level_steps=500)
    print("Seed:", seed)
    print("Success:", result["success"])
    print("Status:", result["status"])
    print("High-level steps:", result["high_level_steps"])
    print("Low-level actions:", result["low_level_actions"])
    print("Belief:", runtime.belief.summary())
    env.close()
    if not result["success"]:
        raise SystemExit(f"Planner did not finish the task: {result['status']}")


if __name__ == "__main__":
    main()
