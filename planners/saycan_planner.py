"""Backward-compatible entry point backed by the observation-grounded planner.

New experiments should import UnifiedSayCanPlanner directly. This adapter keeps
the historical constructor and (success, history) return shape while removing
all reads of environment-internal positions and grid cells.
"""
from skills.observation_runtime import ObservationRuntime
from planners.unified_saycan_planner import UnifiedSayCanPlanner


class SayCanPlanner:
    def __init__(self, env, movement=None, picker=None, opener=None, placer=None, relevance_scorer=None):
        if isinstance(env, ObservationRuntime):
            self.runtime = env
        else:
            self.runtime = ObservationRuntime(env)
            obs = env.unwrapped.gen_obs()
            self.runtime.reset(obs, initial_inventory="empty")
        self.planner = UnifiedSayCanPlanner(self.runtime, relevance_scorer=relevance_scorer)
        self.library = self.planner.library

    def is_task_complete(self):
        done = set(self.runtime.belief.completed_subgoals)
        return (
            "place" in done
            and self.runtime.belief.inventory_status == "empty"
            and ("open_door" in done or "unlock_door" in done)
        )

    def run(self, task, max_high_level_steps=100):
        result = self.planner.run(task, max_high_level_steps=max_high_level_steps)
        return bool(result["success"]), result["history"]
