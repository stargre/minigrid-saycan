"""Generic terminal interaction skill."""
from minigrid.core.actions import Actions
from skills.base import UnifiedSkill


class DoneSkill(UnifiedSkill):
    name = "Done"

    def step(self, target=None, **kwargs):
        if target is not None:
            pos, _ = self._nearest(target)
            if pos is None:
                return self._result(False, "target_not_found", "Find the target before completing the task.")
            result = self.navigator.step_to_adjacent(pos)
            if result.get("status") != "target_adjacent":
                result["skill"] = self.name
                return result
        result = self.runtime.execute_action(Actions.done)
        # MiniGrid's done action may end the episode; task-level success remains
        # a separate predicate owned by the environment adapter/evaluator.
        return self._result(True, "done_issued", "The required MiniGrid done action was issued.", actions_executed=result["actions_executed"])
