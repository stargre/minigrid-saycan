"""High-level navigation skills distinct from primitive path control."""
from skills.base import UnifiedSkill
from perception.belief_state import is_walkable
from skills.target import TargetSpec


class NavigateToSkill(UnifiedSkill):
    name = "NavigateTo"

    def step(self, target, **kwargs):
        pos, cell = self._nearest(target)
        if pos is None:
            return self._result(False, "target_not_found", "Find the target before navigating.")
        result = self.navigator.step_to_adjacent(pos)
        if result.get("status") == "target_adjacent":
            return self._result(True, "skill_complete", "The agent is adjacent to and facing the target.", target_position=list(pos))
        result["skill"] = self.name
        return result


class ReachSkill(UnifiedSkill):
    """Navigate onto a known walkable target cell, e.g. a green goal tile."""

    name = "Reach"

    def __init__(self, runtime):
        super().__init__(runtime)
        self._active_target_key = None
        self._active_target_pos = None

    def step(self, target, **kwargs):
        spec = TargetSpec.from_value(target)
        if spec.key != self._active_target_key:
            self._active_target_key = spec.key
            self._active_target_pos = None

        if self._active_target_pos is None:
            pos, cell = self._nearest(spec.to_dict())
            if pos is None:
                return self._result(
                    False, "target_not_found",
                    "Find the target before attempting to reach its cell.",
                )
            if not is_walkable(cell):
                return self._result(
                    False, "target_not_walkable",
                    "Reach requires a known walkable target cell; use NavigateTo for objects such as doors.",
                    target_position=list(pos),
                )
            self._active_target_pos = tuple(pos)

        target_pos = tuple(self._active_target_pos)
        if tuple(self.belief.agent_pos) == target_pos:
            self.belief._mark_completed("reach")
            self._active_target_pos = None
            self._active_target_key = None
            return self._result(
                True, "skill_complete",
                "The agent reached the target cell according to its belief-state position.",
                target_position=list(target_pos),
            )

        result = self.navigator.step_towards_cell(
            target_pos,
            allow_goal=True,
        )
        if tuple(self.belief.agent_pos) == target_pos:
            self.belief._mark_completed("reach")
            self._active_target_pos = None
            self._active_target_key = None
            return self._result(
                True, "skill_complete",
                "The agent reached the target cell; the position was tracked from observations and action history.",
                actions_executed=result.get("actions_executed", 0),
                target_position=list(target_pos),
                terminated=self.runtime.terminated,
                reward=self.runtime.last_reward,
            )

        result["skill"] = self.name
        result["target_position"] = list(target_pos)
        return result
