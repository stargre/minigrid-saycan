"""Door interaction skills."""
from minigrid.core.actions import Actions
from skills.base import UnifiedSkill
from skills.target import TargetSpec


class OpenDoorSkill(UnifiedSkill):
    name = "OpenDoor"

    def step(self, target, **kwargs):
        pos, cell = self._nearest(target)
        if pos is None:
            return self._result(False, "target_not_found", "Find the door before opening it.")
        state = cell.get("state", "unknown")
        if state == "open":
            return self._result(True, "skill_complete", "The door is already observed open.")
        if state == "locked":
            return self._result(False, "requires_key", "The door is locked; find and pick up its matching key, then call UnlockDoor.")
        if state != "closed":
            return self._result(False, "unknown_door_state", "Door state is not known to be closed.")
        result = self.navigator.step_to_adjacent(pos)
        if result.get("status") != "target_adjacent":
            result["skill"] = self.name
            return result
        action = self.runtime.execute_action(Actions.toggle)
        new_cell = self.belief.cell_at(pos)
        success = bool(new_cell and new_cell.get("state") == "open")
        return self._result(
            success, "skill_complete" if success else "open_not_confirmed",
            "Door opening confirmed by observation." if success else "The toggle action did not produce observable open-door evidence.",
            actions_executed=action["actions_executed"], target_position=list(pos),
        )


class UnlockDoorSkill(UnifiedSkill):
    name = "UnlockDoor"

    def step(self, target, **kwargs):
        pos, cell = self._nearest(target)
        if pos is None:
            return self._result(False, "target_not_found", "Find the door before unlocking it.")
        state = cell.get("state", "unknown")
        if state == "open":
            return self._result(True, "skill_complete", "The door is already open.")
        if state != "locked":
            return self._result(False, "not_locked", "The observed door is not locked.")
        held = getattr(self.belief, "held_object", None)
        required_key = {"object": "key", "color": cell.get("color")}
        if not held or not TargetSpec.from_value(required_key).matches(held):
            return self._result(False, "requires_key", "A matching key must be found and picked up first.")
        result = self.navigator.step_to_adjacent(pos)
        if result.get("status") != "target_adjacent":
            result["skill"] = self.name
            return result
        action = self.runtime.execute_action(Actions.toggle)
        new_cell = self.belief.cell_at(pos)
        success = bool(new_cell and new_cell.get("state") in {"open", "closed"})
        return self._result(
            success, "skill_complete" if success else "unlock_not_confirmed",
            "The lock interaction was confirmed by the updated observation." if success else "Unlocking was not confirmed by the updated observation.",
            actions_executed=action["actions_executed"], target_position=list(pos),
        )
