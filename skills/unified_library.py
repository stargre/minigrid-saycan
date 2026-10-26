"""Observation-grounded reusable skill implementations.

Every public call executes at most one MiniGrid primitive action. A skill keeps
its internal progress and reports success only after observing evidence.
"""
from __future__ import annotations

from typing import Any

from minigrid.core.actions import Actions
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

from perception.belief_state import BeliefState, DIRECTIONS
from perception.observation_summary import observation_to_belief
from skills.approach_door import ApproachDoorSkill
from skills.exploration import ExploreSkill
from skills.explore_for_door import ExploreForDoorSkill
from skills.interaction import OpenDoorSkill
from skills.manipulation import PickSkill, PlaceSkill
from skills.navigation import ObservationNavigator
from skills.observation_explorer import ObservationExplorer
from skills.target import (
    PICKABLE_OBJECTS, TargetSpec, canonical_object, relation_offset,
)


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class UnifiedSkill:
    name = "Skill"

    def __init__(self, runtime):
        self.runtime = runtime
        self.belief = runtime.belief
        self.navigator = ObservationNavigator(runtime)
        self.max_internal_steps = 1

    def _result(self, success, status, reason, **kwargs):
        return {
            "success": bool(success),
            "status": status,
            "reason": reason,
            "skill": self.name,
            "actions_executed": int(kwargs.pop("actions_executed", 0)),
            **kwargs,
        }

    def _positions(self, target):
        spec = TargetSpec.from_value(target)
        return [
            (tuple(pos), cell)
            for pos, cell in self.belief.known_map.items()
            if spec.matches(cell)
        ]

    def _nearest(self, target):
        items = self._positions(target)
        if not items:
            return None, None
        return min(items, key=lambda item: _dist(item[0], self.belief.agent_pos))


class ExploreSkill(UnifiedSkill):
    name = "Explore"

    def __init__(self, runtime):
        super().__init__(runtime)
        self.explorer = ObservationExplorer(runtime)

    def step(self, **kwargs):
        result = self.explorer.step()
        result["skill"] = self.name
        return result


class FindSkill(UnifiedSkill):
    name = "Find"

    def step(self, target, **kwargs):
        spec = TargetSpec.from_value(target)
        pos, cell = self._nearest(spec.to_dict())
        if pos is not None:
            return self._result(
                True, "target_found",
                "Target is supported by observation evidence in the belief map.",
                target=spec.to_dict(), evidence={"position": list(pos), "cell": dict(cell)},
            )
        result = self.explorer.step()
        result["skill"] = self.name
        result["target"] = spec.to_dict()
        if result.get("status") == "exploration_exhausted":
            result["success"] = False
            result["status"] = "target_not_found"
            result["reason"] = "No reachable unexplored frontier remains; target not found."
        elif result.get("status") == "action_executed":
            result["success"] = False
            result["status"] = "searching"
        return result

    def __init__(self, runtime):
        super().__init__(runtime)
        self.explorer = ObservationExplorer(runtime)


class NavigateToSkill(UnifiedSkill):
    name = "NavigateTo"

    def step(self, target, **kwargs):
        pos, cell = self._nearest(target)
        if pos is None:
            return self._result(False, "target_not_found", "Find the target before navigating.")
        if _dist(pos, self.belief.agent_pos) == 1:
            desired = DIRECTIONS.index((pos[0] - self.belief.agent_pos[0], pos[1] - self.belief.agent_pos[1]))
            if self.belief.agent_dir != desired:
                result = self.navigator.step_to_adjacent(pos)
                result["skill"] = self.name
                return result
            return self._result(True, "skill_complete", "The agent is adjacent to and facing the target.", target_position=list(pos))
        result = self.navigator.step_to_adjacent(pos)
        result["skill"] = self.name
        return result


class PickUpSkill(UnifiedSkill):
    name = "PickUp"

    def step(self, target, **kwargs):
        spec = TargetSpec.from_value(target)
        held = getattr(self.belief, "held_object", None)
        if held and spec.matches(held):
            return self._result(True, "skill_complete", "The target is already held.", held_object=held)
        if self.belief.inventory_status != "empty":
            return self._result(False, "gripper_not_empty", "The gripper is not confirmed empty.")
        pos, cell = self._nearest(spec.to_dict())
        if pos is None:
            return self._result(False, "target_not_found", "The target must be observed before pickup.")
        if canonical_object(cell.get("object")) not in PICKABLE_OBJECTS:
            return self._result(False, "not_pickable", "MiniGrid only permits pickup of supported movable objects.")
        result = self.navigator.step_to_adjacent(pos)
        if result.get("status") != "target_adjacent":
            result["skill"] = self.name
            return result
        action = self.runtime.execute_action(Actions.pickup)
        held = getattr(self.belief, "held_object", None)
        success = bool(held and spec.matches(held))
        return self._result(
            success, "skill_complete" if success else "pickup_not_confirmed",
            "Pickup confirmed by the updated observation." if success else "Pickup was not confirmed by the updated belief.",
            actions_executed=action["actions_executed"], held_object=held,
            target_position=list(pos),
        )


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


class PlaceNearSkill(UnifiedSkill):
    name = "PlaceNear"

    def step(self, object, target, relation="near", **kwargs):
        obj_spec = TargetSpec.from_value(object)
        held = getattr(self.belief, "held_object", None)
        if not held or not obj_spec.matches(held):
            return self._result(False, "not_holding_object", "Pick up the requested object before placing it.")
        dest_pos, _ = self._nearest(target)
        if dest_pos is None:
            return self._result(False, "target_not_found", "Find the destination object before placing.")
        relation_norm = str(relation).strip().lower().replace("-", "_")
        offset = relation_offset(relation_norm)
        if offset is None and relation_norm in {"near", "adjacent", "nearby", "beside", "next_to"}:
            options = [(dest_pos[0] + dx, dest_pos[1] + dy) for dx, dy in DIRECTIONS]
            candidates = [
                pos for pos in options
                if self.belief.cell_at(pos) is not None
                and self.belief.cell_at(pos).get("object") in {"empty", "floor"}
            ]
            if not candidates:
                return self._result(False, "placement_cell_unknown", "No known empty cell adjacent to the destination.")
            target_pos = min(candidates, key=lambda p: _dist(p, self.belief.agent_pos))
        elif offset is not None:
            target_pos = (dest_pos[0] + offset[0], dest_pos[1] + offset[1])
        else:
            return self._result(False, "unsupported_relation", f"Unsupported relation: {relation}")
        cell = self.belief.cell_at(target_pos)
        if cell is None:
            return self._result(False, "placement_cell_unknown", "The relation-derived placement cell has not been observed.")
        if cell.get("object") not in {"empty", "floor"}:
            return self._result(False, "placement_cell_occupied", "The placement cell is not known to be empty.")
        result = self.navigator.step_to_adjacent(target_pos)
        if result.get("status") != "target_adjacent":
            result["skill"] = self.name
            return result
        action = self.runtime.execute_action(Actions.drop)
        after = self.belief.cell_at(target_pos)
        held_after = getattr(self.belief, "held_object", None)
        success = bool(after and obj_spec.matches(after) and not held_after)
        return self._result(
            success, "skill_complete" if success else "placement_not_confirmed",
            "Placement and object release confirmed by observation." if success else "The target relation has not been confirmed by observation.",
            actions_executed=action["actions_executed"], placement_position=list(target_pos),
            destination_position=list(dest_pos), relation=relation_norm,
        )


class MoveObstacleSkill(UnifiedSkill):
    name = "MoveObstacle"

    def __init__(self, runtime):
        super().__init__(runtime)
        self.pickup = PickUpSkill(runtime)
        self.place = PlaceNearSkill(runtime)
        self._phase = "pickup"
        self._active_key = None

    def step(self, target, destination, relation="near", **kwargs):
        key = (TargetSpec.from_value(target).key, TargetSpec.from_value(destination).key, relation)
        if key != self._active_key:
            self._active_key, self._phase = key, "pickup"
        if self._phase == "pickup":
            result = self.pickup.step(target)
            if result.get("success"):
                self._phase = "place"
            result["skill"] = self.name
            result["phase"] = "pickup"
            return result
        result = self.place.step(object=target, target=destination, relation=relation)
        if result.get("success"):
            self._phase = "pickup"
            self._active_key = None
        result["skill"] = self.name
        result["phase"] = "place"
        return result


class DoneSkill(UnifiedSkill):
    name = "Done"

    def step(self, target=None, **kwargs):
        if target is not None:
            pos, _ = self._nearest(target)
            if pos is None:
                return self._result(False, "target_not_found", "Find the target before completing the task.")
            if _dist(pos, self.belief.agent_pos) != 1:
                result = self.navigator.step_to_adjacent(pos)
                result["skill"] = self.name
                return result
        result = self.runtime.execute_action(Actions.done)
        # MiniGrid's done action may end the episode; task-level success remains
        # a separate predicate owned by the environment adapter/evaluator.
        return self._result(True, "done_issued", "The required MiniGrid done action was issued.", actions_executed=result["actions_executed"])


class SkillLibraryV1:
    """Frozen skill registry and uniform execution contract.

    Contract: execute(call) -> {success, status, reason, skill,
    actions_executed, ...}. LLMs never issue primitive MiniGrid actions.
    """

    VERSION = "skill_library_v1"

    def __init__(self, runtime):
        self.runtime = runtime
        self.skills = {
            "Explore": ExploreSkill(runtime),
            "Find": FindSkill(runtime),
            "NavigateTo": NavigateToSkill(runtime),
            "PickUp": PickUpSkill(runtime),
            "OpenDoor": OpenDoorSkill(runtime),
            "UnlockDoor": UnlockDoorSkill(runtime),
            "PlaceNear": PlaceNearSkill(runtime),
            "MoveObstacle": MoveObstacleSkill(runtime),
            "Done": DoneSkill(runtime),
        }

    @property
    def names(self):
        return tuple(self.skills.keys())

    def execute(self, call: dict) -> dict:
        if not isinstance(call, dict) or "name" not in call:
            raise TypeError("call must be a mapping containing 'name'")
        name = str(call["name"])
        if name not in self.skills:
            return {
                "success": False, "status": "unknown_skill",
                "reason": f"Unknown skill: {name}", "skill": name,
                "actions_executed": 0,
            }
        args = dict(call.get("args", {}))
        try:
            result = self.skills[name].step(**args)
        except (TypeError, ValueError) as exc:
            return {
                "success": False, "status": "invalid_arguments",
                "reason": str(exc), "skill": name, "actions_executed": 0,
            }
        result.setdefault("skill", name)
        result.setdefault("actions_executed", 0)
        result.setdefault("success", False)
        result.setdefault("status", "unknown")
        result.setdefault("reason", "")
        return result
