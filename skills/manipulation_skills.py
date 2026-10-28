"""Parameterized pickup, placement, and obstacle-movement skills."""
from minigrid.core.actions import Actions
from perception.belief_state import DIRECTIONS
from skills.base import UnifiedSkill, _dist
from skills.target import PICKABLE_OBJECTS, TargetSpec, canonical_object, relation_offset


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
        target_spec = TargetSpec.from_value(target)
        key = (
            target_spec.key,
            TargetSpec.from_value(destination).key,
            str(relation).strip().lower(),
        )
        if key != self._active_key:
            self._active_key, self._phase = key, "pickup"

        # The belief state is the source of truth for the phase transition.
        # A stale phase flag must never make MoveObstacle skip a required pickup.
        held = getattr(self.belief, "held_object", None)
        holding_target = bool(held and target_spec.matches(held))
        if self._phase == "place" and not holding_target:
            self._phase = "pickup"

        if self._phase == "pickup":
            result = self.pickup.step(target)
            held_after = getattr(self.belief, "held_object", None)
            holding_target_after = bool(
                held_after and target_spec.matches(held_after)
            )
            child_status = result.get("status")

            # A navigator tick may return success=True with
            # status=action_executed. That means only that a primitive
            # action ran; it does not mean the pickup skill has completed.
            if child_status == "skill_complete" and holding_target_after:
                self._phase = "place"
                result["success"] = False
                result["status"] = "phase_complete"
                result["reason"] = (
                    "Target pickup is confirmed by the belief state; "
                    "the next invocation will place it."
                )
            elif child_status == "skill_complete" and not holding_target_after:
                # Fail closed only when the child claims the entire pickup
                # skill is complete but belief does not confirm possession.
                result["success"] = False
                result["status"] = "pickup_not_confirmed"
                result["reason"] = (
                    "PickUp reported skill_complete, but the target is not "
                    "present in the current held-object belief."
                )
            result["skill"] = self.name
            result["phase"] = "pickup"
            return result

        result = self.place.step(
            object=target,
            target=destination,
            relation=relation,
        )
        if result.get("success"):
            self._phase = "pickup"
            self._active_key = None
        elif result.get("status") == "not_holding_object":
            # Resynchronize on the next controller tick instead of retaining
            # a place phase that contradicts the observed inventory.
            self._phase = "pickup"
        result["skill"] = self.name
        result["phase"] = "place"
        return result
