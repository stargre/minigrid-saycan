"""Parameterized pickup, placement, and obstacle-movement skills."""
from minigrid.core.actions import Actions
from perception.belief_state import DIRECTIONS
from skills.base import UnifiedSkill, _dist
from skills.target import PICKABLE_OBJECTS, TargetSpec, canonical_object
from skills.placement import select_placement


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

    def __init__(self, runtime):
        super().__init__(runtime)
        # A placement decision is a skill-level commitment, not something to
        # recompute from the agent's changing position on every controller tick.
        self._active_plan_key = None
        self._active_destination_pos = None
        self._active_placement_pos = None

    def _reset_plan(self, reset_navigator=False):
        self._active_plan_key = None
        self._active_destination_pos = None
        self._active_placement_pos = None
        if reset_navigator:
            self.navigator.reset_target()

    def step(self, object, target, relation="near", **kwargs):
        obj_spec = TargetSpec.from_value(object)
        held = getattr(self.belief, "held_object", None)
        if not held or not obj_spec.matches(held):
            self._reset_plan(reset_navigator=True)
            return self._result(False, "not_holding_object", "Pick up the requested object before placing it.")

        target_spec = TargetSpec.from_value(target)
        destinations = self._positions(target)
        if not destinations:
            self._reset_plan(reset_navigator=True)
            return self._result(False, "target_not_found", "Find the destination object before placing.")

        relation_norm = str(relation).strip().lower().replace("-", "_")
        plan_key = (obj_spec.key, target_spec.key, relation_norm)

        # Keep the same destination and placement cell for the lifetime of this
        # PlaceNear call. Re-select only when the task changes or the committed
        # cells are no longer supported by the current belief.
        if self._active_plan_key != plan_key:
            self._reset_plan(reset_navigator=True)
            self._active_plan_key = plan_key

        dest_pos = self._active_destination_pos
        placement_pos = self._active_placement_pos

        if placement_pos is not None:
            destination_cell = self.belief.cell_at(dest_pos)
            placement_cell = self.belief.cell_at(placement_pos)
            destination_valid = target_spec.matches(destination_cell)
            placement_valid = bool(
                placement_cell
                and placement_cell.get("object") in {"empty", "floor"}
            )
            if not destination_valid or not placement_valid:
                self._reset_plan(reset_navigator=True)
                self._active_plan_key = plan_key
                dest_pos = None
                placement_pos = None

        if placement_pos is None:
            dest_pos, placement_pos, placement_error = select_placement(
                self.belief,
                self.navigator,
                [pos for pos, _ in destinations],
                relation_norm,
            )
            if placement_error == "unsupported_relation":
                return self._result(False, placement_error, f"Unsupported relation: {relation}")
            if placement_error == "placement_cell_unknown":
                return self._result(False, placement_error, "No known empty placement cell is available for the requested relation.")
            if placement_error == "placement_cell_occupied":
                return self._result(False, placement_error, "The relation-derived placement cell is occupied.")
            if placement_error == "no_known_path":
                return self._result(False, placement_error, "Known empty placement cells have no reachable adjacent approach.")

            dest_pos = tuple(dest_pos)
            placement_pos = tuple(placement_pos)
            self._active_destination_pos = dest_pos
            self._active_placement_pos = placement_pos
        else:
            dest_pos = tuple(dest_pos)
            placement_pos = tuple(placement_pos)

        result = self.navigator.step_to_adjacent(placement_pos)
        if result.get("status") != "target_adjacent":
            if result.get("status") in {"no_known_path", "target_not_known"}:
                # Permit a fresh choice after the planner obtains new evidence.
                self._reset_plan(reset_navigator=True)
            result["skill"] = self.name
            result["placement_position"] = list(placement_pos)
            result["destination_position"] = list(dest_pos)
            return result

        action = self.runtime.execute_action(Actions.drop)
        after = self.belief.cell_at(placement_pos)
        held_after = getattr(self.belief, "held_object", None)
        success = bool(after and obj_spec.matches(after) and not held_after)
        if success:
            # Only the skill can confirm that the object occupies the
            # relation-derived cell after release; a generic Drop is not enough.
            self.belief._mark_completed("place")
            self._reset_plan(reset_navigator=True)
        return self._result(
            success, "skill_complete" if success else "placement_not_confirmed",
            "Placement and object release confirmed by observation." if success else "The target relation has not been confirmed by observation.",
            actions_executed=action["actions_executed"], placement_position=list(placement_pos),
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
