"""Observation-grounded features for predicting parameterized skill success."""
from __future__ import annotations

import hashlib
import json
import math

from perception.belief_state import DIRECTIONS, is_walkable
from skills.target import TargetSpec


SKILLS = (
    "Explore", "Find", "NavigateTo", "Reach", "PickUp", "OpenDoor",
    "UnlockDoor", "PlaceNear", "MoveObstacle", "Done",
)
OBJECTS = ("empty", "wall", "door", "key", "ball", "box", "goal", "lava")
COLORS = ("red", "blue", "green", "purple", "yellow", "grey", "unknown")
DOOR_STATES = ("open", "closed", "locked")
HELD_OBJECTS = ("none", "key", "ball", "box")
COMPLETED = ("pick", "pickup", "drop", "place", "unlock_door", "open_door", "reach")


class AffordanceFeatureEncoder:
    """Encode only BeliefState and a SkillCall; never simulator ground truth."""

    schema_version = "minigrid_affordance_features_v1"

    def __init__(self):
        names = [
            "log_known_cells", "frontier_ratio", "walkable_ratio",
            "gripper_empty", "target_known", "target_count",
            "target_nearest_distance", "target_in_front", "position_supplied",
            "position_observed", "destination_known", "memory_probe",
            "grounded_reference", "reactive_obstacles",
        ]
        names += [f"direction:{item}" for item in range(4)]
        names += [f"object_count:{item}" for item in OBJECTS]
        names += [f"door_state_count:{item}" for item in DOOR_STATES]
        names += [f"color_count:{item}" for item in COLORS]
        names += [f"held_object:{item}" for item in HELD_OBJECTS]
        names += [f"held_color:{item}" for item in COLORS]
        names += [f"completed:{item}" for item in COMPLETED]
        names += [f"skill:{item}" for item in SKILLS]
        names += [f"target_object:{item}" for item in OBJECTS + ("object", "unknown")]
        names += [f"target_color:{item}" for item in COLORS]
        names += [f"target_state:{item}" for item in DOOR_STATES + ("unknown",)]
        self.feature_names = tuple(names)
        payload = json.dumps(self.feature_names, separators=(",", ":"))
        self.schema_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def dimension(self):
        return len(self.feature_names)

    @staticmethod
    def _normalized_count(value):
        return min(1.0, math.log1p(max(0, int(value))) / math.log(33.0))

    @staticmethod
    def _target(call):
        value = call.get("args", {}).get("target")
        if value is None:
            return TargetSpec("unknown")
        try:
            return TargetSpec.from_value(value)
        except (TypeError, ValueError):
            return TargetSpec("unknown")

    def encode(self, belief, call):
        known_map = getattr(belief, "known_map", {})
        agent_pos = tuple(getattr(belief, "agent_pos", (0, 0)))
        agent_dir = int(getattr(belief, "agent_dir", 0)) % 4
        args = call.get("args", {})
        target = self._target(call)
        target_positions = [
            tuple(pos) for pos, cell in known_map.items() if target.matches(cell)
        ]
        nearest = min(
            (abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1])
             for pos in target_positions),
            default=32,
        )
        dx, dy = DIRECTIONS[agent_dir]
        front_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
        position = args.get("position")
        requested_pos = None
        if isinstance(position, (tuple, list)) and len(position) == 2:
            requested_pos = (int(position[0]), int(position[1]))

        frontiers = 0
        walkable = 0
        object_counts = {name: 0 for name in OBJECTS}
        state_counts = {name: 0 for name in DOOR_STATES}
        color_counts = {name: 0 for name in COLORS}
        for pos, cell in known_map.items():
            obj = cell.get("object", "unknown")
            color = cell.get("color", "unknown")
            if obj in object_counts:
                object_counts[obj] += 1
            color_counts[color if color in color_counts else "unknown"] += 1
            if obj == "door" and cell.get("state") in state_counts:
                state_counts[cell.get("state")] += 1
            if is_walkable(cell):
                walkable += 1
                if any(
                    (pos[0] + ox, pos[1] + oy) not in known_map
                    for ox, oy in DIRECTIONS
                ):
                    frontiers += 1

        held = getattr(belief, "held_object", None) or {}
        held_object = held.get("object", "none")
        held_color = held.get("color", "unknown")
        destination = args.get("destination")
        destination_known = False
        if destination is not None and not isinstance(destination, (tuple, list)):
            try:
                dest_spec = TargetSpec.from_value(destination)
                destination_known = any(dest_spec.matches(cell) for cell in known_map.values())
            except (TypeError, ValueError):
                destination_known = False

        values = [
            min(1.0, math.log1p(len(known_map)) / math.log(257.0)),
            frontiers / max(1, walkable),
            walkable / max(1, len(known_map)),
            float(getattr(belief, "inventory_status", "unknown") == "empty"),
            float(bool(target_positions)),
            self._normalized_count(len(target_positions)),
            min(1.0, nearest / 32.0),
            float(front_pos in target_positions),
            float(requested_pos is not None),
            float(requested_pos in known_map if requested_pos is not None else False),
            float(destination_known),
            float(bool(args.get("memory_probe"))),
            float(bool(args.get("grounded_reference"))),
            float(bool(args.get("reactive_obstacles"))),
        ]
        values += [float(agent_dir == item) for item in range(4)]
        values += [self._normalized_count(object_counts[item]) for item in OBJECTS]
        values += [self._normalized_count(state_counts[item]) for item in DOOR_STATES]
        values += [self._normalized_count(color_counts[item]) for item in COLORS]
        values += [float(held_object == item) for item in HELD_OBJECTS]
        values += [float(held_color == item) for item in COLORS]
        completed = set(getattr(belief, "completed_subgoals", []))
        values += [float(item in completed) for item in COMPLETED]
        values += [float(call.get("name") == item) for item in SKILLS]
        target_objects = OBJECTS + ("object", "unknown")
        values += [float(target.object_name == item) for item in target_objects]
        target_color = target.color or "unknown"
        values += [float(target_color == item) for item in COLORS]
        target_state = target.state or "unknown"
        values += [float(target_state == item) for item in DOOR_STATES + ("unknown",)]
        if len(values) != self.dimension:
            raise AssertionError(f"feature dimension mismatch: {len(values)} != {self.dimension}")
        return values

    def metadata(self):
        return {
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "feature_names": list(self.feature_names),
            "dimension": self.dimension,
        }
