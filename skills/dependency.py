"""Task parsing and explicit skill prerequisite resolution.

The resolver never executes actions. It exposes prerequisite SkillCalls so a
planner can choose them as ordinary candidates and re-score after each result.
"""
import re
from typing import Optional

from skills.target import (
    PICKABLE_OBJECTS,
    TargetSpec,
    parse_target_specs,
    relation_offset,
)


def make_call(name: str, **args) -> dict:
    target = args.get("target")
    target_key = TargetSpec.from_value(target).key if target is not None else ""
    object_key = (
        TargetSpec.from_value(args["object"]).key if args.get("object") is not None else ""
    )
    destination_key = (
        TargetSpec.from_value(args["destination"]).key
        if args.get("destination") is not None and not isinstance(args["destination"], (tuple, list))
        else ""
    )
    relation = str(args.get("relation", ""))
    suffix = ":".join(part for part in (target_key, object_key, destination_key, relation) if part)
    call_id = name.lower() + (":" + suffix if suffix else "")
    return {
        "id": call_id,
        "name": name,
        "args": args,
        "description": _describe_call(name, args),
    }


def _describe_call(name: str, args: dict) -> str:
    if name in {"Find", "NavigateTo", "PickUp", "OpenDoor", "UnlockDoor", "Done"}:
        target = args.get("target")
        label = TargetSpec.from_value(target).display_name if target is not None else "target"
        verbs = {
            "Find": "Find",
            "NavigateTo": "Navigate to",
            "PickUp": "Pick up",
            "OpenDoor": "Open",
            "UnlockDoor": "Unlock",
            "Done": "Finish at",
        }
        return f"{verbs[name]} {label}"
    if name == "PlaceNear":
        obj = TargetSpec.from_value(args.get("object", "object")).display_name
        target = TargetSpec.from_value(args.get("target", "target")).display_name
        return f"Place {obj} {args.get('relation', 'near')} {target}"
    if name == "MoveObstacle":
        obj = TargetSpec.from_value(args.get("target", "box")).display_name
        destination = args.get("destination", "a safe location")
        if not isinstance(destination, (tuple, list)):
            destination = TargetSpec.from_value(destination).display_name
        return f"Move {obj} near {destination}"
    return "Explore the environment"


class TaskDependencyResolver:
    """Generates parameterized candidates and prerequisite chains."""

    def parse_targets(self, task: str, belief=None) -> list[TargetSpec]:
        targets = parse_target_specs(task)
        seen = {item.key for item in targets}

        if belief is not None:
            for _, cell in belief.known_map.items():
                object_name = cell.get("object", "unknown")
                if object_name in {"ball", "key", "door", "box", "goal"}:
                    spec = TargetSpec(
                        object_name=object_name,
                        color=cell.get("color") if cell.get("color") != "unknown" else None,
                        state=None,
                    )
                    if spec.key not in seen:
                        targets.append(spec)
                        seen.add(spec.key)

        text = str(task).lower()
        doors = [item for item in targets if item.object_name == "door"]
        locked_door_known = bool(
            belief is not None
            and belief.find_objects("door", state="locked")
        )
        if "unlock" in text or locked_door_known:
            for door in doors or [TargetSpec("door")]:
                key = TargetSpec("key", color=door.color)
                if key.key not in seen:
                    targets.append(key)
                    seen.add(key.key)

        return targets

    def candidate_calls(self, task: str, belief=None) -> list[dict]:
        text = str(task).lower()
        targets = self.parse_targets(task, belief)
        pick_verbs = any(word in text for word in ("pick", "take", "grab", "get", "carry", "fetch"))
        place_verbs = any(word in text for word in ("place", "put", "drop", "set down"))
        move_verbs = any(word in text for word in ("move", "push", "clear", "obstacle", "blocked"))
        unlock_intent = "unlock" in text or bool(
            belief is not None and belief.find_objects("door", state="locked")
        )
        done_intent = any(word in text for word in ("done", "finish", "complete", "go to the door", "reach the door"))

        calls = [make_call("Explore")]
        for target in targets:
            calls.append(make_call("Find", target=target.to_dict()))
            if any(word in text for word in ("go to", "navigate", "reach", "approach")) or target.object_name == "door":
                calls.append(make_call("NavigateTo", target=target.to_dict()))
            if target.object_name in PICKABLE_OBJECTS and (
                pick_verbs or (unlock_intent and target.object_name == "key")
            ):
                calls.append(make_call("PickUp", target=target.to_dict()))
            if target.object_name == "door":
                calls.append(make_call("OpenDoor", target=target.to_dict()))
                calls.append(make_call("UnlockDoor", target=target.to_dict()))
            if target.object_name == "box" and move_verbs:
                destinations = [item for item in targets if item.key != target.key and item.object_name not in PICKABLE_OBJECTS]
                if destinations:
                    calls.append(
                        make_call(
                            "MoveObstacle",
                            target=target.to_dict(),
                            destination=destinations[0].to_dict(),
                            relation="near",
                        )
                    )

        if place_verbs:
            movable = [item for item in targets if item.object_name in PICKABLE_OBJECTS]
            destinations = [item for item in targets if item.object_name not in PICKABLE_OBJECTS]
            relation = "near"
            for candidate in ("above", "below", "left_of", "right_of", "near"):
                if re.search(r"\b" + candidate.replace("_of", "") + r"\b", text):
                    relation = candidate
                    break
            if re.search(r"\b(left|right)\s+of\b", text):
                relation = "left_of" if re.search(r"\bleft\s+of\b", text) else "right_of"
            for obj in movable:
                for destination in destinations:
                    calls.append(
                        make_call(
                            "PlaceNear",
                            object=obj.to_dict(),
                            target=destination.to_dict(),
                            relation=relation,
                        )
                    )

        if done_intent:
            target = next((item for item in targets if item.object_name in {"door", "goal"}), None)
            if target is not None:
                calls.append(make_call("Done", target=target.to_dict()))

        # Stable de-duplication matters for deterministic tie-breaking.
        unique = {}
        for call in calls:
            unique.setdefault(call["id"], call)
        return list(unique.values())

    @staticmethod
    def _known_positions(belief, target) -> list[tuple[tuple[int, int], dict]]:
        spec = TargetSpec.from_value(target)
        return [
            (tuple(pos), cell)
            for pos, cell in belief.known_map.items()
            if spec.matches(cell)
        ]

    @staticmethod
    def _held_matches(belief, target) -> bool:
        held = getattr(belief, "held_object", None)
        if not held:
            return False
        return TargetSpec.from_value(target).matches(held)

    def prerequisites_for(self, call: dict, belief) -> list[dict]:
        """Return an ordered, deduplicated list of prerequisites."""
        name = call.get("name")
        args = call.get("args", {})
        target = args.get("target")
        prerequisites = []

        def add(name_, **kwargs):
            prerequisites.append(make_call(name_, **kwargs))

        positions = self._known_positions(belief, target) if target is not None else []
        if name in {"NavigateTo", "PickUp", "OpenDoor", "UnlockDoor", "Done"} and target is not None and not positions:
            add("Find", target=target)

        if name == "PickUp" and target is not None:
            if belief.inventory_status != "empty" and not self._held_matches(belief, target):
                return []
            if not positions:
                add("Find", target=target)

        if name in {"OpenDoor", "UnlockDoor"} and target is not None:
            door_pos = positions[0][0] if positions else None
            door_cell = positions[0][1] if positions else None
            if door_cell and door_cell.get("state") == "locked":
                key = {"object": "key", "color": door_cell.get("color")}
                if not self._held_matches(belief, key):
                    add("Find", target=key)
                    add("PickUp", target=key)
            if door_pos is not None:
                dx = door_pos[0] - belief.agent_pos[0]
                dy = door_pos[1] - belief.agent_pos[1]
                if abs(dx) + abs(dy) != 1:
                    add("NavigateTo", target=target)

        if name == "PlaceNear":
            obj = args.get("object", "object")
            destination = args.get("target")
            if belief.inventory_status == "empty" or not self._held_matches(belief, obj):
                add("Find", target=obj)
                add("PickUp", target=obj)
            if destination is not None and not self._known_positions(belief, destination):
                add("Find", target=destination)
            elif destination is not None:
                offset = relation_offset(args.get("relation", "near"))
                if offset is not None:
                    target_pos = self._known_positions(belief, destination)[0][0]
                    cell_pos = (target_pos[0] + offset[0], target_pos[1] + offset[1])
                    if belief.cell_at(cell_pos) is None:
                        add("Explore")

        if name == "MoveObstacle":
            obstacle = args.get("target")
            destination = args.get("destination")
            if obstacle is not None and not self._known_positions(belief, obstacle):
                add("Find", target=obstacle)
            if obstacle is not None and not self._held_matches(belief, obstacle):
                add("PickUp", target=obstacle)
            if destination is not None and not self._known_positions(belief, destination):
                add("Find", target=destination)
            if destination is not None:
                add("PlaceNear", object=obstacle, target=destination, relation=args.get("relation", "near"))

        if name == "Done" and target is not None and positions:
            target_pos = positions[0][0]
            if abs(target_pos[0] - belief.agent_pos[0]) + abs(target_pos[1] - belief.agent_pos[1]) != 1:
                add("NavigateTo", target=target)

        unique = {}
        for item in prerequisites:
            if item["id"] != call.get("id"):
                unique.setdefault(item["id"], item)
        return list(unique.values())
