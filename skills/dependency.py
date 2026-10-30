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
        text = str(task).lower()

        # Official Fetch missions name exactly one color/type target, while
        # the observation may contain several distractor keys and balls.
        # Restrict Fetch candidate generation to the explicitly requested
        # pickable target before adding any objects from the belief map.
        fetch_intent = (
            "fetch" in text
            or bool(re.search(
                r"\bget\s+(?:a|the)\s+"
                r"(?:(?:red|blue|green|purple|yellow|grey|gray)\s+)?"
                r"(?:key|ball)\b",
                text,
            ))
        )
        # Official GoToObject missions ask the agent to approach a single
        # color/type target and issue Done; pickup is explicitly not part of
        # this task. Restrict candidates to that mission target so observed
        # distractor keys, balls, or boxes cannot become alternative goals.
        explicit_object_targets = [
            item for item in targets
            if item.object_name in PICKABLE_OBJECTS and item.color
        ]
        acquisition_intent = (
            any(word in text for word in ("pick", "take", "grab", "carry", "fetch"))
            or bool(re.search(r"\bget\s+(?:a|an|the)\s+", text))
        )
        object_movement_intent = any(
            phrase in text
            for phrase in (
                "go to", "navigate", "reach", "approach", "enter",
                "arrive at", "get to", "cross to",
            )
        )
        goto_object_intent = (
            object_movement_intent
            and not acquisition_intent
            and len(explicit_object_targets) == 1
        )
        if goto_object_intent:
            return explicit_object_targets

        if fetch_intent:
            explicit_fetch_targets = [
                item for item in targets
                if item.object_name in PICKABLE_OBJECTS
            ]
            if explicit_fetch_targets:
                return explicit_fetch_targets

        seen = {item.key for item in targets}

        if belief is not None:
            # Explicitly named colors constrain candidates of the same object
            # type. Other observed objects remain in the belief map, but must
            # not become selectable targets for a task such as "go to the
            # green door" merely because a blue door is also visible.
            explicit_colors = {}
            for item in targets:
                if item.color:
                    explicit_colors.setdefault(item.object_name, set()).add(item.color)

            for _, cell in belief.known_map.items():
                object_name = cell.get("object", "unknown")
                if object_name in {"ball", "key", "door", "box", "goal"}:
                    color = cell.get("color") if cell.get("color") != "unknown" else None
                    requested_colors = explicit_colors.get(object_name, set())
                    if requested_colors and color not in requested_colors:
                        continue
                    spec = TargetSpec(
                        object_name=object_name,
                        color=color,
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
        pick_verbs = (
            any(word in text for word in ("pick", "take", "grab", "carry", "fetch"))
            or bool(re.search(r"\bget\s+(?:a|an|the)\s+", text))
        )
        place_verbs = bool(re.search(r"\b(?:place|put|drop|set\s+down)\b", text))
        move_verbs = any(word in text for word in ("move", "push", "clear", "obstacle", "blocked"))
        unlock_intent = "unlock" in text or bool(
            belief is not None and belief.find_objects("door", state="locked")
        )
        movement_intent = any(
            phrase in text
            for phrase in ("go to", "navigate", "reach", "approach", "enter", "arrive at", "get to", "cross to")
        )
        # GoToDoor missions use phrases such as "go to the red door".
        # They require the terminal Done action, not opening the door.
        object_navigation_intent = (
            movement_intent
            and not pick_verbs
            and any(target.object_name in PICKABLE_OBJECTS for target in targets)
        )
        done_intent = (
            any(word in text for word in ("done", "finish", "complete", "go to the door", "reach the door"))
            or (
                movement_intent
                and any(target.object_name == "door" for target in targets)
            )
            or object_navigation_intent
        )
        goal_entry_intent = any(
            phrase in text
            for phrase in ("reach", "enter", "arrive at", "get to", "go to", "navigate to", "cross to")
        )

        calls = [make_call("Explore")]
        for target in targets:
            calls.append(make_call("Find", target=target.to_dict()))
            if target.object_name == "goal" and goal_entry_intent:
                calls.append(make_call("Reach", target=target.to_dict()))
            elif movement_intent or target.object_name == "door":
                calls.append(make_call("NavigateTo", target=target.to_dict()))
            if target.object_name in PICKABLE_OBJECTS and (
                pick_verbs or (unlock_intent and target.object_name == "key")
            ):
                calls.append(make_call("PickUp", target=target.to_dict()))
            # Only expose door-manipulation candidates when the task asks
            # for opening/unlocking, or observation evidence shows a locked door.
            # A GoToDoor benchmark terminates on toggle, so opening its target is wrong.
            if target.object_name == "door" and (
                "open" in text or unlock_intent
            ):
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
            task_targets = parse_target_specs(task)
            place_matches = list(re.finditer(
                r"\b(?:place|put|drop|set\s+down)\b", text
            ))
            place_match = place_matches[-1] if place_matches else None

            # Resolve the relation and destination relative to the placement
            # verb. This avoids confusing an earlier door/key mention with
            # the actual placement destination in compound instructions.
            relation_patterns = (
                (r"\bon\s+top\s+of\b", "above"),
                (r"\babove\b", "above"),
                (r"\b(?:below|under|beneath)\b", "below"),
                (r"\bleft\s+of\b", "left_of"),
                (r"\bright\s+of\b", "right_of"),
                (r"\b(?:near|beside|next\s+to|adjacent\s+to)\b", "near"),
            )
            relation = "near"
            relation_start = None
            relation_end = None

            if place_match is not None:
                relation_text_start = place_match.end()
                relation_text = text[relation_text_start:]
                for pattern, normalized in relation_patterns:
                    match = re.search(pattern, relation_text)
                    if match is not None:
                        relation = normalized
                        relation_start = relation_text_start + match.start()
                        relation_end = relation_text_start + match.end()
                        break

            destination = None
            if relation_end is not None:
                destination_candidates = parse_target_specs(text[relation_end:])
                if destination_candidates:
                    destination = destination_candidates[0]

            # Fallback for concise or slightly different natural-language
            # forms where the relation phrase is omitted.
            if destination is None and place_match is not None:
                after_place = parse_target_specs(text[place_match.end():])
                destination_candidates = [
                    item for item in after_place
                    if item.object_name not in PICKABLE_OBJECTS
                    and item.object_name != "object"
                ]
                if destination_candidates:
                    destination = destination_candidates[0]

            if destination is None:
                destination_candidates = [
                    item for item in task_targets
                    if item.object_name not in PICKABLE_OBJECTS
                    and item.object_name != "object"
                ]
                if destination_candidates:
                    destination = destination_candidates[-1]

            # Prefer an explicitly named object after the placement verb.
            # For pronouns such as "place it", resolve to the most recently
            # mentioned movable object before the placement instruction.
            moving_object = None
            if place_match is not None:
                object_segment_end = (
                    relation_start if relation_start is not None else len(text)
                )
                object_segment = text[place_match.end():object_segment_end]
                explicit_movable = [
                    item for item in parse_target_specs(object_segment)
                    if item.object_name in PICKABLE_OBJECTS
                    or item.object_name == "object"
                ]
                if explicit_movable:
                    moving_object = explicit_movable[-1]

                if moving_object is None:
                    preceding_movable = [
                        item for item in parse_target_specs(text[:place_match.start()])
                        if item.object_name in PICKABLE_OBJECTS
                        or item.object_name == "object"
                    ]
                    if preceding_movable:
                        moving_object = preceding_movable[-1]

            # A terse instruction such as "place it near the goal" may rely
            # on the object already held by the agent.
            held = getattr(belief, "held_object", None) if belief is not None else None
            if moving_object is None and held:
                moving_object = TargetSpec.from_value(held)

            if moving_object is None:
                movable_targets = [
                    item for item in task_targets
                    if item.object_name in PICKABLE_OBJECTS
                    or item.object_name == "object"
                ]
                if movable_targets:
                    moving_object = movable_targets[-1]

            if moving_object is not None and destination is not None:
                already_held = bool(
                    belief is not None
                    and self._held_matches(belief, moving_object)
                )
                if not already_held:
                    calls.append(
                        make_call("PickUp", target=moving_object.to_dict())
                    )
                calls.append(
                    make_call(
                        "PlaceNear",
                        object=moving_object.to_dict(),
                        target=destination.to_dict(),
                        relation=relation,
                    )
                )

        if done_intent:
            target = next((
                item for item in targets
                if item.object_name in {"door", "goal"}
                or (object_navigation_intent and item.object_name in PICKABLE_OBJECTS)
            ), None)
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
        if name in {"NavigateTo", "Reach", "PickUp", "OpenDoor", "UnlockDoor", "Done"} and target is not None and not positions:
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
