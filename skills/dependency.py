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


_COLOR_PATTERN = r"red|blue|green|purple|yellow|grey|gray"
_ORDERED_DOORS_PATTERN = re.compile(
    rf"\bopen\s+(?:the\s+)?({_COLOR_PATTERN})\s+door\s+"
    rf"(?:and\s+)?then\s+(?:open\s+)?(?:the\s+)?"
    rf"({_COLOR_PATTERN})\s+door\b",
    re.IGNORECASE,
)


def ordered_door_colors(task: str) -> list[str]:
    """Return an explicitly ordered two-door sequence, if one is requested."""
    match = _ORDERED_DOORS_PATTERN.search(str(task))
    if not match:
        return []
    return [color.lower() for color in match.groups()]


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
    position = args.get("position")
    position_key = ""
    if isinstance(position, (tuple, list)) and len(position) == 2:
        position_key = f"at:{int(position[0])},{int(position[1])}"
    mode_key = ":".join(
        mode for flag, mode in (
            (args.get("memory_probe"), "memory_probe"),
            (args.get("grounded_reference"), "grounded_reference"),
            (args.get("reactive_obstacles"), "reactive_obstacles"),
        ) if flag
    )
    suffix = ":".join(
        part for part in (
            target_key, object_key, destination_key, relation, position_key,
            mode_key,
        ) if part
    )
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

    @staticmethod
    def _matching_hallway_target(belief):
        """Resolve MiniGrid Memory's cue using only the observed belief map."""
        if belief is None:
            return None
        objects = [
            (tuple(pos), cell)
            for pos, cell in belief.known_map.items()
            if cell.get("object") in PICKABLE_OBJECTS
            and cell.get("color") == "green"
        ]
        if len(objects) < 3:
            return None
        by_x = {}
        for item in objects:
            by_x.setdefault(item[0][0], []).append(item)
        for endpoint_x, endpoint_items in sorted(by_x.items()):
            endpoint_types = {cell.get("object") for _, cell in endpoint_items}
            if len(endpoint_items) != 2 or len(endpoint_types) != 2:
                continue
            cue_types = {
                cell.get("object") for pos, cell in objects
                if pos[0] != endpoint_x
            }
            matching = [
                item for item in endpoint_items
                if item[1].get("object") in cue_types
            ]
            if len(matching) == 1:
                return matching[0]
        return None

    def parse_targets(self, task: str, belief=None) -> list[TargetSpec]:
        targets = parse_target_specs(task)
        text = str(task).lower()

        # LockedRoom names the room containing the key rather than explicitly
        # naming its entrance (for example, "from the grey room"). In the
        # environment each room is identified by its single same-colour door,
        # so ground that phrase as a door target that can be found and opened.
        seen = {item.key for item in targets}
        for match in re.finditer(
            rf"\b(?:from|in)\s+(?:the\s+)?({_COLOR_PATTERN})\s+room\b",
            text,
        ):
            room_door = TargetSpec("door", color=match.group(1).lower())
            if room_door.key not in seen:
                targets.append(room_door)
                seen.add(room_door.key)

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
        ) and not any(
            marker in text
            for marker in (
                "unlock", "open", " goal", " room", "place", "put", "drop",
            )
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
        navigation_only = not any(
            word in text
            for word in (
                "find", "search", "locate", "open", "unlock", "pick",
                "take", "grab", "carry", "fetch", "place", "put", "drop",
                "move", "clear",
            )
        )
        goto_object_intent = (
            object_movement_intent
            and not acquisition_intent
            and navigation_only
            and len(explicit_object_targets) == 1
            and all(
                item.object_name in PICKABLE_OBJECTS
                for item in targets
            )
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
            locked_doors = []
            if locked_door_known:
                locked_doors = [
                    TargetSpec(
                        "door",
                        color=(
                            cell.get("color")
                            if cell.get("color") != "unknown" else None
                        ),
                    )
                    for _, cell in belief.find_objects("door", state="locked")
                ]
            # Observation evidence has priority: do not invent a key for every
            # unrelated open/closed door just because one locked door exists.
            for door in locked_doors or doors or [TargetSpec("door")]:
                key = TargetSpec("key", color=door.color)
                if key.key not in seen:
                    targets.append(key)
                    seen.add(key.key)

        return targets

    def candidate_calls(self, task: str, belief=None) -> list[dict]:
        text = str(task).lower()
        ordered_doors = ordered_door_colors(task)
        active_ordered_door = None
        if ordered_doors:
            active_ordered_door = ordered_doors[-1]
            if belief is not None:
                for color in ordered_doors:
                    already_open = bool(
                        belief.find_objects("door", color=color, state="open")
                    )
                    if not already_open:
                        active_ordered_door = color
                        break
        memory_intent = bool(
            "matching object" in text and "end of the hallway" in text
        )
        if memory_intent:
            resolved = self._matching_hallway_target(belief)
            if resolved is None:
                return [make_call("Explore", memory_probe=True)]
            position, cell = resolved
            target = TargetSpec(
                object_name=cell.get("object"),
                color=(
                    cell.get("color")
                    if cell.get("color") != "unknown" else None
                ),
            )
            return [make_call(
                "NavigateTo", target=target.to_dict(),
                position=list(position), grounded_reference=True,
            )]

        reactive_goal_navigation = (
            text.strip(" .") == "get to the green goal square"
        )
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
        navigation_only = not any(
            word in text
            for word in (
                "find", "search", "locate", "open", "unlock", "pick",
                "take", "grab", "carry", "fetch", "place", "put", "drop",
                "move", "clear",
            )
        )
        object_navigation_intent = (
            movement_intent
            and not pick_verbs
            and navigation_only
            and any(target.object_name in PICKABLE_OBJECTS for target in targets)
            and all(target.object_name in PICKABLE_OBJECTS for target in targets)
        )
        door_navigation_intent = (
            movement_intent
            and navigation_only
            and any(target.object_name == "door" for target in targets)
            and all(target.object_name == "door" for target in targets)
        )
        done_intent = (
            any(word in text for word in ("done", "finish", "complete"))
            or door_navigation_intent
            or object_navigation_intent
        )
        goal_entry_intent = any(
            phrase in text
            for phrase in ("reach", "enter", "arrive at", "get to", "go to", "navigate to", "cross to")
        )
        explicit_task_targets = parse_target_specs(task)
        explicit_pickable_targets = [
            item for item in explicit_task_targets
            if item.object_name in PICKABLE_OBJECTS
        ]
        locked_key_colors = {
            cell.get("color")
            for _, cell in (
                belief.find_objects("door", state="locked")
                if belief is not None else []
            )
        }

        calls = [make_call(
            "Explore", reactive_obstacles=reactive_goal_navigation
        )]
        for target in targets:
            calls.append(make_call(
                "Find", target=target.to_dict(),
                reactive_obstacles=reactive_goal_navigation,
            ))
            if target.object_name == "goal" and goal_entry_intent:
                calls.append(make_call(
                    "Reach", target=target.to_dict(),
                    reactive_obstacles=reactive_goal_navigation,
                ))
            elif movement_intent and any(
                requested.matches(target.to_dict())
                for requested in explicit_task_targets
            ):
                calls.append(make_call("NavigateTo", target=target.to_dict()))
            explicitly_requested_pickup = bool(
                pick_verbs
                and any(
                    requested.matches(target.to_dict())
                    for requested in explicit_pickable_targets
                )
            )
            required_unlock_key = bool(
                unlock_intent
                and target.object_name == "key"
                and (
                    not locked_key_colors
                    or target.color in locked_key_colors
                )
            )
            if target.object_name in PICKABLE_OBJECTS and (
                explicitly_requested_pickup or required_unlock_key
            ):
                calls.append(make_call("PickUp", target=target.to_dict()))
            # Closed doors are traversal affordances even when the mission does
            # not literally say "open". Keep every observed door position in
            # the call identity so opening one blue door does not suppress all
            # other blue doors. GoToDoor is the exception: toggling its target
            # ends the benchmark incorrectly, so that navigation-only task must
            # still use Done instead of OpenDoor.
            if target.object_name == "door":
                known_doors = (
                    self._known_positions(belief, target.to_dict())
                    if belief is not None else []
                )
                if known_doors:
                    for door_pos, door_cell in known_doors:
                        state = door_cell.get("state")
                        position = list(door_pos)
                        ordered_door_is_active = bool(
                            not ordered_doors
                            or target.color == active_ordered_door
                        )
                        if state == "closed" and ordered_door_is_active and (
                            "open" in text or not door_navigation_intent
                        ):
                            calls.append(make_call(
                                "OpenDoor", target=target.to_dict(),
                                position=position,
                            ))
                        if state == "locked" and unlock_intent:
                            calls.append(make_call(
                                "OpenDoor", target=target.to_dict(),
                                position=position,
                            ))
                            calls.append(make_call(
                                "UnlockDoor", target=target.to_dict(),
                                position=position,
                            ))
                elif (
                    ("open" in text or unlock_intent)
                    and (
                        not ordered_doors
                        or target.color == active_ordered_door
                    )
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

        # Some official missions describe only the final object, while a
        # movable ball/box physically blocks the door on the route to it.
        # Promote that observed obstruction to an explicit SayCan candidate.
        # Keys are excluded because acquiring them is usually the intended
        # unlock prerequisite, not obstacle relocation.
        if belief is not None:
            explicit_targets = parse_target_specs(task)
            barrier_doors = [
                tuple(pos) for pos, cell in belief.known_map.items()
                if cell.get("object") == "door"
                and cell.get("state") in {"closed", "locked"}
            ]
            for obstacle_pos, cell in belief.known_map.items():
                if cell.get("object") not in {"ball", "box"}:
                    continue
                if any(spec.matches(cell) for spec in explicit_targets):
                    continue
                pos = tuple(obstacle_pos)
                if not any(
                    abs(pos[0] - door[0]) + abs(pos[1] - door[1]) == 1
                    for door in barrier_doors
                ):
                    continue
                obstacle = TargetSpec(
                    object_name=cell.get("object"),
                    color=(
                        cell.get("color")
                        if cell.get("color") != "unknown" else None
                    ),
                )
                calls.append(make_call(
                    "MoveObstacle", target=obstacle.to_dict(),
                    position=list(pos),
                ))

            # MiniGrid keeps a key in the agent's hand after the lock is
            # opened. Once no locked door remains, relocate that spent key to
            # a safe observed cell so an explicitly requested ball/box can be
            # acquired. This is task-conditioned and never drops a key while
            # it is still needed by an observed lock.
            held = getattr(belief, "held_object", None)
            if (
                held
                and held.get("object") == "key"
                and pick_verbs
                and explicit_pickable_targets
                and not any(spec.matches(held) for spec in explicit_pickable_targets)
                and not belief.find_objects("door", state="locked")
            ):
                calls.append(make_call(
                    "MoveObstacle",
                    target=TargetSpec.from_value(held).to_dict(),
                ))

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
    def _known_positions(
        belief, target, position=None
    ) -> list[tuple[tuple[int, int], dict]]:
        spec = TargetSpec.from_value(target)
        positions = [
            (tuple(pos), cell)
            for pos, cell in belief.known_map.items()
            if spec.matches(cell)
        ]
        if position is None:
            return positions
        requested = tuple(int(value) for value in position)
        return [item for item in positions if item[0] == requested]

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
        position = args.get("position")
        prerequisites = []

        def add(name_, **kwargs):
            prerequisites.append(make_call(name_, **kwargs))

        positions = (
            self._known_positions(belief, target, position)
            if target is not None else []
        )
        if name in {"NavigateTo", "Reach", "PickUp", "OpenDoor", "UnlockDoor", "Done"} and target is not None and not positions:
            add("Find", target=target)

        if name == "PickUp" and target is not None:
            if belief.inventory_status != "empty" and not self._held_matches(belief, target):
                return []
            if not positions:
                add("Find", target=target)
            target_spec = TargetSpec.from_value(target)
            if target_spec.object_name == "key":
                for door_pos, door_cell in belief.known_map.items():
                    if door_cell.get("object") != "door" or door_cell.get("state") != "locked":
                        continue
                    if target_spec.color and door_cell.get("color") != target_spec.color:
                        continue
                    for obstacle_pos, obstacle_cell in belief.known_map.items():
                        if obstacle_cell.get("object") not in {"ball", "box"}:
                            continue
                        if (
                            abs(tuple(obstacle_pos)[0] - tuple(door_pos)[0])
                            + abs(tuple(obstacle_pos)[1] - tuple(door_pos)[1])
                            != 1
                        ):
                            continue
                        obstacle = TargetSpec(
                            object_name=obstacle_cell.get("object"),
                            color=(
                                obstacle_cell.get("color")
                                if obstacle_cell.get("color") != "unknown"
                                else None
                            ),
                        )
                        add(
                            "MoveObstacle",
                            target=obstacle.to_dict(),
                            position=list(obstacle_pos),
                        )

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
                    add("NavigateTo", target=target, position=position)

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
            if obstacle is not None and not self._known_positions(
                belief, obstacle, position
            ):
                add("Find", target=obstacle)
            if obstacle is not None and not self._held_matches(belief, obstacle):
                add("PickUp", target=obstacle, position=position)
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
