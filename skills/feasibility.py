"""Deterministic feasibility evaluator based only on BeliefState."""
from skills.navigation import ObservationNavigator
from skills.target import PICKABLE_OBJECTS, TargetSpec
from skills.placement import select_placement
from skills.dependency import TaskDependencyResolver, make_call
from skills.manipulation_skills import select_clear_drop_cell
from perception.belief_state import DIRECTIONS, is_walkable


class SkillFeasibilityEvaluator:
    def __init__(self, runtime, dependency_resolver=None):
        self.runtime = runtime
        self.belief = runtime.belief
        self.navigator = ObservationNavigator(runtime)
        self.dependencies = dependency_resolver or TaskDependencyResolver()

    def _positions(self, target, position=None):
        spec = TargetSpec.from_value(target)
        positions = [(tuple(pos), cell) for pos, cell in self.belief.known_map.items() if spec.matches(cell)]
        if position is None:
            return positions
        requested = tuple(int(value) for value in position)
        return [item for item in positions if item[0] == requested]

    def _has_frontier(self):
        for pos, cell in self.belief.known_map.items():
            if not is_walkable(cell):
                continue
            for dx, dy in DIRECTIONS:
                if self.belief.cell_at((pos[0] + dx, pos[1] + dy)) is None:
                    return True
        return False

    def _reachable_adjacent(self, target_pos):
        start = tuple(self.belief.agent_pos)
        for dx, dy in DIRECTIONS:
            candidate = (target_pos[0] + dx, target_pos[1] + dy)
            if self.navigator._safe_walkable(self.belief.cell_at(candidate)):
                if self.navigator._find_path(start, candidate) is not None:
                    return True
        return False

    def _held_matches(self, target):
        held = getattr(self.belief, "held_object", None)
        return bool(held and TargetSpec.from_value(target).matches(held))

    def evaluate(self, call):
        name, args = call.get("name", ""), call.get("args", {})
        target = args.get("target")
        position = args.get("position")
        positions = self._positions(target, position) if target is not None else []
        score, reason = 0.0, "Preconditions are not satisfied."

        if self.runtime.terminated or self.runtime.truncated:
            reason = "The episode has ended."
        elif name == "Explore":
            score = float(self._has_frontier())
            reason = "A reachable frontier remains." if score else "No known unexplored frontier remains."
        elif name == "Find":
            if target is not None and self._held_matches(target):
                score = 0.0
                reason = "Target is already held; no search is required."
            else:
                score = float(not positions and self._has_frontier())
                reason = "Search can reveal new observations." if score else ("Target is already known." if positions else "No reachable unexplored frontier remains.")
        elif name == "Reach":
            if positions:
                pos, cell = min(
                    positions,
                    key=lambda item: abs(item[0][0] - self.belief.agent_pos[0])
                    + abs(item[0][1] - self.belief.agent_pos[1]),
                )
                if not is_walkable(cell):
                    reason = "Reach requires a known walkable target cell; use NavigateTo for non-walkable objects."
                else:
                    path = self.navigator._find_path(
                        self.belief.agent_pos,
                        pos,
                        allow_goal=True,
                    )
                    score = float(path is not None)
                    reason = "A known path to the target cell exists." if score else "No known path to the target cell."
            else:
                reason = "Find the target first."
        elif name == "NavigateTo":
            if positions:
                pos, target_cell = min(
                    positions,
                    key=lambda item: (
                        abs(item[0][0] - self.belief.agent_pos[0])
                        + abs(item[0][1] - self.belief.agent_pos[1])
                    ),
                )
                dx = pos[0] - self.belief.agent_pos[0]
                dy = pos[1] - self.belief.agent_pos[1]
                already_positioned = bool(
                    abs(dx) + abs(dy) == 1
                    and (dx, dy) == DIRECTIONS[int(self.belief.agent_dir)]
                )
                target_spec = TargetSpec.from_value(target)
                if (
                    position is not None
                    and target_spec.object_name == "door"
                    and target_cell.get("state") == "open"
                ):
                    score = 0.0
                    reason = (
                        "This exact door is already open; continue exploration "
                        "through it instead of navigating back to it."
                    )
                elif already_positioned:
                    score = 0.0
                    reason = "The agent is already adjacent to and facing the target."
                else:
                    score = float(self._reachable_adjacent(pos))
                    reason = "A known route to an adjacent cell exists." if score else "No known route to an adjacent cell."
            else:
                reason = "Find the target first."
        elif name == "PickUp":
            if self.belief.inventory_status != "empty" and not self._held_matches(target):
                reason = "The gripper is not confirmed empty."
            elif positions:
                spec = TargetSpec.from_value(target)
                if spec.object_name not in PICKABLE_OBJECTS and spec.object_name != "object":
                    reason = "The target is not a supported pickable object."
                elif spec.object_name == "key" and any(
                    door.get("state") == "locked"
                    and (not spec.color or door.get("color") == spec.color)
                    and any(
                        abs(tuple(obj_pos)[0] - tuple(door_pos)[0])
                        + abs(tuple(obj_pos)[1] - tuple(door_pos)[1]) == 1
                        and obj.get("object") in {"ball", "box"}
                        for obj_pos, obj in self.belief.known_map.items()
                    )
                    for door_pos, door in self.belief.known_map.items()
                    if door.get("object") == "door"
                ):
                    reason = (
                        "Clear the movable object blocking the matching locked "
                        "door before occupying the gripper with its key."
                    )
                else:
                    pos = min((p for p, _ in positions), key=lambda p: abs(p[0]-self.belief.agent_pos[0])+abs(p[1]-self.belief.agent_pos[1]))
                    score = float(self._reachable_adjacent(pos))
                    reason = "The target is known and reachable." if score else "No known route to the target."
            else:
                reason = "Find the target before pickup."
        elif name in {"OpenDoor", "UnlockDoor"}:
            if positions:
                pos, cell = min(positions, key=lambda item: abs(item[0][0]-self.belief.agent_pos[0])+abs(item[0][1]-self.belief.agent_pos[1]))
                state = cell.get("state", "unknown")
                dx, dy = pos[0]-self.belief.agent_pos[0], pos[1]-self.belief.agent_pos[1]
                front = (dx, dy) == DIRECTIONS[int(self.belief.agent_dir)] and abs(dx)+abs(dy) == 1
                if not front:
                    reason = "The door must be adjacent and directly in front."
                elif name == "OpenDoor" and state == "closed":
                    score, reason = 1.0, "An unlocked closed door is directly in front."
                elif name == "UnlockDoor" and state == "locked":
                    key = {"object": "key", "color": cell.get("color")}
                    if self._held_matches(key):
                        score, reason = 1.0, "A matching key is held and the locked door is directly in front."
                    else:
                        reason = "Find and pick up a matching key first."
                elif name == "OpenDoor" and state == "locked":
                    reason = "The door is locked; use UnlockDoor after acquiring a key."
                else:
                    reason = "The requested door operation is already complete or not applicable."
            else:
                reason = "Find the door first."
        elif name == "PlaceNear":
            obj, destination = args.get("object", "object"), args.get("target")
            if not self._held_matches(obj):
                reason = "The requested object must be held before placement."
            else:
                dests = self._positions(destination) if destination is not None else []
                if not dests:
                    reason = "Find the destination first."
                else:
                    _, _, placement_error = select_placement(
                        self.belief,
                        self.navigator,
                        [pos for pos, _ in dests],
                        args.get("relation", "near"),
                    )
                    score = float(placement_error is None)
                    reason = (
                        "A known empty placement cell with a reachable adjacent approach exists."
                        if score else {
                            "unsupported_relation": "Unsupported placement relation.",
                            "placement_cell_unknown": "No known empty placement cell is available.",
                            "placement_cell_occupied": "The relation-derived placement cell is occupied.",
                            "no_known_path": "Known empty placement cells have no reachable adjacent approach.",
                        }.get(placement_error, "Placement is not currently feasible.")
                    )
        elif name == "MoveObstacle":
            obstacle, destination = args.get("target"), args.get("destination")
            holding_obstacle = self._held_matches(obstacle)
            if not positions and not holding_obstacle:
                reason = "Find the obstacle first."
            elif self.belief.inventory_status != "empty" and not self._held_matches(obstacle):
                reason = "The gripper must be empty before moving an obstacle."
            else:
                source_pos = (
                    positions[0][0]
                    if positions else tuple(self.belief.agent_pos)
                )
                obstacle_reachable = bool(
                    holding_obstacle or self._reachable_adjacent(source_pos)
                )
                if destination is None:
                    clear_cell = select_clear_drop_cell(
                        self.belief, self.navigator, source_pos
                    )
                    score = float(obstacle_reachable and clear_cell is not None)
                    reason = (
                        "The blocking object is reachable and has a safe observed drop cell."
                        if score else "No safe observed relocation plan is currently available."
                    )
                elif not self._positions(destination):
                    reason = "Find the destination first."
                else:
                    score = float(obstacle_reachable)
                    reason = "The obstacle is known and reachable." if score else "No known route to the obstacle."
        elif name == "Done":
            if positions:
                pos = positions[0][0]
                score = float(abs(pos[0]-self.belief.agent_pos[0])+abs(pos[1]-self.belief.agent_pos[1]) == 1)
                reason = "The target is adjacent; done can be issued." if score else "Navigate to the target first."
            else:
                reason = "Find the target before issuing done."
        else:
            reason = f"Unknown skill: {name}"

        prerequisites = []
        if score <= 0:
            prerequisites = self.dependencies.prerequisites_for(call, self.belief)
            if not prerequisites and name not in {"Explore", "Find"} and self._has_frontier():
                prerequisites = [make_call("Explore")]
        return {"feasibility": float(score), "reason": reason, "prerequisites": prerequisites}
