"""Deterministic feasibility evaluator based only on BeliefState."""
from skills.navigation import ObservationNavigator
from skills.target import PICKABLE_OBJECTS, TargetSpec, relation_offset
from skills.dependency import TaskDependencyResolver
from perception.belief_state import DIRECTIONS, is_walkable


class SkillFeasibilityEvaluator:
    def __init__(self, runtime, dependency_resolver=None):
        self.runtime = runtime
        self.belief = runtime.belief
        self.navigator = ObservationNavigator(runtime)
        self.dependencies = dependency_resolver or TaskDependencyResolver()

    def _positions(self, target):
        spec = TargetSpec.from_value(target)
        return [(tuple(pos), cell) for pos, cell in self.belief.known_map.items() if spec.matches(cell)]

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
        positions = self._positions(target) if target is not None else []
        score, reason = 0.0, "Preconditions are not satisfied."

        if self.runtime.terminated or self.runtime.truncated:
            reason = "The episode has ended."
        elif name == "Explore":
            score = float(self._has_frontier())
            reason = "A reachable frontier remains." if score else "No known unexplored frontier remains."
        elif name == "Find":
            score = float(not positions and self._has_frontier())
            reason = "Search can reveal new observations." if score else ("Target is already known." if positions else "No reachable unexplored frontier remains.")
        elif name == "NavigateTo":
            if positions:
                pos = min((p for p, _ in positions), key=lambda p: abs(p[0]-self.belief.agent_pos[0])+abs(p[1]-self.belief.agent_pos[1]))
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
                    relation = str(args.get("relation", "near")).lower()
                    offset = relation_offset(relation)
                    dest_pos = dests[0][0]
                    if offset is None and relation not in {"near", "adjacent", "nearby", "beside", "next_to"}:
                        reason = "Unsupported placement relation."
                    else:
                        candidates = [(dest_pos[0]+offset[0], dest_pos[1]+offset[1])] if offset is not None else [(dest_pos[0]+dx, dest_pos[1]+dy) for dx,dy in DIRECTIONS]
                        empty = [p for p in candidates if self.belief.cell_at(p) is not None and self.belief.cell_at(p).get("object") in {"empty","floor"}]
                        score = float(any(self._reachable_adjacent(p) for p in empty))
                        reason = "A known empty placement cell is reachable." if score else "No known reachable empty placement cell."
        elif name == "MoveObstacle":
            obstacle, destination = args.get("target"), args.get("destination")
            if not positions:
                reason = "Find the obstacle first."
            elif destination is None or not self._positions(destination):
                reason = "Find the destination first."
            elif self.belief.inventory_status != "empty" and not self._held_matches(obstacle):
                reason = "The gripper must be empty before moving an obstacle."
            else:
                score = float(self._reachable_adjacent(positions[0][0]))
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
                prerequisites = [{"id":"Explore","name":"Explore","args":{},"description":"Explore the environment"}]
        return {"feasibility": float(score), "reason": reason, "prerequisites": prerequisites}
