from collections import deque

from minigrid.core.actions import Actions
from perception.belief_state import DIRECTIONS, is_walkable


class ApproachDoorSkill:
    """
    观测驱动的门接近技能。

    仅使用 BeliefState 中的已知地图。
    目标格和当前 waypoint 在有效时保持不变，
    避免每次转向后切换路径，导致原地转圈。
    """

    def __init__(self, runtime, color="yellow"):
        self.runtime = runtime
        self.belief = runtime.belief
        self.color = color

        self.door_pos = None
        self.goal_pos = None
        self.waypoint = None

    @staticmethod
    def _neighbors(pos):
        x, y = pos
        return [
            (x + dx, y + dy)
            for dx, dy in DIRECTIONS
        ]

    @staticmethod
    def _distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _find_path(self, start, goal):
        """只允许经过已知且可通行的格子。"""
        if start == goal:
            return [start]

        queue = deque([start])
        parent = {start: None}

        while queue:
            current = queue.popleft()

            for nxt in self._neighbors(current):
                if nxt in parent:
                    continue

                cell = self.belief.cell_at(nxt)

                if cell is None or not is_walkable(cell):
                    continue

                parent[nxt] = current

                if nxt == goal:
                    path = []
                    node = goal

                    while node is not None:
                        path.append(node)
                        node = parent[node]

                    return list(reversed(path))

                queue.append(nxt)

        return None

    def _find_door(self):
        # 优先保留已经追踪的门，避免切换目标。
        if self.door_pos is not None:
            cell = self.belief.cell_at(self.door_pos)

            if (
                cell is not None
                and cell["object"] == "door"
                and cell["color"] == self.color
            ):
                return self.door_pos, cell

            self.door_pos = None
            self.goal_pos = None
            self.waypoint = None

        doors = self.belief.find_objects(
            "door", color=self.color
        )

        if not doors:
            return None, None

        start = tuple(self.belief.agent_pos)

        # 已知地图中的最近目标；选定后持续追踪。
        door_pos, cell = min(
            doors,
            key=lambda item: self._distance(start, item[0]),
        )

        self.door_pos = tuple(door_pos)
        return self.door_pos, cell

    def _choose_goal(self, start, door_pos, exclude=None):
        """
        选择一格已知可通行、且与门相邻的位置。
        只在首次选择或原目标失效时调用。
        """
        candidates = []

        for goal in self._neighbors(door_pos):
            if exclude is not None and goal == exclude:
                continue

            cell = self.belief.cell_at(goal)

            if cell is None or not is_walkable(cell):
                continue

            path = self._find_path(start, goal)

            if path is not None:
                candidates.append((len(path), goal))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    @staticmethod
    def _direction_between(start, end):
        delta = (
            end[0] - start[0],
            end[1] - start[1],
        )
        return DIRECTIONS.index(delta)

    def _turn_action(self, desired_direction):
        difference = (
            desired_direction - self.belief.agent_dir
        ) % 4

        if difference == 0:
            return None
        if difference == 1:
            return Actions.right
        if difference == 3:
            return Actions.left

        # 180 度时固定右转；下一轮继续朝同一个 waypoint。
        return Actions.right

    def _execute_one(self, action, desired_direction=None):
        door_pos = self.door_pos
        goal_pos = self.goal_pos
        waypoint = self.waypoint

        result = self.runtime.execute_action(action)

        return {
            "success": False,
            "status": "action_executed",
            "action": action.name,
            "door_estimated_position": list(door_pos),
            "goal_estimated_position": (
                list(goal_pos) if goal_pos is not None else None
            ),
            "waypoint_estimated_position": (
                list(waypoint) if waypoint is not None else None
            ),
            "planned_direction": desired_direction,
            "estimated_position": list(self.belief.agent_pos),
            "actions_executed": 1,
            "obs": result["obs"],
        }

    def step(self):
        start = tuple(self.belief.agent_pos)

        door_pos, door = self._find_door()

        if door is None:
            return {
                "success": False,
                "status": "no_known_door",
                "reason": "No yellow door is known in the belief map.",
                "actions_executed": 0,
            }

        if door.get("state") == "open":
            return {
                "success": True,
                "status": "door_already_open",
                "reason": "The observed door is already open.",
                "actions_executed": 0,
            }

        if door.get("state") != "closed":
            return {
                "success": False,
                "status": "unknown_door_state",
                "actions_executed": 0,
            }

        # 目标相邻格一旦确定，就持续使用。
        if self.goal_pos is not None:
            cell = self.belief.cell_at(self.goal_pos)

            if (
                self._distance(self.goal_pos, door_pos) != 1
                or cell is None
                or not is_walkable(cell)
            ):
                self.goal_pos = None
                self.waypoint = None

        if self.goal_pos is None:
            self.goal_pos = self._choose_goal(
                start, door_pos
            )

            if self.goal_pos is None:
                return {
                    "success": False,
                    "status": "no_known_path",
                    "reason": (
                        "No known traversable route to a cell "
                        "adjacent to the door."
                    ),
                    "door_estimated_position": list(door_pos),
                    "actions_executed": 0,
                }

        goal = tuple(self.goal_pos)

        # 到达门旁后，面向门即可完成接近技能。
        if start == goal:
            self.waypoint = None
            desired = self._direction_between(start, door_pos)
            action = self._turn_action(desired)

            if action is not None:
                return self._execute_one(action, desired)

            return {
                "success": True,
                "status": "target_adjacent",
                "reason": "Adjacent to the door and facing it.",
                "door_estimated_position": list(door_pos),
                "goal_estimated_position": list(goal),
                "estimated_position": list(start),
                "actions_executed": 0,
            }

        # 到达当前路标后，才重新计算下一个路标。
        if self.waypoint == start:
            self.waypoint = None

        # 转向期间保留原路标，不因新观测立即改变方向。
        if self.waypoint is not None:
            waypoint_cell = self.belief.cell_at(self.waypoint)

            if (
                self._distance(start, self.waypoint) != 1
                or waypoint_cell is None
                or not is_walkable(waypoint_cell)
            ):
                self.waypoint = None

        if self.waypoint is None:
            path = self._find_path(start, goal)

            if path is None:
                old_goal = self.goal_pos
                self.goal_pos = self._choose_goal(
                    start, door_pos, exclude=old_goal
                )
                self.waypoint = None

                if self.goal_pos is None:
                    return {
                        "success": False,
                        "status": "no_known_path",
                        "reason": (
                            "The current route is unavailable "
                            "and no alternative known route exists."
                        ),
                        "door_estimated_position": list(door_pos),
                        "actions_executed": 0,
                    }

                goal = tuple(self.goal_pos)
                path = self._find_path(start, goal)

                if path is None or len(path) < 2:
                    return {
                        "success": False,
                        "status": "replanning_failed",
                        "actions_executed": 0,
                    }

            if len(path) < 2:
                return {
                    "success": False,
                    "status": "invalid_waypoint",
                    "actions_executed": 0,
                }

            self.waypoint = tuple(path[1])

        waypoint = tuple(self.waypoint)
        desired = self._direction_between(start, waypoint)
        action = self._turn_action(desired)

        # 先转向；下一轮仍然朝向同一个 waypoint。
        if action is not None:
            return self._execute_one(action, desired)

        # 已面向 waypoint，执行前进一步。
        return self._execute_one(Actions.forward, desired)
