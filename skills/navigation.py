from collections import deque

from minigrid.core.actions import Actions
from perception.belief_state import DIRECTIONS, is_walkable


class ObservationNavigator:
    """
    只依据 BeliefState 中已知地图进行导航。

    不读取 env.grid、真实坐标或环境 info。
    每次 step 只执行一个底层动作。
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.belief = runtime.belief

        self._active_target = None
        self._approach_pos = None

    @staticmethod
    def _safe_walkable(cell):
        """
        不把绿色目标格当作普通导航路径。

        避免智能体站上目标格后，观测记忆丢失目标信息。
        """
        return (
            cell is not None
            and cell.get("object") != "goal"
            and is_walkable(cell)
        )

    @staticmethod
    def _neighbors(pos):
        x, y = pos
        return [
            (x + dx, y + dy)
            for dx, dy in DIRECTIONS
        ]

    @staticmethod
    def _distance(a, b):
        return (
            abs(a[0] - b[0])
            + abs(a[1] - b[1])
        )

    def reset_target(self):
        self._active_target = None
        self._approach_pos = None

    def _find_path(self, start, goal):
        """只在已知且可通行的格子中执行 BFS。"""
        start = tuple(start)
        goal = tuple(goal)

        if start == goal:
            return [start]

        if not self._safe_walkable(
            self.belief.cell_at(goal)
        ):
            return None

        queue = deque([start])
        parent = {start: None}

        while queue:
            current = queue.popleft()

            for nxt in self._neighbors(current):
                if nxt in parent:
                    continue

                cell = self.belief.cell_at(nxt)

                if not self._safe_walkable(cell):
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

    @staticmethod
    def _turn_action(current_direction, desired_direction):
        difference = (
            desired_direction - current_direction
        ) % 4

        if difference == 0:
            return None

        if difference == 1:
            return Actions.right

        if difference == 3:
            return Actions.left

        # 相差 180 度时，分两次右转。
        return Actions.right

    def _execute(self, action, reason):
        result = self.runtime.execute_action(action)

        return {
            "success": True,
            "status": "action_executed",
            "action": action.name.lower(),
            "reason": reason,
            "actions_executed": result["actions_executed"],
            "estimated_position": list(
                self.belief.agent_pos
            ),
            "direction": int(
                self.belief.agent_dir
            ),
            "known_cells": len(
                self.belief.known_map
            ),
        }

    def step_towards_cell(self, goal_pos):
        """
        向已知、可通行的目标格移动一步。

        返回：
        - action_executed
        - at_target_cell
        - target_not_known
        - no_known_path
        """
        goal = tuple(int(v) for v in goal_pos)
        start = tuple(self.belief.agent_pos)

        if self.belief.cell_at(goal) is None:
            return {
                "success": False,
                "status": "target_not_known",
                "reason": "The target cell has not been observed.",
                "actions_executed": 0,
            }

        path = self._find_path(start, goal)

        if path is None:
            return {
                "success": False,
                "status": "no_known_path",
                "reason": (
                    "No route exists through the currently "
                    "known traversable cells."
                ),
                "actions_executed": 0,
            }

        if len(path) == 1:
            return {
                "success": True,
                "status": "at_target_cell",
                "reason": "Already at the target cell.",
                "actions_executed": 0,
            }

        next_pos = path[1]

        delta = (
            next_pos[0] - start[0],
            next_pos[1] - start[1],
        )

        desired_direction = DIRECTIONS.index(delta)
        current_direction = int(
            self.belief.agent_dir
        )

        if current_direction != desired_direction:
            action = self._turn_action(
                current_direction,
                desired_direction,
            )

            return self._execute(
                action,
                "Turning toward the next known waypoint.",
            )

        return self._execute(
            Actions.forward,
            "Moving to the next known traversable cell.",
        )

    def step_to_adjacent(self, target_pos):
        """
        接近目标，最终停在目标相邻格并面向目标。

        适用于：
        - 门前交互
        - 红球拾取
        - 向目标格放置物体

        不会进入目标物体所在的格子。
        """
        target = tuple(int(v) for v in target_pos)
        start = tuple(self.belief.agent_pos)

        if self.belief.cell_at(target) is None:
            return {
                "success": False,
                "status": "target_not_known",
                "reason": "The target has not been observed.",
                "actions_executed": 0,
            }

        # 目标改变时，清除旧路径。
        if self._active_target != target:
            self.reset_target()
            self._active_target = target

        # 只要原来的接近位置仍然有效，就尽量保持。
        if self._approach_pos is not None:
            cell = self.belief.cell_at(
                self._approach_pos
            )

            path = self._find_path(
                start,
                self._approach_pos,
            )

            if (
                not self._safe_walkable(cell)
                or path is None
            ):
                self._approach_pos = None

        if self._approach_pos is None:
            candidates = []

            for dx, dy in DIRECTIONS:
                candidate = (
                    target[0] + dx,
                    target[1] + dy,
                )

                cell = self.belief.cell_at(candidate)

                if not self._safe_walkable(cell):
                    continue

                path = self._find_path(
                    start,
                    candidate,
                )

                if path is None:
                    continue

                candidates.append(
                    (
                        len(path),
                        self._distance(start, candidate),
                        candidate,
                    )
                )

            if not candidates:
                return {
                    "success": False,
                    "status": "no_known_path",
                    "reason": (
                        "No known reachable adjacent cell "
                        "is available."
                    ),
                    "actions_executed": 0,
                }

            candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2],
                )
            )

            self._approach_pos = candidates[0][2]

        approach = self._approach_pos

        if start != approach:
            return self.step_towards_cell(approach)

        # 已经站在目标相邻格，调整朝向。
        delta = (
            target[0] - start[0],
            target[1] - start[1],
        )

        desired_direction = DIRECTIONS.index(delta)
        current_direction = int(
            self.belief.agent_dir
        )

        if current_direction != desired_direction:
            action = self._turn_action(
                current_direction,
                desired_direction,
            )

            return self._execute(
                action,
                "Facing the observed target.",
            )

        return {
            "success": True,
            "status": "target_adjacent",
            "reason": (
                "The agent is adjacent to the target "
                "and facing it."
            ),
            "target_position": list(target),
            "approach_position": list(approach),
            "actions_executed": 0,
        }
