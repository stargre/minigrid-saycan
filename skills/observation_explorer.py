from minigrid.core.actions import Actions
from perception.belief_state import DIRECTIONS

from skills.navigation import ObservationNavigator


class ObservationExplorer:
    """
    通用观测驱动探索器。

    只在已知地图中移动。
    到达探索前沿后转向未知区域，利用新观测扩展地图。
    不读取完整地图或隐藏状态。
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.belief = runtime.belief
        self.navigator = ObservationNavigator(runtime)

        self.active_frontier = None
        self.pending_frontier = None
        self.blocked_edges = set()

    @staticmethod
    def _distance(a, b):
        return (
            abs(a[0] - b[0])
            + abs(a[1] - b[1])
        )

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
            "known_cells": len(
                self.belief.known_map
            ),
        }

    def _update_pending_frontier(self):
        if self.pending_frontier is None:
            return

        edge = self.pending_frontier
        _, unknown_pos = edge

        # 转向后，若新观测仍没有揭示该格，
        # 不要永远重复尝试同一条边。
        if unknown_pos not in self.belief.known_map:
            self.blocked_edges.add(edge)

        if self.active_frontier == edge:
            self.active_frontier = None

        self.pending_frontier = None

    def _candidate_frontiers(self):
        start = tuple(self.belief.agent_pos)
        candidates = []

        for pos, cell in list(
            self.belief.known_map.items()
        ):
            if not self.navigator._safe_walkable(cell):
                continue

            path = self.navigator._find_path(
                start,
                pos,
            )

            if path is None:
                continue

            for direction_id, (dx, dy) in enumerate(
                DIRECTIONS
            ):
                unknown_pos = (
                    pos[0] + dx,
                    pos[1] + dy,
                )

                edge = (pos, unknown_pos)

                if unknown_pos in self.belief.known_map:
                    continue

                if edge in self.blocked_edges:
                    continue

                candidates.append(
                    (
                        len(path),
                        self._distance(start, pos),
                        pos[0],
                        pos[1],
                        direction_id,
                        pos,
                        unknown_pos,
                    )
                )

        return candidates

    def step(self):
        self._update_pending_frontier()

        # 有界循环：即使连续遇到不可探索的边缘，
        # 也不会无限循环在同一次调用中。
        max_attempts = max(
            16,
            len(self.belief.known_map) * 4,
        )

        for _ in range(max_attempts):
            start = tuple(self.belief.agent_pos)

            # 优先继续上次选择的探索前沿。
            edge = self.active_frontier

            if edge is not None:
                frontier_pos, unknown_pos = edge

                cell = self.belief.cell_at(frontier_pos)
                path = self.navigator._find_path(
                    start,
                    frontier_pos,
                )

                if (
                    unknown_pos in self.belief.known_map
                    or edge in self.blocked_edges
                    or not self.navigator._safe_walkable(cell)
                    or path is None
                ):
                    self.active_frontier = None

            if self.active_frontier is None:
                candidates = self._candidate_frontiers()

                if not candidates:
                    return {
                        "success": False,
                        "status": "exploration_exhausted",
                        "reason": (
                            "No reachable unexplored frontier "
                            "remains in the known map."
                        ),
                        "actions_executed": 0,
                    }

                candidates.sort(
                    key=lambda item: item[:5]
                )

                selected = candidates[0]
                frontier_pos = selected[5]
                unknown_pos = selected[6]

                self.active_frontier = (
                    frontier_pos,
                    unknown_pos,
                )

            frontier_pos, unknown_pos = (
                self.active_frontier
            )

            path = self.navigator._find_path(
                start,
                frontier_pos,
            )

            if path is None:
                self.active_frontier = None
                continue

            # 先沿已知路径前往前沿位置。
            if start != frontier_pos:
                result = self.navigator.step_towards_cell(
                    frontier_pos
                )

                if result["status"] == "action_executed":
                    result["skill"] = "explore"
                    return result

                self.active_frontier = None
                continue

            # 已经到达前沿，转向未知区域。
            delta = (
                unknown_pos[0] - frontier_pos[0],
                unknown_pos[1] - frontier_pos[1],
            )

            desired_direction = DIRECTIONS.index(delta)

            if (
                self.belief.agent_dir
                != desired_direction
            ):
                self.pending_frontier = (
                    frontier_pos,
                    unknown_pos,
                )

                return self._execute(
                    Actions.right
                    if (
                        desired_direction
                        - self.belief.agent_dir
                    ) % 4 in (1, 2)
                    else Actions.left,
                    "Turning toward an unexplored frontier.",
                )

            # 已经面向该格但仍然无法观测时，
            # 放弃这条边并尝试其他前沿。
            self.blocked_edges.add(
                (frontier_pos, unknown_pos)
            )
            self.active_frontier = None

        return {
            "success": False,
            "status": "exploration_stalled",
            "reason": (
                "The explorer could not find a productive "
                "frontier within the attempt limit."
            ),
            "actions_executed": 0,
        }
