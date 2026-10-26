from collections import deque

from minigrid.core.actions import Actions
from perception.belief_state import DIRECTIONS, is_walkable


class ExploreForDoorSkill:
    """
    基于 BeliefState 的门搜索技能。

    约束：
    1. 只读取观测建立的信念地图。
    2. 不读取环境全局地图、真实坐标或 step() 的 info。
    3. 通过 frontier 探索未知区域。
    4. 每次只执行一个底层动作，然后重新观测。
    5. 一旦发现目标门，立即交还控制权。
    """

    def __init__(self, runtime, color="yellow"):
        self.runtime = runtime
        self.belief = runtime.belief
        self.color = color

        self.frontier_pos = None
        self.frontier_unknown = None
        self.attempted_frontiers = set()
        self.action_count = 0

    @staticmethod
    def _neighbors(pos):
        x, y = pos
        return [
            (x + dx, y + dy)
            for dx, dy in DIRECTIONS
        ]

    def _find_door(self):
        """只从已经观测并写入信念地图的对象中寻找门。"""
        doors = self.belief.find_objects(
            "door",
            color=self.color,
        )

        if not doors:
            return None

        start = tuple(self.belief.agent_pos)

        return min(
            doors,
            key=lambda item: (
                abs(item[0][0] - start[0])
                + abs(item[0][1] - start[1])
            ),
        )

    def _target_found(self):
        found = self._find_door()

        if found is None:
            return None

        _, cell = found

        return {
            "success": True,
            "status": "target_found",
            "reason": (
                "The target door was observed and recorded "
                "in the belief map."
            ),
            "target": {
                "object": "door",
                "color": self.color,
                "state": cell.get("state", "unknown"),
            },
            "actions_executed": 0,
            "known_cell_count": len(self.belief.known_map),
        }

    def _find_path(self, start, goal):
        """只经过信念地图中已知且可通行的格子。"""
        start = tuple(start)
        goal = tuple(goal)

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

    def _frontier_is_valid(self):
        if (
            self.frontier_pos is None
            or self.frontier_unknown is None
        ):
            return False

        pos = tuple(self.frontier_pos)
        unknown = tuple(self.frontier_unknown)
        pair = (pos, unknown)

        cell = self.belief.cell_at(pos)

        return (
            cell is not None
            and is_walkable(cell)
            and self.belief.cell_at(unknown) is None
            and pair not in self.attempted_frontiers
        )

    def _select_frontier(self):
        """
        选择距离当前智能体最近的可达 frontier。

        frontier 是已知可通行格与未知格相邻的边界。
        规划只使用 known_map，不推测未知格内部是什么。
        """
        start = tuple(self.belief.agent_pos)
        candidates = []
        path_cache = {}

        for pos, cell in list(self.belief.known_map.items()):
            pos = tuple(pos)

            if not is_walkable(cell):
                continue

            path = self._find_path(start, pos)

            if path is None:
                continue

            path_cache[pos] = path

            for dx, dy in DIRECTIONS:
                unknown = (pos[0] + dx, pos[1] + dy)
                pair = (pos, unknown)

                if self.belief.cell_at(unknown) is not None:
                    continue

                if pair in self.attempted_frontiers:
                    continue

                candidates.append(
                    (len(path) - 1, pos, unknown)
                )

        if not candidates:
            return None

        # 先选择路径最短的边界；同分时使用稳定的排序，
        # 避免每轮随机切换目标。
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1][0],
                item[1][1],
                item[2][0],
                item[2][1],
            )
        )

        _, self.frontier_pos, self.frontier_unknown = (
            candidates[0]
        )

        return self.frontier_pos, self.frontier_unknown

    @staticmethod
    def _turn_action(current_direction, desired_direction):
        difference = (
            desired_direction - current_direction
        ) % 4

        if difference == 0:
            return None

        if difference == 3:
            return Actions.left

        # 包括相差 180 度的情况：每次只向右转一步，
        # 下一轮继续检查当前朝向。
        return Actions.right

    def _execute(self, action):
        """唯一的动作执行入口；不读取环境隐藏状态。"""
        result = self.runtime.execute_action(action)
        self.action_count += 1

        found = self._target_found()

        if found is not None:
            found.update({
                "action": action.name,
                "actions_executed": 1,
                "obs": result["obs"],
                "estimated_position": list(
                    self.belief.agent_pos
                ),
                "known_cell_count": len(
                    self.belief.known_map
                ),
            })
            return found

        if result["terminated"] or result["truncated"]:
            status = "episode_ended"
        else:
            status = "action_executed"

        return {
            "success": False,
            "status": status,
            "action": action.name,
            "actions_executed": 1,
            "obs": result["obs"],
            "estimated_position": list(
                self.belief.agent_pos
            ),
            "known_cell_count": len(
                self.belief.known_map
            ),
        }

    def step(self):
        """
        执行一次搜索决策。

        返回：
        - target_found：已观测到目标门
        - action_executed：完成一个探索动作
        - exploration_exhausted：没有可继续探索的 frontier
        - episode_ended：环境回合已经结束
        """
        found = self._target_found()

        if found is not None:
            return found

        # 在同一位置跳过已经检查过的 frontier，
        # 避免不断朝同一个未知方向转圈。
        for _ in range(64):
            if not self._frontier_is_valid():
                self.frontier_pos = None
                self.frontier_unknown = None

            if self.frontier_pos is None:
                selected = self._select_frontier()

                if selected is None:
                    return {
                        "success": False,
                        "status": "exploration_exhausted",
                        "reason": (
                            "No reachable unattempted frontier "
                            "remains in the current belief map."
                        ),
                        "actions_executed": 0,
                        "known_cell_count": len(
                            self.belief.known_map
                        ),
                    }

            start = tuple(self.belief.agent_pos)
            frontier = tuple(self.frontier_pos)
            unknown = tuple(self.frontier_unknown)
            pair = (frontier, unknown)

            path = self._find_path(start, frontier)

            if path is None:
                self.attempted_frontiers.add(pair)
                self.frontier_pos = None
                self.frontier_unknown = None
                continue

            # 第一步：沿已知可通行路线前往 frontier。
            if start != frontier:
                next_pos = path[1]
                delta = (
                    next_pos[0] - start[0],
                    next_pos[1] - start[1],
                )
                desired = DIRECTIONS.index(delta)

                action = (
                    Actions.forward
                    if self.belief.agent_dir == desired
                    else self._turn_action(
                        self.belief.agent_dir,
                        desired,
                    )
                )

                return self._execute(action)

            # 第二步：到达 frontier 后，朝未知邻格观察。
            delta = (
                unknown[0] - frontier[0],
                unknown[1] - frontier[1],
            )
            desired = DIRECTIONS.index(delta)

            if self.belief.agent_dir == desired:
                # 当前观测已经朝向这个格子，但它仍然未知。
                # 不再重复转向，换一个 frontier。
                self.attempted_frontiers.add(pair)
                self.frontier_pos = None
                self.frontier_unknown = None
                continue

            action = self._turn_action(
                self.belief.agent_dir,
                desired,
            )
            result = self._execute(action)

            if result["status"] == "target_found":
                return result

            # 只有实际转到目标方向后，才判断这个未知格
            # 是否仍不可见。如果仍未知，记录这次尝试。
            if self.belief.agent_dir == desired:
                if self.belief.cell_at(unknown) is None:
                    self.attempted_frontiers.add(pair)

                self.frontier_pos = None
                self.frontier_unknown = None

            elif self.belief.cell_at(unknown) is not None:
                self.frontier_pos = None
                self.frontier_unknown = None

            return result

        return {
            "success": False,
            "status": "exploration_exhausted",
            "reason": "Could not find a new frontier to explore.",
            "actions_executed": 0,
            "known_cell_count": len(self.belief.known_map),
        }
