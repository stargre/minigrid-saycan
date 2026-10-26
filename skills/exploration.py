
from collections import deque, defaultdict

from minigrid.core.actions import Actions
from minigrid.core.constants import (
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
    STATE_TO_IDX,
)

OBJECT_NAMES = {v: k for k, v in OBJECT_TO_IDX.items()}
COLOR_NAMES = {v: k for k, v in COLOR_TO_IDX.items()}
STATE_NAMES = {v: k for k, v in STATE_TO_IDX.items()}

# MiniGrid 世界坐标：0=东，1=南，2=西，3=北
DIRECTIONS = [
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
]
DIRECTION_TO_ID = {
    delta: i for i, delta in enumerate(DIRECTIONS)
}


class ExploreSkill:
    """基于当前局部观测进行目标接近与探索。"""

    def __init__(self, movement):
        self.movement = movement
        self.view_counts = defaultdict(int)
        self.action_count = 0

    @staticmethod
    def _decode(image, x, y):
        """注意：MiniGrid image 的前两个轴是 x、y，不是 row、column。"""
        obj_id, color_id, state_id = (
            int(v) for v in image[x, y]
        )
        return {
            "object": OBJECT_NAMES.get(obj_id, "unknown"),
            "color": COLOR_NAMES.get(color_id, "unknown"),
            "state": STATE_NAMES.get(state_id, "unknown"),
        }

    @staticmethod
    def _world_offset(forward, lateral, direction):
        """将智能体视角的前向、横向偏移转为世界坐标偏移。"""
        hx, hy = DIRECTIONS[int(direction)]
        rx, ry = -hy, hx

        return (
            forward * hx + lateral * rx,
            forward * hy + lateral * ry,
        )

    @staticmethod
    def _walkable(cell):
        if cell["object"] in {"empty", "floor", "goal"}:
            return True

        return (
            cell["object"] == "door"
            and cell["state"] == "open"
        )

    def _observed_map(self, obs):
        """
        构建当前观测中已知的局部地图。
        坐标以当前智能体所在格为原点。
        """
        image = obs["image"]
        width, height, _ = image.shape
        direction = int(obs["direction"])
        known = {}

        # 正确使用 image[x, y]。
        for x in range(width):
            for y in range(height):
                cell = self._decode(image, x, y)

                if cell["object"] == "unseen":
                    continue

                forward = height - 1 - y
                lateral = x - width // 2

                offset = self._world_offset(
                    forward, lateral, direction
                )
                known[offset] = cell

        known[(0, 0)] = {
            "object": "empty",
            "color": "unknown",
            "state": "unknown",
        }
        return known

    def _find_target(
        self,
        obs,
        inventory_status,
        completed_subgoals,
    ):
        image = obs["image"]
        width, height, _ = image.shape
        direction = int(obs["direction"])
        completed = set(completed_subgoals or [])
        candidates = []

        # 这里同样必须使用 image[x, y]。
        for x in range(width):
            for y in range(height):
                cell = self._decode(image, x, y)
                obj = cell["object"]
                color = cell["color"]
                state = cell["state"]

                name = None
                priority = 99

                if (
                    obj == "ball"
                    and color == "red"
                    and inventory_status != "red_ball"
                    and "pick" not in completed
                ):
                    name = "pick"
                    priority = 0

                elif (
                    obj == "door"
                    and color == "yellow"
                    and state != "open"
                    and "open_door" not in completed
                ):
                    name = "open_door"
                    priority = 1

                elif (
                    obj == "goal"
                    and color == "green"
                    and inventory_status == "red_ball"
                    and "place" not in completed
                ):
                    name = "place"
                    priority = 2

                if name is None:
                    continue

                forward = height - 1 - y
                lateral = x - width // 2

                dx, dy = self._world_offset(
                    forward, lateral, direction
                )

                candidates.append({
                    "name": name,
                    "object": obj,
                    "color": color,
                    "state": state,
                    "forward": forward,
                    "lateral": lateral,
                    "world_dx": dx,
                    "world_dy": dy,
                    "distance": abs(dx) + abs(dy),
                    "priority": priority,
                })

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda t: (t["distance"], t["priority"]),
        )

    def _path_to_adjacent(self, known, target_pos, direction):
        """在当前可见地图中寻找通往目标相邻格的最短路径。"""
        start = (0, 0)
        tx, ty = target_pos

        goals = set()

        for dx, dy in DIRECTIONS:
            pos = (tx + dx, ty + dy)

            if pos == start:
                goals.add(pos)
            elif pos in known and self._walkable(known[pos]):
                goals.add(pos)

        if not goals:
            return None

        if start in goals:
            return [start]

        # 优先沿当前朝向前进，再尝试右、左、后方。
        neighbor_order = [
            DIRECTIONS[(direction + i) % 4]
            for i in (0, 1, 3, 2)
        ]

        queue = deque([start])
        parent = {start: None}
        reached = None

        while queue:
            current = queue.popleft()

            if current in goals:
                reached = current
                break

            for dx, dy in neighbor_order:
                neighbor = (
                    current[0] + dx,
                    current[1] + dy,
                )

                if neighbor in parent or neighbor not in known:
                    continue

                if neighbor == target_pos:
                    continue

                if not self._walkable(known[neighbor]):
                    continue

                parent[neighbor] = current
                queue.append(neighbor)

        if reached is None:
            return None

        path = []
        current = reached

        while current is not None:
            path.append(current)
            current = parent[current]

        return list(reversed(path))

    @staticmethod
    def _turn_action(current_direction, desired_direction):
        diff = (desired_direction - current_direction) % 4

        if diff == 0:
            return None
        if diff == 1:
            return Actions.right
        if diff == 3:
            return Actions.left

        # 相差 180° 时先右转一次，下一轮再重新判断。
        return Actions.right

    def _act(self, action):
        result = self.movement.execute_action(action)
        self.action_count += 1

        return {
            "status": "action_executed",
            "action": int(action),
            "obs": result["obs"],
            "terminated": result["terminated"],
            "truncated": result["truncated"],
        }

    def _approach_visible_target(self, obs, target):
        direction = int(obs["direction"])
        dx = target["world_dx"]
        dy = target["world_dy"]
        distance = abs(dx) + abs(dy)

        # 已经相邻：必须正对目标，才能交给操作技能。
        if distance == 1:
            desired = DIRECTION_TO_ID[(dx, dy)]

            if direction == desired:
                return {
                    "status": "target_adjacent",
                    "target": target,
                    "obs": obs,
                    "reason": "Target directly ahead and adjacent.",
                }

            result = self._act(
                self._turn_action(direction, desired)
            )
            result["target"] = target
            return result

        known = self._observed_map(obs)
        path = self._path_to_adjacent(
            known, (dx, dy), direction
        )

        if path is None:
            # 仅在当前局部地图无路可走时转向重新观察。
            result = self._act(Actions.right)
            result["target"] = target
            result["reason"] = "No local path; re-observing."
            return result

        if len(path) < 2:
            return {
                "status": "target_adjacent",
                "target": target,
                "obs": obs,
                "reason": "Already at an adjacent cell.",
            }

        next_pos = path[1]
        delta = (
            next_pos[0] - path[0][0],
            next_pos[1] - path[0][1],
        )
        desired = DIRECTION_TO_ID[delta]

        if direction == desired:
            action = Actions.forward
        else:
            action = self._turn_action(direction, desired)

        result = self._act(action)
        result["target"] = target
        result["local_path_length"] = len(path) - 1
        return result

    def _explore_unknown_area(self, obs):
        """只有当前没有可见任务目标时才进行探索。"""
        image = obs["image"]
        direction = int(obs["direction"])
        signature = (image.tobytes(), direction)
        self.view_counts[signature] += 1

        known = self._observed_map(obs)
        front = DIRECTIONS[direction]
        front_cell = known.get(front)

        if (
            front_cell is not None
            and self._walkable(front_cell)
            and self.view_counts[signature] <= 1
        ):
            return self._act(Actions.forward)

        return self._act(Actions.right)

    def step(
        self,
        obs,
        inventory_status="empty",
        completed_subgoals=None,
    ):
        target = self._find_target(
            obs,
            inventory_status,
            completed_subgoals,
        )

        if target is not None:
            return self._approach_visible_target(obs, target)

        result = self._explore_unknown_area(obs)
        result["target"] = None
        return result
