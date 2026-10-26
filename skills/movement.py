from collections import deque

from minigrid.core.actions import Actions


# MiniGrid 的方向定义：
# 0 = 右，1 = 下，2 = 左，3 = 上
DIRECTIONS = [
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
]

DIRECTION_TO_ID = {
    delta: direction
    for direction, delta in enumerate(DIRECTIONS)
}


class MovementSkill:
    """基于 BFS 的网格移动技能。"""

    def __init__(self, env):
        self.env = env
        self.base_env = env.unwrapped
        self.action_count = 0

    def _is_walkable(self, pos):
        """判断指定格子当前是否允许智能体进入。"""
        x, y = pos

        if not (
            0 <= x < self.base_env.width
            and 0 <= y < self.base_env.height
        ):
            return False

        obj = self.base_env.grid.get(x, y)

        # 空格可以通过；其他物体根据其碰撞规则判断。
        return obj is None or obj.can_overlap()

    def _find_path(self, start, target):
        """使用 BFS 寻找最短路径，返回包含起终点的坐标列表。"""

        if not self._is_walkable(target):
            return None

        queue = deque([start])
        parent = {start: None}

        while queue:
            current = queue.popleft()

            if current == target:
                break

            x, y = current

            for dx, dy in DIRECTIONS:
                neighbor = (x + dx, y + dy)

                if neighbor in parent:
                    continue

                if not self._is_walkable(neighbor):
                    continue

                parent[neighbor] = current
                queue.append(neighbor)

        if target not in parent:
            return None

        # 从目标沿 parent 指针回溯到起点
        path = []
        current = target

        while current is not None:
            path.append(current)
            current = parent[current]

        return list(reversed(path))

    def _execute(self, action):
        """执行一个底层动作，并返回执行结果。"""
        result = self.env.step(int(action))
        self.action_count += 1

        obs, reward, terminated, truncated, info = result

        return {
            "obs": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    def _turn_to(self, desired_direction):
        """通过左右转向，让智能体面朝指定方向。"""

        while int(self.base_env.agent_dir) != desired_direction:
            current = int(self.base_env.agent_dir)
            difference = (desired_direction - current) % 4

            # 选择较短的转向路径；相差 180 度时统一向右转。
            if difference in (1, 2):
                action = Actions.right
            else:
                action = Actions.left

            result = self._execute(action)

            if result["terminated"] or result["truncated"]:
                return False

        return True

    def move_to(self, target):
        """将智能体移动到指定坐标。"""

        target = tuple(target)
        start = tuple(self.base_env.agent_pos)
        initial_action_count = self.action_count

        path = self._find_path(start, target)

        if path is None:
            return {
                "success": False,
                "reason": "Target is blocked or unreachable.",
                "start": start,
                "target": target,
                "path": None,
                "actions_executed": 0,
            }

        for next_pos in path[1:]:
            current_pos = tuple(self.base_env.agent_pos)

            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]

            desired_direction = DIRECTION_TO_ID[(dx, dy)]

            if not self._turn_to(desired_direction):
                return {
                    "success": False,
                    "reason": "Episode ended while turning.",
                    "start": start,
                    "target": target,
                    "path": path,
                    "actions_executed": (
                        self.action_count - initial_action_count
                    ),
                }

            result = self._execute(Actions.forward)

            if result["terminated"] or result["truncated"]:
                return {
                    "success": False,
                    "reason": "Episode ended while moving.",
                    "start": start,
                    "target": target,
                    "path": path,
                    "actions_executed": (
                        self.action_count - initial_action_count
                    ),
                }

            # 检查环境中的真实位置是否符合预期
            actual_pos = tuple(self.base_env.agent_pos)

            if actual_pos != next_pos:
                return {
                    "success": False,
                    "reason": "Actual position differs from planned position.",
                    "start": start,
                    "target": target,
                    "path": path,
                    "actions_executed": (
                        self.action_count - initial_action_count
                    ),
                }

        return {
            "success": tuple(self.base_env.agent_pos) == target,
            "reason": "Arrived." if tuple(self.base_env.agent_pos) == target
                      else "Target not reached.",
            "start": start,
            "target": target,
            "path": path,
            "actions_executed": (
                self.action_count - initial_action_count
            ),
        }

    # Public interfaces for higher-level skills.
    def find_path(self, start, target):
        return self._find_path(tuple(start), tuple(target))

    def turn_to(self, desired_direction):
        return self._turn_to(desired_direction)

    def execute_action(self, action):
        return self._execute(action)
