from minigrid.core.actions import Actions
from minigrid.core.constants import (
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
    STATE_TO_IDX,
)


OBJECT_NAMES = {v: k for k, v in OBJECT_TO_IDX.items()}
COLOR_NAMES = {v: k for k, v in COLOR_TO_IDX.items()}
STATE_NAMES = {v: k for k, v in STATE_TO_IDX.items()}

# MiniGrid 的世界坐标方向：0=东，1=南，2=西，3=北。
DIRECTIONS = [
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
]


def decode_cell(encoded):
    obj_id, color_id, state_id = (int(v) for v in encoded)
    return {
        "object": OBJECT_NAMES.get(obj_id, "unknown"),
        "color": COLOR_NAMES.get(color_id, "unknown"),
        "state": STATE_NAMES.get(state_id, "unknown"),
    }


def is_walkable(cell):
    """只根据已观测到的格子判断能否尝试进入。"""
    obj = cell["object"]

    if obj in {"empty", "floor", "goal"}:
        return True

    return obj == "door" and cell["state"] == "open"


class BeliefState:
    """
    根据局部观测和已执行动作维护环境估计。

    不读取 env.grid、env.agent_pos 或其他环境内部状态。
    初始坐标 (0, 0) 仅用于建立自己的相对坐标系。
    """

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.agent_pos = (0, 0)
        self.agent_dir = 0
        self.known_map = {}
        self.inventory_status = "unknown"
        self.completed_subgoals = []
        self.observation_count = 0
        self.last_obs = None

    def reset(self, obs, initial_inventory="empty"):
        self.reset_state()
        self.agent_dir = int(obs["direction"])
        self.inventory_status = initial_inventory
        self.last_obs = obs
        self._integrate_observation(obs)

    @staticmethod
    def _front_cell(obs):
        """读取观测中紧邻智能体正前方的格子。"""
        image = obs["image"]
        width, height, _ = image.shape
        return decode_cell(image[width // 2, height - 2])

    @staticmethod
    def _world_offset(forward, lateral, direction):
        """将自我中心坐标转换为信念地图中的相对坐标。"""
        hx, hy = DIRECTIONS[int(direction)]
        rx, ry = -hy, hx

        return (
            forward * hx + lateral * rx,
            forward * hy + lateral * ry,
        )

    def _integrate_observation(self, obs):
        """按照 MiniGrid 的 image[x, y] 索引整合局部观测。"""
        image = obs["image"]
        width, height, _ = image.shape
        center = width // 2
        direction = int(obs["direction"])

        self.observation_count += 1

        for x in range(width):
            for y in range(height):
                cell = decode_cell(image[x, y])

                if cell["object"] == "unseen":
                    continue

                forward = height - 1 - y
                lateral = x - center

                dx, dy = self._world_offset(
                    forward, lateral, direction
                )

                pos = (
                    self.agent_pos[0] + dx,
                    self.agent_pos[1] + dy,
                )

                # agent 标记只代表当前观测中的智能体位置，
                # 不应作为障碍物存入地图。
                if cell["object"] == "agent":
                    cell = {
                        "object": "empty",
                        "color": "unknown",
                        "state": "unknown",
                    }

                self.known_map[pos] = {
                    **cell,
                    "last_seen": self.observation_count,
                }

        self.known_map[self.agent_pos] = {
            "object": "empty",
            "color": "unknown",
            "state": "unknown",
            "last_seen": self.observation_count,
        }

    def update(self, previous_obs, action, next_obs):
        """
        利用动作前的观测、执行的动作和动作后的观测更新信念。

        不使用 env.step() 返回的 info，因为其中可能包含隐藏状态。
        """
        action_id = int(action)
        before_front = self._front_cell(previous_obs)
        before_inventory = self.inventory_status

        # 根据前方格子的已知通行属性估计是否发生了移动。
        if action_id == int(Actions.forward):
            if is_walkable(before_front):
                dx, dy = DIRECTIONS[
                    int(previous_obs["direction"])
                ]
                self.agent_pos = (
                    self.agent_pos[0] + dx,
                    self.agent_pos[1] + dy,
                )

        after_front = self._front_cell(next_obs)

        # 只有前方原本是红球、执行拾取后红球消失，
        # 才把持有状态更新为 red_ball。
        if (
            action_id == int(Actions.pickup)
            and before_inventory == "empty"
            and before_front["object"] == "ball"
            and before_front["color"] == "red"
            and not (
                after_front["object"] == "ball"
                and after_front["color"] == "red"
            )
        ):
            self.inventory_status = "red_ball"
            self._mark_completed("pick")

        # 只有观察到同一扇门从关闭变为打开，才确认开门。
        if (
            action_id == int(Actions.toggle)
            and before_front["object"] == "door"
            and before_front["state"] == "closed"
            and after_front["object"] == "door"
            and after_front["state"] == "open"
        ):
            self._mark_completed("open_door")

        # 只有原本持有红球，并在动作后观察到前方出现红球，
        # 才更新持有状态。是否放置在正确位置另行验证。
        if (
            action_id == int(Actions.drop)
            and before_inventory == "red_ball"
            and after_front["object"] == "ball"
            and after_front["color"] == "red"
        ):
            self.inventory_status = "empty"

        # 朝向以动作后的公开观测为准。
        self.agent_dir = int(next_obs["direction"])

        self._integrate_observation(next_obs)
        self.last_obs = next_obs

        self._check_placement()

    def _mark_completed(self, skill_name):
        if skill_name not in self.completed_subgoals:
            self.completed_subgoals.append(skill_name)

    def _check_placement(self):
        """
        通过已观测的红球和绿色目标判断是否完成放置。

        任务中的“above”按 MiniGrid 世界坐标解释为 y - 1。
        """
        goals = self.find_objects("goal", color="green")
        if not goals:
            return

        for goal_pos, _ in goals:
            target_pos = (goal_pos[0], goal_pos[1] - 1)
            target = self.known_map.get(target_pos)

            if (
                target is not None
                and target["object"] == "ball"
                and target["color"] == "red"
            ):
                self._mark_completed("place")
                return

    def find_objects(self, object_name, color=None, state=None):
        """查询自己曾经观测到的、当前记忆中仍存在的对象。"""
        matches = []

        for pos, cell in self.known_map.items():
            if cell["object"] != object_name:
                continue
            if color is not None and cell["color"] != color:
                continue
            if state is not None and cell["state"] != state:
                continue

            matches.append((pos, cell))

        return matches

    def is_known(self, pos):
        return tuple(pos) in self.known_map

    def cell_at(self, pos):
        return self.known_map.get(tuple(pos))

    def summary(self):
        """生成适合日志或规划器使用的状态摘要。"""
        return {
            "estimated_position": list(self.agent_pos),
            "direction": self.agent_dir,
            "known_cell_count": len(self.known_map),
            "inventory_status": self.inventory_status,
            "completed_subgoals": list(self.completed_subgoals),
            "observation_count": self.observation_count,
        }
