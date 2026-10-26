from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Door, Ball, Goal


class SayCanGridEnv(MiniGridEnv):
    """支持随机物体位置和门位置的 SayCan 实验环境。"""

    def __init__(self, render_mode=None, grid_size=13):
        mission_space = MissionSpace(
            mission_func=lambda: (
                "Pick up the red ball, open the yellow door, "
                "and place the ball above the green goal."
            )
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=500,
            see_through_walls=False,
            agent_view_size=5,
            render_mode=render_mode,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # 中央隔墙，只有一扇门可以通行。
        self.wall_x = width // 2
        self.door_y = int(self.np_random.integers(2, height - 2))
        self.door_pos = (self.wall_x, self.door_y)

        for y in range(1, height - 1):
            if y != self.door_y:
                self.grid.set(self.wall_x, y, Wall())

        self.put_obj(
            Door("yellow", is_locked=False),
            *self.door_pos,
        )

        # 智能体从左侧固定位置出发。
        self.agent_pos = (2, 2)
        self.agent_dir = 0

        # 目标固定在右侧，目标放置位置为绿色目标正上方。
        self.goal_pos = (width - 2, height - 2)
        self.target_pos = (width - 2, height - 3)

        self.put_obj(Goal(), *self.goal_pos)

        # 红球随机出现在门的左侧或右侧。
        side_id = int(self.np_random.integers(0, 2))
        self.ball_side = "left" if side_id == 0 else "right"

        if self.ball_side == "left":
            x_candidates = range(1, self.wall_x)
        else:
            x_candidates = range(self.wall_x + 1, width - 1)

        excluded = {
            tuple(self.agent_pos),
            self.goal_pos,
            self.target_pos,
        }

        candidates = [
            (x, y)
            for x in x_candidates
            for y in range(1, height - 1)
            if (x, y) not in excluded
        ]

        if not candidates:
            raise RuntimeError("No valid position for the red ball.")

        ball_index = int(
            self.np_random.integers(0, len(candidates))
        )
        self.ball_pos = tuple(
            int(v) for v in candidates[ball_index]
        )

        self.put_obj(Ball("red"), *self.ball_pos)

        self.mission = (
            "Pick up the red ball, open the yellow door, "
            "and place it above the green goal."
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = (
            super().step(action)
        )

        obj = self.grid.get(*self.target_pos)
        success = isinstance(obj, Ball) and obj.color == "red"

        if success:
            reward = 1.0
            terminated = True
        elif terminated:
            reward = 0.0
            terminated = False

        info["is_success"] = success
        info["ball_pos"] = self.ball_pos
        info["ball_side"] = self.ball_side
        info["door_pos"] = self.door_pos
        info["target_pos"] = self.target_pos

        return obs, reward, terminated, truncated, info
