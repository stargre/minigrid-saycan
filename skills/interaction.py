from minigrid.core.actions import Actions
from perception.observation_summary import observation_to_belief


class OpenDoorSkill:
    """
    基于当前局部观测执行开门操作。

    所有动作均通过 ObservationRuntime 执行，
    不读取全局地图、真实门坐标或环境 info。
    """

    def __init__(self, runtime, color="yellow"):
        self.runtime = runtime
        self.color = color

    def step(self):
        obs = self.runtime.last_obs

        if obs is None:
            raise RuntimeError(
                "Call runtime.reset(obs) before opening the door."
            )

        belief = observation_to_belief(obs)

        doors = [
            item
            for item in belief["visible_objects"]
            if item["object"] == "door"
            and item["color"] == self.color
        ]

        if not doors:
            return {
                "success": False,
                "status": "target_not_visible",
                "reason": "The target door is not visible.",
                "actions_executed": 0,
            }

        # 只依据当前观测选择最近的目标门。
        door = min(
            doors,
            key=lambda item: (
                abs(item["relative_offset"]["forward"])
                + abs(item["relative_offset"]["lateral"])
            ),
        )

        offset = door["relative_offset"]
        forward = int(offset["forward"])
        lateral = int(offset["lateral"])
        state = door.get("state", "unknown")

        target = {
            "object": "door",
            "color": self.color,
            "state": state,
            "relative_offset": {
                "forward": forward,
                "lateral": lateral,
            },
        }

        if state == "open":
            return {
                "success": True,
                "status": "skill_complete",
                "reason": "The door is already open.",
                "target": target,
                "door_open": True,
                "actions_executed": 0,
            }

        if state == "locked":
            return {
                "success": False,
                "status": "requires_key",
                "reason": "The door is locked.",
                "target": target,
                "actions_executed": 0,
            }

        if state != "closed":
            return {
                "success": False,
                "status": "unknown_door_state",
                "reason": "The door state cannot be determined.",
                "target": target,
                "actions_executed": 0,
            }

        # MiniGrid 的 toggle 只能与正前方的门交互。
        if forward != 1 or lateral != 0:
            return {
                "success": False,
                "status": "needs_approach",
                "reason": (
                    "The door is not directly in front. "
                    "ApproachDoorSkill must approach it first."
                ),
                "target": target,
                "actions_executed": 0,
            }

        # 唯一的底层动作入口：执行动作并更新信念状态。
        result = self.runtime.execute_action(Actions.toggle)
        next_obs = result["obs"]

        # 只通过执行后的新观测验证开门结果。
        next_belief = observation_to_belief(next_obs)

        opened = any(
            item["object"] == "door"
            and item["color"] == self.color
            and item.get("state") == "open"
            and item["relative_offset"]["forward"] == 1
            and item["relative_offset"]["lateral"] == 0
            for item in next_belief["visible_objects"]
        )

        return {
            "success": opened,
            "status": (
                "skill_complete"
                if opened
                else "open_not_confirmed"
            ),
            "reason": (
                "Door opening confirmed by the next observation."
                if opened
                else "Toggle executed, but opening was not confirmed."
            ),
            "target": target,
            "door_open": opened,
            "action": "toggle",
            "actions_executed": result["actions_executed"],
            "obs": next_obs,
        }
