from minigrid.core.constants import (
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
    STATE_TO_IDX,
)

OBJECT_NAMES = {v: k for k, v in OBJECT_TO_IDX.items()}
COLOR_NAMES = {v: k for k, v in COLOR_TO_IDX.items()}
STATE_NAMES = {v: k for k, v in STATE_TO_IDX.items()}


def observation_to_belief(
    obs,
    inventory_status="unknown",
    completed_subgoals=None,
):
    """
    仅根据 MiniGrid 的局部观测和已知执行历史构建 belief。

    MiniGrid 编码图像的空间索引为 image[x, y]：
      x 轴：局部视野的横向位置
      y 轴：局部视野的纵向位置，越靠上越远离智能体

    不读取环境内部状态、全局坐标或完整地图。
    """
    image = obs["image"]

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Unexpected observation shape: {image.shape}"
        )

    width, height, channels = image.shape

    visible_objects = []
    visible_obstacles = []

    ignored = {"unseen", "empty", "floor", "agent"}
    center_x = width // 2

    # 注意：MiniGrid 使用 image[x, y]，不是 image[row, col]。
    for x in range(width):
        for y in range(height):
            obj_id = int(image[x, y, 0])
            color_id = int(image[x, y, 1])
            state_id = int(image[x, y, 2])

            obj = OBJECT_NAMES.get(obj_id, "unknown")

            if obj in ignored:
                continue

            color = COLOR_NAMES.get(color_id, "unknown")

            # 视野已经旋转为智能体朝前的规范坐标系。
            forward = height - 1 - y
            lateral = x - center_x

            if lateral < 0:
                side = f"{abs(lateral)} cells left"
            elif lateral > 0:
                side = f"{lateral} cells right"
            else:
                side = "center"

            if forward > 0:
                relative_position = (
                    f"{forward} cells forward, {side}"
                )
            else:
                relative_position = side

            item = {
                "object": obj,
                "color": color,
                "relative_position": relative_position,
                "relative_offset": {
                    "forward": int(forward),
                    "lateral": int(lateral),
                },
            }

            if obj == "door":
                item["state"] = STATE_NAMES.get(
                    state_id, "unknown"
                )

            if obj == "wall":
                visible_obstacles.append(item)
            else:
                visible_objects.append(item)

    return {
        "mission": str(obs.get("mission", "")),
        "view_shape": [int(height), int(width)],
        "visible_objects": visible_objects,
        "visible_obstacles": visible_obstacles,
        "inventory_status": inventory_status,
        "completed_subgoals": list(completed_subgoals or []),
        "information_limits": [
            "Only the current local observation is available.",
            "Unobserved object positions and map cells are unknown.",
            "No global environment coordinates are provided.",
        ],
    }
