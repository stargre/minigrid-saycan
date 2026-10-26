from minigrid.core.constants import (
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
    STATE_TO_IDX,
)
from minigrid.core.actions import Actions

from environments.saycan_grid import SayCanGridEnv
from skills.movement import MovementSkill
from skills.exploration import ExploreSkill
from utils.visualization import FrameRecorder


OBJECT_NAMES = {v: k for k, v in OBJECT_TO_IDX.items()}
COLOR_NAMES = {v: k for k, v in COLOR_TO_IDX.items()}
STATE_NAMES = {v: k for k, v in STATE_TO_IDX.items()}

ACTION_NAMES = {
    int(Actions.left): "left",
    int(Actions.right): "right",
    int(Actions.forward): "forward",
    int(Actions.pickup): "pickup",
    int(Actions.drop): "drop",
    int(Actions.toggle): "toggle",
    int(Actions.done): "done",
}


def visible_objects(obs):
    image = obs["image"]
    width, height, _ = image.shape
    result = []

    for x in range(width):
        for y in range(height):
            obj_id, color_id, state_id = (
                int(v) for v in image[x, y]
            )

            obj = OBJECT_NAMES.get(obj_id, "unknown")
            color = COLOR_NAMES.get(color_id, "unknown")
            state = STATE_NAMES.get(state_id, "unknown")

            if obj in {"door", "ball", "goal"}:
                result.append({
                    "object": obj,
                    "color": color,
                    "state": state,
                    "forward": height - 1 - y,
                    "lateral": x - width // 2,
                    "cell_xy": (x, y),
                })

    return result


env = SayCanGridEnv(render_mode="rgb_array")
obs, _ = env.reset(seed=42)

recorder = FrameRecorder(env)
movement = MovementSkill(recorder)
explorer = ExploreSkill(movement)

base = env.unwrapped

print("=" * 70)
print("EXPLORATION DIAGNOSTICS")
print("Observation shape:", obs["image"].shape)
print("Initial observation direction:", int(obs["direction"]))
print("Initial visible objects:", visible_objects(obs))
print("=" * 70)

for i in range(12):
    before_obs = obs
    before_dir = int(before_obs["direction"])
    before_visible = visible_objects(before_obs)

    result = explorer.step(
        before_obs,
        inventory_status="empty",
        completed_subgoals=[],
    )

    obs = result["obs"]
    after_dir = int(obs["direction"])

    action = result.get("action")
    action_name = (
        ACTION_NAMES.get(action, str(action))
        if action is not None
        else "NONE"
    )

    print(f"\nDecision {i + 1}")
    print("  Action:", action_name)
    print(f"  Direction: {before_dir} -> {after_dir}")
    print("  Selected target:", result.get("target"))
    print("  Before:", before_visible)
    print("  After: ", visible_objects(obs))

    # 仅供离线诊断，绝不传入 ExploreSkill 或规划器。
    print(
        "  DEBUG ONLY - true agent position:",
        tuple(int(v) for v in base.agent_pos),
    )
    print(
        "  DEBUG ONLY - true door position:",
        getattr(base, "door_pos", None),
    )

    if result["status"] == "target_adjacent":
        print("Target is adjacent; stopping diagnostic.")
        break

    if result.get("terminated") or result.get("truncated"):
        print("Episode ended.")
        break

print("\nDiagnostics completed.")
env.close()
