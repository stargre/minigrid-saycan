import numpy as np

from minigrid.core.world_object import Door

from environments.saycan_grid import SayCanGridEnv
from perception.observation_summary import observation_to_belief
from skills.movement import MovementSkill
from skills.interaction import OpenDoorSkill
from utils.visualization import FrameRecorder


env = SayCanGridEnv(render_mode="rgb_array")
env.reset(seed=42)
base = env.unwrapped

# 仅用于单元测试：在随机生成的场景中寻找一扇未上锁的黄门。
# 真实技能不得使用这里的全局地图信息。
doors = []

for x in range(base.width):
    for y in range(base.height):
        obj = base.grid.get(x, y)

        if (
            isinstance(obj, Door)
            and obj.color == "yellow"
            and not obj.is_locked
            and not obj.is_open
        ):
            doors.append((x, y))

assert doors, "No closed, unlocked yellow door in this scene."

found = None

for door_x, door_y in doors:
    neighbors = [
        (door_x, door_y - 1),
        (door_x - 1, door_y),
        (door_x + 1, door_y),
        (door_x, door_y + 1),
    ]

    for x, y in neighbors:
        if not (0 <= x < base.width and 0 <= y < base.height):
            continue

        cell = base.grid.get(x, y)
        if cell is not None and not cell.can_overlap():
            continue

        for direction in range(4):
            base.agent_pos = np.array([x, y], dtype=int)
            base.agent_dir = direction

            obs = base.gen_obs()
            belief = observation_to_belief(obs)

            visible_door = any(
                item["object"] == "door"
                and item["color"] == "yellow"
                and item.get("state") == "closed"
                and item["relative_offset"]["forward"] == 1
                and item["relative_offset"]["lateral"] == 0
                for item in belief["visible_objects"]
            )

            if visible_door:
                found = (obs, (x, y), direction)
                break

        if found is not None:
            break

    if found is not None:
        break

if found is None:
    print("Could not establish a front-facing door observation.")
    print("Door coordinates discovered by test fixture:", doors)
    raise RuntimeError(
        "Check MiniGrid view coordinates and observation parsing."
    )

obs, test_position, test_direction = found

print("=" * 60)
print("OBSERVATION-BASED OPEN DOOR TEST")
print("Test setup position:", test_position)
print("Test setup direction:", test_direction)
print("Visible door:", [
    item for item in observation_to_belief(obs)["visible_objects"]
    if item["object"] == "door"
])
print("=" * 60)

recorder = FrameRecorder(env)
movement = MovementSkill(recorder)
opener = OpenDoorSkill(recorder, movement)

# 从这里开始，技能只接收局部观测。
result = opener.step_from_observation(obs)

print("Execution result:", result)

gif_path = recorder.save_gif(
    "outputs/observation_open_door.gif",
    duration=500,
)

print("GIF:", gif_path.resolve())
print("GIF frames:", len(recorder.frames))

env.close()

assert result["success"], "Observation-based door interaction failed."
assert result["actions_executed"] == 1

print("PASS: observation-based door interaction succeeded.")
