from pathlib import Path

from environments.saycan_grid import SayCanGridEnv
from skills.movement import MovementSkill
from skills.manipulation import PickSkill, PlaceSkill
from skills.interaction import OpenDoorSkill
from utils.visualization import FrameRecorder


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

env = SayCanGridEnv(render_mode="rgb_array")
env.reset(seed=42)

# 所有技能共享同一个环境、记录器和动作计数器
recorder = FrameRecorder(env)
movement = MovementSkill(recorder)

picker = PickSkill(recorder, movement)
opener = OpenDoorSkill(recorder, movement)
placer = PlaceSkill(recorder, movement)

results = {}

print("=" * 60)
print("TASK 1: Pick up the red ball")
results["pick"] = picker.pick((2, 2))
print(results["pick"])

if not results["pick"]["success"]:
    raise RuntimeError("Task failed at pick().")

print("\n" + "=" * 60)
print("TASK 2: Open the yellow door")
results["open_door"] = opener.open_door((4, 3))
print(results["open_door"])

if not results["open_door"]["success"]:
    raise RuntimeError("Task failed at open_door().")

print("\n" + "=" * 60)
print("TASK 3: Place the red ball")
results["place"] = placer.place((6, 5))
print(results["place"])

if not results["place"]["success"]:
    raise RuntimeError("Task failed at place().")

# 最终检查
base_env = env.unwrapped
final_object = base_env.grid.get(6, 5)

print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("Final agent position:", tuple(base_env.agent_pos))
print("Door open:", base_env.grid.get(4, 3).is_open)
print("Object at target:", final_object)
print("Carrying:", base_env.carrying)
print("Total low-level actions:", movement.action_count)

assert base_env.grid.get(4, 3).is_open
assert final_object is not None
assert getattr(final_object, "color", None) == "red"
assert base_env.carrying is None

gif_path = recorder.save_gif(
    OUTPUT_DIR / "full_saycan_task.gif",
    duration=500,
)

print("Full-task GIF:", gif_path.resolve())
print("Recorded frames:", len(recorder.frames))

env.close()

print("\nPASS: complete task succeeded.")
