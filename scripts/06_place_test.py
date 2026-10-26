from pathlib import Path

from minigrid.core.world_object import Ball

from environments.saycan_grid import SayCanGridEnv
from skills.movement import MovementSkill
from skills.manipulation import PlaceSkill
from utils.visualization import FrameRecorder


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

env = SayCanGridEnv(render_mode="rgb_array")
env.reset(seed=42)

# 独立测试 place()：提前打开通往目标区域的门
env.unwrapped.grid.get(4, 3).is_open = True


# 将红球放到智能体手中，用于独立测试 place()
ball_pos = (2, 2)
ball = env.unwrapped.grid.get(*ball_pos)

assert isinstance(ball, Ball)
assert ball.color == "red"

env.unwrapped.grid.set(*ball_pos, None)
env.unwrapped.carrying = ball

recorder = FrameRecorder(env)
movement = MovementSkill(recorder)
placer = PlaceSkill(recorder, movement)

target_pos = (6, 5)

print("Initial position:", tuple(env.unwrapped.agent_pos))
print("Target position:", target_pos)
print("Initially carrying:", env.unwrapped.carrying.color)

result = placer.place(target_pos)

print("\nPlace result:")
for key, value in result.items():
    print(f"  {key}: {value}")

print("\nFinal position:", tuple(env.unwrapped.agent_pos))
print("Carrying:", env.unwrapped.carrying)
print("Object at target:", env.unwrapped.grid.get(*target_pos))

gif_path = recorder.save_gif(
    OUTPUT_DIR / "place_animation.gif",
    duration=700,
)

print("\nAnimation saved:", gif_path.resolve())
print("Recorded frames:", len(recorder.frames))

assert result["success"], "Place skill failed!"
assert env.unwrapped.carrying is None
assert env.unwrapped.grid.get(*target_pos) is ball

env.close()

print("\nPASS: place() works correctly.")
