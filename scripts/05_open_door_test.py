from pathlib import Path

from environments.saycan_grid import SayCanGridEnv
from skills.movement import MovementSkill
from skills.interaction import OpenDoorSkill
from utils.visualization import FrameRecorder


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. 初始化环境
env = SayCanGridEnv(render_mode="rgb_array")
env.reset(seed=42)

# 2. 开启逐步画面记录
recorder = FrameRecorder(env)

# 3. 创建移动和开门技能
movement = MovementSkill(recorder)
opener = OpenDoorSkill(recorder, movement)

door_pos = (4, 3)

print("Initial position:", tuple(env.unwrapped.agent_pos))
print("Target door position:", door_pos)

# 4. 执行高层开门技能
result = opener.open_door(door_pos)

print("\nOpen door result:")
for key, value in result.items():
    print(f"  {key}: {value}")

# 5. 检查门的真实状态
door = env.unwrapped.grid.get(*door_pos)

print("\nFinal position:", tuple(env.unwrapped.agent_pos))
print("Door open:", door.is_open)
print("Total low-level actions:", movement.action_count)

# 6. 保存动图
gif_path = recorder.save_gif(
    OUTPUT_DIR / "open_door_animation.gif",
    duration=700,
)

print("\nAnimation saved:", gif_path.resolve())
print("Recorded frames:", len(recorder.frames))

# 7. 自动验收
assert result["success"], "Open door skill failed!"
assert door.is_open, "The door is still closed!"

env.close()

print("\nPASS: open_door() works correctly.")
