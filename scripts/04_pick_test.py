from pathlib import Path

from environments.saycan_grid import SayCanGridEnv
from skills.movement import MovementSkill
from skills.manipulation import PickSkill
from utils.visualization import FrameRecorder


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. 初始化环境
env = SayCanGridEnv(render_mode="rgb_array")
env.reset(seed=42)

# 2. 开启可视化记录
recorder = FrameRecorder(env)

# 3. 创建技能
movement = MovementSkill(recorder)
picker = PickSkill(recorder, movement)

print("Initial position:", tuple(env.unwrapped.agent_pos))
print("Target object position:", (2, 2))

# 4. 执行拾取
result = picker.pick((2, 2))

print("\nPick result:")
for key, value in result.items():
    print(f"  {key}: {value}")

# 5. 检查实际持有状态
carrying = env.unwrapped.carrying

print("\nFinal position:", tuple(env.unwrapped.agent_pos))
print(
    "Carrying object:",
    type(carrying).__name__ if carrying else None,
)
print(
    "Carrying color:",
    getattr(carrying, "color", None) if carrying else None,
)
print("Total low-level actions:", movement.action_count)

# 6. 保存完整动作 GIF
gif_path = recorder.save_gif(
    OUTPUT_DIR / "pick_animation.gif",
    duration=700,
)

print("\nAnimation saved:", gif_path.resolve())
print("Recorded frames:", len(recorder.frames))

# 7. 验收
assert result["success"], "Pick skill failed!"
assert carrying is not None, "Nothing is being carried."
assert carrying.color == "red", "The carried object is not red."

env.close()

print("\nPASS: pick() works correctly.")
