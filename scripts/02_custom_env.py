from pathlib import Path
from PIL import Image

from environments.saycan_grid import SayCanGridEnv


output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

env = SayCanGridEnv(render_mode="rgb_array")

obs, info = env.reset(seed=42)

print("Mission:", obs["mission"])
print("Agent position:", env.agent_pos)
print("Agent direction:", env.agent_dir)
print("Target position:", env.target_pos)
print("Action space:", env.action_space)

# 保存自定义环境的初始画面
frame = env.render()
Image.fromarray(frame).save(
    output_dir / "saycan_initial.png"
)

env.close()

print("Custom environment initialized successfully.")
print("Image saved to outputs/saycan_initial.png")
