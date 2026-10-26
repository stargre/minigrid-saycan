import gymnasium as gym
from pathlib import Path
from PIL import Image
import minigrid 

# 保存实验结果
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 创建环境
env = gym.make(
    "MiniGrid-Empty-5x5-v0",
    render_mode="rgb_array"
)

# 初始化环境
obs, info = env.reset(seed=42)

print("Mission:", obs["mission"])
print("Observation image shape:", obs["image"].shape)
print("Agent direction:", obs["direction"])
print("Action space:", env.action_space)

# MiniGrid 的底层动作编号
action_names = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}
print("Available actions:", action_names)

# 保存初始环境图像
frame = env.render()
Image.fromarray(frame).save(output_dir / "initial.png")

# 让智能体依次执行几个动作
actions = [1, 2, 2, 0, 2]

for step, action in enumerate(actions, start=1):
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step={step}, "
        f"Action={action_names[action]}, "
        f"Reward={reward:.4f}, "
        f"Terminated={terminated}, "
        f"Truncated={truncated}, "
        f"Direction={obs['direction']}"
    )

    # 保存每一步的环境图像
    frame = env.render()
    Image.fromarray(frame).save(
        output_dir / f"step_{step:02d}.png"
    )

    if terminated or truncated:
        print("Episode finished.")
        break

env.close()
print("Finished. Images saved in outputs/")
