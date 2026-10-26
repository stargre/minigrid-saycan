from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from environments.saycan_grid import SayCanGridEnv
from skills.movement import MovementSkill


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class FrameRecorder:
    """记录每一步环境状态，并为动图添加文字说明。"""

    ACTION_NAMES = {
        0: "TURN LEFT",
        1: "TURN RIGHT",
        2: "FORWARD",
        3: "PICKUP",
        4: "DROP",
        5: "TOGGLE",
        6: "DONE",
    }

    def __init__(self, env):
        self.env = env
        self.frames = []

        # 记录初始状态
        self._record(
            f"Step 0 | START | pos={self._position()}"
        )

    def __getattr__(self, name):
        # 将其他环境接口转交给原始 MiniGrid 环境
        return getattr(self.env, name)

    def _position(self):
        return tuple(
            int(v) for v in self.env.unwrapped.agent_pos
        )

    def _record(self, label):
        frame = self.env.render()

        if frame is None:
            return

        image = Image.fromarray(
            np.asarray(frame, dtype=np.uint8)
        ).convert("RGB")

        # 在画面顶部添加步骤、动作和当前位置
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (0, 0, image.width, 32),
            fill=(25, 25, 25),
        )
        draw.text(
            (8, 9),
            label,
            fill=(255, 255, 255),
        )

        self.frames.append(image)

    def step(self, action):
        # 先让真实环境执行动作
        result = self.env.step(action)

        # 再记录动作执行后的环境状态
        action_id = int(action)
        action_name = self.ACTION_NAMES.get(
            action_id, str(action_id)
        )

        step_id = len(self.frames)
        self._record(
            f"Step {step_id} | {action_name} "
            f"| pos={self._position()}"
        )

        return result


def main():
    # 创建支持图像渲染的环境
    base_env = SayCanGridEnv(render_mode="rgb_array")
    base_env.reset(seed=42)

    # 使用记录器包装环境，不修改原有技能代码
    recorder = FrameRecorder(base_env)
    movement = MovementSkill(recorder)

    print("Initial position:", recorder._position())

    # 执行高层移动技能
    result = movement.move_to((3, 3))

    print("\nMovement result:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    print("\nFinal position:", recorder._position())
    print("Total low-level actions:", movement.action_count)

    # 保存 GIF：初始状态 + 每个动作执行后的状态
    gif_path = OUTPUT_DIR / "move_to_animation.gif"

    if recorder.frames:
        recorder.frames[0].save(
            gif_path,
            save_all=True,
            append_images=recorder.frames[1:],
            duration=700,
            loop=0,
            optimize=False,
            disposal=2,
        )

    base_env.close()

    print("\nAnimation saved:", gif_path.resolve())
    print("Recorded frames:", len(recorder.frames))

    assert result["success"], "Movement test failed!"
    assert recorder._position() == (3, 3)

    print("PASS: movement and animation generation succeeded.")


if __name__ == "__main__":
    main()
