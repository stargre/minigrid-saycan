from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class FrameRecorder:
    """记录环境每一步的状态，并生成带动作说明的 GIF。"""

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
        self.step_index = 0

        self._record(
            f"Step 0 | START | pos={self._position()}"
        )

    def __getattr__(self, name):
        # 其他属性和方法交给原始环境
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

        # 在场景上方单独增加信息栏，不遮挡网格
        header_height = 34
        canvas_width = max(image.width, 420)

        canvas = Image.new(
            "RGB",
            (canvas_width, image.height + header_height),
            (30, 30, 30),
        )

        x_offset = (canvas_width - image.width) // 2
        canvas.paste(image, (x_offset, header_height))

        draw = ImageDraw.Draw(canvas)

        carrying = self.env.unwrapped.carrying
        if carrying is not None:
            color = getattr(carrying, "color", "object")
            label += f" | holding={color}"

        draw.text(
            (8, 10),
            label,
            fill=(255, 255, 255),
        )

        self.frames.append(canvas)

    def step(self, action):
        # 先执行动作，再记录动作完成后的状态
        result = self.env.step(action)

        self.step_index += 1
        action_id = int(action)
        action_name = self.ACTION_NAMES.get(
            action_id, str(action_id)
        )

        self._record(
            f"Step {self.step_index} | {action_name} "
            f"| pos={self._position()}"
        )

        return result

    def save_gif(self, path, duration=700):
        """将已记录的所有帧保存为 GIF。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.frames:
            raise RuntimeError("No frames have been recorded.")

        self.frames[0].save(
            path,
            save_all=True,
            append_images=self.frames[1:],
            duration=duration,
            loop=0,
            optimize=False,
            disposal=2,
        )

        return path
