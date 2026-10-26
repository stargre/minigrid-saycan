from pathlib import Path

from environments.saycan_grid import SayCanGridEnv
from skills.movement import MovementSkill
from skills.manipulation import PickSkill, PlaceSkill
from skills.interaction import OpenDoorSkill
from planners.saycan_planner import SayCanPlanner
from utils.visualization import FrameRecorder


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

env = SayCanGridEnv(render_mode="rgb_array")
env.reset(seed=SEED)

base = env.unwrapped

print("=" * 60)
print("RANDOMIZED SCENE")
print("Map size:", base.width, "x", base.height)
print("Agent position:", tuple(int(v) for v in base.agent_pos))
print("Ball side:", base.ball_side)
print("Ball position:", base.ball_pos)
print("Door position:", base.door_pos)
print("Goal position:", base.goal_pos)
print("Target position:", base.target_pos)
print("=" * 60)

recorder = FrameRecorder(env)
movement = MovementSkill(recorder)
picker = PickSkill(recorder, movement)
opener = OpenDoorSkill(recorder, movement)
placer = PlaceSkill(recorder, movement)

planner = SayCanPlanner(
    recorder,
    movement,
    picker,
    opener,
    placer,
)

task = (
    "Pick up the red ball, open the yellow door, "
    "and place it above the green goal."
)

success, history = planner.run(task, max_high_level_steps=10)

gif_path = recorder.save_gif(
    OUTPUT_DIR / "saycan_planner.gif",
    duration=500,
)

sequence = [item["skill"] for item in history]

# 当前任务及场景下的预期顺序。
expected = (
    ["pick", "open_door", "place"]
    if base.ball_side == "left"
    else ["open_door", "pick", "place"]
)

print("\n" + "=" * 60)
print("FINAL REPORT")
print("Seed:", SEED)
print("Ball side:", base.ball_side)
print("Task success:", success)
print("Actual skill sequence:", sequence)
print("Expected skill sequence:", expected)
print("High-level skills:", len(history))
print("Low-level actions:", movement.action_count)
print("GIF:", gif_path.resolve())
print("GIF frames:", len(recorder.frames))

env.close()

assert success, "Planning failed."
assert sequence == expected, (
    f"Unexpected skill sequence: {sequence}; expected {expected}"
)

print("\nPASS: randomized-scene planning test succeeded.")
