from pathlib import Path

from environments.saycan_grid import SayCanGridEnv
from skills.observation_runtime import ObservationRuntime
from skills.approach_door import ApproachDoorSkill
from skills.interaction import OpenDoorSkill
from utils.visualization import FrameRecorder


SEED = 11
MAX_APPROACH_STEPS = 80

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

env = SayCanGridEnv(render_mode="rgb_array")
obs, _ = env.reset(seed=SEED)

recorder = FrameRecorder(env)

runtime = ObservationRuntime(recorder)
runtime.reset(obs, initial_inventory="empty")

approacher = ApproachDoorSkill(runtime, color="yellow")
opener = OpenDoorSkill(runtime, color="yellow")

print("=" * 64)
print("OBSERVATION-BASED DOOR APPROACH + OPEN TEST")
print("Seed:", SEED)
print("=" * 64)

approach_result = None

for step in range(1, MAX_APPROACH_STEPS + 1):
    result = approacher.step()
    approach_result = result

    status = result.get("status", "unknown")
    action = result.get("action", "none")
    position = result.get(
        "estimated_position",
        list(runtime.belief.agent_pos),
    )
    direction = int(runtime.belief.agent_dir)

    print(
        f"Approach {step:02d}: "
        f"status={status}, "
        f"action={action}, "
        f"estimated_position={position}, "
        f"direction={direction}"
    )

    if status == "target_adjacent":
        break

    if status != "action_executed":
        print("Approach stopped:", result.get("reason", status))
        break

print("-" * 64)

door_result = None

if (
    approach_result is not None
    and approach_result.get("status") == "target_adjacent"
):
    print("Approach complete. Executing observation-based door opening.")

    door_result = opener.step()

    # 不打印 obs 数组，避免日志过长。
    print("Door result:")

    for key, value in door_result.items():
        if key != "obs":
            print(f"  {key}: {value}")
else:
    print("Door opening skipped because approach did not complete.")

gif_path = recorder.save_gif(
    OUTPUT_DIR / "observation_door_approach_and_open.gif",
    duration=450,
)

print("-" * 64)
print("GIF:", gif_path.resolve())
print("GIF frames:", len(recorder.frames))
print("Total low-level actions:", runtime.action_count)

env.close()

assert (
    approach_result is not None
    and approach_result.get("status") == "target_adjacent"
), "The agent failed to approach the door."

assert door_result is not None and door_result.get("success"), (
    "Door opening was not confirmed by the next observation."
)

assert door_result["actions_executed"] == 1, (
    "Opening the closed door should require exactly one toggle action."
)

print("=" * 64)
print("PASS: observation-based approach and door opening succeeded.")
print("=" * 64)
