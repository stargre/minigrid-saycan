import os
from pathlib import Path

from environments.saycan_grid import SayCanGridEnv
from skills.observation_runtime import ObservationRuntime
from planners.observation_task_controller import (
    ObservationTaskController,
)
from utils.visualization import FrameRecorder


SEED = int(os.environ.get("SEED", "0"))
MAX_ACTIONS = 450

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

env = SayCanGridEnv(render_mode="rgb_array")

obs, _ = env.reset(seed=SEED)

recorder = FrameRecorder(env)
runtime = ObservationRuntime(recorder)
runtime.reset(obs, initial_inventory="empty")

controller = ObservationTaskController(runtime)

print("=" * 68)
print("OBSERVATION-BASED FULL TASK TEST")
print("Seed:", SEED)
print("Initial observation shape:", obs["image"].shape)
print("Max low-level actions:", MAX_ACTIONS)
print("=" * 68)

result = controller.run(
    max_low_level_actions=MAX_ACTIONS,
    log_every=10,
)

gif_path = recorder.save_gif(
    OUTPUT_DIR / f"observation_full_task_seed_{SEED}.gif",
    duration=350,
)

print("=" * 68)
print("FINAL REPORT")
print("Success:", result["success"])
print("Status:", result["status"])
print("Decisions:", result["decisions"])
print("Low-level actions:", result["low_level_actions"])
print("Known cells:", result["known_cells"])
print("Inventory:", result["inventory_status"])
print("Completed subgoals:", result["completed_subgoals"])
print("GIF:", gif_path.resolve())
print("GIF frames:", len(recorder.frames))
print("=" * 68)

env.close()

assert result["success"], (
    "Full observation-based task failed. "
    f"Status: {result['status']}; "
    f"completed: {result['completed_subgoals']}"
)

assert result["inventory_status"] == "empty"
assert "pick" in result["completed_subgoals"]
assert "open_door" in result["completed_subgoals"]
assert "place" in result["completed_subgoals"]

print("PASS: all observation-based skills completed the task.")
