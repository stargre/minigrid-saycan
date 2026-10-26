from pathlib import Path

from environments.saycan_grid import SayCanGridEnv
from skills.movement import MovementSkill
from skills.exploration import ExploreSkill
from utils.visualization import FrameRecorder


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

env = SayCanGridEnv(render_mode="rgb_array")
obs, info = env.reset(seed=42)

recorder = FrameRecorder(env)
movement = MovementSkill(recorder)
explorer = ExploreSkill(movement)

inventory_status = "empty"
completed_subgoals = []

print("=" * 60)
print("LOCAL OBSERVATION EXPLORATION TEST")
print("View size:", obs["image"].shape[:2])
print("Initial inventory:", inventory_status)
print("=" * 60)

max_decisions = 40
status = None

for decision in range(max_decisions):
    result = explorer.step(
        obs,
        inventory_status=inventory_status,
        completed_subgoals=completed_subgoals,
    )

    obs = result["obs"]
    status = result["status"]

    target = result.get("target")
    if target:
        target_text = (
            f"{target['name']} "
            f"(forward={target['forward']}, "
            f"lateral={target['lateral']})"
        )
    else:
        target_text = "no visible task target"

    print(
        f"Decision {decision + 1:02d}: "
        f"status={status}, target={target_text}, "
        f"low_level_actions={movement.action_count}"
    )

    if status == "target_adjacent":
        print("Target reached interaction range.")
        break

    if result.get("terminated") or result.get("truncated"):
        print("Episode ended.")
        break

gif_path = recorder.save_gif(
    OUTPUT_DIR / "exploration_test.gif",
    duration=350,
)

print("=" * 60)
print("FINAL REPORT")
print("Decisions:", decision + 1)
print("Low-level actions:", movement.action_count)
print("Final status:", status)
print("GIF:", gif_path.resolve())
print("GIF frames:", len(recorder.frames))
print("=" * 60)

env.close()
