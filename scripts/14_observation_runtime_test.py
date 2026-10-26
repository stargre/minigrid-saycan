from pathlib import Path

from minigrid.core.actions import Actions

from environments.saycan_grid import SayCanGridEnv
from perception.belief_state import BeliefState
from skills.observation_runtime import ObservationRuntime
from utils.visualization import FrameRecorder


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

env = SayCanGridEnv(render_mode="rgb_array")
obs, _ = env.reset(seed=42)

recorder = FrameRecorder(env)
belief = BeliefState()
runtime = ObservationRuntime(recorder, belief)
runtime.reset(obs)

print("=" * 60)
print("OBSERVATION RUNTIME TEST")
print("Initial observation shape:", obs["image"].shape)
print("Initial belief:", belief.summary())
print("=" * 60)

# 固定动作序列只用于验证底层接口，不读取真实地图来选动作。
actions = [
    Actions.forward,
    Actions.right,
    Actions.forward,
    Actions.left,
    Actions.forward,
    Actions.right,
    Actions.forward,
]

for step, action in enumerate(actions, start=1):
    result = runtime.execute_action(action)
    obs = result["obs"]

    print(
        f"Step {step}: action={action.name}, "
        f"estimated_position={belief.agent_pos}, "
        f"direction={belief.agent_dir}, "
        f"known_cells={len(belief.known_map)}, "
        f"inventory={belief.inventory_status}"
    )

    assert "info" not in result, (
        "Runtime must not expose environment info."
    )

    assert belief.agent_dir == int(obs["direction"]), (
        "Belief direction differs from the observation."
    )

    if result["terminated"] or result["truncated"]:
        print("Episode ended.")
        break

gif_path = recorder.save_gif(
    OUTPUT_DIR / "observation_runtime_smoke.gif",
    duration=400,
)

print("=" * 60)
print("FINAL REPORT")
print("Actions executed:", runtime.action_count)
print("Belief summary:", belief.summary())
print("GIF:", gif_path.resolve())
print("GIF frames:", len(recorder.frames))
print("=" * 60)

env.close()

assert runtime.action_count > 0
assert len(belief.known_map) > 0

print("PASS: observation runtime and belief state initialized.")
