from pathlib import Path

from environments.saycan_grid import SayCanGridEnv
from perception.belief_state import BeliefState, DIRECTIONS
from skills.observation_runtime import ObservationRuntime
from skills.approach_door import ApproachDoorSkill
from utils.visualization import FrameRecorder


env = SayCanGridEnv(render_mode="rgb_array")

# 只根据初始局部观测寻找一个门位于侧前方的测试场景。
selected_obs = None
selected_seed = None
selected_offset = None

for seed in range(100):
    obs, _ = env.reset(seed=seed)

    probe = BeliefState()
    probe.reset(obs, initial_inventory="empty")

    doors = probe.find_objects(
        "door",
        color="yellow",
        state="closed",
    )

    if not doors:
        continue

    door_pos, _ = min(
        doors,
        key=lambda item: (
            abs(item[0][0] - probe.agent_pos[0])
            + abs(item[0][1] - probe.agent_pos[1])
        ),
    )

    dx = door_pos[0] - probe.agent_pos[0]
    dy = door_pos[1] - probe.agent_pos[1]

    hx, hy = DIRECTIONS[probe.agent_dir]
    rx, ry = -hy, hx

    forward = dx * hx + dy * hy
    lateral = dx * rx + dy * ry

    if forward > 1 and lateral != 0:
        selected_obs = obs
        selected_seed = seed
        selected_offset = (forward, lateral)
        break

if selected_obs is None:
    env.close()
    raise RuntimeError(
        "Could not find a visible off-axis door using observations."
    )

print("=" * 60)
print("OBSERVATION-BASED DOOR APPROACH TEST")
print("Selected seed:", selected_seed)
print("Initial observed door offset:", selected_offset)
print("=" * 60)

recorder = FrameRecorder(env)
runtime = ObservationRuntime(recorder)
runtime.reset(selected_obs, initial_inventory="empty")
skill = ApproachDoorSkill(runtime)

final_result = None
max_actions = 40

for step in range(1, max_actions + 1):
    result = skill.step()
    final_result = result

    print(
        f"Step {step:02d}: "
        f"status={result['status']}, "
        f"action={result.get('action', 'none')}, "
        f"estimated_position={runtime.belief.agent_pos}, "
        f"direction={runtime.belief.agent_dir}, "
        f"known_cells={len(runtime.belief.known_map)}"
    )

    if result["status"] == "target_adjacent":
        break

    if result["status"] != "action_executed":
        print("Approach paused:", result.get("reason", ""))
        break

gif_path = recorder.save_gif(
    Path("outputs/observation_door_approach.gif"),
    duration=350,
)

print("=" * 60)
print("FINAL STATUS:", final_result["status"])
print("ACTIONS:", runtime.action_count)
print("GIF:", gif_path.resolve())
print("=" * 60)

env.close()

assert final_result["status"] == "target_adjacent", (
    "Door approach did not finish. Inspect the step log and GIF."
)

print("PASS: agent reached the door and faced it.")
