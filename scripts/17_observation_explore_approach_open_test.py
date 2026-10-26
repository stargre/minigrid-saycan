from pathlib import Path

from environments.saycan_grid import SayCanGridEnv
from perception.belief_state import BeliefState
from skills.observation_runtime import ObservationRuntime
from skills.explore_for_door import ExploreForDoorSkill
from skills.approach_door import ApproachDoorSkill
from skills.interaction import OpenDoorSkill
from utils.visualization import FrameRecorder


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEARCH_STEPS = 250
MAX_APPROACH_STEPS = 100


def observation_contains_yellow_door(obs):
    """只根据给定的局部观测判断门是否已经可见。"""
    probe = BeliefState()
    probe.reset(obs, initial_inventory="empty")

    return bool(
        probe.find_objects("door", color="yellow")
    )


env = SayCanGridEnv(render_mode="rgb_array")

# 测试夹具只检查初始观测，不读取环境内部门坐标。
selected_seed = None
initial_obs = None

for seed in range(1000):
    obs, _ = env.reset(seed=seed)

    if not observation_contains_yellow_door(obs):
        selected_seed = seed
        initial_obs = obs
        break

if initial_obs is None:
    env.close()
    raise RuntimeError(
        "Could not find a seed where the yellow door "
        "is absent from the initial local observation."
    )

print("=" * 68)
print("OBSERVATION-BASED SEARCH -> APPROACH -> OPEN")
print("Selected seed:", selected_seed)
print("Initial yellow door visible: False")
print("=" * 68)

# 记录器只用于可视化；决策使用 runtime 和 belief。
recorder = FrameRecorder(env)
runtime = ObservationRuntime(recorder)
runtime.reset(initial_obs, initial_inventory="empty")

# ------------------------------------------------------------
# Stage 1: Search for the door.
# ------------------------------------------------------------
searcher = ExploreForDoorSkill(runtime, color="yellow")
search_result = None
door_found = False

print("\n[STAGE 1] Explore unknown areas")

for step in range(1, MAX_SEARCH_STEPS + 1):
    result = searcher.step()
    search_result = result

    print(
        f"Search {step:03d}: "
        f"status={result['status']}, "
        f"action={result.get('action', 'none')}, "
        f"estimated_position={result.get('estimated_position', list(runtime.belief.agent_pos))}, "
        f"known_cells={result.get('known_cell_count', len(runtime.belief.known_map))}"
    )

    if result["status"] == "target_found":
        door_found = True
        print("Yellow door discovered from observations.")
        break

    if result["status"] != "action_executed":
        print("Search stopped:", result.get("reason", result["status"]))
        break

# ------------------------------------------------------------
# Stage 2: Approach the observed door.
# ------------------------------------------------------------
approach_result = None

if door_found:
    print("\n[STAGE 2] Approach the observed door")

    approacher = ApproachDoorSkill(runtime, color="yellow")

    for step in range(1, MAX_APPROACH_STEPS + 1):
        result = approacher.step()
        approach_result = result

        print(
            f"Approach {step:03d}: "
            f"status={result.get('status')}, "
            f"action={result.get('action', 'none')}, "
            f"estimated_position={result.get('estimated_position', list(runtime.belief.agent_pos))}"
        )

        if result.get("status") == "target_adjacent":
            print("Reached the door and faced it.")
            break

        if result.get("status") != "action_executed":
            print(
                "Approach stopped:",
                result.get("reason", result.get("status")),
            )
            break
else:
    print("\n[STAGE 2] Skipped: door was not discovered.")

# ------------------------------------------------------------
# Stage 3: Open the door using the current observation.
# ------------------------------------------------------------
door_result = None

if (
    approach_result is not None
    and approach_result.get("status") == "target_adjacent"
):
    print("\n[STAGE 3] Open the observed door")

    opener = OpenDoorSkill(runtime, color="yellow")
    door_result = opener.step()

    # 不打印 obs 数组，避免日志过长。
    for key, value in door_result.items():
        if key != "obs":
            print(f"{key}: {value}")
else:
    print("\n[STAGE 3] Skipped: door approach did not complete.")

# Save the animation even if a stage fails, so failures can be inspected.
gif_path = recorder.save_gif(
    OUTPUT_DIR / "explore_approach_open.gif",
    duration=350,
)

print("\n" + "=" * 68)
print("FINAL REPORT")
print("Seed:", selected_seed)
print("Initially visible:", observation_contains_yellow_door(initial_obs))
print("Door discovered:", door_found)
print(
    "Approach status:",
    None if approach_result is None
    else approach_result.get("status"),
)
print(
    "Door opened:",
    None if door_result is None
    else door_result.get("success"),
)
print("Total low-level actions:", runtime.action_count)
print("GIF:", gif_path.resolve())
print("GIF frames:", len(recorder.frames))
print("=" * 68)

env.close()

assert not observation_contains_yellow_door(initial_obs), (
    "Test setup invalid: the door was initially visible."
)
assert door_found, (
    "Search failed to discover the door. Inspect the GIF and logs."
)
assert (
    approach_result is not None
    and approach_result.get("status") == "target_adjacent"
), "ApproachDoorSkill did not reach the door."
assert door_result is not None and door_result.get("success"), (
    "OpenDoorSkill did not confirm the door opening."
)
assert door_result["actions_executed"] == 1, (
    "Expected exactly one toggle action to open the closed door."
)

print("PASS: explore -> approach -> open succeeded.")
