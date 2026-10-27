"""Behavioral acceptance tests for the generic observation-grounded skill library."""
from minigrid.core.actions import Actions

from environments.saycan_grid import SayCanGridEnv
from skills.observation_runtime import ObservationRuntime
from skills.unified_library import SkillLibraryV1


PROGRESS_STATUSES = {"searching", "action_executed", "phase_complete"}


def run_find(library, runtime, target, max_calls=300):
    """Keep invoking the same Find skill until it reaches a terminal status."""
    result = None
    for _ in range(max_calls):
        if runtime.terminated or runtime.truncated:
            return {
                "success": False,
                "status": "episode_ended",
                "reason": "Episode ended before Find reached a terminal result.",
            }

        before = runtime.action_count
        result = library.execute({"name": "Find", "args": {"target": target}})
        executed = runtime.action_count - before

        assert executed <= 1, "One Find controller tick must execute at most one primitive action."
        assert {"success", "status", "reason", "skill", "actions_executed"} <= set(result)
        assert "info" not in result

        if result.get("status") not in PROGRESS_STATUSES:
            return result

        if executed == 0 and result.get("status") != "phase_complete":
            return {
                "success": False,
                "status": "skill_stalled",
                "reason": "Find reported progress without executing an action.",
            }

    return {
        "success": False,
        "status": "test_call_limit",
        "reason": f"Find did not terminate within {max_calls} controller ticks.",
    }


def run_seed(seed):
    env = SayCanGridEnv(render_mode=None)
    obs, _ = env.reset(seed=seed)

    # This is a unit-test fixture for Find itself: open the separating door
    # before the agent starts so every generated target lies in the same
    # reachable region. The end-to-end planner test separately validates
    # finding a target across a closed door by selecting OpenDoor.
    env.grid.get(*env.door_pos).is_open = True
    obs = env.gen_obs()

    runtime = ObservationRuntime(env)
    runtime.reset(obs, initial_inventory="empty")
    library = SkillLibraryV1(runtime)

    expected = {
        "Explore", "Find", "NavigateTo", "Reach", "PickUp", "OpenDoor",
        "UnlockDoor", "PlaceNear", "MoveObstacle", "Done",
    }
    assert set(library.names) == expected
    assert len(library.names) == len(set(library.names))
    assert all(callable(getattr(library.skills[name], "step", None)) for name in library.names)

    # Check that primitive actions pass through the observation runtime and that
    # its returned result does not leak environment info.
    before = runtime.action_count
    action_result = runtime.execute_action(Actions.left)
    assert runtime.action_count == before + 1
    assert "info" not in action_result
    assert runtime.belief.agent_dir == int(action_result["obs"]["direction"])

    # Positive case: every randomized custom scene contains a red ball.
    found = run_find(library, runtime, {"object": "ball", "color": "red"})
    assert found["status"] == "target_found", (
        f"seed={seed}: the existing red ball was not found; status={found.get('status')}"
    )

    # Negative case: this custom environment contains no key. Search must stop
    # with an explicit terminal result rather than returning searching forever.
    missing = run_find(library, runtime, {"object": "key", "color": "yellow"})
    assert missing["status"] in {"target_not_found", "exploration_stalled"}, (
        f"seed={seed}: absent-key search did not terminate correctly: {missing.get('status')}"
    )

    env.close()
    return {
        "seed": seed,
        "positive_search": found["status"],
        "negative_search": missing["status"],
        "actions": runtime.action_count,
    }


def main():
    outcomes = [run_seed(seed) for seed in range(20)]
    print("Behavioral acceptance passed for 20 seeds.")
    print("Outcomes:", outcomes)


if __name__ == "__main__":
    main()
