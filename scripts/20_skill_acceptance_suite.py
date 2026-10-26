"""Multi-seed contract smoke suite; hidden state is read only by the test fixture."""
from environments.saycan_grid import SayCanGridEnv
from skills.observation_runtime import ObservationRuntime
from skills.unified_library import SkillLibraryV1


def run_seed(seed):
    env = SayCanGridEnv(render_mode=None)
    obs, _ = env.reset(seed=seed)
    runtime = ObservationRuntime(env)
    runtime.reset(obs, initial_inventory="empty")
    library = SkillLibraryV1(runtime)

    assert len(library.names) == len(set(library.names))
    assert all(hasattr(library.skills[name], "step") for name in library.names)

    # Each call is bounded to one low-level action. Unknown targets must not
    # query hidden coordinates and must return a structured status.
    before = runtime.action_count
    result = library.execute({"name": "Find", "args": {"target": {"object": "key", "color": "yellow"}}})
    assert runtime.action_count - before <= 1
    assert {"success", "status", "reason", "skill", "actions_executed"} <= set(result)
    assert "info" not in result

    # Runtime intentionally strips environment info from the execution result.
    before = runtime.action_count
    from minigrid.core.actions import Actions
    action_result = runtime.execute_action(Actions.left)
    assert runtime.action_count == before + 1
    assert "info" not in action_result
    assert runtime.belief.agent_dir == int(action_result["obs"]["direction"])
    env.close()
    return result["status"]


def main():
    outcomes = {}
    for seed in range(20):
        outcomes[seed] = run_seed(seed)
    print("Skill-library acceptance smoke suite passed for 20 seeds.")
    print("Find outcomes:", outcomes)


if __name__ == "__main__":
    main()
