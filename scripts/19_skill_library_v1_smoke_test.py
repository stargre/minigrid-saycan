"""Integration smoke test for the observation-grounded v1 library."""
import os
from environments.saycan_grid import SayCanGridEnv
from skills.observation_runtime import ObservationRuntime
from skills.unified_library import SkillLibraryV1


def main():
    seed = int(os.getenv("SEED", "42"))
    env = SayCanGridEnv(render_mode=None)
    obs, _ = env.reset(seed=seed)
    runtime = ObservationRuntime(env)
    runtime.reset(obs, initial_inventory="empty")
    library = SkillLibraryV1(runtime)
    expected = {"Explore", "Find", "NavigateTo", "PickUp", "OpenDoor", "UnlockDoor", "PlaceNear", "MoveObstacle", "Done"}
    assert set(library.names) == expected
    result = library.execute({"name": "Find", "args": {"target": {"object": "ball", "color": "red"}}})
    assert result["status"] in {"target_found", "searching", "action_executed", "exploration_exhausted"}
    assert "info" not in result
    print("Skill library:", library.VERSION)
    print("Seed:", seed)
    print("First Find result:", {k:v for k,v in result.items() if k != "obs"})
    print("Belief:", runtime.belief.summary())
    env.close()


if __name__ == "__main__":
    main()
