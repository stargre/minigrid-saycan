"""Independent behavior tests for every public skill in skill_library_v1.

Test fixtures may set up the MiniGrid world. Skill decisions themselves still
go through ObservationRuntime and BeliefState; the suite never passes hidden
coordinates or environment info into a skill.
"""
from minigrid.core.actions import Actions
from minigrid.core.world_object import Ball, Box, Key

from environments.saycan_grid import SayCanGridEnv
from perception.belief_state import DIRECTIONS
from skills.observation_runtime import ObservationRuntime
from skills.unified_library import SkillLibraryV1


PROGRESS_STATUSES = {"action_executed", "searching", "phase_complete"}
EXPECTED_SKILLS = {
    "Explore", "Find", "NavigateTo", "PickUp", "OpenDoor",
    "UnlockDoor", "PlaceNear", "MoveObstacle", "Done",
}


def make_case(seed=0, door_state="closed", fixture=None):
    """Create a test scene, then expose only its observation to the runtime."""
    env = SayCanGridEnv(render_mode=None)
    env.reset(seed=seed)

    door = env.grid.get(*env.door_pos)
    door.is_locked = door_state == "locked"
    door.is_open = door_state == "open"

    if fixture is not None:
        # The following edits are test-fixture setup only, not skill inputs.
        env.grid.set(*env.ball_pos, None)
        fixture_pos = (3, 2)  # adjacent to the known spawn cell (2, 2)
        if fixture == "key":
            env.grid.set(*fixture_pos, Key("yellow"))
        elif fixture == "box":
            env.grid.set(*fixture_pos, Box("blue"))
        elif fixture == "ball_front":
            env.grid.set(*fixture_pos, Ball("red"))
        else:
            raise ValueError(f"Unknown test fixture: {fixture}")

    obs = env.gen_obs()
    runtime = ObservationRuntime(env)
    runtime.reset(obs, initial_inventory="empty")
    library = SkillLibraryV1(runtime)
    return env, runtime, library


def run_until_terminal(runtime, library, call, max_steps=400):
    """Execute one semantic skill repeatedly until its controller is terminal."""
    last = None
    for step_index in range(max_steps):
        if runtime.terminated or runtime.truncated:
            raise AssertionError(
                f"Episode ended before {call['name']} returned a terminal result; "
                f"last={last}"
            )

        before = runtime.action_count
        last = library.execute(call)
        delta = runtime.action_count - before

        assert delta in (0, 1), (
            f"{call['name']} executed {delta} primitive actions in one controller tick"
        )
        assert {
            "success", "status", "reason", "skill", "actions_executed"
        } <= set(last), f"Incomplete result contract: {last}"
        assert "info" not in last, "Environment info leaked into a skill result."
        assert last["actions_executed"] <= 1

        if last["status"] not in PROGRESS_STATUSES:
            return last

        if delta == 0 and last["status"] != "phase_complete":
            raise AssertionError(
                f"{call['name']} reported progress without an action: {last}"
            )

    raise AssertionError(
        f"{call['name']} did not terminate in {max_steps} controller ticks; last={last}"
    )


def find(runtime, library, target):
    result = run_until_terminal(
        runtime, library, {"name": "Find", "args": {"target": target}}
    )
    assert result["status"] == "target_found", (
        f"Find({target}) failed: {result}"
    )
    return result


def test_explore():
    env, runtime, library = make_case(seed=101, door_state="open")
    try:
        before = runtime.action_count
        result = library.execute({"name": "Explore", "args": {}})
        delta = runtime.action_count - before
        assert delta <= 1
        assert result["status"] in {
            "action_executed", "exploration_exhausted", "exploration_stalled"
        }, result
        if result["status"] == "action_executed":
            assert delta == 1
        print("PASS Explore: bounded one-tick observation-driven exploration")
    finally:
        env.close()


def test_find():
    env, runtime, library = make_case(seed=102, door_state="open")
    try:
        found = find(runtime, library, {"object": "ball", "color": "red"})
        assert found["success"]
        assert found.get("evidence", {}).get("position") is not None
    finally:
        env.close()

    env, runtime, library = make_case(seed=103, door_state="open")
    try:
        missing = run_until_terminal(
            runtime, library,
            {"name": "Find", "args": {"target": {"object": "key", "color": "yellow"}}},
        )
        assert missing["status"] == "target_not_found", missing
        print("PASS Find: present target found; absent target terminates explicitly")
    finally:
        env.close()


def test_navigate_to():
    env, runtime, library = make_case(seed=104, door_state="open")
    try:
        found = find(runtime, library, {"object": "ball", "color": "red"})
        target_pos = tuple(found["evidence"]["position"])
        result = run_until_terminal(
            runtime, library,
            {"name": "NavigateTo", "args": {"target": {"object": "ball", "color": "red"}}},
        )
        assert result["success"] and result["status"] == "skill_complete", result
        agent = tuple(runtime.belief.agent_pos)
        assert abs(agent[0] - target_pos[0]) + abs(agent[1] - target_pos[1]) == 1
        delta = (target_pos[0] - agent[0], target_pos[1] - agent[1])
        assert runtime.belief.agent_dir == DIRECTIONS.index(delta)
        print("PASS NavigateTo: adjacent position and target-facing orientation verified")
    finally:
        env.close()


def test_pickup():
    env, runtime, library = make_case(seed=105, door_state="open")
    try:
        find(runtime, library, {"object": "ball", "color": "red"})
        result = run_until_terminal(
            runtime, library,
            {"name": "PickUp", "args": {"target": {"object": "ball", "color": "red"}}},
        )
        assert result["success"] and result["status"] == "skill_complete", result
        held = runtime.belief.held_object
        assert held and held["object"] == "ball" and held["color"] == "red", held
        print("PASS PickUp: held-object state verified from updated observation")
    finally:
        env.close()


def test_open_door():
    env, runtime, library = make_case(seed=106, door_state="closed")
    try:
        find(runtime, library, {"object": "door", "color": "yellow"})
        result = run_until_terminal(
            runtime, library,
            {"name": "OpenDoor", "args": {"target": {"object": "door", "color": "yellow"}}},
        )
        assert result["success"] and result["status"] == "skill_complete", result
        assert "open_door" in runtime.belief.completed_subgoals
        print("PASS OpenDoor: closed unlocked door opened and verified")
    finally:
        env.close()


def test_unlock_door():
    env, runtime, library = make_case(seed=107, door_state="locked", fixture="key")
    try:
        find(runtime, library, {"object": "door", "color": "yellow"})

        before = runtime.action_count
        missing_key = library.execute({
            "name": "UnlockDoor",
            "args": {"target": {"object": "door", "color": "yellow"}},
        })
        assert missing_key["status"] == "requires_key", missing_key
        assert runtime.action_count == before, "UnlockDoor acted without a matching key."

        find(runtime, library, {"object": "key", "color": "yellow"})
        picked = run_until_terminal(
            runtime, library,
            {"name": "PickUp", "args": {"target": {"object": "key", "color": "yellow"}}},
        )
        assert picked["success"], picked

        find(runtime, library, {"object": "door", "color": "yellow"})
        unlocked = run_until_terminal(
            runtime, library,
            {"name": "UnlockDoor", "args": {"target": {"object": "door", "color": "yellow"}}},
        )
        assert unlocked["success"] and unlocked["status"] == "skill_complete", unlocked
        assert "unlock_door" in runtime.belief.completed_subgoals
        print("PASS UnlockDoor: refuses missing key; matching key unlocks door")
    finally:
        env.close()


def test_place_near():
    env, runtime, library = make_case(seed=108, door_state="open")
    try:
        find(runtime, library, {"object": "ball", "color": "red"})
        picked = run_until_terminal(
            runtime, library,
            {"name": "PickUp", "args": {"target": {"object": "ball", "color": "red"}}},
        )
        assert picked["success"], picked
        goal = find(runtime, library, {"object": "goal", "color": "green"})
        goal_pos = tuple(goal["evidence"]["position"])

        # The destination was observed by Find; the relation cell must also be
        # known and empty before PlaceNear is allowed to execute a drop.
        relation_pos = (goal_pos[0], goal_pos[1] - 1)
        assert runtime.belief.cell_at(relation_pos) is not None, (
            "The cell above the observed goal should be visible in this fixture."
        )
        result = run_until_terminal(
            runtime, library,
            {
                "name": "PlaceNear",
                "args": {
                    "object": {"object": "ball", "color": "red"},
                    "target": {"object": "goal", "color": "green"},
                    "relation": "above",
                },
            },
        )
        assert result["success"] and result["status"] == "skill_complete", result
        placed = runtime.belief.cell_at(relation_pos)
        assert placed and placed["object"] == "ball" and placed["color"] == "red", placed
        assert runtime.belief.inventory_status == "empty"
        print("PASS PlaceNear: above relation, placement, and release verified")
    finally:
        env.close()


def test_move_obstacle():
    env, runtime, library = make_case(seed=109, door_state="open", fixture="box")
    try:
        # Establish observation evidence for both semantic arguments before
        # asking the composite skill to execute its pickup/place phases.
        box = find(runtime, library, {"object": "box", "color": "blue"})
        find(runtime, library, {"object": "goal", "color": "green"})
        box_pos = tuple(box["evidence"]["position"])
        box_cell = runtime.belief.cell_at(box_pos)
        assert box_cell and box_cell["object"] == "box" and box_cell["color"] == "blue", (
            f"Box evidence was lost before MoveObstacle: {box_cell}"
        )
        assert runtime.belief.held_object is None, (
            "MoveObstacle fixture must begin with an empty gripper."
        )
        result = run_until_terminal(
            runtime, library,
            {
                "name": "MoveObstacle",
                "args": {
                    "target": {"object": "box", "color": "blue"},
                    "destination": {"object": "goal", "color": "green"},
                    "relation": "near",
                },
            },
        )
        assert result["success"] and result["status"] == "skill_complete", result
        assert runtime.belief.inventory_status == "empty"
        placement = tuple(result["placement_position"])
        destination = tuple(result["destination_position"])
        assert abs(placement[0] - destination[0]) + abs(placement[1] - destination[1]) == 1
        cell = runtime.belief.cell_at(placement)
        assert cell and cell["object"] == "box" and cell["color"] == "blue", cell
        print("PASS MoveObstacle: pickup/place composition and final relation verified")
    finally:
        env.close()


def test_done():
    env, runtime, library = make_case(seed=110, door_state="open", fixture="ball_front")
    try:
        find(runtime, library, {"object": "ball", "color": "red"})
        before = runtime.action_count
        result = library.execute({
            "name": "Done",
            "args": {"target": {"object": "ball", "color": "red"}},
        })
        assert result["success"] and result["status"] == "done_issued", result
        assert runtime.action_count == before + 1
        assert result["actions_executed"] == 1
        print("PASS Done: interaction-range check and MiniGrid done action verified")
    finally:
        env.close()


def test_uniform_api():
    env, runtime, library = make_case(seed=111, door_state="open")
    try:
        assert set(library.names) == EXPECTED_SKILLS
        assert len(library.names) == len(set(library.names))
        before = runtime.action_count
        unknown = library.execute({"name": "NotASkill", "args": {}})
        invalid = library.execute({"name": "PickUp", "args": {}})
        assert unknown["status"] == "unknown_skill"
        assert invalid["status"] == "invalid_arguments"
        assert runtime.action_count == before
        print("PASS API: registry, uniform results, and invalid-call safety verified")
    finally:
        env.close()


def main():
    tests = [
        test_uniform_api,
        test_explore,
        test_find,
        test_navigate_to,
        test_pickup,
        test_open_door,
        test_unlock_door,
        test_place_near,
        test_move_obstacle,
        test_done,
    ]
    for test in tests:
        test()
    print(f"Skill behavior suite passed: {len(tests)} test groups.")


if __name__ == "__main__":
    main()
