# Frozen Robot Skill Library v1

## Purpose and boundary

The skill_library_v1 API is the low-level execution contract for MiniGrid SayCan experiments. The planner chooses semantic skills; it never chooses primitive turns or forwards. Skills are parameterized by object descriptions and relations rather than separate implementations for each object.

The skill layer is observation-grounded: navigation and search use BeliefState.known_map, and ObservationRuntime discards environment info. Test fixtures may inspect hidden state to set up and score a test, but skills must not.

## Public skills

| Skill | Arguments | Contract |
|---|---|---|
| Explore() | none | Expand the belief map by moving to reachable frontiers. |
| Find(target) | semantic object description | Return target evidence if known; otherwise explore one action at a time. |
| NavigateTo(target) | target | Reach an adjacent cell and face a non-walkable target such as a door. |
| Reach(target) | known walkable target cell | Navigate onto a walkable destination such as a green goal tile; verify position from observations and action history. |
| PickUp(target) | movable object | Navigate, pick up, and verify the held object. |
| OpenDoor(door) | door | Open an unlocked closed door and verify its observed state. |
| UnlockDoor(door) | locked door | Requires a matching held key; interact and verify state change. |
| PlaceNear(object,target,relation) | held object, destination, relation | Derive a relation cell, navigate, drop, and verify object release and placement. |
| MoveObstacle(object,destination,relation) | movable obstacle and destination | Compose generic pickup and placement phases. |
| Done(target) | optional target | Navigate to interaction range, then issue MiniGrid done. |

Every execute(call) returns a dictionary with success, status, reason, skill, and actions_executed. A skill invocation performs at most one primitive action; the planner then observes and re-scores.

## SayCan score and dependencies

The feasibility evaluator is deterministic and observation-grounded. It returns a binary feasibility score plus a reason and prerequisite calls. The planner computes:

score(skill) = relevance(task, skill) * feasibility(state, skill)

An LLM may provide relevance only. It must not override feasibility or inspect hidden state. KeywordRelevanceScorer is an offline smoke-test fallback; research runs should inject QwenSkillRelevanceScorer or another LLM relevance adapter.

Example dependencies:

- Locked door: Find(key) -> PickUp(key) -> NavigateTo(door) -> UnlockDoor(door).
- Object placement: Find(object) -> PickUp(object) -> Find(target) -> PlaceNear(object,target,relation).
- Goal entry: Find(goal) -> Reach(goal), which differs from NavigateTo(goal): Reach enters the walkable target cell, while NavigateTo stops adjacent to the target.
- Blocked route: move a movable obstacle to a known safe placement location, then recompute navigation from the updated belief map.

Prerequisites are inserted as ordinary candidates; they are not executed behind the planner's back. After every primitive action, the planner refreshes feasibility and relevance.

## Install and run

Install requirements.txt. Then run:

    python -m unittest discover -s tests -v
    python scripts/19_skill_library_v1_smoke_test.py
    python scripts/20_skill_acceptance_suite.py
    python scripts/21_skill_behavior_suite.py
    python scripts/22_hidden_state_audit.py
    python scripts/08_saycan_planner_test.py

For LLM scoring, set DASHSCOPE_API_KEY in the shell environment. Never commit API keys or .env files.

## Acceptance checklist

1. Every skill accepts semantic target arguments and uses the same execution contract.
2. No skill reads env.grid, env.agent_pos, env.carrying, env.*_pos, or step(...).info for decisions.
3. Unknown targets cause bounded exploration; known targets stop search immediately.
4. Primitive actions go only through ObservationRuntime.execute_action.
5. Door, pickup, and placement success is based on updated observations, not action intent.
6. Run scripts/22_hidden_state_audit.py; decision modules must not read hidden environment state, and primitive actions must pass through ObservationRuntime.execute_action.
7. No infinite loops: search and planner have action/decision limits and explicit failure statuses.
8. Run scripts/20_skill_acceptance_suite.py and scripts/21_skill_behavior_suite.py; every public skill must pass its independent behavior checks.
9. Run scripts/22_hidden_state_audit.py and verify the guarded end-to-end tasks pass.
10. Run scripts/08_saycan_planner_test.py over multiple seeds and verify both planner-reported and ground-truth success.
9. The unit suite and multi-seed environment suite must pass before tagging this version as frozen.
10. A single explore -> approach -> open pass is not sufficient evidence of skill-library completeness.

## Scope note

MiniGrid DoorKey, Fetch, BlockedUnlockPickup, PutNear, GoToDoor, Crossing, and DynamicObstacles have different mechanics. The semantic skill APIs are designed to generalize, but the current SayCanGridEnv only instantiates one unlocked yellow door, one red ball, and one green goal. Reach is included to distinguish entering a walkable goal tile from merely approaching an object. Do not claim those official environment families are validated until environment adapters and randomized acceptance tests for them are added.


## Official DoorKey acceptance

Run the official-environment integration test after the custom-environment suites:

    python scripts/23_official_doorkey_acceptance.py

By default this evaluates MiniGrid-DoorKey-5x5-v0 and MiniGrid-DoorKey-6x6-v0
for seeds 0-4. The runtime receives a step-only environment proxy; ground-truth
position, door state, and terminal reward are inspected only after planning ends.
Per-episode results are saved to `outputs/doorkey_eval/results.json`. A seed
passes only when the planner reports completion, the environment terminates
with positive reward at the goal, the door is open/unlocked, and the belief
state records both unlock and open evidence. Failures are retained in the JSON
report so the first broken skill/dependency can be diagnosed without changing
the primitive controllers blindly.


## DoorKey completion-state fix

For DoorKey transitions, a single successful toggle may move a matching-key
locked door directly to the open state. The belief state records both
`unlock_door` and `open_door` when that transition is observed. Task
completion also preserves successful historical Find evidence: an object does
not become "never found" merely because it was picked up or the agent later
crossed its map cell. The final environment-level acceptance test still checks
the real terminal state independently.


## Explicit Find-target obligations

Task completion parses targets only from explicit `Find`, `Search`, or
`Locate` clauses. Objects mentioned as operands of later actions (for example,
`UnlockDoor(yellow door)` or `Reach(green goal)`) do not create extra Find
obligations. This keeps natural-language task semantics aligned with the
planner's subgoal checks while preserving evidence requirements for genuinely
requested searches.
