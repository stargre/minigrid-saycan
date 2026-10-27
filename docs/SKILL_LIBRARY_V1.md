# Frozen Robot Skill Library v1

## Purpose and boundary

The skill_library_v1 API is the low-level execution contract for MiniGrid SayCan experiments. The planner chooses semantic skills; it never chooses primitive turns or forwards. Skills are parameterized by object descriptions and relations rather than separate implementations for each object.

The skill layer is observation-grounded: navigation and search use BeliefState.known_map, and ObservationRuntime discards environment info. Test fixtures may inspect hidden state to set up and score a test, but skills must not.

## Public skills

| Skill | Arguments | Contract |
|---|---|---|
| Explore() | none | Expand the belief map by moving to reachable frontiers. |
| Find(target) | semantic object description | Return target evidence if known; otherwise explore one action at a time. |
| NavigateTo(target) | target | Reach an adjacent cell and face the target. |
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
- Blocked route: move a movable obstacle to a known safe placement location, then recompute navigation from the updated belief map.

Prerequisites are inserted as ordinary candidates; they are not executed behind the planner's back. After every primitive action, the planner refreshes feasibility and relevance.

## Install and run

Install requirements.txt. Then run:

    python -m unittest discover -s tests -v
    python scripts/19_skill_library_v1_smoke_test.py
    python scripts/20_skill_acceptance_suite.py
    python scripts/21_skill_behavior_suite.py
    python scripts/08_saycan_planner_test.py

For LLM scoring, set DASHSCOPE_API_KEY in the shell environment. Never commit API keys or .env files.

## Acceptance checklist

1. Every skill accepts semantic target arguments and uses the same execution contract.
2. No skill reads env.grid, env.agent_pos, env.carrying, env.*_pos, or step(...).info for decisions.
3. Unknown targets cause bounded exploration; known targets stop search immediately.
4. Primitive actions go only through ObservationRuntime.execute_action.
5. Door, pickup, and placement success is based on updated observations, not action intent.
6. No infinite loops: search and planner have action/decision limits and explicit failure statuses.
7. Run scripts/20_skill_acceptance_suite.py and scripts/21_skill_behavior_suite.py; every public skill must pass its independent behavior checks.
8. Run scripts/08_saycan_planner_test.py over multiple seeds and verify both planner-reported and ground-truth success.
9. The unit suite and multi-seed environment suite must pass before tagging this version as frozen.
10. A single explore -> approach -> open pass is not sufficient evidence of skill-library completeness.

## Scope note

MiniGrid DoorKey, Fetch, BlockedUnlockPickup, PutNear, GoToDoor, Crossing, and DynamicObstacles have different mechanics. The semantic skill APIs are designed to generalize, but the current SayCanGridEnv only instantiates one unlocked yellow door, one red ball, and one green goal. Do not claim those official environment families are validated until environment adapters and randomized acceptance tests for them are added.
