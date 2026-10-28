# Skill Library v1.0 — Architecture and Freeze Contract

This document defines the modular source layout and behavioral contract for the observation-grounded MiniGrid SayCan skill library.

## Layer responsibilities

1. **Environment adapter** (environments/) exposes the simulator step interface. Decision modules must not inspect simulator internals.
2. **Perception and belief** (perception/) builds a local map and task evidence from observations and action history.
3. **Runtime gateway** (skills/observation_runtime.py) is the only path to primitive MiniGrid actions and deliberately discards info.
4. **Reusable skills** are parameterized semantic controllers. A registry call advances at most one primitive action.
5. **Feasibility and dependencies** provide deterministic precondition checks and prerequisite candidates; they do not execute actions.
6. **Planner** selects a semantic skill using relevance × feasibility, runs it under budgets, and replans after observations.
7. **Evaluation** may inspect ground truth after planning ends; ground truth must never feed decisions.

## Skill modules

- base.py: shared helpers and result structure.
- search.py: Explore and Find(target).
- navigation_skills.py: NavigateTo(target) and Reach(target).
- doors.py: OpenDoor(door) and UnlockDoor(door).
- manipulation_skills.py: PickUp(target), PlaceNear(object,target,relation), MoveObstacle(object,destination,relation).
- interaction_skill.py: Done(target).
- registry.py: canonical ten-skill registry and uniform execute(call).
- unified_library.py: backwards-compatible import facade; it contains no skill implementations.
- target.py: target parsing/matching and relation helpers.
- navigation.py: low-level observation-grounded path controller.
- observation_explorer.py: bounded frontier exploration.
- dependency.py: candidate generation and prerequisite chains.
- feasibility.py: deterministic feasibility from belief state.
- library.py: compatibility adapter for older callers; new code should use SkillLibraryV1.

## Execution result

Every call returns success, status, reason, skill and actions_executed. Status values distinguish terminal completion from intermediate progress such as action_executed, searching and phase_complete. The planner owns retry, timeout and task-level completion decisions.

## Validation gates

Run python -m unittest discover -s tests -v, then scripts 19, 20, 21 and 22, followed by the end-to-end multi-seed task and script 23 official DoorKey suite. The official DoorKey default matrix is two environment IDs × five seeds = ten episodes. Do not claim coverage of other environment families without dedicated tests.

## Change policy

Refactors may move implementations and update imports, but must preserve public skill names/arguments, status semantics, one-primitive-action-per-registry-call invariant, hidden-state boundary and scoring formula. Update scripts/22_hidden_state_audit.py whenever decision modules are added or moved. Freeze only after the complete validation matrix passes.
