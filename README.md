# MiniGrid SayCan: Observation-Grounded Robot Skills

This repository separates high-level task planning from a fixed low-level skill library. The current implementation includes the original experimental environment and a new observation-grounded \`skill_library_v1\` API.

## Quick start

Use Python 3.10+ in a virtual environment:

\`bash
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m unittest discover -s tests -v
python scripts/19_skill_library_v1_smoke_test.py
\`

To use Qwen for relevance scoring, set DASHSCOPE_API_KEY in your shell and inject QwenSkillRelevanceScorer into UnifiedSayCanPlanner. Do not put API keys in source control.

## New stable interfaces

- skills.unified_library.SkillLibraryV1: Explore, Find(target), NavigateTo(target), PickUp(target), OpenDoor(door), UnlockDoor(door), PlaceNear(object,target,relation), MoveObstacle(object,destination,relation), Done(target).
- skills.feasibility.SkillFeasibilityEvaluator: deterministic physical feasibility based on the agent's belief state.
- skills.dependency.TaskDependencyResolver: semantic candidates and prerequisite chains.
- planners.unified_saycan_planner.UnifiedSayCanPlanner: selects skills by relevance × feasibility and re-evaluates after every action.
- llm.skill_relevance.QwenSkillRelevanceScorer: Qwen relevance adapter; KeywordRelevanceScorer is only an offline fallback.

Every skill call returns success, status, reason, skill, and actions_executed. A skill step executes at most one primitive action; all primitive actions go through ObservationRuntime, which does not expose environment info.

## Scope and claims

The existing SayCanGridEnv is a custom task with a red ball, an unlocked yellow door, and a green goal. The generic skill API is designed to parameterize different objects, but official MiniGrid families such as DoorKey, Fetch, BlockedUnlockPickup, PutNear, GoToDoor, Crossing, and DynamicObstacles must each be run through acceptance tests before claiming support. See docs/SKILL_LIBRARY_V1.md for architecture, limits, and the freeze checklist.
