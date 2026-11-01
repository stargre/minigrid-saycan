# MiniGrid SayCan — Skill Library v1.0

A modular, observation-grounded SayCan reproduction for MiniGrid. The planner
scores the same semantic skill candidates with task relevance (**Say**) and
observation-grounded affordance (**Can**), selects the maximum product, executes
a closed-loop skill through one runtime gateway, updates belief, and replans.

> Strict acceptance covers Empty, GoToDoor, GoToObject, Fetch, PutNear, Unlock
> and DoorKey. Can may be either the deterministic baseline or the bundled
> calibrated skill-success model; the learned mode preserves deterministic
> hard safety constraints. Keyword scoring is an offline baseline and must not
> be reported as an LLM experiment.

**Verified baseline:** 97/97 unit tests and 140/140 official MiniGrid episodes
(20 seeds x 7 families) with learned Can on Python 3.12.7 and MiniGrid 3.2.0.
The held-out affordance set reaches 98.63% accuracy, 0.00994 Brier score and
0.00839 expected calibration error. Qwen smoke acceptance previously passed
7/7 on Python 3.12.15 with a valid DashScope setup; Qwen plus the new model must
be rerun wherever that API credential is available. See
[the acceptance protocol](docs/MINIGRID_ACCEPTANCE.md) for details.

## Architecture

Task text → UnifiedSayCanPlanner → dependency resolution + feasibility + relevance scoring → SkillLibraryV1 → semantic skill controller → ObservationRuntime.execute_action → MiniGrid step → BeliefState.update → replan.

## Repository layout

### Environment and perception
- **environments/saycan_grid.py** — custom task environment for baseline regression.
- **perception/belief_state.py** — observation/history-derived local map, estimated pose, inventory and completed-subgoal evidence.
- **perception/observation_summary.py** — converts a single observation into a structured summary.
- **skills/observation_runtime.py** — only permitted primitive-action gateway; discards environment info.
- **skills/navigation.py** — observation-grounded path search and one-step navigation primitives.
- **skills/observation_explorer.py** — frontier exploration controller.
- **skills/target.py** — target parsing/matching, canonical object names and spatial-relation helpers.

### Modular skill library
- **skills/base.py** — shared result contract and target/map helpers.
- **skills/search.py** — Explore and Find(target).
- **skills/navigation_skills.py** — NavigateTo(target) and Reach(target).
- **skills/doors.py** — OpenDoor(door) and UnlockDoor(door).
- **skills/manipulation_skills.py** — PickUp(target), PlaceNear(object,target,relation), and composite MoveObstacle.
- **skills/interaction_skill.py** — Done(target).
- **skills/registry.py** — canonical registry and uniform execute(call) API.
- **skills/unified_library.py** — compatibility facade; existing imports remain valid.
- **skills/dependency.py** — candidate calls and prerequisites; never executes actions.
- **skills/feasibility.py** — deterministic feasibility evaluation from belief state only.
- **affordance/** — observation-grounded feature encoding, calibrated learned
  skill-success model, dataset writer and `hard mask × learned probability`
  evaluator.
- **skills/library.py** — legacy adapter retained for older callers; new code should use SkillLibraryV1.

### Planning and model adapters
- **planners/unified_saycan_planner.py** — candidate scoring, selection, bounded skill execution, replanning and belief-side completion checks.
- **planners/saycan_planner.py** — backward-compatible planner adapter.
- **planners/observation_task_controller.py** — standalone observation-driven task controller retained for callers that still import it.
- **llm/skill_relevance.py** — keyword fallback and parameterized Qwen relevance adapter.
- **requirements.txt** — Python dependencies.

### Tests and evaluation
- **tests/test_skill_contract.py** — target parsing, dependency-call construction and relevance-score contract.
- **tests/test_task_completion.py** — task-completion evidence regression tests.
- **tests/test_refactor_integrity.py** — imports, registry contract, planner wiring and one-step action invariant.
- **scripts/19_skill_library_v1_smoke_test.py** — registry/import smoke test.
- **scripts/20_skill_acceptance_suite.py** — search/acceptance regression over seeds.
- **scripts/21_skill_behavior_suite.py** — independent behavioral checks for all ten public skills.
- **scripts/22_hidden_state_audit.py** — static and guarded-runtime audit for hidden-state access and action-gateway bypass.
- **scripts/23_official_doorkey_acceptance.py** — official DoorKey 5x5/6x6 integration tests.
- **scripts/25_official_basic_env_acceptance.py** — Empty, GoToDoor and Unlock official environment acceptance tests.
- **scripts/26_official_putnear_acceptance.py** — official PutNear integration tests.
- **scripts/27_official_fetch_qwen_e2e.py** — Qwen-API-only Fetch end-to-end tests with per-action GIF recording and post-run oracle checks.
- **scripts/28_official_gotoobject_qwen_e2e.py** — Qwen-API-only GoToObject end-to-end tests with per-action GIF recording and post-run oracle checks.
- **scripts/29_official_gotodoor_qwen_e2e.py** — Qwen-API-only GoToDoor end-to-end tests with per-action GIF recording and post-run oracle checks.
- **scripts/30_minigrid_saycan_acceptance.py** — unified strict acceptance over seven official task families, with smoke/standard/extended profiles.
- **scripts/31_all_minigrid_envs_gif_audit.py** — enumerates the complete installed MiniGrid catalog and writes GIFs, final frames, JSON traces and an HTML gallery.
- **scripts/32_collect_affordance_data.py** — collects belief/SkillCall/outcome
  JSONL records from real skill executions.
- **scripts/33_train_affordance.py** — trains and temperature-calibrates the
  NumPy affordance MLP using an environment/seed-group split.
- **scripts/34_evaluate_affordance.py** — reports held-out accuracy, Brier,
  calibration error, ROC-AUC and per-skill metrics.
- **tests/test_fetch_task_contract.py** — Fetch mission parsing and distractor-target filtering regressions.
- **scripts/08_saycan_planner_test.py** — one end-to-end custom task; accepts SEED.
- **scripts/09_batch_evaluation.py** — five task-language variants with CSV/JSON metrics.

Early tutorial scripts and tests tied to superseded primitive controllers are intentionally excluded from the v1.0 workflow.

## Install

Python 3.10+ is recommended.

    python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

For the exact versions used by the recorded acceptance result:

    python -m pip install -r requirements-tested.txt

## Verification commands

    python -m unittest discover -s tests -v
    python scripts/19_skill_library_v1_smoke_test.py
    python scripts/20_skill_acceptance_suite.py
    python scripts/21_skill_behavior_suite.py
    python scripts/22_hidden_state_audit.py
    SEED=0 python scripts/08_saycan_planner_test.py
    python scripts/23_official_doorkey_acceptance.py
    python scripts/25_official_basic_env_acceptance.py
    python scripts/26_official_putnear_acceptance.py --scorer keyword --seeds 0 1 2
    python scripts/30_minigrid_saycan_acceptance.py --profile standard
    python scripts/30_minigrid_saycan_acceptance.py --profile extended
    python scripts/31_all_minigrid_envs_gif_audit.py
    python scripts/32_collect_affordance_data.py
    python scripts/33_train_affordance.py
    python scripts/34_evaluate_affordance.py
    python scripts/30_minigrid_saycan_acceptance.py --profile extended \
      --affordance-model models/minigrid_affordance_v1.json

    # Qwen API is required; this runner never switches to keyword scoring.
    export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"
    python scripts/27_official_fetch_qwen_e2e.py \
      --env-ids MiniGrid-Fetch-6x6-N2-v0 --seeds 0
    python scripts/27_official_fetch_qwen_e2e.py \
      --env-ids MiniGrid-Fetch-5x5-N2-v0 MiniGrid-Fetch-6x6-N2-v0 MiniGrid-Fetch-8x8-N3-v0 \
      --seeds 0 1 2 3 4

    # Official GoToObject: the target is approached and confirmed with Done; it is not picked up.
    python scripts/28_official_gotoobject_qwen_e2e.py \
      --env-ids MiniGrid-GoToObject-6x6-N2-v0 MiniGrid-GoToObject-8x8-N2-v0 \
      --seeds 0 1 2 3 4

    python scripts/09_batch_evaluation.py

Multi-seed regression:

    mkdir -p outputs/multiseed
    for seed in $(seq 0 19); do
      SEED=$seed python scripts/08_saycan_planner_test.py || exit 1
    done

To use Qwen, configure DASHSCOPE_API_KEY and inject QwenSkillRelevanceScorer into UnifiedSayCanPlanner. With the key unset, the default scorer is offline keyword scoring; do not report that as an LLM-planner experiment.

The unified Qwen gate requires real API use and fails if any request falls back:

    python scripts/30_minigrid_saycan_acceptance.py --scorer qwen --profile smoke

To inspect current behavior outside the accepted seven-family scope, use the
[all-environment visual audit](docs/ALL_MINIGRID_VISUAL_AUDIT.md). With
MiniGrid 3.2.0 it enumerates 82 registered environments and keeps failures,
unsupported tasks and missing optional dependencies explicit.

The observation-grounded traversal layer now treats ordinary closed doors and
movable door blockers as implicit physical prerequisites. It opens each
observed unlocked door, clears a blocking ball/box before occupying the gripper
with a key, and releases a spent key before acquiring the mission target. These
behaviors use only the belief map. Memory references are resolved from the
observed cue before Qwen receives an endpoint candidate, and dynamic-obstacle
tasks use per-action collision checks plus observation-based pose correction.

The official Fetch demo is deliberately Qwen-API-only: it requires DASHSCOPE_API_KEY, records the API request count, fails if the planner falls back to keyword scoring, and writes one trajectory GIF plus final frame and JSON traces per environment/seed. Qwen scores the next semantic skill; deterministic feasibility supplies the observation-grounded affordance score. The combined relevance × feasibility score selects a skill, whose controller executes primitive actions through ObservationRuntime before the planner updates its belief and replans. The Fetch mission target is hard-constrained during candidate generation so distractor objects cannot become pickup candidates.

The official GoToObject demo is also Qwen-API-only and covers MiniGrid-GoToObject-6x6-N2-v0 and MiniGrid-GoToObject-8x8-N2-v0. Its mission target is hard-constrained against distractors. The intended skill sequence is Find → NavigateTo → Done: the object must remain unpicked, and acceptance requires the official environment to terminate with positive reward after the agent reaches the requested object and issues Done. The runner records per-action GIFs, Qwen requests, skill traces, and post-run oracle checks.

## Public skill contract

A call has the shape {"name":"PickUp","args":{"target":{"object":"ball","color":"red"}}}. Every result includes success, status, reason, skill and actions_executed. One registry invocation advances a skill by at most one primitive action. The planner may repeatedly invoke the controller under explicit budgets.

LLMs supply relevance scores only. In baseline mode Can is deterministic; in
learned mode it is a deterministic hard mask times a calibrated, learned
skill-success probability. All physical actions pass through
ObservationRuntime.execute_action. Ground-truth state is allowed in test setup
and post-run evaluation, never as input to skill decisions.

The learned-affordance mode implements the SayCan core as language relevance
times learned skill-success probability, with deterministic feasibility kept
as a hard safety mask. The bundled calibrated model and fully reproducible
collection/training/evaluation workflow are documented in
[docs/LEARNED_AFFORDANCE.md](docs/LEARNED_AFFORDANCE.md). Candidate traces keep
`hard_feasibility`, `learned_affordance` and their combined Can score separate.

## Freeze policy

Do not tag v1.0 as frozen until unit tests, the unified extended acceptance,
all-skill behavior suite, hidden-state audit, custom-task regression and the
intended LLM-backed acceptance pass on the same commit. Preserve JSON traces and
record Python/dependency versions. Passing the offline baseline alone is not an
LLM SayCan result.
