# All-environment visual capability audit

`scripts/31_all_minigrid_envs_gif_audit.py` is a discovery and visualization
runner for the complete environment catalog installed with MiniGrid. It is not
an acceptance shortcut: unsupported mechanisms remain visible as failures.

## What it produces

For every environment that can be instantiated, the runner writes:

- `trajectory.gif` with a bounded number of rendered frames;
- `final_frame.png`;
- `result.json` with mission, planner/environment success, action counts,
  Say/Can scores and the semantic skill trace.

At the output root it writes:

- `manifest.json` with all cases and aggregate counts;
- `index.html`, a local gallery containing all GIFs and statuses.

The default output is `outputs/all_minigrid_visual_audit/` and is intentionally
git-ignored.

## Usage

Audit all registered environments with the offline scorer:

```bash
python scripts/31_all_minigrid_envs_gif_audit.py
```

Open the generated gallery:

```bash
python -m http.server 8000 -d outputs/all_minigrid_visual_audit
# then visit http://localhost:8000/
```

Filter the catalog with shell-style patterns:

```bash
python scripts/31_all_minigrid_envs_gif_audit.py \
  --match "*Door*" "*Unlock*" "*Fetch*"
```

Run selected environments with Qwen (recommended before a costly all-catalog
LLM run):

```bash
python scripts/31_all_minigrid_envs_gif_audit.py \
  --scorer qwen \
  --env-ids MiniGrid-DoorKey-5x5-v0 MiniGrid-Fetch-5x5-N2-v0
```

Use `--strict` in CI when every attempted case in a selected subset is expected
to pass. The default catalog audit always finishes so all failure GIFs remain
available for diagnosis.

## Current seed-0 catalog result

Tested with MiniGrid 3.2.0 and the offline scorer:

| Classification | Count |
|---|---:|
| Registered environments | 82 |
| Attempted | 75 |
| Passed | 60 |
| Failed | 15 |
| Unsupported (empty mission) | 1 |
| Unavailable (missing WFC extras) | 6 |

This single-seed catalog run is an effect audit, not a statistical acceptance
result. The strict seven-family suite remains the regression gate.

The optimized seed-0 run passes every registered KeyCorridor, MultiRoom,
Memory and DynamicObstacles
variant. It also passes BlockedUnlockPickup, UnlockPickup and the basic
ObstructedMaze-1Dl task by treating ordinary doors, door blockers and spent
keys as observation-grounded physical prerequisites.

The remaining capability gaps include LockedRoom's cross-room key search,
multi-obstacle ObstructedMaze variants and ordered RedBlueDoors interaction.
WFC environments require `minigrid[wfc]` before they can be instantiated.
`MiniGrid-Playground-v0` has no mission and therefore cannot be evaluated as a
language-conditioned SayCan task without a supplied instruction.

## Qwen accuracy and grounded references

Before the Memory fix, the seed-0 keyword audit passed 54/75 while Qwen passed
48/75. The six-case difference was exactly the six Memory variants. Their
mission says "matching object" but both endpoints are green; the previous Qwen
payload did not contain the start-room cue, so the LLM had to guess between a
green key and green ball. Keyword happened to choose the correct seed-0 branch
through deterministic ID ordering, which was not real grounding.

The resolver now withholds both endpoint candidates, actively observes the
start cue, and exposes only the endpoint whose object type matches that cue.
Qwen receives explicit `memory_probe` and `grounded_reference` progress hints.
This preserves the Say/Can boundary: language relevance does not guess a
perceptual referent that the observation-grounding layer can resolve exactly.

DynamicObstacles use a separate `reactive_obstacles` mode. Balls are treated as
transient occupancy rather than permanent walls; the controller checks the
next cell before each forward action and replans after every observation.
Static visual landmarks determine whether a forward action actually moved the
agent, preventing belief-pose drift when a ball moves into the intended cell.

## Why and where traversal was fixed

- `skills/dependency.py` derives per-position `OpenDoor` and `MoveObstacle`
  calls from observed topology. Call IDs include positions, so opening one door
  does not suppress another same-colored door.
- `skills/feasibility.py` turns physical ordering into Can constraints: a
  blocker must be cleared before its lock's key occupies the gripper, and
  already-open doors cannot attract stale navigation.
- `skills/manipulation_skills.py` selects an observed reachable drop cell away
  from still-closed doors, moves blocking balls/boxes, and releases a spent key
  before target pickup.
- `planners/unified_saycan_planner.py` commits a ready topology-changing
  interaction before unrelated navigation moves the agent away. Every
  candidate still appears in the complete Say/Can ranking and trace.
- `skills/observation_explorer.py` performs the two-stage Memory probe: observe
  the cue, then search the original hallway direction for endpoint choices.
- `skills/navigation.py` implements transient-obstacle path planning and a
  per-action collision guard.
- `perception/belief_state.py` confirms forward motion from static visual
  landmarks so moving objects cannot corrupt the estimated pose.
- `llm/skill_relevance.py` tells Qwen when a reference has been uniquely
  grounded or a reactive controller is required.
