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
| Passed | 41 |
| Failed | 34 |
| Unsupported (empty mission) | 1 |
| Unavailable (missing WFC extras) | 6 |

This single-seed catalog run is an effect audit, not a statistical acceptance
result. The strict seven-family suite remains the regression gate.

Major remaining capability gaps exposed by the audit include dynamic-obstacle
reaction, KeyCorridor/locked-room acquisition chains, multi-room traversal
through unrequested closed doors, obstructed-maze obstacle relocation and
ordered multi-door tasks. WFC environments require `minigrid[wfc]` before they
can be instantiated. `MiniGrid-Playground-v0` has no mission and therefore
cannot be evaluated as a language-conditioned SayCan task without a supplied
instruction.

