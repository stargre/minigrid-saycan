# MiniGrid SayCan acceptance protocol

## What is being accepted

This repository is a MiniGrid adaptation of the SayCan planning pattern:

1. **Say** assigns a semantic relevance score to every parameterized skill
   candidate.
2. **Can** assigns an observation-grounded affordance score to the same
   candidate set.
3. The planner selects the maximum `relevance * affordance` score, executes a
   closed-loop semantic skill, updates belief from observations and action
   history, and replans.

The original SayCan work uses learned value functions for affordance. This
MiniGrid implementation provides both a deterministic baseline and a calibrated
neural skill-success estimator trained from real skill executions over the
partial-observation belief map. Learned Can is multiplied by a deterministic
hard-safety mask. The keyword scorer remains an offline regression baseline;
only a successful Qwen run is an LLM-backed SayCan experiment.

References:

- [SayCan paper](https://arxiv.org/abs/2204.01691)
- [Official MiniGrid environments](https://minigrid.farama.org/environments/minigrid/)

## Acceptance matrix

The unified runner uses the official mission from each observation and covers:

| Family | Official environment | Required physical outcome |
|---|---|---|
| Empty | `MiniGrid-Empty-5x5-v0` | Reach the goal and terminate with reward |
| GoToDoor | `MiniGrid-GoToDoor-5x5-v0` | Face the requested door and issue Done |
| GoToObject | `MiniGrid-GoToObject-6x6-N2-v0` | Face the requested object and issue Done |
| Fetch | `MiniGrid-Fetch-5x5-N2-v0` | Hold the requested object and terminate with reward |
| PutNear | `MiniGrid-PutNear-6x6-N2-v0` | Release the requested object next to its target |
| Unlock | `MiniGrid-Unlock-v0` | Acquire the matching key and open the door |
| DoorKey | `MiniGrid-DoorKey-5x5-v0` | Unlock/open the door and reach the goal |

Each case must satisfy all of the following:

- planner-side success;
- a post-run simulator oracle;
- positive-reward terminal state where the official environment defines one;
- primitive-action count equality between the guarded environment and
  `ObservationRuntime`;
- no decision-time access to `env.unwrapped`, the simulator grid, true pose or
  `info`;
- in Qwen mode, at least one real API request and zero API-error fallbacks.

## Commands

Install the exact tested top-level dependencies:

```bash
python -m pip install -r requirements-tested.txt
```

Run the deterministic regression baseline:

```bash
python scripts/30_minigrid_saycan_acceptance.py --profile smoke
python scripts/30_minigrid_saycan_acceptance.py --profile standard
python scripts/30_minigrid_saycan_acceptance.py --profile extended
```

Profiles use 1, 5 and 20 seeds per family respectively. Any failed episode
produces a non-zero exit code. Full JSON traces are written under `outputs/`.

Run the LLM-backed acceptance:

```bash
export DASHSCOPE_API_KEY="..."
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export QWEN_MODEL="qwen-plus"
python scripts/30_minigrid_saycan_acceptance.py --scorer qwen --profile smoke
```

The API key and endpoint must belong to the same DashScope region. The runner
does not silently count keyword fallback as a Qwen success.

## Verified result

On Python 3.12.7, MiniGrid 3.2.0, Gymnasium 1.4.0 and NumPy 2.5.4:

- unit tests: **97/97 passed**;
- offline extended acceptance: **140/140 passed** (20 seeds x 7 families);
- learned-Can extended acceptance: **140/140 passed** (20 seeds x 7 families);
- learned affordance held-out accuracy: **98.63%**, Brier score **0.00994**,
  expected calibration error **0.00839**, ROC-AUC **0.99747**;
- Qwen smoke acceptance: **7/7 passed** on Python 3.12.15 with MiniGrid 3.2.0,
  Gymnasium 1.4.0 and NumPy 2.5.4 using a valid DashScope configuration.

Therefore the seven-family MiniGrid control/grounding baseline and the Qwen
smoke gate are accepted. This does not imply support for every registered
MiniGrid environment; use the separate all-environment visual audit to expose
the remaining capability gaps.

## MuJoCo migration boundary

Keep these interfaces stable when replacing MiniGrid:

- semantic call schema (`name`, `args`, `description`);
- relevance scorer contract (`candidate_id -> [0, 1]`);
- affordance evaluator contract (`feasibility`, `reason`, prerequisites);
- runtime action gateway and action accounting;
- post-run oracle isolation and JSON report schema.

Replace the MiniGrid belief decoder, low-level skill policies, feature encoder
and oracle with MuJoCo-specific implementations. The learned evaluator and
high-level planner interfaces remain unchanged; retrain the value model from
MuJoCo skill outcomes.
