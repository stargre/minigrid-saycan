# Learned affordance for MiniGrid SayCan

This repository implements the central SayCan factorization as:

```text
score(skill) = Say(instruction, skill) × Can(belief, skill)
Can = deterministic hard safety mask × learned skill-success probability
```

The deterministic mask prevents physically invalid actions (for example,
unlocking without a matching key). The learned model predicts the probability
that a hard-feasible, parameterized skill will complete successfully from the
current observation-derived belief. The language model never supplies Can.

## What is learned

`AffordanceFeatureEncoder` converts the partial belief and a `SkillCall` into an
85-dimensional vector. It includes observed topology summaries, frontiers,
inventory, door states, target visibility/distance, skill identity and target
arguments. It never reads `env.grid`, true agent coordinates or simulator
inventory.

`LearnedAffordanceModel` is a small NumPy MLP trained with weighted binary
cross entropy on real skill execution outcomes. A held-out environment/seed
split is used for validation and temperature scaling. The saved JSON contains
the feature schema hash, normalization statistics, weights, calibration
temperature and training metrics. Loading fails closed if the schema changed.

## Reproduce the pipeline

Collect 20 seeds across the default 16 official environments:

```bash
python scripts/32_collect_affordance_data.py \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
```

Train and calibrate:

```bash
python scripts/33_train_affordance.py
```

Evaluate only the held-out environment/seed groups:

```bash
python scripts/34_evaluate_affordance.py
```

Run the strict policy acceptance with learned Can:

```bash
python scripts/30_minigrid_saycan_acceptance.py \
  --profile extended \
  --affordance-model models/minigrid_affordance_v1.json
```

Use Qwen Say together with learned Can:

```bash
export DASHSCOPE_API_KEY="YOUR_KEY"
python scripts/30_minigrid_saycan_acceptance.py \
  --scorer qwen --profile smoke \
  --affordance-model models/minigrid_affordance_v1.json
```

## Recorded baseline

The bundled model was trained from 1,492 actual skill executions collected
from 320 successful episodes (20 seeds × 16 environment configurations). The
environment/seed-group validation split contains 292 executions. Its recorded
metrics are:

- accuracy: 0.9863;
- Brier score: 0.00994;
- 10-bin expected calibration error: 0.00839;
- ROC-AUC: 0.99747.

Learned-affordance policy acceptance passes 140/140 official episodes over the
seven strict task families. A separate 20-seed stress set covering LockedRoom,
RedBlueDoors and ObstructedMaze-2Dl passes 60/60.
The complete seed-0 catalog remains 64/75, identical to deterministic Can;
all 11 failures are the already documented hidden-box/large ObstructedMaze
scope. Skills absent from training data never receive an extrapolated neural
score: the evaluator records `unseen_skill_fallback` and retains deterministic
Can for that skill.

These metrics measure the present MiniGrid skill distribution. They do not
claim that the model generalizes to MuJoCo observations. For migration, retain
the evaluator interface and replace the feature encoder and skill policies;
collect new outcomes and retrain the success model on the new embodiment.

## Scientific boundary

This is a genuine learned affordance because its probability comes from skill
execution outcomes, not from deterministic feasibility labels. It is still a
MiniGrid reproduction rather than the original robot system: the low-level
skills are deterministic controllers and the input is a compact belief vector,
whereas the original work learned robot skill value functions from robot data.
The hard mask is an engineering safety layer and should be reported separately
from the learned probability in experiments; both are included in every
candidate ranking trace.
