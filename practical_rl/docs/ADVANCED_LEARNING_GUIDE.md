# Advanced practical RL: lectures 11–25

This guide extends the foundational implementations through maximum-entropy,
model-based, offline, exploratory, goal-conditioned, preference-based, transfer,
and hierarchical reinforcement learning.  Every implementation is deliberately
small enough to trace from its objective to the corresponding PyTorch update.

## 1. Soft Actor-Critic

File: `edu_rl/advanced/sac.py`

SAC learns two Q-functions and uses their minimum to reduce optimistic errors.
Its target includes future policy entropy:

```text
y = r + gamma (1-terminal) [min(Q1_target,Q2_target) - alpha log pi(a'|s')]
```

The actor minimizes `alpha log pi - min(Q1,Q2)`.  A separate optimizer adjusts
`log(alpha)` toward a target entropy.  Compare this carefully with DDPG: SAC is
stochastic, explicitly values entropy, and uses two critics.

## 2. CEM, MPC, Dyna, and MBPO

File: `edu_rl/advanced/model_based.py`

- CEM samples action sequences, keeps an elite subset, and refits its Gaussian.
- MPC executes only the first planned action and replans from the next state.
- Dyna-Q mixes real Q-updates with updates sampled from a learned tabular model.
- The MBPO-style implementation trains a probabilistic dynamics ensemble,
  starts short synthetic rollouts from real replay states, and mixes real and
  model transitions in SAC updates.

Model disagreement is logged as an epistemic-uncertainty diagnostic.  Increasing
synthetic horizon uses more imagined experience but exposes the learner to more
compounding model error.

## 3. Offline RL

File: `edu_rl/advanced/offline_rl.py`

All trainers consume an immutable `OfflineDataset`; only the separate generator
interacts with an environment.

- SAC+BC adds dataset action likelihood to the actor objective.
- AWAC performs weighted behavioral cloning with exponentiated advantages.
- IQL learns an expectile value, uses it for in-distribution Q targets, and
  extracts a policy with advantage-weighted regression.
- CQL adds a log-sum-exp critic penalty over random and policy actions while
  preserving values on dataset actions.

Inspect predicted Q-values as well as evaluation return.  Offline failure often
appears first as unreasonable values for actions absent from the dataset.

## 4. UCB, count bonuses, and RND

File: `edu_rl/advanced/exploration.py`

UCB adds a confidence bonus that shrinks with an arm's visit count.  The tabular
count-bonus example transfers that idea to states.  RND replaces explicit counts
with prediction error against a fixed random target network.  The target must
remain frozen; only the predictor is optimized.

## 5. Goal-conditioned learning and HER

Files: `edu_rl/envs/bitflip.py`, `edu_rl/advanced/her.py`

The Q-network observes both current state and desired goal.  Future-strategy HER
relabels each transition with goals achieved later in the same episode, then
recomputes reward and termination.  It does not pretend the original commanded
goal was reached.

## 6. GAIL, DIAYN, and Skew-Fit

File: `edu_rl/advanced/imitation_skills.py`

- GAIL alternates an expert/learner discriminator with PPO updates using
  discriminator-derived reward.
- DIAYN samples one skill per episode, trains a skill classifier, and rewards
  states that make the selected skill identifiable.
- Skew-Fit trains a small goal VAE with inverse-density weighting, samples new
  goals from it, and trains a goal-conditioned SAC policy.  Coverage is measured
  independently of return.

GAIL is adversarial because the discriminator opposes the policy.  DIAYN's
classifier is cooperative: better skill prediction gives the policy more reward.

## 7. GRPO and miniature RLHF

File: `edu_rl/advanced/preference_rl.py`

A small GRU emits sequences for several prompts.  GRPO samples a group of
completions per prompt and normalizes reward within that group—there is no value
network.  The policy uses PPO-style ratios and a frozen reference-policy penalty.

The RLHF exercise creates pairwise preferences from a hidden oracle, trains a
Bradley–Terry reward model, and then runs GRPO against learned reward.  Both true
and learned reward are retained so reward-model exploitation can be observed.

## 8. Successor features and hierarchical options

File: `edu_rl/advanced/transfer_hierarchy.py`

Successor features predict discounted future feature occupancy.  New linear
reward weights immediately turn those predictions into Q-functions.  GPI takes
the best action suggested by any stored policy at each state.

The fixed-options experiment performs semi-MDP Q-learning; an option lasting
`k` primitive steps bootstraps with `gamma**k`.  Option-Critic then learns option
values, intra-option action policies, and termination probabilities jointly.

## Experiment discipline

Use `--preset quick` only to verify wiring.  `learn` provides a reasonable first
run; `full` is deliberately expensive.  Compare at least three seeds, preserve
the exact dataset for offline comparisons, and change one mechanism at a time.

