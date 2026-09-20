# Learning guide

## 1. Exact planning and sampled learning

Start in `envs/tiny_mdp.py` and `envs/gridworld.py`.  Both expose transition
probabilities `P[s,a,s']` and conditional rewards `R[s,a,s']`.  This lets
`dynamic_programming.py` compute ground truth before `tabular_learning.py`
attempts to recover similar values from samples.

The central contrast is:

```text
Bellman expectation: V(s) <- sum_a pi(a|s) sum_s' P(s'|s,a)[r + gamma V(s')]
Bellman optimality:  V(s) <- max_a          sum_s' P(s'|s,a)[r + gamma V(s')]
TD(0):               V(s) <- V(s) + alpha [r + gamma V(s') - V(s)]
Q-learning:          Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]
```

Watch how exact methods sweep all state/action successors while TD and
Q-learning use one observed successor.  Also inspect the explicit terminal
mask: true termination removes the bootstrap; a time limit does not.

## 2. Behavioral cloning and DAgger

`imitation.py` uses the optimal GridWorld policy as an expert.  Behavioral
cloning minimizes cross-entropy on expert states.  DAgger then executes the
learner and asks the expert to label the learner's states.

The key experiment is to reduce the initial expert dataset.  BC can have low
training loss while performing badly because its own mistakes lead to states
absent from that dataset.  DAgger repairs the dataset rather than merely fitting
the original labels for more epochs.

## 3. REINFORCE

In `reinforce.py`, compare these weights on the same log-probability gradient:

```text
full return:       total trajectory return
reward-to-go:      return after the current action
baseline version:  reward-to-go - V_phi(s)
```

Reward-to-go removes rewards that occurred before an action.  The baseline
does not change the population gradient when it is action-independent, but it
usually reduces variance.  Returns and critic predictions are detached from the
actor loss because they are weights, not a path through the environment.

## 4. Actor–critic, n-step returns, and GAE

`actor_critic.py` keeps the training loop fixed and swaps only the advantage
estimator.  `td0` relies most heavily on the critic, `nstep` uses several real
rewards before bootstrapping, and `gae` mixes all trace lengths exponentially.

`core/returns.py` highlights a subtle distinction:

- `terminated` masks a nonexistent continuation value.
- `episode_ends` also includes truncations and rollout boundaries; it stops a
  trace from crossing a reset while retaining the valid final bootstrap.

Compare GAE lambda values 0, 0.95, and 1 across multiple seeds.

## 5. DQN and Double DQN

`dqn.py` contains three separate networks/roles even though there are only two
neural-network objects:

1. The online network predicts the selected action's Q value.
2. A frozen target network supplies a stable regression label.
3. In Double DQN, the online network selects the next action while the target
   network evaluates it.

Replay breaks short-range temporal correlations and reuses transitions.  It
does not guarantee coverage.  A target network slows label movement.  It does
not by itself remove maximization bias.  The Huber loss limits the influence of
large early TD errors.

## 6. DDPG

`ddpg.py` replaces discrete maximization with a deterministic actor:

```text
critic target: r + gamma Q_target(s', mu_target(s'))
actor loss:   -mean Q(s, mu(s))
```

The critic is frozen during the actor backward pass: gradients must flow
through its action input into the actor, but the actor optimizer must not modify
critic parameters.  Exploration noise affects collection only.  Evaluation
should use the deterministic actor without noise.

## 7. Importance sampling and PPO

`importance_sampling.py` first isolates the probability-ratio idea in a
one-step bandit.  Inspect ordinary versus weighted estimates and the effective
sample size as behavior and target policies separate.

`ppo.py` saves each sampled action's old log-probability, then forms

```text
ratio = exp(log pi_new(a|s) - log pi_old(a|s)).
```

Clipped PPO limits the incentive from ratios outside a chosen interval; it does
not literally constrain every update.  KL-penalty PPO instead subtracts an
empirical drift penalty and adapts its coefficient.  Track the actual KL and
clip fraction, not just return.

For continuous actions the policy stores the pre-tanh sample.  Recomputing its
log-probability includes the tanh Jacobian correction, which is required for a
valid likelihood ratio.

## 8. Natural gradient and TRPO-style line search

`natural_policy_gradient.py` never builds the Fisher matrix explicitly.
Automatic differentiation computes Fisher-vector products, conjugate gradient
approximately solves `F x = g`, and the result is scaled to a KL budget.

Natural policy gradient accepts that local step directly.  The TRPO-style
variant backtracks until the sampled surrogate improves and the measured mean
KL lies within budget.  This is an educational compact implementation: the
guarantee discussed in theory uses stronger distributional assumptions than a
finite empirical batch can establish.

## Suggested investigations

- Plot estimator variance for REINFORCE variants before comparing reward.
- Deliberately remove DQN's target network and observe target/loss oscillation.
- Compare DQN and Double DQN's predicted maxima with evaluation returns.
- Sweep GAE lambda while holding every other configuration fixed.
- Track PPO's clip fraction as update epochs increase.
- Tighten TRPO's KL budget and inspect accepted line-search fractions.
- Compare CPU and CUDA throughput only after correctness is established; these
  small environments are often limited by environment stepping, not the MLP.

