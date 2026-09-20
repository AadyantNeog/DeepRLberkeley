# Remaining course algorithms: practical guide

This guide covers the algorithms added in the final implementation phase. The
code favors visible equations and small experiments over benchmark performance.
Every family can be run with `python -m edu_rl.run_remaining FAMILY`; use
`--quick` only to verify wiring and `--preset learn` to observe learning.

## 1. Coordinate-ascent VI and EM

File: `edu_rl/advanced/generative_models.py`

`coordinate_ascent_gaussian_vi` starts with a Gaussian written in information
form:

```text
p(x) proportional to exp(-1/2 x^T Lambda x + eta^T x)
```

Under a mean-field approximation, factor `i` has variance `1/Lambda_ii` and its
mean is updated using the current means of every other factor. The factorized
mean converges to the exact Gaussian mean, but its variances omit correlations.

The diagonal Gaussian-mixture implementation alternates:

```text
E-step: r_nk = p(z_n=k | x_n, current parameters)
M-step: weighted maximum-likelihood means, variances, and mixture weights
```

The log likelihood should never decrease, up to floating-point noise. The unit
test checks this invariant.

## 2. VAE, CVAE, and sequential VAE

File: `edu_rl/advanced/generative_models.py`

All three use the reparameterization `z = mean + std * epsilon` and minimize a
negative evidence lower bound:

```text
reconstruction loss + beta * KL(q(z|x) || N(0,I))
```

- `VAE` models unconditioned data.
- `ConditionalVAE` models actions conditioned on observations.
- `SequentialVAE` uses GRUs and one stochastic latent per time step.

Try increasing `beta`. Reconstructions usually become less exact while the
latent distribution becomes smoother and easier to sample.

## 3. Flow matching, Reflow, diffusion, and autoregression

File: `edu_rl/advanced/generative_models.py`

Flow matching regresses a vector field on straight paths from Gaussian noise
`x0` to data `x1`:

```text
x_t = (1-t)x0 + t*x1
target velocity = x1 - x0
```

Sampling integrates the learned ODE. Reflow first samples source/destination
pairs from a trained flow, then fits a second, straighter transport between each
pair. A straighter field needs fewer integration steps.

The diffusion policy corrupts dataset actions according to a DDPM schedule and
learns to predict the added noise. Its reverse process supports a Q-gradient at
every denoising step. The autoregressive policy instead applies the probability
chain rule, conditioning every action coordinate on preceding coordinates.

## 4. Soft Q-learning, maximum-entropy IRL, and GCL

File: `edu_rl/advanced/maxent_irl.py`

Soft value iteration replaces `max Q` with:

```text
V(s) = alpha * logsumexp_a(Q(s,a) / alpha)
```

The induced policy is Boltzmann in Q. Model-free Soft Q-learning uses the same
soft value in its TD target.

Maximum-entropy IRL learns linear reward weights by matching expert and learner
feature occupancies. Guided Cost Learning learns a trajectory energy model and
corrects its discriminator logits by the current proposal log probability. It
alternates reward updates with a new soft policy, making the changing sampler
part of the learning algorithm rather than an ignored source of bias.

The implementation uses exact tabular dynamics so occupancies and proposal
probabilities can be inspected directly.

## 5. Random shooting, differentiable planning, and MVE

File: `edu_rl/advanced/model_based_extensions.py`

Random shooting samples complete action sequences once and executes the first
action from the best sequence. Compare it with CEM in `model_based.py`, which
repeatedly refits its sampling distribution.

Differentiable planning treats the action sequence as trainable parameters and
backpropagates return through the dynamics. This is powerful when the model is
smooth, but it can exploit model errors or fail on discontinuities.

Model Value Expansion builds a target from `H` model rewards and a terminal
value bootstrap:

```text
y = r_0 + gamma*r_1 + ... + gamma^(H-1)*r_(H-1) + gamma^H*V(s_H)
```

True termination masks every reward and bootstrap after it; time-limit
truncation does not.

## 6. MOPO and COMBO

File: `edu_rl/advanced/model_based_extensions.py`

Both methods fit a dynamics ensemble and start short model rollouts from states
in a fixed offline dataset.

- MOPO subtracts ensemble disagreement from predicted reward. This pessimism
  discourages the policy from exploiting uncertain model regions.
- The compact COMBO path performs conservative Q-learning on real plus model
  transitions, pushing down values for unsupported actions in model-generated
  regions. `combo_conservative_penalty` exposes the paper's real-versus-model
  conservative term separately for study and testing.

These are educational, low-dimensional variants. The paper-scale versions use
larger ensembles, tuned rollout schedules, normalization, and benchmark-specific
training details.

## 7. BRAC, IDQL, FQL, and diffusion steering

File: `edu_rl/advanced/offline_extensions.py`

BRAC first fits a behavior policy `mu(a|s)`. Its actor balances Q maximization
against a sampled forward-KL penalty `log pi - log mu`, keeping learned actions
near dataset support.

IDQL combines an IQL critic with a diffusion behavior model. At inference it
generates several in-distribution actions and resamples them using exponentiated
Q advantages. The critic and expressive policy are trained separately.

The compact FQL implementation trains a conditional flow behavior policy, then
distills its multi-step sampler into a one-step actor while improving that actor
with the offline critic. Diffusion steering takes the alternative approach: it
retains iterative denoising and injects Q gradients into each reverse step.

Actions are normalized to `[-1,1]` only inside the generative model and are
explicitly mapped back to environment bounds before critic evaluation.

## 8. Density pseudo-counts and CTS

File: `edu_rl/advanced/exploration_extensions.py`

A density pseudo-count asks: what literal visit count would have caused the
same increase from `p_n(x)` to `p_(n+1)(x)`?

```text
N_hat(x) = p_n(x) * (1 - p_(n+1)(x)) / (p_(n+1)(x) - p_n(x))
```

The intrinsic bonus is proportional to `1/sqrt(N_hat)`. The CTS example is an
online binary density: KT estimators predict at several context depths, while a
fixed-share Bayesian update lets probability mass switch between depths as the
sequence structure changes. This is the essential mechanism in CTS exploration,
shown without the image-specific bit-plane engineering used on Atari.

## 9. Goal-conditioned empowerment

File: `edu_rl/advanced/exploration_extensions.py`

Empowerment is channel capacity from action sequences to achieved future goals:

```text
max over p(action sequence) of I(action sequence ; achieved goal | state)
```

The code enumerates short action sequences in Four Rooms and solves the channel
capacity with Blahut-Arimoto. In deterministic dynamics it reduces to the log of
the number of independently reachable outcomes; the same solver also supports
stochastic channels.

## 10. TD3 and continuous successor features

File: `edu_rl/advanced/continuous_extensions.py`

TD3 adds three mechanisms to DDPG:

1. twin critics and a minimum target;
2. noise on target actions to smooth sharp value errors;
3. delayed actor and target-network updates.

The continuous successor-feature experiment predicts discounted future RBF
feature occupancy under a fixed policy. A learned reward vector maps the same
representation to values, illustrating how dynamics/policy knowledge can be
reused when a task's linear reward changes.

## Recommended order

1. CAVI and EM, then VAE/CVAE/sequential VAE.
2. Flow matching, Reflow, diffusion, and autoregressive policies.
3. Soft Q-learning, maximum-entropy IRL, and GCL.
4. Shooting, differentiable planning, MVE, MOPO, and COMBO.
5. BRAC, IDQL, FQL, and diffusion steering.
6. Pseudo-counts, CTS, and empowerment.
7. TD3 and continuous successor features.

For each method, first read its equation-level test, run the focused variant
with `--quick`, then run at least three `learn` seeds. A quick run proves only
that the data path works; it does not establish convergence or rank algorithms.
