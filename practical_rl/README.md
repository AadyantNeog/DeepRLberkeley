# Practical RL: lectures 1–25 in code

This folder is a from-scratch companion to the lecture notes.  Its purpose is
to expose the equations and failure modes—not to compete with a production RL
framework.  NumPy is used when a table makes the mathematics visible; PyTorch
is introduced only when automatic differentiation or neural approximation is
part of the algorithm.

## What is implemented

| Topic | Implementation | Suggested environment |
|---|---|---|
| Exact planning | policy evaluation, policy iteration, value iteration | TinyMDP, GridWorld |
| Sample-based prediction/control | first-visit Monte Carlo, TD(0), Q-learning | GridWorld |
| Imitation learning | behavioral cloning, DAgger | GridWorld with an exact expert |
| Monte Carlo policy gradient | full-return REINFORCE, reward-to-go, value baseline | CartPole |
| Actor–critic | TD(0), n-step, GAE | CartPole |
| Value-based deep RL | DQN and Double DQN with replay/targets | CartPole or Acrobot |
| Continuous off-policy RL | DDPG | Pendulum |
| Off-policy estimation | ordinary and weighted importance sampling | Bernoulli bandit |
| Local policy optimization | clipped PPO and adaptive KL-penalty PPO | CartPole or Pendulum |
| Trust-region geometry | natural policy gradient and TRPO-style line search | CartPole or Pendulum |

The post-Lecture-10 phase is also implemented:

| Area | Implementations |
|---|---|
| Maximum-entropy RL | SAC with twin critics and automatic temperature |
| Model-based RL | CEM, MPC, Dyna-Q, ensemble dynamics, MBPO-style rollouts |
| Offline RL | SAC+BC, AWAC, IQL, CQL |
| Exploration and goals | UCB, count bonuses, RND, goal-conditioned DQN, HER |
| Imitation and skills | GAIL, DIAYN, Skew-Fit |
| Preference RL | GRPO, Bradley–Terry reward modeling, miniature RLHF |
| Transfer and hierarchy | successor features, GPI, options, Option-Critic |

The final coverage phase fills the remaining lecture algorithms:

| Area | Implementations |
|---|---|
| Variational inference | Gaussian mean-field CAVI, Gaussian-mixture EM, VAE, CVAE, sequential VAE |
| Generative policies | flow matching, Reflow, conditional diffusion, autoregressive continuous actions |
| Maximum-entropy imitation | standalone Soft Q-learning, maximum-entropy IRL, Guided Cost Learning |
| Model-based extensions | random shooting, differentiable planning, MVE, MOPO, COMBO |
| Modern offline RL | BRAC, IDQL, FQL, diffusion Q-steering |
| Exploration | density pseudo-counts, Context Tree Switching, goal-conditioned empowerment |
| Continuous control/transfer | TD3, neural continuous successor features |

## Environment

No virtual environment is required.  The project uses the machine's existing
Python/PyTorch installation.  Dependencies are recorded in `requirements.txt`;
Gymnasium Classic Control and TensorBoard were installed for this workspace.

From this folder, verify everything with:

```powershell
$rlPython = 'C:\Users\Aadyant Neog\AppData\Local\Programs\Python\Python313\python.exe'
& $rlPython -m pytest
```

## Running experiments

Start with the exact tabular reference:

```powershell
$rlPython = 'C:\Users\Aadyant Neog\AppData\Local\Programs\Python\Python313\python.exe'
& $rlPython -m edu_rl.run tabular
```

Then work through the neural algorithms:

```powershell
# Imitation learning: both BC and DAgger
& $rlPython -m edu_rl.run imitation

# All three REINFORCE estimators, or choose --variant baseline
& $rlPython -m edu_rl.run reinforce --variant all

# TD(0), n-step, and GAE actor–critic
& $rlPython -m edu_rl.run actor-critic --variant all

# Standard and Double DQN
& $rlPython -m edu_rl.run dqn --variant all

# Continuous-action DDPG
& $rlPython -m edu_rl.run ddpg

# Clipped and KL-penalty PPO
& $rlPython -m edu_rl.run ppo --variant all

# The same PPO implementation with a squashed Gaussian policy
& $rlPython -m edu_rl.run ppo --variant clip --env Pendulum-v1

# Natural gradient and TRPO-style backtracking
& $rlPython -m edu_rl.run natural --variant all

# Continuous-policy trust-region path
& $rlPython -m edu_rl.run natural --variant trpo --env Pendulum-v1

# A transparent importance-sampling diagnostic
& $rlPython -m edu_rl.run importance-sampling
```

Add `--quick` to any command for a short integration check.  A quick run only
checks that data flows through the implementation; it is deliberately too short
to judge learning performance.  Normal runs save the configuration, CSV
metrics, TensorBoard events, plots, and model state under `runs/`.

For TensorBoard:

```powershell
& $rlPython -m tensorboard.main --logdir runs
```

## Recommended learning workflow

1. Read the matching section in `docs/LEARNING_GUIDE.md`.
2. Locate the equation in the algorithm file.
3. Run the relevant unit test and inspect its hand-computed expectation.
4. Run one seed with `--quick` to understand the output files.
5. Run at least three full seeds before comparing an algorithm or ablation.
6. Change one mechanism at a time: baseline, lambda, target network, Double-DQN
   selection, PPO clip range, or KL budget.

RL results are distributions, not single numbers.  A single lucky seed is not
evidence that one method is superior.

## Advanced experiments

Read `docs/ADVANCED_LEARNING_GUIDE.md`, then use the advanced runner:

```powershell
# Verify every advanced implementation without a long training run
& $rlPython -m edu_rl.run_advanced all --quick

# Examples of focused learning runs
& $rlPython -m edu_rl.run_advanced sac --preset learn
& $rlPython -m edu_rl.run_advanced model-based --variant mbpo --preset learn
& $rlPython -m edu_rl.run_advanced offline --variant iql --preset learn
& $rlPython -m edu_rl.run_advanced exploration --variant rnd --preset learn
& $rlPython -m edu_rl.run_advanced her --preset learn
& $rlPython -m edu_rl.run_advanced skills --variant diayn --preset learn
& $rlPython -m edu_rl.run_advanced preference --variant rlhf --preset learn
& $rlPython -m edu_rl.run_advanced transfer --variant option-critic --preset learn
```

For the final coverage phase, read `docs/REMAINING_LEARNING_GUIDE.md` and use:

```powershell
# Verify all final-phase implementations and artifact generation
& $rlPython -m edu_rl.run_remaining all --quick

# Focused examples
& $rlPython -m edu_rl.run_remaining generative --variant flow --preset learn
& $rlPython -m edu_rl.run_remaining maxent --variant maxent-irl --preset learn
& $rlPython -m edu_rl.run_remaining model-based --variant mopo --preset learn
& $rlPython -m edu_rl.run_remaining offline --variant idql --preset learn
& $rlPython -m edu_rl.run_remaining exploration --variant empowerment --preset learn
& $rlPython -m edu_rl.run_remaining continuous --variant td3 --preset learn
```

Every run writes its configuration, CSV metrics, TensorBoard events, plots, and
primary model checkpoint under `runs/`.  The quick preset checks execution; it
is too short to demonstrate convergence.

Together, the three runners provide the course's important algorithmic spine.
Large systems mentioned as case studies—such as Dreamer and SLAC—are not labeled
as single algorithms here: faithfully reproducing them requires substantial
architectures, replay systems, and benchmark suites beyond an educational
from-scratch implementation.
