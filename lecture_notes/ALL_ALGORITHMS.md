# CS 185/285 Deep Reinforcement Learning: Algorithm Reference

This file is a procedure-first reference for the algorithms covered across Lectures 1-25. It intentionally omits derivations and extended theory. Each entry states what to do, what its symbols mean, when the method is appropriate, the assumptions it relies on, and its main limitations.

## Scope

- Included: every method for which the lectures give an update rule, optimization objective, or operational loop; important wrappers such as action chunking and model-predictive control; and the algorithmic recipes revisited in the review lectures.
- Grouped variants: closely related variants share an entry when their steps differ by only one target or loss. The difference is always stated explicitly.
- Not expanded as full algorithms: systems or methods that the course only names or uses as examples without specifying their procedure are listed in the final appendix.
- “Repeat until done” below means until a stated iteration budget is exhausted or a validation/evaluation criterion stops improving. It is not a convergence guarantee.

## Contents

1. [Imitation and expressive behavior models](#1-imitation-and-expressive-behavior-models)
2. [Policy gradients, value estimation, and actor-critic](#2-policy-gradients-value-estimation-and-actor-critic)
3. [Dynamic programming and value-based RL](#3-dynamic-programming-and-value-based-reinforcement-learning)
4. [Data reuse and constrained policy optimization](#4-data-reuse-and-constrained-policy-optimization)
5. [Variational inference and maximum-entropy control](#5-variational-inference-and-maximum-entropy-control)
6. [Inverse RL, adversarial imitation, and preference optimization](#6-inverse-rl-adversarial-imitation-and-preference-optimization)
7. [Model-based RL and planning](#7-model-based-reinforcement-learning-and-planning)
8. [Offline and offline-to-online RL](#8-offline-and-offline-to-online-reinforcement-learning)
9. [Exploration and unsupervised skill discovery](#9-exploration-and-unsupervised-skill-discovery)
10. [Goal-conditioned, transfer, and hierarchical RL](#10-goal-conditioned-transfer-and-hierarchical-reinforcement-learning)
11. [Lecture coverage map](#11-lecture-coverage-map)
12. [Methods named but not fully specified](#12-methods-named-or-illustrated-but-not-specified-as-full-algorithms)
13. [Selection reminders](#13-selection-reminders)

## Global notation and symbols

| Symbol | Meaning |
|---|---|
| $s_t$, $o_t$, $a_t$, $r_t$ | State, observation, action, and reward at environment time $t$ |
| $s_{t+1}$, $d_t$ | Next state and true terminal indicator; $d_t=1$ means no bootstrap is allowed |
| $T$ or $H$ | Episode horizon or sampled trajectory length |
| $\tau=(s_0,a_0,r_0,\ldots)$ | A trajectory; in flow models, internal flow time is written $u$ to avoid overloading $\tau$ |
| $P(s'\mid s,a)$ | Environment transition distribution |
| $\rho_0(s)$ | Initial-state distribution |
| $\pi_\theta(a\mid s)$ | Policy with parameters $\theta$ |
| $\pi_\beta$ or $\mu$ | Behavior policy that generated stored data |
| $\pi_{\mathrm{old}}$ | Frozen policy that generated the current on-policy batch |
| $\gamma\in[0,1)$ | Discount factor; finite-horizon work may use $\gamma=1$ |
| $G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k$ | Monte Carlo reward-to-go from time $t$ |
| $V^\pi(s)$ | Expected discounted return after starting at $s$ and following $\pi$ |
| $Q^\pi(s,a)$ | Expected discounted return after taking $a$ in $s$, then following $\pi$ |
| $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ | Advantage of $a$ relative to the policy's usual action at $s$ |
| $\widehat A_t$ | Sampled or learned advantage estimate |
| $\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)$ | One-step temporal-difference residual |
| $\mathcal D$ | Dataset or replay buffer of transitions/trajectories |
| $B$ | Minibatch size |
| $\alpha$ | Learning rate unless an entry explicitly uses it as entropy temperature |
| $\eta$ | Generic learning rate or Lagrange-multiplier step |
| $\lambda$ | GAE trace parameter, regularization strength, or temperature as stated locally |
| $\bar\phi$, $\theta^-$ | Frozen/target parameters copied or softly updated from online parameters |
| $\operatorname{sg}[x]$ | Stop-gradient: use the numeric value of $x$ but do not differentiate through it |
| $\mathcal H(\pi(\cdot\mid s))$ | Shannon entropy for discrete actions; differential entropy for continuous action densities |
| $D_{\mathrm{KL}}(p\|q)$ | $\mathbb E_{x\sim p}[\log p(x)-\log q(x)]$; the first argument supplies the expectation |
| $\epsilon$ | Small exploration, clipping, support, or numerical constant as stated locally |

Unless an entry says otherwise, neural objectives are estimated on minibatches and optimized with stochastic gradient descent/ascent. Terminal masking must distinguish a true terminal state from a time-limit truncation: a true terminal removes the bootstrap; a truncation normally does not.

**Conventions that affect correctness:**

- In a finite-horizon task, include remaining time in the state or use time-indexed values. A task-defined horizon is a genuine terminal; an external collection cutoff is different.
- Construct and detach the **entire** Bellman target before critic regression. During actor improvement, hold critic parameters fixed but retain derivatives through its action input for pathwise gradients. For score-function gradients, hold sampled actions and return/advantage weights fixed.
- For $J=\mathbb E[\sum_t\gamma^t r_t]$, the exact trajectory policy gradient is $\mathbb E[\sum_t\gamma^t\nabla\log\pi(a_t\mid s_t)Q^\pi(s_t,a_t)]$. Uniform transition averages in practical actor-critic/PPO recipes suppress this outer weighting; they are surrogates, not literally that initial-state gradient. Sampling from discounted occupancy accounts for the weighting instead. For $\gamma=1$ and complete finite episodes, the outer factor is one.
- A sample cutoff does not make a partial reward sum a complete Monte Carlo return. Bootstrap its missing tail when appropriate. At resets, use the final observation **before** reset, never the next episode's initial observation, for that bootstrap.
- The entries are compact course recipes. Where a lecture simplifies a named research method, the distinction from the full paper is stated locally; a named method's guarantee does not automatically cover a schematic variant.

## 1. Imitation and expressive behavior models

### 1.1 Behavioral cloning (BC)

**Covered in:** Lectures 2-4 and 21.

**Use when:** Expert demonstrations are available and direct interaction, reward design, or online learning is undesirable.

**Assumptions:** Demonstration observations contain enough information to choose the action; train and deployment state distributions are sufficiently close; the supervised policy family can represent the expert's action distribution.

**Local notation:** $\mathcal D_E=\{(o_i,a_i^E)\}$ is the expert dataset. The policy may condition on $s_i$, $o_i$, or a history $h_i=(o_{\le i},a_{<i})$.

**Algorithm:**

1. Collect expert trajectories and convert them into input-action pairs. Preserve temporal history when the current observation is not sufficient.
2. Choose a policy distribution appropriate for the action space: categorical for discrete actions, Gaussian or another density for continuous actions, or one of the expressive policies below for multimodal data.
3. Split trajectories, not randomly mixed frames, into training and validation sets when leakage across a trajectory would matter.
4. Sample a minibatch of expert pairs.
5. Compute the negative log-likelihood
   $$L_{\mathrm{BC}}(\theta)=-\frac1B\sum_{i=1}^B\log\pi_\theta(a_i^E\mid o_i).$$
   Fixed-variance Gaussian likelihood reduces, up to scale and constants, to mean-squared action error.
6. Update $\theta$ to decrease $L_{\mathrm{BC}}$.
7. Repeat Steps 4-6, select a checkpoint using held-out demonstrations, and then evaluate by rolling out the policy in the environment rather than relying only on supervised loss.

**Limitations:** Small action errors can move the learner into states absent from $\mathcal D_E$ and compound over time. BC optimizes imitation, not reward improvement; it may outperform noisy demonstrations through generalization but has no general guarantee of doing so. A unimodal likelihood can average incompatible expert modes. Hidden expert information causes causal confusion or irreducible ambiguity.

### 1.2 DAgger (dataset aggregation)

**Covered in:** Lectures 2, 3, and 21.

**Use when:** BC suffers from covariate shift and an expert can label states visited by the learner.

**Assumptions:** The expert can be queried on learner-visited observations; querying is affordable and safe; the expert action is meaningful at off-demonstration states.

**Local notation:** $\pi_E$ is the expert, $\pi_i$ is the learner at iteration $i$, and $\beta_i$ is the probability of executing the expert rather than the learner during data collection.

**Algorithm:**

1. Collect an initial expert dataset $\mathcal D_0$ and train $\pi_0$ by BC.
2. For iteration $i=0,1,\ldots$ choose a mixing probability $\beta_i$, usually decreasing toward zero.
3. Roll out a mixture policy: at each visited state, execute the expert with probability $\beta_i$ and the learner with probability $1-\beta_i$.
4. At every visited state, ask the expert which action it would take, even when the learner's action was executed.
5. Add every newly labeled pair $(o,\pi_E(o))$ to the aggregate dataset: $\mathcal D_{i+1}=\mathcal D_i\cup\mathcal D_{\mathrm{new}}$.
6. Retrain or continue training the learner on the entire aggregate dataset, not only the newest rollout.
7. Reduce $\beta_i$ according to the chosen schedule and repeat Steps 3-6.
8. Evaluate learner checkpoints without expert mixing and deploy a suitable one. The final iterate is a common choice, but the no-regret guarantee is not automatically a guarantee for that last checkpoint.

**Limitations:** It requires interactive expert labels and may visit dangerous states. An expert may be unable to recover from arbitrary learner mistakes. It does not by itself fix missing observations, causal confusion, or a policy class that cannot express multimodal actions.

### 1.3 Intervention-based DAgger

**Covered in:** Lecture 2.

**Use when:** Continuous expert labeling is too expensive but a supervisor can monitor rollouts and take control near failures.

**Assumptions:** The supervisor can detect danger early, intervene quickly, and provide recovery actions; the intervention rule has acceptable false-positive and false-negative rates.

**Algorithm:**

1. Train an initial policy with ordinary demonstrations.
2. Let the learner control the system while an expert monitors it.
3. When the expert judges the state unsafe or unrecoverable for the learner, transfer control to the expert.
4. Record pre-intervention context, the expert takeover action, and recovery segment. Use only states with actual expert labels for the BC loss; do not label earlier states with a later takeover action or silently treat the learner's failing actions as expert targets.
5. Return control only after reaching a state from which the learner can safely continue.
6. Aggregate intervention/recovery data with the original data and retrain the policy.
7. Repeat monitored deployment, intervention, aggregation, and retraining until interventions are sufficiently rare.

**Limitations:** Safety depends on timely human detection and control handoff. Collected data are biased toward the intervention policy and threshold. Rare catastrophic failures may still be missed, and frequent interventions can prevent the learner from experiencing useful recoverable errors.

### 1.4 Autoregressive discretization for continuous actions

**Covered in:** Lectures 3 and 21.

**Use when:** Continuous expert actions are multimodal and action dimension is moderate.

**Assumptions:** Each action coordinate can be discretized finely enough; an ordering of coordinates is acceptable; sequential coordinate sampling fits the latency budget.

**Local notation:** $a=(a_1,\ldots,a_d)$ and each $a_j$ is assigned to one of $K_j$ bins.

**Algorithm:**

1. Choose bounds and bins for each action coordinate.
2. Convert every continuous expert action into bin indices.
3. Factor the joint policy as
   $$\pi(a\mid o)=\prod_{j=1}^d\pi(a_j\mid o,a_1,\ldots,a_{j-1}).$$
4. For a training example, feed the observation and the ground-truth earlier action coordinates to the model (teacher forcing).
5. Sum the categorical cross-entropies for all coordinates and update the model.
6. At deployment, sample or choose the first coordinate, feed that chosen coordinate back to the model, and continue one coordinate at a time.
7. Convert each selected bin to a continuous value, optionally sampling within the bin.

**Limitations:** Results depend on coordinate order and bin resolution. Fine discretization increases output size; coarse bins reduce precision. Sequential sampling adds latency, and exposure bias arises because training conditions on expert prefixes while deployment conditions on model-generated prefixes.

### 1.5 Conditional flow-matching policy

**Covered in:** Lectures 3, 4, and 21.

**Use when:** Continuous actions are strongly multimodal and the policy can afford several internal integration steps per environment action.

**Assumptions:** Base noise and actions have compatible dimensions; demonstration actions are available; a conditional vector field can approximate the transport from noise to the behavior distribution.

**Local notation:** $x_0\sim\mathcal N(0,I)$ is base noise, $x_1=a^E$ is an expert action, $u\in[0,1]$ is flow time, $x_u=(1-u)x_0+ux_1$, and $v_\theta(o,x_u,u)$ is the learned velocity.

**Training algorithm:**

1. Sample an expert pair $(o,x_1)$ from the dataset.
2. Independently sample $x_0\sim\mathcal N(0,I)$ and $u\sim\operatorname{Uniform}[0,1]$.
3. Form the interpolation $x_u=(1-u)x_0+ux_1$.
4. Compute the target velocity $v^*=x_1-x_0$.
5. Minimize
   $$L_{\mathrm{FM}}(\theta)=\|v_\theta(o,x_u,u)-v^*\|_2^2.$$
6. Repeat for minibatches until validation sampling quality is acceptable.

**Sampling algorithm:**

1. At one environment decision, sample $x^{(0)}\sim\mathcal N(0,I)$.
2. Choose an integration count $M$ and $\Delta u=1/M$.
3. For $m=0,\ldots,M-1$, set $u_m=m/M$ and update by forward Euler:
   $$x^{(m+1)}=x^{(m)}+\Delta u\,v_\theta(o,x^{(m)},u_m).$$
4. Execute $a=x^{(M)}$ after applying required action bounds.

**Limitations:** Sampling is slower than one network pass. Too few integration steps bias the action distribution; many steps increase latency. Straight-line conditional pairings need not produce a globally straight learned field. The method imitates the dataset and does not improve actions using reward by itself.

### 1.6 Reflow distillation

**Covered in:** Lectures 4 and 21.

**Use when:** A trained flow policy is accurate but its multi-step sampler is too slow.

**Assumptions:** The original sampler is good enough to serve as a teacher; a second model can represent a straighter or direct noise-to-action mapping.

**Algorithm:**

1. Freeze the trained flow-matching teacher.
2. Sample observations $o$ from the dataset and base noises $x_0$ from the base distribution.
3. Run the accurate, many-step teacher sampler to produce endpoints $x_1$.
4. Save the dependent triples $(o,x_0,x_1)$. Do not randomly re-pair $x_0$ with unrelated endpoints.
5. Train a new conditional flow field on interpolations between each saved $x_0$ and its teacher-produced $x_1$, using target velocity $x_1-x_0$.
6. Evaluate the new model with progressively fewer integration steps.
7. If needed, repeat distillation using the newest accurate sampler as teacher.

**Limitations:** It inherits teacher errors and adds a distillation approximation. One-step sampling is reliable only if the learned transport is sufficiently straight. Generating teacher pairs can be expensive.

### 1.7 Conditional diffusion action policy

**Covered in:** Lectures 3, 12, 18, and 21.

**Use when:** An expressive multimodal continuous-action behavior model is needed and iterative denoising is acceptable.

**Assumptions:** A noise schedule is specified; the denoising network can predict injected noise or an equivalent target; actions can be normalized to a stable range.

**Local notation:** $k\in\{1,\ldots,K\}$ is diffusion time, $\beta_k$ is the forward noise variance, $\alpha_k=1-\beta_k$, $\bar\alpha_k=\prod_{j=1}^k\alpha_j$, $\bar\alpha_0=1$, and $\varepsilon\sim\mathcal N(0,I)$. Choose $\bar\alpha_K$ close to zero so the noised data match the Gaussian sampling start.

**Training algorithm:**

1. Normalize dataset actions to the model's internal range.
2. Sample $(o,a)$, diffusion step $k$, and noise $\varepsilon$.
3. Create a noisy action
   $$a_k=\sqrt{\bar\alpha_k}a+\sqrt{1-\bar\alpha_k}\,\varepsilon.$$
4. Predict the noise $\varepsilon_\theta(o,a_k,k)$.
5. Minimize $\|\varepsilon_\theta(o,a_k,k)-\varepsilon\|_2^2$ and repeat.

**Sampling algorithm:**

1. Draw $a_K\sim\mathcal N(0,I)$.
2. For $k=K,K-1,\ldots,1$, a standard DDPM noise-prediction sampler uses
   $$\mu_k=\frac1{\sqrt{\alpha_k}}\left(a_k-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\varepsilon_\theta(o,a_k,k)\right),\qquad a_{k-1}=\mu_k+\sigma_k z_k,$$
   with independent $z_k\sim\mathcal N(0,I)$ and a specified reverse variance, for example $\sigma_k^2=\beta_k(1-\bar\alpha_{k-1})/(1-\bar\alpha_k)$.
3. Add the schedule-prescribed reverse noise when $k>1$; the usual DDPM sampling convention omits fresh noise at the final step. The output is still random because of the initial noise and earlier draws. A deterministic sampler such as DDIM is deterministic conditional on its initial noise, not a deterministic policy across fresh noise draws.
4. Map $a_0$ back to environment action bounds and execute it.

**Limitations:** Sampling requires many network calls, and quality is sensitive to the schedule and normalization. Direct policy-gradient likelihoods are inconvenient; backpropagating Q through the whole denoising chain is expensive. An expressive model can still omit rare behavior modes.

The displayed reverse update is the [DDPM noise-prediction sampler](https://arxiv.org/abs/2006.11239); other diffusion samplers require their own consistent reverse rule.

### 1.8 Action chunking

**Covered in:** Lectures 3 and 4.

**Use when:** Low-level actions are highly correlated, observations arrive more slowly than controls, or predicting a short coherent plan is easier than predicting isolated actions.

**Assumptions:** A short open-loop chunk is safe enough; demonstrations contain aligned action sequences; the environment does not require immediate replanning after every primitive action.

**Algorithm:**

1. Choose a chunk length $K$.
2. Convert each demonstration index into a target block $(a_t,\ldots,a_{t+K-1})$, masking steps beyond the trajectory end.
3. Train the policy to predict a distribution over the entire chunk conditioned on the current observation/history.
4. At deployment, predict a chunk.
5. Execute either all $K$ actions open loop or only a prefix, then observe again and predict a new chunk.
6. If overlapping chunks are produced, combine the predictions with the chosen temporal-aggregation rule before execution.

**Limitations:** Long chunks cannot react quickly to disturbances and make the output distribution high-dimensional. Short chunks provide less temporal abstraction. Alignment and variable episode endings need careful masking.

### 1.9 Goal-conditioned and multitask behavioral cloning with relabeling

**Covered in:** Lectures 3, 4, 21, and 24.

**Use when:** One policy should imitate many tasks/goals, or play data can be turned into goal-reaching examples.

**Assumptions:** A task context $c$ or goal $g$ can be attached to each example; the goal is observable/representable; relabeled goals are genuinely achieved by the recorded future.

**Algorithm:**

1. Store demonstrations as trajectories with their commanded context when one exists.
2. For every training state-action pair, attach the original context and train $\pi_\theta(a\mid o,c)$ by maximum likelihood.
3. For hindsight relabeling, choose a future state from the same recorded trajectory and treat it as an achieved goal $g$ for the preceding actions.
4. Add the relabeled $(o,g,a)$ examples without changing the recorded state transitions or actions.
5. Retain suitable successful commanded-goal demonstrations alongside hindsight examples. Do not blindly clone failed attempts under their original goals: BC would reinforce those actions. Keeping failed commanded-goal transitions for an RL loss is a separate use of data.
6. Train the shared context-conditioned policy on the combined data.
7. At deployment, provide the desired context/goal and roll out the policy.

**Limitations:** Relabeling cannot invent behavior that the dataset never performed. Always replacing commanded goals creates an unrealistically easy training distribution. Conflicting task gradients can interfere, and ambiguous task labels can encourage averaging.

### 1.10 Broad pretraining followed by narrow post-training

**Covered in:** Lectures 4 and 14.

**Use when:** A large, broad, lower-quality behavior dataset and a much smaller high-quality target dataset are both available.

**Assumptions:** Features learned from broad data transfer to the target behavior; post-training data specify the desired behavior well enough; catastrophic forgetting can be controlled.

**Algorithm:**

1. Train a behavior model by likelihood maximization on the broad dataset.
2. Evaluate broad capabilities and save the pretrained checkpoint.
3. Construct a smaller curated dataset representing desired behavior or interaction style.
4. Fine-tune the pretrained policy on the curated data, usually with a smaller learning rate and controlled number of updates.
5. Mix some pretraining data or add regularization when preserving broad competence is necessary.
6. Evaluate both target behavior and retained capabilities; adjust the data mixture or stopping point accordingly.

**Limitations:** Pretraining can encode undesirable correlations, and fine-tuning can erase useful capabilities. Likelihood on broad data does not distinguish good from bad behavior. The recipe gives no guarantee that knowledge and behavior preference separate cleanly.

## 2. Policy gradients, value estimation, and actor-critic

### 2.1 Monte Carlo policy evaluation

**Covered in:** Lecture 6.

**Use when:** Complete episodes can be sampled and an unbiased, non-bootstrapped value target is preferred.

**Assumptions:** Episodes terminate or returns are otherwise bounded; sampled trajectories come from the policy being evaluated unless off-policy correction is added.

**Algorithm:**

1. Roll out the fixed policy $\pi$ to obtain complete trajectories.
2. Traverse each trajectory backward to compute every $G_t$.
3. For tabular first-visit evaluation, update a state's running mean using only its first occurrence in each episode. For every-visit evaluation, use every occurrence.
4. For function approximation, minimize
   $$L_V(\phi)=\frac1B\sum_t(V_\phi(s_t)-G_t)^2.$$
5. Repeat with new trajectories until the evaluation budget is exhausted.

**Limitations:** Returns have high variance and are unavailable until enough future reward has been observed. Long episodes require storage and delayed updates. Off-policy use needs importance correction.

### 2.2 TD(0) policy evaluation

**Covered in:** Lectures 6 and 22.

**Use when:** Online or low-latency value updates are needed and some bootstrap bias is acceptable.

**Assumptions:** Data cover the states of interest; the target policy is fixed or changes slowly; the learning-rate/function-approximation combination is stable enough.

**Algorithm:**

1. Observe $(s_t,r_t,s_{t+1},d_t)$ while following the policy.
2. Form the stopped-gradient target
   $$y_t=r_t+\gamma(1-d_t)\operatorname{sg}[V_\phi(s_{t+1})].$$
3. Compute $\delta_t=y_t-V_\phi(s_t)$.
4. Update tabular value by $V(s_t)\leftarrow V(s_t)+\alpha\delta_t$, or update neural parameters to reduce $(V_\phi(s_t)-y_t)^2$.
5. Continue on each transition.

**Limitations:** An inaccurate bootstrap makes finite-time targets biased; tabular on-policy TD can nevertheless converge to the correct value under the usual visitation and diminishing-step-size conditions. Function approximation can leave persistent error. Off-policy semi-gradient TD can diverge even with linear approximation.

### 2.3 $n$-step returns

**Covered in:** Lectures 6, 8, and 22.

**Use when:** A tunable compromise is needed between one-step TD bias and Monte Carlo variance.

**Assumptions:** Up to $n$ future rewards can be buffered; bootstrapping is valid at the cutoff; terminal masks are correct.

**Algorithm:**

1. From time $t$, collect at most $n$ consecutive rewards from one episode, stopping at termination, truncation, or the available segment boundary. Let $h\le n$ be the actual number collected.
2. Sum those $h$ observed discounted rewards; never include rewards from a reset episode.
3. Set $m=0$ if the endpoint is a true terminal, otherwise $m=1$.
4. Use the detached target
   $$G_t^{(h)}=\sum_{k=0}^{h-1}\gamma^kr_{t+k}+\gamma^h m V(s_{t+h}).$$
   For an uncorrected multi-step Q-learning target, replace $V$ by $\max_aQ$. A truncation keeps its endpoint bootstrap but shortens the reward window and exponent to $h$.
5. Update the value/Q estimate and slide the window forward.

**Limitations:** Larger $n$ typically delays updates and increases sampling variance; this is a tendency, not a universal monotonic law. Value evaluation requires continuation actions from the evaluated policy or an appropriate correction. A final max does not correct behavior-policy actions inside a replayed multi-step sequence (see §3.9).

### 2.4 REINFORCE with full-trajectory return

**Covered in:** Lectures 5 and 21.

**Use when:** A simple, unbiased on-policy policy-gradient baseline is needed and episodic interaction is available.

**Assumptions:** The stochastic policy is differentiable in $\theta$; sampled trajectories come from the current policy; return variance is tolerable.

**Algorithm:**

1. Freeze the current policy parameters for data collection.
2. Sample $N$ complete trajectories from $\pi_\theta$.
3. Compute each trajectory return $R(\tau_i)=\sum_t\gamma^tr_{i,t}$.
4. Construct the gradient estimate
   $$\widehat g=\frac1N\sum_{i=1}^N\sum_t\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t})R(\tau_i).$$
5. Equivalently minimize the pseudo-loss $-\sum_{i,t}\operatorname{sg}[R(\tau_i)]\log\pi_\theta(a_{i,t}\mid s_{i,t})$.
6. Take one policy update, discard or mark the batch stale, collect fresh on-policy data, and repeat.

**Limitations:** It assigns the same total return to every action and therefore typically has high variance. It is sample inefficient and sensitive to reward scale. After any policy update, the same uncorrected batch is no longer exactly on-policy; small changes can make reuse a tolerable approximation but do not restore exact unbiasedness.

### 2.5 Reward-to-go REINFORCE with a baseline

**Covered in:** Lectures 5, 6, 21, and 22.

**Use when:** A lower-variance on-policy policy gradient is desired without relying on bootstrapped actor-critic targets.

**Assumptions:** Same as REINFORCE; the baseline cannot depend on the sampled action unless an appropriate correction is used.

**Algorithm:**

1. Collect fresh trajectories with the current policy.
2. Compute $G_t$ separately for every time step by backward accumulation.
3. Compute a baseline $b(s_t)$, commonly a learned state value. For the clean unbiasedness argument, use a baseline fixed before sampling the evaluated actions, or an independent/cross-fitted estimate. Same-batch fitting and a self-including batch mean need the qualification below.
4. Form $\widehat A_t=G_t-b(s_t)$. Detach it from the actor computation graph.
5. Optionally normalize advantages within the batch using their sample mean and standard deviation.
6. Minimize
   $$L_\pi(\theta)=-\frac1N\sum_{i=1}^N\sum_t\gamma^t\log\pi_\theta(a_{i,t}\mid s_{i,t})\operatorname{sg}[\widehat A_{i,t}].$$
   This is the initial-state discounted objective with $N$ complete episodes. The common uniform-transition version omits the outer $\gamma^t$ as described in the global conventions.
7. If $b=V_\phi$, update $\phi$ by regression to $G_t$ using a separate value loss.
8. Discard the stale batch after the update and repeat with fresh data.

**Limitations:** Reward-to-go still has high variance. An inaccurate but conditionally action-independent baseline does not itself bias the population gradient. A baseline fitted to the very samples it weights can introduce finite-sample dependence; a self-including mean of $N$ independent full-trajectory returns scales that estimator's expectation by $1-1/N$. Advantage normalization also changes the finite-sample estimator, not just its fixed scale. Long-delayed credit remains difficult.

### 2.6 Basic batch actor-critic

**Covered in:** Lecture 6.

**Use when:** Lower variance than Monte Carlo policy gradient is worth accepting bootstrap bias.

**Assumptions:** On-policy batches can be collected; the critic can learn useful values on the visited distribution; actor and critic updates are kept sufficiently synchronized.

**Algorithm:**

1. Collect a batch of trajectories with $\pi_\theta$.
2. For each transition, build a critic target, commonly $y_t=r_t+\gamma(1-d_t)V_{\bar\phi}(s_{t+1})$.
3. Update $V_\phi$ to reduce $(V_\phi(s_t)-y_t)^2$; use target or stopped-gradient values in $y_t$.
4. Compute $\widehat A_t=r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t)$, or use an $n$-step/GAE estimate.
5. Detach $\widehat A_t$ and update the actor with $-\log\pi_\theta(a_t\mid s_t)\widehat A_t$.
6. Optionally add an entropy bonus to the actor objective.
7. Collect a new on-policy batch and repeat.

**Limitations:** Critic error biases the actor. Simultaneously moving actor and critic create a coupled, nonstationary optimization. One-step advantages can propagate delayed rewards slowly.

### 2.7 Online actor-critic and A3C

**Covered in:** Lectures 6 and 22.

**Use when:** Updates should occur during interaction; A3C is useful when multiple independent workers can decorrelate experience.

**Assumptions:** Environments can run asynchronously; stale worker gradients remain acceptable; the shared optimizer can handle concurrent updates.

**Algorithm:**

1. Initialize shared actor and critic parameters.
2. Give each worker a local parameter copy and an independent environment.
3. Each worker collects a short rollout with its local policy.
4. Bootstrap from the final state when it is nonterminal, then compute returns or advantages backward through the rollout.
5. Accumulate local actor gradients from $-\log\pi(a_t\mid s_t)\widehat A_t$ and critic gradients from value error; optionally include entropy.
6. Apply the accumulated gradients asynchronously to the shared parameters.
7. Refresh the worker's local parameters from the shared copy and continue.

**Limitations:** Gradients are computed using stale parameters and results can depend on concurrency details. Modern accelerators often favor synchronous batched collection instead. It remains on-policy or near-on-policy and can be sample inefficient.

### 2.8 Generalized advantage estimation (GAE)

**Covered in:** Lectures 6, 14, and 22.

**Use when:** An on-policy actor-critic needs a smooth bias-variance trade-off across all $n$-step cutoffs.

**Assumptions:** A value function is available; trajectories preserve temporal order; $\lambda_{\mathrm{GAE}}\in[0,1]$ is tuned.

**Algorithm:**

1. Save pre-update value predictions and compute $\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)$, using the final pre-reset observation at truncations.
2. Traverse the rollout backward with $A_{T}=0$.
3. Set
   $$\widehat A_t=\delta_t+\gamma\lambda_{\mathrm{GAE}}c_t\widehat A_{t+1},$$
   where $c_t=1$ only if the next stored step belongs to the same episode and rollout segment; otherwise $c_t=0$. At a time-limit reset, $d_t=0$ can retain the value bootstrap while $c_t=0$ stops the trace from entering a new episode.
4. Form value targets $\widehat V_t=\widehat A_t+V(s_t)$ using the pre-update value predictions.
5. Optionally normalize $\widehat A_t$ across the batch.
6. Use $\widehat A_t$ in the policy objective and regress the critic to $\widehat V_t$.

**Limitations:** Smaller $\lambda_{\mathrm{GAE}}$ generally increases reliance on critic predictions; larger values generally increase sampling variance. Neither trend is a universal monotonic law. Incorrect terminal/truncation masks contaminate all earlier advantages. GAE does not make stale data on-policy.

**Endpoints:** $\lambda=0$ gives the one-step residual. $\lambda=1$ gives Monte Carlo return minus the current-state baseline only for a complete terminal episode; a cut segment retains a bootstrapped tail. With exact values and on-policy continuation, every $\lambda$ has the correct conditional advantage expectation.

### 2.9 Generic off-policy Q actor-critic

**Covered in:** Lectures 6 and 22.

**Use when:** Replay and high sample reuse are important, especially for continuous actions where enumerating $\arg\max_aQ(s,a)$ is impossible.

**Assumptions:** The replay buffer covers relevant states/actions; the next-action sampler represents the policy being evaluated (or a slowly tracking target copy); learned Q-values are reliable near actor choices. Policy evaluation does not require that this actor already maximize Q.

**Algorithm:**

1. Interact with a behavior policy derived from the actor plus exploration and store $(s,a,r,s',d)$ in replay.
2. Sample a replay minibatch.
3. Generate a fresh next action $a'\sim\pi_{\bar\theta}(\cdot\mid s')$ from a target actor.
4. Form $y=r+\gamma(1-d)Q_{\bar\phi}(s',a')$ and update the critic to reduce $(Q_\phi(s,a)-y)^2$.
5. Sample fresh current actions $\tilde a\sim\pi_\theta(\cdot\mid s)$; do not use the recorded action for actor improvement.
6. Update the actor to increase $Q_\phi(s,\tilde a)$. For a score-function update, maximize $\log\pi_\theta(\tilde a\mid s)\operatorname{sg}[Q_\phi(s,\tilde a)]$ with the sampled action detached. For a reparameterized update, differentiate Q through the sampled action while holding critic parameters fixed.
7. Slowly update target actor and critic parameters.
8. Repeat interaction and multiple replay updates according to the chosen update-to-data ratio.

**Limitations:** It ignores or approximates replay/current state-distribution mismatch. Actor maximization exploits critic errors. Bootstrapping, off-policy data, and function approximation form the “deadly triad,” so divergence is possible.

## 3. Dynamic programming and value-based reinforcement learning

### 3.1 Iterative dynamic-programming policy evaluation

**Covered in:** Lecture 7.

**Use when:** The finite MDP transition and reward tables are known and the value of a fixed policy is needed.

**Assumptions:** State/action spaces can be enumerated; $P$ and expected one-step rewards are exact; either $\gamma<1$ or the episodic problem is proper.

**Algorithm:**

1. Fix policy $\pi$ and initialize $V_0(s)$ for every state, with terminal values set to zero.
2. For each nonterminal state, compute
   $$V_{k+1}(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\left[r(s,a,s')+\gamma V_k(s')\right].$$
3. Perform the update synchronously from the old table, or use a documented in-place ordering.
4. Compute $\Delta=\max_s|V_{k+1}(s)-V_k(s)|$.
5. Repeat Steps 2-4 until $\Delta$ is below a chosen tolerance.

**Limitations:** It requires a known model and complete sweeps over the state/action space. Memory and computation scale poorly with dimension. Approximate stopping introduces residual error.

### 3.2 Tabular policy iteration

**Covered in:** Lecture 7.

**Use when:** A known, small finite MDP must be solved exactly or used as a reference implementation.

**Assumptions:** Same as dynamic-programming evaluation; ties in greedy improvement are resolved consistently.

**Algorithm:**

1. Initialize any policy $\pi_0$.
2. **Evaluate:** run iterative policy evaluation for $\pi_k$ until the desired accuracy is reached.
3. **Improve:** for every state compute
   $$Q^{\pi_k}(s,a)=\sum_{s'}P(s'\mid s,a)[r(s,a,s')+\gamma V^{\pi_k}(s')].$$
4. Set $\pi_{k+1}(s)$ to a maximizing action, or put probability only on maximizing actions.
5. If the improved policy is unchanged at every state, stop; otherwise return to Step 2.

**Limitations:** Exact evaluation can be expensive, and enumeration is impossible in large or continuous spaces. Approximate evaluation/improvement can oscillate or lose monotonic-improvement guarantees.

### 3.3 Tabular value iteration

**Covered in:** Lectures 7 and 20.

**Use when:** A known finite MDP should be solved without fully evaluating each intermediate policy.

**Assumptions:** The Bellman optimality operator is a contraction, normally through $\gamma<1$; state/action spaces and transitions are enumerable.

**Algorithm:**

1. Initialize $V_0(s)$ for every state.
2. For each state perform the Bellman optimality backup
   $$V_{k+1}(s)=\max_a\sum_{s'}P(s'\mid s,a)[r(s,a,s')+\gamma V_k(s')].$$
3. Track $\Delta=\max_s|V_{k+1}(s)-V_k(s)|$ and repeat until below tolerance.
4. Extract a greedy policy using the same one-step model and the final value table.

**Limitations:** Requires an exact model and full sweeps. A small Bellman residual does not compensate for missing states or a wrong model. The method is impractical for high-dimensional continuous state spaces.

### 3.4 Fitted value iteration

**Covered in:** Lectures 7 and 22.

**Use when:** A model is available but the state space is too large for a value table.

**Assumptions:** States can be sampled from a useful training distribution; the value approximator can fit Bellman targets; action maximization and model expectations are tractable or sampleable.

**Algorithm:**

1. Initialize value approximator $V_{\phi_0}$.
2. At outer iteration $k$, freeze $V_{\phi_k}$.
3. Sample training states $s_i$.
4. For every candidate action, estimate the model-based backup and set
   $$y_i=\max_a\mathbb E_{s'\sim P(\cdot\mid s_i,a)}[r(s_i,a,s')+\gamma V_{\phi_k}(s')].$$
5. Fit new parameters $\phi_{k+1}$ to minimize $\sum_i(V_\phi(s_i)-y_i)^2$.
6. Freeze the fitted network, rebuild targets, and repeat outer iterations.
7. Extract a greedy policy through the model.

**Limitations:** Projection back into a restricted function class can destroy the tabular contraction and cause oscillation/divergence. Performance depends on the sampled-state distribution and model accuracy. Continuous-action maximization may be expensive.

### 3.5 Fitted Q-iteration (FQI)

**Covered in:** Lectures 7 and 20.

**Use when:** A fixed transition dataset should support off-policy value learning without a known dynamics model.

**Assumptions:** The dataset covers state-action pairs needed by the learned greedy policy; the action maximum is computable; the Q approximator can fit useful Bellman targets.

**Algorithm:**

1. Start with dataset $\mathcal D=\{(s_i,a_i,r_i,s'_i,d_i)\}$ and initialize $Q_{\phi_0}$.
2. At outer iteration $k$, freeze the current Q-function.
3. For every dataset transition compute
   $$y_i=r_i+\gamma(1-d_i)\max_{a'}Q_{\phi_k}(s'_i,a').$$
4. Fit $Q_{\phi_{k+1}}$ to the fixed supervised dataset $(s_i,a_i,y_i)$, ideally enough to make the regression error small.
5. Freeze the new Q-function, rebuild every target, and repeat.
6. Deploy $\pi(s)=\arg\max_aQ_\phi(s,a)$, optionally retaining exploration during further collection.

**Limitations:** Unsupported actions may receive arbitrary high values and be selected by the maximum. Approximation and sampling errors enter at every iteration. Neural FQI need not converge, even though tabular Bellman iteration does.

### 3.6 Watkins Q-learning

**Covered in:** Lectures 7, 8, and 22.

**Use when:** A finite/discretized action MDP can be learned online without knowing its transition model.

**Assumptions:** In the tabular convergence setting, every state-action pair is visited infinitely often, step sizes satisfy stochastic-approximation conditions, rewards are bounded, and $\gamma<1$; practical neural versions do not inherit that guarantee.

**Algorithm:**

1. Initialize $Q(s,a)$.
2. At state $s_t$, choose an exploratory action, commonly $\epsilon$-greedy with respect to $Q$.
3. Execute it and observe $(r_t,s_{t+1},d_t)$.
4. Form $y_t=r_t+\gamma(1-d_t)\max_{a'}Q(s_{t+1},a')$.
5. Update
   $$Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha_t[y_t-Q(s_t,a_t)].$$
6. Move to $s_{t+1}$ and repeat; reset the environment after true termination.
7. For deployment, use the greedy policy.

**Limitations:** Tabular storage does not scale. The max creates overestimation bias. Sparse rewards make exploration hard, and neural approximation adds instability.

### 3.7 Greedy, $\epsilon$-greedy, and Boltzmann action selection

**Covered in:** Lecture 7.

**Use when:** A value method needs a behavior policy for exploration.

**Assumptions:** Actions can be enumerated and their Q-values computed.

**Algorithm:**

1. **Greedy:** choose an action uniformly among $\arg\max_aQ(s,a)$.
2. **$\epsilon$-greedy:** with probability $\epsilon$, sample an action from a specified exploration distribution (usually uniform); otherwise take a greedy action.
3. **Boltzmann:** choose
   $$\pi(a\mid s)=\frac{\exp(Q(s,a)/\tau)}{\sum_b\exp(Q(s,b)/\tau)},$$
   computing logits with a numerically stable log-sum-exp; $\tau>0$ is temperature.
4. If using a schedule, decay $\epsilon$ or $\tau$ but retain enough exploration for the problem.

**Limitations:** Random exploration is not directed and can be ineffective for long-horizon sparse rewards. Boltzmann probabilities are sensitive to Q scale. Premature decay can permanently miss useful actions.

### 3.8 Deep Q-network (DQN)

**Covered in:** Lectures 8 and 22.

**Use when:** Observations are high-dimensional, actions are discrete and enumerable, and off-policy replay is desired.

**Assumptions:** Replay covers relevant behavior; a target network changes slowly; the network and optimizer are sufficiently stable; action count is small enough to output/evaluate every Q-value.

**Algorithm:**

1. Initialize online $Q_\phi$, target $Q_{\bar\phi}\leftarrow Q_\phi$, and replay buffer $\mathcal D$.
2. Choose an $\epsilon$-greedy action from $Q_\phi$, execute it, and store $(s,a,r,s',d)$.
3. Sample a random replay minibatch.
4. Compute fixed targets
   $$y=r+\gamma(1-d)\max_{a'}Q_{\bar\phi}(s',a').$$
5. Update $\phi$ to reduce a squared or Huber loss between $Q_\phi(s,a)$ and $y$.
6. Every chosen number of environment or gradient steps, hard-copy $\bar\phi\leftarrow\phi$; alternatively use a slow Polyak update.
7. Control data collection rate, gradient-update rate, and target-update rate separately.
8. Continue until the interaction budget is exhausted; evaluate with exploration disabled.

**Limitations:** DQN can diverge or become badly overoptimistic. Replay makes learning distribution-dependent. Prioritized replay is a separate extension with its own sampling-bias correction; ordinary DQN here uses uniform replay. DQN does not directly handle continuous actions.

### 3.9 Multi-step DQN

**Covered in:** Lecture 8.

**Use when:** Rewards should propagate through the Q-function faster than with one-step DQN.

**Assumptions:** Replay stores ordered $n$-step sequences or precomputed $n$-step transitions; the behavior/target-policy mismatch over those steps is acceptable or corrected.

**Algorithm:**

1. Collect transitions while maintaining an $n$-step queue.
2. When the queue is long enough or the episode ends, compute the discounted reward sum from its first transition.
3. Store only the discounted reward sum $R_t^{(h)}=\sum_{k=0}^{h-1}\gamma^k r_{t+k}$, endpoint $s_{t+h}$, actual length $h$, and true-terminal flag. Flush the queue at resets and never cross an episode boundary.
4. When sampling this record for training, construct a fresh detached target $y=R_t^{(h)}+\gamma^h(1-d)\max_{a'}Q_{\bar\phi}(s_{t+h},a')$.
5. Recompute the bootstrap with the current target network on each use; storing a bootstrapped total permanently would leave stale labels even after target updates.
6. Continue ordinary target-network and replay updates.

**Limitations:** Unlike the one-step target, it contains uncorrected behavior-policy continuation actions between the logged first action and the final bootstrap. This biases evaluation of the greedy continuation unless policies match or corrections are supplied. Larger $n$ typically adds variance and complicates replay. Terminal handling errors are amplified.

### 3.10 Tabular Double Q-learning

**Covered in:** Lecture 8.

**Use when:** Max-induced overestimation in tabular Q-learning is material.

**Assumptions:** Two independently updated estimates can be maintained; both receive adequate data.

**Algorithm:**

1. Initialize two tables $Q^A$ and $Q^B$.
2. Select behavior actions using a combination such as $Q^A+Q^B$.
3. After observing a transition, randomly choose which table to update.
4. If updating $Q^A$, select $a^*=\arg\max_aQ^A(s',a)$ but evaluate it with $Q^B$:
   $$y=r+\gamma(1-d)Q^B(s',a^*).$$
5. Update only $Q^A(s,a)$ toward $y$. Reverse the roles when updating $Q^B$.
6. Repeat interaction and alternating updates.

**Limitations:** It doubles storage and can introduce underestimation. The two estimates become correlated through shared data. It does not solve exploration or approximation error.

### 3.11 Double DQN

**Covered in:** Lecture 8.

**Use when:** DQN's target maximum is systematically optimistic.

**Assumptions:** Online and target networks differ enough to partially separate action selection from evaluation.

**Algorithm:**

1. Run the ordinary DQN interaction and replay loop.
2. For each next state, select
   $$a^*=\arg\max_aQ_\phi(s',a)$$
   with the online network.
3. Evaluate that selected action with the target network:
   $$y=r+\gamma(1-d)Q_{\bar\phi}(s',a^*).$$
4. Regress $Q_\phi(s,a)$ toward $y$.
5. Update the target network on the normal schedule.

**Limitations:** Online and target estimates are not independent, so bias is reduced rather than eliminated. It can underestimate and retains all other DQN failure modes.

### 3.12 Clipped double Q-learning

**Covered in:** Lectures 8, 13, and 22.

**Use when:** Continuous-control actor-critic or soft Q-learning needs a conservative target against critic overestimation.

**Assumptions:** Two critics can be trained; moderate pessimistic bias is safer than optimistic actor exploitation.

**Algorithm:**

1. Maintain critics $Q_{\phi_1}$ and $Q_{\phi_2}$ with separate parameters.
2. Sample the next action using the target actor/policy.
3. Evaluate it with both target critics and take the minimum.
4. Build the target using $\min(Q_{\bar\phi_1},Q_{\bar\phi_2})$, plus entropy terms if required by the parent algorithm.
5. Fit both critics independently to the same target using their own prediction losses.
6. Use the actor objective of the parent algorithm: standard SAC uses the minimum of online critics, while standard TD3 updates the actor through its first critic. Clipped double Q specifies the target minimum, not a universal actor loss.

**Limitations:** Taking a minimum shifts estimates downward relative to either critic and can cause underestimation. It is not guaranteed to lie below the true value: both critics may share a positive error. Critics trained on identical data can remain highly correlated. It adds compute but does not fix poor coverage.

### 3.13 Deep deterministic policy gradient (DDPG)

**Covered in:** Lectures 8 and 22.

**Use when:** Actions are continuous, replay efficiency matters, and a deterministic actor is appropriate.

**Assumptions:** Q is differentiable with respect to action; the actor can approximate a maximizing action; exploration noise provides enough coverage.

**Local notation:** $\mu_\theta(s)$ is the deterministic actor and $\rho$ is the Polyak averaging rate.

**Algorithm:**

1. Initialize actor $\mu_\theta$, critic $Q_\phi$, their target copies, and replay.
2. Execute $a_t=\mu_\theta(s_t)+\text{exploration noise}$, clip to bounds, and store the transition.
3. Sample a replay minibatch.
4. Compute $a'=\mu_{\bar\theta}(s')$ and $y=r+\gamma(1-d)Q_{\bar\phi}(s',a')$.
5. Update the critic to reduce $(Q_\phi(s,a)-y)^2$.
6. Update the actor by minimizing $-Q_\phi(s,\mu_\theta(s))$, backpropagating through the critic's action input but not updating critic parameters in this step.
7. Soft-update targets, for example $\bar\phi\leftarrow\rho\bar\phi+(1-\rho)\phi$ and similarly for the actor.
8. Repeat interaction and replay updates.

**Limitations:** DDPG is brittle and highly sensitive to Q error, reward scaling, exploration noise, and hyperparameters. Deterministic policies can explore poorly and readily exploit narrow critic artifacts.

### 3.14 Twin delayed DDPG (TD3)

**Covered in:** Lectures 6 and 22 as a practical descendant of off-policy actor-critic; implemented in the course companion.

**Use when:** DDPG is desired but critic overestimation and sharp action-value errors make it unstable.

**Assumptions:** Same as DDPG; twin critics, delayed actor updates, and target-action smoothing are affordable.

**Local notation:** $\xi$ is target-action noise, $c>0$ is its clipping magnitude, and $a_{\min},a_{\max}$ are environment action bounds.

**Algorithm:**

1. Maintain a deterministic actor, two critics, and target copies of all three.
2. Collect replay data using the actor plus exploration noise.
3. For a replay batch, compute target action
   $$a'=\operatorname{clip}(\mu_{\bar\theta}(s')+\operatorname{clip}(\xi,-c,c),a_{\min},a_{\max}),$$
   with $\xi$ drawn from small zero-mean noise.
4. Form $y=r+\gamma(1-d)\min_jQ_{\bar\phi_j}(s',a')$.
5. Update both critics toward $y$.
6. Only once every chosen number of critic updates, update the actor to maximize $Q_{\phi_1}(s,\mu_\theta(s))$.
7. On those delayed steps, also soft-update all target networks.

**Limitations:** Pessimism may become excessive, and smoothing can blur genuinely sharp optimal actions. It retains deterministic exploration problems and adds tuning parameters.

## 4. Data reuse and constrained policy optimization

### 4.1 Ordinary and self-normalized importance sampling

**Covered in:** Lectures 9, 14, and 22.

**Use when:** Estimating a target-policy expectation from data generated by another policy, or correcting a limited amount of policy mismatch.

**Assumptions:** Support overlap: if the target assigns positive probability, the behavior must also assign positive probability; behavior probabilities are known or estimable.

**Local notation:** $w_i=p(x_i)/q(x_i)$ for target density $p$ and behavior density $q$; trajectory weights are products of action-probability ratios because environment dynamics cancel.

**Algorithm:**

1. For each sample/trajectory from $q$, compute its likelihood ratio. For a trajectory,
   $$w_i=\prod_t\frac{\pi(a_{i,t}\mid s_{i,t})}{\pi_\beta(a_{i,t}\mid s_{i,t})}.$$
2. Compute ordinary IS estimate $\widehat\mu_{\mathrm{IS}}=\frac1N\sum_iw_if(x_i)$ for an unbiased estimate under exact ratios.
3. Or compute self-normalized/weighted IS
   $$\widehat\mu_{\mathrm{WIS}}=\frac{\sum_iw_if(x_i)}{\sum_iw_i},$$
   which trades finite-sample bias for often lower variance.
4. For time-local objectives, remove only ratios belonging to actions that cannot causally affect the measured future quantity.
5. Monitor effective sample size $(\sum_iw_i)^2/\sum_iw_i^2$ and ratio extremes before trusting the result.

**Limitations:** Products of ratios have variance that grows rapidly with horizon. Missing support cannot be repaired by weighting. Clipping or normalization reduces variance but introduces bias.

### 4.2 Local off-policy policy-gradient surrogate

**Covered in:** Lectures 9, 10, and 22.

**Use when:** A recently collected on-policy batch should support several small actor updates.

**Assumptions:** The new policy remains close to $\pi_{\mathrm{old}}$; old-policy advantages are accurate; ignoring the changed state distribution is acceptable locally.

**Algorithm:**

1. Collect a batch with $\pi_{\mathrm{old}}$ and freeze a copy of that policy.
2. Estimate $\widehat A_t$ under the old policy.
3. For candidate parameters $\theta$, compute the action ratio
   $$r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}.$$
4. Optimize the surrogate $\frac1B\sum_tr_t(\theta)\widehat A_t$ for a small number of steps.
5. Measure the KL divergence between old and new policies on batch states.
6. Stop/reject updates if policy change exceeds the chosen local region; then collect fresh data.

**Limitations:** The ratio corrects sampled actions but not the new policy's state-visitation distribution. Unconstrained repeated updates can exploit finite-batch errors and move arbitrarily far from the data-generating policy.

### 4.3 Clipped proximal policy optimization (PPO-Clip)

**Covered in:** Lectures 9, 10, 14, and 22.

**Use when:** A robust, relatively simple on-policy actor-critic should reuse each batch for multiple minibatch epochs.

**Assumptions:** Batch data are fresh from $\pi_{\mathrm{old}}$; policy changes remain local; advantage and value estimates are adequate.

**Local notation:** $\epsilon_{\mathrm{clip}}>0$ is the allowed ratio band half-width; the formula below writes it as $\epsilon$ for brevity.

**Algorithm:**

1. Copy $\pi_{\mathrm{old}}\leftarrow\pi_\theta$ and collect an on-policy rollout batch.
2. Compute returns/value targets and GAE advantages using the pre-update critic; normalize advantages if desired.
3. For several shuffled minibatch epochs compute $r_t(\theta)=\exp(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t))$.
4. Maximize
   $$L^{\mathrm{clip}}=\frac1B\sum_t\min\left(r_t\widehat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t\right).$$
5. Simultaneously or separately fit the value function and optionally add policy entropy; use explicit coefficients for each term.
6. Track actual KL, clip fraction, entropy, value loss, and explained variance. Stop an epoch early if KL is too large.
7. After the fixed epochs, discard the batch, set the updated policy as the new behavior policy, and collect fresh data.

**Limitations:** Clipping is not a hard trust region and does not prevent all large policy changes. Results depend strongly on batch size, epoch count, advantage scale, value loss, and optimizer settings. PPO remains on-policy and interaction hungry.

### 4.4 Adaptive KL-penalty PPO

**Covered in:** Lectures 10, 14, and 22.

**Use when:** Policy movement should be controlled by an interpretable average KL target rather than ratio clipping.

**Assumptions:** KL can be evaluated on sampled old-policy states; a penalty coefficient can be adapted fast enough to enforce the desired scale.

**Algorithm:**

1. Collect an old-policy batch and estimate advantages.
2. Initialize or retain KL penalty $\beta>0$.
3. For several actor steps, maximize
   $$\mathbb E[r_t(\theta)\widehat A_t]-\beta\,\mathbb E[D_{\mathrm{KL}}(\pi_{\mathrm{old}}(\cdot\mid s_t)\|\pi_\theta(\cdot\mid s_t))].$$
4. Measure the resulting mean KL $\widehat D$.
5. Increase $\beta$ when $\widehat D$ exceeds the target $\delta$ and decrease it when KL is too small; use a bounded multiplicative or dual update.
6. Fit the critic, finish the inner epochs, then collect a new batch.

**Limitations:** It is a soft constraint and can temporarily overshoot. The dual/penalty dynamics add tuning and can oscillate. Average KL can hide large changes at rare states.

### 4.5 Natural policy gradient (NPG)

**Covered in:** Lectures 4, 10, and 22.

**Use when:** A policy step should be scaled by change in the policy distribution rather than Euclidean parameter distance.

**Assumptions:** The local quadratic KL approximation is accurate; Fisher-vector products and a stable linear solve are available; the batch is on-policy.

**Algorithm:**

1. Collect on-policy data and estimate advantages.
2. Compute the ordinary surrogate gradient $g$ at the current parameters.
3. Define $F$ as the Hessian of conditional policy KL averaged over batch states, evaluated at the unchanged reference policy, usually accessed through products $Fv$. This equals the action-expectation Fisher under regularity. An outer-product estimate from finitely many sampled action scores is another estimator and need not equal that Hessian exactly.
4. Approximately solve $Fx=g$ with conjugate gradient, adding damping if needed.
5. Scale $x$ to meet a KL budget $\delta$, for example
   $$\Delta\theta=\sqrt{\frac{2\delta}{x^TFx}}\,x.$$
   Skip the actor step if $g=0$ or the solve/curvature is invalid. If scaling with $F+\eta I$ instead, state that damped metric explicitly and still check actual KL.
6. Update $\theta\leftarrow\theta+\Delta\theta$ and verify the measured KL.
7. Collect fresh data and repeat.

**Limitations:** Fisher estimation and conjugate gradients are costly and noisy. The method relies on local approximations and damping choices. Without line search, the realized step can violate the intended constraint.

### 4.6 Trust-region policy optimization (TRPO)

**Covered in:** Lecture 10.

**Use when:** More faithful trust-region enforcement is desired than PPO provides and extra second-order computation is acceptable.

**Assumptions:** Same local surrogate/KL assumptions as NPG; an on-policy batch is available; backtracking can evaluate surrogate improvement and KL.

**Algorithm:**

1. Collect on-policy data, estimate advantages, and compute the local surrogate gradient $g$.
2. Use Fisher-vector products and conjugate gradient to obtain the natural-gradient direction $x\approx F^{-1}g$.
3. Scale $x$ to the proposed trust-region boundary.
4. Starting with the full proposed step, evaluate candidate parameters on the same batch.
5. Accept the first candidate that both improves the surrogate by an adequate amount and satisfies mean KL $\le\delta$.
6. If either condition fails, multiply the step by a backtracking factor and try again.
7. If all candidates fail, keep the old actor parameters.
8. Update the critic separately and collect a new batch.

**Limitations:** It is computationally heavier and more complex than PPO. Guarantees use idealized exact quantities, while implementation uses finite samples, approximate solves, and average KL. Backtracking may reject all steps.

## 5. Variational inference and maximum-entropy control

### 5.1 Coordinate-ascent variational inference (CAVI)

**Covered in:** Lectures 11, 12, and 22.

**Use when:** A latent-variable posterior is intractable but a factorized variational family has tractable coordinate updates.

**Assumptions:** The approximation is $q(z)=\prod_jq_j(z_j)$; expectations under the other factors can be computed; coordinate optimization stays inside a tractable family.

**Local notation:** $x$ is observed, $z$ is latent, and the evidence lower bound is $\mathcal L(q)=\mathbb E_q[\log p(x,z)-\log q(z)]$.

**Algorithm:**

1. Choose the variational factors and initialize every $q_j(z_j)$.
2. For coordinate $j$, hold all other factors fixed.
3. Compute
   $$\log q_j^*(z_j)=\mathbb E_{q_{-j}}[\log p(x,z)]+\text{constant}.$$
4. Normalize the factor or update the parameters of its recognized distribution family.
5. Move to the next coordinate and repeat a full sweep.
6. Evaluate the ELBO after each sweep and stop when its improvement is below tolerance.

**Limitations:** Mean-field factorization discards posterior correlations and often underestimates uncertainty. Exact coordinate updates cannot decrease the ELBO, but a global optimum is not guaranteed; convergence to a stationary/coordinatewise solution requires regularity. Updates can require model-specific algebra and per-datum factors can be costly.

### 5.2 Expectation-maximization (EM)

**Covered in:** Lecture 11.

**Use when:** Maximum-likelihood parameters are needed for a latent-variable model whose posterior or a useful approximation can be computed.

**Assumptions:** The E-step posterior/variational update and M-step expected complete-data optimization are tractable enough; data are modeled by the chosen latent family.

**Algorithm:**

1. Initialize model parameters $\theta_0$.
2. **E-step:** compute $q(z)=p_{\theta_k}(z\mid x)$, or optimize a variational approximation to it.
3. **M-step:** with $q$ fixed, maximize
   $$\theta_{k+1}=\arg\max_\theta\mathbb E_q[\log p_\theta(x,z)].$$
4. Evaluate data log likelihood or the ELBO.
5. Alternate E- and M-steps until improvement is below tolerance.

**Limitations:** EM is sensitive to initialization; convergence to a global or even strict local maximum is not guaranteed. Exact EM makes likelihood nondecreasing. Generalized EM retains that property when the exact E-step is followed by an M-step that increases the expected complete-data objective; variational EM generally guarantees only ELBO improvement. Degenerate mixture likelihoods can be unbounded without regularization.

### 5.3 Variational autoencoder (VAE)

**Covered in:** Lectures 12 and 22.

**Use when:** A high-dimensional generative model and amortized latent representation are needed.

**Assumptions:** The encoder $q_\phi(z\mid x)$ is reparameterizable; the decoder likelihood and prior density are evaluable; an ELBO is an acceptable training objective.

**Algorithm:**

1. For a minibatch $x$, compute encoder mean $\mu_\phi(x)$ and log variance $\log\sigma_\phi^2(x)$.
2. Sample $\varepsilon\sim\mathcal N(0,I)$ and reparameterize $z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon$.
3. Decode $z$ to parameters of $p_\theta(x\mid z)$.
4. Compute reconstruction negative log likelihood $-\log p_\theta(x\mid z)$.
5. Compute $D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))$, usually analytically for diagonal Gaussians.
6. Minimize reconstruction loss plus the KL term. The standard negative ELBO uses $\beta=1$; using another coefficient gives a $\beta$-VAE objective, which is not in general the original negative ELBO.
7. To generate, sample $z\sim p(z)$ and sample/decode $x\sim p_\theta(x\mid z)$.

**Limitations:** The ELBO may favor blurry/averaged reconstructions, and the approximate posterior can be poor. Large KL weight can cause posterior collapse; small weight can make prior samples bad. Learned latents are not guaranteed to contain control-relevant information.

### 5.4 Conditional VAE (CVAE) policy

**Covered in:** Lecture 12.

**Use when:** Multimodal actions or outcomes must be generated conditional on an observation/context.

**Assumptions:** Training provides paired context-action data; latent sampling at test time is acceptable; encoder and decoder are expressive enough.

**Algorithm:**

1. For each pair $(o,a)$, encode $q_\phi(z\mid o,a)$.
2. Reparameterize and sample $z$.
3. Decode a conditional action distribution $p_\theta(a\mid o,z)$.
4. Minimize
   $$-\mathbb E_{q_\phi}[\log p_\theta(a\mid o,z)]+\beta D_{\mathrm{KL}}(q_\phi(z\mid o,a)\|p(z\mid o)),$$
   using a fixed or learned conditional prior. Set $\beta=1$ for the standard conditional ELBO; other weights change the training objective.
5. At deployment, sample $z$ from the prior conditioned only on $o$, decode an action, and execute/sample it.

**Limitations:** The training encoder sees the target action but deployment does not, creating reliance on prior matching. The decoder may ignore $z$. Latent prior mismatch can produce poor actions, and iterative flow/diffusion policies may fit complex modes better.

### 5.5 Sequential VAE / latent state-space model

**Covered in:** Lectures 12, 16, and 22.

**Use when:** Partially observed sequences should be compressed into latent state for prediction, planning, or actor-critic learning.

**Assumptions:** A Markovian latent state is adequate; sequential encoder, dynamics, and decoder distributions are trainable; sequence data cover relevant dynamics.

**Local notation:** $z_t$ is latent state; $p_\theta(z_{t+1}\mid z_t,a_t)$ is latent dynamics; $p_\theta(o_t\mid z_t)$ is the observation decoder; $q_\phi$ is the posterior encoder.

**Algorithm:**

1. Sample an ordered sequence $(o_{0:T},a_{0:T-1})$.
2. Run the inference model through the sequence to produce $q_\phi(z_t\mid o_{\le t},a_{<t})$ or a smoothing posterior that also uses future observations.
3. Reparameterize latent samples at each time step.
4. Predict each observation from its latent and each next latent from the current latent/action.
5. Sum reconstruction/prediction terms and KL terms between posterior latents and dynamics-prior predictions to form the sequential negative ELBO.
6. Backpropagate through the unrolled sequence and update encoder, dynamics, and decoder.
7. For control, initialize latent belief from real observations, imagine latent rollouts, and train/execute a planner or actor using the latent state.

**Limitations:** Long-sequence optimization is costly and can suffer posterior collapse or compounding model error. Latent states may omit reward-relevant detail. Posterior information available during training may be unavailable online.

### 5.6 Exact control-as-inference backward recursion

**Covered in:** Lecture 12.

**Use when:** A small finite-horizon control problem is deterministic, or when the optimistic stochastic-transition interpretation is intentionally desired.

**Assumptions:** Dynamics/rewards are known; states/actions can be enumerated; an action prior $p_0(a\mid s)$ and temperature $\alpha>0$ are specified.

**Local notation:** Introduce an optimality variable with $p(O_t=1\mid s_t,a_t)\propto\exp(r(s_t,a_t)/\alpha)$; $\beta_t$ is a backward message.

**Algorithm:**

1. Initialize the terminal backward message to one, or to the terminal optimality likelihood.
2. For $t=T-1,\ldots,0$, compute the action message
   $$\beta_t(s,a)=\exp(r(s,a)/\alpha)\sum_{s'}P(s'\mid s,a)\beta_{t+1}(s').$$
3. Sum action messages under the action prior:
   $$\beta_t(s)=\sum_a p_0(a\mid s)\beta_t(s,a).$$
4. Normalize to recover the posterior action distribution
   $$\pi(a_t\mid s_t,O_{t:T}=1)\propto p_0(a_t\mid s_t)\beta_t(s_t,a_t).$$
5. At execution, sample from this posterior and repeat the recursion when the horizon/state changes.

**Limitations:** With stochastic dynamics, conditioning on success treats lucky transitions as if the agent controlled them, producing an optimistically wrong controller. Enumeration does not scale, and the reward-to-probability mapping depends on temperature.

### 5.7 Soft value iteration

**Covered in:** Lectures 12, 13, and 22.

**Use when:** A known MDP should be solved for reward plus policy entropy rather than reward alone.

**Assumptions:** Dynamics and reward are known; actions can be summed/integrated; $\alpha>0$ and an optional action prior $p_0(a\mid s)$ are defined.

**Choose the objective first:** with a normalized prior, the formulas below optimize reward minus $\alpha D_{\mathrm{KL}}(\pi\|p_0)$. For ordinary reward-plus-Shannon-entropy control on discrete actions, omit $p_0$ from the sums, giving $V=\alpha\log\sum_a\exp(Q/\alpha)$. A uniform normalized prior adds a $-\alpha\log|\mathcal A|$ per-step constant. It preserves policy preferences only when that constant's accumulated contribution is policy-independent (for example, fixed horizon and fixed action count). In continuous actions, specify the integration measure; differential entropy depends on action coordinates.

**Algorithm:**

1. Initialize $V_0(s)$.
2. Compute the dynamics-preserving soft Q backup
   $$Q_{k+1}(s,a)=r(s,a)+\gamma\mathbb E_{s'\sim P(\cdot\mid s,a)}[V_k(s')].$$
3. Compute
   $$V_{k+1}(s)=\alpha\log\sum_a p_0(a\mid s)\exp(Q_{k+1}(s,a)/\alpha),$$
   using log-sum-exp stabilization.
4. Repeat Steps 2-3 until the value change is below tolerance.
5. Extract
   $$\pi(a\mid s)=\frac{p_0(a\mid s)\exp(Q(s,a)/\alpha)}{\sum_b p_0(b\mid s)\exp(Q(s,b)/\alpha)}.$$

**Limitations:** Requires a known model and tractable action integration. Large $\alpha$ can sacrifice reward for randomness; small $\alpha$ recovers numerically difficult near-max behavior. Entropy is not always a desirable objective.

### 5.8 Model-free soft Q-learning

**Covered in:** Lecture 13.

**Use when:** Entropy-regularized value learning is desired without a known dynamics model, especially with discrete actions.

**Assumptions:** Off-policy data cover useful state-action pairs; the soft action log-sum-exp/integral is tractable or can be approximated.

Use the same entropy/prior convention as §5.7: retaining normalized $p_0$ gives KL-to-prior regularization; omitting it gives the ordinary entropy form.

**Algorithm:**

1. Initialize online/target soft Q-functions and replay.
2. Collect and store transitions with an exploratory policy.
3. For each next state compute
   $$V_{\bar\phi}(s')=\alpha\log\sum_{a'}p_0(a'\mid s')\exp(Q_{\bar\phi}(s',a')/\alpha).$$
4. Build $y=r+\gamma(1-d)V_{\bar\phi}(s')$.
5. Fit $Q_\phi(s,a)$ to $y$ using replay.
6. Update the target Q-function slowly.
7. Act by sampling from the Boltzmann policy induced by $Q_\phi$ and the action prior.

**Limitations:** Exact log-sum-exp is difficult in continuous action spaces. Exponentials amplify Q errors and require numerical stabilization. It inherits replay coverage and bootstrapping problems.

### 5.9 Entropy-regularized policy gradient

**Covered in:** Lecture 13.

**Use when:** A stochastic policy should continue exploring or represent several good behaviors.

**Assumptions:** Policy entropy and its gradient are computable; the entropy coefficient matches the reward scale.

**Algorithm:**

1. Decide whether entropy is just a local actor bonus or part of the full objective $J_{\mathrm{ent}}=\mathbb E[\sum_t\gamma^t(r_t+\alpha\mathcal H(\pi(\cdot\mid s_t)))]$.
2. For the full objective, evaluate future entropy as well as future reward. One consistent convention is $V_{\mathrm{ent}}(s)=\mathbb E_a[Q_{\mathrm{ent}}(s,a)]+\alpha\mathcal H(\pi(\cdot\mid s))$ and $Q_{\mathrm{ent}}(s,a)=\mathbb E[r+\gamma(1-d)V_{\mathrm{ent}}(s')]$. Here Q includes future entropy but excludes entropy at the current state.
3. Estimate $\widehat A_t^{\mathrm{ent}}$ from this Q minus a state baseline, and detach it for the score term.
4. Minimize the pseudo-loss
   $$L(\theta)=-\mathbb E\!\left[\sum_t\gamma^t\left(\log\pi_\theta(a_t\mid s_t)\operatorname{sg}[\widehat A_t^{\mathrm{ent}}]+\alpha\mathcal H(\pi_\theta(\cdot\mid s_t))\right)\right],$$
   on fresh trajectories. The explicit entropy derivative handles the current distribution; the score-weighted future entropy handles actions that lead to later high-entropy states.
5. If instead $\widehat A_t$ uses reward alone, the same-looking actor loss is a common **local entropy bonus**. It omits the effect of actions on future entropy and is not the exact gradient of the full objective above. Optionally adapt/anneal $\alpha$ and recollect data.

**Limitations:** A fixed coefficient is scale sensitive. Too much entropy prevents precise control; too little gives no practical benefit. The method retains ordinary policy-gradient variance and sample inefficiency.

### 5.10 Soft actor-critic (SAC)

**Covered in:** Lectures 13 and 22.

**Use when:** Sample-efficient continuous-action learning and a stochastic entropy-regularized policy are desired.

**Assumptions:** Reparameterizable stochastic actor; replay coverage; differentiable critics; bounded/squashed actions handled with the correct log-probability Jacobian.

**Algorithm:**

1. Initialize stochastic actor $\pi_\theta$, twin critics $Q_{\phi_1},Q_{\phi_2}$, target critics, replay, and temperature $\alpha$.
2. Sample an action from the actor, execute it, and store the transition.
3. For a replay batch, reparameterize $a'\sim\pi_\theta(\cdot\mid s')$ and compute its corrected $\log\pi_\theta(a'\mid s')$.
4. Build
   $$y=r+\gamma(1-d)\left[\min_jQ_{\bar\phi_j}(s',a')-\alpha\log\pi_\theta(a'\mid s')\right].$$
5. Fit both critics to $y$.
6. Reparameterize $a\sim\pi_\theta(\cdot\mid s)$ and minimize actor loss
   $$L_\pi=\mathbb E[\alpha\log\pi_\theta(a\mid s)-\min_jQ_{\phi_j}(s,a)].$$
7. If using automatic temperature tuning, update $\log\alpha$ so policy entropy approaches a target entropy.
8. Soft-update target critics and repeat interaction/replay updates.

**Limitations:** SAC still exploits critic errors and can fail with poor replay coverage. Temperature, reward scale, action squashing, and terminal masks must be correct. It is compute-heavy and not safe for a fixed offline dataset without additional constraints.

### 5.11 Score-function and reparameterized gradient estimators

**Covered in:** Lectures 5, 6, and 12.

**Use when:** A gradient must pass through an expectation over samples from a parameterized distribution.

**Assumptions:** The score-function estimator requires an evaluable differentiable log density; the reparameterized estimator additionally requires samples expressible as a differentiable transform of parameter-independent noise and a differentiable downstream computation.

**Local notation:** $z\sim q_\phi(z)$, objective $J(\phi)=\mathbb E[f(z)]$, and $z=g_\phi(\varepsilon)$ for base noise $\varepsilon\sim p(\varepsilon)$ when reparameterization exists.

**Algorithm:**

1. **Score-function form:** sample $z$, compute detached weight $f(z)-b$, and estimate
   $$\nabla_\phi J\approx\nabla_\phi\log q_\phi(z)\,\operatorname{sg}[f(z)-b],$$
   where $b$ is an optional action/sample-independent baseline.
2. Average the score estimates over samples and update $\phi$ in the ascent direction.
3. **Reparameterized form:** sample $\varepsilon$ independently of $\phi$, compute $z=g_\phi(\varepsilon)$, evaluate $f(z)$, and backpropagate through $f(g_\phi(\varepsilon))$ into $\phi$.
4. Prefer distinct data points with one noise sample each before spending the same compute on many noise samples per point, unless variance measurements justify otherwise.

**Limitations:** Score gradients work for discrete samples but usually have high variance. Pathwise gradients are unavailable for many discrete distributions and invalid when an unknown/nondifferentiable environment blocks the path. Biased relaxations require separate justification.

These formulas assume $f$ has no direct parameter dependence beyond its sampled input. For $f_\phi(z)$, the score estimator also needs $\mathbb E[\nabla_\phi f_\phi(z)]$. Pathwise gradients are often lower variance, but there is no universal variance ordering.

## 6. Inverse RL, adversarial imitation, and preference optimization

### 6.1 Maximum-entropy inverse reinforcement learning

**Covered in:** Lectures 13, 14, and 22.

**Use when:** Demonstrations should reveal a transferable reward or intent, not merely a policy.

**Assumptions:** Expert behavior is approximately Boltzmann-rational under a reward in the chosen class; the environment model or an inner RL solver is available; feature/reward ambiguity is acceptable.

**Local notation:** $r_\psi(s,a)$ is learned reward. The simple trajectory model is $p_\psi(\tau\mid s_0)\propto p_{\mathrm{base}}(\tau\mid s_0)\exp(R_\psi(\tau))$, with a normalizer for each fixed start/context. Temperature is absorbed into reward. This globally reweighted trajectory model matches ordinary soft forward control in the deterministic-dynamics setting. With stochastic dynamics it generally reweights lucky transitions too; it must not be silently equated to a realizable policy with unchanged physics (see §§5.6–5.7).

**Algorithm:**

1. Initialize reward parameters $\psi$.
2. In the deterministic setting, solve the maximum-entropy forward-control problem under $r_\psi$ and the demonstrations' start distribution. For stochastic dynamics, use a maximum-**causal**-entropy formulation that preserves the transition law, rather than claiming the simple exponential trajectory density is the policy distribution.
3. Estimate reward-feature gradients on expert demonstrations.
4. Estimate the same gradients on trajectories from the current maximum-entropy policy.
5. Update $\psi$ to increase expert reward features and decrease model-policy reward features; for linear rewards this is expert minus learner feature occupancy.
6. Re-solve the forward-control problem under the updated reward.
7. Alternate reward and policy optimization until occupancy statistics stop changing materially.

**Limitations:** The nested inner RL solve is expensive. Many rewards explain the same behavior; shaping and scaling ambiguities remain. Misspecified rationality, dynamics, or reward features produce misleading rewards.

**Why the distinction matters:** selecting a risky action cannot make its lucky outcome more likely. A globally reward-weighted trajectory density can do exactly that statistically; a causal policy can only change actions. The feature-matching loop needs the forward model appropriate to the chosen IRL formulation. See [Levine's control-as-inference review](https://arxiv.org/abs/1805.00909).

### 6.2 Guided cost learning (GCL)

**Covered in:** Lectures 13 and 14.

**Use when:** Maximum-entropy IRL must use samples rather than enumerate the partition function.

**Assumptions:** The proposal policy's trajectory likelihood is available; it covers high-reward expert-like trajectories; importance weights can be controlled.

**Algorithm:**

1. Start with demonstrations, reward/cost model, and proposal policy.
2. Sample trajectories from the proposal and record their proposal log probabilities.
3. For proposal trajectory $\tau_j$ from a single policy in the trajectory-density setting of §6.1 (conditional on a common start/context), compute the unnormalized importance weight
   $$w_j=\frac{\exp(R_\psi(\tau_j))}{\prod_t\pi(a_{j,t}\mid s_{j,t})},$$
   because the common initial-state and dynamics factors cancel.
4. Normalize the proposal weights $\widetilde w_j=w_j/\sum_kw_k$.
5. Estimate the reward gradient as the average $\nabla_\psi R_\psi$ on demonstrations minus $\sum_j\widetilde w_j\nabla_\psi R_\psi(\tau_j)$ on proposal samples, and update $\psi$ in that direction.
6. Use the updated reward to partially improve a maximum-entropy policy rather than solving it from scratch to completion.
7. If pooling trajectories from proposal policies $\pi_k$ with proportions $\omega_k$, use the mixture denominator $\sum_k\omega_k\prod_t\pi_k(a_{j,t}\mid s_{j,t})$ for a mixture-importance estimator. Do not multiply per-step policy mixtures: that describes a different sampler. An explicitly per-proposal IS estimator is another option but needs each proposal's own support conditions.
8. Alternate importance-weighted reward fitting and entropy-regularized policy improvement.

**Limitations:** Poor proposal coverage gives high-variance or biased partition estimates. Reward and sampler co-adapt, creating instability. It requires tractable policy trajectory probabilities and remains reward-ambiguous.

### 6.3 Generative adversarial imitation learning (GAIL)

**Covered in:** Lectures 13, 14, and 22.

**Use when:** Expert occupancy should be matched without explicitly recovering a reusable reward.

**Assumptions:** Expert and learner state-action samples can be compared; the discriminator and policy optimizers remain sufficiently balanced; online policy interaction is available.

**Local notation:** Let $D_\psi(s,a)$ be the probability that a sample is expert. Other label conventions reverse the reward signs.

**Algorithm:**

1. Initialize policy and discriminator.
2. Roll out the current policy to collect learner state-action pairs.
3. Train $D_\psi$ by binary cross-entropy to distinguish expert pairs from learner pairs.
4. Assign learner transitions a discriminator-derived reward such as $-\log(1-D_\psi(s,a))$ or $\log D_\psi(s,a)$, consistently with the chosen labels.
5. Improve the policy with an on-policy RL algorithm such as PPO using that learned reward, often with entropy regularization.
6. Collect fresh learner trajectories and alternate Steps 3-5.

**Limitations:** Adversarial training can be unstable and reward scale changes as the discriminator learns. At perfect occupancy matching, the discriminator contains little reusable reward information. There is no explicit task-reward objective rewarding improvement beyond the expert; incidental improvement is possible, not guaranteed. The two example discriminator rewards are different surrogate choices, not numerically interchangeable losses.

### 6.4 Bradley-Terry preference reward model

**Covered in:** Lecture 14.

**Use when:** Humans or an automated judge can compare outputs more reliably than assign scalar rewards.

**Assumptions:** Pairwise preference probability is well modeled by a logistic function of additive scores; raters are sufficiently consistent; compared samples cover policy-relevant behavior.

**Local notation:** $R_\psi(y\mid x)$ scores response/trajectory $y$ for context $x$.

**Algorithm:**

1. Present pairs $(y^A,y^B)$ for the same context and collect which item is preferred, plus ties if supported.
2. Score both items with the reward model.
3. Predict
   $$P(A\succ B)=\sigma(R_\psi(y^A\mid x)-R_\psi(y^B\mid x)).$$
4. Minimize binary cross-entropy against the observed preference.
5. Validate on held-out raters/comparisons and calibrate or normalize scores if needed.
6. Freeze or periodically refresh the reward model before using it as an RL reward.

**Limitations:** Scores are identifiable only up to additive constants and can inherit rater bias. Pairwise consistency may fail. The optimized policy can exploit reward-model errors outside the comparison distribution.

### 6.5 RL from human feedback (RLHF) with PPO

**Covered in:** Lectures 14 and 22.

**Use when:** A pretrained sequence policy should be aligned to preferences that are difficult to express as a programmatic reward.

**Assumptions:** A useful pretrained/reference model exists; comparison labels are available; the reward model generalizes near updated policies; policy drift can be constrained.

**Algorithm:**

1. Pretrain the sequence policy by next-token likelihood and optionally supervised fine-tune it on curated responses.
2. For sampled prompts, generate multiple candidate responses.
3. Collect pairwise rankings and train a Bradley-Terry reward model.
4. Freeze a reference copy $\pi_{\mathrm{ref}}$ and initialize the trainable policy from it.
5. Sample new responses from the trainable policy and score them with the reward model; optionally use token/process rewards when available.
6. Subtract a KL penalty that discourages departure from $\pi_{\mathrm{ref}}$.
7. Estimate token-level advantages using a learned value model/GAE.
8. Update the policy with PPO and fit the value model.
9. Periodically collect fresh comparisons and retrain the reward model when policy behavior has shifted materially.

**Limitations:** The policy can reward-hack the learned model, and KL regularization only limits rather than removes this risk. Human labels are costly and biased. Long sequences make credit assignment difficult, and the multi-model pipeline is expensive.

### 6.6 Group relative policy optimization (GRPO)

**Covered in:** Lecture 14.

**Use when:** Sequence-policy optimization should avoid a learned value network and repeated sampling per prompt is affordable.

**Assumptions:** Several responses can be sampled for the same prompt; within-group reward comparison is meaningful; group size is large enough for a usable baseline.

**Algorithm:**

1. Freeze $\pi_{\mathrm{old}}$ for the iteration.
2. For each prompt $x$, sample a group of $K$ responses $y_1,\ldots,y_K$.
3. Score every response and compute a prompt-specific mean and standard deviation.
4. Set normalized group advantages, for example
   $$\widehat A_i=\frac{R_i-\overline R_x}{s_x+\epsilon}.$$
5. Compute **token-level** current/old ratios, conditioned on the prompt and preceding generated tokens. A full-response product ratio produces a different, usually much higher-variance variant.
6. Optimize the tokenwise PPO clipped surrogate, averaging within each response and then across responses in the original formulation, plus a KL penalty to a frozen reference policy. The old rollout policy is refreshed each iteration; the reference policy anchors behavior across iterations. See [DeepSeekMath's GRPO formulation](https://arxiv.org/abs/2402.03300).
7. Repeat minibatch updates within a controlled KL region, then sample new groups with the updated policy.

**Limitations:** It replaces value-model memory with generation cost. Small groups give noisy baselines; identical rewards give no learning signal. A response-level advantage may assign the same credit to every token, and reward-model exploitation remains possible.

## 7. Model-based reinforcement learning and planning

### 7.1 Iterative learned-model RL

**Covered in:** Lectures 15, 16, 22, and 25.

**Use when:** Real interaction is expensive but a learned dynamics model can support planning or synthetic policy training.

**Assumptions:** Transition data can be collected; the learned model predicts decision-relevant outcomes near policy visitation; reward is known or learned separately.

**Local notation:** $\widehat P_\psi(s'\mid s,a)$ is learned dynamics; a deterministic model $f_\psi(s,a)$ predicts $s'$ directly.

**Algorithm:**

1. Collect an initial transition dataset using a safe exploratory, demonstration, or existing policy.
2. Train deterministic dynamics by next-state regression or probabilistic dynamics by maximizing transition likelihood.
3. Validate multi-step rollouts and uncertainty on held-out transitions, not only one-step average error.
4. Use the model either to plan actions directly or to produce synthetic interaction for an RL algorithm.
5. Improve the policy only a controlled amount while the model is fixed.
6. Execute the improved policy in the real environment and add newly visited transitions to the dataset.
7. Refit the model on all relevant data and repeat model fitting, restrained improvement, and real collection.

**Limitations:** The policy actively seeks and exploits model errors. Small one-step errors compound, and average prediction quality can hide errors in decision-critical regions. Real safety is not guaranteed during corrective collection.

### 7.2 Bootstrap dynamics ensemble and disagreement

**Covered in:** Lectures 15, 16, and 19.

**Use when:** Model-based planning/training needs a practical estimate of epistemic uncertainty.

**Assumptions:** Different bootstrap samples/initializations produce meaningfully different plausible models; ensemble disagreement correlates with lack of data.

**Algorithm:**

1. Create $M$ dynamics models with independent initializations.
2. Give each model a bootstrap resample of the transition dataset, or independently resample minibatches/masks during training.
3. Fit every model by deterministic or probabilistic next-state likelihood.
4. For a candidate $(s,a)$, obtain each model's prediction.
5. Compute disagreement, such as variance of predicted next-state means or rewards.
6. Use disagreement to penalize uncertain plans, reject them, or propagate multiple model particles through planning.
7. Retrain the ensemble as real data grow.

**Limitations:** Ensembles can agree while all are wrong, especially under shared architecture/data bias. Disagreement mixes optimization noise with epistemic uncertainty. Training and planning cost scale with ensemble size.

### 7.3 Random-shooting planning

**Covered in:** Lectures 16 and 25.

**Use when:** A dynamics model exists, actions are continuous, and a simple parallel optimizer over a short horizon is sufficient.

**Assumptions:** Candidate action sequences can be sampled and simulated cheaply; the proposal distribution covers useful plans.

**Algorithm:**

1. Choose planning horizon $H$ and candidate count $N$.
2. Sample $N$ complete action sequences from a proposal distribution within action bounds.
3. Starting from current state, roll every sequence through the model.
4. Sum predicted discounted rewards and any terminal value for each sequence.
5. Select the highest-return sequence.
6. Execute the whole sequence for open-loop control, or only its first action/prefix when wrapped in MPC.

**Limitations:** The probability of sampling a narrow good plan collapses with horizon and action dimension. It wastes samples around obviously poor regions and readily exploits model error.

### 7.4 Cross-entropy method (CEM) planning

**Covered in:** Lectures 8 and 16.

**Use when:** Random shooting is too inefficient but derivative-free continuous planning through a model is still preferred.

**Assumptions:** A simple proposal, usually a factorized Gaussian over action sequences, can concentrate around a good plan; candidate rollouts are cheap.

**Algorithm:**

1. Initialize proposal mean and variance over an $H$-step action sequence.
2. Sample $N$ sequences from the proposal and clip/squash them to valid actions.
3. Simulate every sequence and compute predicted return.
4. Retain the top $K$ elite sequences.
5. Refit proposal mean/variance to the elites, optionally smoothing new parameters with old ones and enforcing a variance floor.
6. Repeat sampling, evaluation, elite selection, and refitting for a fixed number of CEM iterations.
7. Return the best sequence or proposal mean, usually executing only its first action under MPC.

**Limitations:** A factorized Gaussian poorly represents several disjoint good plans and can collapse prematurely. Computation grows with candidates, iterations, horizon, and ensemble particles. Results remain model-biased.

### 7.5 Model-predictive control (MPC)

**Covered in:** Lecture 16.

**Use when:** A planner and model are available but open-loop plans become inaccurate after disturbances or model errors.

**Assumptions:** Planning is fast enough to repeat during execution; current state/belief can be estimated after each executed prefix.

**Algorithm:**

1. Observe the current state or update the current belief state.
2. Optimize an action sequence with random shooting, CEM, gradients, or another planner.
3. Execute only the first action or short prefix.
4. Observe the actual next state rather than trusting the predicted one.
5. Shift/warm-start the previous plan if useful, then replan from the new state.
6. Repeat until termination.

**Limitations:** Replanning can be computationally expensive and may miss real-time deadlines. MPC corrects future prediction drift but cannot undo a bad executed action. Short horizons can be myopic unless a terminal value is accurate.

### 7.6 Differentiable trajectory optimization

**Covered in:** Lecture 16 and the course companion.

**Use when:** Learned/known dynamics and reward are differentiable and action-sequence gradients are informative.

**Assumptions:** Dynamics are smooth enough; gradients through $H$ model steps are numerically stable; action constraints have a differentiable parameterization or projection.

**Algorithm:**

1. Initialize unconstrained action parameters $u_{0:H-1}$.
2. Map each $u_t$ to a valid action, for example with a scaled $\tanh$.
3. Roll the state forward through the differentiable dynamics.
4. Compute predicted return plus a terminal value if used.
5. Backpropagate negative return through the entire rollout to the action parameters.
6. Take several optimizer steps, retaining the best valid sequence.
7. Execute the plan open loop or use only a prefix under MPC.

**Limitations:** Long-horizon gradients vanish/explode and local optima depend on initialization. Discontinuous contacts/rewards break smooth optimization. The optimizer can exploit differentiable model artifacts aggressively.

### 7.7 Policy learning through a learned model

**Covered in:** Lectures 15 and 16.

**Use when:** Deployment needs a fast reactive policy rather than online planning at every state.

**Assumptions:** Model rollouts near the policy are sufficiently accurate; policy updates are restrained or real data can correct errors.

**Algorithm:**

1. Fit a dynamics model from real transitions.
2. Initialize a policy and, if needed, a value/critic.
3. Generate model trajectories from real initial states or replay states.
4. Update the policy with model-free RL inside the learned simulator, or directly differentiate predicted return through dynamics.
5. Limit rollout horizon and/or the size of each policy improvement.
6. Evaluate the new policy in the real environment, add data, refit the model, and repeat.

**Limitations:** Fully optimizing in a fixed imperfect model can create extreme exploitation and oscillation. Backpropagation through long rollouts is unstable. A learned policy is faster at deployment than MPC but gives up feedback from online replanning.

### 7.8 Dyna-Q

**Covered in:** Lectures 4 and 16.

**Use when:** Real transitions are scarce but a learned tabular/local model can cheaply provide extra Q-learning updates.

**Assumptions:** Previously visited state-action dynamics can be modeled; synthetic one-step samples are useful; planning updates do not overwhelm corrective real data.

**Algorithm:**

1. Initialize Q-values, transition/reward model, and a set of observed state-action pairs.
2. Take a real exploratory action and observe $(s,a,r,s')$.
3. Apply one ordinary Q-learning update from the real transition.
4. Update the learned model for $(s,a)$ and mark the pair as observed.
5. Repeat $K$ planning updates: sample a previously observed pair, sample/predict its reward and next state from the model, and apply the same Q-learning update.
6. Continue with the next real transition.

**Limitations:** A biased model produces repeated biased Q updates. Uniformly sampling old pairs can waste planning on irrelevant regions. Tabular Dyna does not directly solve high-dimensional generalization.

### 7.9 Short-branched-rollout Dyna / MBPO-style training

**Covered in:** Lecture 16.

**Use when:** A learned model should improve the real-sample efficiency of an off-policy actor-critic while limiting compounding error.

**Assumptions:** A probabilistic ensemble is accurate for short rollouts near real replay states; the off-policy learner can mix real and synthetic data.

**Algorithm:**

1. Collect real transitions into a real replay buffer.
2. Train a probabilistic dynamics ensemble on real replay.
3. Sample starting states from the real buffer rather than from arbitrary model states.
4. From each start, roll out the current policy in a sampled ensemble model for a short horizon $h$.
5. Store predicted transitions in a separate model buffer.
6. Train an off-policy algorithm such as SAC on minibatches with a specified real/model ratio.
7. Periodically collect new real data, refit the ensemble, clear/refresh stale model data, and adjust $h$ conservatively.

**Limitations:** Longer rollouts add data but magnify model bias. Synthetic data can dominate learning and exploit uncertainty. Model-buffer freshness, mixture ratio, and rollout horizon are sensitive hyperparameters.

### 7.10 Model value expansion (MVE)

**Covered in:** Lecture 16 as part of the Dyna design space; implemented in the course companion.

**Use when:** A learned model can improve critic targets over a short horizon without replacing model-free learning.

**Assumptions:** Short model rollouts are accurate enough; a terminal value function is available; true termination is modeled/masked correctly.

**Algorithm:**

1. To update $Q(s,a)$, sample a real transition $(s,a,r,s',d)$ from replay. Its reward and first action must stay attached to that pair.
2. From $\widehat s_0=s'$, roll the target/current policy through the learned model for at most $H$ additional steps. Let $h$ be the actual length before a predicted terminal.
3. Form the continuation estimate
   $$\widehat G(s')=\sum_{k=0}^{h-1}\gamma^k\widehat r_k+\gamma^h m_{\mathrm{end}}\widehat V(\widehat s_h),$$
   where $m_{\mathrm{end}}=0$ at predicted true termination and 1 otherwise.
4. Regress $Q(s,a)$ to the detached target $y=r+\gamma(1-d)\widehat G(s')$. If $d=1$, use $y=r$ and skip the model rollout.
5. For a state-value update starting directly at $s$, a fully modeled rollout from $s$ instead targets $V(s)$. Do not attach a rollout starting at $s'$ to $Q(s,a)$ without the real reward and outer discount.
6. Keep $H$ short or mix horizons when model error is significant. With $H=0$, the Q construction reduces to ordinary one-step bootstrapping. This is the target-construction idea in [model-based value expansion](https://arxiv.org/abs/1803.00101), not its entire training recipe.

**Limitations:** Model bias and value bias trade off as $H$ changes. Incorrect terminal predictions corrupt all later terms. MVE needs a reliable policy-conditioned rollout distribution.

### 7.11 MOPO-style model-based offline RL

**Covered in:** Lectures 18 and 19.

**Use when:** A fixed offline dataset should be augmented with model rollouts while discouraging policy exploitation of uncertain dynamics.

**Assumptions:** Ensemble disagreement is a useful uncertainty proxy; a penalty coefficient can make uncertain transitions unattractive; short model rollouts stay near data.

**Algorithm:**

1. Fit a probabilistic dynamics ensemble on the fixed offline dataset.
2. Estimate uncertainty $u(s,a)$ from ensemble disagreement.
3. Start short policy rollouts from real dataset states.
4. For each model transition, replace predicted reward by
   $$\widetilde r(s,a)=\widehat r(s,a)-\lambda u(s,a).$$
5. Store penalized model transitions in a model buffer.
6. Train an off-policy policy/critic on a chosen mixture of real and penalized synthetic transitions.
7. Repeat model rollouts and policy updates without collecting new real data.

**Limitations:** Disagreement can miss shared model bias; excessive penalty prevents improvement and weak penalty permits exploitation. The method cannot correct model errors with online data, and rollout/mixing choices are sensitive.

### 7.12 COMBO-style conservative model-based offline RL

**Covered in:** Lectures 18 and 19.

**Use when:** Model-generated data are useful offline but unsupported model-region values should be pushed down directly in the critic.

**Assumptions:** Short model rollouts provide locally useful samples; conservative critic regularization distinguishes real support from model/policy samples.

**Algorithm:**

1. Fit a dynamics model/ensemble to the offline data.
2. Generate short policy rollouts from real dataset states and keep real and model transitions distinguishable.
3. Build Bellman targets from a mixture of real and model samples.
4. Add a conservative critic term of the schematic form
   $$L_{\mathrm{cons}}=\mathbb E_{s\sim\rho_{\mathrm{model}},a\sim\pi}[Q(s,a)]-\mathbb E_{(s,a)\sim\mathcal D}[Q(s,a)],$$
   or the corresponding sampled/log-sum-exp COMBO form. Minimizing it lowers policy actions in model regions relative to real dataset actions.
5. Update the critic on Bellman error plus this conservative term.
6. Update the actor against the conservative critic.
7. Refresh synthetic rollouts as the policy changes.

**Limitations:** Conservative penalties can under-value valid improvements. Model errors and critic conservatism interact in hard-to-tune ways. The compact course recipe omits many benchmark-specific details of full COMBO.

### 7.13 Domain randomization

**Covered in:** Lecture 25.

**Use when:** A policy is trained in simulation but must transfer across unknown real dynamics.

**Assumptions:** The simulator exposes relevant parameters; the randomization distribution contains plausible real systems; one policy can solve the sampled family.

**Local notation:** $\xi$ contains masses, friction, delays, sensors, textures, or other simulator parameters.

**Algorithm:**

1. Choose a distribution $p(\xi)$ over plausible simulator parameters.
2. At every episode or scheduled interval, sample $\xi\sim p(\xi)$ and configure the simulator.
3. Train one policy across all sampled simulators with the selected RL algorithm.
4. Evaluate on held-out parameter combinations and adversarial boundary cases.
5. If real calibration data exist, update the range/distribution without collapsing it to a single fitted simulator.
6. Transfer the robust policy to the real system and monitor failures.

**Limitations:** Robustness covers only variations represented by $p(\xi)$. Overly broad ranges make learning unnecessarily hard; narrow ranges miss real conditions. The policy can still exploit nonrandomized simulator defects.

## 8. Offline and offline-to-online reinforcement learning

All algorithms in this section treat $\mathcal D$ as fixed during offline training. No method is allowed to query the environment until an entry explicitly begins an online phase.

### 8.1 Actor-critic with behavioral cloning (SAC+BC style)

**Covered in:** Lecture 18.

**Use when:** A simple offline baseline is needed and a standard actor-critic implementation is available.

**Assumptions:** Dataset actions have useful quality; a BC coefficient can keep the actor near support; critic targets are sufficiently stable.

**Algorithm:**

1. Load the immutable dataset and initialize actor/critic/targets.
2. Train the critic with its standard offline Bellman target.
3. At dataset states, sample actor actions and compute their Q-values.
4. At the same states, evaluate log likelihood of the recorded dataset actions.
5. Update the actor to maximize
   $$\mathbb E_{s\sim\mathcal D,a\sim\pi}[Q(s,a)]+\lambda\mathbb E_{(s,a_D)\sim\mathcal D}[\log\pi(a_D\mid s)],$$
   plus SAC entropy if used.
6. Tune $\lambda$ using offline diagnostics and permitted held-out/evaluation protocol; never silently use test-environment interaction as training data.

**Limitations:** Weak BC permits OOD actions; strong BC reduces the method to imitation. The critic can still extrapolate incorrectly. The best coefficient depends on dataset quality and reward scale.

### 8.2 Behavior-regularized actor-critic (BRAC)

**Covered in:** Lectures 17 and 18.

**Use when:** Offline policy improvement should be explicitly penalized for leaving the learned behavior distribution.

**Assumptions:** A behavior model $\pi_\beta(a\mid s)$ can be estimated; the chosen forward/reverse KL reflects the desired coverage/mode-seeking trade-off.

**Algorithm:**

1. Fit $\pi_\beta$ by maximum likelihood on dataset actions.
2. Initialize actor, critic, and targets.
3. Define a divergence penalty $D(\pi_\theta(\cdot\mid s),\pi_\beta(\cdot\mid s))$.
4. Include the penalty in the actor objective, in the reward/critic target, or in both, using one documented formulation consistently.
5. Update the critic with regularized Bellman targets built only from the offline data.
6. Update the actor to maximize Q minus the behavior divergence.
7. Monitor both Q-values and divergence from behavior.

**Limitations:** The behavior model can assign misleading density in continuous high dimensions. KL penalizes even safe changes inside support and does not equal a true support constraint. Reverse and forward KL have different mode-dropping/averaging failures.

### 8.3 Advantage-weighted regression (AWR) and AWAC

**Covered in:** Lecture 18.

**Use when:** Stable supervised-style offline policy extraction is preferred over direct actor maximization.

**Assumptions:** Dataset advantages can be estimated; useful actions appear in data; weighted maximum likelihood can represent the improved distribution.

**Algorithm:**

1. Learn $Q(s,a)$ from offline transitions. In AWR, returns/value regression may replace a full Q critic; in AWAC, use actor-critic targets.
2. Estimate $V(s)$ and compute $A(s,a_D)=Q(s,a_D)-V(s)$ for dataset actions.
3. Compute weights $w=\exp(A/\lambda)$, usually normalized or clipped for stability.
4. Update the actor by minimizing
   $$L_\pi=-\mathbb E_{(s,a_D)\sim\mathcal D}[w(s,a_D)\log\pi_\theta(a_D\mid s)].$$
5. Alternate critic/value updates and weighted actor regression.
6. For online AWAC, append new transitions and continue the same update with a controlled old/new data mixture.

**Limitations:** Very small temperature collapses effective sample size; large temperature approaches BC. Bad actions receive near-zero rather than explicit negative pressure. Improvement is limited by actions in the dataset and by projection error.

### 8.4 Implicit Q-learning (IQL)

**Covered in:** Lecture 18.

**Use when:** Offline critic learning should avoid evaluating actor-generated OOD actions.

**Assumptions:** High-value actions exist in the dataset; an upper expectile approximates the best supported action values; weighted BC is adequate for extraction.

**Algorithm:**

1. Initialize $Q_\phi$, target $Q_{\bar\phi}$, value $V_\psi$, and actor.
2. For dataset $(s,a)$, compute $u=Q_{\bar\phi}(s,a)-V_\psi(s)$.
3. Fit the value with asymmetric squared loss
   $$L_V=|\tau-\mathbf1\{u<0\}|u^2,$$
   using upper expectile $\tau>0.5$.
4. Fit Q to $y=r+\gamma(1-d)V_\psi(s')$. This target does not query a new action.
5. Soft-update the target Q.
6. Compute dataset advantage $A=Q(s,a)-V(s)$ and weight $w=\exp(A/\lambda)$ with clipping as needed.
7. Train the actor by weighted behavioral cloning.

**Limitations:** If good actions are absent, IQL cannot invent them. Expectile and temperature are sensitive. Actor extraction can underuse the critic, and transition-distribution shift remains.

### 8.5 Conservative Q-learning (CQL)

**Covered in:** Lectures 18 and 22.

**Use when:** Offline Q-values for unsupported actions must be pushed below supported data-action values.

**Assumptions:** The conservative action sampler/log-sum-exp covers actions an optimizer might exploit; the penalty coefficient is strong enough but not destructive.

**Algorithm:**

1. Initialize critic, target critic, and an actor or discrete greedy policy.
2. Compute the ordinary offline Bellman error.
3. At each dataset state, evaluate Q on broad candidate actions: all discrete actions, or samples from uniform/current-policy/proposal distributions.
4. Compute a conservative term such as
   $$\log\sum_a\exp Q(s,a)-Q(s,a_D),$$
   using importance corrections for nonuniform continuous-action proposals.
5. Update the critic on Bellman error plus $\alpha_{\mathrm{CQL}}$ times the conservative term.
6. Update the actor using the conservative critic or take a discrete greedy policy.
7. Optionally adapt the conservative coefficient to a target gap.

**Limitations:** Excessive conservatism undervalues good actions and can prevent improvement. Continuous-action log-sum-exp is sampling dependent. The critic may be conservative in the wrong places if proposals miss problematic actions.

### 8.6 Two-buffer offline-to-online SAC baseline

**Covered in:** Lecture 18.

**Use when:** Prior offline data should help online learning without a brittle separate offline-pretraining handoff.

**Assumptions:** Online interaction is now allowed; the offline dataset contains some useful coverage/successes; SAC can learn from a mixture of old and new data.

**Algorithm:**

1. Keep the fixed prior dataset in one buffer and create an initially empty online replay buffer.
2. Initialize SAC, optionally from scratch as in the lecture baseline.
3. Interact online and add each new transition only to the online buffer.
4. Build each training minibatch from a fixed mixture, illustrated in the lecture as half offline and half online.
5. Perform ordinary SAC critic, actor, temperature, and target updates on the mixed batch.
6. Continue adding online data; optionally change the mixture only by an explicit schedule.

**Limitations:** The fixed ratio is not universal and stale offline data can slow adaptation. Starting from scratch may waste useful offline pretraining. The method does not by itself solve safety on early online steps.

### 8.7 IDQL: expressive behavior model and candidate selection

**Covered in:** Lecture 18.

**Use when:** Offline actions are multimodal and direct optimization of Q outside behavior support is unsafe.

**Assumptions:** An IQL-style critic ranks supported actions reliably; a diffusion/flow/CVAE behavior model generates in-distribution candidates; inference can afford $N$ samples.

**Algorithm:**

1. Train an IQL critic/value pair on the offline dataset.
2. Independently train an expressive behavior model $\widehat\pi_\beta(a\mid s)$ by imitation.
3. At a decision state, sample $N$ candidate actions from the behavior model.
4. Evaluate all candidates with the learned Q-function.
5. In the lecture's best-of-$N$ simplification, execute the highest-Q candidate. In the general IDQL extraction, resample candidates with normalized critic-derived weights for the policy implicitly defined by the critic loss. Arbitrary softmax-Q weights are not automatically those weights.
6. Repeat candidate generation and selection at every state.

**Limitations:** Inference cost is linear in $N$, and random candidates scale poorly in high dimensions. The maximum amplifies critic ranking error. A misspecified behavior model can still generate OOD actions or omit rare good modes.

The distinction between the lecture's hard-ranking recipe and the implicit-policy weighting is explicit in the [IDQL paper](https://arxiv.org/abs/2304.10573).

### 8.8 FQL: flow distillation with Q improvement

**Covered in:** Lecture 18.

**Use when:** A multimodal flow behavior model is useful offline but deployment needs a cheap reparameterizable actor.

**Assumptions:** Shared latent noise establishes meaningful teacher-student pairing; the simple actor can approximate/distill useful flow modes; the offline critic is reliable locally.

**Algorithm:**

1. Train a conditional flow behavior model on dataset actions.
2. Train Q on dataset transitions with a detached actor-critic target $y=r+\gamma(1-d)Q_{\bar\phi}(s',\mu_\theta(s',z'))$, with fresh $z'$ and a documented twin-critic aggregation if used. FQL's behavior restriction comes from actor distillation; a separate IQL/CQL critic is not required.
3. For dataset state $s$, sample base noise $z$.
4. Run the flow sampler with $(s,z)$ to obtain teacher action $a_F$ and detach that output for the actor loss.
5. Feed the same $(s,z)$ to a one-step actor to obtain $a_\theta$.
6. Update the actor with a combined loss such as
   $$L_\pi=\lambda_{\mathrm{distill}}\|a_\theta-a_F\|_2^2-Q(s,a_\theta),$$
   so it imitates the paired flow action while shifting toward larger Q.
7. Alternate flow-matching BC, critic fitting, actor fitting, and target-Q updates. The flow teacher learns from BC, not the actor's Q gradient; it need not be frozen for the entire run. Deploy the one-step actor with fresh latent noise.

**Limitations:** Finite actor capacity or poor distillation can lose modes. The noise-conditioned one-step actor is nevertheless expressive and need not be Gaussian or unimodal. Too much Q pressure leaves support; too much distillation gives little improvement. See the [FQL paper](https://proceedings.mlr.press/v267/park25f.html) for the complete method.

### 8.9 Diffusion/flow latent steering

**Covered in:** Lecture 18.

**Use when:** A pretrained behavior generator should supply the action space for a steering policy. Specify whether steering uses new online interaction or an explicit offline construction.

**Assumptions:** Typical latent codes decode to behavior-like actions and the actor can be constrained near that region. Online latent actor-critic needs a differentiable latent critic, not necessarily a differentiable generator; differentiating an action-space critic through $G$ does require a differentiable generator.

**Algorithm:**

1. Train an expressive behavior generator $a=G(s,w)$ using BC, with simple prior $w\sim p(w)$.
2. Freeze $G$.
3. For an **online steering phase**, choose $w$, execute $G(s,w)$, and store $(s,w,r,s',d)$ for an off-policy learner in latent action space. This is an explicit transition from offline pretraining to online interaction.
4. A fixed offline tuple $(s,a,r,s')$ does not contain the latent that generated $a$. Do not attach an arbitrary $w$ to that transition. Fully offline steering requires a specified valid action-to-latent inference procedure, or an action-space critic trained on recorded $a$ and queried at $G(s,w)$.
5. Update the latent actor using its latent critic in the online construction, or through $Q(s,G(s,w))$ in the action-critic variant. The latter retains offline extrapolation concerns and may require differentiating the generator.
6. Penalize or bound latent codes so they remain in the prior region used to train $G$.
7. Deploy by selecting $w$, decoding $a$, and executing $a$.

**Limitations:** A latent actor can discover adversarial or low-density codes that decode badly. Latent geometry need not align with action value. Training details depend strongly on whether the generator is a diffusion model, flow, or VAE.

## 9. Exploration and unsupervised skill discovery

### 9.1 UCB1 for stochastic multi-armed bandits

**Covered in:** Lecture 19.

**Use when:** Repeated choices have stationary unknown reward means and uncertainty should guide exploration.

**Assumptions:** Rewards are bounded/sub-Gaussian; arm reward distributions are stationary and independent across pulls; there is no state transition.

**Local notation:** $N_t(a)$ is the number of pulls of arm $a$, $\widehat\mu_t(a)$ its sample mean, and $t$ total pulls.

Here $c>0$ controls optimism; larger $c$ explores more aggressively.

**Algorithm:**

1. Pull every arm once and record its reward.
2. At each later time compute
   $$\operatorname{UCB}_t(a)=\widehat\mu_t(a)+c\sqrt{\frac{\log t}{N_t(a)}}.$$
3. Select the arm with the largest upper-confidence value.
4. Observe reward and incrementally update that arm's count and empirical mean.
5. Repeat for the interaction budget.

**Limitations:** It assumes stationarity and does not directly solve long-horizon credit assignment. The confidence bonus is sensitive to reward scaling and constant $c$. Counts do not generalize across similar states/actions.

### 9.2 Tabular count-bonus exploration

**Covered in:** Lecture 19.

**Use when:** States or state-action pairs can be counted and sparse extrinsic reward requires directed novelty.

**Assumptions:** A meaningful discrete identity/count exists; visiting low-count states is useful; intrinsic and extrinsic reward scales can be balanced.

**Local notation:** $\beta>0$ scales intrinsic reward and $c>0$ prevents division by zero.

**Algorithm:**

1. Initialize visit counts $N(s)$ or $N(s,a)$.
2. On each visit, compute intrinsic reward such as
   $$r_t^{\mathrm{int}}=\frac{\beta}{\sqrt{N(s_t)+c}}.$$
3. Train the chosen RL algorithm using $r_t^{\mathrm{total}}=r_t^{\mathrm{ext}}+r_t^{\mathrm{int}}$.
4. Increment the relevant count after defining consistently whether the current visit is counted before or after the bonus.
5. Continue until bonuses decay in familiar regions; evaluate using extrinsic reward alone.

**Limitations:** Exact counts fail in large/continuous observation spaces and treat visually different but equivalent states as novel. Stochastic observations can create endless novelty. Intrinsic reward may distract from the task.

### 9.3 Density-model pseudo-count exploration

**Covered in:** Lecture 19.

**Use when:** Exact counts are impossible but an online density model can generalize visitation across complex observations.

**Assumptions:** Updating on an observation increases its assigned probability; density learning progress tracks useful novelty; probabilities before and after one update are available.

**Local notation:** $p_n$ and $p_{n+1}$ are the probability model immediately before and after learning from $x$; $\beta>0$ scales the bonus and $c>0$ is a numerical stabilizer. The formula uses probability masses in $[0,1]$ for discrete/quantized observations, not arbitrary continuous densities that can exceed one.

**Algorithm:**

1. Fit/update an online density model on observations seen so far.
2. Before training on the new observation $x$, record $p_n(x)$.
3. Update the density model once on $x$ and record $p_{n+1}(x)$.
4. Compute
   $$\widehat N(x)=\frac{p_n(x)(1-p_{n+1}(x))}{p_{n+1}(x)-p_n(x)}.$$
5. Clamp/handle invalid numerical values explicitly; the formula requires $p_{n+1}(x)>p_n(x)$.
6. Give the RL agent intrinsic reward $\beta/\sqrt{\widehat N(x)+c}$ and combine it with extrinsic reward.
7. Continue updating both density model and RL learner online.

**Limitations:** Density and useful novelty are not the same. Generalization can make unseen states look familiar, while model underfitting can make common states look novel. A probability-decreasing update yields invalid counts.

### 9.4 Context-tree switching (CTS) density estimation

**Covered in:** Lecture 19 at a high level.

**Use when:** A sequential binary/image density is needed for pseudo-counts and local contexts are informative.

**Assumptions:** Observations can be serialized into symbols/bits; a bounded set of context depths is adequate; online probability assignment is more important than high-quality generation.

**Course-level procedure:**

1. Serialize each observation in a fixed order so previously encoded symbols form context.
2. At several context depths, maintain online symbol predictors such as Krichevsky-Trofimov estimators.
3. Before seeing the next symbol, mix predictions from the candidate context depths using current switching weights.
4. Multiply/accumulate symbol probabilities to obtain the observation density.
5. Observe the symbol, update each compatible context predictor, and update mixture weights so well-predicting depths gain mass while a small fixed-share probability permits switching.
6. Use the density before/after an observation in the pseudo-count formula.

**Limitations:** The lecture treats the original image-specific CTS design as antiquated and does not prescribe all engineering details. Serialization/context choices are domain dependent, and a good compressor is not necessarily a good exploration representation.

### 9.5 Random network distillation (RND)

**Covered in:** Lecture 19.

**Use when:** A scalable prediction-error novelty bonus is needed without explicit density estimation.

**Assumptions:** A fixed random target creates learnable but initially unpredictable features; predictor error falls mainly on visited states; observations are normalized appropriately.

**Algorithm:**

1. Initialize target network $f_{\mathrm{tgt}}$ randomly and freeze it permanently.
2. Initialize trainable predictor $f_\psi$ with the same output shape.
3. For each observed next state, compute
   $$r^{\mathrm{int}}=\|f_\psi(s')-f_{\mathrm{tgt}}(s')\|_2^2.$$
4. Normalize/clamp intrinsic rewards with running statistics and add them to extrinsic reward using coefficient $\beta$.
5. Train the RL algorithm on the combined reward.
6. Train only the predictor to minimize the same squared error on visited observations.
7. Evaluate the final policy using the task reward separately from novelty reward.

**Limitations:** Prediction error reflects capacity and optimization, not literal visitation. Stochastic/noisy observations can remain permanently novel. Generalization can remove bonuses from unseen states, and forgetting can make old states novel again.

### 9.6 DIAYN-style skill discovery

**Covered in:** Lecture 23.

**Use when:** A repertoire of diverse temporally extended skills should be learned without an external task reward.

**Assumptions:** Skill-conditioned states can become distinguishable; the chosen state representation captures useful differences; an RL learner can optimize the nonstationary intrinsic reward.

**Local notation:** $z$ is a categorical skill sampled from prior $p(z)$, $\pi(a\mid s,z)$ is the skill policy, and $q_\phi(z\mid s)$ is a classifier.

**Algorithm:**

1. Sample one skill $z$, usually uniformly, at the start of an episode or segment.
2. Hold $z$ fixed and roll out $\pi_\theta(a\mid s,z)$.
3. Store visited $(s,z)$ pairs.
4. Train $q_\phi(z\mid s)$ by multiclass cross-entropy.
5. Give the policy intrinsic reward $r(s,z)=\log q_\phi(z\mid s)-\log p(z)$; the prior term is constant for uniform skills.
6. Update the skill-conditioned policy with an RL algorithm using this reward, commonly including entropy regularization.
7. Alternate rollout, classifier, and policy updates.

**Limitations:** Perfect distinguishability can occur between nearby, uninteresting behaviors, so mutual information need not produce geometric coverage. The discriminator reward is nonstationary. Skills may exploit irrelevant state coordinates or learn destructive behavior.

### 9.7 Goal-conditioned empowerment / unsupervised goal reaching

**Covered in:** Lecture 23.

**Use when:** Skills should be grounded as reachable goal states rather than arbitrary latent IDs.

**Assumptions:** Valid goals can be sampled from a learned/data distribution; the policy and reward/classifier can measure progress or success; off-policy goal learning is available for sparse rewards.

**Algorithm:**

1. Maintain a goal proposal distribution $p(g)$ over states believed reachable.
2. Sample a goal and condition the policy as $\pi(a\mid s,g)$.
3. Roll out the policy while holding the goal fixed.
4. Assign either classifier reward $\log q(g\mid s)$ or a stationary tolerance reward $\mathbf1\{\|h(s)-g\|\le\epsilon\}$.
5. Update the goal-conditioned policy with RL; use hindsight relabeling when rewards are sparse.
6. Add achieved states to the valid-goal data and improve the proposal distribution.
7. Repeat with attention to broad goal coverage.

**Limitations:** Random points in observation space are usually invalid goals. A learned classifier can reward distinguishability without reaching the intended geometry. Equality rewards are sparse, and the goal representation/distance can dominate results.

### 9.8 Skew-Fit

**Covered in:** Lecture 23.

**Use when:** Unsupervised goal-conditioned learning should flatten coverage over reachable states instead of repeatedly sampling already common goals.

**Assumptions:** A generative goal model can estimate/sample reached-state density; inverse-density weighting is numerically stable; an off-policy goal learner can reuse collected transitions.

**Algorithm:**

1. Initialize a goal model $p_\phi(g)$ from available visited states.
2. Sample goals from the model and attempt them with the goal-conditioned policy.
3. Store newly visited/achieved states and transitions.
4. Update the goal-conditioned policy with an off-policy algorithm and goal reward, including relabeling as needed.
5. Estimate each visited goal's current density $p_\psi(g)$.
6. Fit the next goal model by weighted maximum likelihood with
   $$w(g)=p_\psi(g)^{\alpha_{\mathrm{skew}}},\qquad \alpha_{\mathrm{skew}}<0.$$
7. Resample from the reweighted model and repeat.

**Limitations:** Reweighting emphasizes already reached data; discovering new regions still requires exploration and useful generator/policy generalization. Generated goals can extend beyond observed support but may be unreachable. Density errors produce extreme weights; strong negative exponents are unstable. Uniform coverage may not match downstream usefulness.

### 9.9 Go-Explore frontier recipe

**Covered in:** Lecture 23 at a conceptual level.

**Use when:** Exploration requires returning reliably to promising discovered states before expanding outward.

**Assumptions:** Visited states can be grouped into cells/representations; the agent can return to a selected cell through reset, stored trajectory, or goal-conditioned control.

**Course-level procedure:**

1. Maintain an archive of discovered cells and metadata such as visitation, return, or novelty.
2. Select a promising frontier cell from the archive.
3. Return to that cell with the available mechanism.
4. Explore outward from the cell and add newly reached cells or better trajectories to the archive.
5. Repeat frontier selection, return, and outward exploration.
6. In full systems, robustify the resulting behavior so it survives stochasticity; this stage was not specified in detail in the lecture.

**Limitations:** The course presents the frontier principle, not a complete implementation. Results depend on cell representation and reliable return. Deterministic reset assumptions can fail in real/stochastic environments.

## 10. Goal-conditioned, transfer, and hierarchical reinforcement learning

### 10.1 Generic goal-conditioned RL / universal value function

**Covered in:** Lectures 23 and 24.

**Use when:** One policy should solve many goals with shared dynamics.

**Assumptions:** Goal $g$ can be included in the state/context; reward $r_g$ can be recomputed for stored transitions; training goals cover or generalize to deployment goals.

**Algorithm:**

1. Define goal representation $g$ and goal-conditioned reward $r_g(s,a,s')$.
2. Condition policy and value functions on goal: $\pi(a\mid s,g)$ and $Q(s,a,g)$.
3. Sample a goal from the training distribution at episode start.
4. Roll out the goal-conditioned behavior and store the commanded goal with every transition.
5. Sample replay transitions and update the off-policy critic using rewards recomputed for their associated goals.
6. Update the actor/greedy policy using the same goal-conditioned critic.
7. Mix commanded and relabeled goals as appropriate, and evaluate on the deployment goal distribution.

**Limitations:** Sparse goal rewards require exploration/relabeling. Goal distributions create task imbalance and gradient interference. Generalization depends heavily on goal representation and dynamics coverage.

### 10.2 Hindsight experience replay (HER)

**Covered in:** Lectures 3 and 24.

**Use when:** Goal-conditioned tasks have sparse success rewards and most attempted trajectories fail at their commanded goals.

**Assumptions:** Achieved goals can be extracted from states; reward and terminal status can be recomputed for another goal; the base RL method is off-policy.

**Algorithm:**

1. Roll out the policy for commanded goal $g_{\mathrm{cmd}}$ and store the full episode.
2. Store original transitions with $g_{\mathrm{cmd}}$, including failed rewards.
3. For a transition at time $t$, sample one or more achieved goals from later in the same episode (or the final state).
4. Copy the transition with the new goal; do not change its state, action, or next state.
5. Recompute reward and goal-conditioned terminal flag using the relabeled goal.
6. Add relabeled transitions to replay while retaining a chosen fraction of original commanded-goal data.
7. Train the goal-conditioned off-policy learner on the mixture.

**Limitations:** Relabeling does not make every transition successful: rewards must be recomputed step by step. With stochastic dynamics, choosing goals from realized futures can preferentially select lucky outcomes and bias the conditional transition distribution; retaining original goals does not by itself remove this [hindsight bias](https://arxiv.org/abs/2207.01115). Goals must not change the physical transition law. Ordinary on-policy REINFORCE cannot use relabeled trajectories without further correction. Relabeled goals remain limited to achieved outcomes.

### 10.3 Successor-representation learning

**Covered in:** Lecture 24.

**Use when:** The dynamics and a fixed policy should be represented independently of a state-based reward for rapid reevaluation.

**Assumptions:** Discrete states are enumerable or represented one-hot; policy $\pi$ is fixed during evaluation; future visitation predicts relevant returns.

**Local notation:** $\mu^\pi(s)$ is the normalized discounted future-state occupancy vector.

The displayed recurrence assumes a continuing process, including explicit absorbing terminal states if necessary. Then the occupancy vector sums to one. If instead episodes stop and terminal continuation is masked out, it is a sub-probability occupancy over preterminal states; use that convention consistently with the reward vector.

**Algorithm:**

1. Initialize vector prediction $\mu^\pi(s)$ for every state.
2. Follow policy $\pi$ and observe $s_t,s_{t+1}$.
3. Build target
   $$y_t=(1-\gamma)e_{s_t}+\gamma\mu^\pi(s_{t+1}),$$
   where $e_{s_t}$ is the one-hot vector for the current state.
4. Update $\mu^\pi(s_t)$ toward $y_t$ by tabular TD or vector regression.
5. Repeat over transitions until predictions stabilize.
6. For a new state-reward vector $r$, compute $V^\pi(s)=\mu^\pi(s)^Tr/(1-\gamma)$.

**Limitations:** Output dimension equals the number of states and does not directly extend to continuous states. It evaluates one policy; greedy improvement changes visitation and invalidates the old representation for exact evaluation.

### 10.4 Successor features and rapid reward transfer

**Covered in:** Lecture 24.

**Use when:** Many tasks share dynamics and their rewards are linear in a common low-dimensional feature map.

**Assumptions:** $r(s,a,s')\approx\phi(s,a,s')^Tw$; features contain all reward-relevant events; policy-specific successor features can be learned.

**Algorithm:**

1. Choose or learn feature vector $\phi$.
2. For fixed policy $\pi$, initialize $\psi^\pi(s,a)$.
3. From transition $(s,a,s')$, sample/choose $a'\sim\pi(\cdot\mid s')$ and form
   $$y=\phi(s,a,s')+\gamma(1-d)\psi^\pi(s',a').$$
4. Regress $\psi^\pi(s,a)$ toward $y$.
5. For a new task, collect a small reward dataset and fit $w$ by linear regression $r\approx\phi^Tw$.
6. Recover $Q^\pi(s,a)=\psi^\pi(s,a)^Tw$ and optionally take one greedy improvement step.

**Limitations:** Transfer fails when rewards are not represented by $\phi$. Successor features are policy dependent. One greedy step from one stored policy is not generally optimal for a new task.

### 10.5 Generalized policy improvement (GPI)

**Covered in:** Lecture 24.

**Use when:** A library of policies/successor features should be recombined for a new linear-reward task.

**Assumptions:** Each library policy has reasonably accurate successor features; the new reward weights can be estimated; the library spans useful behavior.

**Algorithm:**

1. Pretrain policies $\pi_1,\ldots,\pi_K$ and their successor features $\psi^{\pi_k}$.
2. For the new task, fit reward weights $w$.
3. At current state, evaluate every action under every stored policy:
   $$Q_k(s,a)=\psi^{\pi_k}(s,a)^Tw.$$
4. Choose
   $$a^*=\arg\max_a\max_kQ_k(s,a).$$
5. Repeat the statewise maximization at every decision; the selected source policy may change from state to state.
6. Optionally add the resulting policy to the library and learn its successor features.

**Limitations:** It is only as good as the feature map, estimates, and policy library. Evaluating a large library is costly. Approximation error can make the statewise maximum exploit the noisiest estimate.

### 10.6 Continuous successor density by classification (C-learning style)

**Covered in:** Lecture 24.

**Use when:** Future-state occupancy is needed in a continuous space where one-hot successor representations are impossible.

**Assumptions:** Positive future states and negative marginal states can be sampled; a classifier can estimate their density ratio; the future-time sampling rule matches the desired discounting.

**Algorithm:**

1. Sample a transition context $(s_t,a_t)$ from replay.
2. For an on-policy Monte Carlo estimate, sample a positive future state $s_f^+=s_{t+1+k}$ with $\Pr(k)=(1-\gamma)\gamma^k$. This convention estimates occupancy starting at the next state. Finite trajectory endings need an absorbing-state/tail convention; renormalizing over only available futures changes the estimated distribution.
3. Independently sample a negative state $s_f^-$ from the replay state marginal.
4. Train binary classifier $C_\omega(s,a,s_f)$ to label conditional futures as 1 and marginal negatives as 0, using equal class weights for the odds identity below.
5. Convert classifier odds to a density ratio:
   $$\frac{p^\pi(s_f\mid s,a)}{p(s_f)}=\frac{C_\omega(s,a,s_f)}{1-C_\omega(s,a,s_f)}.$$
6. Use the ratio, or its proportional form when $p(s_f)$ cancels, for goal values/action selection.
7. Refresh positives/negatives as the policy/replay distribution changes.

**Limitations:** With positive class prior $p_+$, multiply classifier odds by $(1-p_+)/p_+$ to recover the density ratio. Futures in arbitrary replay evaluate the behavior continuation, not automatically the current policy. The loop above is Monte Carlo future classification; full [C-learning](https://arxiv.org/abs/2011.08909) adds recursive, policy-conditioned classification updates. Uniform future sampling estimates a different object.

### 10.7 Semi-MDP Q-learning with fixed options

**Covered in:** Lecture 24.

**Use when:** Useful temporally extended skills/options are already available and should shorten the high-level decision horizon.

**Assumptions:** Each option has an initiation rule, intra-option policy, and termination condition or fixed duration; cumulative option rewards and duration are observed.

**Local notation:** Option $o$ starts at $s_t$, runs $h$ primitive steps, accumulates $R_o=\sum_{k=0}^{h-1}\gamma^kr_{t+k}$, and ends at $s_{t+h}$.

**Algorithm:**

1. At a high-level decision state, choose an available option with an exploratory policy over $Q(s,o)$.
2. Execute its low-level policy until termination or fixed duration, recording all primitive rewards.
3. Compute $R_o$ and actual duration $h$.
4. Form target
   $$y=R_o+\gamma^h(1-d)\max_{o'}Q(s_{t+h},o').$$
5. Update $Q(s_t,o_t)$ toward $y$.
6. Choose the next option and repeat.

**Limitations:** Performance depends on option quality and coverage. Long options can ignore important feedback; short options provide little abstraction. Discovering good options may be as hard as solving the original task.

### 10.8 Fixed-duration skill/goal hierarchy

**Covered in:** Lecture 24.

**Use when:** A practical hierarchy should reuse a goal- or skill-conditioned low-level policy without learned option termination.

**Assumptions:** The low-level policy can follow a context for $K$ steps; high-level contexts form a useful action space; each level can be trained on its own time scale.

**Algorithm:**

1. Pretrain or jointly train low-level $\pi_{\mathrm{low}}(a\mid s,z)$, where $z$ is a skill, subgoal, or command.
2. Every $K$ primitive steps, let high-level policy choose a new $z$.
3. Hold $z$ fixed while the low level emits and executes $K$ actions, or until environment termination.
4. Aggregate the discounted primitive rewards into one high-level reward and store the high-level transition.
5. Train the high-level policy as a semi-MDP learner over $z$.
6. Train/fine-tune the low level with its skill/goal objective, taking care not to destabilize the high-level transition semantics.

**Limitations:** Fixed duration may be inappropriate for variable-length subtasks. Nonstationary low-level learning makes the high-level environment nonstationary. Poor subgoal spaces make the hierarchy harder than a flat policy.

## 11. Lecture coverage map

| Lecture | Algorithmic coverage in this file |
|---:|---|
| 1 | Definitions and applications only; no executable algorithm introduced |
| 2 | BC, DAgger, intervention-based DAgger |
| 3 | Autoregressive actions, flow matching, action chunking, goal relabeling |
| 4 | Reflow, pretrain/post-train, goal-conditioned BC, generic RL loop; later algorithms previewed only |
| 5 | Full-return and reward-to-go/baseline REINFORCE |
| 6 | MC/TD/$n$-step evaluation, batch/online/off-policy actor-critic, GAE |
| 7 | DP evaluation, policy/value iteration, FVI/FQI, Q-learning, exploration rules |
| 8 | DQN, multi-step targets, Double Q/DDQN, clipped double Q, DDPG, CEM preview |
| 9 | Importance sampling, local off-policy surrogate, PPO-Clip |
| 10 | KL-PPO, NPG, TRPO |
| 11 | CAVI and EM |
| 12 | VAE/CVAE/sequential VAE, exact control-as-inference messages, soft value iteration |
| 13 | Soft Q-learning, entropy policy gradient, SAC, max-ent IRL; GCL/GAIL in slide continuation |
| 14 | GCL/GAIL, Bradley-Terry reward model, RLHF, GRPO |
| 15 | Iterative learned-model RL, ensemble uncertainty |
| 16 | Random shooting, CEM, MPC, differentiable/model-based policy learning, Dyna, MBPO/MVE family, latent models |
| 17 | Offline constraints and BRAC setup |
| 18 | SAC+BC, BRAC, AWR/AWAC, IQL, CQL, two-buffer online transition, IDQL, FQL, latent steering |
| 19 | MOPO/COMBO, UCB, count bonuses, pseudo-counts, CTS, RND |
| 20 | Value iteration and FQI analysis; no new training algorithm |
| 21 | Review of imitation, flow, goal conditioning, and policy gradient algorithms |
| 22 | Review of policy optimization, actor-critic, value methods, VI, maximum-entropy RL, IRL, RLHF, and model-based RL |
| 23 | DIAYN, goal-conditioned empowerment, Skew-Fit, Go-Explore principle |
| 24 | Goal-conditioned RL, HER, successor representations/features, GPI, C-learning-style classification, options/hierarchy |
| 25 | Domain randomization and the course algorithm atlas; no new optimization update |

## 12. Methods named or illustrated but not specified as full algorithms

The lectures mention the following as examples, relatives, or research systems without supplying enough procedural detail to justify inventing a complete course recipe here:

- **ALVINN:** early end-to-end driving system used as a behavioral-cloning case study.
- **Normalizing flows:** named as an expressive generative alternative; conditional flow matching is the fully specified course procedure.
- **CMA-ES:** named as a continuous optimizer related to CEM; its covariance adaptation is not taught.
- **MuZero and Monte Carlo tree search:** named in the value/model-based atlas, but no full search/training loop is developed.
- **SLAC and Dreamer:** examples on the latent model-based spectrum; the lecture teaches the sequential-latent recipe rather than either complete system.
- **MBA:** named with MVE and MBPO in the Dyna-style design space but not procedurally expanded.
- **Option-Critic:** cited as a classic way to learn options and termination jointly; its gradient updates are not presented.
- **Meta-learning:** appears only in the unreached slide-only continuation of Lecture 24.
- **AlphaGo, $\pi_0$, GNM, and ARCHER:** application/system examples rather than fully taught algorithms in the supplied course record.

## 13. Selection reminders

- Use **BC** when demonstrations are strong and interaction is unavailable; use **DAgger** only when interactive expert labels are feasible.
- Use **PPO/TRPO/NPG** when fresh on-policy interaction is acceptable; use **DQN/SAC/TD3** when replay and sample reuse matter.
- Use **DQN** for enumerable discrete actions; use **SAC/TD3/DDPG** for continuous actions, with SAC usually the more robust stochastic baseline.
- Use **model-based planning** when a model is credible and online compute is available; use a learned actor when deployment latency matters.
- Use **offline RL** only with explicit support control, pessimism, or in-distribution targets; ordinary online Q-learning/SAC is unsafe on a fixed dataset.
- Use **HER** for sparse goal rewards, **RND/pseudo-counts** for novelty, and **DIAYN/Skew-Fit** when pretraining broad behavior without a task reward.
- Treat all neural deep-RL guarantees as conditional on coverage, representation, optimization, and approximation quality; tabular convergence statements do not automatically transfer.
