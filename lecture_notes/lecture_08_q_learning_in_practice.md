---
title: "Lecture 8 - Q-Learning in Practice"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 8
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 8, Q-Learning in Practice.txt"
source_slides: "../lectures/Lecture 08 - Q-Learning in Practice.pdf"
transcript_lines: 7555
slide_pages: 36
status: "source-incomplete"
---

# Lecture 8: Q-Learning in Practice

> **Source warning:** The supplied transcript ends at line 7,555 in the middle of an audience question about the theoretical failure mode discussed at the end. Every supplied line is covered below, but the missing remainder is not reconstructed.

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Recap: epsilon-greedy Q-learning and replay | lines 1-217 |
| 2 | Why Q-learning is not ordinary gradient descent | lines 218-598 |
| 3 | Target networks and the DQN family | lines 599-1620 |
| 4 | The practical Q-learning process | lines 1621-1971 |
| 5 | Update-to-data ratio | lines 1972-2286 |
| 6 | Multi-step targets and value propagation | lines 2287-3165 |
| 7 | Overestimation in Q-learning | lines 3166-3441 |
| 8 | Double Q-learning and clipped double Q-learning | lines 3442-4674 |
| 9 | Practical implementation and debugging advice | lines 4675-5355 |
| 10 | Maximization with continuous actions | lines 5356-5733 |
| 11 | DDPG as approximate continuous-action maximization | lines 5734-6063 |
| 12 | Tabular convergence and the Bellman contraction | lines 6064-6465 |
| 13 | Function approximation as projection | lines 6466-6906 |
| 14 | Why fitted value methods can diverge | lines 6907-7555 |

## Part I - Stabilizing and accelerating deep Q-learning

## 1. Recap: epsilon-greedy Q-learning and replay

**Transcript coverage:** lines 1-217

### What the lecturer said - transcript only

The lecturer resumed the previous lecture's deep Q-learning algorithm. At a state $s_t$, the agent usually selects the action with the largest current $Q_\phi(s_t,a)$, but with probability $\epsilon$ it instead chooses a random action. That is the epsilon-greedy exploration rule.

After executing the action, the agent receives $s_{t+1}$ and $r_t$ and stores $(s_t,a_t,r_t,s_{t+1})$ in a replay buffer. It samples a mini-batch from that buffer and regresses the current Q-value toward a bootstrapped target containing the reward plus the discounted largest next-state Q-value. The replay buffer breaks up temporal correlations and allows old experience to be reused.

The lecturer framed the lecture as a discussion of the adjustments needed to make this apparently simple recipe work reliably with neural networks.

### Source reconciliation

Slides 2-3 write the target as $y_i=r_i+\gamma\max_{a'}Q_\phi(s_i',a')$ and the squared regression loss as $\frac{1}{2}(Q_\phi(s_i,a_i)-y_i)^2$. The transcript describes the same procedure verbally.

### Additional explanation

Replay changes the data distribution seen by an update: the learner trains on a mixture of experience produced by several earlier versions of the behavior policy. This reuse is one reason Q-learning can be much more sample-efficient than an on-policy method, but it also means the target and the sampled data are both nonstationary.

## 2. Why Q-learning is not ordinary gradient descent

**Transcript coverage:** lines 218-598

### What the lecturer said - transcript only

Although the update is commonly implemented as regression, it is not gradient descent on a fixed supervised-learning objective. The same network appears on both sides: its parameters determine the prediction $Q_\phi(s,a)$ and the target $r+\gamma\max_{a'}Q_\phi(s',a')$. As the parameters change, the target moves.

The correct implementation stops the gradient through the target. Backpropagation is applied only through the prediction. If one differentiates through both sides, the computation is no longer the Q-learning update that was derived. The lecturer compared the situation to fitting a network to a dataset whose labels are continuously rewritten by the network itself.

Full fitted Q-iteration would freeze all targets, solve the resulting regression problem, recompute every target with the new function, and repeat. Online deep Q-learning interleaves those stages: it takes only a small number of regression steps before changing the target again. Actor-critic has the same moving-target issue when its critic bootstraps from its own estimates.

### Source reconciliation

Slide 4 explicitly marks the target branch with “no gradient,” while slide 5 contrasts a fixed supervised target with the repeated target-recompute and regression stages of fitted Q-iteration.

### Additional explanation

Stopping the gradient makes the update a **semi-gradient**. It treats the bootstrap value as a temporary label even though that label depends on the parameters globally. This distinction is central to the “deadly triad”: function approximation, bootstrapping, and off-policy data can interact in ways that ordinary supervised-learning intuition does not predict.

## 3. Target networks and the DQN family

**Transcript coverage:** lines 599-1620

### What the lecturer said - transcript only

The standard stabilizing device is a target network $Q_{\bar\phi}$. The online network $Q_\phi$ is trained against targets computed with $Q_{\bar\phi}$, and $\bar\phi$ is updated more slowly. The original DQN scheme copies the online parameters to the target parameters only periodically; the lecturer gave 10,000 environment steps as the characteristic update interval in the original setup.

This makes the labels temporarily stationary. Between target-network updates, the learner is closer to ordinary regression. When the target is copied, the labels move abruptly, after which another block of regression steps follows. The lecturer emphasized that the number is a design choice rather than a universal constant.

The target-network idea converts the earlier deep Q-learning procedure into the recognizable DQN family: collect epsilon-greedy experience, store it in a replay buffer, sample batches, form targets with the target network, update the online network, and periodically update the target.

An alternative is a gradual or Polyak update. In the convention used in the lecture, a coefficient close to one retains most of the old target parameters:

$$
\bar\phi \leftarrow \tau\bar\phi+(1-\tau)\phi,
\qquad \tau\approx 0.999.
$$

The lecturer said that hard periodic copies and gradual updates can both work. The replay buffer does not need to be cleared when the target network changes. In response to questions, he distinguished the target network from an independently trained second estimator: it is deliberately a delayed version of the online network. He also noted the original DQN replay capacity of roughly one million transitions and explained that the target update interval, buffer capacity, and learning dynamics should not be confused with one another.

### Source reconciliation

Slides 6-8 show the hard copy every 10,000 steps and the Polyak formula. The coefficient convention matters: here $\tau$ multiplies the **old target**, so $\tau=0.999$ is a slow update. Other sources use the opposite convention.

### Additional explanation

The target network introduces a second time scale. The online estimator is allowed to fit a nearly fixed target, while the target estimator tracks it slowly. If the target changes too quickly, training can chase its own errors; if it changes too slowly, value information propagates sluggishly.

## 4. The practical Q-learning process

**Transcript coverage:** lines 1621-1971

### What the lecturer said - transcript only

The lecturer reorganized DQN into four concurrent-looking processes:

1. interact with the environment and add transitions to replay;
2. sample training data from replay;
3. update the target network on its slower schedule; and
4. perform regression updates on the online network.

Thinking in terms of processes makes the important rates explicit. Data collection, gradient updates, and target-network updates need not occur once each in a fixed lockstep. In a distributed or large implementation, they may genuinely run as separate processes.

An audience question asked whether the same procedure could be used offline. The lecturer said that if no new transitions are collected, one has offline Q-learning, but the distributional problems become more severe; those problems would be covered later in the course.

### Source reconciliation

Slides 10-11 visualize the pipeline and identify replay sampling, target updates, and regression as separable operations.

### Additional explanation

This process view is useful because “one training step” is ambiguous. Reporting environment steps, gradient steps, target copies, and replay insertions separately makes experiments reproducible and exposes hidden sample-efficiency tradeoffs.

## 5. Update-to-data ratio

**Transcript coverage:** lines 1972-2286

### What the lecturer said - transcript only

The update-to-data ratio (UTD ratio) is the number of gradient updates performed per newly collected transition. Increasing it can improve sample efficiency because each expensive environment observation is used more thoroughly. It also consumes more computation and increases the risk of overfitting to the current replay contents or magnifying bootstrap error.

The lecturer described ratios above roughly 5-10 as often risky in standard implementations, while noting that specially designed methods have made ratios around 100 work. There is therefore no universally correct UTD ratio: it is a computational and statistical tradeoff.

### Source reconciliation

Slide 12 defines the UTD ratio as training updates divided by environment steps.

### Additional explanation

At high UTD, the critic changes substantially while the replay distribution barely changes. Techniques used in high-UTD algorithms commonly target critic overfitting, Q-value bias, or ensemble uncertainty rather than treating more updates as automatically beneficial.

## 6. Multi-step targets and value propagation

**Transcript coverage:** lines 2287-3165

### What the lecturer said - transcript only

Using a grid-world example, the lecturer showed that a one-step backup moves reward information only one state backward per effective target update. A target network that is copied only every 1,000 or 10,000 steps can therefore make long-range propagation slow.

An $n$-step target inserts several observed rewards before bootstrapping:

$$
y_t^{(n)}=r_t+\gamma r_{t+1}+\cdots+\gamma^{n-1}r_{t+n-1}
+\gamma^n\max_a Q_{\bar\phi}(s_{t+n},a).
$$

This moves information backward by $n$ time steps at once and often improves practical performance. A modest value such as $n=5$ is common.

The cost is that the intermediate actions came from the behavior policy. One-step Q-learning is off-policy because its next-action value uses a maximization rather than the action actually taken. A multi-step return includes rewards produced by several actual behavior actions, so it is no longer cleanly off-policy and can be biased downward when those actions are exploratory or poor. The lecturer described Retrace as a more advanced method that uses importance weighting and a dynamically varying effective horizon to address this problem.

In Q&A, the lecturer clarified that a target-network update does not itself perform all value propagation; it changes the labels on which subsequent regression operates. The numerical grid example was intended to isolate this propagation mechanism.

### Source reconciliation

Slide 13 displays the $n$-step return. The transcript's automatic captions render the name “Retrace” inconsistently; the slide and the algorithmic description identify the intended term.

### Additional explanation

The tradeoff resembles the bias-variance spectrum of $n$-step actor-critic returns. Longer returns depend less on a possibly inaccurate bootstrap value, but depend more on the sampled behavior trajectory and usually have higher variance.

## Part II - Bias, practical advice, and continuous actions

## 7. Overestimation in Q-learning

**Transcript coverage:** lines 3166-3441

### What the lecturer said - transcript only

Q-learning often predicts values larger than the returns the policy actually obtains. The lecturer pointed to empirical plots in which predicted Q-values rise while realized returns do not, including Atari comparisons between predicted and actual values.

The cause is the maximization in the target. If estimates for several actions contain zero-mean noise, the maximum preferentially selects whichever estimate received a positive error. Even if each action estimator is unbiased individually, the maximum is biased upward:

$$
\mathbb E[\max(X_1,X_2)]
\geq
\max(\mathbb E[X_1],\mathbb E[X_2]).
$$

Bootstrapping can then propagate this selected positive error into preceding states.

### Source reconciliation

Slides 15-17 show learning curves and the max/Jensen argument. Slide 16 identifies Double DQN as a method that makes predicted Atari values track actual values more closely.

### Additional explanation

The bias does not require every Q-value to be too high. It arises because one noisy estimate is used both to **select** an action and to **evaluate** that selected action. Decoupling those roles is the key idea behind double estimators.

## 8. Double Q-learning and clipped double Q-learning

**Transcript coverage:** lines 3442-4674

### What the lecturer said - transcript only

In idealized double Q-learning, two independently noisy estimators are maintained. One estimator selects the maximizing action, and the other evaluates it. Since the evaluation noise is independent of the selection noise, selecting a positive error in one estimator does not systematically select a positive error in the other.

Deep Q-learning already has two networks, but they are not independent: the target network is a delayed copy of the current network. Practical Double DQN nevertheless uses them to partially separate selection and evaluation. The current network selects the next action, while the target network evaluates it:

$$
a^* = \arg\max_{a'}Q_\phi(s',a'),
\qquad
y=r+\gamma Q_{\bar\phi}(s',a^*).
$$

This usually reduces overestimation and is a simple, widely useful modification.

A more general ensemble approach uses multiple critics. Clipped double Q-learning takes the smaller of two target estimates:

$$
y=r+\gamma\min_{j\in\{1,2\}}Q_{\bar\phi_j}(s',a^*).
$$

The minimum intentionally introduces pessimism and is especially useful when an actor explicitly searches for actions that exploit critic errors, as in continuous-control actor-critic methods. The lecturer warned that excessive pessimism can cause underestimation; the technique trades one bias for another rather than producing a perfectly unbiased estimator.

In Q&A, he noted that the original double-Q idea predates target networks. The delayed target in DQN was introduced for stability, while double estimation was introduced to address maximization bias.

### Source reconciliation

Slides 18-20 distinguish true double Q-learning, practical Double DQN, and clipped double Q-learning. In implementations of clipped double Q-learning, the target action and target critics depend on the surrounding algorithm; the equation above captures the lecture's central minimum-of-critics idea.

### Additional explanation

Double DQN reduces correlation between selection and evaluation only partially because the networks share training history. Clipped double critics are more conservative: an action must look good to both critics to receive a high target.

## 9. Practical implementation and debugging advice

**Transcript coverage:** lines 4675-5355

### What the lecturer said - transcript only

The lecturer offered several practical recommendations:

- A replay buffer should be large enough to decorrelate samples, but making it arbitrarily large retains very old, less relevant behavior. Capacity is a hyperparameter.
- Learning curves may have an S-shape: little visible progress, a rapid improvement phase, and a plateau. Stopping during the flat beginning can falsely suggest failure.
- Exploration schedules matter. Epsilon is commonly annealed from a high value to a lower value, and learning-rate schedules may also help.
- Large temporal-difference errors can destabilize training. Gradient clipping or the Huber loss can reduce the influence of outliers while remaining quadratic near zero and linear for large residuals.
- Random seeds matter. Q-learning runs can vary substantially, so multiple seeds are required before drawing conclusions.
- Diagnostics should compare returns, predicted values, losses, and exploration behavior rather than relying on one curve.

During the intermission Q&A, the lecturer revisited why target networks and double estimators are conceptually distinct and stepped through the grid-world propagation example numerically.

### Source reconciliation

Slides 21-22 separate simple tips from more advanced stabilizers and explicitly mention Huber loss. Slide 23 marks the intermission.

### Additional explanation

A decreasing loss is not sufficient evidence that Q-learning works: the labels themselves move. Episodic return is the primary outcome measure, while the scale and drift of predicted Q-values provide a useful consistency check.

## 10. Maximization with continuous actions

**Transcript coverage:** lines 5356-5733

### What the lecturer said - transcript only

For discrete actions, the target can evaluate every action and take a maximum. In a continuous action space there are infinitely many candidates, so exhaustive enumeration is impossible. The Q-learning target therefore requires solving a continuous optimization problem at every next state.

One option is stochastic optimization: sample random actions, retain promising ones, refit a proposal distribution, and repeat. The lecturer named the cross-entropy method (CEM) and CMA-ES as examples. These methods can be effective in moderately sized action spaces—he gave roughly 40 dimensions as a plausible upper range—but become expensive because each target evaluation now contains an inner optimization loop.

### Source reconciliation

Slides 24-26 show the continuous maximization problem and list random sampling, CEM, and CMA-ES.

### Additional explanation

Continuous-action maximization can also be performed with gradients through $Q(s,a)$, but a non-convex critic may contain many local optima or erroneous high-value regions. An actor amortizes this repeated optimization.

## 11. DDPG as approximate continuous-action maximization

**Transcript coverage:** lines 5734-6063

### What the lecturer said - transcript only

Deep Deterministic Policy Gradient (DDPG) introduces a deterministic actor $\mu_\theta(s)$ whose job is to approximate
$\arg\max_a Q_\phi(s,a)$. The critic target uses the target actor and target critic, while the actor is improved by differentiating the critic with respect to its action input:

$$
y=r+\gamma Q_{\bar\phi}\bigl(s',\mu_{\bar\theta}(s')\bigr),
$$

$$
\nabla_\theta J
\approx
\mathbb E_s\!\left[
\nabla_aQ_\phi(s,a)\big|_{a=\mu_\theta(s)}
\nabla_\theta\mu_\theta(s)
\right].
$$

The result can be viewed either as Q-learning with a learned approximate maximizer or as an off-policy actor-critic method with a deterministic actor. Replay buffers, target networks, double critics, and other Q-learning stabilizers remain relevant.

### Source reconciliation

Slides 27-28 present DDPG in exactly these two interpretations.

### Additional explanation

Because the actor is optimized against the learned critic, it can discover actions where critic error is high. This is why clipped double critics and target-policy smoothing became important descendants of the same family.

## Part III - What theory does and does not guarantee

## 12. Tabular convergence and the Bellman contraction

**Transcript coverage:** lines 6064-6465

### What the lecturer said - transcript only

For a finite tabular MDP, define the Bellman optimality operator

$$
(\mathcal BQ)(s,a)
=r(s,a)+\gamma\,
\mathbb E_{s'\sim p(\cdot\mid s,a)}
\left[\max_{a'}Q(s',a')\right].
$$

Q-value iteration repeatedly applies this operator: $Q_{k+1}=\mathcal BQ_k$. The operator is a contraction in the infinity norm:

$$
\|\mathcal BQ_1-\mathcal BQ_2\|_\infty
\leq \gamma\|Q_1-Q_2\|_\infty.
$$

Because $0\leq\gamma<1$, repeated application brings functions closer together and yields a unique fixed point $Q^*$ satisfying $Q^*=\mathcal BQ^*$. This is the familiar reason tabular value iteration converges.

### Source reconciliation

Slides 30-31 display the Bellman operator and its infinity-norm contraction. The transcript walks through the intuition that taking a maximum and an expectation cannot enlarge the largest pointwise difference, after which multiplication by $\gamma$ shrinks it.

### Additional explanation

The contraction is about the exact operator on the complete table of values. It does not automatically apply after that table is replaced by a restricted neural-network family and a finite-sample regression procedure.

## 13. Function approximation as projection

**Transcript coverage:** lines 6466-6906

### What the lecturer said - transcript only

Fitted value iteration with function approximation can be represented as two operations. First apply the Bellman operator; then project the resulting function back into the representable function class $\Omega$:

$$
V_{k+1}=\Pi_\Omega\mathcal BV_k.
$$

The projection corresponds to supervised fitting—for example, choosing the neural-network parameters whose predictions are closest to the Bellman targets in a squared-error norm. With finite samples, the procedure also contains sampling and optimization errors that the ideal projection notation omits.

The same abstraction applies to fitted Q-iteration. A larger function class can make the projection error smaller because the Bellman-updated function is more likely to be representable.

### Source reconciliation

Slides 32 and 34 depict the Bellman update followed by projection. Slide 32 labels $\Omega$ as the set of functions representable by the chosen approximator.

### Additional explanation

Projection is not merely numerical approximation. It couples otherwise independent states through shared parameters: fitting one target can change predictions elsewhere. That coupling is the mechanism by which local errors can feed back into later bootstrap targets.

## 14. Why fitted value methods can diverge

**Transcript coverage:** lines 6907-7555

### What the lecturer said - transcript only

The Bellman operator contracts in the infinity norm, while least-squares projection is naturally non-expansive in a data-weighted Euclidean norm. Two maps can each be contractions in different norms without their composition being a contraction in any one useful norm. Consequently, $\Pi_\Omega\mathcal B$ can diverge or enter an orbit even though the exact Bellman operator converges.

The lecturer illustrated this geometrically: one operation shrinks distance according to one geometry, the next shrinks according to another, yet alternating them can move the iterate around or outward. This is a basic reason fitted value iteration, fitted Q-iteration, and bootstrapped critics in actor-critic can be unstable with function approximation.

Increasing model capacity can help by reducing projection error, but it does not constitute a general proof of convergence. The practical implication is not that deep Q-learning never works; it is that its success depends on stabilizing choices such as replay, target networks, conservative updates, useful representations, and suitable data coverage. The supplied transcript cuts off while an audience member begins a question about this failure mode, so neither the complete question nor the answer is available.

### Source reconciliation

Slides 33-36 explicitly state that composition of the Bellman contraction and projection need not be a contraction, extend the point from fitted value iteration to fitted Q-iteration and actor-critic, and list the implications. The final slide is visible, but it cannot supply the missing spoken Q&A.

### Additional explanation

This analysis explains why supervised-learning metrics alone cannot certify stability. The learner is not fitting a fixed ground-truth function: its current approximation changes the target operator that generates the next training problem.

## Consolidated takeaways

1. Deep Q-learning uses semi-gradient regression against targets generated by a delayed copy of the critic.
2. Target networks slow target motion; replay buffers reuse and decorrelate experience.
3. Data collection, critic updates, and target updates have separate rates, summarized partly by the UTD ratio.
4. Multi-step returns propagate reward faster but sacrifice the clean one-step off-policy property.
5. Maximization of noisy estimates causes systematic overestimation.
6. Double DQN separates action selection from evaluation; clipped double Q deliberately adds pessimism.
7. Learning curves, exploration schedules, robust losses, Q-value calibration, and multiple seeds are important practical diagnostics.
8. Continuous-action Q-learning needs an optimizer; DDPG amortizes that optimizer with a deterministic actor.
9. Exact tabular Bellman updates contract, but projection into a function class can destroy the contraction.
10. Target networks and other practical devices improve stability without supplying a universal convergence guarantee.

## Key equations

### One-step DQN target

$$
y=r+\gamma\max_{a'}Q_{\bar\phi}(s',a'),
\qquad
\mathcal L(\phi)=\frac12\bigl(Q_\phi(s,a)-y\bigr)^2,
$$

with no gradient through $y$.

### Polyak target update

$$
\bar\phi\leftarrow\tau\bar\phi+(1-\tau)\phi.
$$

### $n$-step target

$$
y_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}
+\gamma^n\max_aQ_{\bar\phi}(s_{t+n},a).
$$

### Double DQN target

$$
y=r+\gamma Q_{\bar\phi}
\left(s',\arg\max_{a'}Q_\phi(s',a')\right).
$$

### Bellman contraction

$$
\|\mathcal BQ_1-\mathcal BQ_2\|_\infty
\leq\gamma\|Q_1-Q_2\|_\infty.
$$

### Fitted update

$$
Q_{k+1}=\Pi_\Omega\mathcal BQ_k.
$$

The contraction of $\mathcal B$ does not by itself imply contraction of $\Pi_\Omega\mathcal B$.

## Glossary

- **Action prior:** a baseline distribution over actions before conditioning on an objective; it appears later in control-as-inference lectures.
- **Bellman operator:** the map that replaces each value by immediate reward plus discounted optimal next-state value.
- **Clipped double Q-learning:** a conservative target using the smaller of two critic estimates.
- **DDPG:** an off-policy continuous-control algorithm with a deterministic actor and bootstrapped critic.
- **Double DQN:** a DQN target in which the online network selects the action and the target network evaluates it.
- **Huber loss:** a residual loss that is quadratic near zero and linear for large magnitude errors.
- **Maximization bias:** upward bias caused by selecting the maximum among noisy estimates.
- **Polyak update:** gradual averaging of online parameters into target parameters.
- **Projection:** fitting a Bellman-updated function back into a restricted approximating class.
- **Replay buffer:** stored transitions sampled for later training.
- **Semi-gradient:** an update that differentiates the prediction while treating a parameter-dependent bootstrap target as fixed.
- **Target network:** a delayed critic used to produce more slowly changing regression targets.
- **Update-to-data ratio:** gradient updates performed per newly collected environment transition.

## Self-check questions

1. Why is the DQN regression target not a fixed supervised label?
2. Where must gradient flow be stopped in the Q-learning loss, and why?
3. How do hard target copies and Polyak updates differ?
4. What tradeoff is controlled by the UTD ratio?
5. Why does an $n$-step target propagate reward faster?
6. Why does an $n$-step target weaken Q-learning's off-policy property?
7. How can individually unbiased action values produce an upward-biased maximum?
8. Which network selects and which evaluates in Double DQN?
9. Why can clipped double Q-learning underestimate values?
10. What diagnostics would reveal rising Q predictions without rising returns?
11. Why is continuous-action maximization an inner optimization problem?
12. In what two ways can DDPG be interpreted?
13. What norm makes the exact Bellman operator a contraction?
14. Why does projection invalidate the direct tabular convergence argument?

## Source coverage checklist

| Topic | Inclusive range | Coverage status |
|---:|---:|---|
| 1 | 1-217 | Accounted for |
| 2 | 218-598 | Accounted for |
| 3 | 599-1620 | Accounted for |
| 4 | 1621-1971 | Accounted for |
| 5 | 1972-2286 | Accounted for |
| 6 | 2287-3165 | Accounted for |
| 7 | 3166-3441 | Accounted for |
| 8 | 3442-4674 | Accounted for |
| 9 | 4675-5355 | Accounted for |
| 10 | 5356-5733 | Accounted for |
| 11 | 5734-6063 | Accounted for |
| 12 | 6064-6465 | Accounted for |
| 13 | 6466-6906 | Accounted for |
| 14 | 6907-7555 | Accounted for; source ends mid-question |

**Coverage result:** All 7,555 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. All 36 slide pages were visually inspected; the transcript truncation is not repaired or extended.
