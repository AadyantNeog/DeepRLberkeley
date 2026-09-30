---
title: "Lecture 7 - Value-Based Reinforcement Learning"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 7
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 7, Value-Based RL.txt"
source_slides: "../lectures/Lecture 07 - Value-Based RL.pdf"
transcript_lines: 4357
slide_pages: 20
status: "complete"
---

# Lecture 7: Value-Based Reinforcement Learning

## Lecture map

**Equation conventions.** Analytic Bellman equations below use an absorbing terminal state with value zero. In code, use $y=r+\gamma m\max_{a'}Q_{\rm ref}(s',a')$, where $m=0$ at a true terminal and $m=1$ otherwise, and detach the entire target. At a collection cutoff, bootstrap from the final pre-reset observation. Finite-horizon values require a time index or remaining time in the state.

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Removing the actor: Q-learning from off-policy actor-critic | lines 1-502 |
| 2 | Initial Q-learning Q&A and the second derivation | lines 503-786 |
| 3 | The policy-iteration framework | lines 787-969 |
| 4 | Tabular dynamic-programming policy evaluation | lines 970-1467 |
| 5 | Tabular policy iteration and value iteration | lines 1468-1770 |
| 6 | Curse of dimensionality and fitted value iteration | lines 1771-2148 |
| 7 | Q-evaluation and fitted Q-iteration | lines 2149-2535 |
| 8 | Function-approximation caveats and Q-network forms | lines 2536-2757 |
| 9 | The full fitted Q-iteration recipe and its schedules | lines 2758-3096 |
| 10 | Why fitted Q-iteration is off-policy | lines 3097-3225 |
| 11 | Bellman error and what fitted Q-iteration optimizes | lines 3226-3462 |
| 12 | Online Watkins Q-learning as a limiting schedule | lines 3463-3599 |
| 13 | Exploration: greedy, epsilon-greedy, and Boltzmann policies | lines 3600-4218 |
| 14 | Replay-buffer Q-learning and the instability warning | lines 4219-4357 |

## Part I - Actor-critic without the actor

## 1. Removing the actor: Q-learning from off-policy actor-critic

**Transcript coverage:** lines 1-502

### What the lecturer said - transcript only

After policy gradients and actor-critic, the lecture would simplify rather than add machinery: remove the actor and retain only a critic. These are **value-based methods**, and students would implement them in Homework 3.

The starting point was the previous lecture’s off-policy actor-critic. Take an action with the latest policy, store the resulting transition in a replay buffer, and later sample a mini-batch—perhaps 64 or 128—from that buffer. A replay buffer is commonly a ring buffer containing a large recent history, possibly the latest one million transitions, evicting the oldest as new data arrives. The training batch need not contain the just-collected transition.

For each replay transition, the critic target is immediate reward plus the expected current-policy Q-value at the stored next state. The next action is sampled from the latest policy rather than the behavior policy that originally produced the transition, which makes the method off-policy. After Q-regression, the actor is updated using either a likelihood-ratio or reparameterized policy gradient.

For a small enumerable discrete action space, such as four Atari buttons, a separate actor is unnecessary. Evaluate $Q(s,a)$ for every action and define a deterministic policy that puts probability one on an argmax action:

$$
\pi_Q(a\mid s)=
\begin{cases}
1,&a=a^*(s),\\
0,&\text{otherwise},
\end{cases}
$$

where $a^*(s)$ is one selected maximizer under a fixed tie-breaking convention. Under this policy, the expectation over the next action becomes a maximum. Delete the actor-gradient and actor-update steps; the target becomes

$$
y_i=r(s_i,a_i)+\gamma\max_{a'}Q_\theta(s_i',a').
$$

The resulting three-step, critic-only off-policy algorithm is Q-learning: collect and store a transition using the policy implicit in Q, load a replay batch, compute max-Q targets, and take one or a few squared-error critic steps. The lecturer joked that the six-minute lecture could end there, because this already supplied the core needed for the homework; the remainder would explain the principle underneath it.

### Source reconciliation

Slides 2-4 visually cross out the actor update and replace the current-policy expectation with a max. The deck’s deterministic-policy notation assumes one chosen argmax; ties require an implementation rule but were not discussed in the transcript.

### Additional explanation

The policy still exists behaviorally even though it has no independent parameters. It is a deterministic computation derived from Q, so improving Q changes both evaluation and action selection.

Choose one maximizer $a^*(s)$ using a fixed tie-breaking rule. The deterministic policy assigns probability one to that selected action, not probability one to every tied maximizer.

## 2. Initial Q-learning Q&A and the second derivation

**Transcript coverage:** lines 503-786

### What the lecturer said - transcript only

Asked whether Q-learning must be off-policy, the lecturer said nothing requires a large replay buffer. Capacity one makes the update effectively on-policy, though it then behaves like batch-size-one learning and is not attractive. The practical trade-off between more and less off-policy data is complicated; empirically, large buffers often work well, but there is no simple characterization of which problems require which degree of recency.

Q can be viewed as a predictive model of the consequences of an action under the future greedy policy. The lecturer connected this intuition to Peter Dayan’s successor-representation work: if rewards are state indicators, value-like predictions become probabilities or occupancies of reaching different states, resembling a multi-step model. He suggested this might reappear in model-based RL.

The rest of the lecture would derive the same Q-learning endpoint from another perspective. Removing the actor from actor-critic is direct, but policy iteration and dynamic programming expose more of the structure of value and Q-functions.

### Source reconciliation

The successor-representation reference and buffer-capacity discussion are transcript-only Q&A; the slides transition directly to policy iteration.

### Additional explanation

Q is not a full dynamics model because it collapses all possible futures into expected discounted return for a specified continuation policy. Successor representations retain more predictive structure by separating expected future state occupancy from rewards.

**Correction to the replay Q&A:** a buffer of capacity one does not by itself make Q-learning on-policy. Even on the newest transition, its bootstrap follows a greedy target policy while an epsilon-greedy behavior policy can choose another action. Buffer age and behavior/target-policy agreement are separate issues. A successor representation measures expected discounted visit counts, not simply the probability of ever reaching a state; repeated visits can make a count exceed one.

## Part II - Policy iteration and dynamic programming

## 3. The policy-iteration framework

**Transcript coverage:** lines 787-969

### What the lecturer said - transcript only

At a high level, **policy iteration** alternates two familiar stages:

1. **Policy evaluation:** estimate how good a policy is, often through $V^\pi$ or $Q^\pi$ for every state or state-action pair.
2. **Policy improvement:** use that evaluation to construct a better policy.

Actor-critic and Q-learning can both be viewed through this green evaluation / blue improvement framework. A conventional improvement step makes the new policy greedy with respect to the evaluated Q-function. Policy evaluation can represent either Q or V because they are linked through the Bellman relationship.

A conventional introductory RL course often begins with value functions and dynamic programming and only later reaches policy gradients. This course had taken the reverse route, so the lecturer would briefly traverse the classic path while students already knew the destination.

### Source reconciliation

Slides 5-6 show the same three-stage course diagram with evaluation and greedy improvement substituted for learned actor optimization.

### Additional explanation

Policy iteration separates “understand the current policy” from “make it greedy with respect to that understanding.” Value iteration later interleaves these operations so tightly that an explicit policy table disappears.

## 4. Tabular dynamic-programming policy evaluation

**Transcript coverage:** lines 970-1467

### What the lecturer said - transcript only

The derivation began with deliberately unrealistic assumptions: the state and action spaces are both small and discrete, and the complete transition probabilities $p(s'\mid s,a)$ are known. A $4\times4$ grid world has 16 states and perhaps four actions—up, down, left, right. Each $V^\pi(s)$ can be stored as one of 16 table entries, while the transition model is a $16\times16\times4$ tensor over current state, next state, and action.

With the model known, policy evaluation can perform exact Bellman expectation backups:

$$
V^\pi(s)\leftarrow
\mathbb E_{a\sim\pi(\cdot\mid s)}
\left[
r(s,a)+\gamma
\mathbb E_{s'\sim p(\cdot\mid s,a)}[V^\pi(s')]
\right].
$$

All expectations are finite sums or small vector inner products. The right side uses the current V estimate, so it is bootstrapped, but it introduces no sampling error. This is dynamic programming rather than trial-and-error RL: no policy need be run in the world.

After evaluation, the improved policy selects the action with the greatest Q-value. For a deterministic policy, the evaluation backup simplifies by substituting $a=\pi(s)$.

One student worried that repeatedly adding rewards to the current estimate might explode. With bounded rewards and $\gamma<1$, the geometric series bounds value by $R_{\max}/(1-\gamma)$. Another asked why not roll out the tiny problem to the end; that is possible, but the model-based dynamic-programming backup evaluates expectations exactly without sampling.

For a fixed policy and known finite model, policy evaluation also has a closed-form solution obtained by solving linear equations. Optimal control has no comparable linear closed form because the max operator is nonlinear.

### Source reconciliation

Slide 7 explicitly states the known-model and small-discrete-space assumptions. Its tensor dimensions match the 16-state, four-action spoken example.

### Additional explanation

In matrix form for a fixed policy,

$$
V^\pi=r^\pi+\gamma P^\pi V^\pi,
\qquad
V^\pi=(I-\gamma P^\pi)^{-1}r^\pi,
$$

when the inverse exists. Iterative evaluation avoids forming that inverse and generalizes more naturally to later approximate methods.

Here $V$ is a column vector and $P^\pi_{ss'}=P(s'\mid s,\pi)$ is row-stochastic. For a finite discounted MDP, $\gamma<1$ ensures the inverse exists. A column-stochastic convention requires transposing the transition matrix.

## 5. Tabular policy iteration and value iteration

**Transcript coverage:** lines 1468-1770

### What the lecturer said - transcript only

Tabular policy iteration repeatedly applies the exact value backup until policy evaluation converges, then stores for each state the greedy action

$$
\pi'(s)\in\arg\max_a Q^\pi(s,a).
$$

In the tabular setting, a deterministic policy is literally a table of action indices. This method is often the first algorithm taught in a conventional RL course. It does not extend directly to large unknown environments because it exhaustively enumerates states and depends on known transition probabilities.

The two-step procedure can be simplified. A table of Q-values has one row per state and one column per action; greedy improvement selects the maximum entry in every row. The value under that greedy choice is simply $\max_a Q(s,a)$. Therefore an explicit policy table can be skipped: store values, derive the policy by argmax when needed, and insert the max directly into the backup. This yields value iteration:

$$
V(s)\leftarrow
\max_a\left[
r(s,a)+\gamma
\mathbb E_{s'\sim p(\cdot\mid s,a)}V(s')
\right].
$$

Every backup here is still exhaustive over all states and actions and uses the known transition model; there are no samples. At inference time the action is the maximizing action. The difference from explicit policy iteration is representational: value iteration stores no separate policy table.

### Source reconciliation

Slides 8-9 show greedy policy extraction as selecting highlighted cells from a Q table, then eliminate the explicit policy in favor of alternating Q and max-V backups.

### Additional explanation

Policy iteration performs substantial evaluation for one fixed policy before improving it. Value iteration performs a greedy optimality backup at every sweep, effectively interleaving evaluation and improvement.

This is an algorithmic difference, not merely the removal of an explicit policy table. Exact policy iteration evaluates each policy to its fixed point; value iteration usually changes the implicit greedy policy before such evaluation finishes. Both can be implemented with or without storing a separate policy.

## Part III - Fitted value iteration

## 6. Curse of dimensionality and fitted value iteration

**Transcript coverage:** lines 1771-2148

### What the lecturer said - transcript only

A table works only for a small state space. A $200\times200$ RGB image with 255 channel values creates an astronomically large set of possible states—larger than can be stored one entry at a time. This exponential blow-up with dimensionality is the **curse of dimensionality**.

Replace the table by a neural network $V_\theta:S\to\mathbb R$. Rather than write each Bellman-updated value into a table, use it as a supervised regression target. Training on sampled states may allow the network to generalize over the enormous state space, at the cost of approximation error.

Fitted value iteration forms

$$
y_i=\max_{a_i}\left[
r(s_i,a_i)+\gamma
\mathbb E_{s_i'\sim p(\cdot\mid s_i,a_i)}V_\theta(s_i')
\right],
$$

then chooses parameters minimizing

$$
\frac12\sum_i\|V_\theta(s_i)-y_i\|^2.
$$

This is a valid deep-RL-style algorithm when actions can be enumerated and the transition model is known. For every action in the max, it must evaluate possible next states and their values. That requirement makes it impractical when dynamics are unknown.

In Q&A, the lecturer clarified that $\arg\min_\theta$ in the second step selects neural-network parameters, not an action; it is ordinary supervised regression.

### Source reconciliation

Slides 10-12 show the image-space state count, label it the curse of dimensionality, and underline that fitted value iteration needs counterfactual outcomes for different actions.

### Additional explanation

Fitted methods separate a target-computation phase from a regression phase. The target should be treated as fixed during each supervised fit even though it was created from an earlier network estimate.

The image-count example should use **256** values for an 8-bit channel (0 through 255). A $200\times200$ RGB array therefore has $256^{120000}$ possible encodings. The dimensionality argument is unchanged.

## 7. Q-evaluation and fitted Q-iteration

**Transcript coverage:** lines 2149-2535

### What the lecturer said - transcript only

To remove the need for known transitions, the lecturer returned to policy iteration. Evaluate a fixed policy through Q rather than V:

$$
Q^\pi(s,a)=r(s,a)+\gamma
\mathbb E_{s'\sim p(\cdot\mid s,a)}
[Q^\pi(s',\pi(s'))].
$$

This can be fit from sampled transitions $(s_i,a_i,s_i')$ because the current action $a_i$ is part of the sample and the continuation action is obtained from the policy.

Apply the same max substitution that turned policy iteration into value iteration. Since the new greedy policy’s state value is $\max_{a'}Q(s',a')$, define targets

$$
y_i=r(s_i,a_i)+\gamma\max_{a_i'}Q_\theta(s_i',a_i')
$$

and regress $Q_\theta(s_i,a_i)$ onto $y_i$. This is **fitted Q-iteration**. It does not need to simulate unobserved current actions. The stored $(s_i,a_i)$ says which first action was actually taken; maximizing that current action would destroy the advantage of using the observed transition. Only the next action is maximized inside the learned Q-function.

The lecturer clarified terminology:

- Policy evaluation estimates V or Q for a fixed policy.
- Policy iteration alternates evaluation and improvement to find a better policy.
- Value iteration or Q-value iteration substitutes the greedy policy directly into the backup, eliminating the explicit improvement step.

### Source reconciliation

Slide 13’s central contrast is that a Q-function can test next actions without simulating their transitions. The transcript Q&A explicitly rejects an additional max over the sampled current action.

### Additional explanation

The Q-network acts as a learned counterfactual evaluator over actions at a given next state. The environment supplies one transition for the logged current action; the network supplies comparable predictions for all candidate continuation actions.

For stochastic-policy evaluation the continuation is $\mathbb E_{a'\sim\pi(\cdot\mid s')}Q^\pi(s',a')$; writing $Q^\pi(s',\pi(s'))$ assumes a deterministic policy. Fitted Q-iteration instead uses a maximum to seek $Q^*$. During learning, its current approximation need not equal the actual return of its current greedy policy.

## 8. Function-approximation caveats and Q-network forms

**Transcript coverage:** lines 2536-2757

### What the lecturer said - transcript only

Tabular value iteration and tabular Q-value iteration are fixed-point algorithms guaranteed to converge under their standard assumptions, though this is nontrivial. Fitted value and fitted Q-iteration with neural networks are not generally valid fixed-point algorithms and are not guaranteed to converge at all. The lecturer called this the dark side of the theory and deferred a fuller discussion to Friday.

Fitted Q-iteration has only one learned network. V is not separately represented; it is derived as $V(s)=\max_a Q(s,a)$.

There are two common Q-network interfaces. A general network consumes both $(s,a)$ and outputs one scalar. With a small discrete action set, a more convenient network consumes only $s$ and outputs a vector containing one Q-value per action. The vector form makes enumeration and argmax immediate. This implementation detail does not change the mathematics.

The method can train from off-policy transitions and avoids a high-variance policy gradient. Its major negative is the missing nonlinear-function-approximation convergence guarantee.

### Source reconciliation

Slide 13 explicitly lists “works even for off-policy samples” and “only one network” as benefits and “no convergence guarantees for non-linear function approximation” as the cost.

### Additional explanation

The guarantee is lost not merely because a neural network has approximation error. Bootstrapped targets depend on the network being updated, so projection by regression and the Bellman optimality operator interact in ways that need not contract.

Fitted methods are still interpretable as approximate fixed-point iterations. What is absent is a general contraction or convergence guarantee for the projected update, not the existence of a fixed-point formulation.

## 9. The full fitted Q-iteration recipe and its schedules

**Transcript coverage:** lines 2758-3096

### What the lecturer said - transcript only

Full fitted Q-iteration has three stages:

1. Collect $N$ transitions using a chosen collection policy.
2. For a fixed target network snapshot, compute $y_i=r_i+\gamma\max_{a'}Q_\theta(s_i',a')$.
3. Take $S$ gradient steps fitting Q to those targets, and repeat target recomputation/regression for $K$ iterations before or alongside more collection.

The dataset size, collection policy, number of fitted iterations, and number of gradient steps are algorithm parameters. Different schedules yield substantially different algorithms. The policy used for action selection is implicitly the greedy argmax of Q, unless an exploratory collection policy is substituted.

Minimizing each regression fully can be extremely expensive. If Q begins near zero, one Bellman-target refresh propagates roughly one reward step, the next propagates another, and a thousand-step dependency may require roughly a thousand target refreshes. If each regression takes a million optimizer steps, this creates an infeasible total. Fewer gradient steps with more frequent target recomputation may propagate reward faster, but recomputing either too often or too rarely can be harmful. This becomes a delicate hyperparameter studied in practical deep Q-learning.

The lecturer corrected his own handwritten policy symbol during Q&A and reiterated that inference runs Q, obtains one value per action, and chooses the argmax; Q is effectively a roundabout representation of the deterministic policy.

### Source reconciliation

Slide 14 labels the main degrees of freedom as dataset size $N$, collection policy, fitted iterations $K$, and gradient steps $S$.

### Additional explanation

Modern implementations often make the target snapshot explicit as a separate target network. That stabilizing device is not introduced as part of this lecture’s algorithm; it is deferred to the next lecture.

## Part IV - Back to Q-learning

## 10. Why fitted Q-iteration is off-policy

**Transcript coverage:** lines 3097-3225

### What the lecturer said - transcript only

Fitted Q-iteration can use data from other policies because the max in the target evaluates the current greedy policy at $s_i'$, while the sampled environment transition remains valid after conditioning on its logged $(s_i,a_i)$. Given a state and action, the distribution of the next state does not depend on which broader behavior policy selected that action.

One can therefore view learning less as a strict collect-update alternation and more as continually “turning the crank” over a transition dataset, occasionally adding new experience to the bucket.

### Source reconciliation

Slide 16 circles the logged $(s_i,a_i)$ and the max at $s_i'$ separately to show why behavior-policy identity drops out after conditioning.

### Additional explanation

Off-policy validity still requires coverage: the dataset must contain useful transitions for the state-action regions whose Q-values matter. The Bellman target cannot recover dynamics for pairs absent from data merely because the algebra is off-policy.

## 11. Bellman error and what fitted Q-iteration optimizes

**Transcript coverage:** lines 3226-3462

### What the lecturer said - transcript only

The fitted regression hides that its target also depends on Q. Substituting $y$ yields the Bellman error

$$
\mathcal E(\theta)=\frac12
\mathbb E_{(s,a,s')\sim\beta}
\left[
\left(
Q_\theta(s,a)-
\left(r(s,a)+\gamma\max_{a'}Q_\theta(s',a')\right)
\right)^2
\right],
$$

where $\beta$ denotes the replay-buffer data distribution. This is not ordinary supervised learning with a target independent of the predictor.

If the error is exactly zero for all relevant state-action pairs, Q satisfies the Bellman optimality equation and is the optimal Q-function $Q^*$; the associated greedy policy $\pi^*$ maximizes reward. In the tabular case, reaching this fixed point is tractable under the usual assumptions.

With samples and function approximation, the error will not be exactly zero. Unlike policy gradient, fitted Q-iteration does not directly optimize current-policy return. The mapping from a nonzero Bellman error to greedy-policy quality is complex and was deferred to later theory. Most clean guarantees are lost when tables are replaced by neural networks.

In Q&A, the lecturer explained that $\beta$ was merely notation for sampling from the replay buffer.

### Source reconciliation

Slide 17 labels $\mathcal E$ as an error and states that zero error implies the optimal Bellman equation. It also warns in red that most guarantees disappear beyond the tabular case.

### Additional explanation

The fitted algorithm normally treats the right-hand Q inside each target as fixed during a regression phase. Directly differentiating both sides of the displayed residual would define a different optimization procedure from the alternating target-and-fit recipe.

Distinguish the **expected Bellman residual** from the displayed **sampled squared TD error**. Let $Y_Q=r+\gamma m\max_{a'}Q(s',a')$ and $(\mathcal T^*Q)(s,a)=\mathbb E[Y_Q\mid s,a]$. Then

$$
\mathbb E[(Q(s,a)-Y_Q)^2\mid s,a]
=(Q(s,a)-(\mathcal T^*Q)(s,a))^2
+\operatorname{Var}(Y_Q\mid s,a).
$$

Even $Q^*$ can have positive sampled squared error because rewards and transitions are stochastic. Zero **expected Bellman residual at every state-action pair** identifies $Q^*$ in the discounted finite MDP; zero loss on a finite dataset does not. For example, a terminal reward equally likely to be 0 or 2 has $Q^*=1$ but expected squared TD error 1.

Differentiating the sampled squared error through the target also differentiates its variance term. To estimate the gradient of the squared expected residual without that extra term generally requires two independent successor samples conditional on the same $(s,a)$, the double-sampling problem. Semi-gradient Q-learning avoids that optimization problem by treating the target as fixed.

## 12. Online Watkins Q-learning as a limiting schedule

**Transcript coverage:** lines 3463-3599

### What the lecturer said - transcript only

The fitted-Q template includes many algorithms depending on $N$, $K$, and $S$. Setting collection batch size, target refresh, and gradient update counts to one gives classic online Q-learning, also called Watkins Q-learning: take one action, observe $(s_i,a_i,r_i,s_i')$, immediately form

$$
y_i=r_i+\gamma\max_{a'}Q_\theta(s_i',a'),
$$

and immediately take one squared-error gradient step on that transition. The lecturer emphasized that the textbook online algorithm is a special case of the general fitted recipe rather than an unrelated method.

### Source reconciliation

Slide 18 places full fitted Q-iteration above the one-transition online update and marks the collection policy as an open choice because the method is off-policy.

### Additional explanation

For a scalar prediction, a gradient-descent step can be written

$$
\theta\leftarrow\theta-\alpha
\nabla_\theta Q_\theta(s_i,a_i)
\left(Q_\theta(s_i,a_i)-y_i\right).
$$

Its target is still bootstrapped and nonstationary even though only one transition is used.

The tabular convergence theorem needs a finite stationary MDP, bounded rewards, $\gamma<1$, infinitely many visits to every relevant state-action pair, and per-pair learning rates satisfying $\sum_k\alpha_k=\infty$ and $\sum_k\alpha_k^2<\infty$. Its update is $Q(s,a)\leftarrow Q(s,a)+\alpha[y-Q(s,a)]$. Replacing the table with shared neural-network parameters does not inherit this theorem.

## 13. Exploration: greedy, epsilon-greedy, and Boltzmann policies

**Transcript coverage:** lines 3600-4218

### What the lecturer said - transcript only

Because Q-learning is off-policy, its behavior during learning need not be the final greedy policy. At deployment, choose an argmax action. During learning, a randomly initialized Q-function may slightly prefer one action; an always-greedy deterministic policy can then repeat that action, collect nearly identical data, and fail to cover the MDP. Adding randomness to data collection is the **exploration problem**.

The simplest default is epsilon-greedy. With probability $1-\epsilon$, take the greedy action; with the remaining probability choose uniformly among the other $|\mathcal A|-1$ actions:

$$
\pi_\epsilon(a\mid s)=
\begin{cases}
1-\epsilon,&a\text{ is the selected greedy action},\\
\epsilon/(|\mathcal A|-1),&\text{otherwise}.
\end{cases}
$$

It is easy to implement, better than always taking argmax, and a good starting point for Homework 3, although it mixes the current policy with random behavior and can be inefficient.

Boltzmann exploration instead uses a softmax over Q-values:

$$
\pi_{\mathrm{B}}(a\mid s)\propto\exp(Q(s,a)).
$$

It prefers high-Q actions while assigning nonzero probability to alternatives. An action with value near negative one billion receives negligible probability, so Boltzmann exploration is useful when clearly catastrophic actions should be avoided. Epsilon-greedy is more uniform among non-greedy actions. Choice depends on the task; the lecturer suggested epsilon-greedy for simplicity and Boltzmann for large action spaces where some actions are easily recognized as very bad.

Both are **action-space exploration**. The real goal is state-space coverage. In a grid world, random left/right/up/down actions create a random walk, whereas useful exploration would deliberately reach varied states. If reaching a distant state requires a particular sequence of length $L$, its probability under naive action randomization may decay exponentially, as a student observed. More advanced exploration methods address this by changing rewards to encourage novel states; those methods were deferred to a later exploration lecture.

A question about the state of the art prompted the same answer: substantially better but more complex methods exist, and simple random-action strategies are only a baseline.

### Source reconciliation

Slide 19 writes the two behavior policies and explicitly postpones deeper exploration. The epsilon-greedy formula assumes a unique greedy action; ties need a convention not supplied in the lecture.

### Additional explanation

A temperature parameter is often included as $\exp(Q(s,a)/T)$, but the slide and transcript present the proportional softmax without introducing $T$. That extra parameter is therefore not attributed to the lecturer here.

The displayed epsilon-greedy convention requires at least two actions and one selected greedy action. Another common convention explores uniformly over **all** actions; then the selected greedy action has probability $1-\epsilon+\epsilon/|\mathcal A|$ and every other action has $\epsilon/|\mathcal A|$. State which convention is used.

Boltzmann exploration trusts the learned value scale. An incorrectly low value can suppress a useful action, and an incorrectly high value can favor a dangerous one; the softmax is not a safety guarantee. Compute it with numerically stable logits rather than exponentiating very large raw values.

## 14. Replay-buffer Q-learning and the instability warning

**Transcript coverage:** lines 4219-4357

### What the lecturer said - transcript only

Putting the lecture back together, replay-buffer Q-learning proceeds as follows:

1. Take a step using an exploratory policy, such as epsilon-greedy or Boltzmann, and append the transition to replay.
2. Sample a mini-batch from the replay buffer.
3. Compute $y_i=r_i+\gamma\max_{a'}Q_\theta(s_i',a')$.
4. Regress $Q_\theta(s_i,a_i)$ onto the targets, usually for a small number of gradient steps.
5. Repeat.

Despite its appealing simplicity, coding exactly this version will almost certainly fail with deep networks. Function approximation means the procedure need not minimize Bellman error to zero and has no general convergence guarantee. The next lecture would introduce the practical stabilization tricks without which deep Q-learning generally does not work. Used correctly, those additions make Q-learning simple and effective.

### Source reconciliation

Slide 20 gives the three-line replay algorithm but places two prominent warnings beside it: the naive implementation will almost certainly not work, and the next lecture will explain how to make it work. These warnings are essential parts of the source, not optional caveats.

### Additional explanation

The central unresolved issue is that the network both generates and chases moving bootstrap targets while the sampled distribution is off-policy. The following lecture’s engineering techniques are part of the practical algorithm, not cosmetic improvements.

The lecture's strong failure warning is practical advice, not a theorem that every unstabilized neural Q-learning run must fail. Conversely, replay and target networks improve practice without guaranteeing convergence in every environment.

## Consolidated takeaways

1. With a small discrete action space, an actor can be replaced by greedy action selection from Q.
2. The resulting critic-only replay algorithm is Q-learning.
3. Policy iteration alternates policy evaluation and greedy improvement.
4. Dynamic programming assumes small enumerable spaces and a known transition model, so it needs no samples.
5. Value iteration substitutes the greedy improvement directly into the value backup.
6. Neural fitted value iteration replaces table writes with regression but still needs counterfactual transitions for every candidate action.
7. Fitted Q-iteration needs only observed current-action transitions because Q predicts candidate next actions directly.
8. Tabular optimality backups have fixed-point guarantees; nonlinear fitted approximations generally do not.
9. A discrete Q-network can output all action values in one forward pass.
10. Fitted Q-iteration’s data size, target-refresh schedule, and regression-step count define materially different algorithms.
11. The method is off-policy because the logged transition remains valid conditional on $(s,a)$ while the max evaluates the new greedy continuation.
12. Zero expected Bellman residual everywhere identifies $Q^*$; stochastic sampled TD errors can remain nonzero even at $Q^*$.
13. Online Watkins Q-learning is the one-sample, one-target, one-gradient-step limit of fitted Q-iteration.
14. Exploration must prevent a premature deterministic greedy policy from restricting data coverage.
15. Epsilon-greedy and Boltzmann exploration randomize actions but do not solve hard state-space exploration.
16. Naive neural replay-buffer Q-learning can be unstable; the next lecture develops practical stabilization methods.

## Key equations

### Greedy policy induced by Q

$$
\pi_Q(s)\in\arg\max_a Q(s,a),
\qquad
V_Q(s)=\max_a Q(s,a).
$$

### Bellman expectation backup

$$
V^\pi(s)=
\mathbb E_{a\sim\pi(\cdot\mid s),\,s'\sim p(\cdot\mid s,a)}
[r(s,a)+\gamma V^\pi(s')].
$$

### Bellman optimality backup for V

$$
V(s)\leftarrow
\max_a\left[r(s,a)+\gamma
\mathbb E_{s'\sim p(\cdot\mid s,a)}V(s')\right].
$$

### Fitted Q-iteration target and loss

$$
y_i=r_i+\gamma m_i\max_{a'}Q_{\bar\theta}(s_i',a'),
\qquad
L(\theta)=\frac{1}{2N}\sum_{i=1}^{N}
(Q_\theta(s_i,a_i)-y_i)^2,
$$

where $\bar\theta$ denotes the parameter snapshot used to form fixed, detached regression targets during one fit, and $m_i$ masks true terminals.

### Bellman optimality equation

$$
Q^*(s,a)=r(s,a)+\gamma
\mathbb E_{s'\sim p(\cdot\mid s,a)}
\left[\max_{a'}Q^*(s',a')\right].
$$

### Online Q-learning update

$$
\theta\leftarrow\theta-\alpha
\nabla_\theta Q_\theta(s_i,a_i)
\left(Q_\theta(s_i,a_i)-
\left[r_i+\gamma m_i\max_{a'}Q_\theta(s_i',a')\right]
\right).
$$

### Simple exploratory policies

$$
\pi_\epsilon(a\mid s)=
\begin{cases}
1-\epsilon,&a\text{ is the selected greedy action},\\
\epsilon/(|\mathcal A|-1),&\text{otherwise},
\end{cases}
$$

$$
\pi_{\mathrm B}(a\mid s)=
\frac{\exp(Q(s,a))}{\sum_{a'}\exp(Q(s,a'))}.
$$

## Glossary

- **Bellman error:** discrepancy between a value prediction and a Bellman backup target.
- **Bellman expectation equation:** recursive value relationship for a fixed policy.
- **Bellman optimality equation:** recursive relationship containing a max over next actions.
- **Boltzmann exploration:** sampling actions from a softmax of Q-values.
- **Curse of dimensionality:** exponential growth of a tabular state space with representation dimension.
- **Dynamic programming:** exact planning with an enumerable state-action space and known transition model.
- **Epsilon-greedy:** mostly taking a greedy action while reserving probability $\epsilon$ for alternatives.
- **Fitted Q-iteration:** alternating max-Q target construction and supervised Q regression.
- **Fitted value iteration:** value iteration using function approximation and supervised regression.
- **Policy evaluation:** computing V or Q for a fixed policy.
- **Policy improvement:** replacing a policy by one greedy with respect to its evaluated Q-function.
- **Policy iteration:** alternating policy evaluation and policy improvement.
- **Q-learning:** a value-based method that learns Q and obtains its policy by greedy action selection.
- **Replay distribution:** the empirical distribution $\beta$ represented by stored transitions.
- **Tabular representation:** storing a separate scalar or action index for every discrete state or state-action pair.
- **Value iteration:** repeatedly applying Bellman optimality backups, interleaving greedy improvement with partial evaluation.
- **Watkins Q-learning:** the classic online one-transition form of Q-learning.

## Self-check questions

1. Under what action-space condition can the actor be removed from actor-critic?
2. Why does the expected next Q-value become a max under the implicit policy?
3. Why can Q-learning remain off-policy even with replay capacity one?
4. In what limited sense can Q be viewed as a predictive world model?
5. Distinguish policy evaluation, policy improvement, and policy iteration.
6. Which assumptions allow tabular dynamic programming to avoid sampling entirely?
7. Why does $\gamma<1$ prevent the bounded-reward value iteration from exploding?
8. Why does a fixed policy have a linear-system solution while optimal control does not?
9. How does value iteration eliminate an explicit policy table?
10. What is the curse of dimensionality in the image-state example?
11. Why does fitted value iteration still require known transition dynamics?
12. Why does fitted Q-iteration maximize only over the next action, not the sampled current action?
13. Which fixed-point guarantee is lost with nonlinear function approximation?
14. Compare the two common Q-network input/output interfaces.
15. What do $N$, $K$, and $S$ control in the fitted-Q recipe?
16. Why can fitting each target set completely be computationally wasteful?
17. What conditional-independence argument makes fitted Q-iteration off-policy?
18. What does zero Bellman error imply, and why is nonzero error harder to interpret?
19. How is online Watkins Q-learning obtained from fitted Q-iteration?
20. Why can a deterministic greedy policy get trapped by random initialization?
21. Compare epsilon-greedy and Boltzmann treatment of a catastrophically low-Q action.
22. Why does randomizing actions fail to guarantee broad state coverage?
23. Why did the lecturer warn that the final replay-buffer algorithm would almost certainly fail if implemented naively with a deep network?

## Source coverage checklist

| Topic | Inclusive range | Coverage status |
|---:|---:|---|
| 1 | 1-502 | Accounted for |
| 2 | 503-786 | Accounted for |
| 3 | 787-969 | Accounted for |
| 4 | 970-1467 | Accounted for |
| 5 | 1468-1770 | Accounted for |
| 6 | 1771-2148 | Accounted for |
| 7 | 2149-2535 | Accounted for; current-action max confusion resolved explicitly |
| 8 | 2536-2757 | Accounted for; convergence caveat preserved |
| 9 | 2758-3096 | Accounted for; handwritten policy-symbol correction disclosed |
| 10 | 3097-3225 | Accounted for |
| 11 | 3226-3462 | Accounted for |
| 12 | 3463-3599 | Accounted for |
| 13 | 3600-4218 | Accounted for |
| 14 | 4219-4357 | Accounted for; naive deep-Q failure warning preserved |

**Coverage result:** All 4,357 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. All 20 slide pages were visually inspected; the policy/value-iteration diagrams, fitted-Q targets, Bellman-error slide, exploration formulas, and final warning were checked against the spoken account.
