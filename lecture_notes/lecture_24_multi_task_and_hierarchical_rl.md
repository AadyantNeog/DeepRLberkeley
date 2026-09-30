---
title: "Lecture 24 - Multi-Task and Hierarchical Reinforcement Learning"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 24
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 24, Multi-Task and Hierarchical RL.txt"
source_slides: "../lectures/Lecture 24 - Multi-Task RL.pdf"
transcript_lines: 8131
slide_pages: 53
status: "source-incomplete"
source_qualifier: "unspoken-slides-isolated"
---

# Lecture 24: Multi-Task and Hierarchical Reinforcement Learning

> **Source warning:** The transcript ends at line 8131 in the middle of the lecturer's final sentence about the guest lectures. The technical lecture had already been adjourned, but the missing tail cannot be reconstructed. Slides 41-53 introduce meta-learning and were not reached in the supplied recording; they are isolated in a slide-only appendix.

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Schedule and the shift to generalist models | lines 1-250 |
| 2 | Multi-task RL: definition, contexts, and efficiency | lines 251-808 |
| 3 | Transfer and where prior knowledge can live | lines 809-1309 |
| 4 | Reducing multi-task RL to ordinary RL | lines 1310-1567 |
| 5 | Ray interference and the relabeling remedy | lines 1568-2200 |
| 6 | Task distributions and relabeling as data augmentation | lines 2201-2467 |
| 7 | Goal-conditioned RL and sparse goal rewards | lines 2468-3031 |
| 8 | Hindsight relabeling, off-policy learning, and goal geometry | lines 3032-3823 |
| 9 | Intermission and goal-conditioned values as models | lines 3824-4369 |
| 10 | Successor representations | lines 4370-4966 |
| 11 | Successor features | lines 4967-5791 |
| 12 | Rapid transfer and generalized policy improvement | lines 5792-6544 |
| 13 | Continuous successor representations by classification | lines 6545-7366 |
| 14 | Options and practical hierarchical RL | lines 7367-8131 |

## 1. Schedule and the shift to generalist models

**Transcript coverage:** lines 1-250

### What the lecturer said - transcript only

This was the last standard deep technical lecture. The following Wednesday would be a shorter, quiz-free course-perspectives lecture. Kevin Frans would give a guest lecture on RL and language models the next Friday, and Sihong Park would discuss offline-to-online algorithms the Wednesday after. Both guest sessions would begin at 9:00 rather than the usual 8:00 and last about 50 minutes. The final Friday had no class so students could work on final projects.

The lecture concerned multi-task and hierarchical RL. The lecturer began with machine learning's shift from separate task-specific models to general-purpose foundation models. Earlier workflows trained a separate architecture and carefully labeled dataset for segmentation, classification, captioning, visual question answering, and so on. Current systems instead pretrain one large model on diverse data and objectives, then prompt, adapt, or fine-tune it for a target task.

Large-scale next-token prediction, image-text contrastive learning, CLIP, Segment Anything, and large language models were examples. A target task may still need examples, but it no longer needs enough carefully curated data to train an entire model from scratch. The shared foundation supplies reusable knowledge.

### Source reconciliation

Slides 1-4 show the schedule-independent title, specialist-to-foundation-model transition, and examples including CLIP, SAM, and LLMs. The transcript supplies all end-of-semester logistics.

### Additional explanation

The central analogy is that multi-task RL aims to pretrain reusable *decision-making* structure—policies, values, dynamics, representations, or skills—rather than only predictive representations.

## 2. Multi-task RL: definition, contexts, and efficiency

**Transcript coverage:** lines 251-808

### What the lecturer said - transcript only

Generalist models are now common in daily use; the other common learned system people encounter is a specialized contextual bandit serving advertisements. The key generalist idea is multi-task learning: data from many tasks reduces how much labeled or interactive data any one task requires.

Ordinary RL maximizes expected return under a policy trajectory distribution. Multi-task RL adds a task context $\omega$:

$$
\max_\theta
\mathbb E_{\omega\sim p(\omega)}
\mathbb E_{\tau\sim p_\theta(\tau\mid\omega)}
\left[r_\omega(\tau)\right],
$$

where

$$
p_\theta(\tau\mid\omega)
=p(s_1)\prod_{t=1}^{H}
\pi_\theta(a_t\mid s_t,\omega)
p(s_{t+1}\mid s_t,a_t).
$$

For a finite training set, the outer expectation is a task-weighted sum. A task could also change dynamics or state/action spaces, depending on how the environment is defined; model-free algorithms often avoid writing those dynamics explicitly.

The context representation is domain-specific. It may be a walking direction, puck target, natural-language command, goal state, Atari game index, or destination on a map. A language-model prompt is a task context, so Homework 4 already implemented multi-task RL.

Two hoped-for benefits are:

- **joint training efficiency:** experience and features shared across related tasks accelerate each task;
- **transfer:** the shared model provides a strong initialization for a new related task.

Sharing is easier to imagine for related language or mathematics problems than for unrelated activities such as pancake flipping and dog walking. Model capacity and training speed are domain-dependent; a larger shared model can require more computation per step yet fewer optimization steps or better parallel utilization.

### Source reconciliation

Slides 5-6 display the context-conditioned objective and the efficiency/pretraining motivation. The transcript provides the broad list of context types and the Homework 4 connection.

### Additional explanation

The outer distribution $p(\omega)$ is part of the problem specification: it expresses which tasks matter and how frequently. Multi-task performance is therefore not one scalar notion independent of task weighting.

The shared-transition formulation assumes tasks differ in reward/context but have compatible state/action semantics and dynamics. General multi-task MDPs can also have task-dependent initial states or transition kernels; retain those dependencies when defining the augmented process.

## 3. Transfer and where prior knowledge can live

**Transcript coverage:** lines 809-1309

### What the lecturer said - transcript only

Multi-task experience can explain why a person finds Montezuma's Revenge easier than a fresh RL agent. The pixels contain semantic cues learned elsewhere: skulls suggest danger, keys suggest doors, and video-game conventions identify relevant objects. That knowledge is outside the single game's MDP reward signal. Watching *Indiana Jones* or playing other games may supply useful priors.

Prior task structure can be stored and transferred in several places:

- **policy and Q-function:** jointly pretrain actor and critic, then continue training on the target task;
- **dynamics model:** reuse predictive structure across related environments even when optimal actions differ;
- **features or hidden states:** use a pretrained encoder and train only a smaller task-specific control head.

For image tasks, a pretrained vision encoder is often an obvious empirical choice. Language tasks almost certainly begin from a pretrained language model, which is itself a pretrained policy. Which mechanism is best depends on the domain.

### Source reconciliation

Slides 7-8 use the Atari example and enumerate Q-function, policy, model, and representation transfer. The transcript expands the semantic-prior argument.

### Additional explanation

Representation transfer changes what the learner can perceive easily; policy/value transfer changes what it already prefers; model transfer changes what futures it can predict. These are complementary rather than mutually exclusive.

## 4. Reducing multi-task RL to ordinary RL

**Transcript coverage:** lines 1310-1567

### What the lecturer said - transcript only

Multi-task RL is not a fundamentally different formal problem. Fuse the task MDPs into one joint MDP: sample $\omega$ once at the beginning and append it to every state. The augmented state $(s,\omega)$ is Markov because the agent knows which MDP it occupies, while the sampled context remains fixed for the trajectory.

Consequently, any ordinary RL algorithm becomes a multi-task algorithm if its policy and value functions receive the context and each rollout uses the appropriate task reward. The basic recipe is:

1. choose an input representation for $\omega$;
2. feed it into $\pi_\theta(a\mid s,\omega)$ and the critic;
3. sample a task for each trajectory and use $r_\omega$.

There is nevertheless a full lecture because multi-task optimization is often harder, and because the fixed, known context has special structure that task-aware methods can exploit.

### Source reconciliation

Slides 10-11 show the joint-MDP reduction and the three-step implementation. The transcript adds why a fixed context is structurally different from an ordinary, opaque state component.

### Additional explanation

The reduction proves representational equivalence, not equal learning difficulty. An algorithm can solve the augmented MDP in principle while still suffering severe optimization interference in practice.

Observed task context that remains fixed within a rollout can be included in the Markov state. When task identity is hidden, a policy needs inference from experience; this is a partially observed/meta-RL problem rather than an ordinary fully observed context-conditioned MDP.

## 5. Ray interference and the relabeling remedy

**Transcript coverage:** lines 1568-2200

### What the lecturer said - transcript only

In joint RL training, tasks often progress at different times rather than improving together. The lecturer cited Schaul and colleagues' “ray interference” work. When the agent first succeeds on an easy task, exploiting that reward can suppress exploration needed by harder tasks. Training all Atari games with one context-conditioned Q-network may produce competence on one game and failure on the rest—or failure everywhere.

For a Gaussian policy, the intuition is explicit. High reward far from the current mean tends to expand variance; high reward near the mean contracts it. Success on one task tells shared parameters to narrow around its discovered behavior, while unsolved tasks still need broader exploration. More generally, shared updates can favor the first rewarding task at the expense of others. The lecturer compared it to allocating effort toward the one of five jobs that begins paying more.

The same architecture trained by supervised behavioral cloning did not show the same plateau/interference pattern in the cited experiment. Easy tasks can provide a curriculum when structure transfers, but can also interfere when their exploitation dominates.

One special multi-task remedy is **relabeling**. If all task reward functions can be evaluated on stored transitions, experience collected under one context can be assigned rewards under other contexts. An off-policy replay transition stores $(s_i,a_i,s_i',\omega_i)$. During training, resample a context $\widetilde\omega_i$, recompute its reward, and update the actor and critic as if the transition had been collected for that task.

This is data augmentation: no new environment transition is collected, but the same observation can teach what is good or bad for multiple tasks.

### Source reconciliation

Slides 12-13 show the RL-versus-supervised interference plots and the context-resampling actor-critic template.

### Additional explanation

Relabeling works only when a stored physical transition is meaningful across tasks. It is natural for different destinations in one world, but a Breakout frame is not a valid Montezuma's Revenge transition.

Relabeling must recompute task-dependent **termination**, as well as reward, and retain time/horizon information where needed. Shared observations do not suffice when tasks change physical dynamics. A logged action remains fixed data; changing the goal does not make that action an on-policy sample for the new goal.

## 6. Task distributions and relabeling as data augmentation

**Transcript coverage:** lines 2201-2467

### What the lecturer said - transcript only

The recomputed reward may be low, which is useful: a transition can be labeled as bad for a task it did not accomplish. The resampling distribution deserves care.

The nominal $p(\omega)$ comes from the test problem. If math tasks occur ten times as often as literature tasks, optimizing the stated objective suggests training on math ten times as often. Matching the test distribution is a safe default.

One may deliberately deviate to create a curriculum. If the agent already masters common math tasks but struggles with rare literature tasks, oversampling the latter could improve learning dynamics even though it differs from evaluation frequencies. A relabeling proposal may also depend on the original task, $p(\widetilde\omega\mid\omega)$, so a northward trajectory is relabeled with nearby directions rather than arbitrary tasks.

This choice is rich and problem-dependent. Relabeling extracts counterfactual task information from existing data, but only when cross-task reward evaluation is semantically valid.

### Source reconciliation

Slide 13 provides the context-resampling actor-critic template. The transcript adds the distinction between matching the evaluation distribution, deliberately oversampling weak tasks, and conditioning a relabeling proposal on the original task.

### Additional explanation

There are two distributions to distinguish: the evaluation task distribution defines what “good overall performance” means, while the training/curriculum distribution controls optimization. Changing the second can help optimization, but without correction it changes task weighting in the objective. To estimate the original task expectation from $q(\omega)$, use $p(\omega)/q(\omega)$ where support permits. A curriculum may deliberately optimize a different weighting; that is distinct from an unbiased estimate of the evaluation objective.

## 7. Goal-conditioned RL and sparse goal rewards

**Transcript coverage:** lines 2468-3031

### What the lecturer said - transcript only

Goal-conditioned RL is a particularly natural multi-task setting. Learn

$$
\pi(a\mid s,g)
$$

to reach any commanded goal. The goal may be a full state or a subset, such as position without velocity or orientation.

Common rewards are

$$
r(s,a,g)=\mathbf 1[s=g],
$$

$$
r(s,a,g)=\mathbf 1[\lVert s-g\rVert\le\epsilon],
$$

or a step cost

$$
r(s,a,g)=-\mathbf 1[s\ne g].
$$

Exact equality is unsuitable for floating-point continuous states, so a tolerance is common. Adding a constant preserves policy ordering for fixed-length returns, or continuing discounted returns where it adds the same $c/(1-\gamma)$. It can change ordering when reward accumulation stops at a policy-dependent terminal time. Success rewards and step costs are therefore not generally equivalent.

With $\gamma=1$ and termination at the goal, the step-cost value has the appealing interpretation of negative expected time-to-goal. One reward definition covers every goal, and a neural policy may generalize zero-shot to unseen but structurally similar goals.

Goal-conditioned RL is usually harder than single-goal RL and often needs relabeling. It also cannot express every task. “Reach a destination while avoiding a forbidden region” depends on the path, not only the final state. A language problem may correspond to a set of acceptable answers rather than one goal state.

### Source reconciliation

Slides 14-16 define goal-conditioned policies, list reward choices, and cite early goal-reaching, universal value-function, hindsight-replay, and C-learning papers.

### Additional explanation

Goal-conditioned value functions are universal value functions: one network represents values for a family of reward functions indexed by $g$.

For a precise hitting-time convention, charge $-1$ for each action taken before the first arrival at $g$, including the action that enters it, then terminate. If $\tau_g$ is the number of those actions and $\mathbb E_\pi\tau_g<\infty$, then $V^\pi(s,g)=-\mathbb E_\pi\tau_g$ at $\gamma=1$. A one-time $+1$ reward on goal entry instead maximizes success probability at $\gamma=1$ and is indifferent to arrival time among certain successes. Unreachable goals can give infinite hitting time; finite-horizon truncation or $\gamma<1$ changes the exact interpretation.

## 8. Hindsight relabeling, off-policy learning, and goal geometry

**Transcript coverage:** lines 3032-3823

### What the lecturer said - transcript only

Sparse distant goals may be almost impossible to hit randomly. Yet every trajectory reaches *some* states. Treat failed attempts for $g^*$ as successful experience for an achieved future goal. As the learned reachable region expands, generalization supports slightly farther goals, and repeated learning can eventually reach the original target without a separate novelty bonus.

An online goal-conditioned actor-critic stores transitions with the commanded goal. When sampling replay data, it may use the achieved future state $g_{\mathrm{reached}}$ half the time and retain $g_{\mathrm{commanded}}$ half the time. Always replacing with achieved goals creates hindsight bias: it makes the training task distribution look easier than deployment and removes the negative signal from failing the intended task.

The achieved goal can be the trajectory's terminal state or a future state sampled a geometrically distributed number of steps after the current transition. The commanded goal is exactly what was originally supplied to the behavior policy. Many other mixing strategies exist.

Relabeling requires an off-policy algorithm. Changing $g$ changes the augmented state and therefore changes which actions the current context-conditioned policy would have taken. The stored actions were generated under the old commanded goal, so REINFORCE is not valid on the relabeled data. Q-learning and off-policy actor-critic can use it; PPO can tolerate only limited off-policy deviation.

With $r=-1$ until termination, $\gamma=1$, and $V(g,g)=0$,

$$
V^\pi(s,g)=
\begin{cases}
-1+\mathbb E_{a\sim\pi(\cdot\mid s,g),\,s'\sim P}[V^\pi(s',g)],&s\ne g,\\
0,&s=g.
\end{cases}
$$

The negative **optimal** value, under the conditions explained below, is minimum expected time to goal and behaves like a directed distance. It is not symmetric because MDP dynamics need not be reversible—an egg can be scrambled but not necessarily unscrambled—so it is a quasimetric. It obeys a triangle-style inequality:

$$
V^*(s,g)\ge V^*(s,w)+V^*(w,g)
$$

under the negative-distance sign convention (equivalently, positive time-to-goal obeys the usual $d(s,g)\le d(s,w)+d(w,g)$).

This geometry supports waypoint planning with short-horizon goal values and auxiliary consistency losses. The lecturer traced the idea through Leslie Kaelbling's 1993 *Learning to Achieve Goals*, while noting Kaelbling had recently pointed him to an even earlier Bellman source.

### Source reconciliation

Slides 17-19 show hindsight relabeling, the off-policy loop, and the triangle relation. Slide 19 writes the inequality using its value convention; the spoken interpretation is preserved with the sign made explicit here.

### Additional explanation

Hindsight relabeling densifies reward without fabricating transitions. It changes only which reward function evaluates the already observed dynamics.

**Two different hindsight biases.** Changing the goal frequencies changes the training objective. Separately, choosing a goal from the realized future can select favorable transition noise: although physical dynamics are goal-independent, the relabeled dataset may satisfy $p_{\mathcal D}(s'\mid s,a,g)\ne P(s'\mid s,a)$. For example, relabeling only successful lottery outcomes as the desired goal hides losing outcomes. Mixing commanded goals back in does not generally eliminate this stochastic selection bias, and merely choosing an off-policy algorithm does not correct it. See [USHER](https://arxiv.org/abs/2207.01115).

**When the triangle inequality holds.** The displayed policy-evaluation equation alone does not imply it. Define $d^*(s,g)=\inf_\pi\mathbb E_\pi\tau_g=-V^*(s,g)$ for a proper shortest-path problem. If a policy can reach waypoint $w$ and then switch to a policy reaching $g$, that feasible concatenation proves $d^*(s,g)\le d^*(s,w)+d^*(w,g)$. Use infinite values for unreachable pairs; finite distances need finite hitting-time assumptions. An arbitrary fixed goal-conditioned policy can be needlessly slow to $g$ and need not satisfy this inequality. Approximate neural values also need not satisfy it.

The optimal backup is $V^*(s,g)=\max_a[-1+\mathbb E_P V^*(s',g)]$ for $s\ne g$, with $V^*(g,g)=0$. This differs from the policy-averaged evaluation backup above.

## 9. Intermission and goal-conditioned values as models

**Transcript coverage:** lines 3824-4369

### What the lecturer said - transcript only

The class paused at 9:00 and resumed around 9:06.

Successor representations and successor features were introduced as a classic conceptual connection among goal-conditioned RL, model-free values, and model-based prediction. They were presented more as conceptual grounding than as the most immediately practical deep-RL tool.

The lecturer's bold claim was that a goal-conditioned value resembles a model. A transition model predicts future states; a time-to-goal value tells how easily every state can be reached and can define a reachability graph for planning.

To make this relationship precise, fix a policy $\pi$ and consider policy evaluation. RL repeatedly generates samples, evaluates the policy, and improves it; a model's immediate role here is to support evaluation. For a state-only reward,

$$
V^\pi(s_t)
=\sum_{t'=t}^{\infty}\gamma^{t'-t}
\mathbb E_{s_{t'}\sim p_\pi(\cdot\mid s_t)}[r(s_{t'})].
$$

The notation $s_{t'}$ means a state at an arbitrary future time, not merely the next state.

### Source reconciliation

Slides 20-23 mark the intermission and begin the policy-evaluation rearrangement. The transcript states the conceptual purpose before performing the algebra.

### Additional explanation

A one-step model answers “what happens next?” A successor representation answers “which states will occupy my discounted future if I continue with this policy?”

A reachability/time-to-goal function is model-like, but generally does not identify the full one-step stochastic transition law. Successor predictions below are specific to a continuation policy; they are not arbitrary-action world simulators.

## 10. Successor representations

**Transcript coverage:** lines 4370-4966

### What the lecturer said - transcript only

Expanding the expectation, exchanging the time and state sums, and grouping each state's discounted visitation gives

$$
V^\pi(s_t)
=\sum_s
\left[
\sum_{t'=t}^{\infty}
\gamma^{t'-t}p_\pi(s_{t'}=s\mid s_t)
\right]r(s).
$$

The bracketed weights sum to $1/(1-\gamma)$. Normalize them to define the discounted future-state distribution

$$
\mu_i^\pi(s_t)
=(1-\gamma)
\sum_{t'=t}^{\infty}
\gamma^{t'-t}p_\pi(s_{t'}=i\mid s_t).
$$

This can be interpreted as sampling a future time from a geometric stopping process: at each step the agent survives with probability $\gamma$ and ceases with probability $1-\gamma$. Nearer future states receive more weight.

For discrete states, $\mu^\pi(s)$ is a vector and

$$
V^\pi(s)=\frac{1}{1-\gamma}\,\mu^\pi(s)^\top r.
$$

It is the **successor representation**, introduced by Peter Dayan in 1993. It obeys a TD-style recursion:

$$
\mu_i^\pi(s_t)
=(1-\gamma)\mathbf 1[s_t=i]
+\gamma\mathbb E_{a_t\sim\pi,s_{t+1}\sim P}
[\mu_i^\pi(s_{t+1})].
$$

This is a vectorized family of Bellman backups with a pseudo-reward for visiting each possible state.

### Source reconciliation

Slides 24-26 display the normalization, inner-product value identity, and recursion. The lecturer called out a slide typo during this derivation; the consistent normalized equations are used above.

### Additional explanation

The representation separates dynamics and policy from reward. Once $\mu^\pi$ is known, evaluating the same policy under a new state reward is only an inner product.

**Normalization convention.** Standard successor counts use $M^\pi(s,i)=\mathbb E_\pi\sum_{k\ge0}\gamma^k\mathbf1[S_{t+k}=i]$, whereas this section uses $\mu^\pi=(1-\gamma)M^\pi$. Accordingly $V^\pi=M^\pi r=\mu^\pi r/(1-\gamma)$. The normalized geometric time has $\Pr(K=k)=(1-\gamma)\gamma^k$ for $k=0,1,\ldots$, so it includes the current state. With terminal states, include a zero-reward absorbing continuation to retain total mass one; dropping post-terminal visits instead gives a sub-probability measure.

## 11. Successor features

**Transcript coverage:** lines 4967-5791

### What the lecturer said - transcript only

Full successor representations are often harder than an ordinary scalar value: they need one output per state, become enormous in large spaces, and are not directly defined as finite vectors for continuous states.

Successor features compress them. Choose features $\phi(s)\in\mathbb R^d$ such that the reward is linear:

$$
r(s)=\phi(s)^\top w.
$$

Define the discounted future feature expectation

$$
\psi^\pi(s_t)
=\mathbb E_\pi\left[
\sum_{k=0}^{\infty}\gamma^k\phi(s_{t+k})
\,\middle|\,s_t
\right].
$$

Then

$$
V^\pi(s)=\psi^\pi(s)^\top w.
$$

The same weights that express reward from features express value from successor features. If there are far fewer features than states, this is much cheaper. Rewards are often simple enough that features need only represent salient events—key pickup, door opening, contact with a hazard—while ignoring most pixel configurations.

The TD recursion is

$$
\psi^\pi(s_t)
=\phi(s_t)+\gamma
\mathbb E_{a_t\sim\pi,s_{t+1}\sim P}
[\psi^\pi(s_{t+1})].
$$

A state-action version is

$$
\psi^\pi(s_t,a_t)
=\phi(s_t)
+\gamma\mathbb E_{s_{t+1},a_{t+1}\sim\pi}
[\psi^\pi(s_{t+1},a_{t+1})],
$$

and recovers

$$
Q^\pi(s,a)=\psi^\pi(s,a)^\top w.
$$

With one-hot $\phi(s)$, these unnormalized successor features equal the standard successor counts $M^\pi$, or $\mu^\pi/(1-\gamma)$ under the preceding section's normalized convention. The lecture assumes the designer supplies or learns the features; the framework itself does not choose them.

### Source reconciliation

Slides 27-28 derive the linear reward/value relation and show state and state-action feature backups. The transcript supplies concrete Atari-style feature examples.

### Additional explanation

Successor features factor transfer into two questions: what events can happen under a policy, represented by $\psi^\pi$, and how valuable are those events for the current task, represented by $w$.

The identities require reward to be linear in the chosen features. For transition rewards use matching features $\phi(s,a,s')$ and expected discounted sums of those same features. If reward approximation has uniform error at most $\eta$, its fixed-policy discounted value error is at most $\eta/(1-\gamma)$. Reusing successor features without reevaluation assumes the same dynamics and continuation policy; changing reward is the part handled by the new weights.

## 12. Rapid transfer and generalized policy improvement

**Transcript coverage:** lines 5792-6544

### What the lecturer said - transcript only

One use is rapid Q-function recovery:

1. pretrain $\psi^\pi(s,a)$ without knowing the final reward;
2. collect a small set of reward samples;
3. solve linear regression for $w$ in $r\approx\phi^\top w$;
4. recover $Q^\pi(s,a)\approx\psi^\pi(s,a)^\top w$ and act greedily.

This resembles model-based pretraining: learn world/policy knowledge before the task reward, then adapt rapidly once reward examples arrive. A linear reward need not be severely restrictive if features form a rich superset of the simple events rewards depend on.

However,

$$
\pi'(s)=\arg\max_a\psi^\pi(s,a)^\top w
$$

is only one step of policy improvement from the policy whose successor features were learned. It is better than blindly following $\pi$ but is not generally optimal; full policy iteration would need to evaluate and improve repeatedly.

A stronger method pretrains successor features $\psi^{\pi_k}$ for many policies. After fitting $w$, recover every policy's Q-function and choose

$$
\pi'(s)
=\arg\max_a\max_k
\psi^{\pi_k}(s,a)^\top w.
$$

This generalized policy improvement takes the best action offered by the best stored policy separately in each state. It can outperform every single member by composing where each is strong, although it is optimal only if the library is sufficiently rich. Continuous policy/skill spaces can replace the finite library if the maximization is tractable.

The method does not specify which base policies or features to learn. Skill discovery and goal-conditioned RL can generate policies; VAEs or other representation-learning methods can generate features. The reward-agnostic iterative pretraining is separated from rapid task-time fitting.

### Source reconciliation

Slides 29-30 contrast one-step improvement with the multi-policy maximum and cite Barreto et al.'s successor-feature transfer work.

### Additional explanation

Generalized policy improvement is a library-based analogue of transfer: pretraining supplies multiple predictive “what I can accomplish” maps, and a new reward selects among them.

With exact Q-values, shared dynamics, bounded discounted rewards, and exact action maximization, GPI guarantees a policy at least as good as each base policy for the new reward. It evaluates all candidate actions under all base continuation policies; it does not merely choose among the actions each base actor emits. Approximate successor features, fitted reward weights, and approximate maximization weaken this guarantee. Transfer quality depends on both feature coverage and the policy library. See [Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312).

## 13. Continuous successor representations by classification

**Transcript coverage:** lines 6545-7366

### What the lecturer said - transcript only

For a continuous state, $\mathbf 1[s=i]$ is almost surely zero and point probabilities are not the desired density. A norm ball defines a legitimate probability but changes the object being estimated. Probability densities can exceed one and need different treatment.

Reframe the successor density as binary classification. Let $F=1$ mean a candidate $s_{\mathrm{future}}$ is a future state reached after $(s_t,a_t)$ under $\pi$. Construct:

- positives from actual discounted future states of that transition;
- negatives from states sampled from the overall replay marginal $p(s)$.

The Bayes-optimal classifier is

$$
p(F=1\mid s,a,s_f)
=\frac{p^\pi(s_f\mid s,a)}
{p^\pi(s_f\mid s,a)+p(s_f)}.
$$

Similarly,

$$
p(F=0\mid s,a,s_f)
=\frac{p(s_f)}
{p^\pi(s_f\mid s,a)+p(s_f)}.
$$

Taking odds gives the density ratio

$$
\frac{p(F=1\mid s,a,s_f)}
{p(F=0\mid s,a,s_f)}
=\frac{p^\pi(s_f\mid s,a)}{p(s_f)}.
$$

Multiplying by $p(s_f)$ recovers the successor density. If only relative action values or an argmax is needed, the state-only marginal may be an irrelevant proportional factor and can sometimes be omitted.

To match the discounted successor definition, sample the positive future time with geometric weighting: the next step proportional to $\gamma$, two steps ahead to $\gamma^2$, and so on. Uniform future sampling is possible but defines a different future-visitation object.

Training is ordinary cross-entropy on replay positives and negatives. The lecturer associated this recursive-classification perspective with C-learning and described an on-policy form, while noting that off-policy variants can also be derived.

### Source reconciliation

Slides 31-33 present continuous successor classification, the Bayes odds identity, and the C-learning algorithm.

### Additional explanation

This is contrastive density-ratio estimation: classification is used because estimating a normalized high-dimensional continuous density directly is difficult, whereas distinguishing conditional futures from marginal states can be practical.

The displayed odds assume equal positive/negative class priors and compatible support. With positive prior $\rho$, odds instead equal $\frac{\rho}{1-\rho}\frac{p^\pi(s_f\mid s,a)}{p(s_f)}$. Omit the marginal only when comparing actions for the **same** candidate goal; it cannot generally be dropped when integrating across goals to reconstruct an arbitrary reward value.

Distinguish a future-only successor measure from the current-inclusive one:

$$
\mu_+^\pi(\cdot\mid s,a)
=(1-\gamma)\sum_{k=0}^\infty\gamma^k
\Pr_\pi(S_{t+1+k}\in\cdot\mid s,a),
\qquad
\mu^\pi(\cdot\mid s,a)=(1-\gamma)\delta_s+\gamma\mu_+^\pi(\cdot\mid s,a).
$$

Sampling strictly later positives estimates the first object after normalization. The full measure includes a current-state atom, which in continuous spaces is not an ordinary density; later dynamics can also be singular. The density-ratio derivation requires a common dominating measure/support. Future samples from old replay trajectories follow their behavior continuation, not automatically the current $\pi$; a recursive off-policy method needs a derived correction/backup.

## 14. Options and practical hierarchical RL

**Transcript coverage:** lines 7367-8131

### What the lecturer said - transcript only

Classical hierarchical RL uses **options**, temporally extended mini-policies. An option contains:

- an initiation set $I_o$ where it may begin;
- an option policy $\pi_o(a\mid s)$;
- a termination rule or set $\beta_o$ where control returns to the high level.

Augment the action space with primitive actions and options. If option $o_t$ acts for $h$ steps, a Q-learning-style backup becomes

$$
Q(s_t,o_t)
\leftarrow
\sum_{k=0}^{h-1}\gamma^k r(s_{t+k})
+\gamma^h\max_{o'}Q(s_{t+h},o').
$$

Long options shorten the high-level effective horizon. A 10,000-step game with 1,000-step coherent options can look like a roughly ten-decision high-level problem, and good options can dramatically simplify exploration.

The unresolved issue is where options come from. If supplied, good options are clearly beneficial. If option discovery is learned end-to-end with the high-level policy, it may be as hard as learning a flat policy and erase much of the advantage. The Option-Critic architecture was cited as a classic attempt to learn both.

Practical hierarchies simplify the formalism. Use one context-conditioned low-level policy rather than a separate policy per option; allow it to start anywhere; omit learned termination; and switch the context every fixed $K$ steps. Train the low-level policy with skill discovery, goal-conditioned RL, or another multi-task method. Train a high-level policy whose “action” is the low-level skill, goal, or language command.

Goal hierarchies can solve long-horizon navigation/manipulation by having the high level choose subgoals for a strong goal-conditioned policy. The lecturer cited work by Nachum and colleagues on long-horizon ant puzzle tasks. Language hierarchies can let a low level produce one utterance while a high level reasons over multiple conversational turns; the ARCHER work was mentioned as an example.

The lecturer adjourned after 10:00, said the following Wednesday would be his final lecture, and strongly encouraged attendance at the two guest lectures. The supplied transcript then ends mid-sentence—“So when you're doing”—so no further closing statement can be recovered.

### Source reconciliation

Slides 34-40 cover classical options, fixed-duration skill hierarchies, goals, and language. Slides 41-53 on meta-learning were not reached in the transcript.

### Additional explanation

Fixed-duration context switching yields a high-level MDP at decision boundaries with discount $\gamma^K$. Variable option durations give a semi-Markov decision process. Its practical appeal is modularity: each layer can use familiar algorithms.

The displayed option backup is a target, not an exact replacement for a stochastic Q estimate. For observed duration $h$, use

$$
y=\sum_{k=0}^{h-1}\gamma^k r_{t+k}
+\gamma^h m\max_{o'\in\mathcal O(s_{t+h})}Q_{\mathrm{target}}(s_{t+h},o'),
\qquad Q(s_t,o_t)\leftarrow Q(s_t,o_t)+\eta[y-Q(s_t,o_t)].
$$

Here $m=0$ at environment termination; ending an option alone does not terminate the environment. Maximize over options available at that state. In the general options formalism, $\beta_o(s)$ is a termination probability, not necessarily a deterministic set. If the low-level policies keep changing, the high-level transition law changes too, so old option replay is not automatically valid. Good options can help, but arbitrary discovered options offer no guaranteed efficiency gain. Retaining primitive actions preserves the original policy possibilities.

## Slide-only appendix: meta-learning material not reached

Slides 41-53 were visually inspected but not discussed in the supplied recording. They introduce:

- supervised meta-learning as learning a function that consumes a small training set;
- generic learning versus meta-learning across datasets/tasks;
- the meta-RL objective over a distribution of MDPs;
- task context inferred from experience rather than supplied directly;
- recurrent policies whose hidden state persists across episodes of a meta-episode;
- emergent exploration when the recurrent agent maximizes total meta-episode return;
- meta-RL as a partially observed RL problem whose latent variable is task identity.

These concepts are not attributed to the lecturer's spoken Lecture 24 account.

## Consolidated takeaways

- Multi-task RL is ordinary RL on a context-augmented joint MDP, but shared optimization can create severe task interference.
- Generalist training can improve efficiency and transfer knowledge through policies, values, models, or representations.
- Context relabeling is a task-aware data-augmentation method; its validity depends on shared transition semantics and off-policy learning.
- Goal-conditioned RL turns arbitrary achieved future states into training goals, greatly densifying sparse reward.
- Optimal proper hitting-time values define directed distances; arbitrary policy values and approximate learned values need not obey the triangle inequality.
- Successor representations predict discounted future occupancy; successor features factor task reward from policy-dependent future features.
- A library of successor features supports generalized policy improvement on a new linear reward.
- Continuous successor densities can be learned through future-versus-marginal classification.
- Hierarchical RL reduces high-level horizon when coherent reusable options or subgoal policies are already available; discovering them remains the central difficulty.

## Key equations

1. **Multi-task RL**

   $$
   \max_\theta\mathbb E_{\omega\sim p(\omega),\tau\sim p_\theta(\tau\mid\omega)}[r_\omega(\tau)].
   $$

2. **Goal-conditioned step-cost value**

   $$
   V^*(s,g)=\max_a\{-1+\mathbb E_P[V^*(s',g)\mid s,a]\}\quad(s\ne g),
   \qquad V^*(g,g)=0.
   $$

3. **Successor representation**

   $$
   \mu_i^\pi(s)
   =(1-\gamma)\sum_{k=0}^{\infty}\gamma^k
   p_\pi(s_{t+k}=i\mid s_t=s).
   $$

4. **Successor feature factorization**

   $$
   r(s)=\phi(s)^\top w
   \quad\Longrightarrow\quad
   Q^\pi(s,a)=\psi^\pi(s,a)^\top w.
   $$

5. **Generalized policy improvement**

   $$
   \pi'(s)=\arg\max_a\max_k\psi^{\pi_k}(s,a)^\top w.
   $$

6. **Successor-density odds**

   $$
   \frac{p(F=1\mid s,a,s_f)}{p(F=0\mid s,a,s_f)}
   =\frac{p^\pi(s_f\mid s,a)}{p(s_f)}.
   $$

   These odds assume balanced classes and the specified future-sampling distribution.

7. **Option target and sampled update**

   $$
   y=\sum_{k=0}^{h-1}\gamma^k r_{t+k}
   +\gamma^h m\max_{o'\in\mathcal O(s_{t+h})}Q_{\mathrm{target}}(s_{t+h},o'),
   \qquad Q(s_t,o_t)\leftarrow Q(s_t,o_t)+\eta[y-Q(s_t,o_t)].
   $$

## Glossary

- **Task context:** Variable identifying which reward, goal, instruction, or environment is active.
- **Ray interference:** Multi-task learning pattern in which progress on one task suppresses progress on others.
- **Relabeling:** Re-evaluating stored experience under a different task context.
- **Goal-conditioned policy:** Policy that receives a desired goal in addition to current state.
- **Hindsight relabeling:** Treating achieved future states as alternative goals for failed trajectories.
- **Quasimetric:** Nonnegative directed distance satisfying a triangle inequality, without requiring symmetry; unreachable pairs may need infinite distance.
- **Successor representation:** Expected discounted future-state counts for a policy; multiplying by $1-\gamma$ gives the normalized occupancy convention used here.
- **Successor feature:** Discounted future expectation of a feature vector.
- **Generalized policy improvement:** Statewise choice of the best action suggested by multiple evaluated policies.
- **Density-ratio classification:** Estimating a conditional density relative to a marginal through classifier odds.
- **Option:** Temporally extended action defined by initiation, internal policy, and termination.
- **High-level policy:** Policy that selects skills, goals, instructions, or options rather than primitive actions.

## Self-check questions

1. How does context augmentation reduce multi-task RL to a single MDP?
2. Why can the first easy task learned interfere with harder tasks?
3. Distinguish the evaluation task distribution from a curriculum distribution.
4. When is context relabeling semantically invalid?
5. Why retain some commanded-goal failures, and why does mixing them in not generally remove stochastic hindsight bias?
6. Why must relabeled goal learning be off-policy?
7. Under which optimality and hitting-time assumptions does a goal-conditioned value define a directed distance?
8. Derive $V^\pi(s)=\mu^\pi(s)^\top r/(1-\gamma)$.
9. Why do the same weights $w$ recover value from successor features?
10. Why is greedy improvement from one policy's successor features not generally optimal?
11. How does a binary classifier recover a successor-density ratio?
12. When does hierarchy truly shorten the learning problem rather than merely move its difficulty to option discovery?

## Source coverage checklist

- [x] All supplied transcript lines 1-8131 are mapped exactly once in increasing, non-overlapping ranges.
- [x] Schedule, questions, intermission, optional historical notes, and truncated closing are retained.
- [x] Transcript, slide-only meta-learning material, and added explanation are explicitly separated.
- [x] All 53 slide pages were rendered and visually inspected.
- [x] Displayed mathematics uses Markdown-compatible LaTeX delimiters.
- [x] The incomplete final sentence is explicitly marked rather than reconstructed.

**Coverage result:** All 8,131 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. The truncated closing and unspoken meta-learning slides are isolated rather than reconstructed.
