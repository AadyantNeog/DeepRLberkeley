---
title: "Lecture 5 - Policy Gradients"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 5
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 5, Policy Gradients.txt"
source_slides: "../lectures/Lecture 05 - Policy Gradients.pdf"
transcript_lines: 4843
slide_pages: 23
status: "complete"
---

# Lecture 5: Policy Gradients

## Lecture map

**Reading conventions.** This lecture initially uses a fixed finite horizon $H$ and an undiscounted sum of rewards. For a finite-horizon MDP, values and policies may depend on time; include the remaining time in the state when suppressing that index. The transcript sections preserve the lecture account; the additional explanations supply qualifications needed to use the formulas correctly.

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Opening logistics and why begin with policy gradients | lines 1-267 |
| 2 | RL objective and the sample-model-improve loop | lines 268-501 |
| 3 | Why ordinary backpropagation through experience fails | lines 502-852 |
| 4 | Likelihood-ratio derivation of the policy gradient | lines 853-1267 |
| 5 | Removing unknown dynamics from the gradient | lines 1268-1519 |
| 6 | Monte Carlo estimation and REINFORCE | lines 1520-2028 |
| 7 | Policy gradients as reward-weighted maximum likelihood | lines 2029-2361 |
| 8 | Trial-and-error interpretation and temporal credit assignment | lines 2362-2604 |
| 9 | Policy gradients under partial observability | lines 2605-2840 |
| 10 | Preset-position chess and high variance | lines 2841-3543 |
| 11 | Constant baselines for variance reduction | lines 3544-4107 |
| 12 | Causality and reward-to-go | lines 4108-4407 |
| 13 | The weighted supervised pseudo-loss | lines 4408-4779 |
| 14 | Practical tuning guidance and closing | lines 4780-4843 |

## 1. Opening logistics and why begin with policy gradients

**Transcript coverage:** lines 1-267

### What the lecturer said - transcript only

After resolving a display problem by unplugging and reconnecting it three times, the lecturer joked that the procedure was “not Markovian.” He then announced that the exam would be Tuesday, April 14 at 7:00 p.m. Students should mark the time, remain in Berkeley, and be available in person; the room was less important than the time.

Policy gradients would be the first reinforcement-learning algorithm taught. When the lecturer began teaching the course about ten years earlier, this ordering was unusual because other methods were more common. Policy gradients have since become one of the most widely used classes of RL methods.

The lecturer asked which RL-trained systems students were likely to encounter regularly. Language models were one answer and are likely to have been trained with policy gradients. Recommendation systems used for advertisements, Netflix, and YouTube were the other; those systems use RL and may personalize from a user profile, but typically are not trained with policy gradients.

### Source reconciliation

Slides 2-3 visually frame policy gradients as the lecture’s first RL algorithm and use language-model post-training as the prominent contemporary example. The exam announcement and opening troubleshooting appear only in the transcript.

### Additional explanation

The motivating examples distinguish an application domain from a particular optimizer. “Uses RL” does not imply “uses policy gradients”: different settings favor policy-gradient, value-based, bandit, or other estimators.

## 2. RL objective and the sample-model-improve loop

**Transcript coverage:** lines 268-501

### What the lecturer said - transcript only

The objective from the previous lecture is the total reward accumulated along a trajectory, averaged under the trajectory distribution. That distribution is over sequences of states and actions, is determined jointly by the MDP and the policy, and is denoted $p_\theta(\tau)$ to emphasize its dependence on the policy parameters $\theta$. By the probability chain rule and the MDP’s conditional independences,

$$
p_\theta(\tau)=p(s_1)\prod_{t=1}^{H}\pi_\theta(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t).
$$

The lecturer again divided RL algorithms into three parts: generate samples, analyze or model those samples, and use that result to improve the policy. Policy gradients have an especially simple analysis step: sum the rewards in each sampled trajectory to estimate the policy’s average reward. The improvement step computes the gradient of that objective and performs gradient ascent.

Writing the objective as $J(\theta)$, policy evaluation can be done by running the policy in the world, adding the rewards in each sampled trajectory, and averaging the returns. This Monte Carlo average is an unbiased estimate of expected return; good, bad, and intermediate trajectories all contribute their observed totals.

### Source reconciliation

Slides 4-6 show the three-stage anatomy visually and write both the expectation and its Monte Carlo approximation. The deck uses a finite-horizon trajectory with time indices matching the formulas above.

### Additional explanation

If $R(\tau)=\sum_{t=1}^{H}r(s_t,a_t)$, then

$$
J(\theta)=\mathbb E_{\tau\sim p_\theta}[R(\tau)],
\qquad
\widehat J(\theta)=\frac{1}{N}\sum_{i=1}^{N}R(\tau^{(i)}).
$$

The challenge is not estimating $J(\theta)$ from rollouts; it is differentiating an expectation whose sampling distribution itself depends on $\theta$.

## 3. Why ordinary backpropagation through experience fails

**Transcript coverage:** lines 502-852

### What the lecturer said - transcript only

A tempting implementation is to run the policy in a simulator, collect perhaps twenty trajectories, compute their differentiable rewards in PyTorch, and backpropagate the average with respect to $\theta$. In the rollout loop, however, each policy action is passed to a dynamical system—possibly a simulator and possibly the real world—to produce the next state. PyTorch does not know how that black box operates. It may differentiate a reward with respect to recorded states and actions, but it cannot trace how $\theta$ produced later states through unknown dynamics; as represented in the computation graph, the desired derivative is therefore zero or disconnected.

The obstacle is not that enumerating all trajectories is intractable, because the proposal already uses samples. It also remains even if the reward is a simple differentiable quadratic. The missing derivative is through the environment.

A simulator can technically be written as a differentiable program, including entirely in PyTorch. People do this, but the resulting gradients can be numerically ill behaved, a subject deferred to model-based RL. The homework simulators do not provide such differentiation, and a physical real-world system cannot provide an ordinary autograd graph.

Two general approaches differentiate stochastic computations. Pathwise derivatives are ordinary backpropagation through a differentiable computation. The alternative used here is the likelihood-ratio method, which can differentiate expectations without differentiating through the sampled world.

### Source reconciliation

Slide 7 depicts the broken computational path through the environment. Slide 8 names the two estimator families “pathwise derivative” and “likelihood ratio.”

### Additional explanation

The environment still affects the eventual gradient estimate through which trajectories it generates. What disappears is the need to know or differentiate its transition function.

A black-box environment blocks the full pathwise derivative through the rollout. This does not mean that every possible derivative is zero: a differentiable reward could supply a direct action derivative, for example, while still omitting the action's effect on later states and rewards. The score-function derivation accounts for the full change in the trajectory distribution without differentiating the environment.

## 4. Likelihood-ratio derivation of the policy gradient

**Transcript coverage:** lines 853-1267

### What the lecturer said - transcript only

The lecturer introduced shorthand $R(\tau)=\sum_t r(s_t,a_t)$ and used $p_\theta(\tau)$ for the complete trajectory distribution. For continuous trajectories, expected return is an integral; for discrete trajectories the analogous operation is a sum:

$$
J(\theta)=\int p_\theta(\tau)R(\tau)\,d\tau.
$$

Only $p_\theta(\tau)$ depends on $\theta$, so the derivative is moved inside the integral. The key log-derivative identity follows from the derivative of a logarithm:

$$
p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)
=p_\theta(\tau)\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}
=\nabla_\theta p_\theta(\tau).
$$

Substitution converts the derivative into an expectation:

$$
\nabla_\theta J(\theta)
=\mathbb E_{\tau\sim p_\theta}
\left[\nabla_\theta\log p_\theta(\tau)R(\tau)\right].
$$

This is valuable because expectations under $p_\theta$ can be estimated simply by sampling trajectories from the current policy. A student connected the form to maximum likelihood; another asked how to compute $\nabla\log p_\theta(\tau)$, which the lecturer identified as the remaining step.

The lecturer called this the single most important slide of the day, strongly hinted that it would appear on the exam, and described the expression as perhaps the most important formula in reinforcement learning and sometimes as the policy-gradient theorem.

In response to a question about regularity or boundedness conditions for interchanging the derivative and integral, the lecturer answered that expectation and differentiation are linear and therefore can always be swapped.

### Source reconciliation

The transcript’s “can always swap” statement is retained as spoken. The corresponding derivation is displayed on slides 9-10.

### Additional explanation

Mathematically, linearity alone does not guarantee that differentiation may pass through an integral. Standard sufficient conditions include differentiability plus an integrable dominating bound (or a theorem with comparable regularity assumptions). Policy-gradient presentations commonly leave these technical conditions implicit.

The identity is also called the **score-function estimator**. It replaces a derivative through a sample with a derivative of that sample’s log probability.

The usual derivation also assumes suitable common support and that rewards have no explicit dependence on $\theta$ other than through the trajectory. An explicitly parameter-dependent reward would contribute its own derivative.

## 5. Removing unknown dynamics from the gradient

**Transcript coverage:** lines 1268-1519

### What the lecturer said - transcript only

The remaining task was to simplify $\log p_\theta(\tau)$. Taking the logarithm of the trajectory-factorization product turns it into a sum:

$$
\log p_\theta(\tau)=\log p(s_1)+
\sum_{t=1}^{H}\left[
\log\pi_\theta(a_t\mid s_t)+
\log p(s_{t+1}\mid s_t,a_t)
\right].
$$

The initial-state probability does not depend on the policy, so its derivative with respect to $\theta$ is zero. The environment transition probabilities also do not depend on which policy parameters are used, so their derivatives are zero. Only the policy terms remain:

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_{t=1}^{H}\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

Expanding $R(\tau)$ yields

$$
\nabla_\theta J(\theta)=
\mathbb E_{\tau\sim p_\theta}
\left[
\left(\sum_{t=1}^{H}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right)
\left(\sum_{t=1}^{H}r(s_t,a_t)\right)
\right].
$$

Every quantity inside this expectation is available from rollouts and the policy network. In Q&A, the lecturer clarified that the reward sum came from the definition of $R(\tau)$ and that taking the logarithm was required precisely because the previous identity contained $\log p_\theta(\tau)$. He called this the second most important slide.

### Source reconciliation

Slides 11-12 visually strike out the initial-state and transition-model score terms, emphasizing that the dynamics need not be known.

### Additional explanation

The simplification requires that environment dynamics and the initial-state distribution do not themselves change as a differentiable function of the policy parameter. Their statistical effect remains encoded in the sampled states.

Inside $\nabla_\theta\log p_\theta(\tau)$, the realized states and actions are held fixed. This is why the dynamics terms have zero parameter derivative even though changing the policy will change which states occur on the next rollout.

## 6. Monte Carlo estimation and REINFORCE

**Transcript coverage:** lines 1520-2028

### What the lecturer said - transcript only

As with objective evaluation, the expectation in the gradient can be replaced by an average over trajectories sampled from the current policy. This gives an unbiased gradient estimate. A small gradient-ascent step should then improve the policy locally:

$$
\widehat{\nabla_\theta J(\theta)}=
\frac{1}{N}\sum_{i=1}^{N}
\left(\sum_{t=1}^{H}\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})\right)
\left(\sum_{t=1}^{H}r(s_t^{(i)},a_t^{(i)})\right).
$$

The resulting algorithm is REINFORCE: sample trajectories from the policy, compute the estimator, and update in the gradient direction. The lecturer noted that “REINFORCE” is a tortured acronym whose expansion he did not remember. This is the first complete RL algorithm introduced in the course and is already sufficient to implement a minimal version.

Several questions clarified sampling. The probability density does not need to be inserted explicitly when averaging samples drawn from that very distribution, just as a random sample of people can estimate average age. Explicit enumeration would require trajectory probabilities, which are usually unknown; the method instead assumes access to interactions that generate trajectories. The world’s influence remains in the samples, and those samples must be produced by the policy whose gradient is being estimated.

The lecturer confirmed that a reward with widely varying magnitudes makes the algorithm less stable and previewed high variance as the reason. The policy score terms are neural-network log-probability gradients, such as cross-entropy gradients. When a student noticed the slide omitted $1/N$, the lecturer explicitly identified it as a typo and observed that this constant omission does not change the gradient direction.

### Source reconciliation

Slide 13’s displayed estimator omits the factor $1/N$. This is not silently repaired: the transcript calls it a typo at lines 2000-2028. The normalized Monte Carlo estimator is written above, while the omission and its limited effect are documented here.

### Additional explanation

Multiplying an estimator by $N$ changes its scale, so the learning rate must absorb that factor even though the direction is unchanged. “Unbiased” applies to the normalized estimator under on-policy sampling and the derivation’s assumptions.

## 7. Policy gradients as reward-weighted maximum likelihood

**Transcript coverage:** lines 2029-2361

### What the lecturer said - transcript only

The policy-gradient expression resembles the gradient of the maximum-likelihood objective used for imitation learning. Maximum likelihood increases the probabilities of all dataset actions. Policy gradient applies the same kind of log-probability gradient but multiplies every trajectory’s terms by its total reward, increasing high-reward samples and decreasing negative-reward samples.

For continuous control, the policy may be a multivariate Gaussian whose neural network outputs the mean. Its log probability is proportional to a covariance-weighted squared difference between the network output and the sampled action. Differentiating it produces that error, weighted by inverse covariance, backpropagated through the neural network, and the policy-gradient estimator then multiplies it by return. For discrete actions, the network produces logits, softmax gives the action distribution, and automatic differentiation supplies the corresponding cross-entropy/log-probability gradient.

A student asked whether reward must always be negative. The lecturer deferred the full answer but highlighted the real issue: adding one million to every reward leaves the underlying task unchanged while making all multipliers positive. This sensitivity motivates a later correction. Another student pointed out a missing constant in the Gaussian expression; the lecturer agreed that a constant was omitted and said it did not matter much there.

### Source reconciliation

Slides 14-15 place maximum likelihood and policy gradient side by side and show the Gaussian score. The spoken correction about an omitted constant is preserved rather than silently rewriting the slide derivation as exact.

### Additional explanation

For a fixed covariance $\Sigma$ and mean $\mu_\theta(s)$,

$$
\log\pi_\theta(a\mid s)=
-\frac12(a-\mu_\theta(s))^\top\Sigma^{-1}(a-\mu_\theta(s))+C,
$$

where $C$ is constant with respect to the mean parameters when $\Sigma$ is fixed. Autograd should be applied to the sampled action’s log probability, with the return treated as a weight rather than differentiated through the rollout.

For the mean, $\nabla_\mu\log\pi(a\mid s)=\Sigma^{-1}(a-\mu)$. A positive return weight therefore pulls the mean toward the sampled action; a negative weight pushes it away. If covariance is learned, the Gaussian normalization term, including $-\tfrac12\log\det\Sigma$, is no longer constant.

## 8. Trial-and-error interpretation and temporal credit assignment

**Transcript coverage:** lines 2362-2604

### What the lecturer said - transcript only

Using shorthand in which $\nabla\log\pi(\tau)$ denotes the sum of per-time-step policy scores, maximum likelihood makes every sampled example more probable. Policy gradient makes a trajectory more or less probable according to its return. If good outcomes have positive rewards and bad outcomes negative rewards, the intuitive result is to make good trials more likely and bad trials less likely. This formalizes trial-and-error learning: generate trials, judge them by accumulated reward, and update their probabilities accordingly.

The interpretation is imperfect when even bad trajectories have positive rewards, an issue deferred to baselines. It also assigns a single total return to all decisions in a trajectory. If a trajectory is half good and half bad, its entire probability is increased or decreased according to the net outcome. In chess, a genuinely good move followed by a later blunder may therefore receive a negative update. The lecturer agreed that a better method would separately strengthen good portions and weaken bad portions.

### Source reconciliation

The corresponding slides contrast maximum-likelihood updates with return-weighted trial updates and motivate time-local credit. The transcript supplies the positive-reward qualification and the chess Q&A.

### Additional explanation

This is the **temporal credit-assignment problem**: a delayed outcome says how the episode ended but not which earlier decisions caused that outcome. Reward-to-go will remove rewards that occurred before an action, while actor-critic methods later estimate more localized credit.

## 9. Policy gradients under partial observability

**Transcript coverage:** lines 2605-2840

### What the lecturer said - transcript only

Under partial observability, the policy receives observations rather than Markov states, so it has the form $\pi_\theta(a_t\mid o_t)$. Asked how policy gradient must change, a student suggested either differentiating through an unrolled observation-generating process if possible or simply using the same algorithm. The lecturer endorsed the latter.

The derivation never actually used the Markov property of the policy input. Even if a transition or observation probability depends on the entire preceding history, its score with respect to $\theta$ is still zero so long as that process is independent of the policy parameters. Therefore the same policy-gradient expression applies to observations. Rewards must still be obtainable somehow; subject to that requirement, no change is needed.

### Source reconciliation

Slide 16 explicitly replaces states by observations and retains the same estimator. The transcript’s claim concerns the likelihood-ratio derivation, not whether a memoryless observation policy is expressive enough to act optimally.

### Additional explanation

The estimator remains valid, but partial observability can still hurt performance because $o_t$ may omit information needed to choose an action. A recurrent policy or explicit history can improve the policy class without changing the score-function principle.

## 10. Preset-position chess and high variance

**Transcript coverage:** lines 2841-3543

### What the lecturer said - transcript only

The basic algorithm is correct but numerically troublesome. To build intuition, the lecturer introduced preset-position chess. Instead of always starting from the standard board, each game begins from a randomly selected difficult configuration. The reward is $+1$ for a win and $-1$ for a loss. Ideally, good moves would receive positive multipliers and bad moves negative ones.

Several confounders prevent that on individual trajectories:

- An easy initial position may produce a win despite mediocre play, positively reinforcing every move.
- A difficult initial position may produce a loss despite good play, negatively reinforcing every move.
- A strong opening followed by a later mistake makes the earlier good moves receive a negative multiplier.
- A bad move may receive a positive multiplier if the opponent later makes a random mistake.

These errors average out with sufficiently many samples. In the infinite-sample limit, good moves occur slightly more often in successful trajectories and the expectation is the true gradient. The problem is that a finite sample may require many trajectories before this signal dominates noise.

The random starting positions were introduced purely as a didactic construction and are sampled from an initial-state distribution. Every action from every sampled trajectory participates in the update, and in the basic estimator every time step in one trajectory receives the same total-return multiplier.

Variance measures how far a finite-sample estimate deviates from its true expectation. An estimator has high variance when that deviation is large on average at practical sample counts, even though it converges to the correct expectation with infinitely many samples. The lucky starts, later blunders, and opponent mistakes are concrete manifestations of this one general problem.

In Q&A, the lecturer said that a richer chess evaluation could instead change the reward function. A value function was not exactly the same object, but value functions would later provide a way to reduce policy-gradient variance. Shorter trajectories also reduce variance; however, if reward arrives only at the end, shortening the trajectory enough to omit that outcome removes the learning signal. Improving policy gradients therefore largely means variance reduction so that stable gradients can be obtained from fewer samples.

### Source reconciliation

Slides 17-18 show the preset-position example and list the three principal mis-crediting cases. The spoken Q&A adds the didactic nature of the randomized setup and the terminal-reward caveat.

### Additional explanation

The estimator can be unbiased and still be practically unusable: unbiasedness constrains its average over repeated datasets, while variance determines how noisy any one affordable dataset is. This distinction is central throughout policy-gradient design.

Adding a constant to every reward preserves action preferences when every trajectory has the same number of reward terms (or the same infinite discounted constant stream). It can change the task when episode length depends on the policy: a positive per-step offset can reward staying alive longer. Subtracting an action-independent baseline from a gradient weight is a different operation and does not change the reward objective.

## 11. Constant baselines for variance reduction

**Transcript coverage:** lines 3544-4107

### What the lecturer said - transcript only

Random initial circumstances influence return even when the agent did nothing unusually good or bad. The desired update should make better-than-average behavior more likely and worse-than-average behavior less likely. The lecturer therefore replaced $R(\tau)$ by $R(\tau)-b$, with $b$ chosen as the current batch’s average trajectory reward.

He justified the change by isolating the baseline term and using the log-derivative identity in reverse:

$$
\mathbb E_{\tau\sim p_\theta}
[\nabla_\theta\log p_\theta(\tau)b]
=b\int \nabla_\theta p_\theta(\tau)\,d\tau
=b\nabla_\theta\int p_\theta(\tau)\,d\tau
=b\nabla_\theta 1=0.
$$

Thus subtracting a constant baseline does not alter the expected gradient, although it changes finite-sample estimates and can lower variance. Any constant $b$ is unbiased in expectation; the mean reward is useful and intuitive but not the theoretically optimal variance-minimizing baseline. The exact optimal baseline can be derived by calculating the estimator’s variance, but it is complicated and almost never used in that form. Better practical baselines would be introduced later.

This is different from shifting the reward function once by a fixed constant because the batch-average baseline is recomputed at each iteration for the current policy. In a stock-trading example, as the policy grows more profitable, its standard rises: the relevant question is whether a trajectory did better than the policy’s current norm, not whether it beat a permanently fixed bad trader.

The lecturer clarified that all the previously listed failures are instances of variance and that the baseline is a general variance-reduction method, not only a correction for lucky or unlucky initial states. A student asked whether $b$ could depend on the trajectory and still move outside the integral; the lecturer deferred that important question to Friday.

### Source reconciliation

Slides 19-20 present the centered-return estimator and the zero-expectation proof. “Average reward is not the optimal baseline” is stated both visually and orally.

### Additional explanation

The proof above only covers a baseline independent of the sampled trajectory/action in the relevant expectation. Later, a state-dependent baseline is allowed because, conditional on a state, it does not depend on the sampled action. Care is also needed when estimating the baseline from the same finite batch: the clean population identity and a particular finite-sample implementation are related but not identical claims.

For example, let $z_i=\nabla_\theta\log p_\theta(\tau_i)$ and use the mean return of the same $N$ independent episodes, $\bar R=N^{-1}\sum_iR_i$. Then

$$
\mathbb E\!\left[\frac1N\sum_i z_i(R_i-\bar R)\right]
=\left(1-\frac1N\right)\nabla_\theta J.
$$

Each episode partly subtracts its own return, so this particular estimator has a finite-batch bias (and is zero when $N=1$). A baseline from independent data, or the mean of the other episodes, avoids that self-inclusion effect. A good baseline reduces noise; it need not equal the exact value to preserve the population score-gradient identity.

## 12. Causality and reward-to-go

**Transcript coverage:** lines 4108-4407

### What the lecturer said - transcript only

The next improvement uses causality: a decision now cannot change what happened yesterday. More formally, an action at time $t'$ cannot affect a reward at an earlier time $t<t'$. Expanding the two sums in the basic estimator otherwise multiplies every time-step policy score by rewards both before and after that action.

Past rewards can therefore be removed. Each score at time $t$ is multiplied only by the sum of rewards from $t$ through the end:

$$
\widehat Q_t=\sum_{t'=t}^{H}r(s_{t'},a_{t'}).
$$

This remaining return is called **reward-to-go** and is sometimes written $\widehat Q_t$. The lecturer said one can prove that the expected policy score multiplied by past rewards is zero, using an argument morally similar to the baseline proof, but did not work through the more involved derivation.

Reward-to-go reduces variance because it removes unrelated terms and multiplies each score by a smaller-magnitude sum. It is valid because future actions do not affect past rewards. The lecturer recommended always using this form and said there is no reason to retain the full-trajectory return in the estimator.

A different baseline is now appropriate at each time: $b_t$ is the average reward-to-go from time $t$ through the end, whereas the earlier single $b$ was the average whole-trajectory return. This answered a student’s question about the two notations.

### Source reconciliation

Slide 21 displays the causality estimator. Its reward sum begins at the action’s time index rather than at the beginning of the episode.

### Additional explanation

With a time-indexed baseline, the estimator has the form

$$
\widehat{\nabla_\theta J(\theta)}=
\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{H}
\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})
\left(\widehat Q_t^{(i)}-b_t\right).
$$

Reward-to-go fixes only one kind of irrelevant credit. A late reward may still depend weakly on a particular early action, so long-horizon variance remains.

The causal argument is conditional: given the history before $a_t$, earlier rewards are fixed and $\mathbb E[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\mid\text{history}]=0$. Their expected gradient contribution is therefore zero. This is the justification, rather than a claim that a shorter sum always has smaller magnitude or guarantees lower total-gradient variance for every problem.

For the discounted objective $J_\gamma=\mathbb E[\sum_{t=1}^H\gamma^{t-1}r_t]$, define $G_t=\sum_{k=t}^H\gamma^{k-t}r_k$. Its exact trajectory gradient is

$$
\nabla_\theta J_\gamma
=\mathbb E\!\left[\sum_{t=1}^H\gamma^{t-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)(G_t-b_t(s_t))\right].
$$

There are two discounts: one inside the return and one outside the score term. The outside factor disappears only under an appropriate discounted state-sampling convention, or when $\gamma=1$.

## 13. The weighted supervised pseudo-loss

**Transcript coverage:** lines 4408-4779

### What the lecturer said - transcript only

Explicitly computing every $\nabla\log\pi$ and then multiplying gradients is inconvenient. Instead, one constructs a computation graph whose derivative equals the desired policy gradient. The estimator is the derivative of a reward-weighted supervised-learning expression when sampled trajectories and their weights are treated as fixed.

The lecturer called this a **pseudo-loss** because its numerical value is not itself a meaningful mathematical objective for the full sampling process. It exists to make PyTorch, JAX, TensorFlow, or another automatic-differentiation package return the intended derivative. It resembles maximum likelihood except that every sampled action log probability is weighted by its reward-to-go, after subtracting any baseline.

For discrete actions, the underlying supervised term is cross-entropy; for a Gaussian policy it corresponds to the Gaussian negative log likelihood, often implemented through a squared-error-like expression. The shown TensorFlow-style pseudocode starts with states and actions, runs the policy to obtain logits, and forms per-sample likelihood losses. Moving from behavioral cloning to policy gradient adds a tensor of $\widehat Q$ values with shape $N\times T\times1$, computed by cumulative sums, and multiplies the per-step loss by those values before averaging and differentiating. Only the highlighted weighting changes.

The lecturer said this was essentially the core of Homework 2, with extensions. If a baseline is used, it is folded into the stored Q-value weights. In Q&A, he defined those weights as the sum of rewards from the current time step to the end minus the selected baseline. Whether rewards should be discounted was deferred until the following week. The current $\widehat Q$ is only a sampled sum of rewards; later value functions would improve its estimate, but at this point it is not a learned value function.

### Source reconciliation

Slide 22 labels the expression a “pseudo-loss,” and slide 23 shows the TensorFlow-like implementation comparison. Depending on whether an API returns log likelihood or negative log likelihood, an implementation must choose the sign so that gradient descent produces policy improvement.

### Additional explanation

A conventional loss for gradient descent is

$$
L_{\mathrm{PG}}(\theta)=
-\frac{1}{N}\sum_{i,t}
\operatorname{stopgrad}\!\left(\widehat Q_t^{(i)}-b_t\right)
\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)}).
$$

`stopgrad` expresses the lecturer’s warning: the weights and sampled data are held fixed while differentiating the pseudo-loss. Reusing the pseudo-loss value as though it were the true expected-return objective would be misleading.

## 14. Practical tuning guidance and closing

**Transcript coverage:** lines 4780-4843

### What the lecturer said - transcript only

The estimator provides a basic implementable algorithm, but practical policy gradients need additional care. Their gradients have higher variance than ordinary supervised-learning gradients. Larger batches or mini-batches, smaller learning rates, and more deliberate hyperparameter tuning may therefore be necessary. Adaptive step-size optimizers such as Adam can help. Later lectures would introduce policy-gradient-specific ways to adjust learning rates. The lecturer closed by saying that policy gradients would continue on Friday.

### Source reconciliation

The final slides summarize the policy-gradient estimator and point to the next lecture. The batch-size, learning-rate, optimizer, and tuning guidance comes from the spoken closing.

### Additional explanation

Useful diagnostics include the distribution of episode returns, reward-to-go weights, gradient norms, policy entropy, and the change in action log probabilities after an update. A loss curve alone is not a reliable performance metric because the pseudo-loss changes with newly sampled on-policy data.

Implementation checks: detach the sampled action if it was produced by a differentiable sampling operation, as well as the return/advantage weight. Keep the policy's parameter-to-log-probability computation differentiable, including its recurrent computation when using history. Sum log probabilities over joint-action components and mask padded time steps.

Average episode gradient sums over the number of episodes. Dividing by the total number of transitions instead introduces a random, policy-dependent denominator when episode lengths vary; it is not the same unbiased estimator. A sampled gradient step need not improve return on every iteration, even when its expectation is correct.

## Consolidated takeaways

1. Policy gradients directly differentiate expected return with respect to policy parameters.
2. The likelihood-ratio identity avoids differentiating through unknown environment dynamics.
3. After trajectory factorization, only policy log-probability gradients remain.
4. REINFORCE estimates the expectation with trajectories sampled from the current policy.
5. The method resembles maximum likelihood with each sampled action weighted by return.
6. Partial observability does not invalidate the score-function derivation, although it may require a richer policy.
7. The basic estimator is unbiased but can have severe finite-sample variance.
8. A constant baseline changes variance without changing the population gradient expectation.
9. Reward-to-go removes past rewards that an action could not have caused.
10. A weighted log-likelihood pseudo-loss is the practical autograd implementation.
11. Policy-gradient training generally needs larger batches and more careful step-size tuning than supervised learning.

## Key equations

### Expected-return objective

$$
J(\theta)=\mathbb E_{\tau\sim p_\theta(\tau)}
\left[\sum_{t=1}^{H}r(s_t,a_t)\right].
$$

### Log-derivative identity

$$
\nabla_\theta p_\theta(\tau)
=p_\theta(\tau)\nabla_\theta\log p_\theta(\tau).
$$

### Policy-gradient expression

$$
\nabla_\theta J(\theta)=
\mathbb E_{\tau\sim p_\theta}
\left[
\sum_{t=1}^{H}\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\sum_{t'=1}^{H}r(s_{t'},a_{t'})
\right].
$$

### Baseline identity

$$
\mathbb E_{\tau\sim p_\theta}
[\nabla_\theta\log p_\theta(\tau)b]=0.
$$

### Reward-to-go estimator

$$
\widehat{\nabla_\theta J(\theta)}=
\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{H}
\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})
\left(
\sum_{t'=t}^{H}r(s_{t'}^{(i)},a_{t'}^{(i)})-b_t
\right).
$$

### Policy-gradient pseudo-loss

$$
L_{\mathrm{PG}}(\theta)=
-\frac{1}{N}\sum_{i,t}\widehat A_t^{(i)}
\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)}),
\qquad
\widehat A_t=\widehat Q_t-b_t,
$$

where the sampled weights $\widehat A_t$ are held fixed during differentiation.

## Glossary

- **Baseline:** a quantity subtracted from a return weight to reduce variance without changing the expected policy gradient under the appropriate independence condition.
- **Causality:** the fact that an action cannot influence rewards that occurred before it.
- **Credit assignment:** determining which decisions deserve credit or blame for an observed return.
- **Gradient ascent:** updating parameters in the direction that locally increases an objective.
- **Likelihood-ratio estimator:** an estimator using $\nabla\log p$ to differentiate an expectation without pathwise differentiation.
- **On-policy sampling:** collecting trajectories from the same policy whose objective or gradient is being estimated.
- **Pathwise derivative:** a derivative propagated through a differentiable sampling computation.
- **Policy gradient:** a gradient of expected return with respect to policy parameters.
- **Pseudo-loss:** an autograd construction whose derivative is useful even though its scalar value is not the original RL objective.
- **REINFORCE:** the Monte Carlo score-function policy-gradient algorithm.
- **Reward-to-go:** the sum of rewards from the current time step through the end of the trajectory.
- **Score function:** the log-probability gradient $\nabla_\theta\log p_\theta(x)$.
- **Unbiased estimator:** an estimator whose expectation equals the target quantity.
- **Variance:** the dispersion of finite-sample estimates around their expectation.

## Self-check questions

1. Why can PyTorch differentiate a reward with respect to recorded actions but still fail to differentiate return with respect to policy parameters?
2. Derive $p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)=\nabla_\theta p_\theta(\tau)$.
3. Which terms vanish when differentiating the log trajectory distribution, and why?
4. Write the Monte Carlo REINFORCE estimator, including its normalization.
5. Why does sampling from $p_\theta(\tau)$ remove the need to know each trajectory’s probability explicitly?
6. How does policy gradient differ from ordinary maximum likelihood on the same sampled actions?
7. Why does adding a constant to every reward expose a problem in the naive interpretation?
8. Why does the derivation remain valid when policy inputs are non-Markov observations?
9. In preset-position chess, name three ways a move can receive misleading credit.
10. Explain how an estimator can be unbiased and nevertheless require impractically many samples.
11. Prove that a constant baseline has zero expected score-function contribution.
12. Why is a current-policy mean return different from permanently shifting the reward function?
13. What causal argument permits replacing full return with reward-to-go?
14. What does $b_t$ mean after the reward-to-go change?
15. Why is the weighted log-likelihood expression called a pseudo-loss?
16. Which tensors must be added to turn a supervised action-learning implementation into REINFORCE?
17. Why should return weights be held fixed during pseudo-loss differentiation?
18. What tuning changes did the lecturer recommend for high-variance gradients?

## Source coverage checklist

| Topic | Inclusive range | Coverage status |
|---:|---:|---|
| 1 | 1-267 | Accounted for |
| 2 | 268-501 | Accounted for |
| 3 | 502-852 | Accounted for |
| 4 | 853-1267 | Accounted for; overbroad derivative-integral claim disclosed |
| 5 | 1268-1519 | Accounted for |
| 6 | 1520-2028 | Accounted for; missing-$1/N$ slide typo disclosed |
| 7 | 2029-2361 | Accounted for; spoken constant correction retained |
| 8 | 2362-2604 | Accounted for |
| 9 | 2605-2840 | Accounted for |
| 10 | 2841-3543 | Accounted for |
| 11 | 3544-4107 | Accounted for |
| 12 | 4108-4407 | Accounted for |
| 13 | 4408-4779 | Accounted for |
| 14 | 4780-4843 | Accounted for |

**Coverage result:** All 4,843 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. All 23 slide pages were visually inspected; equations, implementation diagrams, and source-visible typos were checked against the spoken account.
