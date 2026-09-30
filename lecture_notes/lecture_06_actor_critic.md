---
title: "Lecture 6 - Actor-Critic"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 6
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 6, Actor-Critic.txt"
source_slides: "../lectures/Lecture 06 - Actor Critic.pdf"
transcript_lines: 9547
slide_pages: 33
status: "complete"
---

# Lecture 6: Actor-Critic

## Lecture map

**Reading conventions.** Finite-horizon values depend on remaining time, even when the notation suppresses it. In implementation formulas, $m_t=0$ at a true task terminal and $m_t=1$ otherwise. A collection cutoff is not a terminal: bootstrap from its final observation before any environment reset. Exact infinite-horizon discounted claims assume bounded rewards and $0\leq\gamma<1$.

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | REINFORCE recap and reward-to-go as a random variable | lines 1-408 |
| 2 | Replacing sampled reward-to-go by a conditional expectation | lines 409-814 |
| 3 | State-dependent baselines and advantage | lines 815-1224 |
| 4 | $Q^\pi$, $V^\pi$, $A^\pi$, and actor-critic anatomy | lines 1225-1455 |
| 5 | Policy evaluation and the one-step advantage identity | lines 1456-1884 |
| 6 | Monte Carlo value-function regression | lines 1885-2523 |
| 7 | Bootstrapped value targets and their failure modes | lines 2524-3000 |
| 8 | Discounting and infinite-horizon objectives | lines 3001-3447 |
| 9 | Time-invariant transition notation and policy-evaluation examples | lines 3448-3829 |
| 10 | A basic batch actor-critic algorithm | lines 3830-4417 |
| 11 | Online actor-critic and A3C | lines 4418-4824 |
| 12 | Intermission Q&A and practical research advice | lines 4825-5418 |
| 13 | On-policy methods and the critic/baseline bias trade-off | lines 5419-6066 |
| 14 | Eligibility traces and $n$-step returns | lines 6067-6558 |
| 15 | Generalized advantage estimation | lines 6559-7053 |
| 16 | Policy gradient with GAE and advantage normalization | lines 7054-7368 |
| 17 | Why a naive replay-buffer actor-critic is broken | lines 7369-7701 |
| 18 | Repairing the critic with a $Q$-function | lines 7702-8118 |
| 19 | Repairing the actor and accepting state-distribution mismatch | lines 8119-8646 |
| 20 | Complete off-policy actor-critic and replay-buffer practice | lines 8647-9000 |
| 21 | Reparameterized off-policy actor-critic | lines 9001-9547 |

## Part I - Improving policy gradients with values

## 1. REINFORCE recap and reward-to-go as a random variable

**Transcript coverage:** lines 1-408

### What the lecturer said - transcript only

Actor-critic algorithms extend policy-gradient methods and bring back value and Q-functions introduced earlier. The lecturer began by recapping REINFORCE, described as both the simplest policy-gradient method and perhaps the simplest RL algorithm. It directly differentiates expected total reward and applies stochastic gradient ascent.

At each iteration, run the policy in the simulator or world to sample trajectories from $p_\theta(\tau)$; compute per-time-step reward-to-go; form the policy-gradient estimate, using a baseline and causality to lower variance without changing its expectation; and update $\theta$ by gradient ascent. Reward-to-go for time $t$ on trajectory $i$ was denoted $\widehat Q_t^{(i)}$ and can be computed with a cumulative sum.

In the course’s three-part diagram, the orange stage generates samples, the green stage estimates return, and the blue stage improves the policy. Actor-critic changes the green stage by learning a better estimate.

The sampled reward-to-go is a random variable because both the policy and MDP may be stochastic. Starting from the same state and action need not lead to the same next state or future trajectory. If one could return to the same point and replay the future, different completions could yield different sums; that randomness creates policy-gradient variance.

### Source reconciliation

Slides 2-3 show the REINFORCE estimator and place “fit a model to estimate return” in the green stage. The spoken account explicitly says the environment need not be deterministic after a fixed state-action pair.

### Additional explanation

Conditioned on one sampled $(s_t,a_t)$,

$$
\widehat Q_t=\sum_{t'=t}^{H}r(s_{t'},a_{t'})
$$

is one draw from the distribution of possible futures. Its expectation can be stable even when any one draw is extremely noisy.

## 2. Replacing sampled reward-to-go by a conditional expectation

**Transcript coverage:** lines 409-814

### What the lecturer said - transcript only

The sampled $\widehat Q_t$ estimates the expected reward obtained after taking $a_t$ in $s_t$. A lower-variance estimate would average many possible completions from that same state and action. The relevant expectation is over both future policy actions and environment transitions.

This conditional expectation is the true expected reward-to-go:

$$
Q^\pi(s_t,a_t)=
\mathbb E_{\tau\sim\pi}
\left[
\sum_{t'=t}^{H}r(s_{t'},a_{t'})
\middle|s_t,a_t
\right].
$$

Teleporting back to exactly the same situation and repeatedly living out the future—likened to *Groundhog Day*—is generally impossible in the real world. If it were possible, averaging those futures would estimate $Q^\pi$ with progressively lower variance. The exact conditional expectation itself has no remaining future-rollout variance, so replacing the single-sample return by it would improve the policy-gradient estimator.

The lecturer distinguished the sampled return’s unbiasedness from its noise. With more samples it approaches the true value; he noted that this description is asymptotic unbiasedness, while ordinary unbiasedness means that averaging repeated runs of the estimator gives the target expectation.

A student asked why not directly choose $\arg\max_a Q(s,a)$. That is possible and would appear the following week. The argmax is difficult in high-dimensional continuous action spaces, and in practice may be approximated by training another neural network—an operation that can look much like a policy update.

### Source reconciliation

The slides replace a sampled reward-to-go with $Q^\pi(s_t,a_t)$ and depict averaging over possible future completions. The transcript adds the *Groundhog Day* analogy, the unbiasedness clarification, and the continuous-action argmax Q&A.

### Additional explanation

This lecture uses $Q$ in two connected roles: as the conditional expected return that would ideally weight a policy score, and later as a learned critic that approximates that conditional expectation.

An on-policy Monte Carlo return is already unbiased for $Q^\pi(s_t,a_t)$ at every sample size. Convergence of its sample average is consistency, not the definition of asymptotic unbiasedness. Replacing one score-times-return term by its exact conditional expectation removes that term's future-rollout noise; a learned critic can introduce bias, and this does not prove a universal variance reduction for the whole time-summed gradient.

## 3. State-dependent baselines and advantage

**Transcript coverage:** lines 815-1224

### What the lecturer said - transcript only

The prior lecture subtracted a baseline $b$, such as average return or a time-specific average remaining return, so long as it did not depend on the sampled action. A more informative baseline is the expected Q-value in the current state:

$$
V^\pi(s_t)=
\mathbb E_{a_t\sim\pi(\cdot\mid s_t)}[Q^\pi(s_t,a_t)].
$$

Although $V^\pi(s_t)$ depends on state, it does not depend on which action is sampled after conditioning on that state. The same score-function proof therefore shows that it may be subtracted without bias. It is better than a single global average because it represents the average reward specifically from the current state.

The difference

$$
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)
$$

is called the **advantage**. It measures how much better or worse action $a_t$ is than the policy’s average action in that state. A positive advantage increases an action’s probability; a negative advantage decreases it. This directly resolves the desired comparison in the preset-position chess example.

The lecturer emphasized that this discussion temporarily assumes access to the exact expectations. Previous reward-to-go and average-baseline quantities were noisy, single-sample estimates. The rest of the lecture would address how to approximate the expected values.

Subtracting the baseline still matters even with exact $Q$. If every chess outcome receives an added reward of 100, wins yield 101 and losses 99; all Q-values are positive, so a finite-sample update tries to increase all sampled actions. Subtracting $V$ centers the weights around the relevant state’s mean and yields a cleaner gradient from fewer samples.

In Q&A, the lecturer corrected a handwritten symbol on the slide: what looked like a capital $T$ was meant to be the horizon $H$.

### Source reconciliation

Slide 5 contains the oddly written horizon symbol explicitly corrected in the transcript. The formulas above use $H$ but retain this correction here rather than silently treating the source as unambiguous.

### Additional explanation

Conditioning explains why a state-dependent baseline is allowed:

$$
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}
[\nabla_\theta\log\pi_\theta(a\mid s)V(s)]
=V(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)=0.
$$

The baseline must be independent of the sampled action within this conditional expectation; it need not be globally constant across states.

Treat the baseline as fixed in the actor derivative, even if actor and critic share parameters. Fitting a baseline using the very same action/return samples introduces statistical dependence that this population proof does not automatically cover. Also, $V^\pi$ is a useful baseline, not necessarily the exact minimum-variance baseline, which depends on the magnitudes of policy score gradients.

## 4. $Q^\pi$, $V^\pi$, $A^\pi$, and actor-critic anatomy

**Transcript coverage:** lines 1225-1455

### What the lecturer said - transcript only

$V$ is a state-value function, usually shortened to value function. $Q$ is a state-action value function, usually shortened to Q-function. $Q^\pi(s_t,a_t)$ is the expected total reward from starting in $s_t$, first taking $a_t$, and then following policy $\pi$ in the MDP. Different policies generally have different Q-functions, so the superscript makes the dependence explicit; $Q^*$ conventionally denotes the Q-function of the best possible policy.

$V^\pi(s_t)$ is the expected total reward from $s_t$ while following $\pi$, equivalently the expectation of $Q^\pi(s_t,a_t)$ over the policy’s action at that state. Advantage is $Q^\pi-V^\pi$. The closer the advantage estimate is to this exact quantity, the lower the corresponding source of gradient variance. By contrast, one sampled reward-to-go minus a baseline is unbiased but high variance—like estimating the average age in a room by asking one person.

In actor-critic, the policy-gradient update in the blue box remains. The green box fits a model to estimate the advantage, commonly through a Q-function or value function. Thus actor-critic methods have two learned models: the **actor**, which is the policy, and the **critic**, which helps estimate return or advantage.

### Source reconciliation

Slide 6 collects all three definitions and labels the previous estimator as an unbiased, high-variance single-sample estimate.

### Additional explanation

The critic does not necessarily criticize in natural language. It supplies a scalar assessment that determines how strongly the actor should increase or decrease the probability of an action.

## Part II - Policy evaluation

## 5. Policy evaluation and the one-step advantage identity

**Transcript coverage:** lines 1456-1884

### What the lecturer said - transcript only

**Policy evaluation** means taking a fixed policy $\pi$ and estimating $V^\pi$ or $Q^\pi$. The name is literal: $V^\pi(s)$ predicts how well the policy will do from $s$, and its value at an initial state evaluates the policy overall.

The Q-function has a recursive decomposition. Once $s_t$ and $a_t$ are fixed, the immediate reward is known under the notation used in the lecture, while the future remains random:

$$
Q^\pi(s_t,a_t)=
r(s_t,a_t)+
\mathbb E_{s_{t+1}\sim p(\cdot\mid s_t,a_t)}
[V^\pi(s_{t+1})].
$$

Replacing the expectation over $s_{t+1}$ by the observed next state introduces only one step of sampling variance and remains unbiased when that next state is sampled from the true dynamics. Substitution into $A^\pi=Q^\pi-V^\pi$ gives the one-step estimate

$$
\widehat A(s_t,a_t)=
r(s_t,a_t)+V^\pi(s_{t+1})-V^\pi(s_t).
$$

This reads as the immediate reward plus the change in predicted future value. It depends only on $V$, which takes a state rather than a state-action pair, so the lecturer chose to learn a value network. This is not the only possible design; learning $Q$ would support a different story used later. Historically, early actor-critic methods followed the value-function route.

The learned network is written $\widehat V_\phi^\pi(s)$ to distinguish it from the exact function and from the actor parameters $\theta$. It may consume a complicated state such as a camera image but outputs one scalar.

### Source reconciliation

Slide 8 first shows the expectation exactly and then the one-next-state approximation. Discount $\gamma$ is introduced later, so this first identity is undiscounted as spoken.

### Additional explanation

The quantity $r_t+\gamma V(s_{t+1})-V(s_t)$ is often called a temporal-difference error. At this point in the lecture $\gamma$ has not yet been inserted, so the displayed version has $\gamma=1$.

For the exact value and a sampled reward/transition,
$\mathbb E[r_t+\gamma m_tV^\pi(s_{t+1})-V^\pi(s_t)\mid s_t,a_t]=A^\pi(s_t,a_t)$.
One residual is a noisy sample of advantage, not the advantage itself. Substituting an approximate value generally changes that conditional expectation.

## 6. Monte Carlo value-function regression

**Transcript coverage:** lines 1885-2523

### What the lecturer said - transcript only

Expected return from initial states satisfies $J(\theta)=\mathbb E_{s_1\sim p(s_1)}[V^\pi(s_1)]$. One policy-evaluation method is Monte Carlo evaluation: generate futures and average observed returns. Multiple restarts from exactly the same state would give better labels but require resetting or teleporting the simulator, so the general method uses one observed future per visited state.

For every state in every sampled trajectory, sum the rewards from that state to the end and call the result $y_t^{(i)}$. Fit the value network with ordinary squared-error regression:

$$
y_t^{(i)}=\sum_{t'=t}^{H}r(s_{t'}^{(i)},a_{t'}^{(i)}),
\qquad
L(\phi)=\frac12\sum_{i=1}^{N}\sum_{t=1}^{H}
\left\|\widehat V_\phi(s_t^{(i)})-y_t^{(i)}\right\|^2.
$$

This may seem circular because noisy returns motivated learning a value function, yet those same noisy returns supervise it. Neural-network generalization makes it useful: similar states observed on different trajectories may have different sampled returns, and regression tends to average those labels into a better estimate of their expectation. If outcomes are bimodal—half good and half bad—the desired value really is their mean, even though no individual trajectory realizes it.

Architecture depends on the state. A 10- to 20-dimensional state can use a standard MLP; images need an appropriate vision network. Large policies and critics may share an encoder and then branch into an action head and a scalar value head to save parameters.

Using the learned value rather than the raw label helps because the model pools information across related states. With one state leading equally often to good and bad outcomes, the network can predict the mean while every individual target lies in one mode.

The on-policy training set generally must be discarded after updating the policy because it estimates $V^\pi$ for the behavior policy that created it. Older samples estimate another policy’s values. Importance sampling or other advanced corrections can enable reuse, but not in the method as currently written.

### Source reconciliation

Slides 9-10 contrast ideal multiple-completion Monte Carlo evaluation with function approximation from one return label per visited state.

### Additional explanation

Supervised regression reduces conditional variance when the function class and optimization generalize appropriately; it may also introduce approximation bias. That is the first explicit bias-variance exchange in the critic.

## 7. Bootstrapped value targets and their failure modes

**Transcript coverage:** lines 2524-3000

### What the lecturer said - transcript only

The recursive value relation suggests a target that uses the existing critic instead of the entire observed return. If the current value estimate is better than one Monte Carlo sample, use it to label the previous state:

$$
y_t=r(s_t,a_t)+\widehat V_\phi(s_{t+1}).
$$

The new training set contains each visited state paired with its immediate reward plus the previously fitted next-state value. Regressing $\widehat V_\phi(s_t)$ onto this target is called a **bootstrapped estimate**. Values near an episode’s end become accurate first and propagate backward, although an implementation normally updates all sampled time steps together rather than literally traversing backward one at a time.

The target is circular: a randomly initialized network supplies poor labels to train itself. Under strong, unrealistic assumptions one can prove convergence to the true value function, but with neural networks convergence to the correct value is not guaranteed. The lecturer promised a fuller treatment during Q-learning and jokingly advised temporarily accepting the problem as deep-learning practitioners.

A second issue appears if the reward is always one and the horizon is absent. Starting near zero, the critic first learns targets near one; on the next iteration it learns two; repeated bootstrapping makes the value grow without bound. The finite horizon $H$ quietly disappeared from the time-homogeneous recursion, effectively converting the problem to infinite horizon. Feeding time $t$ to the critic could restore finite-horizon information, but many important tasks are naturally continuing or variable-length: a chess policy should play until the game ends, and a driving policy should reach its destination rather than operate for an arbitrarily fixed hour.

### Source reconciliation

Slide 11 calls the target bootstrapped. The transcript explicitly warns that neural-network convergence is not guaranteed; the slide alone could otherwise look like an unconditional improvement.

### Additional explanation

Bootstrapping trades a long, noisy observed return for a short target containing the model’s own prediction. It can lower variance but makes the target nonstationary and couples approximation errors across states. Detach the entire target during critic regression; differentiating its bootstrap value would define a different update.

## 8. Discounting and infinite-horizon objectives

**Transcript coverage:** lines 3001-3447

### What the lecturer said - transcript only

Infinite-horizon tasks require a way to keep values finite and express preference over timing. The lecturer’s “philosophical” motivation was mortality: because lives end, reward now is preferred to the same reward 10,000 years later. Introduce a discount factor $\gamma\in[0,1]$, often about $0.99$ and perhaps $0.999$ for high-frequency tasks:

$$
y_t=r(s_t,a_t)+\gamma\widehat V_\phi(s_{t+1}),
\qquad
J_\gamma(\theta)=
\mathbb E_{\tau\sim p_\theta}
\left[\sum_{t=1}^{\infty}\gamma^{t-1}r(s_t,a_t)\right].
$$

Discounting has an exact survival interpretation. Modify the MDP so that from every state there is probability $1-\gamma$ of entering an absorbing zero-reward death state and probability $\gamma$ of following the original transition. Undiscounted expected return in that modified MDP equals discounted return in the original dynamics.

Future rewards are not ignored; a sufficiently larger delayed reward can still dominate after applying $\gamma^k$, just as in compounded-interest calculations. Discounting changes the MDP/objective and can change the optimal policy. It is accepted because it handles continuing tasks conveniently and encodes a plausible timing preference.

### Source reconciliation

Slide 12 explicitly shows the absorbing-state construction and warns that $\gamma$ changes the MDP. The transcript gives $0.99$ and $0.999$ as practical examples, not universal prescriptions.

### Additional explanation

If rewards are bounded by $|r_t|\le R_{\max}$ and $\gamma<1$, then the infinite discounted sum is bounded by $R_{\max}/(1-\gamma)$. This is the technical reason discounting prevents the always-one example from diverging.

For $J_\gamma=\mathbb E[\sum_{t=1}^\infty\gamma^{t-1}r_t]$, the exact trajectory gradient includes $\gamma^{t-1}$ outside each score-times-advantage term. Equivalently, define the normalized discounted occupancy

$$
d_\gamma^\pi(s)=(1-\gamma)\sum_{t=1}^\infty\gamma^{t-1}P_\pi(s_t=s).
\qquad
\nabla_\theta J_\gamma=\frac{1}{1-\gamma}
\mathbb E_{s\sim d_\gamma^\pi,a\sim\pi_\theta}
[\nabla_\theta\log\pi_\theta(a\mid s)A^\pi(s,a)].
$$

Uniformly averaging the time steps of an ordinary rollout is not automatically sampling this discounted occupancy. Implementations often omit the outside discount as a practical surrogate; distinguish that convention from the exact start-state objective. Undiscounted finite episodes are also valid; an infinite undiscounted sum requires additional assumptions or a different objective, such as average reward.

## 9. Time-invariant transition notation and policy-evaluation examples

**Transcript coverage:** lines 3448-3829

### What the lecturer said - transcript only

With time-homogeneous dynamics and an infinite-horizon formulation, the exact time index often does not matter. A sequential tuple $(s_i,a_i,s_i')$ is called a **transition**, where the prime only says that $s_i'$ follows $s_i$ after $a_i$. Algorithms written with primes instead of $t$ and $t+1$ usually operate on transitions and do not care where they appeared in a trajectory.

The regression target becomes

$$
y_i=r(s_i,a_i)+\gamma\widehat V_\phi(s_i'),
$$

with the same squared-error objective. This is a notational change, not a new algorithm. Without discount, an infinite-horizon value may be non-finite, though specially structured episodic problems can remain finite. For chess with only a terminal $+1$ or $-1$ reward, return remains bounded even if no discount is used.

The lecturer gave two policy-evaluation examples. TD-Gammon (Gerald Tesauro, 1992) was described as an early deep-RL game-playing agent competitive with expert humans. A small neural network learned backgammon values from the game outcome. AlphaGo (2016) used a much larger, more advanced computation but the same essential idea: predict expected game outcome from a board state. With reward one for a win and zero for a loss, the value equals win probability. More complicated rewards yield a more complicated expected sum, but the value still predicts how well the policy will do from a state.

### Source reconciliation

Slides 13-14 identify TD-Gammon and AlphaGo and visualize the transition notation. The lecturer also mentioned a $+1/-1$ outcome convention before explaining the separate $1/0$ probability interpretation.

### Additional explanation

With no intermediate rewards and an undiscounted, almost-surely terminating task, $1/0$ terminal rewards give $V^\pi(s)=P_\pi(\text{win}\mid s)$. With $+1/-1$, $V^\pi(s)=2P_\pi(\text{win}\mid s)-1$ when draws are absent. Discounting weights outcomes by their time of arrival, so these probability identities no longer hold as written.

## Part III - Basic actor-critic algorithms

## 10. A basic batch actor-critic algorithm

**Transcript coverage:** lines 3830-4417

### What the lecturer said - transcript only

The batch actor-critic algorithm is REINFORCE augmented with a value critic:

1. Run the current policy and break the resulting trajectories into transitions $(s_i,a_i,s_i')$.
2. Compute targets $y_i=r(s_i,a_i)+\gamma\widehat V_\phi(s_i')$.
3. Refit $\widehat V_\phi(s_i)$ to those targets by supervised regression.
4. Compute $\widehat A_i=r(s_i,a_i)+\gamma\widehat V_\phi(s_i')-\widehat V_\phi(s_i)$.
5. Estimate the policy gradient with $\sum_i\nabla_\theta\log\pi_\theta(a_i\mid s_i)\widehat A_i$.
6. Update the actor by gradient ascent and repeat.

Trajectories may be stored intact in code, but the one-step formulas only require a bucket of transitions, so the mathematical sum no longer needs an explicit time index. The green stage now fits $\widehat V_\phi$; the actor $\pi_\theta$ outputs an action distribution and the critic outputs one number.

In classical RL notation, reward may be returned by the environment along with each transition rather than available as a callable function. A human training a dog is an example where no explicit reward function is known. In homework code the reward is likely directly computable. Sampling $(s_i,a_i,s_i',r_i)$ and evaluating $r(s_i,a_i)$ are therefore two notational versions of the same required information.

There are still two sources of sampling variance: sampled state-action inputs to the policy score and the advantage estimate. Given that the former must currently be sampled, improving $\widehat A$ is the available lever. The $\gamma$ in the advantage comes from the Q-function recursion.

Using the same batch both to fit the critic and to compute the actor gradient is, strictly, a statistical “no-no.” Some implementations use separate batches or old data for the critic. The lecturer said it technically alters the clean analysis but is often acceptable in practice.

### Source reconciliation

Slide 16 presents the six-step batch algorithm. Its transition-only notation does not mean trajectories were never rolled out; the transcript explicitly says they are sampled and then chopped into transitions.

### Additional explanation

This algorithm is on-policy: after a meaningful actor update, old targets no longer evaluate the new policy exactly. The critic and actor learning schedules must therefore remain coordinated.

A boundary-aware one-step implementation freezes a reference value prediction, forms $y_t=r_t+\gamma m_tV_{\rm ref}(s_{t+1})$, and uses $y_t-V_{\rm ref}(s_t)$ as a detached actor weight. The critic regresses toward detached $y_t$. At a true terminal the target is just $r_t$; at an external cutoff it still includes the final state's bootstrap value.

## 11. Online actor-critic and A3C

**Transcript coverage:** lines 4418-4824

### What the lecturer said - transcript only

Because the bootstrapped update needs only one transition, an actor and critic can update before an entire long trajectory finishes. A fully online version takes one environment step, forms one target, takes one critic gradient step to avoid overfitting that sample, computes a one-transition advantage and policy-gradient estimate, updates the actor, and repeats. The first actor-critic algorithm followed this pattern.

The method has serious issues. One critic step leaves $V$ perpetually behind the current policy, creating bias. In principle, repeated bootstrapping can eventually propagate long-horizon reward, though convergence is complicated. Consecutive transitions are not IID, violating assumptions behind ordinary stochastic-gradient behavior, and every update has batch size one.

A3C—described as asynchronous advantage actor-critic—runs multiple workers concurrently so their transitions form a mini-batch. It still faces non-IID data and may need extensive hyperparameter tuning, but it can work and is historically important. Collecting ten or twenty consecutive transitions from one worker exacerbates correlation; ten independent workers are statistically preferable, though ten physical systems may be infeasible.

The lagging critic is no longer exactly $V^\pi$, so the method is biased even with infinite samples. The practical hope is that small policy changes and fast, stable critic learning allow it to catch up. Actor and critic learning rates must be tuned carefully. The lecturer summarized the situation humorously as a mess, though not impossibly hard.

### Source reconciliation

Slides 17-18 move from batch to single-transition online updates and note that practical A3C typically requires multiple workers.

### Additional explanation

Two-time-scale analysis formalizes the intuition that one component should track the other. In practice, update ratios, learning rates, parallelism, and rollout length jointly control that tracking problem.

A3C workers collect their own short rollouts and asynchronously apply updates to shared parameters; their gradients can be stale. It is not simply a synchronized minibatch of independent transitions. A2C is the synchronous related pattern. Temporal correlation affects variance and analysis, but lack of IID data alone does not make every online gradient update invalid.

## 12. Intermission Q&A and practical research advice

**Transcript coverage:** lines 4825-5418

### What the lecturer said - transcript only

The class paused for an intermission until 9:12. During questions, the lecturer returned to the online method and said critic accuracy depends on tuning its learning rate so the value function keeps up with the actor.

Asked about common RL-project failures, he identified starting too late as the largest one. Training is slow and RL adds repeated hyperparameter searches, so a run taking $T$ hours can become roughly $T$ times the number of tuning iterations.

For research more broadly, he recommended a short iteration cycle and infrastructure that makes non-incremental ideas easy to test. There is a delicate balance between important large-scale problems and small experiments. Tiny grid worlds enable rapid experimentation but give weaker evidence of real relevance; large language-model or vision-based robotics experiments are compelling but make unusual ideas expensive to test. Good research finds a useful balance.

The baseline proof was clarified: the original $b$ is a constant with respect to the trajectory variable. More generally, a baseline may depend on other conditioned variables but not on the random variable whose log probability is differentiated—for example, it may depend on $s$ in $\pi(a\mid s)$ but not on $a$. An estimated expected reward is still a constant with respect to the individual trajectory inside that argument.

Students also revisited correlated samples and discussed using recent states or ordering from the end of a trajectory; the lecturer suggested continuing one unclear proposal after class before restarting the lecture.

### Source reconciliation

This intermission and research-advice exchange is transcript-only; the technical deck has no separate slide content to import into it.

### Additional explanation

This intermission material is part of the supplied transcript and is included in the line map even though it is not represented on technical slides.

## Part IV - Practical on-policy actor-critic

## 13. On-policy methods and the critic/baseline bias trade-off

**Transcript coverage:** lines 5419-6066

### What the lecturer said - transcript only

The second half introduced two practical families. On-policy actor-critic leads toward methods such as PPO and behaves like policy gradient made stable. Off-policy actor-critic leads toward sample-efficient methods such as SAC, at the cost of more computation, complexity, and tuning.

The basic batch actor-critic still needs a better advantage estimator and later a better policy-gradient estimator. This lecture would introduce generalized advantage estimation (GAE); importance sampling and PPO’s modified gradient would come later.

Two advantage estimators expose a bias-variance trade-off:

- The one-step critic estimate $r_t+\gamma\widehat V(s_{t+1})-\widehat V(s_t)$ has lower variance, but an imperfect critic introduces bias. Because the critic is normally warm-started and trained for a limited number of steps rather than exactly refitted, it is not truly $V^\pi$.
- The Monte Carlo return minus an average baseline is unbiased in the infinite-sample limit, but has higher variance because it uses a single sampled future.

An imperfect critic can be wrong in arbitrary, gradient-correlated ways; merely lowering its mean squared error without reaching the true value gives no general guarantee of the correct policy gradient. The bias may nevertheless be worthwhile when it buys a large variance reduction.

The critic can instead be used only as a baseline:

$$
\widehat A_t=
\left(\sum_{t'=t}^{H}\gamma^{t'-t}r_{t'}\right)-\widehat V_\phi(s_t).
$$

This remains unbiased because the subtracted function depends on state but not the sampled action, and it usually has lower variance than subtracting one global constant. The lecturer suggested reproducing the one-step conditional score-function proof as a useful exercise and possible exam question.

### Source reconciliation

Slides 21-22 contrast the biased one-step critic estimator with the unbiased Monte Carlo return plus learned state baseline. “Not unbiased if the critic is not perfect” is explicit on the deck.

### Additional explanation

The value network has two conceptually different uses: it can replace unobserved future return inside a bootstrapped Q estimate, which may bias the actor, or it can merely subtract a state-only control variate from an otherwise sampled return, which preserves the expected score gradient.

If $e(s)=V_\phi(s)-V^\pi(s)$, then

$$
\mathbb E[\delta_t^\phi\mid s_t,a_t]-A^\pi(s_t,a_t)
=\gamma\mathbb E[m_te(s_{t+1})\mid s_t,a_t]-e(s_t).
$$

The current-state error cancels in the expected score gradient as a baseline; the successor error generally depends on the action and does not cancel. Conversely, a complete Monte Carlo return minus an inaccurate fixed baseline can still give an unbiased policy gradient, even though it is not conditionally unbiased for the true advantage.

## 14. Eligibility traces and $n$-step returns

**Transcript coverage:** lines 6067-6558

### What the lecturer said - transcript only

Eligibility traces or $n$-step returns interpolate between the one-step actor-critic estimate and the full Monte Carlo estimate. The one-step form has low variance but can inherit large bias from a wrong critic. The full return has no critic-induced bias but high variance.

An $n$-step advantage uses $n$ observed rewards, then bootstraps from the value function:

$$
\widehat A_t^{(n)}=
\sum_{t'=t}^{t+n-1}\gamma^{t'-t}r_{t'}
+\gamma^n\widehat V_\phi(s_{t+n})
-\widehat V_\phi(s_t).
$$

$n=1$ recovers classic one-step actor-critic; taking $n$ to the remaining episode length approaches Monte Carlo. Intermediate $n$ can be a useful compromise. Far-future rewards bring more uncertainty, while the critic’s error at the cutoff is attenuated by $\gamma^n$. A less trusted critic suggests larger $n$; a more trusted critic suggests smaller $n$.

The trajectory must remain ordered so that $s_{t+n}$ and the intervening rewards are available; the transitions can no longer be completely scrambled. A student suggested defining $n$ in physical time rather than simulator steps. The lecturer agreed that, for example, always backing up half a second avoids changing the effective backup duration when simulation frequency changes.

### Source reconciliation

Slide 23 draws a cutoff between near-term sampled rewards and a value-function tail. The upper limit is interpreted inclusively as the next $n$ reward terms, consistent with the spoken “next 20 rewards” example.

### Additional explanation

Increasing $n$ reduces reliance on the critic but adds stochastic environment and policy outcomes. Thus $n$ controls where model error is exchanged for sampling noise.

Count exactly $n$ rewards, from $r_t$ through $r_{t+n-1}$, before bootstrapping at $s_{t+n}$. If a boundary arrives after only $h<n$ transitions, use those $h$ rewards and $\gamma^h mV(s_{t+h})$. Never include rewards from the next reset episode. Bias and variance need not change monotonically with $n$ for an arbitrary imperfect critic.

## 15. Generalized advantage estimation

**Transcript coverage:** lines 6559-7053

### What the lecturer said - transcript only

Rather than choosing one cutoff $n$, generalized advantage estimation combines all possible $n$-step advantages. Each is weighted with exponential falloff, proportional to $\lambda^{n-1}$. This choice is somewhat arbitrary, but it produces a convenient telescoping identity and works well.

Define the one-step temporal-difference residual

$$
\delta_t=r_t+\gamma\widehat V_\phi(s_{t+1})-\widehat V_\phi(s_t).
$$

Then the weighted mixture collapses to

$$
\widehat A_t^{\mathrm{GAE}(\gamma,\lambda)}=
\sum_{t'=t}^{\infty}(\gamma\lambda)^{t'-t}\delta_{t'}.
$$

The expanded interpretation is that at each step there is a $1-\lambda$ tendency to cut the sampled trace and use the critic; longer traces receive exponentially smaller weight. Implementation only requires computing all $\delta_t$ values and a discounted cumulative sum.

GAE is widely used in modern policy-gradient methods, including PPO. Its appeal is practical rather than uniquely theoretically mandated: it represents a sensible mixture of all $n$-step returns, is easy to compute, and works well.

$\gamma$ and $\lambda$ play different roles. $\gamma$ defines the discounted RL objective; lowering it makes the optimal policy more near-term or “hedonistic,” while high $\gamma$ plans farther ahead. $\lambda$ changes the estimator and its bias-variance behavior without changing the objective. With a perfect population estimate, different $\lambda$ values should not define different tasks.

In principle, $\lambda$ could adapt as the critic improves—lower for a trusted value model and higher for a poor one—but reliably measuring critic accuracy is difficult, so this is not usually done.

### Source reconciliation

Slide 24 credits Schulman, Moritz, Levine, Jordan, and Abbeel (2016) and writes the compact GAE identity. The transcript explicitly calls the exponential weighting arbitrary but effective.

### Additional explanation

For a finite rollout, use separate masks for value bootstrapping and for continuation of the advantage trace:

$$
\delta_t=r_t+\gamma m_tV_{\rm ref}(s_{t+1})-V_{\rm ref}(s_t),
\qquad
\widehat A_t=\delta_t+\gamma\lambda c_t\widehat A_{t+1}.
$$

Here $c_t=1$ only when the next residual belongs to the same trajectory and is available in the rollout. At a true terminal, $m_t=c_t=0$. At an external time limit or rollout cutoff, use $m_t=1,c_t=0$: bootstrap from the final pre-reset observation but do not propagate a trace into a reset episode. If the time limit defines the task itself, it is a true terminal and remaining time belongs in the state.

The equivalent finite mixture with $K$ available transitions is

$$
\widehat A_t^{\rm GAE}
=(1-\lambda)\sum_{n=1}^{K-1}\lambda^{n-1}\widehat A_t^{(n)}
+\lambda^{K-1}\widehat A_t^{(K)}.
$$

The last weight absorbs the remaining mass. Using $(1-\lambda)$ on every term loses mass and fails at $\lambda=1$. The endpoints are one-step TD at $\lambda=0$, and the full available return minus the starting value at $\lambda=1$. The latter still depends on the endpoint critic if the rollout was truncated.

For example, with rewards $(0,0,1)$, values $(0.2,0.4,0.6,0)$, $\gamma=1$, and a true terminal after reward 1, the residuals are $(0.2,0.2,0.4)$. At $\lambda=0.5$, backward recursion gives advantages $(0.4,0.4,0.4)$. At $\lambda=1$, the first advantage is $1-0.2=0.8$. See the [original GAE paper](https://arxiv.org/abs/1506.02438) for the estimator's derivation.

## 16. Policy gradient with GAE and advantage normalization

**Transcript coverage:** lines 7054-7368

### What the lecturer said - transcript only

A practical on-policy loop samples trajectories, computes value targets, fits the critic, evaluates GAE, inserts it into the policy-gradient estimator, and updates the policy. The estimator used for critic targets is an independent design choice: GAE itself, one-step bootstrap targets, $n$-step targets, or Monte Carlo targets may be used. The slide showed one-step targets.

In practice, advantages are usually normalized over the batch. Compute their mean $\mu$ and standard deviation $\sigma$, then replace each advantage by $(\widehat A-\mu)/\sigma$. The lecturer described this as a hard-to-justify numerical hack that nevertheless improves optimizer behavior, in a spirit similar to Adam or Adagrad. If $\sigma$ is near zero, clamp it with a small threshold such as about $10^{-6}$ to avoid division by an unstable number.

GAE can be used for both the critic targets and actor weights, but need not be. The lecturer characterized policy gradient with GAE as roughly half of PPO; the other half uses a different importance-sampling-based gradient estimator taught later. He advised implementing the centering simply as specified rather than blending its statistics into the GAE formula.

### Source reconciliation

Slide 25 shows both the GAE actor update and explicit batch centering. Its yellow annotation calls this “the first part of implementing PPO.”

### Additional explanation

Advantage normalization changes finite-batch scaling and can introduce small coupling between examples. It is a practical optimization device, not part of the exact policy-gradient theorem.

A return-like critic target is $\widehat R_t=\widehat A_t^{\rm GAE}+V_{\rm ref}(s_t)$, computed before advantage normalization and then detached. Regressing the critic directly onto advantages would train it to predict the wrong quantity. Keep reference values and actor weights fixed while optimizing a given batch.

## Part V - Off-policy actor-critic

## 17. Why a naive replay-buffer actor-critic is broken

**Transcript coverage:** lines 7369-7701

### What the lecturer said - transcript only

On-policy policy gradient with GAE is relatively simple and computationally quick but discards data after each update. Off-policy actor-critic is more complicated, computationally expensive, and harder to tune, but reuses experience and is more sample efficient.

Starting from online actor-critic, place each new transition in a **replay buffer** instead of immediately learning only from it. For later updates, sample an approximately IID mini-batch of old and new transitions from the buffer. This fixes batch-size-one and strong adjacent-transition correlation.

The naive substitution breaks two assumptions. First, an old $(s_i,a_i)$ pair came from an older behavior policy $\bar\pi$, so $r_i+\gamma\widehat V(s_i')$ does not in general evaluate the current policy. Second, the replayed $a_i$ was not sampled from the current $\pi_\theta(a\mid s_i)$, so inserting it directly into the on-policy score-function estimator is invalid. Both the critic target and actor update must be repaired.

The key change is to learn a Q-function rather than only a value function.

### Source reconciliation

Slides 26-27 intentionally label the first replay-buffer construction “broken” and cross out its on-policy assumptions. This is a negative example, not a recommended algorithm.

### Additional explanation

Replay removes the need for one environment interaction per gradient example, but it changes the data-generating policy. Off-policy algorithms must explicitly decide which quantities may use logged actions and which must be recomputed under the current policy.

## 18. Repairing the critic with a $Q$-function

**Transcript coverage:** lines 7702-8118

### What the lecturer said - transcript only

Learn $\widehat Q_\phi(s,a)$ with the bootstrap target

$$
y_i=r(s_i,a_i)+\gamma
\mathbb E_{a_i'\sim\pi_\theta(\cdot\mid s_i')}
[\widehat Q_\phi(s_i',a_i')].
$$

The stored transition supplies $s_i,a_i,r_i,s_i'$, but the next action is newly sampled from the latest policy at the stored next state. Resampling an action is cheap because the policy is a neural network in memory and needs no new environment transition. The Q-network can then evaluate that hypothetical current-policy action. One action gives a convenient Monte Carlo approximation of the expectation; multiple actions are valid, but if computation permits ten policy evaluations it is usually more useful to draw ten replay states and one action for each than ten actions for one state.

The lecturer explicitly corrected a slide typo: the new action should be sampled from $\pi_\theta(a_i'\mid s_i')$, not conditioned on the unprimed state. Using the latest policy in this target fixes the critic’s action-policy mismatch.

### Source reconciliation

Slide 28 contains the conditioning typo called out in the transcript. The corrected formula above follows the lecturer’s spoken correction, while the source error remains documented here.

### Additional explanation

The logged current action $a_i$ remains a valid input at which to regress Q because the target asks for the return after that particular first action, followed by the current policy thereafter. Only the next action in the bootstrap continuation must come from the target policy.

## 19. Repairing the actor and accepting state-distribution mismatch

**Transcript coverage:** lines 8119-8646

### What the lecturer said - transcript only

The actor has the same action mismatch. For each replayed state $s_i$, sample a fresh $\widetilde a_i\sim\pi_\theta(\cdot\mid s_i)$ and use

$$
\widehat{\nabla_\theta J}=
\frac{1}{N}\sum_{i=1}^{N}
\nabla_\theta\log\pi_\theta(\widetilde a_i\mid s_i)
\widehat Q_\phi(s_i,\widetilde a_i).
$$

The action is new; the state comes from replay. One could estimate $V(s_i)$ by averaging Q over actions and subtract it to form an advantage. In practice, the lecturer said people often use Q directly. Baselines lower variance when new environment samples are expensive, but replay examples are cheap; increasing the replay mini-batch can reduce variance more simply than estimating and subtracting $V$.

Increasing the number sampled means increasing mini-batch size, not merely buffer capacity.

The replayed state distribution still differs from the current policy’s visitation distribution. The lecturer acknowledged this conceptual mismatch. The practical rationale is that learning a good policy on a broader collection of states should not harm its behavior on current-policy states if the model has enough capacity—analogous to a California driver also learning to drive in the UK. This is still one task, not necessarily multitask learning.

Fresh current-policy data must remain represented. Practical buffers often have finite capacity and evict old transitions. Omitting collection or taking too many gradient updates while the policy changes rapidly can leave the buffer unrepresentative and degrade performance. Data collection rate, actor update rate, and buffer recency must be balanced.

### Source reconciliation

Slides 29-30 retain the replay-state mismatch as an unresolved approximation and explicitly say there is nothing in this construction that corrects it.

### Additional explanation

The displayed actor expression is a surrogate gradient over the replay-state distribution, not an exact reconstruction of the original on-policy state-visitation gradient. Its effectiveness relies on coverage and function approximation, not on the earlier unbiased on-policy theorem.

With the replay-state distribution and critic parameters fixed, it is the gradient of $\mathbb E_{s\sim D,a\sim\pi_\theta}[Q_\phi(s,a)]$. Do not additionally differentiate critic parameters through the actor loss or identify this fixed-critic objective with the exact start-state return.

## 20. Complete off-policy actor-critic and replay-buffer practice

**Transcript coverage:** lines 8647-9000

### What the lecturer said - transcript only

The complete algorithm is:

1. Take one step in the simulator or world and append $(s_i,a_i,r_i,s_i')$ to the replay buffer.
2. Sample a mini-batch from the buffer.
3. For each sample, draw $a_i'\sim\pi_\theta(\cdot\mid s_i')$ and form $y_i=r_i+\gamma\widehat Q_\phi(s_i',a_i')$.
4. Update $\phi$ on Q-regression mean squared error; begin with one gradient step and tune the update count if needed.
5. At each replayed $s_i$, draw a current-policy action and estimate the score-function actor gradient weighted by $\widehat Q_\phi$.
6. Update the actor and repeat.

The state-distribution mismatch remains, but the method is an actual implementable algorithm. The intuition is that it learns a good policy over a broader distribution than strictly necessary.

For small low-dimensional states, store the replay buffer in system memory. Disk caching is less attractive because training needs random access. Large systems may use a distributed buffer; the lecturer named DeepMind’s Reverb package as an example. Expectations over policy actions are usually approximated with one sampled action. The most common eviction scheme is a ring buffer, which removes the oldest samples as new ones arrive and thereby favors data closer to the current policy; more selective schemes also exist.

### Source reconciliation

Slide 30 consolidates the algorithm and slide 31 begins the transition to a different actor-gradient implementation. Replay-buffer storage and eviction guidance appears in transcript Q&A.

### Additional explanation

Replay ratio—the number of gradient updates per newly collected transition—is a central stability and efficiency parameter. Too small wastes stored data; too large can exploit critic errors and outrun coverage by fresh samples.

## 21. Reparameterized off-policy actor-critic

**Transcript coverage:** lines 9001-9547

### What the lecturer said - transcript only

The off-policy actor objective contains only differentiable objects held in the computer: the policy and learned Q-network. Unlike the original trajectory-return objective, it contains no unknown dynamics inside the action expectation. Therefore the derivative of Q with respect to action can be used directly.

For a Gaussian policy whose neural network produces mean and scale,

$$
\pi_\theta(a\mid s)=\mathcal N(\mu_\theta(s),\operatorname{diag}(\sigma_\theta(s)^2)),
\qquad
\epsilon\sim\mathcal N(0,I),
\qquad
a=\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon.
$$

A Gaussian sample is thus a deterministic differentiable transformation of noise whose distribution is independent of $\theta$. Rewrite

$$
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}[Q(s,a)]
=
\mathbb E_{\epsilon\sim\mathcal N(0,I)}
[Q(s,\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon)].
$$

After sampling $\epsilon$, PyTorch, JAX, or another autograd system can backpropagate through Q, the sampled action, and the policy outputs. This **reparameterized gradient** uses $\partial Q/\partial a$ and has lower error than the score-function alternative in this setting.

The trick requires a differentiable continuous action and a reparameterizable distribution. It does not directly work for discrete actions. Other continuous parametric distributions are possible, but Gaussian policies are the easiest and common. A network may output both a mean vector and covariance or diagonal scale; the scale may also be fixed, in which case only the mean depends on $\theta$.

The full reparameterized off-policy actor-critic keeps replay collection and Q training unchanged and replaces only the actor step by differentiating

$$
\widehat Q_\phi
(s_i,\mu_\theta(s_i)+\sigma_\theta(s_i)\odot\epsilon_i).
$$

The lecturer named soft actor-critic and TD3 as practical descendants. Strong implementations add Q-estimation techniques taught in later Q-learning lectures and, for SAC, entropy regularization. Deterministic-policy variants would also appear later.

### Source reconciliation

Slides 31-33 show the progression from the score estimator to the reparameterized gradient. The deck writes a univariate-looking $\sigma$ for clarity; the transcript explicitly allows a covariance matrix or diagonal vector.

### Additional explanation

This is a pathwise derivative, the estimator family contrasted with likelihood-ratio gradients in Lecture 5. Moving the environment out of the differentiated expectation makes an often lower-variance pathwise route available, though lower variance is not guaranteed for every problem.

For a diagonal Gaussian, write $a=\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon$ with $\epsilon\sim\mathcal N(0,I)$; its covariance is $\operatorname{diag}(\sigma_\theta^2)$, not $\sigma_\theta$ itself. For a full covariance use $a=\mu_\theta+L_\theta\epsilon$ with $\Sigma_\theta=L_\theta L_\theta^\top$. Freeze critic parameters during the actor step but preserve the derivative through the critic's action input. Detaching that action would destroy this pathwise gradient.

## Consolidated takeaways

1. Reward-to-go is an unbiased but noisy sample of a conditional expected return.
2. $Q^\pi(s,a)$ is expected return after a specified first action; $V^\pi(s)$ averages Q over the policy’s action.
3. $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ compares an action with the policy’s state-specific average.
4. An actor-critic method learns a policy actor and a return-estimating critic.
5. Monte Carlo value fitting averages noisy return labels through function approximation.
6. Bootstrapping replaces a long sampled return with immediate reward plus a learned next-state value.
7. Discounting controls infinite-horizon returns and changes the optimized objective.
8. A one-step critic can lower variance but may bias the actor if the critic is wrong.
9. $n$-step returns and GAE interpolate between critic bias and Monte Carlo variance.
10. GAE’s $\lambda$ changes the estimator; $\gamma$ changes the task objective.
11. Advantage normalization is a practical optimizer heuristic, not part of the exact theorem.
12. Replay reuses data; when behavior differs from the current policy, the data are off-policy.
13. A Q-critic lets current-policy actions be resampled at stored states and next states.
14. The replay-state distribution remains mismatched and must retain sufficient fresh coverage.
15. Reparameterization provides a pathwise actor gradient for differentiable continuous policies, often with lower variance than the score estimator.

## Key equations

Use detached reference predictions for target construction. Here $m_t$ masks true terminals and $c_t$ stops traces at rollout boundaries; at a nonterminal cutoff $m_t=1,c_t=0$. Values at a true terminal are zero. See Section 8 for the exact discounted actor-gradient weighting.

### Value, Q, and advantage

$$
Q^\pi(s_t,a_t)=
\mathbb E_\pi\left[\sum_{t'=t}^{H}\gamma^{t'-t}r_{t'}\middle|s_t,a_t\right],
$$

$$
V^\pi(s_t)=\mathbb E_{a_t\sim\pi(\cdot\mid s_t)}[Q^\pi(s_t,a_t)],
\qquad
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t).
$$

### One-step temporal-difference advantage

$$
\delta_t=r_t+\gamma m_tV_{\rm ref}(s_{t+1})-V_{\rm ref}(s_t).
$$

### Critic regression

$$
y_i=r_i+\gamma m_iV_{\rm ref}(s_i'),
\qquad
L_V(\phi)=\frac{1}{2N}\sum_{i=1}^{N}
(\widehat V_\phi(s_i)-y_i)^2.
$$

### Discounted objective

$$
J_\gamma(\theta)=
\mathbb E_{\tau\sim p_\theta}
\left[\sum_{t=1}^{\infty}\gamma^{t-1}r_t\right].
$$

### $n$-step advantage

$$
\widehat A_t^{(h)}=
\sum_{k=0}^{h-1}\gamma^k r_{t+k}
+\gamma^h m_{t+h-1}V_{\rm ref}(s_{t+h})
-V_{\rm ref}(s_t),\qquad 1\leq h\leq n.
$$

Here $h$ is the actual number of transitions available before the boundary.

### Generalized advantage estimation

$$
\widehat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=\delta_t+\gamma\lambda c_t\widehat A_{t+1}^{\mathrm{GAE}(\gamma,\lambda)}.
$$

Without intervening boundaries, this expands into $\sum_{k\geq0}(\gamma\lambda)^k\delta_{t+k}$. The critic target is the raw advantage plus $V_{\rm ref}(s_t)$.

### Off-policy Q target

$$
y_i=r_i+\gamma m_i
\mathbb E_{a'\sim\pi_\theta(\cdot\mid s_i')}
[Q_{\rm ref}(s_i',a')].
$$

Detach the whole target, including the sampled next action.

### Reparameterized actor objective

$$
a_\theta(s,\epsilon)=\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I),
$$

$$
\nabla_\theta J_{\mathrm{actor}}
\approx\frac{1}{N}\sum_{i=1}^{N}
\nabla_\theta\widehat Q_\phi(s_i,a_\theta(s_i,\epsilon_i)).
$$

## Glossary

- **A3C:** asynchronous advantage actor-critic, an online method using parallel workers.
- **Actor:** the learned policy in an actor-critic method.
- **Advantage:** $Q^\pi(s,a)-V^\pi(s)$, the action’s value relative to the policy average at the state.
- **Bootstrapping:** training a value estimate from a target that contains another learned value estimate.
- **Critic:** a learned value or Q model used to assess and improve the actor.
- **Discount factor:** $\gamma$, which downweights later rewards and defines a discounted objective.
- **GAE:** generalized advantage estimation, an exponentially weighted combination of $n$-step advantage estimates.
- **Monte Carlo evaluation:** estimating expected return from complete sampled returns.
- **Off-policy:** learning about a target policy using data generated partly by other policies.
- **On-policy:** learning from data generated by the current target policy.
- **Policy evaluation:** estimating $V^\pi$ or $Q^\pi$ for a given policy.
- **Q-function:** expected return conditioned on a state and first action.
- **Replay buffer:** storage from which past transitions are sampled for additional learning updates.
- **Reparameterization trick:** expressing a random action as a differentiable transformation of parameter-independent noise.
- **State-value function:** expected return from a state while following a policy.
- **Temporal-difference residual:** $r_t+\gamma V(s_{t+1})-V(s_t)$.
- **Transition:** a tuple containing a state, action, next state, and usually reward.

## Self-check questions

1. Why is reward-to-go random even after conditioning on a state and action?
2. Over which two sources of future randomness is $Q^\pi(s,a)$ averaged?
3. Why may $V(s)$ be subtracted as a baseline even though it varies across states?
4. Interpret advantage in the preset-position chess example.
5. Distinguish $Q^\pi$, $V^\pi$, and $Q^*$.
6. Why is learning a value function called policy evaluation?
7. Derive the one-step advantage from the recursive Q identity.
8. How can regression on noisy Monte Carlo returns improve on any one target?
9. Why must the simple on-policy critic dataset be refreshed after changing the actor?
10. What circularity does a bootstrapped value target introduce?
11. Explain the always-one-reward divergence without a horizon or discount.
12. How does the absorbing-death-state construction reproduce discounting?
13. Why does changing $\gamma$ change the task rather than merely the estimator?
14. List the six steps of batch actor-critic.
15. Why is reusing one batch to fit the critic and update the actor statistically delicate?
16. What problems arise in single-transition online actor-critic, and how does A3C help?
17. Contrast the bias and variance of one-step critic and Monte Carlo-baseline advantages.
18. How does $n$ determine the split between observed rewards and bootstrapped value?
19. Derive GAE’s backward recursion from its discounted residual sum.
20. What different roles do $\gamma$ and $\lambda$ play?
21. Why is advantage normalization described as a practical hack?
22. Which two assumptions fail when replay actions are inserted naively into on-policy actor-critic?
23. Why does learning Q permit resampling a current-policy next action at an old next state?
24. Why might a larger replay batch substitute for subtracting a value baseline?
25. What mismatch remains after both replay actions have been resampled?
26. Why can excessive gradient updates per environment step destabilize replay learning?
27. Derive the Gaussian reparameterization and explain why it cannot directly handle discrete actions.

## Source coverage checklist

| Topic | Inclusive range | Coverage status |
|---:|---:|---|
| 1 | 1-408 | Accounted for |
| 2 | 409-814 | Accounted for |
| 3 | 815-1224 | Accounted for; handwritten-$H$ correction disclosed |
| 4 | 1225-1455 | Accounted for |
| 5 | 1456-1884 | Accounted for |
| 6 | 1885-2523 | Accounted for |
| 7 | 2524-3000 | Accounted for; convergence caveat preserved |
| 8 | 3001-3447 | Accounted for |
| 9 | 3448-3829 | Accounted for |
| 10 | 3830-4417 | Accounted for |
| 11 | 4418-4824 | Accounted for |
| 12 | 4825-5418 | Accounted for; includes intermission Q&A |
| 13 | 5419-6066 | Accounted for |
| 14 | 6067-6558 | Accounted for |
| 15 | 6559-7053 | Accounted for |
| 16 | 7054-7368 | Accounted for |
| 17 | 7369-7701 | Accounted for; intentionally broken algorithm identified |
| 18 | 7702-8118 | Accounted for; next-state conditioning typo disclosed |
| 19 | 8119-8646 | Accounted for |
| 20 | 8647-9000 | Accounted for |
| 21 | 9001-9547 | Accounted for |

**Coverage result:** All 9,547 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. All 33 slide pages were visually inspected; the actor/critic diagrams, GAE derivation, replay-buffer corrections, and reparameterization equations were checked against the spoken account.
