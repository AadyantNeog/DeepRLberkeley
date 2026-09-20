---
title: "Lecture 22 - Midterm Review, Part 2"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 22
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 22, Midterm Review 2.txt"
source_slides: "../lectures/Lecture 22 - Midterm Review Part 2.pdf"
continuation_slides: "../lectures/Lecture 21 - Midterm Review Part 1.pdf"
continuation_slide_range: "17-34"
continuation_slide_pages: 18
transcript_lines: 2810
slide_pages: 73
status: "complete"
source_qualifier: "cross-deck-continuation-disclosed"
---

# Lecture 22: Midterm Review, Part 2

> **Source alignment:** The recording begins by continuing the policy-gradient and actor-critic material on slides 17-34 of the Lecture 21 deck. It then uses the Lecture 22 deck through roughly slide 42. Slides 43-73 of the Lecture 22 deck were not reached. This note follows the transcript as the authority for what was spoken and labels the unused slide material separately.

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Off-policy policy gradient and importance sampling | lines 1-167 |
| 2 | PPO importance-ratio clipping | lines 168-339 |
| 3 | PPO implementation and KL-constrained derivation | lines 340-926 |
| 4 | Online actor-critic and correlated samples | lines 927-1036 |
| 5 | $n$-step returns and generalized advantage estimation | lines 1037-1241 |
| 6 | Off-policy actor-critic and reparameterization | lines 1242-1331 |
| 7 | Q-learning, target networks, Double Q, and DDPG | lines 1332-1511 |
| 8 | Why fitted value iteration need not converge | lines 1512-1690 |
| 9 | Variational inference and amortized inference | lines 1691-1980 |
| 10 | Structured state-space models | lines 1981-2072 |
| 11 | Control as inference and maximum-entropy RL | lines 2073-2434 |
| 12 | Soft actor-critic, inverse RL, and adversarial imitation | lines 2435-2653 |
| 13 | RLHF and model-based RL review | lines 2654-2810 |

## 1. Off-policy policy gradient and importance sampling

**Transcript coverage:** lines 1-167

### What the lecturer said - transcript only

The review resumed from ordinary policy gradient and baselines. Because generating trajectories can be expensive—candidate language-model responses or physical robot trials, for example—it is desirable to take multiple gradient steps from one batch. After the first update, however, the data came from an old policy rather than the current one.

Importance sampling estimates an expectation under $p$ using samples from $q$:

$$
\mathbb E_{x\sim p}[f(x)]
=\mathbb E_{x\sim q}
\left[\frac{p(x)}{q(x)}f(x)\right].
$$

For trajectories, $p$ is the new policy distribution and $q$ is the old one. Initial-state and transition probabilities cancel from their ratio, leaving

$$
\frac{p_{\theta'}(\tau)}{p_\theta(\tau)}
=\prod_{t=1}^{H}
\frac{\pi_{\theta'}(a_t\mid s_t)}
{\pi_\theta(a_t\mid s_t)}.
$$

This exact estimator is unbiased but its variance is impractical. A product of many ratios can vanish or explode exponentially with horizon.

The policy gradient can equivalently be expressed through state-action marginals. Their exact importance ratio contains a state-distribution ratio and an action ratio. The state ratio is unknown, so practical methods omit it and retain only

$$
\frac{\pi_{\theta'}(a_t\mid s_t)}{\pi_\theta(a_t\mid s_t)}.
$$

This omission is biased: infinitely many samples do not recover the exact gradient. The justification is local. When the new policy stays close to the old one, the two state distributions remain close and the resulting objective error is bounded. The lecturer's mental picture was that the approximate objective may differ greatly far away but resemble the true objective in a neighborhood of the old policy.

### Source reconciliation

This segment aligns with Lecture 21 slides 17-18, not with the opening of the Lecture 22 deck. Those slides display the trajectory-ratio cancellation and the one-step approximation.

### Additional explanation

The approximation trades bias for variance. Full-trajectory importance sampling is correct but often unusable; the one-step ratio is stable enough to optimize provided the update is kept local.

## 2. PPO importance-ratio clipping

**Transcript coverage:** lines 168-339

### What the lecturer said - transcript only

There are two PPO mechanisms for limiting departure from the data-generating policy: clip the importance ratio or impose a KL-divergence constraint. Both are used.

Define the one-step ratio

$$
r_t(\theta')
=\frac{\pi_{\theta'}(a_t\mid s_t)}
{\pi_\theta(a_t\mid s_t)}.
$$

When the new policy moves into regions unsupported by old samples, some ratios become very large and others very small. Clipping replaces the ratio with

$$
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon).
$$

This does not constrain the policy parameters themselves. It removes additional objective credit once the sampled action's ratio passes the interval, so the optimizer has little incentive to move farther. The lecturer suggested $\epsilon=0.1$ as a reasonable value to try.

Clipping alone creates a loophole. If a parameter change increases probability on both a good sampled action and a catastrophically bad one, capping the bad action's penalty could make the change appear too favorable. PPO therefore uses the minimum of the unclipped and clipped advantage-weighted terms:

$$
L^{\mathrm{CLIP}}(\theta')
=\mathbb E_t\left[
\min\!\left(
r_t(\theta')\widehat A_t,
\operatorname{clip}(r_t(\theta'),1-\epsilon,1+\epsilon)\widehat A_t
\right)
\right].
$$

This caps gains from favorable samples without capping the cost of making unfavorable samples still more likely. The lecturer called this an important practical detail but not the most profound aspect of PPO.

### Source reconciliation

Lecture 21 slides 19-20 show the clipping rule and the positive-versus-negative-advantage diagrams. The transcript gives the extended “good region containing one terrible outcome” explanation.

### Additional explanation

The minimum makes the surrogate pessimistic. Whenever clipping would make a proposed update look better than its raw ratio does, PPO keeps the worse value.

## 3. PPO implementation and KL-constrained derivation

**Transcript coverage:** lines 340-926

### What the lecturer said - transcript only

Practical clipped PPO repeats this loop:

1. Run the current policy to collect trajectories.
2. Form critic targets and fit the value function.
3. Estimate advantages, usually with GAE.
4. Take several gradient steps on $L^{\mathrm{CLIP}}$ using the same data.
5. Collect a fresh batch and repeat.

Entropy regularization can prevent premature policy collapse and support exploration. Gaussian and categorical entropies are easy to compute. Standardizing advantages by subtracting their mean and dividing by their standard deviation improves gradient conditioning.

The KL-constrained PPO variant has the same goal but constrains policy change more explicitly. Its derivation establishes two facts. First, advantages from the old policy are legitimate because the performance-difference identity gives

$$
J(\theta')-J(\theta)
=\mathbb E_{\tau\sim p_{\theta'}}
\left[\sum_t\gamma^t A^{\pi_\theta}(s_t,a_t)\right].
$$

The old policy's value is a constant with respect to $\theta'$, so replacing a new-policy advantage with the old-policy advantage changes the objective by a harmless constant relationship rather than a crude approximation.

The remaining mismatch is between new-policy states $p_{\theta'}(s_t)$ and old-policy states $p_\theta(s_t)$. For deterministic policies, the same coupling proof as behavioral cloning says that if action disagreement probability is at most $\epsilon$, state-distribution TV grows no faster than $\epsilon t$. For stochastic policies, a maximal coupling constructs a joint distribution whose marginals are the two policies and whose actions agree with probability at least $1-\epsilon$ whenever their TV is at most $\epsilon$. This reduces the stochastic case to the same no-disagreement argument.

Adding and subtracting the old state distribution then bounds the per-time-step objective error by $2\epsilon tC$, where $C$ bounds the action-averaged advantage. An advantage has scale at most $HR_{\max}$, or $R_{\max}/(1-\gamma)$ in discounted form. Summing over time produces another horizon factor. Thus maximizing the computable old-state-distribution surrogate improves the desired objective when the policy constraint keeps the slack sufficiently small.

The lecturer emphasized that this is a bound on objective values before gradients are taken; the surrogate and true gradients need not match. The result says that a small policy-change region prevents their objective values from separating too much.

In practice, KL is more convenient than TV for distributions such as Gaussians. A Lagrangian form penalizes departure from the old policy. Up to constants independent of $\theta'$, forward KL contributes an expectation of $-\log\pi_{\theta'}(a\mid s)$ under old-policy samples. Its multiplier $\beta$ can be adjusted by dual gradient descent: increase the penalty when measured KL is above its target and decrease it when below.

Both clipped and KL-constrained forms are called PPO. The lecturer knew of no universal winner; clipping is common partly because it is extremely easy to implement. Policy gradients are widely used, especially in RL for language models, so students were encouraged to understand these derivations thoroughly.

The discussion then reframed almost all modern policy-gradient algorithms as actor-critic: they contain both a policy and a value function even if convention calls them “policy gradient.” Pure policy-gradient methods are rare; GRPO was mentioned as one example.

### Source reconciliation

This material aligns with Lecture 21 slides 21-28. One transcript rendering spells KL as “kale” and GAE as “GE”; the mathematically and visually supported symbols are used here.

### Additional explanation

PPO's core contract is: reuse a batch, but make only a locally trustworthy change before collecting new data. Clipping and KL penalties are two approximate trust-region mechanisms.

## 4. Online actor-critic and correlated samples

**Transcript coverage:** lines 927-1036

### What the lecturer said - transcript only

The classic online actor-critic algorithm takes one transition $(s_i,a_i,r_i,s_i')$, forms the one-step target

$$
y_i=r_i+\gamma\widehat V_\phi(s_i'),
$$

regresses the critic toward $y_i$, estimates

$$
\widehat A(s_i,a_i)
=r_i+\gamma\widehat V_\phi(s_i')-\widehat V_\phi(s_i),
$$

and updates the policy with $\nabla_\theta\log\pi_\theta(a_i\mid s_i)\widehat A(s_i,a_i)$. Historically, it updated actor and critic each time step with batch size one. That is generally a poor modern choice.

A3C mitigates batch size one by running multiple simulators in parallel—for example, several Atari workers—and combining their simultaneous samples. Simply taking $N$ consecutive steps produces strongly correlated data, which is harmful to stochastic-gradient learning. The lecturer illustrated a regressor repeatedly overfitting one local part of a target function, forgetting others, then receiving huge corrective gradients later.

Monte Carlo estimates are possible, but they require waiting for future rewards. One-step TD is the natural choice if an update must occur immediately. The historical actor-critic lineage then branches into on-policy methods culminating in algorithms such as PPO and off-policy replay-buffer methods such as DDPG, TD3, and SAC.

### Source reconciliation

This discussion aligns with Lecture 21 slides 29-30. The Lecture 22 deck begins with Q-learning and does not contain these opening actor-critic slides.

### Additional explanation

Parallel workers reduce temporal correlation without waiting to accumulate a long sequential minibatch from a single environment. Replay buffers provide a different decorrelation mechanism and enable data reuse.

## 5. $n$-step returns and generalized advantage estimation

**Transcript coverage:** lines 1037-1241

### What the lecturer said - transcript only

An $n$-step advantage estimator is

$$
\widehat A_t^{(n)}
=\sum_{k=0}^{n-1}\gamma^k r_{t+k}
+\gamma^n\widehat V(s_{t+n})
-\widehat V(s_t).
$$

Larger $n$ reduces bias because the possibly inaccurate bootstrapped value is multiplied by $\gamma^n$ and eventually disappears. It increases variance because more of the return is represented by one sampled future rather than an expectation. Thus:

- $n=1$ is attractive when the value function is very accurate;
- $n\to\infty$ becomes Monte Carlo and is attractive in a deterministic system where trajectory variance is zero.

Instead of choosing one $n$, GAE forms an exponentially weighted combination of all $n$-step estimators, with weight proportional to $\lambda^{n-1}$. This choice is somewhat arbitrary but produces a simple implementation:

$$
\widehat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l},
$$

$$
\delta_t=r_t+\gamma\widehat V(s_{t+1})-\widehat V(s_t).
$$

Larger-$n$ contributions receive exponentially smaller weights. The hyperparameter $\lambda$ controls the estimator's bias-variance tradeoff. The discount $\gamma$ has a similar statistical effect but also changes the underlying RL objective, whereas $\lambda$ changes only the estimator. The lecturer was unaware of a standard method that learns more elaborate $n$-step weights, though he agreed it was a reasonable idea.

In a PPO implementation, fit the value function, calculate TD residuals, form GAE, standardize the advantages, and use them in the PPO policy update.

### Source reconciliation

This segment aligns with Lecture 21 slides 31-33, which visualize the bias-variance tradeoff and show the TD-residual form.

### Additional explanation

GAE is an eligibility-trace computation for policy-gradient advantages. Its recursive implementation is efficient: scan backward with $A_t=\delta_t+\gamma\lambda A_{t+1}$.

## 6. Off-policy actor-critic and reparameterization

**Transcript coverage:** lines 1242-1331

### What the lecturer said - transcript only

Off-policy actor-critic stores transitions in a replay buffer, samples nearly independent minibatches, builds Q-function targets, updates the critic, and updates the actor. Reused data makes sampling cheap relative to on-policy rollouts, so variance reduction is less dominant.

The buffer contains states from older policies, and the basic method ignores that state-distribution mismatch. It handles old *actions* with a Q-function: $Q(s,a)$ evaluates the supplied action regardless of which policy generated it, while the next-action expectation is computed under the current policy.

Because the critic is differentiable with respect to a continuous action, the actor commonly uses a reparameterized gradient rather than a score-function gradient. For a Gaussian policy,

$$
a=\mu_\theta(s)+\sigma_\theta(s)\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
$$

and one differentiates $Q(s,a)$ through this deterministic transformation. For discrete actions, the policy expectation can be summed exactly.

GAE cannot be used correctly on arbitrary replay sequences because after one step the stored next action comes from an old behavior policy. Some implementations nevertheless use off-policy multistep returns as an approximation.

### Source reconciliation

This is the final segment corresponding to Lecture 21 slide 34. The transcript's “soft policy” phrase refers to the current-policy next-action expectation used by a soft actor-critic-style update.

### Additional explanation

One-step off-policy bootstrapping separates the observed transition from the current policy: the environment supplies $(s,a,r,s')$, while the learner chooses how to value actions after $s'$.

## 7. Q-learning, target networks, Double Q, and DDPG

**Transcript coverage:** lines 1332-1511

### What the lecturer said - transcript only

Q-learning can be viewed as actor-critic without a separately parameterized actor. Its implicit discrete policy is

$$
\pi(a\mid s)=
\mathbf 1\!\left[a=\arg\max_{a'}Q(s,a')\right].
$$

Plugging that policy into the critic target produces the max:

$$
y_i=r_i+\gamma\max_{a'}Q_{\bar\theta}(s_i',a').
$$

The DQN-style loop collects transitions into replay, samples minibatches, regresses $Q_\theta$ toward targets from a lagged network $Q_{\bar\theta}$, and periodically or gradually updates the target parameters. A separate policy head is possible, but for discrete actions the network already outputs one Q-value per action and the argmax acts like a policy head. Replay buffers and target networks also benefit off-policy actor-critic; they are not unique to Q-learning.

The target network slows changes in regression targets. It is called a target network because its values appear on the right-hand side of the critic regression.

Maximizing a noisy estimator creates positive bias:

$$
\mathbb E[\max(X_1,X_2)]
\ge \max(\mathbb E[X_1],\mathbb E[X_2]).
$$

The argmax selects both genuine value and positive estimation noise. Double Q-learning uses one estimator to select an action and another to evaluate it:

$$
y=r+\gamma Q_B
\left(s',\arg\max_{a'}Q_A(s',a')\right).
$$

In practice, the current and target networks serve these two roles. They are not independent, but are decorrelated enough to reduce overestimation.

DDPG addresses continuous actions by giving actor-critic a deterministic actor $\mu_\theta(s)$ that approximately performs the Q-function argmax. Train it by backpropagating

$$
\nabla_\theta Q_\phi(s,\mu_\theta(s)).
$$

This requires a derivative with respect to action and therefore assumes a continuous action space.

### Source reconciliation

This is where the spoken review begins using the Lecture 22 deck, primarily slides 2-9. The slides also show the dynamic-programming route to fitted Q-iteration; the lecturer said that derivation was worth reviewing but used the actor-critic-without-an-actor interpretation for the concise recap.

### Additional explanation

The target network stabilizes temporal targets; Double Q addresses maximization bias. They solve different problems even though practical Double DQN often uses the same pair of current and target networks for both.

## 8. Why fitted value iteration need not converge

**Transcript coverage:** lines 1512-1690

### What the lecturer said - transcript only

Tabular value iteration and Q-value iteration converge to their fixed points. With nonlinear function approximation, general convergence is not guaranteed.

The Bellman operator $\mathcal B$ is a contraction in the infinity norm:

$$
\lVert\mathcal BV-\mathcal B\widetilde V\rVert_\infty
\le\gamma\lVert V-\widetilde V\rVert_\infty.
$$

Supervised projection $\Pi$ onto a representable function class is a contraction in a Euclidean or two-norm sense:

$$
\lVert\Pi V-\Pi\widetilde V\rVert_2
\le\lVert V-\widetilde V\rVert_2.
$$

Fitted value iteration composes them: $V\leftarrow\Pi\mathcal BV$. Two contractions in different norms need not compose into a contraction in any norm. Geometrically, a Bellman backup can move toward $V^*$ in infinity norm, after which projection onto the function class moves the approximation farther from $V^*$ overall.

The counterexample requires a restricted function class. Very expressive neural networks may make the pathological geometry rare, which is one possible reason deep Q-learning often works in practice despite the absence of a general guarantee. The conclusion to remember is: tabular value iteration converges; fitted nonlinear value iteration need not.

The class then took a break and resumed at 9:18.

### Source reconciliation

Slide 10 provides the carefully drawn geometric counterexample. The transcript explicitly calls it a proof sketch rather than a full proof.

The displayed nonexpansiveness claim for $\Pi$ is guaranteed for exact orthogonal metric projection onto a closed convex set or subspace. A neural-network function class is generally nonconvex, and an SGD fit is only approximate, so its projection step need not satisfy this inequality. This caveat strengthens the lecture's conclusion that nonlinear fitted value iteration lacks a general convergence guarantee; it is not inserted into the transcript-only layer.

### Additional explanation

This interaction among function approximation, bootstrapping, and off-policy data is commonly associated with the “deadly triad.” The lecture's argument isolates the first two ingredients at the operator level.

For a closed convex class $\mathcal F$ with exact Euclidean projection, $\Pi_{\mathcal F}$ is nonexpansive. For a nonconvex neural class, “projection” is best read as shorthand for a supervised fitting operation, not as a mathematical projector with that guarantee.

## 9. Variational inference and amortized inference

**Transcript coverage:** lines 1691-1980

### What the lecturer said - transcript only

Variational inference is a general tool used in several RL contexts: control as inference and inverse RL, learning latent dynamics models, and representation learning.

A latent-variable model introduces a simple prior $p(z)$ and a decoder $p_\theta(x\mid z)$ so that their nonlinear combination can represent a complex data distribution. The mapping may be deterministic, as in related flow methods, but probabilistic notation gives a general derivation.

Marginal likelihood is

$$
\log p_\theta(x)
=\log\int p_\theta(x\mid z)p(z)\,dz,
$$

whose integral is generally intractable. Introduce any tractable $q_i(z)$, multiply and divide by it, and view the integral as an expectation. Jensen's inequality for concave $\log$ gives

$$
\begin{aligned}
\log p_\theta(x_i)
&\ge
\mathbb E_{z\sim q_i}
\left[
\log p_\theta(x_i\mid z)
+\log p(z)
-\log q_i(z)
\right]\\
&=
\mathbb E_{q_i}[\log p_\theta(x_i\mid z)]
-D_{\mathrm{KL}}(q_i(z)\Vert p(z)).
\end{aligned}
$$

The negative expectation of $\log q_i$ is its entropy. The inequality is valid for any $q_i$, but the tightest choice is the posterior $p(z\mid x_i)$. The prior describes $z$ before observing $x$; $q_i$ approximates the posterior for a particular point.

Amortized inference uses one encoder network $q_\phi(z\mid x)$ for all examples instead of separately optimizing each $q_i$. A VAE therefore contains an encoder and decoder. The encoder typically outputs a Gaussian mean and variance, and the reparameterization

$$
z=\mu_\phi(x)+\sigma_\phi(x)\epsilon,
\qquad\epsilon\sim\mathcal N(0,I)
$$

allows gradients through samples. With Gaussian posterior and prior, the KL has a closed form. The decoder outputs parameters of $p_\theta(x\mid z)$. Conditional VAEs additionally feed conditioning information into encoder and decoder; a policy model could take an observation as condition and treat the action as $x$.

### Source reconciliation

Slides 11-18 show Jensen's derivation, the ELBO/KL identity, amortization, reparameterization, conditional models, and a diffusion/flow relationship. The transcript discusses the first five of these and only alludes generally to more advanced VAEs.

### Additional explanation

The two ELBO terms balance reconstruction and latent regularity. The decoder should explain each datum from its sampled latent, while the approximate posterior should not drift arbitrarily far from the prior.

## 10. Structured state-space models

**Transcript coverage:** lines 1981-2072

### What the lecturer said - transcript only

A latent state-space model applies the same framework to a partially observed trajectory. Observations are $o_{1:T}$ and unobserved states are $z_{1:T}$. For variational inference, the latent variable is the entire concatenated sequence, not one $z_t$.

The structured prior is

$$
p(z_{1:T}\mid a_{1:T-1})
=p(z_1)\prod_{t=1}^{T-1}p(z_{t+1}\mid z_t,a_t),
$$

and a factorized decoder is

$$
p_\theta(o_{1:T}\mid z_{1:T})
=\prod_{t=1}^{T}p_\theta(o_t\mid z_t).
$$

The encoder or variational posterior is a modeling choice. A common filtering posterior predicts $z_t$ from current and past observations and actions. Any tractable $q$ gives a valid lower bound, although its structure affects tightness and ease of training.

Independent decoding from each $z_t$ encourages the latent state to contain the current information needed for a Markov representation. A more history-dependent decoder could be more expressive, but if the resulting $z_t$ is not Markov, it cannot be inserted directly into a Q-learning algorithm. Modeling choices must therefore reflect the downstream use of the latent state.

### Source reconciliation

This discussion aligns with slide 19. A visually similar state-space slide is repeated later as slide 43 in the unused model-based section.

### Additional explanation

The learned latent dynamics provide a compact predictive state. The encoder is needed during training and inference from observations; planning or value learning can operate on latent transitions once the state is inferred.

## 11. Control as inference and maximum-entropy RL

**Transcript coverage:** lines 2073-2434

### What the lecturer said - transcript only

To model approximately rational behavior, control as inference augments an MDP with binary optimality variables $O_t$. Assuming rewards are nonpositive for normalization, define

$$
p(O_t=1\mid s_t,a_t)=\exp(r(s_t,a_t)).
$$

Conditioning on all optimality variables being true gives

$$
p(\tau\mid O_{1:T}=1)
\propto
p(\tau)\exp\!\left(\sum_t r(s_t,a_t)\right).
$$

Natural dynamics still determine which trajectories are possible, while reward makes higher-return trajectories exponentially more likely. Equal-quality actions remain equally likely; small mistakes are plausible and large mistakes are exponentially unlikely. This is a reasonable noisy model of human or animal behavior and avoids explaining every demonstration imperfection as a reward feature.

Backward messages are

$$
\beta_t(s_t,a_t)=p(O_{t:T}=1\mid s_t,a_t),
$$

with a state message obtained by integrating actions. Their recursive equations resemble Bellman backups after taking logs. Bayes' rule gives the posterior policy as the state-action message divided by the state message, and therefore

$$
\pi(a_t\mid s_t)
=\exp(Q_t(s_t,a_t)-V_t(s_t))
=\exp(A_t(s_t,a_t)).
$$

The direct probabilistic recursion is overly optimistic under stochastic transitions: a high-value lucky next state dominates a log-expectation-exp. In hindsight, inference can explain success either through a good action or a lucky transition. The lecturer's lottery example illustrated why that is a bad *plan*: observing sudden wealth may make a winning ticket a plausible explanation, but buying a ticket is not a reliable strategy.

Variational control corrects this by choosing

$$
q(s_{1:T},a_{1:T})
=p(s_1)
\prod_t p(s_{t+1}\mid s_t,a_t)q(a_t\mid s_t).
$$

It forces initial-state and transition factors to match the real MDP and leaves only the controllable action distribution free. When inserted into the ELBO, matching environment terms cancel and the objective becomes

$$
\mathbb E_q\left[
\sum_t r(s_t,a_t)+\mathcal H(q(a_t\mid s_t))
\right].
$$

Thus the control objective maximizes both reward and policy entropy. Entropy regularization in implementations such as PPO can be viewed through this framework rather than only as an exploration heuristic. The resulting dynamic program uses expected next values and a soft maximum over actions instead of the optimistic transition log-sum-exp.

### Source reconciliation

Slides 20-28 contain the graphical model, backward messages, policy computation, ELBO cancellation, and variational backward pass. The transcript skips some displayed algebra but explains its meaning and the stochastic-transition correction at length.

### Additional explanation

“Soft” does not mean approximate here. It means that maximization is entropy-regularized, producing a stochastic Boltzmann policy rather than a hard argmax.

## 12. Soft actor-critic, inverse RL, and adversarial imitation

**Transcript coverage:** lines 2435-2653

### What the lecturer said - transcript only

Soft actor-critic implements the maximum-entropy objective in an off-policy actor-critic algorithm. The policy objective includes entropy, and the critic target must credit future entropy:

$$
y=r(s,a)+\gamma
\mathbb E_{a'\sim\pi(\cdot\mid s')}
\left[Q_{\bar\phi}(s',a')-\alpha\log\pi(a'\mid s')\right].
$$

The multiplier $\alpha$ or temperature trades off reward scale and entropy.

Inverse RL parameterizes a reward $r_\psi$ and maximizes the likelihood of expert trajectories under the soft-optimal trajectory model. The log likelihood contains demonstrated reward minus a log partition function

$$
Z_\psi=\int p(\tau)\exp(r_\psi(\tau))d\tau.
$$

Differentiation produces

$$
\nabla_\psi\mathcal L
=
\mathbb E_{\tau\sim p^*}
[\nabla_\psi r_\psi(\tau)]
-
\mathbb E_{\tau\sim p_{\psi,\mathrm{soft}}}
[\nabla_\psi r_\psi(\tau)].
$$

Raise reward on expert demonstrations and lower it on trajectories from the current reward-optimal policy. The gradient vanishes when the two distributions match. The expert expectation is easy to estimate; the model expectation would naively require solving RL to convergence after every reward update. Practical algorithms interleave partial policy optimization and reward updates, sometimes with importance sampling.

Unlike behavioral cloning, inverse RL is dynamics-aware and can avoid BC's compounding-error problem when environment interaction and RL are practical.

The same distribution-matching idea can be implemented as a GAN. Treat the policy as a generator, train a binary discriminator to distinguish policy samples from demonstrations, and use a function of its output as reward. This is simpler, but at convergence the discriminator outputs one half everywhere and does not retain a meaningful recovered reward. The result is primarily a policy.

### Source reconciliation

Slides 29-33 show SAC, the partition-function gradient, the GAN connection, and the regular-discriminator variant. Slide 34 begins RLHF, which continues in the next section.

### Additional explanation

Maximum-entropy inverse RL matches occupancy distributions, not just individual actions. That is the source of its dynamics awareness and also of its computational expense.

## 13. RLHF and model-based RL review

**Transcript coverage:** lines 2654-2810

### What the lecturer said - transcript only

RL from human feedback learns a reward from expressed preferences rather than demonstrations. Sample two or more policy trajectories, ask a person which is preferred, and interpret preference $\tau_i\succ\tau_j$ as evidence that $\tau_i$ has larger total reward. A Bradley-Terry/logistic model uses

$$
p(\tau_i\succ\tau_j)
=
\frac{
\exp\left(\sum_t r_\psi(s_t^{(i)},a_t^{(i)})\right)
}{
\exp\left(\sum_t r_\psi(s_t^{(i)},a_t^{(i)})\right)
+
\exp\left(\sum_t r_\psi(s_t^{(j)},a_t^{(j)})\right)
}.
$$

Train the reward as a binary classifier's logits, then optimize the policy against that learned reward. The sigmoid model also represents noisy preferences.

With little time remaining, the lecturer highlighted model-based RL rather than reviewing every remaining topic. Its central difficulty is distribution shift. Train a model on current data, improve the policy under that model, then collect real data where the improved policy goes. Adding those transitions corrects the model in regions the policy is likely to exploit.

An alternative is to keep the policy in regions where the model is confident. Output entropy measures aleatoric or environmental randomness and is not enough. What matters for trust is epistemic/model uncertainty—uncertainty about the learned model because data are limited. A standard estimate is an ensemble: train multiple networks and either sample a member for rollouts or use their disagreement.

Open-loop planning commits to an action sequence before seeing future states and is effective in deterministic environments. It is suboptimal under important stochastic observations. A student cannot precommit to answers before seeing a randomly selected math test, even if they know how to solve every possible question. Closed-loop control returns a policy whose next action responds to the observed state; RL naturally targets this form.

Common model-based policy-learning methods insert short model rollouts into an off-policy model-free learner. Start branches from states sampled throughout the real replay buffer, roll out the current policy for a short horizon in the learned model, and train the critic or actor-critic on the resulting synthetic transitions. Full model rollouts accumulate too much model error; short rollouts only from initial states miss later parts of trajectories. Replay-started short branches are a compromise, although their starting-state distribution comes from older policies.

The lecturer stopped before reviewing offline RL and the later topics, noting that they were more recent and hopefully fresher. He wished students luck on the exam and said class would resume Wednesday.

### Source reconciliation

RLHF is on slide 34. After the intermission marker, the spoken model-based discussion aligns with slides 36-42. Slide 43 onward was not reached.

### Additional explanation

Branched rollouts use the model where it is most reliable: for short local predictions from real states. They trade synthetic-data volume against compounding model bias.

## Slide-only appendix: material not reached

Slides 43-73 of the Lecture 22 deck were not discussed in the supplied transcript. For source completeness, they contain recap material on:

- repeated latent state-space modeling;
- offline RL distribution shift in Q-learning, model-based RL, and importance-weighted policy gradient;
- policy constraints, behavior cloning regularization, AWAC, IQL, CQL, IDQL, diffusion/flow Q-learning, and MOPO;
- optimism, count bonuses, pseudo-counts, and random-network-distillation-style novelty;
- value-iteration contraction, oracle model estimation, and fitted-Q error propagation from Lecture 20.

These slides are not represented as spoken Lecture 22 material.

## Consolidated takeaways

- Exact trajectory importance sampling is unbiased but its variance grows catastrophically with horizon.
- PPO uses a biased one-step surrogate and keeps it trustworthy through clipping or a KL-based local constraint.
- GAE combines all $n$-step estimators with exponential weights to navigate bias versus variance.
- Off-policy actor-critic and Q-learning reuse replay data; Q-functions allow actions from old policies to be evaluated without an action importance ratio.
- Q-learning is actor-critic with an implicit argmax actor; Double Q reduces maximization bias, while target networks stabilize regression.
- Bellman contraction plus function projection does not guarantee nonlinear fitted-value convergence because the contractions use different norms.
- Variational inference converts an intractable latent marginal likelihood into a tractable lower bound and supports latent dynamics models.
- Control as inference yields entropy-regularized RL; SAC implements its soft Bellman structure.
- Inverse RL learns from demonstrations, RLHF learns from preferences, and both alternate reward learning with policy optimization.
- Model-based RL must manage model exploitation through new data, epistemic uncertainty, or short branched rollouts.

## Key equations

1. **Importance sampling**

   $$
   \mathbb E_p[f]=\mathbb E_q\left[\frac pq f\right].
   $$

2. **PPO clipped surrogate**

   $$
   L^{\mathrm{CLIP}}
   =\mathbb E_t\left[\min(r_t\widehat A_t,
   \operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t)\right].
   $$

3. **Generalized advantage estimation**

   $$
   \widehat A_t^{\mathrm{GAE}}
   =\sum_{l\ge0}(\gamma\lambda)^l
   \left(r_{t+l}+\gamma V_{t+l+1}-V_{t+l}\right).
   $$

4. **DQN target**

   $$
   y=r+\gamma\max_{a'}Q_{\bar\theta}(s',a').
   $$

5. **Variational lower bound**

   $$
   \log p_\theta(x)
   \ge\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
   -D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)).
   $$

6. **Maximum-entropy control objective**

   $$
   \max_\pi\mathbb E_\pi\left[
   \sum_t r(s_t,a_t)+\alpha\mathcal H(\pi(\cdot\mid s_t))
   \right].
   $$

## Glossary

- **Importance ratio:** Likelihood of a sample under a target policy divided by its likelihood under the behavior policy.
- **PPO:** Proximal Policy Optimization, using clipping or a KL mechanism to keep reused-data updates local.
- **Maximal coupling:** Joint construction that makes samples from two distributions agree as often as total variation permits.
- **GAE:** Exponentially weighted mixture of multistep advantage estimators.
- **Replay buffer:** Stored transition collection sampled to reuse and decorrelate data.
- **Target network:** Lagged value network used to construct more stable bootstrap targets.
- **Double Q-learning:** Separate selection and evaluation to reduce maximization bias.
- **ELBO:** Evidence lower bound optimized in variational inference.
- **Amortized inference:** Shared model that predicts approximate posteriors for many observations.
- **Optimality variable:** Auxiliary random variable used to express reward-seeking as probabilistic inference.
- **Soft actor-critic:** Off-policy maximum-entropy actor-critic algorithm.
- **Inverse RL:** Inferring a reward or objective from demonstrated behavior.
- **Epistemic uncertainty:** Uncertainty in a learned model due to limited knowledge or data.
- **Open-loop plan:** Fixed action sequence that cannot react to future observations.
- **Closed-loop policy:** State-dependent decision rule that reacts during execution.

## Self-check questions

1. Why does the exact trajectory importance ratio have high variance?
2. Which part of the exact state-action ratio does PPO omit?
3. Why does PPO take a minimum of clipped and unclipped objectives?
4. How does the imitation-learning coupling proof reappear in the PPO guarantee?
5. Explain the bias-variance effect of increasing $n$ in an $n$-step return.
6. Why is one-step bootstrapping natural for arbitrary replay data?
7. Distinguish target networks from Double Q-learning.
8. Why can two contractions fail to make fitted value iteration converge?
9. Derive the ELBO using Jensen's inequality.
10. Why does variational control force transition probabilities in $q$ to match the real MDP?
11. How does the inverse-RL gradient compare expert and current-policy trajectories?
12. Why is ensemble disagreement more relevant than predictive entropy for model trust?

## Source coverage checklist

- [x] All supplied transcript lines 1-2810 are mapped exactly once in increasing, non-overlapping ranges.
- [x] Administrative transitions, break, questions, optional asides, and stopping remarks are retained.
- [x] The continuation deck, primary deck, unused slides, transcript layer, and added explanation are explicitly separated.
- [x] All 73 pages of the primary deck and the relevant continuation slides were visually inspected.
- [x] Displayed mathematics uses Markdown-compatible LaTeX delimiters.

**Coverage result:** All 2,810 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. The Lecture 21 deck continuation and unused Lecture 22 slides are separately identified.
