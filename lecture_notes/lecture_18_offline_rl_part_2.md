---
title: "Lecture 18 - Offline RL, Part 2"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 18
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 18, Offline RL Algorithms.txt"
source_slides: "../lectures/Lecture 18 - Offline RL Part 2.pdf"
transcript_lines: 8149
slide_pages: 35
status: "complete"
---

# Lecture 18: Offline RL, Part 2

## Lecture map

**Implementation convention.** In every Bellman target, mask true terminals with $m=0$, use $m=1$ otherwise, and detach the full target. External rollout cutoffs retain a bootstrap from the final pre-reset state. Critic parameters and advantage weights are held fixed during actor fitting.

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Offline-RL recap | lines 1-190 |
| 2 | Forward and reverse KL review | lines 191-778 |
| 3 | Choosing a divergence and the ideal support constraint | lines 779-1126 |
| 4 | Explicit policy-constraint placements | lines 1127-1708 |
| 5 | Actor-critic with behavioral cloning | lines 1709-2077 |
| 6 | Advantage-weighted regression and AWAC | lines 2078-3001 |
| 7 | Implicit Q-learning | lines 3002-4351 |
| 8 | Pessimism and conservative Q-learning | lines 4352-5600 |
| 9 | The offline-to-online transition | lines 5601-6415 |
| 10 | Why expressive diffusion and flow policies may help | lines 6416-6961 |
| 11 | IDQL: behavior modeling plus best-of-$N$ | lines 6962-7483 |
| 12 | FQL: distilling a flow behavior model into an actor | lines 7484-7837 |
| 13 | Diffusion steering and deferred material | lines 7838-8149 |

## Part I - Policy constraints

## 1. Offline-RL recap

**Transcript coverage:** lines 1-190

### What the lecturer said - transcript only

The lecturer briefly recapped why ordinary deep RL fails with a fixed dataset. Policy improvement queries the critic or model on actions not represented in the behavior data. Optimization selects favorable extrapolation errors, and no new interaction is available to correct them.

The three remedy families from Part 1 were restated: constrain the policy toward the behavior distribution, make value estimates pessimistic for unsupported actions, or design targets that never query unsupported actions. This lecture would turn those principles into algorithms.

### Source reconciliation

Slides 1-2 list the same three principles and mark the transition from motivation to concrete offline-RL algorithms.

### Additional explanation

These remedies target different points in the actor-critic loop. Policy constraints alter the actor, pessimism alters the critic, and in-distribution target construction alters the data used by the Bellman backup.

## 2. Forward and reverse KL review

**Transcript coverage:** lines 191-778

### What the lecturer said - transcript only

The lecturer reviewed the two KL directions. Reverse KL averages under the learned policy and penalizes it for choosing actions with low behavior probability. Forward KL averages under the behavior distribution and penalizes the learned policy for assigning too little probability to recorded actions.

With a restricted unimodal policy and multimodal data, forward KL tends to cover or average modes, while reverse KL tends to select one mode. The former resembles maximum-likelihood behavioral cloning. The latter resembles maximizing reward with an added behavior log-probability and entropy term.

The lecturer used the mode diagrams to emphasize that neither direction is universally correct. Their behavior depends on the expressiveness of the policy class and on whether mode coverage or best-mode selection is desired.

### Source reconciliation

Slides 3-4 display

$$
D_{\mathrm{KL}}(\pi_\beta\|\pi_\theta)
=\mathbb E_{a\sim\pi_\beta}
\left[\log\frac{\pi_\beta(a\mid s)}{\pi_\theta(a\mid s)}\right]
$$

and

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_\beta)
=\mathbb E_{a\sim\pi_\theta}
\left[\log\frac{\pi_\theta(a\mid s)}{\pi_\beta(a\mid s)}\right].
$$

Automatic transcript renderings of KL as "Kale" are reconciled from the slide labels.

### Additional explanation

If the policy class can exactly represent the behavior distribution, both KL directions have the same zero-divergence optimum. Their mode-covering versus mode-seeking distinction is most visible under approximation constraints or when the policy is deliberately allowed to improve away from exact behavior matching.

## 3. Choosing a divergence and the ideal support constraint

**Transcript coverage:** lines 779-1126

### What the lecturer said - transcript only

Forward KL is often used because it can be estimated from the offline samples without evaluating $\pi_\beta(a\mid s)$. It preserves behavior modes, which may matter for broad response distributions such as language. Reverse KL is attractive when the goal is to choose the best behavior within the dataset, but it requires fitting or otherwise evaluating the behavior density.

Ideally, an offline policy would be allowed to do anything that has adequate support in the data and would be penalized only for truly unsupported actions. Such a support indicator is difficult to estimate in continuous action spaces, where an exact action almost always has zero empirical count.

In response to a question about multimodal behavior, the lecturer noted that a sufficiently expressive policy, such as a flow or diffusion model, can represent several modes without smearing mass between them. This observation later becomes central to offline-to-online algorithms.

### Source reconciliation

Slide 4 states the practical forward/reverse trade-offs. The idealized support penalty on slide 4 is of the form

$$
\mathbb E_{a\sim\pi_\theta}
\left[\mathbf 1\{\pi_\beta(a\mid s)<\epsilon\}\right],
$$

where a small threshold is more meaningful than literal empirical equality to zero in continuous spaces.

### Additional explanation

A divergence penalizes changes even when both old and new actions are well supported. A support constraint would permit more aggressive improvement inside the reliable action set. The difficulty is that density is not support: a normalized density can be small in a valid high-dimensional region, and a learned density can assign probability to interpolations absent from the real data.

## 4. Explicit policy-constraint placements

**Transcript coverage:** lines 1127-1708

### What the lecturer said - transcript only

Policy constraints can enter an actor-critic algorithm in several mathematically related locations.

One option modifies only the actor objective. A reverse-KL regularizer produces an expected-Q objective with $\log\pi_\beta(a\mid s)$ and policy entropy terms. A forward-KL regularizer adds a behavioral-cloning log-likelihood term on dataset actions. Algorithms often called SAC+BC use this simple construction.

A second option treats the divergence as part of reward. Then future constraint violations enter the critic through Bellman backups rather than being charged only at the present actor update. This is natural for reverse KL but requires a behavior model.

A BRAC-like implementation includes a divergence penalty in the critic target and actor objective. The lecturer emphasized that the same regularized-control idea can be written in several places; differences in implementation and estimation still matter even when the ideal objectives are closely related.

### Source reconciliation

Slides 5-7 show the generic regularized actor objective

$$
\max_\theta\;
\mathbb E_{s\sim\mathcal D,\,a\sim\pi_\theta}
[Q_\phi(s,a)]-\lambda D(\pi_\theta,\pi_\beta),
$$

and illustrate actor-only and reward/target placements. BRAC refers to behavior-regularized actor critic.

### Additional explanation

Putting a penalty in reward changes the learned notion of long-term value: the critic anticipates future departures from behavior. Putting it only in the actor constrains the current update while the critic may still estimate an unconstrained return. With exact optimization the formulations can be related, but approximation error makes the distinction practical.

These placements are not generally equivalent even with exact optimization unless the critic definition and policy-evaluation objective are adjusted consistently. A one-state regularized improvement step and an objective penalizing every future policy decision solve different problems. Divergences in dataset-based objectives are normally averaged over sampled states, not guaranteed uniformly over every state.

## 5. Actor-critic with behavioral cloning

**Transcript coverage:** lines 1709-2077

### What the lecturer said - transcript only

Actor-critic plus behavioral cloning is a straightforward baseline. Train the critic with the usual off-policy temporal-difference target. Train the actor to maximize predicted Q while also maximizing likelihood of the recorded action at each dataset state. The cloning term is a forward-KL constraint up to constants independent of the learned policy.

The algorithm is easy to add to an existing actor-critic implementation and often works when its coefficient is tuned. It does not perfectly solve distribution shift: an overly weak cloning term permits OOD actions, while an overly strong term collapses toward pure imitation.

Q&A focused on implementation. Dataset actions are used directly for the log-likelihood term; actions sampled from the current actor are used for the Q-maximization term.

### Source reconciliation

Slide 8 gives the actor objective in the schematic form

$$
\max_\theta\;
\mathbb E_{s\sim\mathcal D,\,a\sim\pi_\theta}[Q_\phi(s,a)]
+\lambda\mathbb E_{(s,a)\sim\mathcal D}[\log\pi_\theta(a\mid s)].
$$

### Additional explanation

The two expectations use different actions at the same sampled states. This matters in code: replacing the dataset action in the BC term with an actor sample removes the behavior constraint.

## Part II - Implicit constraints and in-distribution targets

## 6. Advantage-weighted regression and AWAC

**Transcript coverage:** lines 2078-3001

### What the lecturer said - transcript only

The lecturer derived an implicit policy constraint. Optimize expected Q subject to a reverse-KL constraint from the behavior policy. The optimal nonparametric policy is proportional to the behavior policy multiplied by an exponentiated advantage. The value baseline may be subtracted because it does not change relative action probabilities at a fixed state.

Directly normalizing or sampling from this policy is inconvenient. Instead, use behavior-policy samples and fit a parametric actor by weighted maximum likelihood. Each dataset action receives weight $\exp(A(s,a)/\lambda)$. The practical objective drops the state-dependent normalizer; this is not generally an exact cancellation for a shared actor, as explained below.

This gives advantage-weighted regression (AWR) and, in an actor-critic form, advantage-weighted actor critic (AWAC). Train Q with offline targets, obtain or approximate a state value, compute dataset advantages, and perform weighted behavioral cloning.

The advantages are practical stability and simple supervised actor updates. The limitations are important. Temperature is sensitive. Very bad data actions receive nearly zero weight rather than an explicit negative update. With finite data and a restricted actor, weighted regression may struggle to improve much beyond behavior cloning. The projection uses forward KL to fit a reverse-KL-derived target, so it can inherit mode-averaging behavior. The lecturer characterized it as stable but sometimes slow to improve.

### Source reconciliation

Slides 9-10 show

$$
\pi^*(a\mid s)=\frac{1}{Z(s)}
\pi_\beta(a\mid s)\exp\!\left(\frac{A(s,a)}{\lambda}\right),
$$

and the weighted actor fit

$$
\max_\theta\;
\mathbb E_{(s,a)\sim\mathcal D}
\left[
\exp\!\left(\frac{A(s,a)}{\lambda}\right)
\log\pi_\theta(a\mid s)
\right].
$$

### Additional explanation

As $\lambda$ decreases, weights concentrate on the highest-advantage samples and effective sample size falls. As $\lambda$ increases, weights flatten and the actor approaches ordinary behavioral cloning. Weight clipping or normalization is often used to control variance, though it changes the exact projection.

**Normalizer correction:** $Z(s)$ does not generally cancel from a forward-KL projection averaged over states. The exact normalized projection weights are $e^{A(s,a)/\lambda}/Z(s)$. Omitting $Z(s)$ reweights states; it preserves a separate unrestricted per-state optimum, but can change the optimum of a shared parametric actor. Subtracting a state baseline leaves the normalized target policy unchanged, yet can also change cross-state weights in the unnormalized regression objective.

The reverse-KL solution follows from maximizing $\mathbb E_\pi Q-\lambda D_{\rm KL}(\pi\|\pi_\beta)$ at each state. This derivation does not mean AWR and AWAC have identical critic estimation: AWAC uses off-policy actor-critic evaluation, while AWR's return/value fitting can differ.

## 7. Implicit Q-learning

**Transcript coverage:** lines 3002-4351

### What the lecturer said - transcript only

Implicit Q-learning (IQL) is designed so the critic never evaluates an OOD action. The lecturer considered two similar states represented in the data: one recorded action leads to return $-10$ and another to $+10$. An ordinary behavior-value regression would average them near zero. IQL instead fits a state value toward an upper expectile of the Q-values of dataset actions.

An expectile is not a quantile. It uses an asymmetric squared loss: errors on one side receive a larger weight. With an upper expectile parameter, the fitted $V(s)$ moves toward the high supported Q-values without explicitly taking a maximum over new actions.

The Q target is $r+\gamma V(s')$. Because $V(s')$ was fitted only from actions recorded at $s'$, the target never queries an actor-generated OOD action. Transition randomness remains averaged in Q; the optimistic selection occurs only over actions represented in the data.

The lecturer related this to optimizing within a supported action set

$$
\Omega(s)=\{a:\pi_\beta(a\mid s)\ge\epsilon\}.
$$

After fitting $Q$ and $V$, extract a policy with advantage-weighted regression. The critic and actor are partially decoupled. IQL is stable and simple, but the same weighted-regression extraction can limit final policy quality. Other extraction rules, including actor-critic plus BC, can be paired with the critic.

### Source reconciliation

Slides 11-15 use the residual $x=V(s)-Q(s,a)$ and display the piecewise loss

$$
\ell_2^\tau(x)=
\begin{cases}
(1-\tau)x^2, & x>0,\\
\tau x^2, & x\le 0.
\end{cases}
$$

Equivalently, for $u=Q(s,a)-V(s)$ this is

$$
L_2^\tau(u)=|\tau-\mathbf 1\{u<0\}|u^2
$$

and the IQL objectives

$$
\min_V\;\mathbb E_{(s,a)\sim\mathcal D}
[L_2^\tau(Q_{\bar\phi}(s,a)-V(s))],
$$

$$
\min_Q\;\mathbb E_{(s,a,r,s')\sim\mathcal D}
[(r+\gamma V(s')-Q_\phi(s,a))^2].
$$

The slide equation resolves the transcript's verbal ambiguity about which side receives weight $\tau$; the correction is disclosed here rather than silently inferred.

### Additional explanation

For $\tau>1/2$, positive residuals $Q-V>0$ receive weight $\tau$ and pull $V$ upward more strongly than negative residuals pull it downward. Unlike a sample maximum, the expectile changes smoothly with every supported action and is less sensitive to a single noisy high estimate.

For equally likely values $-10$ and $+10$, the upper expectile is $V=20\tau-10$; at $\tau=0.8$, it is 6, not the 80th percentile (which is 10). It remains sensitive to squared-error outliers. A finite $\tau<1$ produces a soft preference among dataset actions; it is not exactly a hard maximum over a density-thresholded set $\Omega(s)$.

IQL avoids **explicit actor-generated action queries** in critic targets. Function approximation still generalizes across nearby states and can be wrong, and the extracted actor can produce OOD actions. High expectiles are applied to estimates of expected action value, not directly to lucky one-transition returns; the Q regression averages transition randomness. This separation is central to the method.

The [original IQL paper](https://arxiv.org/abs/2110.06169) derives this separation between action selection through expectiles and ordinary transition averaging.

## Part III - Pessimistic critics

## 8. Pessimism and conservative Q-learning

**Transcript coverage:** lines 4352-5600

### What the lecturer said - transcript only

Pessimistic offline RL is especially convenient for theoretical analysis and finite-sample bounds, though not always the easiest approach to implement. For a practical continuous-control baseline, the lecturer would often suggest IQL; for proving sample-complexity or regret properties, pessimism is particularly attractive.

The motivating analogy is adversarial training. A policy or adversarial action distribution $\mu$ searches for actions with high Q-values, including spurious peaks. The critic is trained to push those values down in addition to minimizing its ordinary Bellman error. With a sufficiently large coefficient, this first attempt can force Q below the true Q-function, but it systematically underestimates even supported good actions because $\mu$ never stops finding the current maximum.

Conservative Q-learning (CQL) balances that downward pressure by pushing Q-values upward on dataset actions with the same coefficient. If $\mu$ selects an action close to the data, the two terms cancel approximately. If it selects an OOD action, its value is pushed down while data-action values are pushed up. Under suitable conditions and a carefully chosen coefficient, the expected value of the learned policy can be a lower bound even when every individual action value is not.

The critic can then be paired with an argmax policy for discrete actions or a standard continuous actor. If CQL succeeds, OOD actions have low values and will not be selected.

The adversarial distribution $\mu$ should be regularized, commonly with entropy, so it covers a broad set of high-Q actions rather than flip-flopping between narrow maxima. The entropy-regularized optimum is proportional to $\exp Q$. For discrete actions, its contribution becomes a log-sum-exp and can be computed exactly. For continuous actions, samples from an actor or proposal can be importance weighted. Implementing the analytic or sampled form avoids learning an explicit adversary and its GAN-like instability.

### Source reconciliation

Slides 17-21 give the CQL critic objective in the form

$$
\min_Q\;
\alpha\left(
\mathbb E_{s\sim\mathcal D,\,a\sim\mu(\cdot\mid s)}[Q(s,a)]
-\mathbb E_{(s,a)\sim\mathcal D}[Q(s,a)]
\right)
+\frac12\mathbb E_{\mathcal D}[(Q(s,a)-y)^2],
$$

with entropy-regularized adversary

$$
\max_\mu\;
\mathbb E_{a\sim\mu}[Q(s,a)]+\mathcal H(\mu),
\qquad
\mu^*(a\mid s)\propto\exp(Q(s,a)).
$$

For discrete actions, substituting $\mu^*$ produces $\log\sum_a\exp Q(s,a)$. The transcript's "log sumx" is reconciled to log-sum-exp.

### Additional explanation

CQL penalizes the critic where an optimizer is likely to look, not uniformly over the entire action space. The data-action subtraction is essential: without it, the trivial solution of making every Q-value extremely negative would satisfy pessimism but destroy useful ranking.

More precisely, the subtraction removes the incentive for a uniform downward shift from the **conservative regularizer**: $\log\sum_a e^{Q_a-c}-\mathbb E_\beta[Q_a-c]$ is unchanged by $c$. The Bellman regression term still anchors the overall scale, so sending all values to $-\infty$ does not minimize the complete squared-error objective.

Log-sum-exp is the optimum of $\mathbb E_\mu Q+\mathcal H(\mu)$, not of $\mathbb E_\mu Q$ alone after substitution. For continuous actions, an integral requires a reference measure and proposals with adequate support; Monte Carlo importance weights divide by proposal density, and taking the log introduces finite-sample bias. CQL lower-bound results require their stated sampling, regularization, and approximation assumptions; a trained neural critic is not automatically a certified pointwise lower bound.

## Part IV - From offline pretraining to online improvement

## 9. The offline-to-online transition

**Transcript coverage:** lines 5601-6415

### What the lecturer said - transcript only

Offline-to-online RL aims to pretrain from a fixed dataset and then improve efficiently with live interaction. The lecturer called this an open, cutting-edge problem without one guaranteed textbook solution.

Naively switching a CQL agent to online training can cause an immediate performance collapse. In the displayed experiment, offline performance reached roughly $0.5$ success, then dropped sharply when online interaction began before recovering. The pessimistic critic had assigned very low values to unseen actions. Once those actions were observed, its average Q-values rapidly recalibrated upward, disrupting the actor. In a safety-critical system such as a car, a first-online-step crash would be unacceptable. Other offline algorithms have analogous, though not identical, transition problems.

Offline and online phases demand conflicting biases. Offline training should avoid OOD actions, may prefer pessimism, and can take many gradient steps on fixed data. Online training should improve beyond $\pi_\beta$, may use optimism for exploration, and should learn quickly from new experience.

For many years, an embarrassingly simple baseline outperformed elaborate pretraining. Run ordinary online SAC from scratch with two buffers, drawing half of every training batch from the prior offline dataset and half from the online replay buffer. Do not perform a separate offline pretraining phase.

Offline samples help when they contain rare success, for example a successful trajectory in a delayed-reward task such as Montezuma's Revenge. They anchor value learning to the fact that success is possible. Behavioral cloning alone initializes only the actor; actor-critic training can quickly erase that initialization because the uninitialized critic controls the actor. Offline RL is therefore importantly a way to pretrain the critic. Poorly designed offline pretraining followed by the two-buffer method may cause no lasting harm but also no benefit relative to starting from scratch.

### Source reconciliation

Slides 22-26 show the performance dip and Q-value recalibration, compare offline and online requirements, and diagram the 50/50 two-buffer SAC baseline. The "half" ratio is the displayed baseline, not a universal optimum.

### Additional explanation

The transition is a critic-calibration problem as well as an exploration problem. A conservative critic intentionally encodes ignorance as low value. Online data changes the meaning of that ignorance, so actor and critic learning rates, data mixing, and regularizer annealing must be coordinated.

## 10. Why expressive diffusion and flow policies may help

**Transcript coverage:** lines 6416-6961

### What the lecturer said - transcript only

Recent offline-to-online methods using diffusion or flow-matching actors have empirically surpassed the simple two-buffer baseline. The lecturer said the reason is not yet fully understood and clearly labeled his explanation as speculation.

In fully observed online RL, an optimal deterministic policy exists under standard assumptions, so a simple Gaussian actor is often adequate. In offline RL, a policy constraint may need to represent the complex, multimodal behavior distribution. Forward KL smears mass when the actor cannot cover all modes, while reverse KL drops modes. A sufficiently expressive diffusion or flow policy can match all modes during the offline phase and later concentrate on the best one online.

Directly maximizing expected Q with such an actor is difficult. Diffusion and flow models are trained through per-step denoising or vector-field losses, not an easily evaluated action log likelihood. Policy gradient would need $\nabla\log\pi$, while reparameterization would require backpropagating through the entire iterative sampler. The latter is expensive and unstable. The lecturer suggested that avoiding this long backpropagation may itself be part of why the successful methods work well.

### Source reconciliation

Slides 27-28 contrast expressive behavior distributions with a simple actor and illustrate why a diffusion/flow sampling chain complicates the standard actor objective. The claim about why these models help is explicitly speculative in the transcript.

### Additional explanation

Expressiveness helps only if the model learns the behavior support accurately. A powerful generator can still create off-support samples or omit rare modes. The practical attraction is that supervised diffusion/flow training can fit behavior without exposing the generator itself to a noisy Q-gradient.

## 11. IDQL: behavior modeling plus best-of-$N$

**Transcript coverage:** lines 6962-7483

### What the lecturer said - transcript only

The first simple approach leaves the diffusion or flow policy entirely outside the RL objective. Train a Q-function with IQL or another method that avoids an actor-dependent OOD target. Separately train a highly expressive policy with behavioral cloning so that it approximates $\pi_\beta$.

At test time, sample several actions from that behavior model and select the one with the highest learned Q-value. Because a well-fit expressive generator should produce in-distribution actions, ranking those samples is safer than optimizing Q over the unrestricted action space.

The lecturer described this under names such as rejection sampling, best-of-$N$, guess and check, or sample and rank. It is very simple and surprisingly hard to beat. Its weakness is dimensional scaling: if the behavior distribution is broad and the action space high dimensional, random candidates may miss the best supported action. It is easy to implement by combining an IQL critic with the flow-matching policy machinery from Homework 1.

### Source reconciliation

Slide 29 labels this construction IDQL and diagrams a behavior-policy generator feeding multiple actions to a Q-based selector. The transcript does not expand the acronym; the note retains the slide label without inventing a longer name.

### Additional explanation

Best-of-$N$ improves monotonically with more samples under a fixed, correctly ranked candidate distribution, but its compute grows linearly with $N$. Selection also amplifies Q-estimation error among the samples, though restricting candidates to behavior-like actions reduces the severity of that optimizer's curse.

The monotonic statement is about expected selected score (or a nested set of candidates), not every independent run's actual return. If a desirable region has proposal probability $p$, the chance of at least one candidate there is $1-(1-p)^N$. Best-of-$N$ is sample-and-rank, not exact rejection sampling from a prescribed target density. The [IDQL paper](https://arxiv.org/abs/2304.10573) also discusses weighted policy extraction; the lecture's greedy selector is a particular extraction choice.

## 12. FQL: distilling a flow behavior model into an actor

**Transcript coverage:** lines 7484-7837

### What the lecturer said - transcript only

The second approach trains a flow model by behavioral cloning, but deploys a simpler one-step actor. The spoken explanation calls it Gaussian; the important correction is that Gaussian input noise does not require Gaussian output actions. Write the flow's output as a function of state and base noise $z$. Feed the same $z$ to the actor. For each $z$, regularize the actor toward the action the flow model would have generated while also maximizing Q.

The actor therefore distills a multimodal behavior model but shifts its outputs toward higher-value actions. Because the final actor is simple and reparameterizable, its expected Q can be optimized with the same machinery as SAC without backpropagating through the flow sampler for the Q term.

In Q&A, the lecturer agreed that a VAE could play a similar role. VAEs are more expressive than a Gaussian actor, but flow matching has performed better on the referenced benchmarks. He clarified that the Gaussian actor does not make the critic easier to train; it makes the critic easier to maximize through reparameterization. This method was suggested as one candidate for the offline-to-online final project.

### Source reconciliation

Slide 30 labels the construction FQL and shows an actor objective schematically combining

$$
\mathbb E_z[Q(s,\pi_\theta(s,z))]
$$

with a supervised term that matches the behavior-flow action generated from the same $(s,z)$. The slide establishes the shared-noise correspondence.

### Additional explanation

The shared noise turns distribution matching into paired distillation: each latent code identifies a particular behavior mode, so the actor is not asked to average unrelated modes. The Q term can move each paired output toward a locally better action.

**FQL architecture correction:** the distilled actor is an expressive one-step mapping $a=g_\theta(s,z)$ with Gaussian **input noise**. Its output distribution need not be Gaussian; a nonlinear mapping can remain multimodal. This distinction is explicit in the [FQL paper](https://arxiv.org/abs/2502.02538).

With a fixed flow teacher $g_{\rm flow}$, a representative actor loss is

$$
L_{\rm actor}(\theta)=
\mathbb E_{s,z}\!\left[-Q_\phi(s,g_\theta(s,z))
+\lambda\|g_\theta(s,z)-\operatorname{stopgrad}(g_{\rm flow}(s,z))\|^2\right].
$$

Freeze critic parameters but differentiate its action input; no Q gradient is needed through the teacher's iterative sampler. The shared-noise pairing is a coupling that encourages distributional proximity, not a guarantee that every noise coordinate has an identifiable semantic mode.

## 13. Diffusion steering and deferred material

**Transcript coverage:** lines 7838-8149

### What the lecturer said - transcript only

Diffusion steering changes the action space rather than directly making Q robust or constraining the actor in action space. Train a diffusion or flow model by behavioral cloning. Its base noise $w$ comes from a simple, known distribution; ordinary noise values map to behavior-like actions. Then run an efficient off-policy RL algorithm such as SAC with $w$ as its action. The fixed generative model maps the chosen latent into the environment action.

The actor "steers" behavior by selecting the noise that produces a desired action. If the latent prior is well behaved and the generator accurately maps typical latents into the behavior distribution, the latent action space contains far fewer OOD choices.

An audience member asked why this is specific to diffusion. The lecturer said it is not: any latent-variable model could be used. VAE versions were tried earlier but often worked worse, possibly because the actor discovers adversarial latent codes that decode poorly. He marked that explanation as a guess rather than an established result.

The lecturer then said model-based offline RL remained to be covered after spring break. The recording ends normally after that deferral.

### Source reconciliation

Slide 31 diagrams a learned actor producing $w$, which the fixed diffusion/flow decoder maps to an environment action. Slides 32-35 contain model-based offline-RL material that the lecturer explicitly deferred, so they are isolated below rather than attributed to this transcript segment.

### Additional explanation

The safety argument depends on keeping learned latents in the generator's trained prior region. An unconstrained SAC actor can still output extreme $w$ values, so a latent prior penalty or bounded parameterization may be necessary in practice.

## Slide-only deferred material

Slides 32-35 preview model-based offline RL but were not spoken in the supplied Lecture 18 recording:

- Slide 32 returns to the model-based offline setting.
- Slide 33 illustrates policy exploitation of learned-model errors.
- Slide 34 presents MOPO, which penalizes model-generated reward according to model uncertainty before using short synthetic rollouts.
- Slide 35 presents COMBO, which applies a conservative-value principle to model-generated state-action samples.

These ideas are actually explained at the start of the supplied Lecture 19 transcript and are documented there from speech. They are listed here only to account for every slide page.

## Consolidated takeaways

1. Explicit policy constraints can be placed in the actor, reward, or critic target; forward-KL actor regularization reduces to a behavioral-cloning term.
2. AWR/AWAC derives an exponentiated-advantage policy within a reverse-KL trust region and implements it as weighted behavioral cloning.
3. IQL fits an upper expectile over dataset-action Q-values, creating Bellman targets without actor-generated OOD actions.
4. CQL adversarially lowers high Q-values away from data while raising values on dataset actions; entropy regularization yields a log-sum-exp form.
5. Directly switching a pessimistic offline agent to online learning can cause a severe calibration and performance collapse.
6. A simple 50/50 offline/online replay mixture is a strong baseline, but it does not exploit large datasets through pretraining.
7. Expressive diffusion and flow models may preserve the behavior distribution's modes through offline training, though the causal explanation is not settled.
8. IDQL ranks several behavior-model samples, FQL distills a flow model into a Q-improving actor, and diffusion steering runs RL in the generator's latent action space.
9. Model-based offline slides were deferred in speech and must not be presented as part of the transcript-only lecture account.

## Key equations

### Actor-critic plus behavioral cloning

$$
\max_\theta\;
\mathbb E_{s\sim\mathcal D,\,a\sim\pi_\theta}[Q_\phi(s,a)]
+\lambda\mathbb E_{(s,a)\sim\mathcal D}[\log\pi_\theta(a\mid s)].
$$

### Advantage-weighted target policy

$$
\pi^*(a\mid s)=\frac{1}{Z(s)}
\pi_\beta(a\mid s)\exp\!\left(\frac{A(s,a)}{\lambda}\right).
$$

### Expectile loss

$$
L_2^\tau(u)=|\tau-\mathbf 1\{u<0\}|u^2.
$$

### IQL critic losses

$$
\mathcal L_V=
\mathbb E_{\mathcal D}[L_2^\tau(Q_{\bar\phi}(s,a)-V(s))],
$$

$$
\mathcal L_Q=
\mathbb E_{\mathcal D}[(\operatorname{stopgrad}(r+\gamma mV(s'))-Q_\phi(s,a))^2].
$$

### Conservative Q-learning regularizer

$$
\alpha\left(
\log\sum_a\exp Q(s,a)
-\mathbb E_{a\sim\pi_\beta(\cdot\mid s)}[Q(s,a)]
\right).
$$

### Diffusion steering

$$
w\sim\pi_\theta(w\mid s),
\qquad
a=g_{\mathrm{flow}}(s,w),
$$

with RL performed over $w$ and the behavior generator $g_{\mathrm{flow}}$ mapping latents to actions.

## Glossary

- **SAC+BC / actor-critic plus BC:** Actor-critic with a behavior-cloning log-likelihood term in the actor objective.
- **Behavior regularization:** Penalizing deviation of a learned policy from the offline behavior distribution.
- **AWR:** Advantage-weighted regression, fitting a policy to behavior samples weighted by exponentiated advantage.
- **AWAC:** An actor-critic algorithm whose actor update uses advantage-weighted behavioral cloning.
- **Expectile:** The minimizer of an asymmetric squared-error objective.
- **IQL:** Implicit Q-learning, which uses an upper expectile value to avoid OOD actions in critic targets.
- **Pessimism:** Lowering estimates for actions not sufficiently supported by data.
- **CQL:** Conservative Q-learning, which lowers high non-data Q-values and raises data-action Q-values.
- **Log-sum-exp:** A smooth maximum arising from an entropy-regularized adversarial action distribution.
- **Offline-to-online RL:** Pretraining on a fixed dataset followed by learning from new environment interaction.
- **Best-of-$N$:** Sampling $N$ candidates from a proposal and selecting the highest-scoring one.
- **IDQL:** The slide label for combining an in-distribution critic with behavior-model sampling and ranking.
- **FQL:** The slide label for a flow-based behavior model distilled into a Q-improving actor.
- **Diffusion steering:** Applying RL in the latent-noise action space of a fixed behavior generator.
- **MOPO:** A model-based offline method that penalizes predicted reward by model uncertainty.
- **COMBO:** A model-based offline method that applies conservative value learning to model-generated samples.

## Self-check questions

1. Why does forward-KL actor regularization become a behavioral-cloning term?
2. How does the nonparametric AWR policy combine advantage with behavior support?
3. What happens to effective sample size as the AWR temperature approaches zero?
4. Why does IQL's Bellman target avoid querying actor-generated actions?
5. How is an upper expectile different from a maximum or a quantile?
6. Why does the first pessimistic Q objective underestimate supported good actions?
7. What role does the data-action subtraction play in CQL?
8. Why does entropy regularization turn the CQL adversary into a log-sum-exp term?
9. Why is offline-to-online RL not solved by pretraining only the actor with BC?
10. What competing requirements do the offline and online phases impose?
11. How do IDQL, FQL, and diffusion steering use an expressive behavior model differently?
12. Which four slides contain content explicitly deferred by the lecturer?

## Source coverage checklist

- [x] Transcript lines 1-8149 are assigned once, in monotonic and inclusive ranges.
- [x] The recording ends normally; model-based offline RL is explicitly deferred rather than truncated.
- [x] All 35 slide pages were rendered and visually inspected.
- [x] KL, actor+BC, AWR/AWAC, expectile/IQL, CQL, flow-distillation, and latent-steering equations were checked against the slides.
- [x] Mode, IQL, CQL, offline-to-online, IDQL, FQL, diffusion-steering, MOPO, and COMBO diagrams were visually inspected.
- [x] Slides 32-35 are clearly isolated as deferred slide-only content.
- [x] Transcript errors such as "Kale," "log sumx," "mew," and inconsistent VAE/flow terms are reconciled only where the slides establish the intended notation.
- [x] Lecturer claims, explicitly speculative explanations, slide-only material, and additional explanation are kept separate.

**Coverage result:** Complete for the supplied 8,149-line transcript and 35-page slide deck; four model-based offline slides are accounted for as explicitly deferred material.
