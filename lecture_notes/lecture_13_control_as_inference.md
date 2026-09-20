---
title: "Lecture 13 - Control as Variational Inference"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 13
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 13, Control as Variational Inference.txt"
source_slides: "../lectures/Lecture 13 - Control as Inference.pdf"
transcript_lines: 4474
slide_pages: 34
status: "source-incomplete"
---

# Lecture 13: Control as Variational Inference

> **Source warning:** The supplied transcript ends at line 4,474 during an audience question immediately after the lecturer says that more practical inverse-RL methods will be continued on Friday. Slides 25-34 have no corresponding recorded lecture speech and are therefore isolated in a slide-only appendix.

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Recap: control as inference | lines 1-315 |
| 2 | Why exact inference is optimistically wrong | lines 316-957 |
| 3 | A dynamics-preserving variational posterior | lines 958-1439 |
| 4 | ELBO cancellation yields reward plus entropy | lines 1440-1848 |
| 5 | Soft dynamic programming without transition optimism | lines 1849-2440 |
| 6 | Why maximum-entropy control can help | lines 2441-2592 |
| 7 | Soft Q-learning and entropy-regularized policy gradient | lines 2593-2883 |
| 8 | Soft actor-critic | lines 2884-3246 |
| 9 | Inverse RL and imitation of intent | lines 3247-3554 |
| 10 | Reward ambiguity and maximum-likelihood IRL | lines 3555-3924 |
| 11 | The IRL partition-function gradient | lines 3925-4293 |
| 12 | A correct but expensive nested algorithm | lines 4294-4474 |

## Part I - Repairing control as inference with variational inference

## 1. Recap: control as inference

**Transcript coverage:** lines 1-315

### What the lecturer said - transcript only

The lecture would combine variational inference with control as inference, then begin inverse reinforcement learning, where rewards rather than policies are learned.

The previous lecture augmented an MDP with binary optimality variables $\mathcal O_t$ whose likelihood is the exponential of reward. Conditioning on all optimality variables being true produces

$$
p(\tau\mid\mathcal O_{1:T}=1)
\propto p(\tau)\exp\left(\sum_t r(s_t,a_t)\right).
$$

High-reward trajectories are most likely, while small mistakes are exponentially but not absolutely suppressed. Backward inference produced Q- and V-like log messages, and the policy was the ratio of state-action and state backward messages:

$$
\pi(a\mid s)=\exp(Q(s,a)-V(s))=\exp(A(s,a)).
$$

The procedure resembles inference in a hidden Markov model or extended Kalman filter and mathematically resembles value iteration with soft rather than hard maximization.

### Source reconciliation

Slides 2-5 recap the graphical model, log-message backups, exponential-advantage policy, and summary.

### Additional explanation

The resemblance is exact for the action aggregation under deterministic dynamics. Stochastic dynamics expose the defect addressed next.

## 2. Why exact inference is optimistically wrong

**Transcript coverage:** lines 316-957

### What the lecturer said - transcript only

The problematic exact-inference backup is

$$
Q(s,a)=r(s,a)+
\log\mathbb E_{s'\sim p(\cdot\mid s,a)}[\exp(V(s'))].
$$

Log-expectation-exp is dominated by unusually large outcomes. The lecturer used a lottery ticket: a tiny chance of winning a billion dollars has low expected value, but exponentiating the billion-dollar outcome overwhelms the expectation, after which the logarithm returns a value near the jackpot. This produces an irrationally optimistic decision maker. Optimism may encourage exploration, but it is not a good model of rational planning.

No algebra error caused the problem; the inference question is wrong for control. Probabilistic conditioning asks, “Given that high reward occurred, what action and transition probably occurred?” In hindsight, seeing the lecturer return with a Ferrari and gold chain would make a winning ticket likely. It would update the posterior over the transition outcome even though the agent could not control whether the ticket won.

Planning asks a different, forward-looking question: “If high reward is desired, what should the agent do?” The action distribution may change, but the environment transition probabilities must not. Optimality evidence can be explained either by a good decision or by luck, and exact posterior inference changes both. That is the source of optimistic transition beliefs.

Questions clarified that “the monkey is optimal” technically means it obtained high reward in the model; the reward could result from good actions or luck. The variational repair must preserve the former inference while preventing the latter.

### Source reconciliation

Slides 6-7 identify the log-expectation-exp transition and use the stochastic-outcome problem to motivate variational control.

### Additional explanation

This distinction is the causal core: control may intervene on the action distribution, but it cannot condition nature's transition mechanism into becoming luckier.

## 3. A dynamics-preserving variational posterior

**Transcript coverage:** lines 958-1439

### What the lecturer said - transcript only

The desired approximation is close to the optimality-conditioned posterior over states and actions but is constrained to retain the real initial-state and transition distributions. Variational inference permits choosing any tractable or structurally constrained family and then finding its closest member to the true posterior.

Here the observed variables $x$ in the generic variational notation are the optimality variables, and the latent variables $z$ are all states and actions. The approximate trajectory distribution is chosen as

$$
q(\tau)
=p(s_1)
\left[\prod_{t=1}^{T}q(a_t\mid s_t)\right]
\left[\prod_{t=1}^{T-1}p(s_{t+1}\mid s_t,a_t)\right].
$$

The only free factors are $q(a_t\mid s_t)$, which form a policy. The true initial-state distribution and dynamics are copied into $q$, so conditioning on reward cannot alter them.

The lecturer called this unusual because variational families are ordinarily chosen for computational simplicity, whereas the true dynamics may be unknown or complicated. It is nevertheless a valid restriction. A question asked what exactly was unusual; he answered that $q$ deliberately contains $p$ factors from the original model rather than an independently simple family.

### Source reconciliation

Slides 8-9 contrast the original optimality model with the variational trajectory distribution containing true dynamics and a free policy.

### Additional explanation

“Unknown dynamics” does not prevent sampling $q(\tau)$: execute $q(a\mid s)$ in the real environment. The method need not evaluate the transition density because the same factor later cancels from the ELBO.

## 4. ELBO cancellation yields reward plus entropy

**Transcript coverage:** lines 1440-1848

### What the lecturer said - transcript only

Substituting the chosen $q(\tau)$ into the ordinary variational lower bound produces log terms for the original model and negative log terms for $q$. The initial-state factors cancel. The transition factors also cancel because they are identical. A student correctly noted that the transition sum should stop at $T-1$ rather than run beyond the trajectory; the lecturer acknowledged the slide correction.

The surviving terms are optimality log likelihoods and negative policy log probabilities. Since $\log p(\mathcal O_t=1\mid s_t,a_t)=r(s_t,a_t)$, the ELBO becomes

$$
\mathcal L(q)
=
\mathbb E_{\tau\sim q}
\left[
\sum_t r(s_t,a_t)-\log q(a_t\mid s_t)
\right]
$$

or equivalently

$$
\mathcal L(q)
=\sum_t
\mathbb E_{(s_t,a_t)\sim q}
\left[r(s_t,a_t)+\mathcal H(q(\cdot\mid s_t))\right].
$$

Only the action distribution is optimized. Without the entropy term this is the usual expected-return objective; with it, it is policy gradient with entropy. Questions distinguished the per-state policy $q(a_t\mid s_t)$ from the full trajectory distribution it induces together with fixed initial state and dynamics.

### Source reconciliation

Slide 10 shows the cancellation and explicitly notes the $T-1$ endpoint correction; slide 11 writes the resulting soft Bellman form.

### Additional explanation

The cancellation is why the method both enforces correct dynamics and remains model-free. The fixed environment terms are present in the sampling distribution but absent from the objective's likelihood ratio.

## 5. Soft dynamic programming without transition optimism

**Transcript coverage:** lines 1849-2440

### What the lecturer said - transcript only

At this point one could optimize the reward-plus-entropy objective directly with policy gradient. For connection to inference, the lecturer also derived dynamic programming.

The basic identity is that maximizing an expected function plus the distribution's entropy gives a Gibbs distribution:

$$
\arg\max_q
\left(\mathbb E_{a\sim q}[f(a)]+\mathcal H(q)\right)
\propto\exp(f(a)).
$$

At the final step $Q_T(s,a)=r(s,a)$, so the optimal policy is proportional to $e^{Q_T}$. Recursing backward gives

$$
Q_t(s_t,a_t)
=r(s_t,a_t)
+\mathbb E_{s_{t+1}\sim p(\cdot\mid s_t,a_t)}[V_{t+1}(s_{t+1})],
$$

$$
V_t(s_t)=\log\int\exp(Q_t(s_t,a))\,da,
\qquad
q^*(a_t\mid s_t)=\exp(Q_t(s_t,a_t)-V_t(s_t)).
$$

The crucial repair is that $Q$ now uses the ordinary expectation of next value. Only action selection is soft. There is no log-expectation-exp over stochastic transitions, so lottery optimism disappears. The result is soft value iteration: conventional Bellman expectation plus soft rather than hard action maximization.

The lecturer summarized the variational posterior as using the true initial state and transitions with a learned action distribution. He mentioned two extensions: discounting can be interpreted via a $1-\gamma$ chance of transition to an absorbing death state, and a temperature controls softness. As temperature tends to zero, regular hard value iteration is recovered as a limiting case.

### Source reconciliation

Slides 11-12 summarize the corrected backup and variants. The slides write temperature scaling compactly; the normalized convention is
$V_\alpha(s)=\alpha\log\int e^{Q(s,a)/\alpha}da$.

### Additional explanation

The Gibbs identity follows from rewriting the objective as a constant minus a KL divergence between $q(a)$ and the normalized $e^{f(a)}$ distribution.

## 6. Why maximum-entropy control can help

**Transcript coverage:** lines 2441-2592

### What the lecturer said - transcript only

In response to “why do we need this?”, the lecturer gave three answers. The deeper answer is a relationship to robust control: a high-entropy policy performs the task in many ways, so if one route or behavior fails under a misspecified MDP, alternatives remain. Exiting through both the left and right doors makes the agent less brittle if one is blocked.

A simpler but potentially misleading answer is improved exploration. Randomness can help exploration, but more targeted exploration methods exist. The practitioner answer is that soft actor-critic, used in the homework, often works well.

### Source reconciliation

Slide 12 contains the theoretical summary; the robustness and exploration discussion is primarily spoken.

### Additional explanation

Maximum entropy is not the same as maximizing randomness regardless of reward. The optimization seeks the most random policy among policies that still obtain high return, with the reward/entropy tradeoff set by temperature.

## Part II - Maximum-entropy RL algorithms

## 7. Soft Q-learning and entropy-regularized policy gradient

**Transcript coverage:** lines 2593-2883

### What the lecturer said - transcript only

Soft Q-learning makes a one-line conceptual change to Q-learning: replace the target's hard action maximum with log-sum-exp for discrete actions or log-integral-exp for continuous actions. It is easy for small discrete action spaces but not especially popular because a hard maximum already works well there and the continuous integral is difficult.

For policy gradient, add policy entropy to the expected-return objective. Entropy has closed forms for common distributions such as Gaussians and categorical softmax policies. This prevents premature policy collapse, in which the policy becomes nearly deterministic before finding good rewards. A multiplier controls the entropy weight and can be reduced during training if a deterministic final policy is desired.

Asked when not to add entropy, the lecturer said it saves one term and one hyperparameter if ordinary training already works. It is particularly useful when early collapse is observed.

### Source reconciliation

Slides 13-15 introduce maximum-entropy algorithms, soft Q-learning, and entropy-regularized policy gradient.

### Additional explanation

Soft Q-learning requires a tractable partition integral. Soft actor-critic avoids evaluating that integral directly by representing a policy that approximately samples the high-Q distribution.

## 8. Soft actor-critic

**Transcript coverage:** lines 2884-3246

### What the lecturer said - transcript only

Soft actor-critic (SAC) is the maximum-entropy version of off-policy actor-critic. As in ordinary off-policy actor-critic, transitions enter a replay buffer, a critic is trained from bootstrapped targets, and an actor is optimized against the critic.

The changes are:

- the actor objective includes current policy entropy; and
- the critic target includes entropy at future time steps, commonly by subtracting the next action's log probability.

A representative target and actor objective are

$$
y=r+\gamma
\left(Q_{\bar\phi}(s',a')-\alpha\log\pi_\theta(a'\mid s')\right),
\qquad a'\sim\pi_\theta(\cdot\mid s'),
$$

$$
J_\pi(\theta)
=\mathbb E_{s\sim\mathcal D,\,a\sim\pi_\theta}
\left[Q_\phi(s,a)-\alpha\log\pi_\theta(a\mid s)\right].
$$

The current entropy must appear in the actor objective so its gradient flows into the policy. If it were hidden only inside a stopped-gradient Q target, the actor would not receive that derivative. The future entropy belongs in the critic because $Q$ represents future reward plus future entropy.

The temperature $\alpha$ trades reward against randomness. It is redundant in principle with rescaling all rewards but convenient in practice. It can also be adapted dynamically to reach a desired entropy level. The lecturer identified SAC as one of the most widely used off-policy actor-critic methods and connected it directly to the inference derivation.

### Source reconciliation

Slide 16 gives the SAC update sequence. The transcript first recaps ordinary actor-critic on the slide and then identifies the entropy additions; this ordering is retained above.

### Additional explanation

Modern SAC commonly uses two critics and the smaller target estimate, inheriting clipped double Q-learning from Lecture 8. That detail is part of common implementations but was not the focus of this transcript segment.

## Part III - Beginning inverse reinforcement learning

## 9. Inverse RL and imitation of intent

**Transcript coverage:** lines 3247-3554

### What the lecturer said - transcript only

Forward RL assumes a reward and seeks a policy. Inverse reinforcement learning (IRL) observes behavior and seeks a reward that explains it. The same optimality graphical model can be inverted: rather than infer actions from reward, infer reward from demonstrated actions and states.

Learning reward can imitate **intent** rather than literal motion. Behavior cloning copies actions without reasoning about their consequences. People often infer another person's goal and choose different actions that achieve it. The lecturer showed a child-psychology video in which a child reproduced the adult's apparent goal rather than mechanically copying the exact awkward action.

On an Atari benchmark, the game score supplies a reward. In real driving or robotics, the intended reward may be difficult to specify. Demonstrations can reveal human intent, after which a new policy can optimize that inferred objective while accounting for outcomes.

### Source reconciliation

Slides 17-20 introduce IRL, reuse the inference graph, contrast imitation of actions with imitation of intent, and contrast explicit benchmark scores with real-world reward ambiguity.

### Additional explanation

Inferring intent can improve robustness to embodiment differences: a robot need not have the same body or exact action coordinates as a human if both optimize a shared state-based objective.

## 10. Reward ambiguity and maximum-likelihood IRL

**Transcript coverage:** lines 3555-3924

### What the lecturer said - transcript only

IRL is underdetermined: many rewards can make the same behavior optimal. In a $4\times4$ grid with 16 states, a demonstrated path could be explained by a desirable destination, a later desirable region, avoidance of bad states, or a contrived action-specific reward. One trivial explanation assigns $-\infty$ to every unchosen action and a million to the chosen one, but that is not a satisfying account of intent.

Soft optimality helps use relative trajectory frequencies to disambiguate explanations. Represent reward by $r_\psi(s,a)$, perhaps a neural network or a function of known features, and retain

$$
p(\mathcal O_t=1\mid s_t,a_t,\psi)=\exp(r_\psi(s_t,a_t)).
$$

The induced trajectory density is

$$
p_\psi(\tau\mid\mathcal O_{1:T}=1)
=\frac{1}{Z(\psi)}p(\tau)e^{r_\psi(\tau)},
\qquad
r_\psi(\tau)=\sum_t r_\psi(s_t,a_t).
$$

Given trajectories sampled from an expert's soft-optimal policy, maximum likelihood maximizes their log probability. The lecturer described the state-action trajectories as the latent/integrated variables in the underlying optimality model, while the demonstrations provide observed samples of them for the likelihood objective. The physics term $p(\tau)$ is independent of $\psi$ and can be ignored during reward optimization. The hard term is the normalizer

$$
Z(\psi)=\int p(\tau)e^{r_\psi(\tau)}\,d\tau.
$$

An audience member asked why $Z$ appeared. The lecturer explained that “proportional to” requires division by a normalizer that does not depend on $\tau$ but unfortunately does depend on $\psi$, so it cannot be dropped during learning.

### Source reconciliation

Slides 21-22 show reward ambiguity and the maximum-likelihood optimality model. Slide 22 contains staged/overlaid notation from animation, but the final spoken equation is $p(\mathcal O_t\mid s_t,a_t,\psi)=\exp(r_\psi(s_t,a_t))$.

### Additional explanation

Soft optimality does not make the reward fully identifiable. Potential-based shaping and constant shifts can preserve behavior. It supplies a likelihood principle and uses stochastic frequencies, but inductive bias or regularization is still needed.

## 11. The IRL partition-function gradient

**Transcript coverage:** lines 3925-4293

### What the lecturer said - transcript only

For expert trajectories $\{\tau_i\}_{i=1}^N$, the objective is

$$
\mathcal L(\psi)
=\frac1N\sum_{i=1}^N r_\psi(\tau_i)-\log Z(\psi)+\text{const}.
$$

Differentiating $\log Z$ yields

$$
\nabla_\psi\mathcal L
=
\mathbb E_{\tau\sim\pi^*}
[\nabla_\psi r_\psi(\tau)]
-
\mathbb E_{\tau\sim p_\psi(\tau\mid\mathcal O=1)}
[\nabla_\psi r_\psi(\tau)].
$$

The first expectation is estimated with demonstrations. The factor
$Z^{-1}p(\tau)e^{r_\psi(\tau)}$ in the derivative of the partition function is exactly the soft-optimal trajectory distribution under the current reward, which supplies the second expectation.

The gradient increases reward on expert behavior and decreases reward on behavior preferred by the current learned reward. At a solution where the current reward induces the expert's trajectory distribution, the two feature/reward-gradient expectations match and the gradient is zero.

### Source reconciliation

Slide 23 derives the partition-function gradient and slide 24 labels its expert and current-policy expectations.

### Additional explanation

For a linear reward $r_\psi(s,a)=\psi^\top f(s,a)$, the gradient is exactly expert feature counts minus current-policy feature counts. The neural-reward form generalizes “features” to $\nabla_\psi r_\psi$.

## 12. A correct but expensive nested algorithm

**Transcript coverage:** lines 4294-4474

### What the lecturer said - transcript only

A direct estimator first solves a maximum-entropy RL problem for the current reward—using SAC or entropy-regularized policy gradient—then samples that policy to estimate the negative expectation in the reward gradient. Expert samples estimate the positive expectation.

This is correct but extremely expensive: after every reward update one would ideally solve a complete RL problem again. Intuitively, it repeatedly raises reward on demonstrations, lowers reward on the current policy's behavior, and changes the policy until it aligns with the expert.

At 10:00, the lecturer stopped and said Friday's lecture would make the method practical without this full inner loop. The transcript then records the beginning of a student's question—“So, how do you get from here to here? Like, are you saying that because of the fact that our…”—and ends after a pair of acknowledgments. The remainder of the question and answer is unavailable.

### Source reconciliation

Slide 24 contains the sample estimator and nested maximum-entropy RL idea. The spoken lecture does not reach the subsequent slide material.

### Additional explanation

The computational bottleneck is a moving negative-sample distribution: every reward change changes which trajectories should be sampled. The slide-only methods below reuse and adapt policy samples more efficiently.

## Slide-only appendix: material after the transcript truncation

The following material comes from slides 25-34 only. It is **not** presented as a paraphrase of what the lecturer said because no corresponding speech is present in the supplied transcript.

### A. Lazy policy updates and importance sampling

Slide 25 proposes improving the current policy only a little rather than fully solving maximum-entropy RL after every reward update. Reusing its samples gives a biased negative expectation unless those samples are corrected. It proposes self-normalized importance weights

$$
w_j=\frac{p(\tau_j)e^{r_\psi(\tau_j)}}{\pi(\tau_j)},
$$

$$
\nabla_\psi\mathcal L
\approx
\frac1N\sum_{i=1}^N\nabla_\psi r_\psi(\tau_i)
-\frac{1}{\sum_jw_j}
\sum_{j=1}^M w_j\nabla_\psi r_\psi(\tau_j).
$$

Slide 26 expands the ratio. The initial-state and dynamics factors cancel against the same factors in the policy's trajectory density, leaving

$$
w_j
=\frac{\exp(\sum_t r_\psi(s_t,a_t))}
{\prod_t\pi(a_t\mid s_t)}.
$$

It states that each policy update under the current reward brings the proposal closer to the target soft-optimal distribution.

### B. Guided cost learning

Slide 27 labels the alternating procedure **guided cost learning** (Finn et al., ICML 2016). Starting from a policy and human demonstrations, it generates policy samples, updates a learned reward with importance-weighted policy samples and demonstrations, then updates the policy under that reward. The slide depicts a robotic manipulation application.

### C. Adversarial imitation as a game

Slides 28-29 reframe the alternating process as adversarial imitation learning. The reward is trained to make demonstrations more likely and policy samples less likely; the policy is changed so its samples become harder to distinguish from demonstrations.

### D. Connection to GANs

Slide 30 reviews generative adversarial networks: a generator produces samples and a discriminator is trained to distinguish data from generated samples. The generator is then optimized to fool the discriminator.

Slide 31 gives the optimal discriminator for expert density $p^*$ and policy density $p_\theta$,

$$
D^*(x)=\frac{p^*(x)}{p_\theta(x)+p^*(x)},
$$

and chooses a structured trajectory discriminator whose numerator contains the exponentiated reward and partition term, while its denominator also contains the policy likelihood. In that parameterization, the partition function absorbs the earlier importance-weighting role. Slide 32 summarizes the resulting reward/discriminator and policy updates.

### E. Regular discriminators and GAIL

Slide 33 asks whether one can use an ordinary binary neural-network discriminator and then use $\log D_\psi(\tau)$ as policy reward. It identifies this as the simpler adversarial-imitation approach associated with generative adversarial imitation learning (GAIL). Its advantage is fewer moving parts. Its stated disadvantages are that the discriminator contains no useful information at convergence and generally cannot be reused as a reward for later re-optimization.

Slide 34 compares a maximum-entropy IRL reward model with an ordinary classifier and says the two adversarial processes are closely related. It shows humanoid multi-skill imitation and motion-imitation examples.

## Consolidated takeaways

1. Exact optimality-conditioned inference is optimistic under stochastic dynamics because it conditions transition outcomes on observed success.
2. Control needs a posterior family that changes the policy but preserves the true initial state and dynamics.
3. Copying those dynamics into the variational distribution causes them to cancel from the ELBO.
4. The resulting objective is expected reward plus policy entropy.
5. Soft value iteration uses a standard expected next value and soft action maximization.
6. Maximum-entropy policies can improve robustness, maintain exploration, and work well in practice.
7. Soft Q-learning, entropy-regularized policy gradient, and SAC are direct algorithmic realizations.
8. SAC includes current entropy in the actor and future entropy in critic targets.
9. IRL learns a reward that makes expert trajectories likely under a soft-optimal model.
10. The IRL likelihood gradient is expert reward-gradient statistics minus current-policy statistics.
11. Solving a full RL problem after every reward step is correct but impractical.
12. The unspoken slide continuation introduces importance sampling, guided cost learning, and adversarial imitation/GAIL as more efficient alternatives.

## Key equations

### Variational control objective

$$
\max_q\;
\mathbb E_{\tau\sim q}
\left[\sum_t r(s_t,a_t)+\mathcal H(q(\cdot\mid s_t))\right].
$$

### Soft Bellman equations

$$
Q(s,a)=r(s,a)+\gamma\mathbb E_{s'\mid s,a}[V(s')],
$$

$$
V(s)=\alpha\log\int\exp(Q(s,a)/\alpha)\,da,
\qquad
\pi(a\mid s)=\exp((Q(s,a)-V(s))/\alpha).
$$

### SAC target and actor objective

$$
y=r+\gamma\left(Q_{\bar\phi}(s',a')-\alpha\log\pi_\theta(a'\mid s')\right),
$$

$$
J_\pi(\theta)=
\mathbb E[Q_\phi(s,a)-\alpha\log\pi_\theta(a\mid s)].
$$

### Maximum-entropy IRL density

$$
p_\psi(\tau)
=\frac{1}{Z(\psi)}p(\tau)e^{r_\psi(\tau)},
\qquad
Z(\psi)=\int p(\tau)e^{r_\psi(\tau)}d\tau.
$$

### IRL likelihood gradient

$$
\nabla_\psi\mathcal L
=\mathbb E_{\tau\sim\pi^*}[\nabla_\psi r_\psi(\tau)]
-\mathbb E_{\tau\sim p_\psi}[\nabla_\psi r_\psi(\tau)].
$$

## Glossary

- **Adversarial imitation learning:** alternating optimization in which a reward/discriminator separates demonstrations from policy behavior and the policy tries to eliminate that distinction.
- **Control as variational inference:** inference with a variational trajectory family constrained to preserve the real dynamics.
- **GAIL:** generative adversarial imitation learning, which uses an ordinary discriminator-derived reward to match expert occupancy.
- **Guided cost learning:** sample-based maximum-entropy IRL that alternates reward fitting, policy improvement, and importance correction.
- **Inverse reinforcement learning:** inference of a reward function from demonstrated behavior.
- **Maximum-entropy RL:** optimization of expected reward plus policy entropy.
- **Partition function:** the normalizer $Z$ integrating exponentiated trajectory reward over physically possible trajectories.
- **Reward ambiguity:** the fact that many reward functions can explain the same behavior.
- **Soft actor-critic:** an off-policy actor-critic algorithm optimizing reward plus entropy.
- **Soft Bellman backup:** a Bellman backup with expected next value and soft action aggregation.
- **Transition optimism:** preference for rare favorable next states created by log-expectation-exp over stochastic transitions.
- **Variational trajectory distribution:** the dynamics-preserving trajectory family whose free factor is the policy.

## Self-check questions

1. Why does exact inference favor a lottery ticket more than expected-return planning does?
2. What is the difference between conditioning on high reward in hindsight and planning for high reward?
3. Which factors are fixed and which are optimized in $q(\tau)$?
4. Why do the initial-state and dynamics terms cancel from the ELBO?
5. What objective remains after cancellation?
6. How does the soft Q backup differ from the optimistic exact-inference backup?
7. Give the robustness and exploration interpretations of maximum entropy.
8. What one-line change defines soft Q-learning?
9. Where do current and future entropy appear in SAC, and why?
10. How can SAC's temperature be adapted?
11. Why might inferred intent be more useful than copied expert actions?
12. Give three different rewards that could explain one grid-world path.
13. Why can the IRL partition function not be ignored?
14. Why does the IRL gradient vanish when the learned policy distribution matches the expert?
15. Why is the direct nested IRL algorithm too expensive?
16. Which parts of the importance-sampling and adversarial continuation are slide-only?

## Source coverage checklist

| Topic | Inclusive range | Coverage status |
|---:|---:|---|
| 1 | 1-315 | Accounted for |
| 2 | 316-957 | Accounted for |
| 3 | 958-1439 | Accounted for |
| 4 | 1440-1848 | Accounted for; spoken endpoint correction retained |
| 5 | 1849-2440 | Accounted for |
| 6 | 2441-2592 | Accounted for |
| 7 | 2593-2883 | Accounted for |
| 8 | 2884-3246 | Accounted for |
| 9 | 3247-3554 | Accounted for |
| 10 | 3555-3924 | Accounted for |
| 11 | 3925-4293 | Accounted for |
| 12 | 4294-4474 | Accounted for; source ends mid-question |

**Coverage result:** All 4,474 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. All 34 slide pages were visually inspected. Slides 25-34 lack transcript coverage and are isolated in the slide-only appendix rather than attributed to the lecturer's recorded speech.
