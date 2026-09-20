---
title: "Lecture 9 - Advanced Policy Gradients, Part 1"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 9
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 9, Off-Policy Policy Gradient.txt"
source_slides: "../lectures/Lecture 09 - Advanced Policy Gradients Part 1.pdf"
transcript_lines: 4504
slide_pages: 16
status: "complete"
---

# Lecture 9: Advanced Policy Gradients, Part 1

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Why reuse data in policy gradients? | lines 1-585 |
| 2 | Why repeated on-policy updates are invalid | lines 586-1113 |
| 3 | Importance sampling for trajectories | lines 1114-1566 |
| 4 | Re-deriving the off-policy policy gradient | lines 1567-2244 |
| 5 | State-distribution approximation and a multi-step algorithm | lines 2245-2988 |
| 6 | Finite-sample variance and policy drift | lines 2989-3285 |
| 7 | Clipped importance ratios | lines 3286-3840 |
| 8 | The asymmetric PPO clipped objective | lines 3841-4275 |
| 9 | PPO in practice and closing Q&A | lines 4276-4504 |

## Part I - From on-policy updates to data reuse

## 1. Why reuse data in policy gradients?

**Transcript coverage:** lines 1-585

### What the lecturer said - transcript only

After brief setup and a joke about the room lighting, the lecturer introduced advanced policy-gradient methods as attempts to retain the attractive properties of basic on-policy policy gradients while reducing their wastefulness.

On-policy policy gradients are often stable and conceptually simple. A Monte Carlo return, corresponding to $\lambda=1$ in generalized advantage estimation, remains unbiased even if the critic is inaccurate. This makes the method comparatively forgiving.

The drawback is sample cost. Each iteration collects trajectories, takes essentially one policy step using them, then discards them because the policy has changed. Whether that is unacceptable depends on where samples come from. Samples can be cheap for a language model generating its own tokens or for a simulated game. They can be very expensive for real driving, real robot interaction, or conversations that require human participation. Sim-to-real transfer can reduce physical data needs, but it does not make data reuse irrelevant.

The goal is therefore to perform multiple useful policy updates per collected batch without losing the reliability of the on-policy method.

### Source reconciliation

Slides 2-4 summarize the basic policy-gradient loop, the appeal of GAE/Monte Carlo targets, and the one-step-per-batch problem.

### Additional explanation

“Sample efficiency” here means improvement per environment interaction, not per unit of computation. A method can reuse data and become more sample-efficient while requiring more optimization work on each batch.

## 2. Why repeated on-policy updates are invalid

**Transcript coverage:** lines 586-1113

### What the lecturer said - transcript only

The tempting procedure is to collect one batch and take several ordinary policy-gradient steps on it. This is not valid because the gradient identity assumes that trajectories are distributed according to the same policy whose objective is being differentiated. After the first update, the stored trajectories came from an older policy.

The lecturer illustrated the failure using a Gaussian policy. Suppose the sampled actions happen to lie on one side of its mean and receive a positive advantage. One update shifts the Gaussian toward those samples. Repeatedly reusing the same points as if they were still fresh on-policy samples keeps pushing the mean toward them, even after they are no longer representative of actions that the new policy would generate. The optimizer can overfit a finite batch and move far away from the data-generating policy.

Old data are not useless, but the distribution mismatch must be accounted for. The desired multi-step method therefore needs an off-policy correction.

### Source reconciliation

Slides 5-6 contrast the desired repeated-update loop with the fact that the plain gradient estimator is no longer valid after the first change in policy.

### Additional explanation

This is a statistical issue rather than merely a large-learning-rate issue. A smaller step delays the mismatch, but after enough repeated steps the batch is still drawn from the wrong distribution.

## 3. Importance sampling for trajectories

**Transcript coverage:** lines 1114-1566

### What the lecturer said - transcript only

Importance sampling rewrites an expectation under a target distribution $p$ using samples from another distribution $q$:

$$
\mathbb E_{x\sim p}[f(x)]
=
\mathbb E_{x\sim q}
\left[\frac{p(x)}{q(x)}f(x)\right].
$$

The ratio is valid only where the sampling distribution has support wherever the target distribution does. Even with formal support, very small sampling probabilities can create enormous weights and high variance.

For an MDP trajectory,

$$
p_\theta(\tau)
=p(s_1)\prod_t\pi_\theta(a_t\mid s_t)
p(s_{t+1}\mid s_t,a_t).
$$

When the target and behavior policies operate in the same environment, the initial-state and transition terms cancel in the trajectory likelihood ratio. If $\theta_{\mathrm{old}}$ generated the data, then

$$
\frac{p_\theta(\tau)}{p_{\theta_{\mathrm{old}}}(\tau)}
=\prod_t
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

Thus an off-policy correction can be computed from policy probabilities without knowing the environment dynamics.

### Source reconciliation

Slides 6-7 show the importance-sampling identity and the cancellation of dynamics and initial-state probabilities in the trajectory ratio.

### Additional explanation

The cancellation is one of the most useful properties of likelihood-ratio policy methods. It does not say dynamics are irrelevant to which states appear; it says their explicit likelihood factors are identical in numerator and denominator.

## 4. Re-deriving the off-policy policy gradient

**Transcript coverage:** lines 1567-2244

### What the lecturer said - transcript only

The lecturer re-derived the policy gradient with trajectories sampled from an older policy. Applying importance sampling to the usual likelihood-ratio expression yields a trajectory-weighted estimator. By causality, a reward at time $t$ cannot depend on actions taken after time $t$, so likelihood-ratio factors for future actions can be removed from that reward's term.

In its causal form, a return contribution can be corrected using the product of policy ratios only through the relevant time step:

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim p_{\mathrm{old}}}
\left[
\sum_t
\left(\prod_{t'=1}^{t}
\frac{\pi_\theta(a_{t'}\mid s_{t'})}
{\pi_{\mathrm{old}}(a_{t'}\mid s_{t'})}
\right)
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\widehat Q_t
\right].
$$

The on-policy formula is recovered when the policies are equal and every ratio is one.

The lecturer then motivated an approximation used in practice. If advantages are computed under the old policy, they can be treated as measures of how actions should change during a local policy-improvement step. This resembles policy iteration: evaluate the old policy, then improve relative to its advantage estimates.

### Source reconciliation

Slides 8-9 develop the same derivation and explicitly use causality to remove future importance weights.

### Additional explanation

The exact causal estimator still contains products over time. It is unbiased under ideal assumptions, but each random ratio multiplies all earlier ones. That makes the estimator impractical for long horizons even before approximation error in the advantage is considered.

## 5. State-distribution approximation and a multi-step algorithm

**Transcript coverage:** lines 2245-2988

### What the lecturer said - transcript only

Products of importance ratios can grow or shrink exponentially with horizon, producing extreme variance. An equivalent exact expression can be written with a marginal state-action ratio:

$$
\nabla_\theta J(\theta)
=
\mathbb E_{s\sim d^{\pi_{\mathrm{old}}},\,a\sim\pi_{\mathrm{old}}}
\left[
\frac{d^{\pi_\theta}(s)}{d^{\pi_{\mathrm{old}}}(s)}
\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}
\nabla_\theta\log\pi_\theta(a\mid s)
A^{\pi_{\mathrm{old}}}(s,a)
\right].
$$

The state-visitation ratio is hard to calculate. The practical approximation is to ignore it and retain only the one-step action probability ratio:

$$
\nabla_\theta J(\theta)
\approx
\mathbb E_{(s,a)\sim\pi_{\mathrm{old}}}
\left[
\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}
\nabla_\theta\log\pi_\theta(a\mid s)
A^{\pi_{\mathrm{old}}}(s,a)
\right].
$$

This is a first-order or local approximation: it is plausible only while the new and old policies remain close, because then their state distributions have not changed much.

The resulting multi-step algorithm stores the old policy probabilities with the batch, estimates advantages once, and performs several gradient steps using the ratio between the current and old action probabilities. The lecturer gave roughly 10-50 optimization steps on one batch as a common scale, while emphasizing that the safe number depends on step size and policy drift.

### Source reconciliation

Slides 9-11 show the exact state/action-marginal expression, cross out the state ratio as the practical approximation, and present the repeated-update algorithm.

### Additional explanation

Ignoring the state ratio is the central approximation behind this presentation of PPO. A ratio near one at sampled states is evidence of local action-distribution similarity, but it does not by itself prove that visitation distributions match everywhere.

## Part II - Constraining how much a reused batch can change the policy

## 6. Finite-sample variance and policy drift

**Transcript coverage:** lines 2989-3285

### What the lecturer said - transcript only

Even a formally correct importance-sampled estimator can fail with a finite batch. As the current policy moves away from the old one, a few samples can receive very large weights and dominate the update, while actions important under the new policy may be absent from the old batch.

The lecturer returned to the Gaussian illustration: the distribution can become elongated or shift so that the old points no longer represent it, yet optimization continues to reward those points. This is variance and batch overfitting, not a contradiction of the expectation identity.

To compute the ratios, the implementation must retain either the old policy itself or the old log probability for each sampled action. The latter is usually sufficient and cheaper.

### Source reconciliation

Slides 12-13 emphasize that importance sampling is unbiased in expectation but can have unusably high finite-sample variance when the policies are far apart.

### Additional explanation

Log probabilities are stored in practice because products and ratios in probability space can underflow. The ratio is computed as $\exp(\log\pi_\theta-\log\pi_{\mathrm{old}})$.

## 7. Clipped importance ratios

**Transcript coverage:** lines 3286-3840

### What the lecturer said - transcript only

One way to prevent a sample from exerting unbounded influence is to clip its probability ratio. For

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)},
$$

replace the ratio by a value restricted to $[1-\epsilon,1+\epsilon]$. The lecturer described $\epsilon\approx0.1$ as a characteristic value, not a law.

The old policy remains fixed as the anchor throughout the repeated inner updates. Clipping reduces variance and limits how much a sampled action can benefit from moving farther in the same direction, but it introduces bias because the clipped expression is no longer exact importance sampling. The method deliberately accepts bias for a more usable finite-sample estimator.

### Source reconciliation

Slide 14 displays the clipped range and stresses that the reference policy is the policy that generated the batch.

### Additional explanation

Ratio clipping is a samplewise constraint, not a direct global distance bound. Some unsampled actions can still change, and the aggregate KL divergence should therefore be monitored in implementations.

## 8. The asymmetric PPO clipped objective

**Transcript coverage:** lines 3841-4275

### What the lecturer said - transcript only

Simply replacing every ratio with its clipped version creates an undesirable flat region. Once the ratio saturates, the objective cannot distinguish between a benign update and a movement that makes the policy worse.

PPO instead takes the smaller of the unclipped and clipped surrogate terms:

$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb E_t\left[
\min\left(
r_t(\theta)\widehat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\widehat A_t
\right)
\right].
$$

The minimum makes clipping asymmetric and pessimistic. If $\widehat A_t>0$, PPO stops giving extra credit when the sampled action becomes more than $1+\epsilon$ times as probable, but it still penalizes making that good action much less probable. If $\widehat A_t<0$, it stops rewarding the policy for making the bad action less than $1-\epsilon$ as probable, but still penalizes making it more probable. Thus the plateau appears only in the direction that would otherwise over-improve the finite batch.

The lecturer mentioned an alternative of simply stopping optimization when a distance threshold is exceeded, but the clipped objective provides a differentiable, samplewise mechanism.

### Source reconciliation

Slide 15 plots the positive- and negative-advantage cases and shows why the $\min$ is necessary. The word “pessimistic” describes the objective's lower-bound-like local behavior; it is not a theorem that the surrogate is a global lower bound on true return.

### Additional explanation

The sign of the advantage is what reverses the safe clipping direction. Remembering the two cases is more informative than treating the `min` as an arbitrary implementation trick.

## 9. PPO in practice and closing Q&A

**Transcript coverage:** lines 4276-4504

### What the lecturer said - transcript only

The practical PPO loop is:

1. run the current policy and collect a batch;
2. estimate returns and advantages, commonly with GAE;
3. retain the old action log probabilities;
4. optimize the clipped surrogate for multiple mini-batch epochs;
5. fit the critic using the same collected data; and
6. replace the old policy with the updated policy and collect fresh data.

An entropy bonus is often added to discourage premature loss of randomness. The lecturer noted that PPO has been used for demanding examples including language-model fine-tuning and learning acrobatic behaviors such as backflips.

In closing questions, he clarified that GAE supplies advantage estimates for the policy update and return-like targets for critic training. The clipped ratio is always relative to the fixed policy that collected the current batch, not to the immediately preceding mini-batch step. Once the prescribed inner optimization is finished, new data are collected and the anchor is replaced.

### Source reconciliation

Slide 16 provides the complete PPO clipped algorithm and includes the optional entropy term.

### Additional explanation

PPO remains an on-policy family in the broad sense: it reuses each fresh batch for several epochs, but it does not normally maintain a long-lived replay buffer containing data from many old policies.

## Consolidated takeaways

1. Plain policy-gradient updates are reliable but discard a batch after essentially one policy step.
2. Repeating the on-policy gradient on old data violates the sampling assumption and overfits the batch.
3. Importance sampling corrects a distribution mismatch through likelihood ratios; environment dynamics cancel in trajectory ratios.
4. Exact trajectory or state-distribution corrections have prohibitive variance or are difficult to compute.
5. Practical multi-step policy optimization keeps the one-step action ratio and assumes policy changes are local enough to ignore state-distribution shift.
6. The old policy remains the fixed reference throughout all inner updates on a batch.
7. Clipping trades unbiasedness for lower variance and controlled sample influence.
8. PPO's minimum creates the correct asymmetric plateau for positive and negative advantages.
9. PPO combines the clipped actor objective with advantage estimation, critic regression, mini-batches, and often entropy regularization.

## Key equations

### Importance sampling

$$
\mathbb E_{x\sim p}[f(x)]
=\mathbb E_{x\sim q}\left[\frac{p(x)}{q(x)}f(x)\right].
$$

### Trajectory likelihood ratio

$$
\frac{p_\theta(\tau)}{p_{\mathrm{old}}(\tau)}
=\prod_t\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)}.
$$

### PPO probability ratio

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)}.
$$

### PPO clipped surrogate

$$
L^{\mathrm{CLIP}}(\theta)
=\mathbb E_t\left[
\min\left(
r_t(\theta)\widehat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\widehat A_t
\right)
\right].
$$

## Glossary

- **Action probability ratio:** current policy probability divided by the behavior policy probability for a sampled action.
- **Behavior policy:** the policy that generated the stored trajectories.
- **Causality correction:** removal of future policy ratios from a reward term that cannot depend on future actions.
- **Clipped surrogate:** PPO's pessimistic sample objective formed from the minimum of unclipped and clipped ratio terms.
- **GAE:** generalized advantage estimation, a weighted combination of temporal-difference residuals controlled by $\lambda$.
- **Importance sampling:** estimation under one distribution using samples from another and a density ratio.
- **On-policy:** using data sampled from the policy currently being evaluated or optimized.
- **Policy drift:** change between the current policy and the policy that generated a batch.
- **PPO:** proximal policy optimization, a family of repeated-update policy-gradient methods; this lecture focuses on the clipped variant.
- **State-visitation distribution:** the frequency with which a policy reaches each state over its trajectories.
- **Support:** the set of outcomes to which a distribution assigns nonzero probability.

## Self-check questions

1. When is throwing away policy-gradient samples inexpensive, and when is it costly?
2. Why can the same finite Gaussian samples push the mean too far under repeated plain updates?
3. Which MDP factors cancel in a trajectory importance ratio?
4. What support condition is required for importance sampling?
5. How does causality shorten the product of trajectory ratios?
6. Why are products of ratios unusable on long trajectories?
7. Which ratio does the practical approximation ignore?
8. Why must the current policy remain close to the old policy for that approximation to be plausible?
9. What is stored with a PPO batch to compute the ratio efficiently?
10. What bias-variance tradeoff does clipping make?
11. Why is clipping alone insufficient without the asymmetric minimum?
12. How does the clipped objective behave for positive versus negative advantages?
13. Why is PPO still broadly considered on-policy despite multiple epochs of reuse?

## Source coverage checklist

| Topic | Inclusive range | Coverage status |
|---:|---:|---|
| 1 | 1-585 | Accounted for |
| 2 | 586-1113 | Accounted for |
| 3 | 1114-1566 | Accounted for |
| 4 | 1567-2244 | Accounted for |
| 5 | 2245-2988 | Accounted for |
| 6 | 2989-3285 | Accounted for |
| 7 | 3286-3840 | Accounted for |
| 8 | 3841-4275 | Accounted for |
| 9 | 4276-4504 | Accounted for |

**Coverage result:** All 4,504 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. All 16 slide pages were visually inspected and reconciled with the transcript.
