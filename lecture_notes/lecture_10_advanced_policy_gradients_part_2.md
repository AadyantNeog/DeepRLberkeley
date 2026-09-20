---
title: "Lecture 10 - Advanced Policy Gradients, Part 2"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 10
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 10, Advanced Policy Gradient.txt"
source_slides: "../lectures/Lecture 10 - Advanced Policy Gradients Part 2.pdf"
transcript_lines: 7825
slide_pages: 33
status: "complete"
---

# Lecture 10: Advanced Policy Gradients, Part 2

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Recap, two PPO approximations, and agenda | lines 1-862 |
| 2 | Policy gradient as policy iteration | lines 863-1797 |
| 3 | Why an old-policy advantage is valid | lines 1798-2238 |
| 4 | State-distribution change for deterministic policies | lines 2239-2694 |
| 5 | Stochastic policies, coupling, and total variation | lines 2695-3462 |
| 6 | A conservative performance bound | lines 3463-4698 |
| 7 | Policy constraints, Pinsker's inequality, and KL | lines 4699-5271 |
| 8 | KL-penalty PPO and dual adaptation | lines 5272-6099 |
| 9 | Linearizing the trust-region objective | lines 6100-6465 |
| 10 | Fisher geometry and the natural gradient | lines 6466-6957 |
| 11 | Natural-gradient intuition and TRPO | lines 6958-7593 |
| 12 | Review and practical guidance | lines 7594-7825 |

## Part I - A policy-improvement view of policy gradients

## 1. Recap, two PPO approximations, and agenda

**Transcript coverage:** lines 1-862

### What the lecturer said - transcript only

After pre-lecture setup and questions, the lecturer recapped the previous lecture's way of reusing a policy-gradient batch. The exact off-policy expression contains both a state-visitation ratio and an action-probability ratio. The practical method makes two simplifying moves: it uses advantages evaluated under the old policy and it ignores the mismatch between the old and new state distributions.

The first move is easier to justify through policy iteration. The second is safe only if the policy has not moved far. This motivates the idea of a trust region: reuse the batch for many updates—roughly 50-100 was discussed as a possible scale—but explicitly keep the resulting policy close enough to the data-generating one.

The lecture's goals were to justify the local surrogate, derive a distance-based constraint, present a KL-penalty version of PPO, and connect that constraint to natural policy gradient and trust-region policy optimization (TRPO).

### Source reconciliation

Slides 2-4 label the previous lecture's two shortcuts and list the theory and algorithmic goals.

### Additional explanation

“Close” must be defined in action-distribution space, not merely in raw neural-network parameter space. Two small parameter changes can have very different effects on policy probabilities.

## 2. Policy gradient as policy iteration

**Transcript coverage:** lines 863-1797

### What the lecturer said - transcript only

The lecturer started from a return objective

$$
J(\pi)=\mathbb E_{\tau\sim p_\pi}
\left[\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)\right]
$$

and derived the performance-difference identity. Add and subtract the old policy's value at the initial state, express that value as a telescoping discounted sequence, and combine the reward with $\gamma V^\pi(s_{t+1})-V^\pi(s_t)$. The result is

$$
J(\pi')-J(\pi)
=
\mathbb E_{\tau\sim p_{\pi'}}
\left[
\sum_{t=0}^{\infty}\gamma^t A^\pi(s_t,a_t)
\right].
$$

This says that the new policy improves exactly when, under the new policy's own trajectories, it accumulates positive advantage measured relative to the old policy. That is the policy-iteration interpretation: evaluate $\pi$, then choose $\pi'$ to favor actions with positive $A^\pi$.

Questions during the derivation focused on the telescope, the initial-state term, and why the expectation after substitution is under the new trajectory distribution. The lecturer walked through those equalities rather than treating the lemma as a black box.

### Source reconciliation

Slides 5-7 show the policy-iteration loop and the full telescoping derivation of the performance-difference lemma.

### Additional explanation

This identity is exact. The approximations enter only when its expectation under $p_{\pi'}$ is replaced with data from $p_\pi$ and when a finite-sample advantage estimate replaces $A^\pi$.

## 3. Why an old-policy advantage is valid

**Transcript coverage:** lines 1798-2238

### What the lecturer said - transcript only

The identity directly justifies using $A^\pi$ while optimizing a candidate $\pi'$. The advantage is supposed to describe which actions improve on the **old** policy; it need not be recomputed after every tiny inner update.

Applying importance sampling only to the action expectation gives a surrogate of the form

$$
\bar A(\pi')
=
\sum_t
\mathbb E_{s_t\sim p_{\pi}}
\left[
\mathbb E_{a_t\sim\pi}
\left[
\frac{\pi'(a_t\mid s_t)}{\pi(a_t\mid s_t)}
\gamma^t A^\pi(s_t,a_t)
\right]
\right].
$$

The action ratio is exact for changing the action distribution at sampled states. The unresolved issue is that the performance-difference identity needs states visited by $\pi'$, whereas the practical surrogate uses states visited by $\pi$.

### Source reconciliation

Slide 8 highlights that the old-policy advantage is justified while the old-policy state distribution remains the harder shortcut.

### Additional explanation

The surrogate and the true performance improvement agree to first order at $\pi'=\pi$. Their discrepancy grows with the state-distribution shift, which is why controlling policy distance supports repeated local optimization.

## 4. State-distribution change for deterministic policies

**Transcript coverage:** lines 2239-2694

### What the lecturer said - transcript only

The lecturer first considered a deterministic reference policy. Suppose the candidate policy chooses a different action from the reference with probability at most $\epsilon$ in every state. The probability that the two policies have made no different choice by time $t$ is at least $(1-\epsilon)^t$. Therefore the candidate state distribution can be decomposed into a no-mistake component matching the reference and a component conditioned on at least one mistake.

This yields a total-variation bound

$$
D_{\mathrm{TV}}\!\left(p_{\pi'}(s_t),p_\pi(s_t)\right)
\leq 1-(1-\epsilon)^t
\leq \epsilon t,
$$

using $(1-\epsilon)^t\geq1-\epsilon t$ for $\epsilon\in[0,1]$.

The bound becomes looser with time because an early different action may lead to a permanently different future. The lecturer related this compounding-error picture to behavior cloning.

### Source reconciliation

Slides 10-11 display the mixture decomposition and the $\epsilon t$ bound.

### Additional explanation

This is a worst-case statement. In a mixing or forgiving environment, trajectories may reconverge, while the bound assumes that any first disagreement can keep them different forever.

## 5. Stochastic policies, coupling, and total variation

**Transcript coverage:** lines 2695-3462

### What the lecturer said - transcript only

For stochastic policies, closeness is measured by total variation at each state:

$$
D_{\mathrm{TV}}\bigl(\pi'(\cdot\mid s),\pi(\cdot\mid s)\bigr)
=\frac12\sum_a\left|\pi'(a\mid s)-\pi(a\mid s)\right|
\leq\epsilon.
$$

The lecturer used a coupling lemma: if two distributions are within $\epsilon$ in total variation, there exists a joint sampling scheme with the correct marginals under which the two samples agree with probability at least $1-\epsilon$. Couple the two policies this way at every state. They then choose the same action with probability at least $1-\epsilon$ per time step, so the deterministic no-mistake argument applies to the stochastic case as well.

Audience questions asked why a joint distribution may be introduced and whether this claims that independently drawn actions agree. The lecturer clarified that coupling is a proof construction: it shows there exists a joint realization with the stated marginals and agreement probability; it is not a claim about arbitrary independent samples.

### Source reconciliation

Slide 12 states the coupling lemma and carries the state-distribution bound into the general stochastic-policy case.

### Additional explanation

Total variation has an operational meaning: it is the smallest possible disagreement probability over all couplings of the two distributions. That is precisely the quantity the trajectory proof needs.

## 6. A conservative performance bound

**Transcript coverage:** lines 3463-4698

### What the lecturer said - transcript only

If a bounded function $f(s_t)$ is evaluated under two state distributions separated in total variation, their expectations differ by at most a constant times the distance. Applying this to the policy-improvement integrand produces a lower bound on the true improvement using the old-state surrogate minus a penalty that grows with policy distance and time.

The lecturer wrote the error at step $t$ in the form $2\epsilon tC$, where $C$ bounds the relevant statewise advantage contribution. Summing over time gives a conservative guarantee. He characterized $C$ as scaling like $O(Hr_{\max})$ for a finite horizon or $O(r_{\max}/(1-\gamma))$ in a discounted problem.

The message was qualitative rather than that one should use the worst-case constant literally: ignoring the state distribution is reasonable only when the policy is close. The mathematical penalty is deliberately pessimistic and can be much larger than what occurs in practice. Q&A emphasized that the result protects against worst-case state-distribution change and does not predict the exact performance of a particular environment.

### Source reconciliation

Slide 13 shows the surrogate lower bound and labels the error scale. Earlier slides show a tighter $\epsilon t$ state-TV statement, while the performance inequality uses a conservative factor of two when translating total variation into an expectation bound.

### Additional explanation

This is the conceptual origin of a trust region: maximize the empirical advantage surrogate, but restrict the policy movement so the omitted state-distribution term cannot dominate the expected gain.

## Part II - Constraining policy updates

## 7. Policy constraints, Pinsker's inequality, and KL

**Transcript coverage:** lines 4699-5271

### What the lecturer said - transcript only

PPO clipping is a convenient approximate way of discouraging a policy from moving far, but it is not the only approach. A more direct formulation maximizes the importance-sampled advantage surrogate subject to a distributional distance constraint.

Total variation is theoretically convenient but awkward to estimate and differentiate. The lecturer introduced Pinsker's inequality, though the automatic transcript repeatedly renders the name as “Pinkster”:

$$
D_{\mathrm{TV}}(p,q)
\leq\sqrt{\frac12D_{\mathrm{KL}}(p\|q)}.
$$

Thus a small KL divergence implies a small total-variation distance. KL is

$$
D_{\mathrm{KL}}(p_1\|p_2)
=\mathbb E_{x\sim p_1}
\left[\log\frac{p_1(x)}{p_2(x)}\right].
$$

With samples from the old policy, the part involving $\log\pi_{\mathrm{old}}$ is constant when optimizing the new policy. Minimizing the forward old-to-new KL therefore amounts to maximizing the likelihood of the old samples under the new policy.

### Source reconciliation

Slides 16-18 show Pinsker's inequality, the KL definition, and its sample estimate. The transcript's “Pinkster” is a transcription error preserved here as an anomaly; the slide identifies **Pinsker's inequality**.

### Additional explanation

KL is asymmetric. The lecture uses $D_{\mathrm{KL}}(\pi_{\mathrm{old}}\|\pi_{\mathrm{new}})$ because it can be estimated directly with old-policy actions. Different implementations sometimes constrain the reverse direction or a symmetric approximation.

## 8. KL-penalty PPO and dual adaptation

**Transcript coverage:** lines 5272-6099

### What the lecturer said - transcript only

The constrained improvement problem is

$$
\max_{\theta'}\;\bar A(\theta')
\quad\text{subject to}\quad
D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid s)\|\pi_{\theta'}(\cdot\mid s)\right)
\leq\epsilon.
$$

Introduce a Lagrange multiplier $\beta$ and optimize

$$
\mathcal L(\theta',\beta)
=\bar A(\theta')
-\beta\left(D_{\mathrm{KL}}(\pi_\theta\|\pi_{\theta'})-\epsilon\right).
$$

For a fixed $\beta$, take several gradient steps in $\theta'$. Then adapt the multiplier:

$$
\beta\leftarrow\beta+\alpha_\beta
\left(D_{\mathrm{KL}}(\pi_\theta\|\pi_{\theta'})-\epsilon\right).
$$

If the policy moved too far, increase the penalty; if it moved less than the target, decrease it. The lecturer described this as dual gradient descent.

With samples, the KL term becomes a log-likelihood penalty and the advantage term uses the action probability ratio. A practical KL-PPO loop collects data, trains the critic and GAE advantages, fixes the old policy, performs roughly 20-50 inner actor steps, adjusts $\beta$, then adopts the new policy and gathers a fresh batch.

The lecturer contrasted this version with clipped PPO. Both are optional formulations of PPO; clipping is more common because it has fewer moving parts, while a KL penalty gives a direct, interpretable measure of policy change. Questions addressed averaging the KL over sampled states, the sign of the dual update, and why optimization of the actor may be incomplete between multiplier updates.

### Source reconciliation

Slides 19-22 give the constrained problem, Lagrangian, sample objective, and KL-PPO algorithm.

### Additional explanation

An adaptive penalty is a soft constraint: it can temporarily violate the target. A hard trust-region solver instead computes a step designed to remain within the approximate constraint on every outer iteration.

## Part III - Natural gradients and trust regions

## 9. Linearizing the trust-region objective

**Transcript coverage:** lines 6100-6465

### What the lecturer said - transcript only

Natural policy gradient begins from the same constrained surrogate but replaces the objective locally with its first-order Taylor approximation:

$$
\bar A(\theta')
\approx
\bar A(\theta)+g^\top(\theta'-\theta),
\qquad
g=\nabla_\theta J(\theta).
$$

At the reference policy, the gradient of the importance-sampled surrogate is exactly the ordinary on-policy policy gradient because the probability ratio equals one. The remaining problem is to choose a step that optimizes this linear objective while respecting a KL trust region.

### Source reconciliation

Slides 23-25 show the linearization and reduce the surrogate gradient at $\theta'=\theta$ to the familiar policy gradient.

### Additional explanation

Linearization is sensible only locally. The constraint supplies the definition of “local”; without it, maximizing a linear function would send the step to infinity.

## 10. Fisher geometry and the natural gradient

**Transcript coverage:** lines 6466-6957

### What the lecturer said - transcript only

Ordinary gradient ascent constrains Euclidean parameter distance. That is inappropriate for a policy because some parameters can change action probabilities dramatically while others have little effect.

Near $\theta$, the KL divergence has a second-order approximation

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\theta'})
\approx
\frac12(\theta'-\theta)^\top F(\theta'-\theta),
$$

where the Fisher information matrix is

$$
F=
\mathbb E_{s,a\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)^\top
\right].
$$

The trust region is therefore an ellipse in parameter space, narrow in sensitive directions and wide in insensitive directions. Solving the linear objective with this quadratic constraint changes the gradient direction by $F^{-1}$; this is the natural gradient.

An audience member asked about computing an inverse for a seven-billion-parameter model. The lecturer agreed that an explicit $7\text{B}\times7\text{B}$ matrix is impossible and deferred to matrix-free methods discussed next.

### Source reconciliation

Slides 26-28 show the Euclidean ball, Fisher quadratic form, and natural-gradient update. Slide 28 prints
$\alpha=\sqrt{2\epsilon/(g^\top Fg)}$ next to $\theta'=\theta+\alpha F^{-1}g$. Those two displayed formulas are algebraically inconsistent. Solving the stated constrained problem gives

$$
\theta'=\theta+
\sqrt{\frac{2\epsilon}{g^\top F^{-1}g}}\,F^{-1}g.
$$

This correction is isolated here rather than silently substituted into the transcript-only account.

### Additional explanation

The natural gradient is invariant to smooth reparameterizations in the ideal distribution-space geometry. It asks for a direction that changes the policy distribution efficiently, not one that merely moves far in raw parameter coordinates.

## 11. Natural-gradient intuition and TRPO

**Transcript coverage:** lines 6958-7593

### What the lecturer said - transcript only

The lecturer used a one-dimensional quadratic-control example with a Gaussian policy whose mean was controlled by a feedback gain $k$ and whose standard deviation was another parameter. Vanilla gradients could shrink the standard deviation too quickly or move through an ill-conditioned parameterization, whereas a natural gradient respected the sensitivity of the induced action distribution and moved in a more appropriate direction.

TRPO implements the trust-region idea without explicitly constructing or inverting the Fisher matrix. Conjugate gradient approximately solves systems involving $F$ using Fisher-vector products, which can be computed with automatic differentiation. A line search then scales the candidate step to satisfy the empirical KL constraint and improve the surrogate.

The lecturer described TRPO as theoretically motivated but more complex and currently less popular than PPO. PPO usually wins on simplicity, even though its clipping or penalty only approximates the trust-region behavior.

Questions revisited whether the Fisher is singular, why damping may be added, and why matrix-vector products avoid storing the matrix while still approximating $F^{-1}g$.

### Source reconciliation

Slides 29-30 show the Gaussian toy problem and summarize TRPO's use of conjugate gradient and a line search.

### Additional explanation

Conjugate gradient needs only a function that returns $Fv$. It builds an approximate solution in a Krylov subspace, which is why large neural policies can use second-order geometry without materializing a quadratic-size matrix.

## 12. Review and practical guidance

**Transcript coverage:** lines 7594-7825

### What the lecturer said - transcript only

The lecturer closed by reviewing the chain of ideas. The performance-difference lemma validates improvement using the old policy's advantage. Replacing the new state distribution with the old one is safe only locally. Total variation bounds state-distribution drift, KL provides a tractable surrogate distance, and this leads to constrained policy optimization.

Three practical realizations were compared:

- clipped PPO, the simplest and most common;
- KL-penalty PPO, which adapts a soft constraint; and
- TRPO or natural-gradient methods, which more directly use policy-distribution geometry.

The broader lesson was that policy-gradient updates should be judged by how much they change the policy distribution, not only by the optimizer's nominal learning rate or Euclidean parameter step. The lecture ended with applause.

### Source reconciliation

Slides 31-33 provide the review, lessons, and final summary.

### Additional explanation

All three methods are local policy-improvement schemes. None removes the need for reliable advantage estimates, adequate sampling, and diagnostics of actual return.

## Consolidated takeaways

1. The performance-difference lemma expresses exact improvement under a new policy using the old policy's advantages.
2. This exactly justifies the old advantage; replacing the new state distribution with the old one is the approximation.
3. If policies are within $\epsilon$ in per-state total variation, their state distributions can differ by at most a quantity growing roughly as $\epsilon t$.
4. Coupling extends the deterministic disagreement argument to stochastic policies.
5. A conservative performance bound motivates constraining every reused-batch policy update.
6. Pinsker's inequality lets a KL constraint control total variation.
7. KL-PPO uses a Lagrange multiplier adapted to a desired policy distance.
8. Natural gradient linearizes the objective and uses the Fisher matrix as the local policy metric.
9. TRPO approximates $F^{-1}g$ with Fisher-vector products and conjugate gradient.
10. Clipped PPO is more common in practice because it approximates the same local-update principle with less machinery.

## Key equations

### Performance-difference lemma

$$
J(\pi')-J(\pi)
=\mathbb E_{\tau\sim p_{\pi'}}
\left[\sum_{t=0}^\infty\gamma^tA^\pi(s_t,a_t)\right].
$$

### Per-state policy distance

$$
D_{\mathrm{TV}}(\pi',\pi)
=\frac12\sum_a|\pi'(a\mid s)-\pi(a\mid s)|.
$$

### State-distribution bound

$$
D_{\mathrm{TV}}(p_{\pi'}(s_t),p_\pi(s_t))
\leq1-(1-\epsilon)^t\leq\epsilon t.
$$

### Pinsker's inequality

$$
D_{\mathrm{TV}}(p,q)
\leq\sqrt{\frac12D_{\mathrm{KL}}(p\|q)}.
$$

### KL-penalized objective

$$
\mathcal L(\theta',\beta)
=\bar A(\theta')
-\beta\left(D_{\mathrm{KL}}(\pi_\theta\|\pi_{\theta'})-\epsilon\right).
$$

### Natural-gradient trust-region step

$$
\delta\theta
=
\sqrt{\frac{2\epsilon}{g^\top F^{-1}g}}\,F^{-1}g.
$$

The slide's printed scalar normalization is inconsistent with its own $F^{-1}g$ direction; see the source reconciliation in Section 10.

## Glossary

- **Coupling:** a joint distribution with specified marginals, used here to maximize agreement between two policy samples.
- **Dual gradient descent:** adaptation of a Lagrange multiplier according to constraint violation.
- **Fisher information matrix:** the expected outer product of policy score gradients, giving the local KL geometry.
- **KL divergence:** an asymmetric distributional discrepancy used as a practical policy-distance constraint.
- **Natural gradient:** the gradient transformed by $F^{-1}$ to account for distribution-space sensitivity.
- **Performance-difference lemma:** an exact identity expressing return improvement through old-policy advantages evaluated on new-policy trajectories.
- **Pinsker's inequality:** a bound relating total variation to KL divergence.
- **Policy coupling:** a proof construction in which two policies select the same action with maximal possible probability.
- **TRPO:** trust-region policy optimization, which approximately solves a KL-constrained surrogate problem.
- **Trust region:** a neighborhood within which a local approximation is intended to remain accurate.
- **Total variation distance:** half the $L_1$ distance between probability distributions.

## Self-check questions

1. State the performance-difference lemma and identify which policy supplies the advantage.
2. Why does the lemma justify retaining old-policy advantages across inner updates?
3. Which distribution mismatch remains after action importance sampling?
4. How does the no-disagreement probability produce the $\epsilon t$ bound?
5. What role does coupling play for stochastic policies?
6. Why is the resulting performance guarantee conservative?
7. How does Pinsker's inequality motivate a KL constraint?
8. Why is old-to-new KL convenient to estimate from a fresh batch?
9. How should $\beta$ change when the observed KL exceeds its target?
10. What is linearized in natural policy gradient, and what is approximated to second order?
11. Why is a Euclidean parameter ball not a meaningful policy trust region?
12. What does $F^{-1}$ do to the vanilla gradient?
13. How does TRPO avoid constructing a full Fisher matrix?
14. What inconsistency appears in the scalar step-size formula on slide 28?

## Source coverage checklist

| Topic | Inclusive range | Coverage status |
|---:|---:|---|
| 1 | 1-862 | Accounted for |
| 2 | 863-1797 | Accounted for |
| 3 | 1798-2238 | Accounted for |
| 4 | 2239-2694 | Accounted for |
| 5 | 2695-3462 | Accounted for |
| 6 | 3463-4698 | Accounted for |
| 7 | 4699-5271 | Accounted for; transcript name anomaly reconciled |
| 8 | 5272-6099 | Accounted for |
| 9 | 6100-6465 | Accounted for |
| 10 | 6466-6957 | Accounted for; slide equation inconsistency flagged |
| 11 | 6958-7593 | Accounted for |
| 12 | 7594-7825 | Accounted for |

**Coverage result:** All 7,825 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. All 33 slide pages were visually inspected; the transcript naming anomaly and slide 28 equation inconsistency are explicitly identified rather than silently repaired.
