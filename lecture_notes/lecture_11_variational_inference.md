---
title: "Lecture 11 - Variational Inference"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 11
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 11, Variational Inference.txt"
source_slides: "../lectures/Lecture 11 - Variational Inference.pdf"
transcript_lines: 3934
slide_pages: 18
status: "source-incomplete"
---

# Lecture 11: Variational Inference

> **Source warning:** The transcript includes the closing farewell but ends at line 3,934 in the middle of the final teaser: “Where you get rid of all the”. The missing words are not inferred in the transcript-only account.

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Probability models and conditional distributions | lines 1-354 |
| 2 | Discrete and continuous latent-variable models | lines 355-977 |
| 3 | RL uses and the maximum-likelihood obstacle | lines 978-1257 |
| 4 | Posterior inference and expected complete-data likelihood | lines 1258-1695 |
| 5 | Choosing a tractable variational approximation | lines 1696-1914 |
| 6 | Deriving the evidence lower bound with Jensen's inequality | lines 1915-2505 |
| 7 | Entropy and KL divergence | lines 2506-2999 |
| 8 | The ELBO gap and optimization of the posterior | lines 3000-3558 |
| 9 | Coordinate optimization, EM, scaling, and closing Q&A | lines 3559-3934 |

## Part I - Why latent-variable models need inference

## 1. Probability models and conditional distributions

**Transcript coverage:** lines 1-354

### What the lecturer said - transcript only

The lecturer introduced variational inference as a general tool for training and reasoning about models with variables that are not observed. Although the lecture would use probability notation rather than RL notation at first, the same ideas would later apply to policies, partially observed states, and reward-learning problems.

A model $p(x)$ assigns probabilities or densities to observations. A conditional model $p(y\mid x)$ assigns a distribution to an output after an input is given. A stochastic policy is one example: $p(a\mid s)$ can be written as $\pi(a\mid s)$. The point of the notation was to make clear that a neural network can parameterize a probability distribution rather than merely output one deterministic number.

### Source reconciliation

Slides 2-3 introduce latent-variable and probabilistic models and connect conditional distributions to familiar learned mappings.

### Additional explanation

For continuous variables, $p(x)$ is a density, so its value need not itself be a probability in $[0,1]$. Probabilities are obtained by integrating the density over a region.

## 2. Discrete and continuous latent-variable models

**Transcript coverage:** lines 355-977

### What the lecturer said - transcript only

A latent-variable model introduces an unobserved $z$ and writes

$$
p(x)=\int p(x\mid z)p(z)\,dz
$$

for a continuous latent variable, or a sum for a discrete one. A mixture model is the simplest picture: $z$ identifies which component generated a data point. Different data points have different latent assignments.

The lecturer connected this to multimodal imitation. If demonstrations go left or right around an obstacle, a single simple output distribution may average the modes. A latent $z$ can represent which behavioral mode is active. He also related latent randomness to the noise variable used by flow-matching generative policies: a sampled latent determines which output mode is produced.

In a continuous latent model, $p(z)$ can be extremely simple—often a standard Gaussian—while $p_\theta(x\mid z)$ is a neural-network transformation. The marginal $p_\theta(x)$ can then be very complicated even though the prior is simple. Sampling is easy: sample $z\sim p(z)$, then sample $x\sim p_\theta(x\mid z)$. Evaluating or training the marginal is harder because it requires integrating over every possible $z$.

### Source reconciliation

Slides 4-6 show mixture models, a continuous latent $z$, and a neural conditional distribution that turns a simple prior into a complex marginal.

### Additional explanation

The latent variable is not automatically meaningful. Its semantics emerge only through the model, data, and training objective; multiple latent parameterizations can describe the same marginal distribution.

## 3. RL uses and the maximum-likelihood obstacle

**Transcript coverage:** lines 978-1257

### What the lecturer said - transcript only

The lecturer listed several RL uses for latent variables:

- representing a multimodal policy or imitation distribution;
- modeling an unobserved state in a partially observed process;
- representing hidden human preferences or rewards; and
- supporting probabilistic inference for exploration or human-feedback problems.

For data $\mathcal D=\{x_i\}_{i=1}^N$, maximum likelihood would optimize

$$
\max_\theta\frac1N\sum_i\log p_\theta(x_i)
=
\max_\theta\frac1N\sum_i
\log\left(\int p_\theta(x_i\mid z)p(z)\,dz\right).
$$

The log lies outside the integral. For a neural decoder and a high-dimensional latent, this integral is generally intractable. Naively sampling $z$ does not solve the problem: a Monte Carlo average can estimate $p(x)$, but taking a log of that random estimate is not an unbiased estimate of $\log p(x)$.

### Source reconciliation

Slide 7 marks the log-of-integral likelihood as intractable. The transcript additionally explains why “sample and then take the log” is not an unbiased shortcut.

### Additional explanation

The obstacle is sometimes called the marginal-likelihood or evidence problem. Even when generating from the model is cheap, summing over every hidden explanation of an observed point can be computationally prohibitive.

## 4. Posterior inference and expected complete-data likelihood

**Transcript coverage:** lines 1258-1695

### What the lecturer said - transcript only

If the latent variable for $x_i$ were known, one could train on the joint or complete-data likelihood. Since it is not known, a natural alternative is to average over plausible latent explanations using the posterior:

$$
\mathbb E_{z\sim p(z\mid x_i)}
\left[\log p_\theta(x_i,z)\right].
$$

The intuition is not to guess one $z$ with certainty but to weight all possible $z$ values according to how likely they are after observing $x_i$. Computing $p(z\mid x_i)$ is probabilistic inference.

Unfortunately, the exact posterior

$$
p(z\mid x_i)=\frac{p(x_i\mid z)p(z)}{p(x_i)}
$$

contains the same intractable evidence $p(x_i)$ in its denominator. This circularity is why an approximation is needed. The lecturer pointed out that posterior inference is broadly useful: it can infer hidden causes of behavior, latent goals, or uncertain states, not only train a generative model.

### Source reconciliation

Slides 8-9 present the expected log-likelihood idea and identify posterior calculation as probabilistic inference, then list applications.

### Additional explanation

The posterior combines the prior plausibility of a latent with how well that latent explains the observation. Variational inference replaces this hard posterior calculation with optimization over a manageable family.

## Part II - The variational lower bound

## 5. Choosing a tractable variational approximation

**Transcript coverage:** lines 1696-1914

### What the lecturer said - transcript only

For each observation $x_i$, choose an approximate posterior $q_i(z)$. A convenient example is a Gaussian $q_i(z)=\mathcal N(\mu_i,\sigma_i)$, with parameters stored for that data point. This distribution will generally be wrong; the reason for choosing it is that it can be sampled, evaluated, and optimized.

Every legitimate density $q_i$ can be used to construct a lower bound. A poor $q_i$ gives a loose bound, while one close to the true posterior gives a useful one. At this stage the lecturer had not yet supplied the criterion for “close”; that would be derived through KL divergence.

### Source reconciliation

Slides 10-11 introduce the variational-inference section and label the Gaussian approximation “incorrect but very convenient.”

### Additional explanation

The chosen set of $q$ distributions is called the **variational family**. Its expressiveness determines an approximation gap, while its structure determines how easy inference and optimization are.

## 6. Deriving the evidence lower bound with Jensen's inequality

**Transcript coverage:** lines 1915-2505

### What the lecturer said - transcript only

Insert $q_i(z)/q_i(z)$ into the evidence integral:

$$
\log p(x_i)
=
\log\int q_i(z)
\frac{p(x_i\mid z)p(z)}{q_i(z)}\,dz
=
\log\mathbb E_{z\sim q_i}
\left[\frac{p(x_i\mid z)p(z)}{q_i(z)}\right].
$$

Because the logarithm is concave, Jensen's inequality gives

$$
\log\mathbb E[Y]\geq\mathbb E[\log Y].
$$

Therefore

$$
\log p(x_i)
\geq
\mathbb E_{z\sim q_i}
\left[
\log p(x_i\mid z)+\log p(z)-\log q_i(z)
\right].
$$

Using the entropy $\mathcal H(q_i)=-\mathbb E_{q_i}[\log q_i(z)]$, define the evidence lower bound (ELBO)

$$
\mathcal L_i(p,q_i)
=
\mathbb E_{z\sim q_i}
\left[\log p(x_i\mid z)+\log p(z)\right]
+\mathcal H(q_i).
$$

Maximizing this tractable quantity raises a lower bound on the desired log likelihood. The lecturer stressed that it is a valid bound for any $q_i$ but may be inaccurate if $q_i$ is poorly chosen.

### Source reconciliation

Slide 12 contains the full Jensen derivation. Its exported page contains overlapping annotation remnants near the final entropy term, but the preceding algebra and the spoken derivation unambiguously give $+\mathcal H(q_i)$.

### Additional explanation

The ELBO is a surrogate objective with two jobs: fit the generative model and make the approximate posterior useful enough that the surrogate remains close to the evidence.

## 7. Entropy and KL divergence

**Transcript coverage:** lines 2506-2999

### What the lecturer said - transcript only

Entropy is

$$
\mathcal H(q)=-\mathbb E_{z\sim q}[\log q(z)].
$$

It measures randomness or spread. A concentrated distribution has low entropy; a broad distribution has higher entropy. In the ELBO, the expected log joint rewards putting $q_i$ where $p(x_i,z)$ is large, while the entropy discourages $q_i$ from collapsing to an arbitrarily narrow spike. The two terms encourage a distribution that occupies plausible latent regions while remaining broad.

KL divergence is

$$
D_{\mathrm{KL}}(q\|p)
=\mathbb E_{z\sim q}
\left[\log\frac{q(z)}{p(z)}\right]
=-\mathbb E_q[\log p(z)]-\mathcal H(q).
$$

It is nonnegative and is zero only when the distributions agree almost everywhere. It is not symmetric. The lecturer began connecting the ELBO to the KL between the approximate posterior and the true posterior.

### Source reconciliation

Slides 13-14 define entropy and KL and use drawings to illustrate the expected-log-density and spread terms.

### Additional explanation

For $D_{\mathrm{KL}}(q\|p)$, expectations are taken under $q$. If $q$ places mass where $p$ is tiny, the penalty is large. Regions that only $p$ covers do not contribute directly, which is associated with the mode-seeking behavior of this KL direction.

## 8. The ELBO gap and optimization of the posterior

**Transcript coverage:** lines 3000-3558

### What the lecturer said - transcript only

The lecturer asked what makes $q_i$ good and answered: it should have small KL divergence to $p(z\mid x_i)$. Expanding that KL with Bayes' rule gives

$$
\begin{aligned}
D_{\mathrm{KL}}\bigl(q_i(z)\|p(z\mid x_i)\bigr)
&=\mathbb E_{q_i}
\left[\log\frac{q_i(z)}{p(z\mid x_i)}\right]\\
&=-\mathcal L_i(p,q_i)+\log p(x_i).
\end{aligned}
$$

Rearranging,

$$
\log p(x_i)
=
\mathcal L_i(p,q_i)
+D_{\mathrm{KL}}\bigl(q_i(z)\|p(z\mid x_i)\bigr).
$$

Since KL is nonnegative, this is another proof that the ELBO is a lower bound. The KL is exactly the gap. If $q_i$ equals the true posterior, the gap is zero and the bound is tight. If the KL is small—for example, the lecturer used $0.001$ to answer a question—only a small improvement in the bound is required before it must push against the likelihood.

One student asked why the earlier Jensen derivation was necessary if this identity also proves the bound. The lecturer said the KL identity is elegant once known, but beginning by writing the relevant KL is much less intuitive. Another question considered changing the KL while holding the bound fixed; the lecturer redirected attention to the fact that optimizing the bound is the practical operation.

For fixed $p$, $\log p(x_i)$ does not depend on $q_i$. Maximizing the ELBO with respect to $q_i$ therefore minimizes the posterior KL and tightens the bound. The same single objective can train both the model and the approximate posterior.

### Source reconciliation

Slide 15 mistakenly writes $q_i(x_i)$ in the first argument of one displayed KL. The transcript explicitly calls this a typo and says it should be $q_i(z)$. Slides 15-16 otherwise show the gap identity and the consequence that maximizing $\mathcal L_i$ with respect to $q_i$ minimizes KL.

### Additional explanation

There are two different optimization effects:

- changing $q_i$ reduces the gap for the current model;
- changing $p_\theta$ raises the lower bound and changes the true posterior, so the previously fitted $q_i$ may cease to be accurate.

This motivates alternating or simultaneous updates.

## 9. Coordinate optimization, EM, scaling, and closing Q&A

**Transcript coverage:** lines 3559-3934

### What the lecturer said - transcript only

The lecturer turned the bound into an algorithm. For each $x_i$ or mini-batch, sample $z\sim q_i(z)$, use the sample to estimate the model gradient—only $\log p_\theta(x_i\mid z)$ depends on $\theta$ in the presented setup—and update $\theta$. Then update the parameters of $q_i$, such as $\mu_i$ and $\sigma_i$, to maximize the same ELBO. Gaussian entropy and, in some cases, posterior updates may have closed forms; otherwise gradients can be used.

Optimizing $q_i$ tightens the approximate objective. Optimizing $p_\theta$ improves that objective but can increase the KL because the true posterior changes. Alternating updates lets $q_i$ keep up with the moving model. The lecturer identified expectation maximization as a special case: the E-step updates the posterior-like $q$, and the M-step improves the model, often with closed-form or non-gradient updates.

The remaining practical problem is parameter count. If every data point has its own mean and variance, then the total number of parameters is

$$
|\theta|+N\bigl(|\mu_i|+|\sigma_i|\bigr).
$$

That scales with the dataset and does not fit the usual deep-learning workflow of a fixed parameter vector and arbitrary mini-batches.

In Q&A, the lecturer said one should choose a useful variational family, but real posteriors can be so complicated that Gaussian, beta, and other simple choices may all be poor; practitioners often do not gain much by debating among several equally inadequate families. He also clarified that a $q$-update always decreases the relevant KL for fixed $p$, whereas a model update can increase it.

The lecturer thanked the class and said the next lecture would present a more practical version. The transcript ends during the final teaser after “where you get rid of all the,” so the omitted object is not supplied here.

### Source reconciliation

Slides 17-18 show the alternating optimization procedure and the $N$-dependent parameter-count problem. Slide 18 says the next lecture will explain how deep learning makes the method tractable, but that slide text is not used to complete the truncated spoken sentence.

### Additional explanation

The next step is amortization: replace a separate $(\mu_i,\sigma_i)$ with one inference network that maps each $x_i$ to its posterior parameters. That idea belongs to Lecture 12 and is stated here only as orientation, not as missing transcript content.

## Consolidated takeaways

1. Latent variables let a simple prior and conditional decoder represent complex, multimodal marginals.
2. Sampling from a latent model can be easy while evaluating its marginal likelihood is intractable.
3. Exact posterior inference contains the same hard evidence integral as maximum likelihood.
4. Variational inference introduces a tractable approximate posterior $q_i(z)$.
5. Jensen's inequality produces the ELBO: expected log joint plus posterior entropy.
6. The KL from $q_i$ to the true posterior is exactly the gap between the evidence and the ELBO.
7. Maximizing the ELBO with respect to the model improves a likelihood bound; maximizing it with respect to $q_i$ tightens that bound.
8. Alternating posterior and model updates generalizes expectation maximization.
9. A separate variational parameter vector per example scales linearly with dataset size and motivates amortized inference.

## Key equations

### Latent marginal

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz.
$$

### ELBO

$$
\mathcal L_i(p,q_i)
=\mathbb E_{q_i}
\left[\log p_\theta(x_i\mid z)+\log p(z)\right]
+\mathcal H(q_i).
$$

### Evidence decomposition

$$
\log p_\theta(x_i)
=\mathcal L_i(p,q_i)
+D_{\mathrm{KL}}\bigl(q_i(z)\|p_\theta(z\mid x_i)\bigr).
$$

### Model-gradient sample estimate

$$
\nabla_\theta\mathcal L_i
\approx\nabla_\theta\log p_\theta(x_i\mid z),
\qquad z\sim q_i(z).
$$

## Glossary

- **Approximate posterior:** a tractable distribution $q(z)$ used in place of an intractable exact posterior.
- **Complete-data likelihood:** likelihood of observed and latent variables jointly.
- **ELBO:** evidence lower bound, a tractable lower bound on log marginal likelihood.
- **Entropy:** expected negative log density under a distribution; a measure of uncertainty or spread.
- **Evidence:** the marginal likelihood $p(x)$ obtained after integrating out latent variables.
- **Expectation maximization:** alternating posterior/inference and model-maximization updates; a special case of the presented variational scheme.
- **KL divergence:** a nonnegative, asymmetric discrepancy between distributions.
- **Latent variable:** an unobserved variable used to explain variation in observed data.
- **Posterior:** the conditional distribution $p(z\mid x)$ after observing data.
- **Prior:** the distribution $p(z)$ before observing $x$.
- **Variational family:** the restricted set of distributions from which $q$ is chosen.
- **Variational inference:** optimization of a tractable distribution to approximate a posterior and form a usable likelihood bound.

## Self-check questions

1. How can a simple Gaussian prior produce a complicated marginal over $x$?
2. Why is sampling easy while maximum-likelihood evaluation is hard?
3. Why does sampling $z$ and taking a log not give an unbiased log-likelihood estimator?
4. What does the posterior $p(z\mid x)$ represent?
5. Why is the exact posterior itself intractable?
6. What makes a Gaussian $q_i$ convenient but generally incorrect?
7. Derive the ELBO by inserting $q_i(z)/q_i(z)$ and applying Jensen's inequality.
8. What roles do expected log joint and entropy play in the bound?
9. What is the exact relationship between the ELBO, evidence, and posterior KL?
10. Which update tightens the bound, and which update raises the learned objective?
11. How is expectation maximization related to these alternating updates?
12. Why does one $q_i$ per data point fail to scale?
13. What typo appears in the KL expression on slide 15?

## Source coverage checklist

| Topic | Inclusive range | Coverage status |
|---:|---:|---|
| 1 | 1-354 | Accounted for |
| 2 | 355-977 | Accounted for |
| 3 | 978-1257 | Accounted for |
| 4 | 1258-1695 | Accounted for |
| 5 | 1696-1914 | Accounted for |
| 6 | 1915-2505 | Accounted for |
| 7 | 2506-2999 | Accounted for |
| 8 | 3000-3558 | Accounted for; transcript and slide typo flagged |
| 9 | 3559-3934 | Accounted for; source ends mid-sentence |

**Coverage result:** All 3,934 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. All 18 slide pages were visually inspected; equation-export artifacts and the explicit $q_i(x_i)$ typo are reconciled, and the truncated final sentence is not completed from the slide.
