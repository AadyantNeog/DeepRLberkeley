---
title: "Lecture 15 - Model-Based RL, Part 1"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 15
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 15, Model-Based RL.txt"
source_slides: "../lectures/Lecture 15 - Model-Based RL Part 1.pdf"
transcript_lines: 4120
slide_pages: 22
status: "source-incomplete"
---

# Lecture 15: Model-Based RL, Part 1

> **Source warning:** The supplied transcript ends at line 4120 in the middle of an audience question: "How state of the arts is". The recording therefore does not contain the lecturer's answer. The slide deck also ends with a "Next time" slide and cannot reconstruct that missing exchange. All supplied lines are covered below, and the truncation is isolated rather than silently completed.

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Homework, compute, survey, and exam logistics | lines 1-442 |
| 2 | What model-based RL learns and how it is used | lines 443-865 |
| 3 | Failure modes, benefits, and problem categories | lines 866-1512 |
| 4 | Distribution shift and restrained policy improvement | lines 1513-2394 |
| 5 | Decision making under predictive uncertainty | lines 2395-3049 |
| 6 | Aleatoric and epistemic uncertainty | lines 3050-3462 |
| 7 | Bayesian neural networks, ensembles, and truncated Q&A | lines 3463-4120 |

## Part I - Logistics and the model-based recipe

## 1. Homework, compute, survey, and exam logistics

**Transcript coverage:** lines 1-442

### What the lecturer said - transcript only

The lecturer began with Homework 4 logistics. Even the smallest language model in the assignment is large enough to make experiments slow. Students should use the course-provided H100 or A100 resources, read the infrastructure instructions carefully, and test access and code early rather than discovering problems near the deadline.

The best final implementation might take about six hours for a training run, but students were advised to budget up to twelve hours because debugging and reruns can double the apparent requirement. The lecturer asked students to complete the course survey so the staff could understand compute use and difficulties.

The exam would occur after spring break. The staff would release representative practice questions and organize review support. Two lectures and a discussion section would be devoted to review. The lecturer framed these notices as planning guidance rather than new technical material.

### Source reconciliation

The opening slide material supports the transition from course logistics to model-based RL but does not add a technical claim to this segment.

### Additional explanation

The practical warning is part of the experimental method: with multi-hour training jobs, short unit tests, tiny-data runs, checkpointing, and early infrastructure validation are necessary to distinguish an algorithmic failure from a late operational failure.

## 2. What model-based RL learns and how it is used

**Transcript coverage:** lines 443-865

### What the lecturer said - transcript only

Model-free RL does not know or explicitly learn the environment transition process. Model-based RL learns a model that can act as a simulator. Video-generation systems such as Veo 2 and Sora make this idea visually intuitive: a learned model predicts how a scene changes. Their artifacts also foreshadow why an imperfect learned simulator can be dangerous for control.

The basic recipe begins with transition data $\mathcal D$ containing states, actions, and next states, typically collected by some behavior policy $\pi_\beta$. A deterministic dynamics model predicts the next state and may be trained with squared error. A probabilistic model predicts a conditional distribution and may be trained by maximum likelihood.

After fitting the model, one can use it in two broad ways: run an otherwise model-free RL algorithm inside the learned simulator, or plan actions directly through the model. Since synthetic interaction can be much cheaper than real interaction, model-based RL may achieve greater real-world sample efficiency.

### Source reconciliation

Slides 2-5 write the dataset and objectives as

$$
\mathcal D=\{(s_i,a_i,s_i')\}_{i=1}^{N},
$$

$$
\min_\phi\sum_i\|f_\phi(s_i,a_i)-s_i'\|^2
$$

for deterministic prediction, or

$$
\max_\phi\sum_i\log p_\phi(s_i'\mid s_i,a_i)
$$

for a probabilistic model. The slides label $\pi_\beta$ as the data-collection or behavior policy.

### Additional explanation

"Model-based" refers to explicitly using a learned or known prediction model for decision making. Merely learning a representation of state is not sufficient unless predictions from that representation influence planning, policy learning, or value learning.

## 3. Failure modes, benefits, and problem categories

**Transcript coverage:** lines 866-1512

### What the lecturer said - transcript only

The lecturer divided model-based difficulties into three broad categories.

First is a statistics and data-distribution problem. A model can only be trusted in regions sufficiently covered by its data. A maze model trained from behavior confined to the upper-left portion of the maze has little basis for predicting elsewhere.

Second is a modeling problem. The appropriate model class depends strongly on the domain. The lecturer used a Sora cube-stacking video in which cubes appeared, disappeared, or changed as an example of a visually plausible model violating physical consistency. Pixel dynamics can be nonsmooth and unnecessarily difficult to predict. For control, a deliberately simpler or smoother internal model can be more useful than a maximally photorealistic one.

Third is a planning or control problem. Even with a good model, deciding which actions maximize long-horizon reward is difficult. A learned video model may itself be enormous and slower to run than a conventional simulator or even than the physical process it is meant to replace.

The potential benefits remain important. A learned model can improve sample efficiency by supporting many synthetic trials per real transition. It can also exploit external or pretraining data not collected specifically for the final policy. The lecture would therefore treat model-based RL as the interaction of statistics, deep-learning model design, and control.

### Source reconciliation

Slides 6-9 visually contrast limited maze coverage, inconsistent generated physics, and the planning loop. The Sora example is evidence of model artifacts, not a claim that every video model behaves that way.

### Additional explanation

The three categories fail differently. More data may reduce a coverage problem but not repair a structurally unsuitable model. A better model may still be unusable if planning is too expensive. A stronger planner can worsen behavior if it more effectively exploits model errors.

## Part II - Distribution shift and uncertainty

## 4. Distribution shift and restrained policy improvement

**Transcript coverage:** lines 1513-2394

### What the lecturer said - transcript only

The central failure is distribution shift caused by optimizing the model. The lecturer illustrated a landscape in which training data lies on the left. A fitted model extrapolates that moving right raises elevation or reward, so an optimizing policy moves beyond the data and may go over a real cliff. The policy is not passively affected by error; it deliberately searches for the model's most favorable errors.

An iterative procedure can repair this. Execute the current policy, add the newly visited transitions, refit the model, and improve the policy again. This resembles the distribution-correction logic of DAGGER. Eventually the model receives evidence in regions the policy wants to enter.

However, fully optimizing the policy after each model fit can produce extreme failures and oscillation. One model sends the policy far in one direction, new data corrects that region, and the next fit may send it far elsewhere. The lecturer recommended improving the policy only "a bit," for example with one PPO iteration, before collecting more real data. This creates a tension: larger updates promise more improvement, but only while the model remains accurate along the new policy's distribution.

The lecturer called a spurious optimum in the learned model a phantom peak. Two major remedies are a trust region that constrains policy change and an uncertainty-aware probabilistic model that reduces reliance on unsupported predictions.

### Source reconciliation

Slides 10-14 show the cliff and phantom-peak diagrams and express the trust-region idea as a constraint such as

$$
D_{\mathrm{KL}}(\pi_{\text{new}}\,\|\,\pi_\beta)\le \epsilon.
$$

Here the relevant reference may be the policy associated with currently covered data, not necessarily a permanently fixed policy.

### Additional explanation

The model and planner create a feedback loop:

$$
\text{model error}\rightarrow\text{policy exploitation}
\rightarrow\text{new state distribution}\rightarrow\text{larger error}.
$$

Small policy updates shorten that loop by asking the model only local counterfactual questions before acquiring corrective evidence.

## 5. Decision making under predictive uncertainty

**Transcript coverage:** lines 2395-3049

### What the lecturer said - transcript only

The lecturer used a Gaussian-process-like picture near a cliff to discuss uncertainty-aware decisions. Two actions may have the same predicted mean next state but different uncertainty. If one distribution spreads across a catastrophic region, its expected reward can be much lower even though the reward at its mean looks identical.

A controller may evaluate the expectation under its predictive distribution, use a pessimistic estimate for safety, or use an optimistic estimate for exploration. Those are different decisions, not interchangeable mathematical approximations. The expected reward of a random state is generally not the reward evaluated at the expected state:

$$
\mathbb E[r(s)]\ne r(\mathbb E[s]).
$$

In Q&A, the lecturer said that Monte Carlo samples can approximate the expectation, while analytic calculations may be possible for convenient distributions. Variational inference can approximate uncertainty, but only if the learned uncertainty actually corresponds to the unknowns relevant for decision making.

### Source reconciliation

Slides 15-17 show predictive distributions whose mass overlaps a cliff and distinguish expectation, pessimism, and optimism. The diagrams verify that the comparison concerns the full state distribution rather than only a point prediction.

### Additional explanation

For nonlinear rewards, variance changes value through curvature and tail risk. A small probability of catastrophic failure can dominate a large probability of ordinary success. This is why propagating a distribution through the reward is qualitatively different from rolling out only the mean dynamics.

## 6. Aleatoric and epistemic uncertainty

**Transcript coverage:** lines 3050-3462

### What the lecturer said - transcript only

The output variance of a probabilistic model is not automatically the uncertainty needed to prevent model exploitation. An overfit network may be confidently wrong and predict almost zero variance in a region with no data.

The lecturer separated two sources of uncertainty. Aleatoric, or statistical, uncertainty is irreducible randomness in outcomes even if the true model parameters were known. Epistemic, or model, uncertainty is uncertainty about those parameters or about which model is correct because data is limited. More relevant data can reduce epistemic uncertainty but not inherent aleatoric noise.

For safe model-based improvement, epistemic uncertainty is crucial: unsupported state-action inputs should lead to disagreement over possible dynamics. A Bayesian treatment represents a posterior $p(\theta\mid\mathcal D)$ over model parameters and integrates predictions over that posterior. A single maximum-likelihood estimate is only one point and cannot express this uncertainty by itself.

### Source reconciliation

Slides 17-19 use the labels **aleatoric/statistical uncertainty** and **epistemic/model uncertainty** and contrast a point estimate with a posterior over $\theta$:

$$
p(s'\mid s,a,\mathcal D)
=\int p(s'\mid s,a,\theta)p(\theta\mid\mathcal D)\,d\theta.
$$

### Additional explanation

A stochastic next-state head primarily models aleatoric variation conditional on one fitted parameter vector. Epistemic uncertainty requires uncertainty across plausible parameter vectors, functions, or models. The two can coexist and should not be conflated.

## 7. Bayesian neural networks, ensembles, and truncated Q&A

**Transcript coverage:** lines 3463-4120

### What the lecturer said - transcript only

A Bayesian neural network seeks an approximate posterior over its weights. One simple approximation assigns each weight an independent Gaussian with a learned mean and variance, omitting correlations between weights. Variational inference trains this distribution through an evidence-lower-bound objective. The lecturer presented this as conceptually direct but not always the most convenient practical choice.

A bootstrap ensemble is a simpler approximation. Train several separate models and treat their predictions as samples from different plausible explanations of the data. In the classical bootstrap, each member receives a dataset sampled with replacement from the original data. The ensemble then approximates the posterior as a mixture of point masses. Intuitively, agreement indicates confidence and disagreement indicates epistemic uncertainty.

Practical ensembles are often small, usually fewer than ten models. Implementations commonly skip bootstrap resampling and instead train every member on the same dataset with different random initializations and stochastic optimization. This loses the clean bootstrap interpretation, but the lecturer said it often creates enough diversity empirically to be useful.

The recording ends as an audience member begins asking, "How state of the arts is". No complete question or answer is available.

### Source reconciliation

Slides 20-21 show a fully factorized Gaussian approximation,

$$
q_\phi(\theta)=\prod_j\mathcal N(\theta_j;\mu_j,\sigma_j^2),
$$

and an ensemble approximation,

$$
p(\theta\mid\mathcal D)\approx
\frac{1}{M}\sum_{m=1}^{M}\delta(\theta-\theta_m).
$$

Slide 22 only previews the next lecture. It contains no answer to the truncated question and is not used to invent one.

### Additional explanation

Random initialization ensembles work because neural-network training is nonconvex and stochastic: different members can converge to different functions that all fit the observed data. Their disagreement is a useful heuristic, not a calibrated Bayesian posterior. If all networks share the same architecture and biases, they may agree confidently on the same wrong extrapolation.

## Consolidated takeaways

1. Model-based RL learns a dynamics model and uses it for planning or synthetic RL updates.
2. Its promise is real-world sample efficiency and use of external data; its costs include model design, planning complexity, and model-exploitation risk.
3. Policy optimization changes the state-action distribution and actively seeks favorable model errors.
4. Iterative data collection can correct errors, but large model-based policy updates can produce oscillatory, unsafe behavior.
5. Trust regions and uncertainty-aware planning limit how strongly the policy relies on unsupported predictions.
6. Decisions must account for a predictive distribution; evaluating reward at a mean state can hide nonlinear costs and tail risk.
7. Aleatoric uncertainty describes irreducible outcome noise, while epistemic uncertainty describes uncertainty over the model.
8. Bayesian neural networks and ensembles attempt to represent epistemic uncertainty; small random-initialization ensembles are a common practical approximation.

## Key equations

### Deterministic model fitting

$$
\min_\phi\sum_i\|f_\phi(s_i,a_i)-s_i'\|^2.
$$

### Probabilistic model fitting

$$
\max_\phi\sum_i\log p_\phi(s_i'\mid s_i,a_i).
$$

### Posterior predictive dynamics

$$
p(s'\mid s,a,\mathcal D)
=\int p(s'\mid s,a,\theta)p(\theta\mid\mathcal D)\,d\theta.
$$

### Reward under a predictive distribution

$$
\mathbb E_{s'\sim p(\cdot\mid s,a,\mathcal D)}[r(s')]
\ne
r\!\left(\mathbb E[s']\right)
$$

in general.

### Ensemble posterior approximation

$$
p(\theta\mid\mathcal D)\approx
\frac{1}{M}\sum_{m=1}^{M}\delta(\theta-\theta_m).
$$

## Glossary

- **Model-free RL:** RL that does not explicitly learn or use a transition model for decision making.
- **Model-based RL:** RL that learns or uses a model of environment dynamics for planning or policy/value improvement.
- **Behavior policy $\pi_\beta$:** The policy that generated a transition dataset.
- **Model exploitation:** A policy selecting actions because errors in the learned model predict unrealistically favorable outcomes.
- **Distribution shift:** A difference between the state-action distribution used to fit a model and the distribution on which it is later queried.
- **Phantom peak:** A spurious optimum created by model extrapolation error.
- **Trust region:** A constraint that limits how far a new policy moves from a reference policy.
- **Aleatoric uncertainty:** Irreducible randomness in the environment conditional on known inputs and parameters.
- **Epistemic uncertainty:** Reducible uncertainty over the model or its parameters due to limited data.
- **Bayesian neural network:** A neural model that represents a distribution over weights rather than one point estimate.
- **Bootstrap ensemble:** A collection of models trained on resampled datasets and interpreted as approximate posterior samples.
- **Random-initialization ensemble:** A practical ensemble trained from different initial weights, often on the same data.

## Self-check questions

1. What distinguishes model-based from model-free RL?
2. What are the three problem categories the lecturer associated with model-based RL?
3. Why can a better planner produce worse real-world behavior when its model is imperfect?
4. How does iterative real-data collection correct the model's distribution shift?
5. Why might fully optimizing the policy between model fits cause oscillation?
6. When does $\mathbb E[r(s)]$ differ importantly from $r(\mathbb E[s])$?
7. Why does a probabilistic output head not necessarily capture epistemic uncertainty?
8. What is the conceptual difference between a Bayesian neural network and an ensemble?
9. Why can same-data, random-initialization ensembles still disagree?
10. Which question at the end of the source cannot be answered from the supplied recording?

## Source coverage checklist

- [x] Transcript lines 1-4120 are assigned once, in monotonic and inclusive ranges.
- [x] The recording truncation at line 4120 is disclosed; no answer is fabricated.
- [x] All 22 slide pages were rendered and visually inspected.
- [x] Dataset, deterministic/probabilistic fitting, trust-region, posterior-predictive, Bayesian-network, and ensemble notation was checked against the slides.
- [x] The maze-coverage, Sora-artifact, cliff, phantom-peak, and uncertainty diagrams were visually reconciled.
- [x] Transcript variations of Bayesian, Markov, and policy notation are corrected only where the slides establish the intended term.
- [x] Lecturer claims, slide-only notation, and additional explanation are kept separate.

**Coverage result:** Complete for every supplied transcript line and all 22 slides; source incomplete because the final audience question and answer are truncated.
