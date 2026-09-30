---
title: "Lecture 16 - Model-Based RL, Part 2"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 16
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 16, Model-Based RL Algorithms.txt"
source_slides: "../lectures/Lecture 16 - Model-Based RL Part 2.pdf"
transcript_lines: 8500
slide_pages: 38
status: "complete"
---

# Lecture 16: Model-Based RL, Part 2

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Recording fragment and uncertainty recap | lines 1-229 |
| 2 | Deterministic open-loop planning | lines 230-670 |
| 3 | Open-loop versus closed-loop control and MPC | lines 671-1354 |
| 4 | Random shooting, CEM, and planning families | lines 1355-2128 |
| 5 | Planning with epistemic uncertainty | lines 2129-2958 |
| 6 | Policy learning through a model | lines 2959-3537 |
| 7 | Compounding error and short branched rollouts | lines 3538-4125 |
| 8 | Dyna-style algorithms and parallel training | lines 4126-5200 |
| 9 | Latent-state sequence models | lines 5201-5988 |
| 10 | Sequential ELBO and posterior choices | lines 5989-6820 |
| 11 | Deterministic encoders as a practical simplification | lines 6821-7242 |
| 12 | Actor-critic with learned representations | lines 7243-7986 |
| 13 | Model classes, sample-compute trade-offs, and examples | lines 7987-8500 |

## Part I - Planning with learned dynamics

## 1. Recording fragment and uncertainty recap

**Transcript coverage:** lines 1-229

### What the lecturer said - transcript only

The supplied recording opens with a short, fragmentary pre-class exchange before the formal lecture begins. The intelligible lecture content then recaps Part 1: model-based RL fits dynamics from transition data, but an improved policy queries the model outside the data distribution and may exploit prediction errors. An uncertainty-aware model should distinguish confident interpolation from unsupported extrapolation.

The lecturer recalled bootstrap ensembles as the main practical uncertainty tool. Several models trained from different initial conditions, and ideally different bootstrap samples, produce a distribution of predictions. Their disagreement is used as a rough proxy for epistemic uncertainty. This lecture would use that model for planning and policy learning.

### Source reconciliation

Slides 1-3 mark the lecture title and recap model uncertainty. The opening conversational fragment is present only in the transcript and is not converted into course content.

### Additional explanation

The ensemble represents uncertainty over possible worlds, not merely noise that changes independently at every simulated step. That distinction becomes essential when sampling long rollouts.

## 2. Deterministic open-loop planning

**Transcript coverage:** lines 230-670

### What the lecturer said - transcript only

For known deterministic dynamics, planning can be posed as optimization over a finite sequence of actions. The states are constrained by the learned transition function. Because each next state is determined by the preceding state and action, the equality constraints can be eliminated by repeatedly substituting the model into the objective.

The resulting plan commits to the entire action sequence before observing any new state. The lecturer called this open-loop planning. An audience question asked about action chunking. A chunk of several low-level actions can be treated as one higher-level action, although assigning intermediate rewards and handling what happens inside the chunk may require additional bookkeeping.

For stochastic dynamics, the same open-loop idea optimizes expected return. That expectation may be estimated with samples or, for special distributions and dynamics, calculated analytically. The main limitation is conceptual: the action sequence cannot respond to information revealed after execution begins.

### Source reconciliation

Slides 4-6 state the deterministic problem as

$$
\max_{a_{1:H}}\sum_{t=1}^{H}r(s_t,a_t)
\quad\text{subject to}\quad
s_{t+1}=f(s_t,a_t),
$$

and show the equivalent unconstrained objective obtained by rolling $f$ forward from the current state.

### Additional explanation

Open-loop optimization is an optimization over an action vector, not over a policy. Its dimension grows with action dimension times planning horizon. It is appropriate when dynamics are nearly deterministic over the planned interval or when the plan will be recomputed frequently.

## 3. Open-loop versus closed-loop control and MPC

**Transcript coverage:** lines 671-1354

### What the lecturer said - transcript only

A closed-loop controller chooses later actions after observing what actually happened. The lecturer contrasted lane-changing behavior with a plan that blindly assumes the car reached each predicted location. Feedback lets the controller correct disturbances and model errors.

He offered a quiz analogy. Committing all answers before seeing any questions is open loop; reading each question and then answering is closed loop. In stochastic environments, information observed during execution can materially change the best later action, so a fully open-loop plan is generally suboptimal.

Model predictive control (MPC) is a practical compromise. Plan a finite action sequence, execute only the first action or a short prefix, observe the new state, and plan again. Replanning turns an open-loop optimizer into a feedback controller without explicitly learning a complete closed-loop policy. The lecturer also mentioned local policy methods such as LQR, which directly produce state-feedback rules in settings where suitable local approximations are available.

### Source reconciliation

Slides 7-10 diagram open-loop plans, closed-loop policy trees, and the repeated plan-execute-replan cycle of MPC. The lane-change illustrations confirm that replanning is intended to correct both stochastic disturbances and prediction errors.

### Additional explanation

MPC's feedback frequency controls a trade-off. Replanning more often is robust but computationally costly. Executing longer chunks is cheaper but allows prediction errors to accumulate before correction.

MPC still solves an open-loop problem inside each replan; it does not generally value future information as a full policy-tree solution would. Replanning can correct disturbances but does not guarantee stability, constraint satisfaction, or global optimality without further assumptions. A short horizon may need a terminal value or terminal constraint to avoid myopic decisions.

## 4. Random shooting, CEM, and planning families

**Transcript coverage:** lines 1355-2128

### What the lecturer said - transcript only

The planning problem can be abstracted as finding an action vector $A$ that maximizes a return function $J(A)$. Random shooting uses guess and check: sample many candidate action sequences, simulate each through the model, and select the best. It is easy to implement and highly parallel, but the probability of sampling a good candidate falls rapidly with dimensionality. A narrow optimum becomes a needle in a high-dimensional haystack.

The cross-entropy method (CEM) iteratively improves a sampling distribution. Begin with a distribution over action sequences, sample candidates, evaluate them, retain an elite fraction, and refit the distribution to those elites. Repeat until the distribution concentrates. A common implementation uses a factorized Gaussian. CMA-ES is a related evolutionary strategy with richer covariance and momentum-like updates.

These optimizers are attractive because they are simple, derivative free, and compatible with arbitrary learned models. Their limits remain: the lecturer gave roughly thirty to fifty optimization dimensions as a regime beyond which simple sampling may become difficult, and the output remains an open-loop sequence.

Other planning families include Monte Carlo tree search, continuous trajectory optimization and LQR-style methods, and motion-planning trees. They make different assumptions about action spaces, smoothness, and branching.

### Source reconciliation

Slides 11-15 visually specify random shooting and the CEM loop. The Gaussian update on elites is shown schematically rather than as a single mandatory formula. The slide's planning taxonomy separates stochastic optimization, tree search, continuous optimization, and motion planning.

### Additional explanation

CEM improves sample efficiency relative to uniform random shooting by adapting where it samples. A factorized Gaussian still cannot represent multiple disjoint promising plans well; elite selection may collapse prematurely to one local mode. MPC partially mitigates imperfect long-horizon optimization because only a prefix is executed.

Planning dimension is action dimension multiplied by horizon, not action dimension alone. Respect action bounds, retain a nonzero exploration scale, and distinguish the number of candidate sequences from the number of stochastic rollouts used to score each one. The lecture's dimensional ranges are examples, not hard algorithmic limits.

## 5. Planning with epistemic uncertainty

**Transcript coverage:** lines 2129-2958

### What the lecturer said - transcript only

With an ensemble, a rollout should sample one model parameter $\theta$ for the entire simulated trajectory. Each sampled model represents one plausible world. Resampling a different model at every time step would describe a world whose physical laws change from moment to moment.

The lecturer used a door-lock analogy. If the agent is uncertain whether a door is locked, one sampled world should keep it locked throughout the imagined trajectory and another should keep it unlocked. Sampling independently at every step would allow the same door to alternate arbitrarily between locked and unlocked.

Candidate action sequences can be evaluated across sampled ensemble members and optimized for expected performance. This encourages hedging against epistemic uncertainty. It is not fully Bayes-adaptive planning: the open-loop calculation does not explicitly value actions for what their observations will teach the agent.

The lecturer described a robot-hand experiment associated with Nagabandi and colleagues. A model trained from real interaction, combined with CEM and MPC, learned to rotate balls in a dexterous hand after roughly two hours of experience. The controller replanned every couple of dozen low-level steps and did not learn a standalone policy.

### Source reconciliation

Slides 16-18 contrast sampling one ensemble member per trajectory with sampling one per time step and show the robot-hand planning loop. The videos and images are illustrative results; they do not establish a universal two-hour sample requirement.

### Additional explanation

This procedure is sometimes called trajectory sampling. It approximates

$$
\mathbb E_{\theta\sim p(\theta\mid\mathcal D)}
\left[\sum_t r(s_t,a_t)\right]
$$

while preserving temporal correlation within a sampled model. A pessimistic planner could replace the mean with a lower quantile or worst-case aggregation.

Holding model parameters fixed preserves **epistemic** uncertainty about a persistent world. Fresh transition noise should still be sampled at each step for aleatoric stochasticity. Resampling model indices per step is a different approximation to predictive mixtures, rather than a faithful draw from a posterior over fixed dynamics. The displayed expectation also includes transition noise when individual models are stochastic.

## Part II - Learning a policy from a learned model

## 6. Policy learning through a model

**Transcript coverage:** lines 2959-3537

### What the lecturer said - transcript only

Instead of planning at test time, a learned model can generate experience for training a policy. There are two ways to differentiate the resulting objective.

The likelihood-ratio or policy-gradient estimator treats simulated transitions as samples and differentiates only through action log probabilities. The pathwise estimator backpropagates through the policy and the learned dynamics along the rollout. The pathwise route uses more information and can have lower Monte Carlo variance.

Its advantages do not guarantee stability. Real or learned dynamics may be nonsmooth. Long-horizon backpropagation multiplies many Jacobians, leading to exploding, vanishing, or ill-conditioned gradients. The lecturer said likelihood-ratio policy gradients can therefore be more stable except for short, simple, or approximately linear dynamics.

A direct model-based policy-learning loop is: collect real transitions, fit the model, generate synthetic trajectories, update the policy with policy gradients, execute the new policy in the real environment, append data, and repeat.

### Source reconciliation

Slides 19-21 compare the score-function gradient with a pathwise derivative through the rollout computation graph. The transcript's phrase "model-based RL version 2.5" refers to the displayed iterative algorithm, not to a standardized field-wide version number.

### Additional explanation

For an open-loop sequence with later actions held fixed, the pathwise derivative contains products of state Jacobians. With a feedback policy $a_j=\pi_\theta(s_j)$, the appropriate closed-loop Jacobian is instead

$$
A_j=\frac{\partial f}{\partial s}(s_j,a_j)
+\frac{\partial f}{\partial a}(s_j,a_j)
\frac{\partial\pi_\theta}{\partial s}(s_j),
\qquad
\frac{d s_{t+k}}{d a_t}
=A_{t+k-1}\cdots A_{t+1}
\frac{\partial f}{\partial a}(s_t,a_t),
$$

with an identity product when $k=1$. Later Jacobians multiply on the left. This explains why long-rollout differentiation resembles backpropagation through a recurrent network. Policy-parameter gradients additionally include direct parameter effects at every action. Both score and pathwise estimators optimize the **learned** model's return; unbiased sampling in that model does not remove model bias relative to the real environment.

## 7. Compounding error and short branched rollouts

**Transcript coverage:** lines 3538-4125

### What the lecturer said - transcript only

Long synthetic rollouts suffer compounding model error. A small one-step prediction error changes the next input to the model, which can create a larger later error. The lecturer cited the familiar order-$\epsilon H^2$ accumulation under simple assumptions and compared it with compounding error in imitation learning.

Simply truncating every rollout from the initial-state distribution loses important late-task events. A better strategy is to branch short model rollouts from real states stored in a replay buffer. This covers many temporal locations while limiting how long each model-generated branch can drift.

The synthetic transitions are mixed with real transitions and consumed by an off-policy RL algorithm. The real states anchor the rollout-start distribution; the short horizon limits model error; the model still supplies extra transitions around the data. The lecturer described this as a Dyna-style or "version 3.0" recipe rather than a literal software release.

### Source reconciliation

Slides 22-24 illustrate long-rollout drift, truncated initial-state rollouts, and short branches from replay states. The diagrams establish that the branch horizon and start-state distribution are independent design choices.

### Additional explanation

If one-step total-variation error is bounded by $\epsilon$, state-distribution deviation can grow roughly linearly with time, and summing reward error across the horizon can yield quadratic $O(\epsilon H^2)$ return error. Short branches reduce the effective $H$ in that bound.

This reasoning assumes bounded rewards, the same initial distribution and policy, and a suitable uniform or visitation-weighted transition-error bound. Small training MSE alone does not establish those assumptions. Short branches trade model error for reliance on a bootstrap critic and real-state coverage; they do not eliminate long-horizon reasoning.

## 8. Dyna-style algorithms and parallel training

**Transcript coverage:** lines 4126-5200

### What the lecturer said - transcript only

An audience member asked whether the model and policy could reinforce one another's errors. The lecturer agreed that the policy tries to exploit the model, but the real-data collection step supplies corrective evidence. Mixing real and model-generated transitions also hedges against trusting the model completely.

The original Dyna-style Q-learning loop operates online. After a real transition, update the model and reward estimate, perform a normal Q update, then make $K$ additional one-step model samples from previously visited states and use them for more Q updates. Modern "Dyna-style" is broader: it may use arbitrary off-policy actor-critic algorithms and short multi-step rollouts.

Practical systems often run several processes in parallel. Real interaction fills a replay buffer. Model training reads from that buffer. A rollout worker starts from replay states and fills a model-generated buffer. Actor and critic workers train on chosen mixtures, often with current and target networks. Old model rollouts may be evicted as the model changes.

The main knobs are the number of model samples, branch horizon, real-to-model batch ratio, model-update frequency, and compute devoted to planning or training. More synthetic data may improve real-sample efficiency but increases compute and exposure to model bias. The lecturer named MBA, MVE, and MBPO as canonical examples in this design space.

### Source reconciliation

Slides 25-27 show the classical Dyna pseudocode and a modern parallel-process diagram with real and model buffers. Acronyms in the transcript are reconciled from the slide labels, including MBPO (model-based policy optimization).

### Additional explanation

Freshness matters because synthetic data is conditional on the model version that generated it. Retaining old imagined transitions indefinitely mixes incompatible simulators and can preserve errors after the model has been corrected.

## Part III - Latent state-space models

## 9. Latent-state sequence models

**Transcript coverage:** lines 5201-5988

### What the lecturer said - transcript only

High-dimensional observations, especially images, make direct dynamics prediction difficult. Partial observability adds a second problem: one frame may not contain the physical state needed to predict the future. A latent state-space model represents each time step by a compact hidden state and learns transitions in that space.

The lecturer built the model from a sequential variational-autoencoder perspective. The generative model has an initial-state prior, action-conditioned latent dynamics, and an observation decoder. Conditioned on latent states, observations are independent across time. A separate reward model may predict reward from latent state and action.

An encoder approximates the posterior over latent states. A filtering encoder uses observations up to the current time and preceding actions. It may be a recurrent network or Transformer. The latent dynamics and reward models may be ordinary multilayer perceptrons even when the encoder and decoder are large.

### Source reconciliation

Slides 28-30 give the generative factorization

$$
p(s_{1:T},o_{1:T}\mid a_{1:T-1})
=p(s_1)\prod_{t=1}^{T-1}p(s_{t+1}\mid s_t,a_t)
\prod_{t=1}^{T}p(o_t\mid s_t),
$$

and a filtering posterior of the form

$$
q(s_{1:T}\mid o_{1:T},a_{1:T-1})
=\prod_{t=1}^{T}q(s_t\mid o_{1:t},a_{1:t-1}).
$$

### Additional explanation

The latent state need not correspond to named physical quantities. It is trained to preserve information needed for observation reconstruction, future prediction, reward prediction, and ultimately control. Those objectives can pull the representation in different directions.

## 10. Sequential ELBO and posterior choices

**Transcript coverage:** lines 5989-6820

### What the lecturer said - transcript only

Training maximizes a sequential evidence lower bound (ELBO). Its expected log terms reward agreement with latent dynamics and observation reconstruction, while the posterior entropy prevents an arbitrarily concentrated variational distribution. Dynamics training requires the joint posterior over consecutive latent states because each transition term depends on $(s_t,s_{t+1})$.

The simplest filtering factorization treats latent states as conditionally independent once each history prefix is given. This makes all consecutive pairs available for dynamics fitting, but it can underestimate correlations and uncertainty. A flexible decoder can also encourage posterior collapse, in which latent states carry too little useful information.

A full smoothing posterior conditions latent states on the entire observation sequence and represents a joint distribution across time. It is more statistically expressive but more complex. A filtering posterior is more practical for online control, because it uses only past and current information. A single-step encoder depending only on $o_t$ is simpler still. The lecturer said full smoothing constructions are relatively uncommon in the practical systems he sees.

### Source reconciliation

Slides 30-32 show the ELBO schematically as

$$
\mathbb E_q\!\left[
\log p(s_1)+\sum_{t=1}^{T-1}\log p(s_{t+1}\mid s_t,a_t)
+\sum_{t=1}^{T}\log p(o_t\mid s_t)-\log q(s_{1:T}\mid o_{1:T},a_{1:T-1})
\right].
$$

The slide diagrams distinguish smoothing, filtering, and single-observation encoders. The transcript's occasional "VAE" variants are reconciled to variational autoencoder.

### Additional explanation

Smoothing is useful during offline training because future frames disambiguate an earlier hidden state. A deployed controller cannot use unseen future observations, so it ultimately needs a filtering or predictive state. Some systems train with smoothing and distill into a filter.

The product of per-time filtering factors is a variational approximation; it does not equal the exact joint posterior merely because every factor sees a history. Transition expectations must use the chosen joint distribution for consecutive latents. Generative transitions run over $t=1,\ldots,T-1$ and observation terms over $t=1,\ldots,T$.

## 11. Deterministic encoders as a practical simplification

**Transcript coverage:** lines 6821-7242

### What the lecturer said - transcript only

A very simple alternative uses a deterministic single-frame encoder,

$$
s_t=g_\psi(o_t).
$$

The lecture treats its entropy as effectively constant and motivates fitting latent dynamics and reconstructing observations, optionally with a reward-prediction loss. This is close to an ordinary autoencoder augmented with temporal prediction. For continuous point-mass encoders, the entropy argument is only a heuristic; see the correction below.

The simplification is easy to implement and can produce useful low-dimensional representations for Q-learning. It is not a faithful probabilistic treatment of a POMDP: it cannot represent uncertainty over hidden state, and one observation may be insufficient. Nevertheless, useful practical systems occupy a continuum between this deterministic baseline and fully stochastic smoothing models.

### Source reconciliation

Slide 33 shows the deterministic encoding and its reduced training objective. It includes reconstruction, transition, and optional reward terms; those losses should not be mistaken for an exact likelihood when the encoder is deterministic.

### Additional explanation

The deterministic model is best viewed as representation learning with a dynamics regularizer. Its success depends on whether the observation itself is close enough to Markov or whether the encoder architecture receives enough temporal context to infer missing state.

**Entropy correction:** a Dirac encoder in a continuous latent space does not have an ordinary finite differential entropy that can be dropped as a harmless constant. Its KL to a nonsingular continuous prior is generally infinite. Reconstruction and latent-prediction losses are a separate practical objective, not a finite exact ELBO obtained by substituting a point mass. A fixed-small-variance limit can motivate related losses but requires treating divergent constants and scales explicitly.

## 12. Actor-critic with learned representations

**Transcript coverage:** lines 7243-7986

### What the lecturer said - transcript only

The lecturer assembled a practical latent actor-critic loop. Collect observation transitions and store them in replay. Update the encoder, latent dynamics, reward model, and decoder using their prediction objectives. Encode replay observations into latent states. Train a critic with temporal-difference targets and train an actor to maximize the critic.

The learned model can be used only as a representation learner, or it can additionally generate latent rollouts that supplement actor-critic training. A stable likelihood or reconstruction objective may provide a better representation-learning signal than an unstable Q-learning loss by itself.

In Q&A, the lecturer said representation learning may be pretrained and then frozen or trained jointly with RL. Joint training adapts features to the task but couples several nonstationary objectives. A "kitchen-sink" model-based actor-critic includes the real replay path, latent model learning, synthetic branches, and actor-critic updates. Decoding imagined latent states back to pixels is unnecessary if actor and critic operate directly in latent space.

### Source reconciliation

Slides 34-36 diagram the replay-to-model and replay-to-actor-critic paths, followed by the optional imagination branch. The diagrams show a target value or critic network but leave implementation-specific loss coefficients open.

### Additional explanation

Skipping the decoder during imagined rollouts saves compute and avoids compounding pixel-level generation error. A decoder can still be valuable during representation training because reconstruction discourages the encoder from discarding observation information too early.

If the encoder changes, previously stored latent vectors can become stale; storing observations/history and re-encoding them avoids mixing incompatible representations. A latent imagined rollout also needs a reward and continuation/termination model. Stop bootstrap gradients and distinguish true terminal states from externally truncated rollouts just as in ordinary actor-critic.

## 13. Model classes, sample-compute trade-offs, and examples

**Transcript coverage:** lines 7987-8500

### What the lecturer said - transcript only

Latent dynamics may be regularized to be simple, for example linear, or represented with more expressive stochastic models including diffusion. The appropriate choice depends on the domain and the intended use of the model.

Model-based methods often improve real-environment sample efficiency while consuming more computation. A practitioner may use learned representations without extensive model rollouts when a real simulator is cheap, the learned model is inaccurate, or neural simulation is itself expensive. The literature is heterogeneous because different domains favor different compromises.

The lecturer identified SLAC as an example emphasizing stochastic latent representations and Dreamer as an example that learns behavior through latent imagination. The lecture ended normally after brief acknowledgments.

### Source reconciliation

Slides 37-38 summarize the spectrum from representation-only use to latent planning and cite SLAC and Dreamer. The diagrams and citations do not imply that either design dominates in every environment.

### Additional explanation

"Sample efficient" and "compute efficient" are separate axes. A method can reduce expensive robot interactions yet require much more GPU time for model fitting and imagined rollouts. The economically relevant choice depends on which resource is scarce.

## Consolidated takeaways

1. Open-loop planning optimizes an action sequence; closed-loop control adapts future actions to observations.
2. MPC turns an open-loop optimizer into feedback control by executing only a prefix and replanning.
3. Random shooting is simple and parallel; CEM adapts its sampling distribution but still struggles as plan dimension grows.
4. Ensemble rollouts should keep one sampled model fixed per trajectory so that each rollout represents a coherent possible world.
5. Backpropagating through dynamics can reduce estimator variance but creates long products of model Jacobians and can be unstable.
6. Long model rollouts compound error; short branches from real replay states preserve temporal coverage with a shorter synthetic horizon.
7. Dyna-style methods trade real samples for model compute by mixing genuine and imagined transitions.
8. Latent state-space models combine a transition prior, observation decoder, and approximate posterior over hidden states.
9. Smoothing posteriors use future observations, filtering posteriors are deployable online, and deterministic encoders are a useful but non-probabilistic simplification.
10. A learned model may be used for representation only or for latent imagination; the correct choice depends on sample cost, model accuracy, and compute.

## Key equations

### Finite-horizon open-loop planning

$$
a_{1:H}^*=\arg\max_{a_{1:H}}
\sum_{t=1}^{H}r(s_t,a_t)
\quad\text{s.t.}\quad s_{t+1}=f(s_t,a_t).
$$

### Planning under model uncertainty

$$
a_{1:H}^*=\arg\max_{a_{1:H}}
\mathbb E_{\theta\sim p(\theta\mid\mathcal D)}
\left[\sum_{t=1}^{H}r(s_t,a_t)\right],
$$

with one $\theta$ held fixed throughout each sampled trajectory.

### Sequential latent generative model

$$
p(s_{1:T},o_{1:T}\mid a_{1:T-1})
=p(s_1)\prod_{t=1}^{T-1}p(s_{t+1}\mid s_t,a_t)
\prod_{t=1}^{T}p(o_t\mid s_t).
$$

### Sequential ELBO

$$
\mathcal L=
\mathbb E_q\!\left[
\log p(s_1)+\sum_{t=1}^{T-1}\log p(s_{t+1}\mid s_t,a_t)
+\sum_{t=1}^{T}\log p(o_t\mid s_t)-\log q(s_{1:T}\mid o_{1:T},a_{1:T-1})
\right].
$$

### Deterministic latent encoder

$$
s_t=g_\psi(o_t),
$$

trained with reconstruction, latent-transition, and optionally reward-prediction losses.

### One-step Dyna-style value target

$$
y=r+\gamma m\max_{a'}Q_{\bar\phi}(s',a'),
$$

where $(s,a,r,s')$ may be real or generated by the learned model, $m$ masks true termination, and the full target is detached. Model-generated termination is itself an estimate and can be wrong.

## Glossary

- **Open-loop plan:** A fixed action sequence that does not condition later actions on newly observed states.
- **Closed-loop policy:** A rule that selects actions using feedback from the current state or observation history.
- **Model predictive control (MPC):** Repeatedly plan, execute a short prefix, observe, and replan.
- **Random shooting:** Selecting the best of randomly sampled candidate action sequences.
- **Cross-entropy method (CEM):** Iteratively sampling, selecting elites, and refitting a distribution over candidate solutions.
- **Trajectory sampling:** Evaluating plans by sampling coherent model instances or stochastic rollouts.
- **Pathwise gradient:** A derivative propagated through a differentiable model and policy computation graph.
- **Compounding model error:** Growth of rollout error as imperfect predictions become inputs to later predictions.
- **Dyna:** A framework that interleaves real experience, model learning, and value or policy updates from simulated experience.
- **Branched rollout:** A short model-generated trajectory initialized from a real replay state.
- **Latent state-space model:** A sequential generative model with hidden states, learned dynamics, and observation emissions.
- **Filtering posterior:** An approximate state posterior using only observations available up to the current time.
- **Smoothing posterior:** A posterior over states that may condition on the full observation sequence, including future observations.
- **Posterior collapse:** A failure in which the latent variable carries little information because the decoder can model observations without it.
- **Latent imagination:** Training a policy or value function on rollouts generated in a learned latent dynamics model.

## Self-check questions

1. Why is a stochastic open-loop plan generally weaker than a closed-loop policy?
2. How does MPC obtain feedback without learning an explicit global policy?
3. What is the dimensionality problem for random shooting, and how does CEM help?
4. Why should an ensemble member remain fixed across one simulated trajectory?
5. What information does a pathwise gradient use that a score-function gradient ignores?
6. Why can the pathwise gradient become ill-conditioned over a long horizon?
7. Why are short model rollouts from replay states preferable to short rollouts only from initial states?
8. What distinct roles do the real replay buffer and model rollout buffer serve in a Dyna-style system?
9. How do filtering and smoothing posteriors differ?
10. In what sense is a deterministic encoder not a probabilistic POMDP solution?
11. Why might reconstruction improve an actor-critic representation?
12. When might representation-only model learning be preferable to latent rollouts?

## Source coverage checklist

- [x] Transcript lines 1-8500 are assigned once, in monotonic and inclusive ranges.
- [x] The fragmentary pre-class opening is disclosed and not mistaken for lecture content.
- [x] The recording ends normally; no source truncation was detected.
- [x] All 38 slide pages were rendered and visually inspected.
- [x] Planning, MPC, ensemble, Dyna, latent-factorization, ELBO, and actor-critic equations and diagrams were checked against the slides.
- [x] Robot-hand, rollout-branching, parallel-buffer, encoder, and imagination diagrams were visually reconciled.
- [x] Transcript variants of CEM, LQR, Dyna, VAE, MBPO, SLAC, and Dreamer are reconciled only where the slides establish the intended term.
- [x] Lecturer claims, slide-only precision, and additional explanation are kept separate.

**Coverage result:** Complete for the supplied 8,500-line transcript and 38-page slide deck.
