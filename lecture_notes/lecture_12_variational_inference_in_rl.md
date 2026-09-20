---
title: "Lecture 12 - Variational Inference in Reinforcement Learning"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 12
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 12, Variational Inference in RL.txt"
source_slides: "../lectures/Lecture 12 - VI in RL.pdf"
transcript_lines: 9595
slide_pages: 37
status: "complete"
---

# Lecture 12: Variational Inference in Reinforcement Learning

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Recap: the per-data-point variational algorithm | lines 1-403 |
| 2 | Amortized variational inference | lines 404-991 |
| 3 | Score-function and reparameterized gradients | lines 992-1690 |
| 4 | The practical ELBO and VAE computation graph | lines 1691-2416 |
| 5 | Variational autoencoders as generative models | lines 2417-2916 |
| 6 | Representation learning for RL | lines 2917-3716 |
| 7 | Conditional VAEs and multimodal imitation | lines 3717-4131 |
| 8 | Diffusion and flow matching as hierarchical VAEs | lines 4132-4716 |
| 9 | State-space models and intermission | lines 4717-5104 |
| 10 | Approximately rational behavior as inference | lines 5105-6412 |
| 11 | Inference queries and backward-message recursion | lines 6413-7233 |
| 12 | Soft value iteration, optimism, and the action prior | lines 7234-8442 |
| 13 | Policy extraction, temperature, and numerical details | lines 8443-9095 |
| 14 | Forward messages, state marginals, and summary | lines 9096-9595 |

## Part I - Amortized variational inference and VAEs

## 1. Recap: the per-data-point variational algorithm

**Transcript coverage:** lines 1-403

### What the lecturer said - transcript only

The lecturer recapped the preceding lecture. For a latent model $p_\theta(x\mid z)p(z)$ and a separate approximate posterior $q_i(z)$ for every $x_i$, the evidence is lower-bounded by

$$
\mathcal L_i
=\mathbb E_{z\sim q_i}
\left[\log p_\theta(x_i\mid z)+\log p(z)\right]
+\mathcal H(q_i).
$$

For a mini-batch, one can sample $z$ from each $q_i$, update $\theta$ with the expected decoder log-likelihood, and update the parameters of $q_i$ to maximize the same bound and reduce the posterior KL. The method is a correct approximation, but it is a poor modern deep-learning algorithm because the number of variational parameters grows with the dataset.

In opening questions, the lecturer clarified that the entropy gradient must also be included and that the Gaussian entropy has a standard differentiable formula. He also corrected the suggestion that both ELBO terms must be positive: they need not be.

### Source reconciliation

Slides 2-4 review variational inference and the $|\theta|+N(|\mu_i|+|\sigma_i|)$ parameter-count problem.

### Additional explanation

The issue is not mathematical validity. It is storage, mini-batch compatibility, and the inability to infer a posterior for a new $x$ without running a new optimization problem.

## 2. Amortized variational inference

**Transcript coverage:** lines 404-991

### What the lecturer said - transcript only

Sharing one fixed $q(z)$ across all observations would still give a valid lower bound, but usually a very loose one because different data points have different posteriors. The lecturer suggested it might collapse toward something like an average or the prior.

Instead, share one **function**. An inference network with parameters $\phi$ maps $x$ to a distribution:

$$
q_\phi(z\mid x)
=\mathcal N\bigl(\mu_\phi(x),\sigma_\phi(x)\bigr).
$$

Every data point gets different posterior parameters, but all are produced by one fixed-size neural network. The decoder $p_\theta(x\mid z)$ and inference model $q_\phi(z\mid x)$ are trained jointly.

The objective does not change: maximize the same ELBO with respect to $\theta$ and now with respect to the shared $\phi$, averaged over data. The student answer “maximize the ELBO” was therefore correct; changing the parameterization of $q_i$ did not change the variational mathematics. The lecturer called the method amortized because the cost of finding posteriors is spread across all data points, which jointly train one inference procedure.

In Q&A, he defined “amortized” as spreading a cost across multiple things and said the decoder and encoder learning rates need not be equal, though there is often no special reason to make them different.

### Source reconciliation

Slide 5 shows the inference network, and slide 6 writes the joint $\theta$ and $\phi$ update loop.

### Additional explanation

Amortization adds a new possible error: even within the chosen Gaussian family, a shared network may fail to output the best Gaussian for a particular point. In exchange, inference becomes one forward pass and generalizes to unseen inputs.

## 3. Score-function and reparameterized gradients

**Transcript coverage:** lines 992-1690

### What the lecturer said - transcript only

The ELBO depends on $\phi$ through the entropy and through the distribution inside an expectation. Gaussian entropy is easy to differentiate. The expectation can be handled with the same score-function identity as policy gradient. If its non-entropy contents are denoted $r(x_i,z)$, then

$$
\nabla_\phi
\mathbb E_{z\sim q_\phi}[r(x_i,z)]
=
\mathbb E_{z\sim q_\phi}
\left[\nabla_\phi\log q_\phi(z\mid x_i)r(x_i,z)\right].
$$

This estimator handles discrete or continuous $z$, but it has high variance and may need many samples and small learning rates. A question noted that $r$ depends on $\theta$; the lecturer explained that this is harmless because the gradient under discussion is with respect to $\phi$.

For a Gaussian, a lower-variance alternative is the reparameterization trick:

$$
z=\mu_\phi(x)+\epsilon\sigma_\phi(x),
\qquad \epsilon\sim\mathcal N(0,I).
$$

The randomness is now in $\epsilon$, whose distribution does not depend on $\phi$. The objective is an expectation over $\epsilon$, and automatic differentiation can backpropagate through the deterministic map from $\phi$ to $z$ and through the decoder. The price is that the downstream quantity must be differentiable. Ordinary policy gradients cannot generally use this path because unknown environment dynamics block differentiation; the variational model is fully known.

One sample of $\epsilon$ per $x$ is normally used. More samples reduce variance, but for the same computation it is usually better to draw more distinct $x$ values in the mini-batch—such as 256 examples—and one $\epsilon$ for each. The discussion here assumed i.i.d. observations such as a dataset of images.

### Source reconciliation

Slides 6-7 contrast the score-function estimator with reparameterization and note that a single noise sample works well.

### Additional explanation

The score-function estimator observes only how changing probability changes sampled outcomes. The pathwise estimator also uses the local derivative of the outcome with respect to the sample, which usually supplies much richer information per draw.

## 4. The practical ELBO and VAE computation graph

**Transcript coverage:** lines 1691-2416

### What the lecturer said - transcript only

The ELBO can be rearranged into a reconstruction term and a prior-matching term:

$$
\mathcal L_i
=
\mathbb E_{z\sim q_\phi(z\mid x_i)}
\left[\log p_\theta(x_i\mid z)\right]
-D_{\mathrm{KL}}\bigl(q_\phi(z\mid x_i)\|p(z)\bigr).
$$

If both $q_\phi(z\mid x)$ and $p(z)$ are Gaussian, the KL has a standard closed-form expression. The expected reconstruction log-likelihood is estimated with reparameterized samples.

The computation graph is: feed $x_i$ to the encoder, produce $\mu_\phi(x_i)$ and $\sigma_\phi(x_i)$, combine them with sampled $\epsilon$ to form $z$, pass $z$ through the decoder to obtain $p_\theta(x_i\mid z)$, and attach the KL term to the encoder outputs. Differentiating this graph trains both networks. A student asked whether this was similar to a VAE; the lecturer said it was exactly a variational autoencoder.

The KL is in the direction $D_{\mathrm{KL}}(q\|p)$ and was described as mode-seeking, but the prior in the simple case is unimodal, so that label is not the main issue. Its functional role is to make the posterior contain as little information about $x$ as possible while still permitting accurate reconstruction. If $q(z\mid x)=p(z)$ for every $x$, $z$ contains no information and reconstruction fails. The tension encourages compact underlying factors rather than pixel-level memorization. If $z$ and $x$ were truly independent, there would be no need for this latent construction because $p(x\mid z)=p(x)$.

The lecturer closed the comparison by saying policy gradient handles discrete and continuous latents but is high variance; reparameterization is simple and low variance but directly applies to continuous reparameterizable variables. More advanced methods are needed for discrete latents. An audience member also flagged a slide label that should say $\nabla J$; the lecturer agreed.

### Source reconciliation

Slides 8-9 show the reconstruction-minus-KL form and the estimator comparison. The transcript explicitly acknowledges the mislabeled gradient on the comparison slide.

### Additional explanation

For a diagonal Gaussian posterior and standard-normal prior, the closed-form KL is

$$
D_{\mathrm{KL}}(q\|p)
=\frac12\sum_j
\left(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right).
$$

This formula is additional detail; the lecturer advised looking it up rather than deriving it in class.

## 5. Variational autoencoders as generative models

**Transcript coverage:** lines 2417-2916

### What the lecturer said - transcript only

A VAE contains an encoder $q_\phi(z\mid x)$ and decoder $p_\theta(x\mid z)$. The goal is a generative decoder, while the encoder is the variational tool required for tractable training. Once trained, generation does not use the encoder: sample $z\sim p(z)$, then sample or decode $x\sim p_\theta(x\mid z)$.

The lecturer showed faces changing as a latent point moved through a VAE space and wrote the training objective as the dataset average of a reparameterized reconstruction log-likelihood minus the posterior/prior KL.

He asked why test-time prior samples work when training-time decoder inputs come from approximate posteriors. The KL term pushes all per-example posteriors toward and collectively across the prior while the reconstruction term prevents them from all becoming identical. He pictured small posterior “balls” arranged through latent space so that prior samples decode to reasonable images.

A question asked whether multiple $z$ samples can help on complex datasets. The lecturer said yes and named the importance-weighted autoencoder (IWAE) as a more advanced method using multiple importance-weighted samples and a more expressive/tighter objective. He also noted that many modern diffusion systems operate in a learned VAE latent space; Stable Diffusion was given as an example.

### Source reconciliation

Slides 10-12 present the VAE, its generation rule, and the prior/posterior matching intuition.

### Additional explanation

The encoder is essential during training because it proposes latent explanations near an observed point. At generation time there is no observed $x$ to encode, so the prior supplies $z$.

## 6. Representation learning for RL

**Transcript coverage:** lines 2917-3716

### What the lecturer said - transcript only

A VAE can compress complicated observations into latent states for RL. A sample online procedure is:

1. collect transitions and add them to replay;
2. update the VAE from replay observations; and
3. update a Q-function from replay while feeding $z$ in place of the original observation or state.

The same idea can accompany actor-critic. The KL pressure removes redundant information, so—with careful training—the latent can reflect underlying factors of variation. In a chair dataset, a latent subspace may correspond to viewing angle. The lecturer said the model is not discovering concepts because they are human-readable; it is recovering the “knobs” that varied when the data were generated because those knobs provide a compressed representation.

Compared with an ordinary low-dimensional autoencoder bottleneck, a VAE more directly limits information and often produces somewhat better-disentangled factors, though it is far from perfect. Simply reducing dimension can cram or “spaghettify” information without yielding a simple geometry.

Q&A covered several qualifications:

- An Atari image is technically an observation, not necessarily a Markov state. A sequential VAE can incorporate history; “state” was used initially to avoid that complication.
- Prior trajectories can pretrain the representation before RL.
- One can pretrain and freeze the VAE or train it jointly with RL. Joint training is online but less stable; pretraining assumes data are available in advance.
- Training one representation across unrelated Atari games may waste capacity on game-specific variables. It may help across related levels that share factors and where generalization is useful.
- Encoder/decoder architecture is orthogonal to the VAE objective; convolutional networks are natural for images.
- If only the latent encoding is ultimately used, the decoder is still needed to provide a training signal unless another representation objective replaces it.

### Source reconciliation

Slide 13 shows the replay-buffer VAE/Q-learning loop and an example associated with disentangled representation learning.

### Additional explanation

An RL-relevant representation should preserve information needed for reward and dynamics, not merely reconstruction. A plain VAE may spend capacity on visually salient but control-irrelevant detail or discard small reward-critical features.

## 7. Conditional VAEs and multimodal imitation

**Transcript coverage:** lines 3717-4131

### What the lecturer said - transcript only

For conditional modeling of $y$ given $x$, use

$$
q_\phi(z\mid x,y),
\qquad
p_\theta(y\mid x,z),
$$

and optionally a conditional prior $p(z\mid x)$ rather than an unconditional $p(z)$. The encoder receives the training pair $(x,y)$; the decoder receives $x$ and sampled $z$.

For a policy, $x$ can be a state or image and $y$ an action. The latent then represents modes in the action distribution, such as whether a demonstrator goes left or right around a tree. At runtime the encoder is normally unnecessary: sample $z$ from the prior and decode the action. This provides a multimodal imitation-learning alternative to the flow-based model used in the first homework.

The lecturer cited research that learns latent plans from diverse human “play” data and another robotics example with fine-grained bimanual demonstrations, where transformer-based encoder and decoder networks formed a variational policy.

### Source reconciliation

Slides 14-16 show the conditional VAE and the latent-plan and bimanual examples.

### Additional explanation

If one latent is sampled once for an entire trajectory, it can encode a coherent plan. Resampling an independent latent at every action may switch modes and recreate the averaging/inconsistency problem at a temporal level.

## 8. Diffusion and flow matching as hierarchical VAEs

**Transcript coverage:** lines 4132-4716

### What the lecturer said - transcript only

The lecturer related conditional VAEs to diffusion and, less directly, flow matching. A conditional VAE samples one latent $z$ and decodes $a$ from $p(a\mid s,z)$. Diffusion begins with noise and performs many conditional denoising steps. He warned about opposite conventions: in the DDPM notation used on the slide, noise is $a_1$ and the clean action is $a_0$; in the course's flow-matching convention, noise is $a_0$ and the flow runs toward the data endpoint.

Diffusion can be interpreted as a highly structured hierarchical VAE. The intermediate noisy variables are the unobserved latents, the forward noising chain plays the role of a structured variational distribution or prior-side process, and the reverse denoising model is trained through a variational likelihood bound. The lecturer did not derive this more complex bound, but stated that it has the same expected-log-likelihood and KL logic.

Questions clarified that the $p$ model is what generates samples and $q$ is principally a training tool. At generation time the encoder/posterior is not needed. During training, it is needed to evaluate a lower bound and obtain gradients. Before convergence, encoder outputs need not resemble prior samples; the KL term is what gradually makes prior sampling meaningful.

### Source reconciliation

Slide 17 explicitly labels diffusion a hierarchical VAE and includes the DDPM/flow convention warning. The diagram visually labels forward noising and reverse denoising; the transcript notes that the exact flow-matching mathematics differs.

### Additional explanation

This interpretation does not mean every diffusion implementation literally trains a conventional two-network VAE. It says the diffusion objective can be derived as variational inference in a model with a chain of latent variables.

## 9. State-space models and intermission

**Transcript coverage:** lines 4717-5104

### What the lecturer said - transcript only

To model partial observability, treat an entire observation trajectory as $x$ and an entire latent-state trajectory as $z$. Use a structured prior

$$
p(z_{1:T})=p(z_1)\prod_t p_\theta(z_{t+1}\mid z_t,a_t),
$$

a factorized observation decoder

$$
p_\theta(o_{1:T}\mid z_{1:T})
=\prod_t p_\theta(o_t\mid z_t),
$$

and an inference model such as

$$
q_\phi(z_{1:T}\mid o_{1:T})
=\prod_tq_\phi(z_t\mid o_{1:t}).
$$

The learned latent transition is a dynamics model; a transformer could infer the current latent from observation history. The lecturer said this flexible VAE construction would be revisited in model-based RL.

In Q&A, he said the diffusion-as-VAE view helps establish diffusion as principled maximum-likelihood training through a likelihood lower bound. The class then took a three-minute intermission and reconvened at approximately 9:10.

### Source reconciliation

Slides 18-19 show the structured state-space model and mark the intermission.

### Additional explanation

This is often called a sequential VAE or latent state-space model. Unlike an image VAE, its prior encodes temporal coherence and action-conditioned dynamics.

## Part II - Control as probabilistic inference

## 10. Approximately rational behavior as inference

**Transcript coverage:** lines 5105-6412

### What the lecturer said - transcript only

The lecturer shifted from training policies to modeling behavior. If a rational agent maximizes utility, observing its actions and inferring its reward could explain and predict what it will do; RLHF was mentioned as a practical relative of this goal.

Perfect optimization is a poor model of people and animals. In a motor-control experiment, a trained monkey moves a joystick-controlled dot to a target. It generally reaches the target but does not always follow a perfectly straight path. The randomness is structured: large mistakes that miss the goal are rare, while harmless path variation is common. Similarly, a person walking to class may vary the route without going to a random classroom. Good behavior is most likely, while mildly suboptimal behavior has nonzero probability.

To model this, augment the MDP graphical model with binary optimality variables $\mathcal O_t$. The observation $\mathcal O_t=1$ represents the assumption that the agent is trying to be optimal. Choose

$$
p(\mathcal O_t=1\mid s_t,a_t)=\exp(r(s_t,a_t)).
$$

For this to be a Bernoulli probability, the reward is assumed nonpositive. Conditioning on optimality gives

$$
p(\tau\mid\mathcal O_{1:T}=1)
\propto p(\tau)
\exp\left(\sum_t r(s_t,a_t)\right).
$$

Physically impossible trajectories retain probability zero through $p(\tau)$; among possible trajectories, higher reward makes a trajectory exponentially more likely. Slight detours lose a little probability, while goal failure loses much more.

Questions clarified that $\mathcal O$ is neither state nor action but a modeled intent/evidence variable. It could be generalized from binary optimality to a goal that changes the reward. The lecturer cautioned that this is a probabilistic story, not a literal claim about monkey brains or an algorithm by itself; using it requires a model of trajectory physics.

The model is useful because it tolerates noise when inferring reward. A deterministic optimality assumption might invent a bizarre reward for every small detour—such as a special desire to step left at minute 37—whereas a stochastic model gives a parsimonious goal explanation. It also allows probabilistic-inference algorithms to solve planning problems and gives a reason to prefer some stochastic behavior for robustness or exploration.

### Source reconciliation

Slides 20-24 introduce control as inference, the optimality graphical model, the exponential-reward likelihood, and its motivations.

### Additional explanation

Adding a constant to every finite-horizon reward changes the unnormalized trajectory density by a common factor and therefore does not change the conditioned distribution. This is why the nonpositive-reward requirement can be met by shifting rewards.

## 11. Inference queries and backward-message recursion

**Transcript coverage:** lines 6413-7233

### What the lecturer said - transcript only

Three inference quantities were introduced:

1. the backward state-action message $\beta_t(s_t,a_t)=p(\mathcal O_{t:T}=1\mid s_t,a_t)$;
2. the policy $p(a_t\mid s_t,\mathcal O_{1:T}=1)$; and
3. the forward message $\alpha_t(s_t)=p(s_t\mid\mathcal O_{1:t-1}=1)$.

Backward messages lead to policies; forward messages help compute likely state locations.

At the terminal step,

$$
\beta_T(s_T,a_T)=p(\mathcal O_T=1\mid s_T,a_T)=\exp(r(s_T,a_T)).
$$

Introducing $s_{t+1}$, applying the Markov conditional independences, and integrating it out gives

$$
\beta_t(s_t,a_t)
=p(\mathcal O_t=1\mid s_t,a_t)
\mathbb E_{s_{t+1}\sim p(\cdot\mid s_t,a_t)}
[\beta_{t+1}(s_{t+1})],
$$

where

$$
\beta_t(s_t)
=\mathbb E_{a_t\sim p(a_t\mid s_t)}
[\beta_t(s_t,a_t)].
$$

The distribution $p(a_t\mid s_t)$ here is the **action prior**, the action distribution before conditioning on optimality. It was initially assumed uniform.

An extended audience exchange asked why past state/action variables disappear from the future-optimality term. The lecturer pointed to d-separation: conditioning on $s_{t+1}$ makes later variables independent of $s_t,a_t$. He described the derivation as repeated probability chain rules and Markov independence rather than a new RL assumption.

### Source reconciliation

Slides 25-27 list the inference tasks and show the full backward-message factorization. Slide 26 is a staged duplicate of the same setup rather than a distinct topic.

### Additional explanation

The backward message is analogous to the backward variable in a hidden Markov model. It aggregates all future evidence—here, future optimality—into a function of the current state or state-action pair.

## 12. Soft value iteration, optimism, and the action prior

**Transcript coverage:** lines 7234-8442

### What the lecturer said - transcript only

Define log messages

$$
V_t(s_t)=\log\beta_t(s_t),
\qquad
Q_t(s_t,a_t)=\log\beta_t(s_t,a_t).
$$

For a uniform action prior,

$$
V_t(s_t)=\log\int\exp(Q_t(s_t,a_t))\,da_t,
$$

which the lecturer called a “soft max” (two words), or log-sum-exp in a discrete action space. Exponentiation emphasizes large values; at a high scale the largest value dominates, so the expression approaches a hard maximum. When values are small in scale, it behaves more like an average.

The Q recursion is

$$
Q_t(s_t,a_t)
=r(s_t,a_t)
+\log\mathbb E_{s_{t+1}\sim p(\cdot\mid s_t,a_t)}
\left[\exp(V_{t+1}(s_{t+1}))\right].
$$

With deterministic dynamics, the log and exponential cancel and this becomes the ordinary reward-plus-next-value backup, with the hard action maximum replaced by a soft one. With stochastic dynamics, the log-expectation-exp also acts like a soft maximum over possible next states. It therefore favors rare high-value outcomes and makes the decision maker irrationally optimistic. The lecturer deferred the repair to the following lecture.

Students asked for the algebra to be revisited. The lecturer said the important conceptual conclusion was that ordinary probabilistic inference in the constructed graph almost reproduces value iteration; the exponential-reward likelihood was effectively reverse-engineered to obtain that relationship.

The action prior need not truly be uniform. If it is $p(a\mid s)$, then

$$
V(s)=\log\int
\exp\bigl(Q(s,a)+\log p(a\mid s)\bigr)\,da.
$$

The term $\log p(a\mid s)$ can be folded into the reward. A prior favoring small actions is equivalent to an added reward favoring small actions. Therefore one may assume a uniform prior without loss of generality if any nonuniform preference is incorporated into reward.

### Source reconciliation

Slides 28-30 show the log-space recursion, mark the stochastic-transition term “optimistic,” summarize the backward pass, and explain how a nonuniform action prior becomes an added reward term.

### Additional explanation

For a scale parameter $\alpha$, the normalized soft maximum is $\alpha\log\int\exp(Q/\alpha)da$. As $\alpha\to0$, it approaches $\max_a Q$ under standard regularity conditions.

## 13. Policy extraction, temperature, and numerical details

**Transcript coverage:** lines 8443-9095

### What the lecturer said - transcript only

The policy is obtained by Bayes' rule. Past optimality variables can be removed once $s_t$ is conditioned on. With a uniform action prior,

$$
\pi(a_t\mid s_t)
=p(a_t\mid s_t,\mathcal O_{1:T}=1)
=\frac{\beta_t(s_t,a_t)}{\beta_t(s_t)}.
$$

In log space,

$$
\pi(a_t\mid s_t)
=\exp\bigl(Q_t(s_t,a_t)-V_t(s_t)\bigr)
=\exp(A_t(s_t,a_t)).
$$

Thus the highest-advantage action is most probable and less advantageous actions become exponentially less probable. The lecturer said this was a natural probabilistic account of small mistakes.

A temperature $\alpha$ controls sharpness:

$$
\pi(a\mid s)
=\exp\left(\frac{1}{\alpha}(Q(s,a)-V(s))\right).
$$

Equal-value actions are tied randomly. The result is analogous to Boltzmann exploration, and as $\alpha\to0$ it approaches a greedy policy.

Exponentials can overflow. For stable log-sum-exp, subtract the maximum Q-value before exponentiating and add it back after taking the log. The lecturer also returned to the nonpositive-reward assumption: subtracting the largest reward makes $\exp(r)\leq1$ without changing the control solution; this is a mathematical technicality rather than something implementations must directly enforce in the later algorithms.

A question about a practical policy-gradient implementation was deferred. The lecturer said the eventual algorithm would look almost identical to regular policy gradient and retain its practical tricks.

### Source reconciliation

Slides 31-33 derive the ratio of backward messages, express the policy as exponential advantage, and summarize the temperature and Boltzmann connection.

### Additional explanation

$V(s)$ is exactly the log normalizer that makes $\exp(Q-V)$ integrate or sum to one. It is simultaneously a soft value and a partition function in log space.

## 14. Forward messages, state marginals, and summary

**Transcript coverage:** lines 9096-9595

### What the lecturer said - transcript only

The forward message is

$$
\alpha_t(s_t)=p(s_t\mid\mathcal O_{1:t-1}=1).
$$

It is computed from the beginning of the trajectory by integrating the previous state and action, applying the transition model and the policy conditioned on past optimality. The lecturer moved quickly through the algebra because time was short and referred students to the slide for the full factorization.

Combining past and future evidence gives the optimal-trajectory state marginal:

$$
p(s_t\mid\mathcal O_{1:T}=1)
\propto\alpha_t(s_t)\beta_t(s_t).
$$

The backward message is high for states from which the goal can be reached; the forward message is high for states reachable from the start under good behavior. Their intersection identifies states likely in the middle of approximately optimal trajectories.

The lecturer connected this to measured human reaching motions. Variability is small near a fixed start and goal but larger in the middle, where deviations matter less. The probabilistic model reproduces that qualitative pattern.

The final summary was: augment an MDP dynamic Bayes net with optimality variables; solve control as inference using methods analogous to HMM or extended-Kalman-filter inference; and obtain dynamic-programming equations with hard maxima replaced by soft maxima. The next lecture would combine this construction with variational inference to produce practical algorithms. The final transcript lines are brief acknowledgments rather than additional content.

### Source reconciliation

Slides 34-37 give the forward recursion, show the forward/backward “cone” intersection, cite the reaching-variability example, and summarize the lecture.

### Additional explanation

Forward-backward inference is needed when state occupancy itself matters, as in inverse reinforcement learning. Computing only a policy does not directly provide the normalized probability of every intermediate state under the optimality-conditioned trajectory model.

## Consolidated takeaways

1. Amortized inference replaces one variational parameter vector per datum with one network $q_\phi(z\mid x)$.
2. Score-function gradients are general but high variance; reparameterized gradients are low variance when a differentiable continuous path exists.
3. A VAE optimizes reconstruction likelihood minus posterior/prior KL.
4. The KL makes prior samples meaningful and encourages compressed latent factors.
5. VAEs can provide RL representations, multimodal conditional policies, or structured latent dynamics.
6. Diffusion can be interpreted as variational inference in a hierarchical chain of latent noisy variables.
7. A probabilistic control model makes trajectory probability proportional to physical plausibility times exponentiated reward.
8. Backward messages become Q- and V-like in log space; soft max replaces hard max.
9. Exact inference is irrationally optimistic under stochastic dynamics because it conditions transition outcomes on success.
10. A nonuniform action prior can be absorbed into reward.
11. The induced policy is an exponential-advantage or Boltzmann policy, with temperature controlling sharpness.
12. Forward and backward messages multiply to give optimality-conditioned state marginals.

## Key equations

### Amortized ELBO

$$
\mathcal L_i
=\mathbb E_{q_\phi(z\mid x_i)}[\log p_\theta(x_i\mid z)]
-D_{\mathrm{KL}}(q_\phi(z\mid x_i)\|p(z)).
$$

### Reparameterization

$$
z=\mu_\phi(x)+\epsilon\sigma_\phi(x),
\qquad\epsilon\sim\mathcal N(0,I).
$$

### Optimality-conditioned trajectory model

$$
p(\tau\mid\mathcal O_{1:T}=1)
\propto p(\tau)\exp\left(\sum_t r(s_t,a_t)\right).
$$

### Backward recursion

$$
\beta_t(s_t,a_t)
=e^{r(s_t,a_t)}
\mathbb E_{s_{t+1}\mid s_t,a_t}[\beta_{t+1}(s_{t+1})].
$$

### Log-space messages

$$
Q_t(s_t,a_t)=r(s_t,a_t)
+\log\mathbb E_{s_{t+1}\mid s_t,a_t}[e^{V_{t+1}(s_{t+1})}],
$$

$$
V_t(s_t)=\log\int e^{Q_t(s_t,a)}\,da.
$$

### Policy and state marginal

$$
\pi(a\mid s)=e^{Q(s,a)-V(s)},
\qquad
p(s_t\mid\mathcal O_{1:T}=1)\propto\alpha_t(s_t)\beta_t(s_t).
$$

## Glossary

- **Action prior:** action distribution before conditioning on optimality; its log can be incorporated into reward.
- **Amortized inference:** a shared inference network that predicts variational parameters for any input.
- **Backward message:** probability of future optimality given a current state or state-action pair.
- **Conditional VAE:** a VAE that models a conditional distribution such as actions given states.
- **Forward message:** probability of reaching a state conditioned on past optimality evidence.
- **Hierarchical VAE:** a variational latent model with multiple dependent latent levels; diffusion admits this interpretation.
- **Optimality variable:** modeled evidence whose likelihood is proportional to exponentiated reward.
- **Reparameterization trick:** expression of a random sample as a differentiable transform of parameter-free noise.
- **Score-function estimator:** likelihood-ratio gradient estimator, identical in form to policy gradient.
- **Soft max:** log-sum-exp/log-integral-exp aggregation, not the normalized softmax probability operation.
- **State-space model:** a sequential latent model with learned latent dynamics and observation decoding.
- **Temperature:** a scale controlling how close an exponential-advantage policy is to a greedy one.
- **Variational autoencoder:** a latent generative model trained with amortized variational inference.

## Self-check questions

1. Why is one fixed posterior for all $x_i$ valid but usually poor?
2. What quantity is amortized by $q_\phi(z\mid x)$?
3. Compare the score-function and reparameterized gradient estimators.
4. Why is one $\epsilon$ sample per item usually enough in a large mini-batch?
5. How does the VAE ELBO split into reconstruction and prior matching?
6. Why can the decoder generate from prior samples it did not directly see as training inputs?
7. What is the distinction between a low-dimensional bottleneck and a VAE information bottleneck?
8. How can a conditional VAE represent left-versus-right behavior?
9. In what sense is diffusion a hierarchical VAE?
10. What are the prior, decoder, and encoder in the state-space model?
11. Why is perfect reward maximization a poor model of monkey reaching paths?
12. What does $\mathcal O_t$ represent, and why must the initial derivation use nonpositive rewards?
13. Derive the backward recursion using $s_{t+1}$ and d-separation.
14. Why does log-expectation-exp over stochastic next states imply optimism?
15. How can a nonuniform action prior be folded into reward?
16. Why is the policy proportional to exponential advantage?
17. What numerical trick stabilizes log-sum-exp?
18. What information do forward messages add beyond the policy?

## Source coverage checklist

| Topic | Inclusive range | Coverage status |
|---:|---:|---|
| 1 | 1-403 | Accounted for |
| 2 | 404-991 | Accounted for |
| 3 | 992-1690 | Accounted for |
| 4 | 1691-2416 | Accounted for; spoken slide correction retained |
| 5 | 2417-2916 | Accounted for |
| 6 | 2917-3716 | Accounted for |
| 7 | 3717-4131 | Accounted for |
| 8 | 4132-4716 | Accounted for; notation convention reconciled |
| 9 | 4717-5104 | Accounted for |
| 10 | 5105-6412 | Accounted for |
| 11 | 6413-7233 | Accounted for |
| 12 | 7234-8442 | Accounted for |
| 13 | 8443-9095 | Accounted for |
| 14 | 9096-9595 | Accounted for |

**Coverage result:** All 9,595 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. All 37 slide pages were visually inspected, including the graphical-model factorizations, staged derivations, diagrams, and notation warnings.
