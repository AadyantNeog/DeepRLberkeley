---
title: "Lecture 3 - Behavioral Cloning, Part 2"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 3
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 3, Supervised Learning of Behaviors.txt"
source_slides: "../lectures/Lecture 03 - Behavioral Cloning Part 2.pdf"
transcript_lines: 1652
slide_pages: 32
status: "source-incomplete"
---

# Lecture 3: Behavioral Cloning, Part 2

> **Source warning:** The supplied transcript ends at line 1652 in the middle of the flow-matching policy-training recipe. The transcript-only account assigns every supplied line, but it cannot reconstruct the lecturer's missing words. Material visible on the remainder of slide 15 and slides 16-32 is isolated in a slide-only appendix.

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Recap and the case for better imitation models | lines 1-85 |
| 2 | Non-Markovian experts and history-based policies | lines 86-237 |
| 3 | History, optimality, and state sufficiency Q&A | lines 238-358 |
| 4 | History-induced shift and causal confusion | lines 359-656 |
| 5 | Can DAgger fix causal confusion? | lines 657-731 |
| 6 | Multimodal expert behavior | lines 732-793 |
| 7 | Autoregressive discretization | lines 794-1172 |
| 8 | Latent noise for continuous multimodal actions | lines 1173-1259 |
| 9 | Diffusion and flow-matching intuition | lines 1260-1381 |
| 10 | Sampling by integrating a learned vector field | lines 1382-1429 |
| 11 | Conditional flow-matching training | lines 1430-1625 |
| 12 | Flow-matching policy implementation begins | lines 1626-1652 |

## Part I - Models for imitation learning

## 1. Recap and the case for better imitation models

**Transcript coverage:** lines 1-85

### What the lecturer said - transcript only

After joking that the lecture might be slower without coffee, the lecturer framed this as the second imitation-learning lecture and a practical guide to making behavioral cloning work.

Naive behavioral cloning has no general guarantee. The tightrope construction gives a worst-case error proportional to $\epsilon H^2$, where $\epsilon$ is in-distribution error and $H$ is horizon. Small policy errors change later inputs, creating distributional shift and larger errors. DAgger can address this under the stronger assumption that policy-generated states can be labeled, but plain behavioral cloning remains attractive because it is simple and scalable.

The lecture would therefore cover mitigations that do not remove the worst-case theory but can be highly effective: better model classes, careful data collection and augmentation, and multitask ideas. A sufficiently accurate model makes $\epsilon$ so small that even an unfavorable dependence on $H$ may be tolerable. Ordinary checks for overfitting and underfitting remain essential.

### Source reconciliation

Slides 2-3 recap the same four responses to the problem: DAgger, very accurate models, smart data collection or augmentation, and multitask learning.

### Additional explanation

Improving $\epsilon$ and changing the distribution-shift mechanism are different interventions. A larger model can reduce how often drift begins; DAgger or recovery data change what happens once the learner reaches unfamiliar states.

## 2. Non-Markovian experts and history-based policies

**Transcript coverage:** lines 86-237

### What the lecturer said - transcript only

One reason a model may fail to fit an expert is that human behavior is not Markovian. In a proper Markov state, an optimal action can depend only on the current state. Humans, however, have bounded rationality and maintain internal context rather than recomputing an action from scratch at every instant. A driver may still swerve after another car has finished cutting in, even though the current image alone no longer calls for that reaction.

Experts can also act stochastically. When driving around a tree, either left or right may be equally good, and a person may choose inconsistently even though a deterministic optimal policy exists.

A standard policy $\pi_\theta(a_t\mid o_t)$ produces the same action distribution whenever the current observation repeats. To imitate history-dependent behavior, one can instead model $\pi_\theta(a_t\mid o_1,\ldots,o_t)$. For images, a shared visual encoder processes each frame and a sequence model, typically a transformer, combines the resulting tokens. A modern implementation might use a ViT per frame, concatenate tokens, and use a readout token for the action. Language models are a prominent history-based imitation system: predicting from only the current token would be a one-gram model and perform poorly.

### Source reconciliation

Slides 4-7 contrast the Markov policy $\pi_\theta(a_t\mid o_t)$ with the history policy $\pi_\theta(a_t\mid o_1,\ldots,o_t)$ and show shared visual-encoder weights feeding a transformer. The shared weights avoid a separate parameter set for every history position.

### Additional explanation

History helps when observations are partial or the demonstrator’s action depends on remembered context. Sharing the encoder across time preserves a fixed parameter count, while the sequence model handles a variable number of encoded frames.

## 3. History, optimality, and state sufficiency Q&A

**Transcript coverage:** lines 238-358

### What the lecturer said - transcript only

Asked whether a non-Markovian policy can be optimal, the lecturer distinguished observations from states. If observations do not determine the state, history may be required for optimal behavior. If the observation is the state, or is in bijection with it, at least one optimal policy exists that does not require history. Other history-using optimal policies may also exist.

The earlier $\epsilon H^2$ analysis abstracts the entire policy input into $\epsilon$, so it is unchanged by whether history was used to achieve that error. A finer analysis can penalize the larger input space: if history does not improve fit, it can worsen susceptibility to shift. If history lowers $\epsilon$, the tradeoff may still be favorable. These statements are worst-case and may not predict real data.

A valid state must satisfy the Markov property and be adequate for the reward. A constant zero technically evolves Markovianly but cannot represent a task whose objective depends on other facts. If the state is sufficient and Markov, a state-based optimal policy exists. This existence result does not guarantee that imitation learning will recover it.

### Source reconciliation

The slide deck does not add notation beyond the policy forms in slide 4. The transcript itself provides the reward-sufficiency caveat, which is important because transition sufficiency alone does not define a decision-sufficient state.

### Additional explanation

History has a bias-variance-style tradeoff. It can remove ambiguity caused by partial observability, but it also creates a much larger space of possible input sequences. The right question is whether the information gain reduces prediction error enough to offset the harder generalization problem.

## 4. History-induced shift and causal confusion

**Transcript coverage:** lines 359-656

### What the lecturer said - transcript only

More information can hurt by enabling spurious correlations and overfitting. History especially enlarges the surface for distribution shift: without history, only the current image must look familiar; with history, the entire sequence must. Even after a small mistake is corrected and the current scene returns to familiar territory, the anomalous event remains in the transformer’s context.

The lecturer called one common failure **causal confusion**. Imagine a dashboard light that turns on when the driver presses the brake. During a long braking event, the light predicts continued braking far more reliably than the road scene - for example, 49 of 50 subsequent steps may have both the light and brake active. A purely statistical learner can therefore mistake an effect of prior braking for a cause of current braking.

The real autonomous-driving analogue is vehicle pitch: braking tilts the hood downward, so a history-based policy can learn a hood-tilt detector rather than attend to a pedestrian or obstacle. Hiding that misleading cue can improve performance.

Information bottlenecks and other compression methods may discard some spurious features, and temporal subsampling can avoid highly correlated nearby frames. Additional data, input curation, and model design can help. None fully removes policy-induced distribution shift without on-policy data. The lecturer described robust use of history in imitation learning as an open problem and warned that adding history can make results worse.

The issue is not unique to history, only commonly amplified by it. History is indispensable in tasks such as language modeling, where supervised fine-tuning can be viewed as behavioral cloning over token histories. Causal-inference methods have been studied, but making them scalable is difficult.

### Source reconciliation

Slide 8 labels the brake-indicator example “causal confusion” and cites de Haan et al., *Causal Confusion in Imitation Learning*. It poses two explicit questions: whether history mitigates causal confusion and whether DAgger can mitigate it.

### Additional explanation

The spurious feature is predictive under demonstrations because the expert’s earlier action created it. When the learned policy behaves differently, that correlation can break. This is why ordinary held-out validation from the same demonstration process may fail to reveal the problem.

## 5. Can DAgger fix causal confusion?

**Transcript coverage:** lines 657-731

### What the lecturer said - transcript only

The lecturer turned the DAgger question into a possible exam-style exercise. Students debated whether human relabeling would reproduce the same braking labels. His conclusion was that DAgger fixes the problem in principle because causal confusion here is caused by distributional shift and DAgger asymptotically eliminates that shift.

Pathological cases can be slow. If almost every brake-light frame is legitimately paired with continued braking, DAgger may need enough policy data to change the correlation balance one time step at a time. In practice it often helps quickly, but its asymptotic argument does not promise sample efficiency.

### Source reconciliation

Slide 8 asks the question but does not print the answer; the answer above is transcript-only.

### Additional explanation

DAgger exposes the model to histories caused by its own decisions and obtains expert labels there. It can therefore reveal cases where the dashboard cue is present without the action it appeared to predict in expert-only data.

## 6. Multimodal expert behavior

**Transcript coverage:** lines 732-793

### What the lecturer said - transcript only

“Multimodal” here means multiple modes of a distribution, not multiple sensory modalities. In the tree example, left and right are both reasonable expert actions. A categorical policy can assign 50% probability to left, 50% to right, and zero to straight. A unimodal Gaussian policy over continuous steering instead averages the alternatives and may drive straight into the tree.

Continuous control therefore requires either discretization, which permits categorical modes, or a more expressive continuous distribution.

### Source reconciliation

Slide 9 shows the multimodal action histogram whose two side modes would be averaged by a unimodal continuous predictor.

### Additional explanation

The problem is not randomness itself; it is representing several coherent choices. A policy should sample one complete mode. Averaging actions is safe only when intermediate actions are also valid.

## 7. Autoregressive discretization

**Transcript coverage:** lines 794-1172

### What the lecturer said - transcript only

Naive discretization is excellent for a one-dimensional action, but the number of joint bins grows exponentially with dimension. Ten bins per dimension give 100 bins in two dimensions, 1,000 in three, and $10^{56}$ for a 56-dimensional humanoid action.

Producing one independent softmax per dimension avoids the exponential output but loses dependence between components. In driving, “steer right and slow down” and “steer left and speed up” may be coherent modes; independently sampling steering and speed can combine mismatched halves.

Autoregressive discretization solves this by sampling dimensions sequentially. For a three-dimensional action, first sample $a_{t,0}$ from the observation, then sample $a_{t,1}$ conditioned on the observation and $a_{t,0}$, and finally sample $a_{t,2}$ conditioned on everything before it. The probability chain rule represents the full joint distribution while each output has only the per-dimension number of bins. A transformer can implement the decoding sequence.

At inference, the sampled value - not merely its probability vector - must be fed into the next step, because later components need to know which earlier choice was realized. The order is a modeling choice and does not change which joint distributions can be represented by a sufficiently capable, well-trained model.

In Q&A, the lecturer added that long action vectors are possible if their sequence fits in memory, though autoregressive decoding adds latency and is unattractive for a 1 kHz quadcopter controller. There is no standard guarantee that axes align with principal components; improving the ordering or coordinate system is an active research question. The method turns $K^d$ joint bins into roughly $Kd$ categorical outputs while retaining dependencies through conditioning. He deferred a question related to VAEs until later in the course.

### Source reconciliation

Slides 10-11 show

$$
p(a_t\mid s_t)=
p(a_{t,2}\mid s_t,a_{t,0},a_{t,1})
p(a_{t,1}\mid s_t,a_{t,0})
p(a_{t,0}\mid s_t).
$$

The displayed action is $(0.1,1.2,-0.3)^\top$, used only to illustrate component ordering.

### Additional explanation

Autoregressive factorization moves complexity from output width to sampling time. It represents cross-dimensional correlation exactly in principle, but later dimensions cannot be produced in parallel.

## 8. Latent noise for continuous multimodal actions

**Transcript coverage:** lines 1173-1259

### What the lecturer said - transcript only

To retain continuous actions, one can give a single-mode output model an extra random input that selects which mode to produce. In the tree example, one noise value could select left and another right. A continuous latent vector can select among more complex action modes.

The central challenge is forcing the model to use noise meaningfully. Simply concatenating random noise to a Gaussian predictor gives it no reason to associate different noise samples with different actions. Training must create that association. The lecturer named variational autoencoders, normalizing flows, diffusion, and flow matching as methods with this general flavor, then chose flow matching because students would implement it in Homework 1.

### Source reconciliation

Slide 12 depicts a base noise sample $\xi\sim\mathcal N(0,I)$ feeding the model so different noise values can produce different modes.

### Additional explanation

This transforms a deterministic function of observation and noise into a conditional sampler:

$$
\xi\sim p(\xi),\qquad a=f_\theta(o,\xi).
$$

Expressivity comes from partitioning noise space into regions that map to distinct action modes.

## 9. Diffusion and flow-matching intuition

**Transcript coverage:** lines 1260-1381

### What the lecturer said - transcript only

Flow matching is a modern relative of diffusion. Diffusion models are widely used for AI-generated images, video, and speech - high-dimensional continuous data, unlike discrete text generation.

It is easy to corrupt a dog image into noise by repeatedly adding noise but hard to generate the dog directly from noise. Diffusion learns the reverse process from many synthetically corrupted examples. Flow matching uses essentially the same intuition with a simpler implementation, not a claim of universally better performance.

For a one-dimensional illustration, begin with a simple unimodal Gaussian and learn a vector field that transports probability into a complicated multimodal data distribution. Different regions of the base noise flow to different modes. The lecturer emphasized that the transport “time” is internal generation time, not physical environment time; the entire generative process is run at every policy step.

Asked why this would not always produce an average dog, he explained that different initial noise produces different outcomes.

### Source reconciliation

Slide 13 expresses the transport as

$$
x_1=x_0+\int_0^1 v(x_t,t)\,dt,
\qquad x_0\sim p_0,\quad x_1\sim p_{\mathrm{data}}.
$$

### Additional explanation

The vector field is analogous to a velocity assigned to every point and internal time. Integrating its ordinary differential equation moves a random base sample into a data sample while preserving stochastic diversity across different starts.

## 10. Sampling by integrating a learned vector field

**Transcript coverage:** lines 1382-1429

### What the lecturer said - transcript only

To sample, draw $x_0$ from a convenient zero-mean Gaussian and repeatedly update it using the learned vector field. With a step size of 0.1, the lecturer described ten steps from internal time 0 to 1. Each update is the previous point plus velocity times the step size, which is forward Euler integration. The endpoint is returned as the generated sample.

### Source reconciliation

Slide 14 gives

$$
x_{t+\Delta t}\leftarrow x_t+v(x_t,t)\Delta t,
\qquad t\in\{0,\Delta t,2\Delta t,\ldots,1-\Delta t\}.
$$

### Additional explanation

Smaller $\Delta t$ generally improves numerical accuracy but requires more neural-network evaluations. Sampling quality therefore trades off against control-loop latency.

## 11. Conditional flow-matching training

**Transcript coverage:** lines 1430-1625

### What the lecturer said - transcript only

Training lacks a predetermined pairing between base noise and data samples. The flow-matching trick pairs them randomly. For each example, sample a noise point $x_0$, a data point $x_1$, and an internal time $t$ uniformly between zero and one. Linearly interpolate to

$$
x_t=tx_1+(1-t)x_0
$$

and supervise the vector field at $(x_t,t)$ toward velocity $x_1-x_0$.

Random pairings appear contradictory: the same region can receive targets toward different data modes. The lecturer’s intuition was that ambiguous central regions average their targets, while nearer the endpoints the paths separate. At convergence, integrating the learned field produces curved rather than the straight training paths. He said this result can be proven, though the proof is too complex for the lecture.

Random standard initialization, such as Xavier initialization with small output biases, is adequate; the initialization does not encode the pairing. Noise and target require the same dimensionality because integration repeatedly adds a velocity vector to the evolving sample. A student observed that intrinsic data variation may be lower-dimensional than pixels. The lecturer agreed that diffusion and flow matching are not especially good representation learners - the noise coordinates need not match semantic factors - but they are strong generators.

### Source reconciliation

Slide 14 writes the squared flow-matching target as

$$
\left\|v(x_t,t)-(x_1-x_0)\right\|^2.
$$

The slide samples $t\sim p(t)$, with $p(t)=\mathcal U(0,1)$ as the example.

### Additional explanation

Each conditional straight-line flow is easy to supervise. Averaging these conditional velocity targets yields a single marginal velocity field whose density evolution matches the desired data path; this is the mathematical result behind the lecturer’s curved-path intuition.

## 12. Flow-matching policy implementation begins

**Transcript coverage:** lines 1626-1652

### What the lecturer said - transcript only

For a policy, the velocity network is conditioned on three inputs: the environment observation, the internal flow time $\tau$, and the current noisy/intermediate action from the previous integration step. Homework 1 would train this conditional velocity field. The lecturer began the minibatch recipe by saying that each element samples an observation and ground-truth action from the dataset, then begins to say “you sample” when the supplied transcript cuts off.

### Source reconciliation

Slide 15 completes the input notation as $v_\theta(o_t,a_{t,\tau},\tau)$. The remaining displayed training recipe is recorded below as slide-only material, not as reconstructed speech.

### Additional explanation

The environment time $t$ and flow time $\tau$ must remain distinct. At one environment decision, the policy performs many internal $\tau$ integration steps to transform action noise into a sampled action.

## Slide-only appendix: material after the transcript truncation

The following content comes from the visually inspected deck and is **not** attributed to the lecturer’s missing speech.

### A. Completion of flow-matching policy training

Slide 15 completes the minibatch procedure. For each batch element $j$:

$$
(o_t^{(j)},a_t^{(j)})\sim\mathcal D,\qquad
a_{t,0}^{(j)}\sim\mathcal N(0,I),\qquad
\tau^{(j)}\sim p(\tau),
$$

$$
a_{t,\tau}^{(j)}=\tau^{(j)}a_t^{(j)}+\left(1-\tau^{(j)}\right)a_{t,0}^{(j)},
$$

$$
\mathcal L=\sum_{j=1}^{B}
\left\|v_\theta\!\left(o_t^{(j)},a_{t,\tau}^{(j)},\tau^{(j)}\right)
-\left(a_t^{(j)}-a_{t,0}^{(j)}\right)\right\|^2.
$$

The parameters are updated by gradient descent on this loss.

### B. Action chunking and diffusion-policy case study

Slides 16-17 contrast one-step actions with sampling a block $a_{t:t+K}$ conditioned on $o_t$, executing the block, then observing again at $t+K+1$. The deck calls chunking a small detail that helps substantially and cites Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (2023/2024 labeling across the deck). The examples show manipulation tasks including pushing a T-shaped object, moving objects, and food handling.

Slide 18 shows a larger system labeled $\pi_0$ that combines visual-language pretraining, language and joint-angle inputs, a roughly 50-step action chunk, and flow matching across multiple robot embodiments.

### C. Narrow versus broad data

Slides 19-20 recommend intentionally adding mistakes and corrections when their corrective value exceeds the harm of the imperfect actions, and using augmentation to create corrective examples such as side-facing-camera views.

Slides 21-23 frame pretraining as a way to combine broad but behaviorally imperfect data with narrow, high-quality task demonstrations. Broad data exposes many situations; post-training data specifies desired actions. The $\pi_0$ example contrasts about 10,000 hours of diverse pretraining data with roughly 20 hours each of narrower laundry-folding and box-building data. The slide warns that narrow data alone can leave a robot confused after its own mistake.

### D. Multitask and goal-conditioned behavioral cloning

Slides 24-26 ask whether learning many tasks can become easier. A goal-conditioned policy $\pi_\theta(a\mid s,p)$ can reach any specified point $p$, rather than training one $\pi_\theta(a\mid s)$ for one destination.

For each successful trajectory ending at $s_T^i$, slide 26 relabels the final state as goal $g=s_T^i$ and maximizes

$$
\log\pi_\theta(a_t^i\mid s_t^i,g=s_T^i).
$$

The slide flags distribution shift both in ordinary policy visitation and in the distribution of commanded goals.

### E. Learning from play and hindsight-style relabeling

Slides 27-29 summarize *Learning Latent Plans from Play*: collect broad play interactions, train a goal-conditioned model with a latent plan, and execute it to reach specified goal observations.

Slide 30 describes iterated supervised goal reaching: begin with a random policy, collect data under random goals, treat achieved outcomes as demonstrations for the goals actually reached, improve the policy, and repeat. Slide 31 shows large-scale goal-conditioned navigation across several robot platforms and cites Shah et al., *GNM: A General Navigation Model to Drive Any Robot* (2022), with 60 hours in the shown aggregate table. Slide 32 connects the same hindsight principle to Hindsight Experience Replay, noting that its RL interpretation will make more sense after off-policy value-based methods.

## Consolidated takeaways

1. Better model fit reduces the $\epsilon$ factor in behavioral cloning’s worst-case error.
2. Human experts can be non-Markovian because they use memory and have bounded rationality.
3. History can resolve partial observability but enlarges the space in which distribution shift can occur.
4. Causal confusion arises when a policy relies on an effect of an earlier action as if it caused the current expert action.
5. DAgger fixes causal-confusion shift in principle, though pathological correlations may require much data.
6. Multimodal expert behavior demands a policy that samples one coherent mode rather than averaging modes.
7. Autoregressive discretization represents a joint high-dimensional action distribution with linear-size per-dimension outputs.
8. Continuous generative policies use random input to select among action modes.
9. Flow matching learns a vector field that transports simple noise into a complex data distribution.
10. Environment time and internal generative time are distinct.
11. The transcript truncates during the policy-training recipe; later techniques are therefore documented only as slide material.

## Key equations

### History-based policy

$$
\pi_\theta(a_t\mid o_{1:t}).
$$

### Autoregressive action factorization

$$
p(a_t\mid s_t)=\prod_{k=0}^{d-1}
p(a_{t,k}\mid s_t,a_{t,0:k-1}).
$$

### Continuous latent-variable sampler

$$
\xi\sim\mathcal N(0,I),\qquad a_t=f_\theta(o_t,\xi).
$$

### Flow sampling

$$
x_0\sim\mathcal N(0,I),\qquad
x_{\tau+\Delta\tau}=x_\tau+v_\theta(x_\tau,\tau)\Delta\tau.
$$

### Flow-matching interpolation and target

$$
x_\tau=\tau x_1+(1-\tau)x_0,\qquad
v^*(x_\tau,\tau)=x_1-x_0.
$$

### Observation-conditioned policy loss

$$
\mathcal L(\theta)=
\mathbb E\left[
\left\|v_\theta(o_t,a_{t,\tau},\tau)-(a_t-a_{t,0})\right\|^2
\right].
$$

## Glossary

- **Action chunking:** predicting and executing a block of consecutive actions before observing again.
- **Autoregressive discretization:** sequentially sampling discretized action components conditioned on earlier sampled components.
- **Bounded rationality:** the limitation that an expert cannot recompute a globally optimal decision from all current information at every instant.
- **Causal confusion:** reliance on a correlated effect of prior behavior as though it were a cause of the desired current action.
- **Flow matching:** learning a time-dependent vector field that transports a base distribution into a data distribution.
- **Flow time:** the internal integration coordinate used during generation, written $\tau$ here to distinguish it from environment time.
- **Goal-conditioned policy:** a policy whose action distribution is conditioned on a desired outcome or goal as well as current state.
- **History-based policy:** a policy conditioned on a sequence of prior observations rather than only the current one.
- **Mode:** a distinct high-density region of a distribution; in behavior, a coherent alternative such as going left or right.
- **Multimodal behavior:** behavior whose action distribution has multiple valid modes.
- **Non-Markovian behavior:** behavior whose decision depends on history beyond the current state or observation.
- **Pretraining:** learning broad representations or capabilities from a large, diverse dataset before task-specific post-training.
- **Spurious correlation:** a predictive association that does not remain valid under the intended causal or deployment conditions.
- **Vector field:** a function assigning a velocity to each sample location and internal time.

## Self-check questions

1. Why does reducing in-distribution error help even without changing the $H^2$ worst-case dependence?
2. Give two reasons human demonstrations may be non-Markovian.
3. When is history necessary for an optimal policy, and when is a memoryless optimal policy guaranteed to exist?
4. Why can history worsen distribution shift after the current observation has recovered?
5. Explain the brake-light or hood-tilt causal-confusion example.
6. Why does DAgger fix that problem asymptotically but potentially slowly?
7. Why does a unimodal Gaussian average incompatible expert actions?
8. Why are independent per-dimension softmaxes insufficient for correlated actions?
9. Derive the autoregressive factorization for a three-dimensional action.
10. What must be passed from one autoregressive decoding step to the next: probabilities or a sampled value, and why?
11. What role does random noise play in a continuous multimodal policy?
12. Distinguish environment time from flow time.
13. Describe forward-Euler sampling from a learned flow.
14. How are $x_0$, $x_1$, and $x_\tau$ sampled or constructed during training?
15. Why do contradictory straight-line supervision targets yield curved generated paths?
16. What portion of this lecture is missing from the transcript, and how is it separated in these notes?

## Source coverage checklist

| Topic | Inclusive range | Coverage status |
|---:|---:|---|
| 1 | 1-85 | Accounted for |
| 2 | 86-237 | Accounted for |
| 3 | 238-358 | Accounted for |
| 4 | 359-656 | Accounted for |
| 5 | 657-731 | Accounted for |
| 6 | 732-793 | Accounted for |
| 7 | 794-1172 | Accounted for |
| 8 | 1173-1259 | Accounted for |
| 9 | 1260-1381 | Accounted for |
| 10 | 1382-1429 | Accounted for |
| 11 | 1430-1625 | Accounted for |
| 12 | 1626-1652 | Accounted for; source ends mid-sentence |

**Coverage result:** All 1,652 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. All 32 slide pages were visually inspected. The transcript stops during slide 15’s implementation recipe; the remainder of that recipe and slides 16-32 are isolated in the slide-only appendix.
