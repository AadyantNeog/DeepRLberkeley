---
title: "Lecture 23 - Exploration and Skill Learning"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 23
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 23, Exploration and Skill Learning.txt"
source_slides: "../lectures/Lecture 23 - Advanced Exploration.pdf"
transcript_lines: 4024
slide_pages: 18
status: "complete"
---

# Lecture 23: Exploration and Skill Learning

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Advanced-lecture logistics and the exploration problem | lines 1-223 |
| 2 | Unsupervised behavior discovery | lines 224-460 |
| 3 | Entropy and mutual information | lines 461-976 |
| 4 | State marginals and empowerment | lines 977-1474 |
| 5 | From action empowerment to latent skills | lines 1475-1744 |
| 6 | Diversity Is All You Need-style skill learning | lines 1745-2320 |
| 7 | Mutual-information objective and geometric failure modes | lines 2321-2812 |
| 8 | Goal-conditioned empowerment | lines 2813-3292 |
| 9 | Fixed goal rewards and Skew-Fit | lines 3293-3772 |
| 10 | Reweighting, frontier goals, and closing | lines 3773-4024 |

## 1. Advanced-lecture logistics and the exploration problem

**Transcript coverage:** lines 1-223

### What the lecturer said - transcript only

Three instructor-led enrichment lectures remained, followed by two guest lectures: Kevin Frans on language models and Sihong on offline-to-online RL. These advanced topics were intended to inspire final projects and expose students to unusual, experimental ideas rather than only the most central deep-RL machinery. This lecture and Friday's had GradeScope quizzes; the following broader-perspectives lecture did not. If quizzes were not yet visible, the TA likely only needed to publish them.

The technical topic was exploration and skill learning. Ordinary DQN can solve Atari Breakout relatively easily because rewards are immediate and aligned with useful progress. Montezuma's Revenge is much harder for an RL agent despite not necessarily being harder for a person, because its rewards are delayed and provide weaker guidance toward the required sequence of behaviors.

The classical exploration response is to reward rarely visited states. If an agent continually seeks novelty, it may eventually visit every relevant state and discover the task reward. This lecture developed a different view: use reward-free interaction to learn a repertoire of behaviors that can later accelerate goal-directed learning.

### Source reconciliation

Slides 1-2 show the title and the Breakout/Montezuma contrast. The transcript adds schedule and quiz logistics as well as the reward-timing explanation.

### Additional explanation

Novelty bonuses ask “where have I not been?” Skill learning asks “what distinct, controllable behaviors can I acquire before I know the downstream task?” The second objective can serve the first, but it also supports transfer and hierarchy.

## 2. Unsupervised behavior discovery

**Transcript coverage:** lines 224-460

### What the lecturer said - transcript only

An agent may have abundant unsupervised interaction but little task-directed experience. A child plays without explicitly optimizing for a distant career goal, yet that play builds capabilities. Analogously, an artificial agent might:

- learn skills without task supervision and later use them for goals;
- learn reusable subskills for hierarchical RL;
- explore the space of possible behaviors as a subroutine for ordinary reward discovery.

The lecturer imagined a robot exploring a room before anyone specifies that today's task is washing dishes. If its unsupervised phase discovered relevant manipulation behaviors, it could solve the later goal rapidly.

The formal challenge is therefore to remove the reward function from an MDP and still induce potentially useful behavior. Information theory provides objectives for doing this.

### Source reconciliation

Slides 3-4 show child play and the robot-before/after-goal scenario. The transcript explains how unsupervised play can support later control.

### Additional explanation

This is a form of pretraining for control. Instead of pretraining representations only, the agent pretrains a conditional policy whose modes correspond to different behaviors.

## 3. Entropy and mutual information

**Transcript coverage:** lines 461-976

### What the lecturer said - transcript only

For a distribution $p(x)$, entropy is

$$
\mathcal H(X)
=-\mathbb E_{x\sim p(x)}[\log p(x)].
$$

It measures how broad the distribution is: high entropy assigns nonzero mass across many values, whereas low entropy concentrates on few values. For a simple Gaussian, width is a useful visual analogy.

Mutual information measures dependence between two jointly distributed variables:

$$
I(X;Y)
=D_{\mathrm{KL}}(p(x,y)\Vert p(x)p(y))
=\mathbb E_{p(x,y)}
\left[\log\frac{p(x,y)}{p(x)p(y)}\right].
$$

If $Y$ becomes predictable from $X$, mutual information is high. If they are independent, $p(x,y)=p(x)p(y)$ and mutual information is zero. It resembles correlation but captures nonlinear dependence; variables can have high mutual information even when their linear correlation is small. Independence is binary, whereas mutual information quantifies the degree of dependence.

A particularly useful identity is

$$
I(X;Y)
=\mathcal H(Y)-\mathcal H(Y\mid X)
=\mathcal H(X)-\mathcal H(X\mid Y).
$$

It asks how much knowing one variable reduces uncertainty about the other. If knowing $X$ completely determines $Y$, the conditional entropy vanishes; if it changes nothing, the two entropy terms cancel.

### Source reconciliation

Slides 5-7 provide the entropy, KL, mutual-information, and conditional-entropy identities. The transcript supplies the sample-cloud interpretation and the contrast with correlation.

### Additional explanation

Mutual information requires two complementary properties: the predicted variable must vary globally, and it must be predictable conditionally. Either term alone is insufficient for diverse, identifiable behavior.

## 4. State marginals and empowerment

**Transcript coverage:** lines 977-1474

### What the lecturer said - transcript only

Exploration in most RL problems is primarily about states because state spaces are often complex while action spaces are comparatively small. A power-grid controller could be an exception with a very large action space. More generally one can study state-action visitation, but the lecture focused on state marginals.

The state marginal $p_\pi(s)$ is the frequency of states obtained by running policy $\pi$ for many trajectories and ignoring their exact time positions. A replay buffer gives an empirical picture of this distribution. In a time-homogeneous, infinite-horizon Markov environment, the time index is unnecessary. A finite-horizon process can be made time homogeneous by appending time to the state. Multiple trajectories matter; a state need not recur within one trajectory to have a marginal frequency.

High state entropy

$$
\mathcal H(p_\pi(s))
$$

indicates broad coverage. The maximum-entropy distribution over a finite unconstrained set is uniform, although MDP dynamics and fixed initial states often make uniform visitation impossible.

Empowerment is an information-theoretic notion of control authority. In its one-step form,

$$
I(A_t;S_{t+1})
=\mathcal H(S_{t+1})-\mathcal H(S_{t+1}\mid A_t).
$$

It is low if actions barely affect the next state. It is also limited if only a few next states are available even when actions select them reliably. A hallway offers forward/backward choices; an intersection may offer four reliably controllable directions and therefore greater empowerment. High empowerment requires both many possible outcomes and strong action-dependent control over which outcome occurs.

Entropy remains relative even in an already complex environment. Random background pixels may create many states, but entering a different game room can broaden the distribution further. Information-theoretic quantities measure additional bits of variation rather than merely asking whether the state set is “large.”

### Source reconciliation

Slide 8 defines the state marginal, its entropy, and action-next-state empowerment. The transcript provides the replay-buffer, time-index, hallway/intersection, and stochastic-control interpretations.

### Additional explanation

Empowerment is not task reward. It prefers states with options and controllability, which may be a useful generic prior but may also favor behavior irrelevant to a particular downstream goal.

## 5. From action empowerment to latent skills

**Transcript coverage:** lines 1475-1744

### What the lecturer said - transcript only

To turn empowerment into a skill-discovery algorithm, replace the one-step action with a temporally extended skill identifier. Use one policy

$$
\pi(a\mid s,z)
$$

conditioned on a categorical variable $z$. If ten skills are desired, $z$ can simply be an integer from zero through nine, not a continuous VAE-style latent vector.

An intuitive repertoire in a two-dimensional environment sends different skill values toward different regions. Duplicate skills are less useful than distinct ones, and a collection that only explores subtle variants on the right side is less useful than one covering both left and right. The intended properties are:

- different skills should visit distinguishable regions;
- across skills, the repertoire should cover the achievable state space.

### Source reconciliation

Slides 9-10 introduce skill-conditioned policies and state-region diversity. The transcript makes the categorical nature of $z$ explicit and motivates global coverage.

### Additional explanation

The skill is temporally extended because one $z$ is normally held fixed for an episode or segment while the low-level policy produces many primitive actions.

## 6. Diversity Is All You Need-style skill learning

**Transcript coverage:** lines 1745-2320

### What the lecturer said - transcript only

Run multitask RL over skill values, but invent each skill's reward instead of supplying a task reward. Train a discriminator $q_\phi(z\mid s)$ to predict which skill produced a visited state and give the policy intrinsic reward

$$
r(s,z)=\log q_\phi(z\mid s).
$$

For each rollout, sample $z$ uniformly, run $\pi(a\mid s,z)$, and add the resulting $(s,z)$ pairs to the discriminator's classification data. With ten skills, this is a ten-way classifier. It normally uses only the state, although actions can be appended when appropriate.

If two skills reach the same state equally often, neither can receive posterior probability above one half there. They can both increase reward by separating so the classifier identifies each with probability near one. The discriminator and policy are therefore cooperative rather than adversarial: both benefit when skill-conditioned state distributions become easier to distinguish.

With two initially overlapping trajectory clouds, the classifier draws a weak separating boundary; the reward pushes the policies farther apart, enabling a cleaner boundary, and the cycle repeats. With many skills this can spread behavior across the environment.

Examples included HalfCheetah skills that run forward, backward, or perform a flip and sit; an ant moving in different directions or patterns; and MountainCar skills rolling in different directions, including one that reaches the conventional task goal. Some learned skills are uninteresting, such as holding an unusual pose. The intrinsic reward encodes diversity, not downstream completion.

In complex domains, maximizing diversity over raw state may produce destructive or useless distinctions. One can restrict the classifier input to selected state components or a learned summary representation so mutual information is computed over a task-relevant quantity.

State overlap is unavoidable when all trajectories share initial conditions. Perfect classification is not required: a skill still benefits if it makes a state moderately more characteristic of itself than of its peers.

### Source reconciliation

Slides 10-12 illustrate the discriminator loop and example behaviors. The method is identified with Eysenbach et al., *Diversity Is All You Need*.

### Additional explanation

The intrinsic reward is nonstationary because the discriminator learns while the policy learns. Unlike GAN training, their objectives align, but the coupled optimization can still be unstable.

## 7. Mutual-information objective and geometric failure modes

**Transcript coverage:** lines 2321-2812

### What the lecturer said - transcript only

The procedure maximizes skill-state mutual information:

$$
I(Z;S)=\mathcal H(Z)-\mathcal H(Z\mid S).
$$

Sampling skills uniformly maximizes $\mathcal H(Z)$ because the uniform categorical distribution has maximum entropy. Training $q_\phi(z\mid s)$ and rewarding the policy for predictable skill identity reduces $\mathcal H(Z\mid S)$. Thus the algorithm is empowerment between skills and states rather than between primitive actions and next states.

Empowerment-based skill discovery is a substantial research area. More recent work can use representation learning and metric-aware distribution objectives so that skills are rewarded for geometric separation rather than only exact non-overlap.

A fundamental failure case occurs in a nearly deterministic room. Two skills can move only a small distance in opposite directions, become perfectly non-overlapping, and receive the same classification reward they would get after reaching opposite ends of the room. The discriminator is evaluated only on visited states and need not care about unexplored regions. Noise may continue to favor wider separation in principle, but the practical signal can disappear once classification is perfect.

Regularizing the discriminator or using an explicit metric can keep spatial distance relevant. The lecturer illustrated that small logistic-regression weights create a gradual decision surface, encouraging larger separation instead of an arbitrarily sharp boundary between nearby clusters.

Continuous skill variables can be used as well, provided the method can model their distribution. The discrete case was chosen for simplicity.

### Source reconciliation

Slide 13 writes the exact $I(Z;S)$ decomposition. The transcript contributes the deterministic-room counterexample and the regularized-classifier intuition.

### Additional explanation

Mutual information is invariant to how far apart perfectly distinguishable clusters are. Metric-aware objectives add information that pure classification discards.

## 8. Goal-conditioned empowerment

**Transcript coverage:** lines 2813-3292

### What the lecturer said - transcript only

Latent skill IDs have no inherent meaning. A grounded alternative lets the skill variable itself be a goal state $g$ and learns

$$
\pi(a\mid s,g).
$$

This resembles goal-conditioned behavioral cloning, but the policy is now trained with RL. An empowerment-analogous reward is

$$
r(s,g)=\log q(g\mid s),
$$

leading to

$$
I(G;S)=\mathcal H(G)-\mathcal H(G\mid S).
$$

The classifier tries to infer which goal was commanded from the reached state. The easiest way for the policy to make the goal predictable is to reach it quickly and reliably.

The difficult term is now $\mathcal H(G)$. Latent IDs can be sampled uniformly by construction, but goals must be valid states. In a game, random pixel arrays are almost never reachable frames. The learner therefore needs a generative model $p(g)$ that covers achievable states broadly. This turns part of goal discovery into a generative-modeling problem and leaves room for domain-specific design.

In practice, goal-conditioned RL often replaces the learned discriminator reward with a fixed success signal such as

$$
r(s,g)=\mathbf 1[s=g]
$$

or a tolerance-based version $\mathbf 1[\lVert s-g\rVert\le\epsilon]$.

### Source reconciliation

Slides 14-16 replace $z$ with $g$, write the mutual-information objective, and show equality/tolerance rewards.

### Additional explanation

The fixed reward can be viewed as an extremely restricted discriminator: it assumes the commanded goal is the reached state only when they match. This sacrifices adaptive classification for a stable, stationary objective.

## 9. Fixed goal rewards and Skew-Fit

**Transcript coverage:** lines 3293-3772

### What the lecturer said - transcript only

The fixed equality reward still fits the information-theoretic framework if one regards it as restricting $q(g\mid s)$ to a single identity-like rule. It may be less statistically flexible when the policy is imperfect, but it removes the coupled dynamics of a learned reward and is often much easier to optimize.

The major design question becomes how to propose goals. Skew-Fit maintains a generative goal model $p_\phi(g)$, such as a VAE, and repeats:

1. sample $g\sim p_\phi(g)$;
2. attempt the goal with $\pi(a\mid s,g)$;
3. update the policy from collected data using an off-policy method such as SAC or DQN;
4. update the goal model from visited states.

Naively maximum-likelihood fitting $p_\phi$ to replay amplifies current bias. If the policy reaches the left side more reliably, the replay buffer contains more left-side states; the goal model then proposes the left more often, creating still more left-side data.

The repair is weighted maximum likelihood. Downweight frequent states and upweight rare ones:

$$
\max_\phi
\mathbb E_{g\sim\mathcal D}
[w(g)\log p_\phi(g)],
\qquad
w(g)=p_\psi(g)^\alpha,
$$

with negative $\alpha$. Choosing $\alpha=-1$ aims at a uniform distribution over the dataset's support; a softer value such as $-1/2$ avoids an overly aggressive correction.

### Source reconciliation

Slide 17 gives the Skew-Fit loop and weighted likelihood, with $w(g)=p_\psi(g)^\alpha$ for the current learned goal-model density. The transcript develops the positive-feedback failure of ordinary maximum likelihood.

### Additional explanation

Reweighting cannot create arbitrary invalid images because training examples still come only from reached states. It changes density within the empirical support, not the support itself.

## 10. Reweighting, frontier goals, and closing

**Transcript coverage:** lines 3773-4024

### What the lecturer said - transcript only

Inverse-density weighting does not undo the requirement that goals be valid: every positively weighted training point is a real visited state, while impossible states remain outside the data support and retain zero empirical weight.

Uniform coverage is not the only choice. A more aggressive explorer can suppress frequently visited states below uniform and concentrate goal proposals on the frontier of known experience. This intentionally pushes the policy toward rare reachable regions.

Go-Explore exemplifies the strategy: identify a promising frontier state, return to it with a goal-conditioned mechanism, then explore outward from there. Other work combines frontier selection with learned state representations designed to make goal proposals and distances more meaningful. The lecturer named classic papers as entry points and noted substantial more recent research.

The next Friday lecture would cover hierarchical RL and transfer learning.

### Source reconciliation

Slide 18 shows Go-Explore and representation/abstraction examples. The transcript describes their common frontier-goal principle without deriving either full algorithm.

### Additional explanation

Frontier sampling is curriculum construction: choose goals that are reachable enough to train on but far enough from familiar experience to expand coverage.

## Consolidated takeaways

- Reward-free skill learning pretrains behavior rather than waiting for a downstream objective.
- Entropy measures coverage; mutual information measures how much one variable predicts another.
- Empowerment rewards controllable choice: many possible outcomes that actions can select reliably.
- Skill-state mutual information produces distinguishable skill-conditioned state distributions without an external task reward.
- Pure distinguishability does not guarantee broad geometric coverage; nearby deterministic clusters may already be perfectly classifiable.
- Goal-conditioned empowerment grounds skills as desired states but requires a model or proposal mechanism over valid goals.
- Fixed goal-reaching rewards improve stability by replacing a learned discriminator.
- Naive goal-model maximum likelihood reinforces visitation bias; inverse-density reweighting or frontier proposals expand coverage.

## Key equations

1. **Entropy**

   $$
   \mathcal H(X)=-\mathbb E_{p(x)}\log p(x).
   $$

2. **Mutual information**

   $$
   I(X;Y)=D_{\mathrm{KL}}(p(x,y)\Vert p(x)p(y))
   =\mathcal H(Y)-\mathcal H(Y\mid X).
   $$

3. **One-step empowerment**

   $$
   I(A_t;S_{t+1})
   =\mathcal H(S_{t+1})-\mathcal H(S_{t+1}\mid A_t).
   $$

4. **Skill discovery**

   $$
   I(Z;S)=\mathcal H(Z)-\mathcal H(Z\mid S),
   \qquad r(s,z)=\log q_\phi(z\mid s).
   $$

5. **Goal discovery**

   $$
   I(G;S)=\mathcal H(G)-\mathcal H(G\mid S).
   $$

6. **Skewed goal-model fit**

   $$
   \max_\phi\mathbb E_{g\sim\mathcal D}
   [p_\psi(g)^\alpha\log p_\phi(g)],
   \qquad\alpha<0.
   $$

## Glossary

- **State marginal:** Time-aggregated state-visitation distribution induced by a policy.
- **Entropy:** Information-theoretic measure of distributional breadth or uncertainty.
- **Mutual information:** Reduction in uncertainty about one variable obtained from knowing another.
- **Empowerment:** Mutual information between an agent's choices and resulting states; a measure of control authority.
- **Skill-conditioned policy:** Low-level policy that additionally receives a persistent skill identifier.
- **Discriminator:** Classifier predicting which skill or goal produced a state.
- **DIAYN:** Diversity Is All You Need, a mutual-information skill-discovery method.
- **Goal-conditioned RL:** Learning a policy that reaches a supplied goal state.
- **Goal proposal distribution:** Distribution from which training goals are selected.
- **Skew-Fit:** Goal-exploration method that reweights visited states to flatten the learned goal distribution.
- **Frontier:** Boundary between well-explored and not-yet-explored state regions.

## Self-check questions

1. Why can Breakout be easier for DQN than Montezuma's Revenge?
2. Interpret $I(X;Y)=\mathcal H(Y)-\mathcal H(Y\mid X)$.
3. Why is an intersection more empowering than a deterministic hallway?
4. How do uniform skill sampling and discriminator reward optimize the two terms of $I(Z;S)$?
5. Why is DIAYN's discriminator cooperative with the policy rather than adversarial?
6. Give a case where skills are perfectly distinguishable but cover very little space.
7. What changes when the latent skill $z$ is replaced by a goal state $g$?
8. Why can random pixel goals not maximize useful $\mathcal H(G)$?
9. Why does ordinary maximum-likelihood goal fitting amplify current policy bias?
10. How do frontier goals differ from merely uniform goal sampling?

## Source coverage checklist

- [x] All supplied transcript lines 1-4024 are mapped exactly once in increasing, non-overlapping ranges.
- [x] Logistics, audience questions, optional examples, and closing remarks are retained.
- [x] Transcript-derived statements, slide reconciliation, and added explanation are separated.
- [x] All 18 slide pages were rendered and visually inspected.
- [x] Displayed mathematics uses Markdown-compatible LaTeX delimiters.

**Coverage result:** All 4,024 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps.
