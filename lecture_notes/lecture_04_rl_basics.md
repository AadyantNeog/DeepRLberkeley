---
title: "Lecture 4 - Reinforcement Learning Basics"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 4
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 4, Reinforcement Learning Basics.txt"
source_slides: "../lectures/Lecture 04 - RL Basics.pdf"
transcript_lines: 7921
slide_pages: 41
status: "complete"
source_qualifier: "opening-continues-prior-lecture"
---

# Lecture 4: Reinforcement Learning Basics

> **Source warning:** Lines 1-1968 of the supplied Lecture 4 transcript complete material from the preceding imitation-learning lecture: flow matching, action chunking, data design, pretraining, and goal-conditioned behavioral cloning. The supplied Lecture 4 slide deck begins with reinforcement-learning basics, which aligns with the transcript from line 1969 onward. These notes retain the transcript prelude in its supplied file and disclose the mismatch rather than moving or duplicating it silently.

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Flow-matching policy implementation and sampling Q&A | lines 1-558 |
| 2 | Action chunking | lines 559-981 |
| 3 | Recovery-focused demonstrations and augmentation | lines 982-1329 |
| 4 | Broad pretraining and narrow post-training | lines 1330-1725 |
| 5 | Multitask and goal-conditioned behavioral cloning | lines 1726-1968 |
| 6 | From imitation data to rewards and MDPs | lines 1969-2355 |
| 7 | Markov chains and transition operators | lines 2356-2688 |
| 8 | MDPs under a policy and partial observability | lines 2689-2943 |
| 9 | Trajectory distributions and state marginals | lines 2944-3318 |
| 10 | The reinforcement-learning objective | lines 3319-3483 |
| 11 | Finite-horizon objective in linear-algebra form | lines 3484-3987 |
| 12 | Infinite horizon and stationary distributions | lines 3988-4467 |
| 13 | Expectations, discontinuous rewards, and intermission Q&A | lines 4468-5054 |
| 14 | The three-part anatomy of an RL algorithm | lines 5055-5647 |
| 15 | Q-functions and value functions | lines 5648-6039 |
| 16 | Policy improvement from Q and V | lines 6040-6270 |
| 17 | Four broad RL algorithm families | lines 6271-6543 |
| 18 | Model-based, value-based, policy-gradient, and actor-critic loops | lines 6544-6834 |
| 19 | Why multiple algorithms: assumptions and tradeoffs | lines 6835-7065 |
| 20 | Sample efficiency and on-policy versus off-policy learning | lines 7066-7425 |
| 21 | Stability, convergence, and objective mismatch | lines 7426-7776 |
| 22 | Named algorithms, Atari, locomotion, and preview | lines 7777-7921 |

## Part I - Completion of the imitation-learning lecture

## 1. Flow-matching policy implementation and sampling Q&A

**Transcript coverage:** lines 1-558

### What the lecturer said - transcript only

After brief room chatter and a search for a marker, the lecturer resumed the preceding lecture. History can model non-Markovian expert behavior, while flow matching can represent multimodal action distributions. Both matter for Homework 1.

The policy network predicts a vector field rather than an action directly. It receives the environment observation, the current intermediate action at flow time $\tau$, and $\tau$ itself, and returns an action-dimensional velocity. For each minibatch element, training samples an observation-action pair, base noise - ordinarily a zero-mean unit-variance Gaussian - and a flow time. Uniform $\tau\in[0,1]$ is the simplest choice, though nonuniform schedules emphasizing early or late times can sometimes help. The intermediate action is a linear interpolation between noise and the demonstrated action, and the target velocity points from the noise to that action.

At inference, the necessary integration accuracy depends on how curved the learned paths are. A perfectly straight field can be integrated in one forward-Euler step; a highly curved field requires more. The lecturer’s rule of thumb was about 10 steps for actions and 100 for images. Delta-like targets or suitably aligned Gaussian distributions are special cases with straight paths.

If 10 steps are still too expensive, one can distill the sampler. The lecturer called the trick “reflow”: use an accurate, expensive integrator to map many noise samples to actions, save the noise-action pairs, and train a second network to make the mapping in one step. Observations remain an ordinary conditioning input; flow matching does not model them specially.

### Source reconciliation

This transcript segment corresponds to slides 15 and earlier material in the supplied **Lecture 03 - Behavioral Cloning Part 2** deck, not to the Lecture 4 deck. The prior deck visually confirms $v_\theta(o_t,a_{t,\tau},\tau)$ and the squared velocity loss. The transcript’s “forward oiler” and “refflow” are rendered here as forward Euler and reflow only in this reconciliation layer.

### Additional explanation

Distillation trades training cost for deployment speed. The teacher expends many integration evaluations once to create targets; the student amortizes those trajectories into a direct noise-to-action map.

## 2. Action chunking

**Transcript coverage:** lines 559-981

### What the lecturer said - transcript only

Action chunking predicts a sequence rather than one action. It is especially useful for high-frequency continuous control, such as robot manipulation or drones operating around 10-100 Hz, not primarily for language modeling or low-frequency discrete Atari decisions.

A standard Markov policy predicts $a_t$ from $o_t$, executes it, observes $o_{t+1}$, and repeats. A chunked policy predicts perhaps 10 or 50 actions. It may execute the entire block open loop or only a prefix, but executing just one removes the intended benefit. The control problem is effectively redefined so one decision is a sequence of native actions.

The lecturer cited the Diffusion Policy work on tabletop manipulation and said contemporary state-of-the-art imitation policies for robotic manipulation almost certainly use chunking. Why it helps is not settled. It may shorten the effective horizon by a factor equal to the executed prefix, or provide a stronger learning signal because each observation supervises many future actions.

Chunking definitely helps imitation learning, while its role in RL is an active research topic and does not behave identically. It is not merely a response to computational constraints. Chunking and history are independent choices and may be combined, though the lecturer did not know how strongly they synergize. Predicting a long chunk but executing only a prefix can still change representation learning through additional training targets; this is an empirical design choice without a definitive theory.

### Source reconciliation

This segment aligns with slide 16 of the prior lecture deck, which writes

$$
a_{t:t+K}\sim\pi_\theta(a_{t:t+K}\mid o_t)
$$

and cites Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. It is absent from the current Lecture 4 deck.

### Additional explanation

Chunking is a temporal abstraction. It reduces the number of policy queries but also reduces feedback frequency. Executing only a prefix retains replanning while still using the longer prediction task as an auxiliary training signal.

## 3. Recovery-focused demonstrations and augmentation

**Transcript coverage:** lines 982-1329

### What the lecturer said - transcript only

Data quality determines not only imitation accuracy but resilience to mistakes. Many systems are locally recoverable: a car that drifts slightly from its lane can return, unlike one that has already gone off a cliff. Demonstrations that are too perfect may never show such corrections. Introducing limited variability can teach the feedback needed after a small deviation without making the demonstrations disastrous.

The lecturer summarized the tradeoff as “the mistakes hurt, but the corrections help,” often by more than the mistakes hurt. DAgger obtains these examples from learner rollouts, but a cheaper practical alternative is to instruct demonstrators to include controlled mistakes and recoveries or to inject carefully bounded perturbations.

Simply applying the nearest familiar action is not enough. If a car is angled left, “go straight” may be the nearest demonstrated command, but recovery requires steering right. Correction depends on feedback from the current deviation.

Data augmentation can synthesize feedback examples. The earlier NVIDIA three-camera setup paired left- and right-facing views with corrective steering. Such transformations are domain specific but often valuable.

### Source reconciliation

This material corresponds to slide 20 of the prior lecture deck. The current Lecture 4 deck contains no matching slide.

### Additional explanation

Perturbation data should cover the local basin from which recovery is possible. If perturbations are too small, they add little robustness; if too large, their labels may be ambiguous or the system may be unrecoverable.

## 4. Broad pretraining and narrow post-training

**Transcript coverage:** lines 1330-1725

### What the lecturer said - transcript only

The lecturer presented pretraining as a recent and still uncertain imitation-learning recipe. A very large, relatively low-quality dataset first teaches broad knowledge. A much smaller, carefully curated dataset then fine-tunes or post-trains desired behavior.

Language models illustrate the recipe. Web-scale next-token imitation produces a raw completion model with broad knowledge, not necessarily a helpful question-answering assistant. Curated question-answer data then teaches the desired interaction style. The lecturer emphasized that language-model pretraining is still imitation learning: previous tokens are observations and the next token is the action, even when the process is called self-supervised.

Robotics has a similar tension. Broad data contains many situations, including mistakes, but also suboptimal actions. Narrow high-quality data gives good actions but covers fewer states. Pretrain on broad behavior so the model recognizes many situations, then post-train on good behavior to specify what it should do. The $\pi_0$ example used roughly 10,000 hours from diverse robots, followed by much narrower high-quality datasets for tasks such as laundry folding and box assembly.

Pretraining data need not be perfect, but it cannot be useless. Avoiding catastrophic forgetting mainly requires careful hyperparameters. The lecturer highlighted number of epochs and learning rate, suggesting fewer epochs in pretraining and more in post-training. He said mixing or alternating the phases was too recent for a confident conclusion.

### Source reconciliation

This segment corresponds to slides 21-23 of the prior deck, not the current Lecture 4 deck. The prior slide labels the system $\pi_0$ and displays about 10,000 hours of pretraining data and about 20 hours for each shown post-training task.

### Additional explanation

Pretraining aims to separate **coverage** from **behavioral preference**. It is not guaranteed to do so: post-training can overwrite broad competence, and pretraining can imprint undesirable behavior. The recipe works when representations learned from broad data remain useful after narrow behavioral refinement.

## 5. Multitask and goal-conditioned behavioral cloning

**Transcript coverage:** lines 1726-1968

### What the lecturer said - transcript only

Learning several tasks can paradoxically improve imitation. A policy trained only to reach one destination sees narrow trajectories. A policy conditioned on a requested destination can use demonstrations reaching many places. The larger variety covers more states and exposes more recoveries.

The intended goal need not be manually annotated when demonstrations are assumed successful. One can take the final state, or a sufficient statistic such as its location, and relabel the entire trajectory as an attempt to reach that outcome. The lecturer called this goal-conditioned behavioral cloning. It is useful when a diverse demonstration collection contains many behaviors whose original instructions are unavailable.

### Source reconciliation

This material aligns with slides 24-32 of the prior deck. Slide 26 formalizes final-state relabeling as $g=s_T^i$ and maximizes $\log\pi_\theta(a_t^i\mid s_t^i,g=s_T^i)$. It does not appear in the current Lecture 4 deck.

### Additional explanation

Relabeling converts every successful trajectory into supervision for the outcome it actually achieved. The learned policy can then share skills across goals, but it still inherits coverage limits: commanding a goal outside the relabeled goal distribution introduces a second form of shift.

## Part II - Markov decision processes and the RL objective

## 6. From imitation data to rewards and MDPs

**Transcript coverage:** lines 1969-2355

### What the lecturer said - transcript only

The remainder of the lecture would define reinforcement-learning problems rather than give a specific algorithm. The compact mathematical “universe” for sequential decisions is a Markov decision process. For now the setting is fully observed, so $o=s$ and the policy maps states directly to actions.

Imitation learning maximizes likelihood of demonstrated actions. RL instead assumes no good demonstration labels: the agent can interact, explore, and must determine good actions itself. A reward function supplies preferences. In the most general presentation it maps the current state-action pair to a scalar. It says which outcomes or state-action pairs are preferred, not which action sequence to follow.

For driving, the lecturer proposed reward zero most of the time, $+1$ at the destination, and $-1{,}000{,}000$ for an accident. Action dependence can encode inherent costs, such as $-0.001$ for slamming the accelerator to discourage wear. For simpler intuition, students can imagine state-only reward.

The current action cannot affect the current state; it affects the next state. Reward may reasonably be computed from observations because observations depend on state. The RL problem is to choose current actions that avoid undesirable outcomes many steps later. States, actions, transition probabilities, and reward together define the MDP.

### Source reconciliation

Slides 1-4 of the Lecture 4 deck begin here. Slide 4 writes $r(s_t,a_t)$ and says that $s_t$, $a_t$, $r$, and $p(s_{t+1}\mid s_t,a_t)$ define the MDP.

### Additional explanation

Reward is a specification, while the policy is a solution. Sparse reward can clearly state the goal yet provide almost no local hint about which action improves the chance of reaching it.

## 7. Markov chains and transition operators

**Transcript coverage:** lines 2356-2688

### What the lecturer said - transcript only

The lecturer traced MDPs to Markov chains, which contain states but no actions. A chain has a state space and transition probabilities $p(s_{t+1}\mid s_t)$. It satisfies the property that, conditional on the current state, the next state is independent of earlier states.

For a finite space of 10 states, the transition probabilities form a $10\times10$ matrix. The lecturer defined $\mu_t$ as the vector of current state probabilities and the matrix entry indexed by destination $i$ and source $j$ as the probability of entering $i$ from $j$. Then the entire distribution advances through matrix multiplication:

$$
\mu_{t+1}=\mathcal T\mu_t.
$$

The slide initially summed over the wrong state index; a student caught the typo, and the lecturer said the marginalization should sum over $s_t$. Repeated transition gives $\mu_{t+k}=\mathcal T^k\mu_t$. The linear-algebra view will later make analysis compact.

### Source reconciliation

Slide 5 identifies Andrey Markov and writes

$$
p(s_{t+1})=\sum_{s_t}p(s_{t+1}\mid s_t)p(s_t),
\qquad
\mathcal T_{i,j}=p(s_{t+1}=i\mid s_t=j).
$$

The transcript’s repeated “Markoff,” “Marov,” and “baset” are transcription errors; the slide-confirmed terms are Markov and Bayesian network.

### Additional explanation

Under this column-vector convention, every column of $\mathcal T$ sums to one. A different row-vector convention is equally valid, but formulas must use one orientation consistently.

## 8. MDPs under a policy and partial observability

**Transcript coverage:** lines 2689-2943

### What the lecturer said - transcript only

An MDP adds an action space and reward to a Markov chain. The action affects the next-state probability. The lecturer attributed popularization of this formalism to Richard Bellman in the 1950s.

For a fixed policy, an MDP can be converted into a Markov chain over state-action pairs. The probability of the next state-action pair is the environment transition probability multiplied by the probability that the policy selects the next action:

$$
p(s_{t+1},a_{t+1}\mid s_t,a_t)
=p(s_{t+1}\mid s_t,a_t)\pi(a_{t+1}\mid s_{t+1}).
$$

A partially observed MDP adds observations and an observation or emission distribution $p(o_t\mid s_t)$. It too can be written as a Markov chain by including the appropriate variables and applying the probability chain rule. The lecture then returned to full observability.

### Source reconciliation

Slides 6-7 define

$$
\mathcal M=\{\mathcal S,\mathcal A,\mathcal T,r\}
$$

for an MDP and

$$
\mathcal M=\{\mathcal S,\mathcal A,\mathcal O,\mathcal T,\mathcal E,r\}
$$

for a POMDP, where $\mathcal E$ is the emission operator. The transcript calls it an “initial operator” once; the slide confirms **emission operator**.

### Additional explanation

Fixing a policy removes the decision variable from the dynamics. The combined environment-policy system is then simply a stochastic process, allowing Markov-chain tools to analyze visitation and long-run reward.

## 9. Trajectory distributions and state marginals

**Transcript coverage:** lines 2944-3318

### What the lecturer said - transcript only

At each step, the policy produces an action and the environment produces the next state. For a finite horizon $H$, this induces a probability distribution over complete state-action trajectories. It depends on policy parameters because a different policy visits different states and actions.

By the chain rule, trajectory probability is the initial-state probability times the policy and transition probabilities at every step. A state marginal is the probability of being in a particular state at a particular time, obtained by summing or integrating all earlier states and actions out of the trajectory distribution.

Exact marginal expressions are cumbersome, but sampling is simple: run the policy in a simulator or the real world and read $s_t$ from the resulting trajectory. The lecturer’s rule of thumb was to use probability expressions for theory and samples for implementation. Practical algorithms usually never store full visitation probabilities.

### Source reconciliation

Slides 8-10 write

$$
p_\theta(\tau)=p(s_1)\prod_{t=1}^{H}
\pi_\theta(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t).
$$

The deck notes that $s'$ is shorthand for the next state. Its horizon indexing includes the successor $s_{H+1}$ even when the trajectory label visually ends at $(s_H,a_H)$; these notes preserve the product as displayed rather than silently changing its convention.

### Additional explanation

Sampling avoids enumerating exponentially many trajectories. A Monte Carlo average is therefore the basic bridge between the mathematical expectation and an executable algorithm.

## 10. The reinforcement-learning objective

**Transcript coverage:** lines 3319-3483

### What the lecturer said - transcript only

Reward identifies better state-action pairs, but delayed consequences mean the policy must optimize long-term reward. The RL objective is the expected cumulative reward under the trajectory distribution induced by the policy:

$$
\theta^*=\arg\max_\theta
\mathbb E_{\tau\sim p_\theta(\tau)}
\left[\sum_{t=1}^{H}r(s_t,a_t)\right].
$$

The reward function need not include transition probabilities. They are usually unknown; interaction supplies samples of their consequences. Every algorithm in the course aims, directly or indirectly, to maximize this expected cumulative reward.

### Source reconciliation

Slide 11 gives the same objective and emphasizes that the accident reward arrives too late to serve as a direct action label; the policy must optimize long-term consequences.

### Additional explanation

The expectation averages both policy randomness and environment randomness. Maximizing a single lucky trajectory would not solve the stochastic decision problem.

## 11. Finite-horizon objective in linear-algebra form

**Transcript coverage:** lines 3484-3987

### What the lecturer said - transcript only

Linearity of expectation rewrites expected total reward as a sum of expected per-time-step rewards under state-action marginals. For a fixed policy, the MDP becomes a Markov chain over state-action tuples.

If there are $n$ states and $m$ actions, the marginal vector $\mu_t$ has $nm$ entries such as $p(s_t=1,a_t=1)$ and $p(s_t=1,a_t=2)$. A policy-specific transition matrix maps one tuple distribution to the next. A reward vector $\tilde r$ contains $r(s,a)$ in the same tuple order. The expected reward at time $t$ is the inner product $\mu_t^\top\tilde r$, and

$$
\mu_t=\mathcal T_\theta^{,t-1}\mu_1.
$$

Thus the finite-horizon objective can be written compactly with powers of the transition matrix. The lecturer stressed that real RL algorithms normally do not know this matrix; the representation is for theoretical analysis.

A student identified another slide typo and the lecturer corrected it aloud, though the transcript does not preserve the full corrected expression. For continuous states or actions, these vectors are formally infinite-dimensional objects and measure-theoretic care is needed; the lecturer declined to pursue that machinery.

### Source reconciliation

Slides 12-13 visually confirm

$$
\mathcal T_{\theta,i,j}
=p(s'=s_i\mid s=s_j,a=a_j)\pi_\theta(a'=a_i\mid s'=s_i),
$$

$$
J(\theta)=\sum_{t=1}^{H}\mu_t^\top\tilde r
=\left[\sum_{t=1}^{H}\mathcal T_\theta^{,t-1}\mu_1\right]^\top\tilde r.
$$

This slide rendering supplies the corrected formula that the transcript says was fixed but does not fully state.

### Additional explanation

The formula exposes two ways $\theta$ matters: it changes the transition matrix of the combined policy-environment chain and therefore changes every future visitation marginal.

## 12. Infinite horizon and stationary distributions

**Transcript coverage:** lines 3988-4467

### What the lecturer said - transcript only

An undiscounted infinite sum can diverge, so the lecturer considered average reward by dividing the first $H$ rewards by $H$ and taking $H\to\infty$. Under the stated reachability and non-periodicity conditions, a Markov chain eventually forgets its initial state and converges to a stationary distribution. He described this with a random walk around Berkeley: after wandering long enough, the starting point no longer affects the distribution.

The transcript renders the relevant terms as “erotic,” “argotic,” and “a periodic.” The intended reachability statement was that any state can eventually be reached from any other; if two islands cannot communicate, the starting island is never forgotten. Periodic oscillation is another obstruction.

The stationary vector $\bar\mu$ is unchanged by a transition:

$$
\bar\mu=\mathcal T_\theta\bar\mu.
$$

Therefore $(\mathcal T_\theta-I)\bar\mu=0$, so $\bar\mu$ is an eigenvector with eigenvalue one. Long-run average reward becomes $\bar\mu^\top\tilde r$. Algorithms usually do not solve this system directly; the value is theoretical.

In Q&A, the lecturer said an ergodic aperiodic chain has a unique stationary distribution. Two disconnected internally ergodic islands can support two stationary distributions. Once the state distribution is stationary, the state-action distribution is also stationary because policy action probabilities are conditioned on state.

### Source reconciliation

Slide 14 uses the correct terms **ergodic** and **aperiodic** and writes

$$
\lim_{H\to\infty}\frac1H\sum_{t=1}^{H}
\mathbb E[r(s_t,a_t)]=\bar\mu^\top\tilde r.
$$

The transcript-only explanation informally equates ergodicity with mutual reachability; in a finite-chain treatment, irreducibility plus aperiodicity gives the convergence result being used.

### Additional explanation

Stationarity is a fixed point of distribution dynamics, not a state at which the physical system stops moving. Individual states continue changing even though their long-run probabilities remain constant.

## 13. Expectations, discontinuous rewards, and intermission Q&A

**Transcript coverage:** lines 4468-5054

### What the lecturer said - transcript only

Finite- and infinite-horizon formulations both optimize expected reward. Expectations matter because outcomes are stochastic: a choice may usually gain 20 dollars but carry a small catastrophic risk.

Sampling also makes discontinuous rewards workable. On a mountain road, reward $+1$ for staying on the road and $-1$ for falling is discontinuous and nondifferentiable as a function of physical state. Yet its expectation is smooth in the probability of falling. RL gradients generally differentiate policy-induced probabilities with respect to policy parameters, not reward with respect to state. This idea would become central in policy gradients.

The lecturer paused for five minutes, planning to resume at 9:12. During the intermission, a student discussed the 2011 Ross DAgger paper. The lecturer clarified that its refined proof shows low cost before visitation distributions necessarily converge; a simpler infinite-iteration mixture argument is less efficient. Asked whether cost could converge while distributions oscillate, he was unsure and guessed that regularity assumptions might rule it out.

Another student asked whether adding recovery mistakes eliminates distribution shift. The lecturer said no: it is a practical mitigation, not a bulletproof guarantee. A learned policy can err even in a recovery state and drift farther. Guarantees require further dynamical assumptions such as smoothness; empirically, mistakes and recoveries tend to produce more robust policies. Combining this data strategy with DAgger can still help.

Finally, he clarified an overly loose lottery example. The action is buy or do not buy; the possible states include a winning or losing ticket. Although outcomes are discontinuous, the expected return varies with the probability induced by the policy.

### Source reconciliation

Slide 15 writes the illustrative parameterization $\pi_\theta(a=\text{fall})=\theta$ and notes that $\mathbb E_{\pi_\theta}[r]$ can be smooth in $\theta$ even when $r(x)$ is not smooth in $x$. The DAgger discussion occurs during the recorded break and has no matching Lecture 4 slide.

### Additional explanation

For the binary road example, if reward is $+1$ for safe and $-1$ for fall with fall probability $\theta$, then

$$
\mathbb E[r]=(1-\theta)(+1)+\theta(-1)=1-2\theta,
$$

which is differentiable even though the outcome reward is discrete.

## Part III - Anatomy and families of RL algorithms

## 14. The three-part anatomy of an RL algorithm

**Transcript coverage:** lines 5055-5647

### What the lecturer said - transcript only

Most RL algorithms repeat three parts:

1. generate samples by running some behavior;
2. use those samples to estimate return or fit a model; and
3. use the estimate to improve the policy.

A simple policy-gradient loop sums trajectory rewards, increases probability of high-return trajectories, and decreases probability of low-return ones. A more complex model-based loop fits a neural next-state predictor and backpropagates or plans through it to improve behavior.

The cost of each box varies. Sampling can be cheap in a fast simulator, language-model generation with a verifier, or programmable Go self-play. It can be slow, dangerous, or expensive for real cars, robots, power grids, or public policy. A fast simulator might operate hundreds of thousands of times faster than real time. Exploration may intentionally collect diverse, informative data rather than follow the current best policy exactly.

Return estimation may be only a sum of rewards or a costly learned world model. Policy improvement may be one gradient step or substantial planning. One model-based strategy trains a world model, generates many additional synthetic samples, and passes them to a learner. Whether only reward or a full transition model is needed depends on the domain.

### Source reconciliation

Slides 17-21 depict the orange/green/blue loop and the two examples. Slide 21 summarizes that Monte Carlo return averaging is computationally cheap, transition-model fitting can be expensive, and real-world sample collection may dominate wall-clock cost.

### Additional explanation

This loop is a diagnostic template. For any new algorithm, identify what distribution generates data, what quantity the estimator learns, and how that estimate changes behavior. Many apparent algorithmic differences become choices inside one of these three stages.

## 15. Q-functions and value functions

**Transcript coverage:** lines 5648-6039

### What the lecturer said - transcript only

The nested expectation over future rewards can be compressed recursively. The Q-function for policy $\pi$ is expected total future reward when the agent starts in state $s_t$, first takes action $a_t$, and follows $\pi$ thereafter. The value function is expected total future reward when it starts in $s_t$ and follows $\pi$ immediately.

The value is the policy-weighted expectation of the Q-function:

$$
V^\pi(s)=\mathbb E_{a\sim\pi(\cdot\mid s)}[Q^\pi(s,a)].
$$

The expected value of $V^\pi(s_1)$ under the initial-state distribution equals the RL objective. Unlike full state marginals, Q-functions and value functions will actually be learned in later homework.

### Source reconciliation

Slides 23-24 define, for a finite endpoint $T$,

$$
Q^\pi(s_t,a_t)=
\mathbb E_\pi\left[\sum_{t'=t}^{T}r(s_{t'},a_{t'})\mid s_t,a_t\right],
$$

$$
V^\pi(s_t)=
\mathbb E_\pi\left[\sum_{t'=t}^{T}r(s_{t'},a_{t'})\mid s_t\right].
$$

### Additional explanation

$Q^\pi$ separates the value of the first action from the policy used later. That makes it the natural object for asking whether replacing the current action improves upon following $\pi$ unchanged.

## 16. Policy improvement from Q and V

**Transcript coverage:** lines 6040-6270

### What the lecturer said - transcript only

If $Q^\pi$ is known exactly, create $\pi'$ by choosing an action that maximizes it in every state. The new policy is at least as good as $\pi$ and is better whenever improvement is possible. The result does not depend on the quality of the initial policy, provided the Q-function and maximization are exact.

Alternatively, increase the probability of actions whose Q-value exceeds the current state value. Because $V^\pi(s)$ is the average Q under $\pi$, an action with $Q^\pi(s,a)>V^\pi(s)$ is better than average. Policy-gradient methods will use this relationship.

Estimating $Q$ or $V$ belongs to the evaluation/modeling box; it does not itself alter the policy. The improvement box then uses the estimate to change the actor.

### Source reconciliation

Slide 25 writes the greedy update

$$
\pi'(a\mid s)=1
\quad\text{if}\quad
a\in\arg\max_{a'}Q^\pi(s,a'),
$$

and highlights $Q^\pi(s,a)-V^\pi(s)$ as the sign of whether an action is above average.

### Additional explanation

The difference $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ is called the advantage. It centers action quality relative to the policy’s own baseline at that state.

## 17. Four broad RL algorithm families

**Transcript coverage:** lines 6271-6543

### What the lecturer said - transcript only

All canonical RL methods aim to maximize expected reward, but they estimate and improve in different ways.

- **Policy gradients** directly estimate the gradient of the RL objective. They are conceptually simple but often sample hungry, making them attractive when simulation is cheap. Sim-to-real humanoid locomotion is a common use.
- **Value-based methods** estimate $V$ or $Q$. A pure method may have no explicit policy and select the Q-maximizing action. DQN for Atari is the classic example and appears in Homework 3.
- **Actor-critic methods** maintain both a policy and a value estimator. The critic evaluates the current policy and trains the actor. These methods can be relatively sample efficient.
- **Model-based RL** learns environment dynamics and uses them for planning, differentiable policy optimization, or generation of synthetic experience.

Under the standard problem assumed here, sampled rewards are directly observed. The critic estimates expected future return, not the immediate reward function itself. Different reward-access assumptions would define different RL problems rather than different algorithms for this canonical one.

### Source reconciliation

Slide 28 lists the same four families and explicitly notes that value-based methods may have no explicit policy, while model-based methods can plan, improve a policy, or feed another learner.

### Additional explanation

The families overlap. Actor-critic combines policy gradients and value estimation; a model-based method may use a value function internally. The labels indicate the central learned object and improvement mechanism, not mutually exclusive software modules.

## 18. Model-based, value-based, policy-gradient, and actor-critic loops

**Transcript coverage:** lines 6544-6834

### What the lecturer said - transcript only

Model-based RL learns $p(s_{t+1}\mid s_t,a_t)$. The task can be easy when rules are known, as in chess, or extremely complex from visual observations. The model can support continuous trajectory optimization or optimal control, discrete search in games, gradients through suitably parameterized stochastic dynamics, or synthetic rollouts used to fit values.

Value-based methods place most difficulty in fitting $V$ or $Q$; policy improvement may simply enumerate a small discrete action set and choose the maximum. Direct policy gradients estimate returns by summing rewards and differentiate the objective with respect to policy parameters. Actor-critic fits values and then uses them in the policy gradient, combining both ideas.

### Source reconciliation

Slides 29-33 name the planning methods as model-predictive control and Monte Carlo tree search. The transcript renders these as “NPC” and “Monte Carlo research”; the slide-confirmed terms are used here only under reconciliation.

### Additional explanation

Planning replaces an amortized action rule with online computation. It can adapt to the current state without storing an explicit policy, but it spends computation at every decision.

## 19. Why multiple algorithms: assumptions and tradeoffs

**Transcript coverage:** lines 6835-7065

### What the lecturer said - transcript only

No single imperfect practical algorithm dominates every problem. Major tradeoffs include sample efficiency, computational cost, stability, ease of tuning, stochastic versus deterministic dynamics, continuous versus discrete actions, and episodic versus continuing interaction.

Q-learning and DQN are natural with a few discrete actions but awkward in high-dimensional continuous action spaces. Some methods work for stochastic systems; others need determinism or special distributional forms. Go naturally supplies resettable episodes. Stock trading is a continuing process that cannot simply return to Monday and repeat. A method should also exploit whether the policy or the world model is easier to represent in the domain.

### Source reconciliation

Slide 34 presents these considerations as different tradeoffs, assumptions, and representational difficulties.

### Additional explanation

Algorithm choice should start from bottlenecks: whether interaction, computation, representation, or optimization stability is scarce. “Most powerful” without a cost model is not a meaningful ranking.

## 20. Sample efficiency and on-policy versus off-policy learning

**Transcript coverage:** lines 7066-7425

### What the lecturer said - transcript only

Sample efficiency asks how much interaction is required for a good policy. An off-policy algorithm can improve a policy without collecting new samples from that exact policy; an on-policy algorithm requires fresh data after every policy change.

Model-based learning is off-policy in principle because any transition samples can train the model, although data near the target policy can still help with distribution shift. Basic policy gradients are on-policy: collect from the latest policy, compute one update, and discard the data.

Past-data reuse is valuable when samples are expensive. When a fast simulator makes samples cheap, on-policy methods can be preferable. Evolutionary search is even less sample efficient; policy gradients are mainstream on-policy methods; actor-critic can be on- or off-policy; Q-learning is classically off-policy; model-based approaches can be most sample efficient.

The lecturer warned that sample efficiency is not wall-clock efficiency. Moving toward model-based methods often exchanges more computation for less real data. A learned video model may generate simulated experience more slowly than real time. The best trade depends on the relative price of interaction and compute.

### Source reconciliation

Slides 35-36 place methods on a qualitative spectrum from model-based deep RL and off-policy Q-learning through actor-critic and on-policy policy gradients to evolutionary or gradient-free methods.

### Additional explanation

Off-policy reuse introduces its own distribution mismatch: the dataset may describe different behavior than the current policy. Importance weighting, conservative updates, or value-learning structure is typically needed to use that data safely.

## 21. Stability, convergence, and objective mismatch

**Transcript coverage:** lines 7426-7776

### What the lecturer said - transcript only

Sample-efficient algorithms are often harder to use. Deep RL may lack convergence guarantees under realistic nonlinear function approximation, even when tabular or linear special cases are understood. Hyperparameters and architecture choices therefore matter greatly, and convergence may still be to an undesirable solution.

Supervised neural-network training usually applies gradient descent to a fixed fitting objective. Many RL methods do not apply gradient descent directly to expected reward. Q-learning is a fixed-point iteration. Model-based RL maximizes model fit, but better average prediction does not necessarily yield a better policy because accuracy can improve in irrelevant regions and worsen in decision-critical ones.

Value fitting minimizes a prediction or Bellman-style error, which is not the same as expected reward; popular nonlinear deep value methods may have no general convergence guarantee. Policy gradient performs ascent on the true RL objective, but its gradients have high variance and practical enhancements can move the update away from the exact gradient.

### Source reconciliation

Slides 37-38 state that policy gradient is the only listed family that literally performs gradient ascent on the true objective. They distinguish convergent model fitting from the absent guarantee that better model fit implies a better policy.

### Additional explanation

There are two layers of optimization: solving the algorithm’s surrogate objective and improving actual return. An algorithm can converge perfectly on the surrogate while producing a poor policy if the surrogate is misaligned with decision quality.

## 22. Named algorithms, Atari, locomotion, and preview

**Transcript coverage:** lines 7777-7921

### What the lecturer said - transcript only

The lecturer named examples students would encounter: Q-learning, DQN, temporal-difference learning, fitted value iteration, REINFORCE, natural policy gradients, methods used in language-model RL, A3C, soft actor-critic, Dyna-like model-based methods, and MuZero. The transcript garbles several names but says they will be introduced later.

He showed a 2013 convolutional Q-learning system playing Atari and a simulated robotic locomotion policy trained with a policy-gradient method. The locomotion example used TRPO, described as similar to PPO; later lectures would explain trust regions and other stability ideas. The next sequence would cover policy gradients, actor-critic, and then value-based methods. The lecture ended with applause.

### Source reconciliation

Slides 39-41 supply the exact names and citations obscured by transcription:

- value methods: Q-learning, DQN, temporal-difference learning, fitted value iteration;
- policy gradients: REINFORCE, natural policy gradient, PPO;
- actor-critic: A3C and SAC;
- model-based: Dyna, MBPO, and MuZero;
- Atari: Mnih et al. (2013), Q-learning with convolutional networks;
- locomotion: Schulman et al. (2016), generalized advantage estimation and trust-region policy optimization with value approximation.

### Additional explanation

These demonstrations preview two very different regimes: a small discrete action set where Q maximization is easy, and continuous high-dimensional control where a parameterized stochastic policy is natural.

## Consolidated takeaways

1. The transcript begins by completing the previous imitation-learning lecture; the current RL deck begins only near line 1969.
2. Flow matching generates actions by integrating an observation-conditioned vector field; reflow can distill many integration steps into one.
3. Action chunks and recovery-focused data are powerful empirical tools for continuous-control imitation.
4. Broad pretraining can supply coverage while narrow post-training specifies desired actions.
5. An MDP consists of states, actions, dynamics, and reward; a POMDP adds observations and emissions.
6. A fixed policy turns an MDP into a Markov chain over state-action pairs.
7. The trajectory distribution factors into initial-state, policy, and transition probabilities.
8. RL maximizes expected cumulative reward, not immediate reward or likelihood of demonstrated actions.
9. State-action marginals can be propagated with a policy-specific transition operator for theoretical analysis.
10. Long-run average reward is an expectation under a stationary distribution when the relevant chain is ergodic and aperiodic.
11. Discontinuous physical rewards can have smooth expectations in policy-induced probabilities.
12. Most RL algorithms alternate data collection, return/model estimation, and policy improvement.
13. $Q^\pi$ evaluates a chosen first action; $V^\pi$ averages $Q^\pi$ under the policy.
14. Greedy or advantage-weighted policy updates improve behavior when values are accurate.
15. Policy-gradient, value-based, actor-critic, and model-based methods make different compromises.
16. Off-policy learning reuses data and is often sample efficient; on-policy learning collects fresh data after updates.
17. Greater sample efficiency can require more computation and produce less stable optimization.

## Key equations

### Flow-matching policy loss

$$
\mathcal L(\theta)=
\mathbb E\left[
\left\|v_\theta(o_t,a_{t,\tau},\tau)-(a_t-a_{t,0})\right\|^2
\right].
$$

### MDP and trajectory distribution

$$
\mathcal M=\{\mathcal S,\mathcal A,\mathcal T,r\},
$$

$$
p_\theta(\tau)=p(s_1)\prod_{t=1}^{H}
\pi_\theta(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t).
$$

### Finite-horizon RL objective

$$
J(\theta)=
\mathbb E_{\tau\sim p_\theta}
\left[\sum_{t=1}^{H}r(s_t,a_t)\right].
$$

### Marginal propagation

$$
\mu_{t+1}=\mathcal T_\theta\mu_t,
\qquad
\mu_t=\mathcal T_\theta^{,t-1}\mu_1.
$$

### Stationary average reward

$$
\bar\mu=\mathcal T_\theta\bar\mu,
\qquad
\bar J(\theta)=\bar\mu^\top\tilde r.
$$

### Value and Q-functions

$$
Q^\pi(s_t,a_t)=
\mathbb E_\pi\left[\sum_{t'=t}^{T}r(s_{t'},a_{t'})\mid s_t,a_t\right],
$$

$$
V^\pi(s_t)=\mathbb E_{a_t\sim\pi(\cdot\mid s_t)}[Q^\pi(s_t,a_t)].
$$

### Advantage and greedy improvement

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s),
\qquad
\pi'(s)\in\arg\max_a Q^\pi(s,a).
$$

## Glossary

- **Action chunking:** predicting multiple consecutive native actions as one higher-level policy output.
- **Actor:** the policy component in an actor-critic method.
- **Critic:** a learned value estimator used to evaluate and improve an actor.
- **Ergodic chain:** in the finite-chain context used here, a chain whose states communicate so the long-run distribution can forget its start; aperiodicity is also required for ordinary convergence.
- **Emission operator:** the observation distribution $p(o\mid s)$ in a POMDP.
- **Finite horizon:** a problem with a fixed, finite number of decision steps.
- **Infinite horizon:** a continuing problem with no fixed terminal time.
- **Markov chain:** a state-only stochastic dynamical process with transition probabilities depending on the current state.
- **Markov decision process:** a Markov dynamical system in which actions affect transitions and rewards score state-action outcomes.
- **Model-based RL:** RL that learns or uses a transition model.
- **Off-policy:** able to learn about or improve a policy from data not freshly generated by that exact policy.
- **On-policy:** requiring data generated by the current policy for each update.
- **POMDP:** a partially observed MDP with latent states and observation emissions.
- **Policy gradient:** a method that estimates a gradient of expected return with respect to policy parameters.
- **Q-function:** expected future return after a specified state-action pair and subsequent use of a policy.
- **Reward:** the scalar objective signal for a state-action outcome.
- **Sample efficiency:** the amount of environment interaction needed to obtain a useful policy.
- **State marginal:** the probability distribution of the state at one time step under a policy.
- **Stationary distribution:** a distribution unchanged by one transition of the Markov chain.
- **Trajectory distribution:** the policy-induced probability distribution over complete state-action sequences.
- **Value function:** expected future return from a state under a policy.

## Self-check questions

1. Which three inputs condition the flow-matching policy’s velocity field?
2. Why can one integration step suffice for a straight vector field, and what is reflow?
3. What are the proposed explanations for why action chunking helps?
4. Why can imperfect demonstrations with recoveries outperform perfect narrow ones?
5. How do broad pretraining and narrow post-training address different data needs?
6. How does final-state relabeling create goal-conditioned demonstrations?
7. What four objects define the MDP in this lecture?
8. Why does reward specify outcomes rather than actions?
9. Derive $\mu_{t+1}=\mathcal T\mu_t$ for a finite Markov chain.
10. How does a fixed policy convert an MDP into a Markov chain?
11. Factor the trajectory distribution into its component probabilities.
12. Why are state marginals hard to enumerate but easy to sample?
13. State the finite-horizon RL objective.
14. What does the policy-specific state-action transition matrix contain?
15. Under what conditions does the long-run distribution forget its starting state?
16. Why can a discontinuous reward have a differentiable expectation?
17. Identify the orange, green, and blue stages of a generic RL algorithm.
18. Distinguish $Q^\pi(s,a)$ from $V^\pi(s)$.
19. Why does $Q^\pi(s,a)>V^\pi(s)$ identify a better-than-average action?
20. Contrast policy-gradient, value-based, actor-critic, and model-based methods.
21. Define on-policy and off-policy learning.
22. Why can a more sample-efficient algorithm be slower in wall-clock time?
23. Why does lower value-prediction or model error not automatically imply higher return?

## Source coverage checklist

| Topic | Inclusive range | Coverage status |
|---:|---:|---|
| 1 | 1-558 | Accounted for; prior-lecture continuation |
| 2 | 559-981 | Accounted for; prior-lecture continuation |
| 3 | 982-1329 | Accounted for; prior-lecture continuation |
| 4 | 1330-1725 | Accounted for; prior-lecture continuation |
| 5 | 1726-1968 | Accounted for; prior-lecture continuation |
| 6 | 1969-2355 | Accounted for |
| 7 | 2356-2688 | Accounted for |
| 8 | 2689-2943 | Accounted for |
| 9 | 2944-3318 | Accounted for |
| 10 | 3319-3483 | Accounted for |
| 11 | 3484-3987 | Accounted for; slide typo disclosed and reconciled |
| 12 | 3988-4467 | Accounted for |
| 13 | 4468-5054 | Accounted for; includes substantive intermission Q&A |
| 14 | 5055-5647 | Accounted for |
| 15 | 5648-6039 | Accounted for |
| 16 | 6040-6270 | Accounted for |
| 17 | 6271-6543 | Accounted for |
| 18 | 6544-6834 | Accounted for |
| 19 | 6835-7065 | Accounted for |
| 20 | 7066-7425 | Accounted for |
| 21 | 7426-7776 | Accounted for |
| 22 | 7777-7921 | Accounted for |

**Coverage result:** All 7,921 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. All 41 pages of the Lecture 4 deck were visually inspected. The first 1,968 transcript lines complete the prior lecture and have no counterpart in the current deck; that source misalignment is disclosed throughout.
