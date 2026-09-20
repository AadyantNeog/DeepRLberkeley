---
title: "Lecture 14 - RL with Sequences and LLMs"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 14
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 14, RL with Sequences & LLMs.txt"
source_slides: "../lectures/Lecture 14 - LLM RL.pdf"
transcript_lines: 7768
slide_pages: 51
status: "complete"
---

# Lecture 14: RL with Sequences and LLMs

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Maximum-entropy inverse RL recap | lines 1-511 |
| 2 | Practical inverse RL and importance sampling | lines 512-958 |
| 3 | Inverse RL as a two-player game | lines 959-1441 |
| 4 | GANs and adversarial imitation learning | lines 1442-2068 |
| 5 | Language-model pretraining and post-training | lines 2069-3262 |
| 6 | One-step and token-level RL formulations | lines 3263-3891 |
| 7 | Policy gradients for language models | lines 3892-4302 |
| 8 | Value baselines, GAE, and GRPO | lines 4303-4848 |
| 9 | Reference-model regularization and PPO | lines 4849-5399 |
| 10 | Preference rewards and the Bradley-Terry model | lines 5400-6492 |
| 11 | Verifiers, judges, and process rewards | lines 6493-6948 |
| 12 | Partial observability and information gathering | lines 6949-7399 |
| 13 | Learning with histories under partial observability | lines 7400-7768 |

## Part I - From inverse RL to adversarial imitation

## 1. Maximum-entropy inverse RL recap

**Transcript coverage:** lines 1-511

### What the lecturer said - transcript only

The lecture resumed the previous discussion of inverse reinforcement learning (IRL). In ordinary RL, a reward function is supplied and an optimal or near-optimal policy is sought. In IRL, expert demonstrations are supplied and the goal is to infer a reward under which those demonstrations look optimal or nearly optimal. This differs from behavioral cloning: rather than directly copying the expert's action distribution, IRL tries to recover an objective that can be optimized.

The lecturer returned to maximum-entropy IRL. It places a probability distribution over trajectories in which higher-reward trajectories are exponentially more probable, while the partition function normalizes over all possible trajectories. The likelihood of the demonstrations therefore increases when their rewards rise, but the normalization term prevents the reward from being raised everywhere without consequence.

Differentiating the demonstration log likelihood gives an intuitive update. Increase reward on expert trajectories and decrease reward on trajectories sampled from the current soft-optimal policy. This gradient is exact in the maximum-entropy model, but using it literally requires repeatedly solving an RL problem for the current reward. That inner loop is computationally expensive.

In response to a question about why one would learn a reward rather than simply imitate, the lecturer emphasized two advantages. First, a policy learned by supervised imitation can suffer compounding errors: once it reaches states not represented in demonstrations, it may make further mistakes. A learned reward can be optimized with interaction or planning and can therefore correct behavior in such states. Second, a reward can sometimes transfer or generalize. For autonomous driving, demonstrations might implicitly reveal a preference for keeping clearance from other vehicles; expressing that preference as a reward can apply in configurations absent from the demonstrations. The lecturer also cautioned that reward learning is not automatically safe from overfitting.

### Source reconciliation

Slides 3-6 make the maximum-entropy trajectory model and gradient explicit. With dynamics and initial-state factors collected into the trajectory probability,

$$
p_\psi(\tau)=\frac{1}{Z_\psi}
p(s_1)\prod_t p(s_{t+1}\mid s_t,a_t)
\exp\!\left(\sum_t r_\psi(s_t,a_t)\right),
$$

and maximizing expert log likelihood yields the difference between expert and model-generated reward-feature expectations. The transcript's automatic renderings of the reward parameter are reconciled to $\psi$ using the slides.

### Additional explanation

The partition function couples all trajectories. Raising the reward of an expert trajectory helps its likelihood, but raising rewards on competing trajectories raises $Z_\psi$ and offsets that benefit. This is why the gradient has a positive expert term and a negative soft-optimal-policy term.

Recovering a reward does not make the problem identifiable in general: many rewards can induce the same optimal behavior. Maximum entropy supplies a probabilistic preference for less arbitrary explanations, while architectural choices, regularization, and the demonstration distribution still determine which reward is recovered.

## 2. Practical inverse RL and importance sampling

**Transcript coverage:** lines 512-958

### What the lecturer said - transcript only

A practical IRL algorithm need not solve the policy to convergence after every reward update. The lecturer described a deliberately "lazy" alternative: partially improve the policy for the current reward, use its trajectories for a reward update, and alternate. This removes much of the nested-loop cost, but the trajectories are then sampled from the current policy rather than exactly from the distribution required by the maximum-entropy likelihood gradient.

Importance sampling can correct that mismatch. A trajectory sampled under the current policy is reweighted by the ratio between its probability under the target maximum-entropy trajectory distribution and its probability under the sampling policy. Initial-state and transition-probability factors appear in both numerator and denominator and cancel. What remains is proportional to the exponential of trajectory return divided by the product of the policy probabilities of the sampled actions.

Because the target density contains an unknown normalizing constant, normalized importance weights are used across the sampled trajectories. As the policy improves toward the target distribution, the importance ratios approach one and the estimator becomes better behaved. However, multiplying action-probability ratios across a long sequence can produce high variance. The lecturer noted that practitioners sometimes omit the importance weights and still obtain useful algorithms, even though that is no longer the exact gradient.

### Source reconciliation

Slides 7-9 display the cancellation explicitly. For a sampled trajectory,

$$
w(\tau)
\propto
\frac{\exp(r_\psi(\tau))}
{\prod_t \pi(a_t\mid s_t)},
\qquad
\tilde w_i=\frac{w(\tau_i)}{\sum_j w(\tau_j)}.
$$

Here $r_\psi(\tau)=\sum_t r_\psi(s_t,a_t)$. The slides confirm that the dynamics cancel only because the target model and sampling process use the same environment dynamics.

### Additional explanation

This is a standard bias-variance-computation trade-off. Fully solving the inner RL problem is expensive; partially updating the policy is cheaper but introduces distribution mismatch; importance sampling restores correctness in principle but may introduce severe variance. Long language or control sequences make the product-form likelihood ratio especially fragile, which motivates clipped ratios and near-on-policy updates later in the lecture.

## 3. Inverse RL as a two-player game

**Transcript coverage:** lines 959-1441

### What the lecturer said - transcript only

The lecturer clarified several pieces of notation in response to questions. The target trajectory distribution is induced by the current learned reward, while $\pi$ is the policy currently used to generate trajectories. Both change during learning. A probability denoted generically by $p$ is not automatically the policy; it may include the entire trajectory probability, including environment factors.

The alternating procedure can be understood as a game. The reward learner tries to assign high reward to expert demonstrations and low reward to trajectories from the current policy. The policy then tries to find trajectories with high reward. If the reward exposes a difference between learner and expert, the policy is pushed to eliminate that difference. At equilibrium, the learner's trajectory distribution matches the expert's, so the reward can no longer distinguish them.

This adversarial view explains why one need not interpret every intermediate reward function as a reusable statement of the true task. During training, the reward is an opponent whose job is to identify how the current policy differs from the demonstrations. The policy's job is to remove that discrepancy.

### Source reconciliation

Slide 10 depicts the alternating game between a reward function and a policy. Its arrows make clear that expert samples remain fixed, while current-policy samples and the learned reward evolve together.

### Additional explanation

The equilibrium statement is distributional. It means that, for the discriminator or reward class being used, expert and learner behavior cannot be separated. If the reward model is too weak, distinct behaviors may look identical to it. If it is too powerful relative to a small dataset, it may memorize the demonstrations and give an unhelpfully easy learning signal.

## 4. GANs and adversarial imitation learning

**Transcript coverage:** lines 1442-2068

### What the lecturer said - transcript only

The preceding game resembles a generative adversarial network (GAN). A GAN trains a generator to produce samples and a discriminator to distinguish generated samples from real data. In adversarial imitation learning, the policy plays the generator and expert demonstrations play the real dataset. A classifier distinguishes expert from policy trajectories, and its output supplies the policy's reward.

Although a classifier could consume an entire trajectory, it is often advantageous to factor the discriminator over states or state-action pairs. That corresponds more closely to a Markovian reward and gives a denser training signal. The resulting adversarial-imitation algorithm has fewer conceptual components than repeatedly solving maximum-entropy IRL: alternate discriminator updates with policy updates using the discriminator-derived reward.

The lecturer stressed important drawbacks. At convergence, an ideal discriminator outputs one-half because expert and learner distributions match. Consequently, the discriminator is not a reusable task reward: its very purpose is to become uninformative once imitation succeeds. With few demonstrations and a powerful discriminator, overfitting is easy. GAN-style mode collapse can also make the learned policy cover only part of expert behavior. The discriminator often has the easier job and therefore needs regularization, such as spectral normalization or a gradient penalty.

Under a particular discriminator parameterization, the adversarial objective is equivalent to maximum-entropy IRL. The lecturer used this connection to show that GAN-like imitation is not an unrelated trick but a practical expression of the same underlying density-matching idea. He cited motion-capture imitation, including humanoid behaviors, as a setting in which these methods have worked.

### Source reconciliation

Slides 11-13 show the discriminator and the special form that recovers maximum-entropy IRL:

$$
D(\tau)=
\frac{\exp(r_\psi(\tau))}
{\exp(r_\psi(\tau))+\pi(\tau)}.
$$

The transcript uses several approximate automatic transcriptions of GAN and discriminator terminology; the slide labels establish the intended terms. The humanoid images on slide 13 illustrate motion-capture imitation rather than an additional derivation.

### Additional explanation

The discriminator can be converted into different reward shapes, for example $-\log(1-D)$ or $\log D-\log(1-D)$. These choices may have the same ideal equilibrium but very different learning dynamics. The central invariant is that policy behavior is rewarded for becoming less distinguishable from expert behavior.

## Part II - Reinforcement learning for language models

## 5. Language-model pretraining and post-training

**Transcript coverage:** lines 2069-3262

### What the lecturer said - transcript only

The lecturer introduced language models as sequence models over tokens. Tokens are roughly frequently occurring chunks of characters, not necessarily whole words. A Transformer maps token embeddings plus positional information through masked self-attention and positionwise multilayer perceptrons. The final softmax defines a distribution over the next token. Masking ensures that a position cannot use future tokens when predicting the next one.

Pretraining performs next-token prediction on a very large and diverse body of text, such as material collected from the web. This objective can force the model to acquire extensive knowledge. However, the raw pretrained model is fundamentally a completion engine. If given a question, it may continue the page rather than behave as a helpful assistant because pretraining alone did not specify the intended interaction pattern.

Post-training teaches the model how to use its pretrained knowledge. Supervised fine-tuning, also called instruction tuning, uses prompt-response examples and is essentially imitation learning. It can make the model follow instructions, but high-quality demonstrations are labor intensive, especially when the desired response requires expert skill.

RL post-training can instead optimize outcomes. The lecturer divided its reward sources into two broad classes. Preference rewards are general because people can compare outputs across many tasks, but they can encourage models to flatter, persuade, or fool the evaluator instead of becoming objectively correct. Verifier or correctness rewards are narrower because they require a checker, judge, sandbox, or ground truth, but they can directly optimize task success. RL with verifiers has been especially important for models that generate reasoning or "thinking" tokens before a final answer.

In Q&A, the lecturer connected language generation to partial observability. A single token by itself is only a small observation, but the complete prefix can serve as the state. For contexts too long to retain literally, a system might summarize or learn a representation, although brute-force attention over long histories remains powerful.

### Source reconciliation

Slides 14-18 visually specify the Transformer pipeline, autoregressive mask, pretraining objective, and the pretraining-to-post-training transition. They distinguish supervised fine-tuning from RL using human preferences or verifiable rewards. The transcript's occasional rendering of "LLM" as another short token is reconciled from these labels.

### Additional explanation

Pretraining and post-training solve different distributional problems. Pretraining estimates what text is likely in the corpus. Post-training changes the conditional distribution so that, for a user prompt, useful or successful responses receive greater probability. The latter may use knowledge acquired during pretraining without requiring demonstrations of every successful behavior.

## 6. One-step and token-level RL formulations

**Transcript coverage:** lines 3263-3891

### What the lecturer said - transcript only

Language generation can be cast as RL in two equivalent ways.

In a one-step formulation, the prompt is the state and the entire completion is one action. Although the action space is then the set of all strings, the policy remains tractable because the probability of a completion factorizes into next-token probabilities by the chain rule.

In a multi-step formulation, each generated token is an action. The state is the prompt followed by all tokens generated so far, and the deterministic transition appends the chosen token. The action space at each step is the vocabulary. This form is useful when rewards or value estimates are attached to intermediate reasoning steps rather than only to the final completion.

An audience question asked about special generation or thought markers. The lecturer said they are simply tokens in the sequence; their semantics come from training. Another question asked about discounting and length. Discounting or an explicit length penalty can be used, but a terminal reward is also a common formulation.

### Source reconciliation

Slides 19-21 write the factorization as

$$
\pi_\theta(a\mid s)
=\prod_{t=1}^{T}\pi_\theta(a_t\mid s,a_{1:t-1}).
$$

They also diagram the token-level state transition $s_{t+1}=(s_t,a_t)$. The two formulations represent the same sampled completion when rewards are assigned consistently.

### Additional explanation

The one-step view makes sequence-level policy gradients compact. The token-level view makes temporal credit assignment explicit. In implementation, a system often combines them: it samples a full response autoregressively, computes a sequence-level reward, and distributes an advantage or KL term across token log probabilities.

## 7. Policy gradients for language models

**Transcript coverage:** lines 3892-4302

### What the lecturer said - transcript only

For a prompt $s$ and completion $a$, the objective is the expected reward of completions sampled from the language model. Applying the score-function identity gives a policy gradient proportional to the reward times the gradient of the completion log probability. Because the completion log probability is a sum of token log probabilities, the update resembles reward-weighted cross-entropy over sampled responses.

Plain REINFORCE is therefore valid for LLMs. In practice, generating responses is expensive, so it is desirable to reuse a batch for more than one gradient step. Importance-ratio methods, including PPO-style updates, allow several minibatch updates while limiting how far the new policy moves from the sampling policy. A practical LLM RL system also needs a variance-reduction baseline, a regularizer, and a reward source.

### Source reconciliation

Slides 22-26 give the sequence-level gradient:

$$
J(\theta)=\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}[r(s,a)],
$$

$$
\nabla_\theta J(\theta)
=\mathbb{E}\!\left[r(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right]
=\mathbb{E}\!\left[r(s,a)\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s,a_{<t})\right].
$$

The slides place this gradient inside the now-familiar sample, score, update loop and then motivate importance-weighted PPO.

### Additional explanation

Calling the update "reward-weighted cross-entropy" is useful but incomplete. Unlike supervised learning, the training targets were sampled from the model itself, and their weights depend on outcomes. The data distribution therefore changes with the policy, which is why stale samples need likelihood-ratio correction or tightly limited reuse.

## 8. Value baselines, GAE, and GRPO

**Transcript coverage:** lines 4303-4848

### What the lecturer said - transcript only

A token-level actor-critic method can learn a value function and use generalized advantage estimation (GAE). The value target is the old value plus the estimated advantage. The value model may be a separate LLM or an additional scalar head on a shared backbone. Value loss should be applied to the generated suffix, not the prompt tokens, because the policy did not choose the prompt.

An audience member asked why not use a more fully off-policy critic. The lecturer described a practical trade-off: sampling LLM outputs is expensive, but maintaining and training a large critic is also computationally expensive. Importance-weighted policy-gradient methods currently occupy a useful middle ground.

The lecturer then described group relative policy optimization (GRPO). For each prompt, sample a group of $K$ completions. Use a completion's reward minus the mean reward of that prompt's group as its advantage. Returning to the same prompt prefix makes these multiple completions an empirical estimate of the prefix's value. This is not the same as subtracting one global average reward, because the baseline is prompt specific. GRPO removes the learned value model and is simpler to implement, though whether it is preferable to a PPO value model remains debated.

### Source reconciliation

Slides 27-29 give the temporal-difference residual and GAE construction,

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
\qquad
\hat A_t^{\mathrm{GAE}}
=\sum_{l\ge 0}(\gamma\lambda)^l\delta_{t+l},
$$

and the GRPO group baseline,

$$
\hat A_i=r_i-\frac{1}{K}\sum_{j=1}^{K}r_j.
$$

The transcript's automatic "GA" rendering is reconciled to GAE from the slide heading.

### Additional explanation

GRPO exchanges learned generalization for repeated sampling. A value network can predict a baseline for a prompt from previous training data, while a group baseline estimates it anew using several completions from that prompt. The latter saves value-model memory but consumes generation compute and can be noisy for small $K$.

## 9. Reference-model regularization and PPO

**Transcript coverage:** lines 4849-5399

### What the lecturer said - transcript only

RL fine-tuning usually regularizes the policy toward a fixed reference language model, commonly the instruction-tuned model immediately before RL. A KL penalty prevents the policy from drifting into nonsensical text or exploiting imperfections in the reward model. The effective reward is the task reward minus a weighted divergence from the reference.

The complete PPO-style recipe therefore samples responses, estimates advantages with a value model or group baseline, fits the value function when one is present, and optimizes a clipped importance-ratio objective. GRPO keeps the policy part but omits the learned value model. In a sampled response, the KL contribution can be viewed as penalizing tokens or completions that are much less likely under the reference than under the current policy.

The broader pipeline is: pretrain a model, instruction-tune it, copy that checkpoint as the fixed reference, and optimize a separate policy copy with RL.

### Source reconciliation

Slides 29-30 write the regularized reward as

$$
\bar r(s,a)=r(s,a)-\beta D_{\mathrm{KL}}
\bigl(\pi_\theta(\cdot\mid s)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid s)\bigr),
$$

and show the clipped PPO surrogate with the probability ratio between the current and data-collection policies. The transcript's "PO" rendering is reconciled to PPO from these slides.

### Additional explanation

The reference model is a behavioral prior, not the data-collection policy used in PPO's importance ratio. They can initially be the same checkpoint, but conceptually they have different roles: the old policy makes the gradient estimator usable for reused samples, while the reference supplies a long-lived constraint on linguistic behavior.

## 10. Preference rewards and the Bradley-Terry model

**Transcript coverage:** lines 5400-6492

### What the lecturer said - transcript only

When objective correctness is unavailable, reward can be learned from comparisons. For one prompt, generate two or more completions and ask a person which is better. The lecturer jokingly called this an "optometrist algorithm": repeatedly ask whether option one or option two is preferable.

The Bradley-Terry model treats a preference as stochastic. The probability that completion $i$ is preferred to completion $j$ is a logistic function of the difference between their scalar rewards. Training maximizes the log probability of the human comparisons. The lecturer connected this to Elo ratings and logistic regression.

The full RLHF loop is to sample responses, obtain comparisons, train a reward model, and optimize the policy against that learned reward using PPO or a related algorithm. In practice, systems often perform only one or a few large rounds of relabeling and reward training rather than continuously requesting labels. The lecturer suggested a rough scale of two or three rounds and perhaps around fifty policy-optimization steps as an illustrative pattern, not a universal prescription. Preference likelihoods are summed over comparisons and raters.

Q&A covered several design choices. Comparing answers to the same prompt makes judgments easier and more meaningful. One can collect multiple suffixes and pairwise comparisons or full rankings. Human judgments contain noise and uncertainty: a weak preference should not be treated as infinitely certain, and the data-collection design must account for how people actually label. The lecturer said detailed human-factors strategies were outside his expertise and emphasized that the learned reward can only be as good as its labelers.

### Source reconciliation

Slides 31-36 give the Bradley-Terry likelihood:

$$
P_\psi(\tau_i\succ\tau_j)
=\frac{\exp(r_\psi(\tau_i))}
{\exp(r_\psi(\tau_i))+\exp(r_\psi(\tau_j))}
=\sigma\!\left(r_\psi(\tau_i)-r_\psi(\tau_j)\right),
$$

with objective

$$
\max_\psi\sum_{(i,j)\in\mathcal D_{\mathrm{pref}}}
\log P_\psi(\tau_i\succ\tau_j).
$$

The slides visually distinguish the reward model from the actor and critic in the RLHF training loop.

### Additional explanation

Only reward differences affect Bradley-Terry probabilities, so adding the same constant to every reward changes nothing. The scale matters: multiplying reward differences changes how deterministic the modeled preferences are. Regularization and label noise therefore affect not only ranking but also the reward magnitudes later seen by the RL algorithm.

## 11. Verifiers, judges, and process rewards

**Transcript coverage:** lines 6493-6948

### What the lecturer said - transcript only

A reward model is often a separate LLM initialized from a related checkpoint and given a scalar output at the final token. Actor, value, and reward networks can use separate backbones or share parts of one model, with different memory and interference trade-offs.

Not every reward must be a preference model. A verifier may check an answer against ground truth, execute code in a sandbox, match a required pattern, or use a learned judge. A process reward grades intermediate reasoning steps instead of only the final answer. Dense process rewards can make exploration easier when correct final answers are rare, because partial progress receives a signal. They can also distract the model from the actual goal if the partial-credit scheme is an imperfect proxy.

An audience member proposed a curriculum: begin with process rewards and later switch to sparse final-outcome rewards. The lecturer considered this sensible but said he did not know a paper establishing it as a standard solution and suggested it as a possible final-project direction.

### Source reconciliation

Slides 37-39 list reward sources and diagram outcome versus process supervision. The slides support the distinction between a terminal scalar reward and rewards attached to intermediate tokens or reasoning steps; they do not turn the proposed curriculum into an endorsed result.

### Additional explanation

Process supervision changes credit assignment. It can shorten the effective horizon by telling the policy which intermediate decisions helped. The risk is reward hacking at the step level: a model may learn to produce reasoning that looks locally acceptable to the process judge without improving the final answer.

## Part III - Partial observability and sequence state

## 12. Partial observability and information gathering

**Transcript coverage:** lines 6949-7399

### What the lecturer said - transcript only

The lecturer returned to POMDPs. In a partially observed problem, the current observation is not Markov: earlier observations can still provide information about the hidden state even after the current observation is known. Examples include a car with blind spots, an agent in Minecraft, and a chatbot whose proper response depends on earlier conversation.

Partial observability creates information-gathering actions. An agent in a maze might climb a hill or look around a corner to learn where it is or what lies ahead. Such actions need not have immediate task reward, but they improve later decisions. If the full state were directly observed, the same information-gathering action would be unnecessary.

A POMDP can also require an optimal stochastic policy. The lecturer described a hidden initial state that is either $A$ or $C$, with a goal at $B$ and no observation revealing which side the agent occupies. Always moving in one deterministic direction fails forever for one of the two starts. Randomizing left and right eventually reaches the goal from either hidden state. The performance difference can amount to infinite regret for the deterministic policy in this construction.

The lecturer distinguished information gathering from exploration. Exploration is needed during learning to acquire knowledge and may disappear after training. Information gathering is part of the deployed optimal behavior whenever the environment itself remains partially observed.

### Source reconciliation

Slides 40-47 depict the blind-spot and maze examples, define the failure of the Markov condition for observations, and illustrate the $A$-$B$-$C$ stochastic-policy construction. The diagrams confirm that the randomization claim depends on the absence of an informative observation.

### Additional explanation

A POMDP separates two kinds of uncertainty: uncertainty because the environment state is hidden now, and uncertainty because the agent has not yet learned the environment. Information-gathering actions address the first. Exploration algorithms address the second. A real system can face both simultaneously.

## 13. Learning with histories under partial observability

**Transcript coverage:** lines 7400-7768

### What the lecturer said - transcript only

Vanilla policy gradient does not require the observation itself to be Markov: the score-function estimator can optimize any differentiable policy over the information it receives. However, the advantage or value estimator used with it may assume a Markov state, so a practitioner must be careful about what information is provided to the critic.

Naive Q-learning is more vulnerable. If identical observations correspond to different hidden situations, one observation does not have a single well-defined optimal action value. A greedy Q policy is deterministic and can fail in cases where stochasticity is necessary.

One formal solution is to define the state as the complete history,

$$
s_t=(o_1,a_1,o_2,a_2,\ldots,o_t).
$$

The history is Markov because the next history contains the entire previous history plus the next action and observation. It need not reconstruct the physical world perfectly; the formal transition distribution depends only on the current history. The obvious cost is that the state grows over time. Long tasks therefore motivate recurrent networks, Transformers, or learned summaries. The lecture ended at the scheduled time and deferred model-based POMDP methods.

### Source reconciliation

Slides 48-51 compare policy-gradient and value-based methods under partial observability and state the history-as-state construction. The transcript's repeated automatic variations of "Markov" are reconciled using the slide terminology.

### Additional explanation

Using the full history makes the mathematical state Markov but does not make learning easy. The representation must still extract the history features that predict future rewards. A recurrent or attention model is therefore an approximation to belief-state inference: it compresses the history into a statistic useful for action selection and value prediction.

## Consolidated takeaways

1. Maximum-entropy IRL raises reward on expert behavior and lowers it on behavior generated by the current soft-optimal policy; practical alternating updates trade exactness for efficiency.
2. Importance sampling corrects the trajectory-distribution mismatch in principle, but sequence likelihood ratios can have high variance.
3. Adversarial imitation turns the reward learner into a discriminator. Its equilibrium goal is distribution matching, so the converged discriminator is not a reusable task reward.
4. LLM pretraining learns next-token prediction, instruction tuning imitates desired responses, and RL post-training directly favors preferred or verifiably successful outcomes.
5. A whole completion may be treated as one action, or each token may be treated as an action with the prefix as state.
6. PPO-style LLM training combines policy gradients, variance-reduction baselines, limited sample reuse, and a KL penalty to a fixed reference model.
7. GRPO replaces a learned value baseline with a prompt-specific mean over several sampled completions.
8. Bradley-Terry reward modeling converts pairwise preferences into a differentiable scalar-reward likelihood, but the result inherits labeler limitations.
9. Verifiers and process rewards can provide more objective or denser signals, while introducing coverage and proxy-design constraints.
10. In a POMDP, information gathering can remain necessary after training; representing the complete history restores the formal Markov property but creates a representation problem.

## Key equations

### Maximum-entropy trajectory model

$$
p_\psi(\tau)=\frac{1}{Z_\psi}p(s_1)
\prod_t p(s_{t+1}\mid s_t,a_t)
\exp\!\left(\sum_t r_\psi(s_t,a_t)\right).
$$

### Importance weight for a policy-sampled trajectory

$$
w(\tau)\propto
\frac{\exp(\sum_t r_\psi(s_t,a_t))}
{\prod_t\pi(a_t\mid s_t)}.
$$

### Autoregressive completion probability

$$
\pi_\theta(a\mid s)=\prod_{t=1}^{T}
\pi_\theta(a_t\mid s,a_{<t}).
$$

### Sequence-level policy gradient

$$
\nabla_\theta J
=\mathbb{E}\!\left[(r(s,a)-b(s))
\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s,a_{<t})\right].
$$

### GAE

$$
\hat A_t^{\mathrm{GAE}}
=\sum_{l\ge 0}(\gamma\lambda)^l
\left(r_{t+l}+\gamma V(s_{t+l+1})-V(s_{t+l})\right).
$$

### GRPO group advantage

$$
\hat A_i=r_i-\frac{1}{K}\sum_{j=1}^{K}r_j.
$$

### KL-regularized reward

$$
\bar r=r-\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}}).
$$

### Bradley-Terry preference probability

$$
P(\tau_i\succ\tau_j)=
\sigma\!\left(r_\psi(\tau_i)-r_\psi(\tau_j)\right).
$$

## Glossary

- **Inverse reinforcement learning (IRL):** Inferring a reward function from demonstrations rather than optimizing a supplied reward.
- **Maximum-entropy IRL:** A probabilistic IRL model in which trajectory probability is proportional to exponentiated return.
- **Partition function:** The normalizer that sums unnormalized probability over possible trajectories.
- **Importance sampling:** Reweighting samples from one distribution to estimate an expectation under another.
- **Adversarial imitation learning:** Imitation in which a discriminator distinguishes expert from policy behavior and supplies the policy reward.
- **Mode collapse:** Failure to cover all important modes of a target behavior distribution.
- **Token:** A vocabulary unit, often a frequently occurring character or word fragment, generated by a language model.
- **Instruction tuning:** Supervised fine-tuning on prompt-response demonstrations.
- **Verifier:** A program, environment, or learned judge that scores whether an output satisfies a checkable criterion.
- **Process reward:** Reward applied to intermediate reasoning or actions rather than only the final outcome.
- **GAE:** Generalized advantage estimation, an exponentially weighted combination of temporal-difference residuals.
- **GRPO:** Group relative policy optimization, using within-prompt sampled rewards as a baseline instead of a learned value model.
- **Reference model:** A fixed language model used as a behavioral prior through a KL penalty during RL fine-tuning.
- **Bradley-Terry model:** A logistic model of pairwise preference based on the difference between scalar scores.
- **POMDP:** A decision process in which the agent observes information correlated with, but not necessarily equal to, the Markov state.
- **Information-gathering action:** An action valuable because it improves the agent's knowledge for later decisions.
- **History state:** The complete observation-action prefix treated as the current state.

## Self-check questions

1. Why does the maximum-entropy IRL gradient contain both an expert term and a current-policy term?
2. Which factors cancel in the trajectory importance ratio, and why can the remaining product have high variance?
3. Why is a converged adversarial-imitation discriminator not a reusable task reward?
4. What different roles do pretraining, instruction tuning, and RL post-training play in an LLM pipeline?
5. How are the one-step completion formulation and token-level formulation mathematically compatible?
6. Why does PPO need an old sampling policy, while KL regularization needs a fixed reference policy?
7. What computation does GRPO remove, and what additional sampling does it require?
8. Why can preference optimization reward persuasion or flattery even when labels are collected correctly?
9. When can process rewards help, and how can they become a misleading proxy?
10. Why can a POMDP require a stochastic policy even when a fully observed MDP has a deterministic optimum?
11. Why is exploration different from information gathering?
12. In what formal sense is the complete history Markov?

## Source coverage checklist

- [x] Transcript lines 1-7768 are assigned once, in monotonic and inclusive ranges.
- [x] Every supplied transcript line is represented; the recording ends normally rather than mid-sentence.
- [x] All 51 slide pages were rendered and visually inspected.
- [x] Maximum-entropy IRL, importance-weight, GAN/IRL, policy-gradient, GAE, GRPO, KL, preference, and POMDP equations were checked against the slides.
- [x] The Transformer, RLHF, process-reward, and POMDP diagrams were visually reconciled with the transcript.
- [x] Transcript errors such as "PO" for PPO, "GA" for GAE, and inconsistent renderings of $\psi$ and "Markov" are disclosed rather than silently adopted.
- [x] Lecturer claims, slide-only precision, and additional explanation are kept separate.

**Coverage result:** Complete for the supplied 7,768-line transcript and 51-page slide deck.
