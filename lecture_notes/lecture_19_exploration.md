---
title: "Lecture 19 - Exploration"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 19
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 19, Exploration.txt"
source_slides: "../lectures/Lecture 19 - Exploration.pdf"
transcript_lines: 4387
slide_pages: 23
status: "complete"
---

# Lecture 19: Exploration

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Exam logistics and model-based offline RL | lines 1-799 |
| 2 | Sparse rewards and temporally extended discovery | lines 800-1450 |
| 3 | Exploration, exploitation, and real-world examples | lines 1451-1879 |
| 4 | Tractability spectrum and reward shaping | lines 1880-2029 |
| 5 | Multi-armed bandits as a POMDP | lines 2030-2509 |
| 6 | Regret and optimism in the face of uncertainty | lines 2510-2989 |
| 7 | UCB count bonuses and logarithmic regret | lines 2990-3394 |
| 8 | Exploration bonuses in large MDPs | lines 3395-3577 |
| 9 | Density models and pseudo-counts | lines 3578-4093 |
| 10 | CTS density estimation and random network distillation | lines 4094-4387 |

## Part I - Completing offline RL

## 1. Exam logistics and model-based offline RL

**Transcript coverage:** lines 1-799

### What the lecturer said - transcript only

After spring break, the lecturer announced that the midterm was about two weeks away. Because this was the first year with an exam, there was no previous exam to distribute. The staff would post representative practice questions, a topic list, and other guidance on Ed, and both lectures the following week would be review lectures. TAs would offer topic-specific review sections. The lecturer described the midterm as a final exam in disguise, scheduled early so students could reserve the formal final period for projects. Exploration and RL theory were the last new topics expected on the exam.

Before exploration, he completed the model-based offline-RL material deferred from Lecture 18. Recall the Dyna-style recipe: learn stochastic dynamics from transition data, start short model rollouts from real replay states, and use those transitions in an off-policy RL algorithm. Short branches limit compounding error, and replay starts keep the simulated state distribution near real data.

In the offline setting, the policy cannot collect corrective transitions. If a model mistake makes an action look favorable, the policy can exploit it indefinitely. The remedy is to punish model rollouts in regions where the dynamics may be wrong. One method subtracts an uncertainty penalty from predicted reward. A bootstrap ensemble can estimate uncertainty through disagreement, and a sufficiently large penalty discourages unfamiliar state-action inputs. This is the model-based pessimism principle used by methods such as MOPO.

An audience member noted that a model can still be wrong where the ensemble appears confident. The lecturer agreed that the scheme is not bulletproof. Agreement is only a proxy for adequate coverage, and the penalty weight must be tuned. Uncertainty must depend on both state and action because a familiar state can be paired with an unfamiliar action.

A related COMBO-style method applies a CQL-like penalty to model-generated state-action pairs. Rather than only modifying reward, train the critic to assign low value to the short-rollout samples that look unlike real data. Negative model reward and lowered model-sample value express similar pessimistic principles.

### Source reconciliation

Slides 1-2 complete the material previewed on Lecture 18 slides 32-35. The reward-penalty form can be summarized as

$$
\tilde r(s,a)=\hat r(s,a)-\lambda u(s,a),
$$

where $u$ is an ensemble-disagreement or other epistemic-uncertainty score. The diagrams visually distinguish MOPO's reward pessimism from COMBO's conservative Q-value treatment.

### Additional explanation

Model-based offline RL compounds two optimizer's curses: the policy can exploit errors in both the transition model and the critic trained on model output. Short rollouts limit temporal accumulation; uncertainty penalties and conservative values limit spatial departure from the data. None replaces the need for coverage assumptions.

## Part II - Why exploration is difficult

## 2. Sparse rewards and temporally extended discovery

**Transcript coverage:** lines 800-1450

### What the lecturer said - transcript only

Exploration is almost the opposite of offline RL. Offline methods avoid unfamiliar regions; an online explorer deliberately seeks new experience.

The lecturer contrasted two Atari games. Breakout is easy for a standard deep-RL agent because breaking each brick immediately yields reward, so local reward points toward winning. Montezuma's Revenge is much harder even though its images are semantically clearer to a person. The agent receives reward for picking up a key and opening a door, but no direct negative reward for being killed by a skull. A locally rewarding strategy is to collect the key, die so it reappears, and repeat until lives run out. Winning requires a long sequence across roughly twenty distinct puzzle levels, and successful events only weakly correlate with completing the game.

People bring prior semantic knowledge about skulls, keys, doors, and goals; a from-scratch RL agent sees pixels and rewards. The lecturer compared this to the card game Mao and to Calvinball. Players begin without being told the rules and infer them only after penalties for violations. To an RL agent, an unknown Markovian environment resembles a coherent but arbitrary rule system whose rules can be discovered only through interaction.

Difficulty grows with temporal extent: many unrewarded decisions may be required before any positive or negative reinforcement reveals whether the sequence was useful. A simulated dexterous-hand task illustrated the same problem. If reward is given only when a ball reaches a marked circle or a nail is fully hammered, random finger motion is exceedingly unlikely to reveal the intended task. The demonstration shown for the manipulation task had therefore been initialized with human examples; learning entirely from scratch would be far harder.

### Source reconciliation

Slides 3-6 show Breakout, Montezuma's Revenge, the Mao quotation, and the manipulation examples. The transcript's variants "Montuma" and "Montazuma" are reconciled to **Montezuma's Revenge** from the slide title. The card-game name is reconciled to **Mao** from slide 5.

### Additional explanation

Sparse reward is not merely a small numerical signal. It is a credit-assignment and discovery problem: the agent must first encounter a successful trajectory before value learning can propagate its consequences backward. The probability of accidental success can decrease exponentially with the number of coordinated choices.

## 3. Exploration, exploitation, and real-world examples

**Transcript coverage:** lines 1451-1879

### What the lecturer said - transcript only

The lecturer gave two equivalent views of exploration. One asks how an agent discovers a high-reward strategy requiring a temporally extended sequence of individually unrewarding behaviors. The other asks whether the agent should try new behavior that might be better or continue the best behavior found so far.

In Montezuma's Revenge, repeatedly collecting the key is exploitation: it is the best known source of reward. Searching for a way through the door is exploration: the intermediate actions may pay nothing, but they could reveal a much better long-term strategy.

Exploitation means doing what current knowledge predicts will yield the highest reward. With no knowledge, it may mean doing nothing. Exploration means doing something not previously tried in the hope of finding a better outcome. Since the agent does not know what will be better, novelty is often the only general signal available.

The lecturer gave three classical examples. A diner can revisit a favorite restaurant or try a new one. An advertising system can show the currently best-performing advertisement or test another that may match a user better. An oil company can drill at the best known location or investigate a new site. Exploration often has a high probability of a worse immediate outcome and a smaller chance of discovering a much better option.

Finding a globally optimal exploration strategy is computationally difficult. Optimality can be formulated through regret or a Bayes-optimal policy that explicitly tracks uncertainty, but exact solutions are feasible only in small problems.

### Source reconciliation

Slides 7-9 state the exploration/exploitation definitions, show the restaurant/advertising/oil examples, and introduce regret versus Bayesian optimality as theoretical criteria.

### Additional explanation

Exploration has instrumental value: novelty is useful only because the resulting information can improve later reward. A novelty-seeking policy that never converts knowledge into exploitation can perform poorly even if it visits every state.

## 4. Tractability spectrum and reward shaping

**Transcript coverage:** lines 1880-2029

### What the lecturer said - transcript only

Exploration problems lie on a spectrum. Multi-armed bandits are one-step decision problems and permit clean theory. Contextual bandits add an observed context. Small tabular MDPs allow Bayesian model identification and value-of-information calculations. Large or infinite MDPs with image observations make exact optimal exploration essentially intractable.

The common practical strategy is to derive methods with optimal or near-optimal guarantees in bandits or small MDPs and apply their principles heuristically to deep RL. The guarantee does not automatically survive that transfer.

An audience member asked whether reward could simply be changed to remove the exploration problem. The lecturer identified this as reward shaping. It is sometimes appropriate when domain knowledge provides useful intermediate goals, as in robotic manipulation. It cannot solve every setting. In advertising, uncertainty over an individual user's response is the point of the problem, so exploration remains necessary.

### Source reconciliation

Slide 10 lists the progression from multi-armed bandits through contextual bandits and structured/tabular MDPs to large deep-RL problems. It presents tractability as a spectrum rather than a sharp boundary.

### Additional explanation

Reward shaping can change the optimal policy if designed carelessly. Potential-based shaping is a classical way to add dense feedback while preserving optimal policies under suitable assumptions, but the lecture's main point is more basic: shaping requires knowledge the agent may not have.

## Part III - Bandits and optimism

## 5. Multi-armed bandits as a POMDP

**Transcript coverage:** lines 2030-2509

### What the lecturer said - transcript only

The term "bandit" comes from a slot machine's nickname, a one-armed bandit. A multi-armed bandit offers several actions or arms. Each produces a stochastic reward, and the learner must identify which is best. The lecturer also offered medication selection as an alternative analogy: each treatment has an unknown probability of success.

For a Bernoulli bandit, action $i$ produces reward one with unknown probability $\theta_i$ and reward zero otherwise. A prior $p(\theta)$, perhaps uniform, represents initial uncertainty. The learner does not need to identify every $\theta_i$ precisely; it only needs enough evidence to find the arm with the largest success probability and then exploit it.

Each pull is a one-step problem, but learning across repeated pulls is sequential. The hidden state of a meta-level POMDP is the vector $\theta=(\theta_1,\ldots,\theta_n)$. Rewards from pulls are also observations about that hidden state. If $\theta$ were observed, the optimal policy would always choose $\arg\max_i\theta_i$. When it is hidden, the history of actions and rewards supports a belief over $\theta$, and the optimal POMDP policy is the Bayes-optimal exploration strategy.

One could solve this idealized POMDP with policy gradients or belief-state planning. That is overkill for a simple bandit but defines an upper standard against which simpler exploration rules can be compared.

### Source reconciliation

Slides 11-12 formalize

$$
r\mid a=i\sim\operatorname{Bernoulli}(\theta_i),
\qquad
\theta\sim p(\theta),
$$

and diagram repeated actions and reward observations as a POMDP over fixed hidden arm parameters.

### Additional explanation

The hidden parameter is constant across pulls. That temporal persistence is why information has value: observing one reward changes the posterior used for all later decisions.

## 6. Regret and optimism in the face of uncertainty

**Transcript coverage:** lines 2510-2989

### What the lecturer said - transcript only

Exploration quality is commonly measured by regret over a horizon $H$. Compare the reward an omniscient agent would expect from playing the best arm for all $H$ steps with the reward actually collected. If the best arm were known immediately, expected regret would be zero. Learning generally incurs regret because some pulls test inferior arms.

Minimizing regret is equivalent to maximizing total reward up to a constant and a sign. The best action in the benchmark is not known to the learner; it is an oracle comparator used for analysis.

Several simple strategies can asymptotically match the regret order of an optimal bandit solver. The lecturer focused on optimism in the face of uncertainty. Track the empirical mean reward $\hat\mu_a$ of each arm. Pure exploitation chooses the largest mean, but a lucky first success can make it stick forever to the wrong arm. Instead, add an uncertainty bonus $C\sigma_a$ and choose the arm with the largest optimistic estimate.

As $C$ approaches zero, the rule becomes exploitation. For a very large $C$, it prioritizes uncertainty. Intermediate values try each arm until evidence makes it sufficiently clear that the arm is not great.

### Source reconciliation

Slides 12-13 write regret as

$$
\operatorname{Reg}(H)
=H\,\mathbb E[r(a^*)]-\sum_{t=1}^{H}r(a_t),
$$

and the optimistic action as

$$
a_t=\arg\max_a\left(\hat\mu_a+C\sigma_a\right).
$$

### Additional explanation

Optimism converts uncertainty into a temporary reward advantage. Sampling an uncertain arm reduces its uncertainty, so the bonus naturally decays unless its empirical reward remains competitive.

## 7. UCB count bonuses and logarithmic regret

**Transcript coverage:** lines 2990-3394

### What the lecturer said - transcript only

A classical upper-confidence-bound (UCB) rule uses the number $N(a)$ of times an arm has been selected. The uncertainty bonus grows when an arm is rarely sampled and shrinks like one over the square root of its count. The logarithm of the total horizon appears in the numerator.

The inverse-square-root dependence comes from concentration or learning-theory bounds; the lecturer referred to Hoeffding's inequality. With this UCB rule, regret grows as $O(\log H)$ in the analyzed bandit setting, which is asymptotically optimal up to constants for the problem class being discussed.

An audience member asked about an explicit trade-off coefficient. The lecturer said one can include and tune a constant even when a theoretical presentation fixes it for a simplified analysis.

The theoretical result does not carry directly to deep multi-step environments, but it motivates using visitation counts as an exploration bonus.

### Source reconciliation

Slide 13 gives the rule

$$
a_t=\arg\max_a
\left[
\hat\mu_a+\sqrt{\frac{2\ln H}{N(a)}}
\right]
$$

and states $\operatorname{Reg}(H)=O(\log H)$. The transcript's automatic "haftings" rendering is reconciled to Hoeffding from the standard bound context; the slide itself cites Auer and colleagues rather than spelling out Hoeffding.

### Additional explanation

The displayed formula is undefined at $N(a)=0$, so an implementation initially pulls every arm or treats an untried arm's bonus as infinite. Logarithmic regret is gap dependent; constants worsen when the best and second-best arms have nearly equal mean reward.

## Part IV - Exploration bonuses for deep RL

## 8. Exploration bonuses in large MDPs

**Transcript coverage:** lines 3395-3577

### What the lecturer said - transcript only

In an MDP, novelty can depend on both state and action. For Atari, the action set is small while the state space is enormous, so practical bonuses often depend only on state. In general, a state-action bonus may be more appropriate.

Augment the environment reward with a bonus that decreases with visitation count, then use the augmented reward in an ordinary algorithm such as PPO, actor-critic, or DQN. Many decreasing functions are possible, including $1/N$ and $1/\sqrt N$, and their scale must be tuned.

Exact counting fails for images and continuous states. Two Atari frames are almost never pixel-identical because characters and objects move. A continuous floating-point state may never repeat exactly. Then every literal count remains near one, even when the agent repeatedly visits semantically similar situations. A useful method must soften equality into familiarity.

### Source reconciliation

Slides 14-16 write the augmented reward generically as

$$
r^+(s,a)=r(s,a)+\beta\,B(N(s,a)),
$$

with $B$ decreasing as count increases, and illustrate why literal pixel counts fail.

### Additional explanation

An intrinsic bonus changes during training because familiarity changes. The resulting reward is nonstationary. Replay-based algorithms must decide whether to store the bonus at collection time or recompute it under the current novelty model.

## 9. Density models and pseudo-counts

**Transcript coverage:** lines 3578-4093

### What the lecturer said - transcript only

A principled way to soften counts is to fit a density model $p_\theta(s)$, or $p_\theta(s,a)$, to visited experience. A new state similar to old states can receive high density and be treated as familiar; a low-density state is treated as novel.

Density is not a count, so the lecturer introduced pseudo-counts. For a discrete empirical distribution, probability equals state count divided by total observations. After observing one more occurrence of a state, both that state's count and the total increase by one. A density model before and after learning the new state can be required to obey analogous equations.

The procedure from Bellemare and colleagues is: fit $p_\theta$ to all states so far, observe $s_i$, update to $p_{\theta'}$ using the enlarged dataset, solve for a pseudo-count $\hat N(s_i)$ from the old and new assigned densities, give the state a decreasing bonus such as $1/\sqrt{\hat N}$, and repeat.

An audience member asked how an unseen state can have nonzero probability. The lecturer said a generalizing density model, such as a VAE-derived likelihood or bound, can assign nonzero density to unseen but similar inputs. The method requires a model that outputs a density; it does not require good generated samples.

Many bonus forms are possible. The original work used an inverse-square-root pseudo-count bonus. In Montezuma's Revenge, pseudo-count exploration let DQN progress through many more levels than ordinary DQN, and modern exploration methods can complete all levels.

### Source reconciliation

Slides 17-19 define

$$
p_\theta(s_i)=\frac{\hat N(s_i)}{\hat n},
\qquad
p_{\theta'}(s_i)=\frac{\hat N(s_i)+1}{\hat n+1}.
$$

Solving gives

$$
\hat n=\frac{1-p_{\theta'}(s_i)}
{p_{\theta'}(s_i)-p_\theta(s_i)},
\qquad
\hat N(s_i)=
\frac{p_\theta(s_i)(1-p_{\theta'}(s_i))}
{p_{\theta'}(s_i)-p_\theta(s_i)}.
$$

The update must increase the assigned probability, $p_{\theta'}(s_i)>p_\theta(s_i)$, for a positive pseudo-count. The transcript's variations of the author's surname are reconciled to Bellemare from the slide citation.

### Additional explanation

Pseudo-counts measure learning progress as much as raw familiarity. A highly generalizing model may increase a state's probability very little because related states already trained it; that produces a large pseudo-count and small novelty bonus. A poorly trained density model can violate the probability-increase condition and yield unstable counts.

## 10. CTS density estimation and random network distillation

**Transcript coverage:** lines 4094-4387

### What the lecturer said - transcript only

The original pseudo-count work used a context-tree-switching (CTS) image model. It factorized pixel probabilities in scan-line order, conditioning a pixel on preceding pixels such as those to its left and above. The lecturer called this model antiquated and declined to dwell on it. The relevant requirement is accurate-enough density evaluation, not high-quality sampling; many other density models could be substituted.

He then moved to a simpler heuristic. A true normalized density is unnecessary if a function can distinguish familiar from novel inputs. Choose a target function $f^*(s,a)$ and train a predictor $\hat f_\theta(s,a)$ only on replay-buffer samples. Prediction error should be low on trained inputs and higher on unfamiliar ones, so squared error can serve as an exploration bonus.

One target is the next state, making dynamics-prediction error a novelty score. An even simpler target is a fixed randomly initialized neural network. Train a second network to match that random target on visited states and use their discrepancy as intrinsic reward. This is random network distillation (RND), which the lecturer described as the most widely used deep-exploration method at present and a strong out-of-the-box starting point.

The lecture ended normally after questions were invited and the class adjourned.

### Source reconciliation

Slides 20-21 summarize the CTS density factorization and Atari results. Slides 22-23 define the error bonus

$$
\mathcal E(s,a)=
\|\hat f_\theta(s,a)-f^*(s,a)\|^2,
$$

with either $f^*(s,a)=s'$ for dynamics prediction or $f^*=f_\phi$ for a fixed random network. The transcript's "R&D" rendering is reconciled to RND from the Burda et al. slide citation.

### Additional explanation

RND avoids explicit density estimation, but prediction error is only a heuristic for visitation. Error can remain high because the predictor lacks capacity, can fall on unseen states through generalization, and can change as optimization proceeds. Observation normalization and control of intrinsic-reward scale are therefore important in practice.

## Consolidated takeaways

1. Model-based offline RL uses pessimistic uncertainty or conservative-value penalties to prevent policies from exploiting learned dynamics.
2. Sparse, delayed rewards make temporally extended behavior difficult to discover even when a human sees obvious semantic clues.
3. Exploration trades immediate use of current knowledge against actions that may reveal a better future strategy.
4. Bayes-optimal exploration is tractable only in small problems; deep RL borrows principles from bandits and tabular MDPs heuristically.
5. A repeated bandit can be treated as a POMDP whose hidden state is the fixed vector of arm reward probabilities.
6. Regret compares an exploring learner with an omniscient best-arm policy.
7. UCB adds an uncertainty bonus to empirical reward and attains logarithmic regret in the analyzed bandit setting.
8. Literal counts fail in image and continuous spaces, motivating density-based familiarity and pseudo-counts.
9. Pseudo-counts convert a density model's probability increase after one observation into an effective count.
10. RND replaces density estimation with prediction error against a fixed random target network.

## Key equations

### Model-uncertainty reward penalty

$$
\tilde r(s,a)=\hat r(s,a)-\lambda u(s,a).
$$

### Bernoulli bandit

$$
r\mid a=i\sim\operatorname{Bernoulli}(\theta_i),
\qquad \theta\sim p(\theta).
$$

### Finite-horizon regret

$$
\operatorname{Reg}(H)
=H\,\mathbb E[r(a^*)]-\sum_{t=1}^{H}r(a_t).
$$

### UCB action rule

$$
a_t=\arg\max_a
\left[
\hat\mu_a+\sqrt{\frac{2\ln H}{N(a)}}
\right].
$$

### MDP reward with an exploration bonus

$$
r^+(s,a)=r(s,a)+\beta B(N(s,a)),
\qquad B'(N)<0.
$$

### Density pseudo-count

$$
\hat N(s)=
\frac{p_\theta(s)(1-p_{\theta'}(s))}
{p_{\theta'}(s)-p_\theta(s)}.
$$

### Random network distillation bonus

$$
r_{\mathrm{int}}(s,a)
=\beta\|\hat f_\theta(s,a)-f_\phi(s,a)\|^2,
$$

where the target-network parameters $\phi$ remain fixed and random.

## Glossary

- **Exploration:** Trying behavior not yet understood in order to discover higher-reward strategies.
- **Exploitation:** Choosing the behavior currently predicted to yield the highest reward.
- **Sparse reward:** A reward signal that occurs only for rare or temporally distant events.
- **Reward shaping:** Adding intermediate feedback using domain knowledge to make a task easier to learn.
- **Multi-armed bandit:** A repeated one-step decision problem with unknown reward distribution for each action.
- **Contextual bandit:** A bandit in which the best arm may depend on an observed context.
- **Bayes-optimal exploration:** The policy maximizing expected return while updating a posterior over unknown environment parameters.
- **Regret:** Reward lost relative to an oracle that always chooses the best action.
- **Optimism in the face of uncertainty:** Treating uncertain choices as if their value were near an upper confidence bound.
- **UCB:** Upper confidence bound, an empirical reward plus an uncertainty bonus.
- **Intrinsic reward:** A learning bonus generated by novelty, uncertainty, or prediction progress rather than the task's external objective.
- **Visitation count:** The number of times a state or state-action pair has been encountered.
- **Density model:** A model assigning normalized probability or density to states or state-action pairs.
- **Pseudo-count:** An effective visitation count inferred from how a density model's probability changes after one observation.
- **CTS:** Context-tree switching, the density estimator used in the original pseudo-count Atari work discussed here.
- **RND:** Random network distillation, using prediction error against a fixed random network as a novelty bonus.
- **MOPO:** Model-based offline policy optimization with a model-uncertainty reward penalty.
- **COMBO:** A model-based offline method using conservative values on model-generated data.

## Self-check questions

1. How do MOPO-style and COMBO-style pessimism differ?
2. Why is Montezuma's Revenge much harder for a reward-driven agent than Breakout?
3. In what sense are temporally extended discovery and exploration-versus-exploitation the same problem?
4. Why can reward shaping help robotics yet fail to eliminate exploration in advertising?
5. What is the hidden state when a bandit is formulated as a POMDP?
6. Why is regret equivalent to negative total reward up to a constant?
7. What behavior does the UCB bonus induce for rarely pulled arms?
8. Why can literal state counts remain one in an Atari or continuous-control task?
9. How do $p_\theta(s)$ and $p_{\theta'}(s)$ determine a pseudo-count?
10. What condition is needed for the pseudo-count to be positive?
11. Why does the density model need evaluable likelihood but not good samples?
12. What are two possible RND-style target functions, and why does fixed random targeting work at all?

## Source coverage checklist

- [x] Transcript lines 1-4387 are assigned once, in monotonic and inclusive ranges.
- [x] The recording ends normally; no source truncation was detected.
- [x] All 23 slide pages were rendered and visually inspected.
- [x] MOPO/COMBO, regret, UCB, intrinsic-bonus, pseudo-count, CTS, and RND equations and diagrams were checked against the slides.
- [x] Breakout, Montezuma's Revenge, Mao, manipulation, tractability-spectrum, bandit, Atari-level, and novelty-error visuals were reconciled.
- [x] Transcript errors in Montezuma, Mao, Markov, Bernoulli, POMDP, Hoeffding, Bellemare, and RND terminology are disclosed and reconciled from the slides or unambiguous technical context.
- [x] Lecturer claims, theoretical scope, heuristic transfer, and additional explanation are kept separate.

**Coverage result:** Complete for the supplied 4,387-line transcript and 23-page slide deck.
