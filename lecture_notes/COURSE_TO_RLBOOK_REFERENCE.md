# CS 185/285 to Sutton and Barto: Reference Guide

This guide maps the Spring 2026 CS 185/285 lecture notes to Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, second edition (the supplied `RLbook2020.pdf`). Use it when a course topic appears and you want the closest treatment in the book.

## How to use this guide

- **Book pages** below are the printed page numbers shown in the book and its table of contents.
- In the supplied PDF, the viewer page is generally **printed page + 22**. For example, book p. 321 is PDF page 343.
- **Direct** means the book treats essentially the same classical idea or algorithm.
- **Foundation** means the book explains useful prerequisites, but the lecture goes materially beyond it.
- **Not covered** means the book is not the right reference for that modern course topic.
- The lectures remain the primary source for the course. The book emphasizes core online RL, especially tabular and linear methods; the course emphasizes modern deep RL.

## Lecture-by-lecture lookup

| Lecture | Course topic | Closest book reference | Match and limits |
|---:|---|---|---|
| [1](lecture_01_introduction.md) | Introduction; RL versus supervised learning; states, actions, rewards; applications | Ch. 1, especially §§1.1-1.4 (pp. 1-7); applications in §§16.5-16.6 (pp. 436-449); future of AI in §17.6 (pp. 475-480) | **Direct** for the classical definition and agent-environment view. The generative-AI and modern deep-RL framing is course-specific. |
| [2](lecture_02_behavioral_cloning.md) | Behavioral cloning; demonstrations; distribution shift; DAgger | §3.1 (pp. 47-52) for the interaction interface; §17.3 (pp. 464-468) for observations versus state | **Not covered** for behavioral cloning, covariate shift, DAgger, and intervention-based imitation. The cited sections only supply state/observation background. |
| [3](lecture_03_behavioral_cloning_part_2.md) | History-based and multimodal policies; autoregressive actions; diffusion and flow matching | §17.3 (pp. 464-468) for partial observability and state construction | **Not covered** for modern imitation models, autoregressive policies, diffusion policies, or flow matching. |
| [4](lecture_04_rl_basics.md) | MDPs; trajectory distributions; objectives; values; policy improvement; RL algorithm families | Ch. 3 (pp. 47-68), especially §§3.1-3.6; generalized policy iteration in §4.6 (p. 86) | **Direct** and the best book companion for the lecture's RL portion. The book is less focused on deep-network implementation and modern algorithm taxonomy. |
| [5](lecture_05_policy_gradients.md) | Likelihood-ratio policy gradient; REINFORCE; reward-to-go; baselines; continuous policies | Ch. 13, especially §§13.1-13.4 (pp. 322-330) and §13.7 (pp. 335-336) | **Direct** for the policy-gradient theorem, REINFORCE, baselines, and continuous-action parameterization. |
| [6](lecture_06_actor_critic.md) | Actor-critic; value baselines; bootstrapping; n-step returns; GAE; off-policy actor-critic | §13.5 (pp. 331-332); TD prediction in §6.1 (pp. 119-123); n-step methods in Ch. 7 (pp. 141-157); λ-returns and TD(λ) in §§12.1-12.2 (pp. 288-294) | **Direct** for basic actor-critic, TD, n-step returns, and eligibility-trace foundations. **Not covered** for GAE, A3C practice, replay-based deep actor-critic, or reparameterized actors. |
| [7](lecture_07_value_based_rl.md) | Policy/value iteration; fitted value and Q iteration; Watkins Q-learning; exploration | Ch. 4 (pp. 73-88); §§6.4-6.6 (pp. 129-134); function approximation in §§9.1-9.4 (pp. 198-209) and §§10.1-10.2 (pp. 243-248) | **Direct** for dynamic programming and tabular TD control. **Foundation** for fitted deep Q methods. |
| [8](lecture_08_q_learning_in_practice.md) | DQN; replay; target networks; multi-step targets; Double Q; clipped double Q; DDPG; divergence | Q-learning and Double Q in §§6.5-6.7 (pp. 131-135); n-step targets in Ch. 7 (pp. 141-157); neural approximation in §9.7 (pp. 223-227); divergence and the deadly triad in §§11.2-11.3 (pp. 260-265); DQN case study in §16.5 (pp. 436-440) | **Direct** for Q-learning, maximization bias, Double Q, experience replay, target networks, and the core instability story. **Not covered** for clipped double Q, DDPG, and most modern debugging practice. |
| [9](lecture_09_advanced_policy_gradients_part_1.md) | Off-policy policy gradients; importance sampling; PPO clipping | Importance sampling in §§5.5-5.9 (pp. 103-115); n-step off-policy corrections in §§7.3-7.4 (pp. 148-151); variance issues in §11.9 (p. 283) | **Foundation** for importance sampling and off-policy correction. **Not covered** for the PPO clipped objective or its implementation. |
| [10](lecture_10_advanced_policy_gradients_part_2.md) | Performance-difference reasoning; KL constraints; natural gradient; TRPO; KL-penalty PPO | Policy-gradient theorem in §13.2 (pp. 324-325); policy improvement/GPI in §§4.2-4.6 (pp. 76-87) | **Foundation only.** The book does not develop the performance-difference bound, Fisher geometry, natural policy gradient, TRPO, or PPO. |
| [11](lecture_11_variational_inference.md) | Latent-variable models; ELBO; KL divergence; EM | No substantial counterpart | **Not covered.** Use the lecture notes or a probabilistic-machine-learning reference. |
| [12](lecture_12_variational_inference_in_rl.md) | Amortized inference; VAE; reparameterization; state-space models; control as inference; soft value iteration | Ch. 8 (pp. 159-188) only for classical planning context | **Not covered** for VI, VAEs, learned latent state-space models, or the control-as-inference derivation. The apparent similarity between soft value iteration and ordinary value iteration should not be treated as equivalence. |
| [13](lecture_13_control_as_inference.md) | Maximum-entropy RL; soft Q-learning; SAC; inverse RL | Basic actor-critic in §13.5 (pp. 331-332); reward design in §17.4 (pp. 469-471) | **Foundation only.** Maximum-entropy control, soft actor-critic, and inverse RL are not treated. |
| [14](lecture_14_rl_with_sequences_and_llms.md) | Adversarial imitation; LLM RL; GAE/GRPO/PPO; preference rewards; POMDP histories | Policy gradients in Ch. 13 (pp. 321-337); importance sampling in §§5.5-5.9 (pp. 103-115); observations and state in §17.3 (pp. 464-468) | **Foundation only.** The book predates RLHF-style post-training, GRPO, preference models, adversarial imitation, and LLM-specific RL. |
| [15](lecture_15_model_based_rl_part_1.md) | Learned dynamics; distribution shift; uncertainty; Bayesian models and ensembles | Models/planning in §8.1 (pp. 159-160); incorrect models in §8.3 (pp. 166-167); expected versus sample updates in §8.5 (pp. 172-173) | **Direct** for the basic model/planning distinction and the risk of a wrong model. **Not covered** for deep ensembles, Bayesian neural networks, and modern epistemic-uncertainty methods. |
| [16](lecture_16_model_based_rl_part_2.md) | Open-loop planning; MPC; random shooting/CEM; Dyna; branched rollouts; latent world models | Ch. 8 (pp. 159-188), especially Dyna §8.2 (pp. 161-165), trajectory sampling §8.6 (pp. 174-176), planning at decision time §§8.8-8.11 (pp. 180-187) | **Direct** for Dyna and classical planning/rollout ideas. **Foundation** for MPC. **Not covered** for CEM practice, short learned-model branches, sequential ELBOs, and modern latent world models. |
| [17](lecture_17_offline_rl_part_1.md) | Offline RL; extrapolation error; policy constraints | Off-policy learning in §§5.5-5.9 (pp. 103-115); divergence and deadly triad in §§11.1-11.3 (pp. 258-265) | **Foundation only.** “Off-policy” in the book is not the same as fixed-dataset offline RL. The divergence material explains why distribution mismatch is dangerous, but the offline-RL formulation and remedies are absent. |
| [18](lecture_18_offline_rl_part_2.md) | BC regularization; AWAC; IQL; CQL; offline-to-online RL; diffusion/flow policies | §§11.1-11.3 (pp. 258-265) for instability background | **Not covered** for the named offline-RL algorithms and offline-to-online transition. |
| [19](lecture_19_exploration.md) | Exploration/exploitation; bandits; regret; optimism; UCB; count and density bonuses; RND | Ch. 2 (pp. 25-42), especially optimistic initialization §2.6 (p. 34) and UCB §2.7 (pp. 35-36); reward-signal design in §17.4 (pp. 469-471) | **Direct** for bandits, optimism, and UCB. **Not covered** for modern high-dimensional pseudo-count methods and random network distillation. |
| [20](lecture_20_rl_theory.md) | Contraction; concentration; simulation lemma; value error; fitted-Q error | Optimal values in §3.6 (pp. 62-66); value iteration §4.4 (pp. 82-84); TD optimality §6.3 (pp. 126-128); prediction objective and semi-gradients §§9.2-9.4 (pp. 199-209) | **Direct** for Bellman optimality and classical convergence intuition. **Not covered in the lecture's form** for concentration inequalities, the simulation lemma, and the finite-sample/approximation-error derivation. |
| [21](lecture_21_midterm_review_part_1.md) | Review: imitation learning and policy gradients | Use the mappings for Lectures 2-5 | Ch. 13 is the principal useful reading. The imitation-learning half has no real counterpart. |
| [22](lecture_22_midterm_review_part_2.md) | Review: PPO, actor-critic, Q-learning, VI, maximum-entropy RL, LLM RL, model-based RL | Use the mappings for Lectures 6-16 | The strongest matches are Chs. 6-8 and 13. PPO, VI, maximum-entropy RL, and RLHF remain outside the book. |
| [23](lecture_23_advanced_exploration.md) | Unsupervised behavior discovery; entropy; mutual information; empowerment; DIAYN; Skew-Fit | General value functions and auxiliary tasks in §17.1 (pp. 459-460); reward design in §17.4 (pp. 469-471) | **Foundation only.** Empowerment, mutual-information skill discovery, DIAYN, goal-conditioned empowerment, and Skew-Fit are not covered. |
| [24](lecture_24_multi_task_and_hierarchical_rl.md) | Multi-task and goal-conditioned RL; HER; successor representations/features; options | General value functions and auxiliary tasks in §17.1 (pp. 459-460); temporal abstraction/options in §17.2 (pp. 461-463) | **Direct** for options and their Bellman/model view. **Not covered** for HER, successor features, generalized policy improvement for transfer, and modern multi-task training. |
| [25](lecture_25_challenges_and_open_problems.md) | Algorithm atlas; simulator exploitation; domain randomization; AlphaGo; continual/reset-free learning; open problems | Wrong models §8.3 (pp. 166-167); AlphaGo/AlphaGo Zero §16.6 (pp. 441-449); remaining issues §17.5 (pp. 472-474); RL and AI §17.6 (pp. 475-480) | **Direct** for the AlphaGo case study and the book's own frontier perspective. The course's modern simulator, continual-learning, reset-free, offline-to-online, and LLM examples are not covered. |

## Topic-first index

Use this table when you know the concept but not the lecture number.

| Topic encountered in the course | Go to these book sections first |
|---|---|
| RL problem, agent-environment loop, reward, policy | §§1.1-1.4 (pp. 1-7), then §3.1 (pp. 47-52) |
| Episodes, continuing tasks, return, discounting | §§3.3-3.4 (pp. 54-57) |
| Value functions and Bellman equations | §§3.5-3.6 (pp. 58-66) |
| Policy evaluation and policy improvement | §§4.1-4.3 (pp. 74-81) |
| Value iteration and generalized policy iteration | §§4.4-4.6 (pp. 82-87) |
| Monte Carlo prediction/control | Ch. 5, especially §§5.1-5.4 (pp. 92-102) |
| Importance sampling and off-policy correction | §§5.5-5.9 (pp. 103-115), then §§7.3-7.4 (pp. 148-151) |
| TD learning and bootstrapping | §§6.1-6.3 (pp. 119-128) |
| SARSA, Q-learning, Expected SARSA | §§6.4-6.6 (pp. 129-134) |
| Maximization bias and Double Q-learning | §6.7 (pp. 134-135) |
| n-step returns | Ch. 7 (pp. 141-157) |
| Eligibility traces and λ-returns | §§12.1-12.2 (pp. 288-294); control and off-policy traces in §§12.7-12.11 (pp. 303-315) |
| Function approximation and semi-gradients | §§9.1-9.4 (pp. 198-209) |
| Neural-network value approximation | §9.7 (pp. 223-227) |
| Off-policy instability / deadly triad | §§11.1-11.3 (pp. 258-265) |
| REINFORCE and policy-gradient theorem | §§13.1-13.4 (pp. 322-330) |
| Basic actor-critic | §13.5 (pp. 331-332) |
| Continuous-action policy parameterization | §13.7 (pp. 335-336) |
| DQN, experience replay, and target networks | §16.5 (pp. 436-440), supported by §§6.5-6.7 and §11.3 |
| Models, planning, and Dyna | §§8.1-8.3 (pp. 159-167) |
| Search, rollouts, and MCTS | §§8.8-8.11 (pp. 180-187) |
| Exploration/exploitation, optimism, UCB | Ch. 2 (pp. 25-42), especially §§2.6-2.7 (pp. 34-36) |
| Partial observability and state construction | §17.3 (pp. 464-468) |
| Auxiliary tasks / general value functions | §17.1 (pp. 459-460) |
| Hierarchical RL and options | §17.2 (pp. 461-463) |
| Reward design | §17.4 (pp. 469-471) |
| Atari/DQN application | §16.5 (pp. 436-440) |
| AlphaGo and AlphaGo Zero | §16.6 (pp. 441-449) |

## Major course topics for which this book is not the main reference

Do not spend time searching this book for detailed treatments of the following:

- behavioral cloning, DAgger, adversarial imitation learning, or modern diffusion/flow policies;
- PPO, TRPO, natural policy gradient, GAE, DDPG, TD3-style clipped double Q, or SAC;
- variational inference, VAEs, control as inference, maximum-entropy RL, or inverse RL;
- RLHF, preference models, GRPO, process rewards, or LLM post-training;
- modern learned world models, CEM-based MPC, ensemble uncertainty, or latent-state deep models;
- offline RL algorithms such as AWAC, IQL, CQL, MOPO, IDQL, or FQL;
- count-density exploration, RND, empowerment, DIAYN, or Skew-Fit;
- HER, successor features, and most modern multi-task or transfer-RL algorithms.

For those topics, the corresponding lecture note is the better starting point; follow its named algorithms and papers rather than trying to infer a treatment from a superficially related classical chapter.

## A compact reading strategy

If reading alongside the course rather than looking up one isolated topic:

1. Read Chs. 1 and 3 with Lectures 1 and 4.
2. Read Ch. 13 with Lectures 5-6.
3. Read Chs. 4, 6, 7, 9, and 11 selectively with Lectures 7-10.
4. Read Ch. 8 with Lectures 15-16.
5. Read Ch. 2 with Lecture 19.
6. Read §§17.1-17.4 with Lectures 23-24.
7. Treat Lectures 2-3, 11-14, and 17-18 as course-note/paper territory rather than book territory.
