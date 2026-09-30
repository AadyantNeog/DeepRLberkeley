---
title: "Lecture 25 - Challenges and Open Problems"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 25
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 25, Challenges and Open Problems.txt"
source_slides: "../lectures/Lecture 25 - Challenges and Open Problems.pdf"
transcript_lines: 4261
slide_pages: 44
status: "complete"
---

# Lecture 25: Challenges and Open Problems

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Course logistics, review, and the RL algorithm atlas | lines 1-223 |
| 2 | Why RL has many algorithms and two meanings of “reinforcement learning” | lines 224-411 |
| 3 | One formalism, very different practical constraints | lines 412-639 |
| 4 | Three perspectives on RL and the engineering-tool view | lines 640-882 |
| 5 | Control as inversion of a mathematical model | lines 883-1086 |
| 6 | Simulator exploitation and domain randomization | lines 1087-1347 |
| 7 | Anything simulatable is controllable: AlphaGo as an example | lines 1348-1545 |
| 8 | LLM sandboxes, verification, and simulation-to-real systems | lines 1546-1758 |
| 9 | Real-world RL and Moravec's paradox | lines 1759-1944 |
| 10 | Easy universes, hard universes, and the limits of written rules | lines 1945-2133 |
| 11 | Open-world messiness and autonomous discovery | lines 2134-2331 |
| 12 | Requirements for learning in a hard universe | lines 2332-2568 |
| 13 | Continual learning, resets, and prior experience | lines 2569-2751 |
| 14 | Reset-free robot learning and offline-to-online adaptation | lines 2752-2943 |
| 15 | The real world as its own simulator and the universal-learning view | lines 2944-3231 |
| 16 | Large low-quality datasets and the limitation of estimating $p(x)$ | lines 3232-3429 |
| 17 | Data as knowledge about possibilities; learning as decision-making | lines 3430-3621 |
| 18 | A universal RL recipe and the dialogue-agent example | lines 3622-3852 |
| 19 | Model-based RL for multi-turn dialogue | lines 3853-4044 |
| 20 | Dialogue results and the lecture's synthesis | lines 4045-4261 |

## 1. Course logistics, review, and the RL algorithm atlas

**Transcript coverage:** lines 1-223

### What the lecturer said - transcript only

This was the lecturer's final regular lecture and was intended as a nontechnical retrospective on the course: what the class had learned, what broader perspectives could be taken from it, and why reinforcement learning looks the way it does.

Kevin Frans would give a one-hour guest lecture on RL and large language models in industry on Friday, beginning at 9:00 rather than 8:00. Head TA Sihong Park would give the following Wednesday's guest lecture on offline and offline-to-online RL, including perspectives that could differ from the lecturer's own. There would be no lecture on the final Friday, May 1, so students could focus on their final projects.

The course material could be organized as an atlas of methods for learning-based control. At the highest level, it separated into:

- **imitation learning**, which learns a policy that copies behavior present in data;
- **reinforcement learning**, which seeks an optimal policy.

The model-free RL branch included policy gradients, value-based methods, actor-critic methods, Q-learning, Q-function actor-critic algorithms, and more advanced practical policy-gradient methods. Model-based methods included planning without a learned policy, such as random shooting, and methods that also learn a policy. Related topics included exploration, unsupervised RL, goal and skill discovery, control as probabilistic inference, and inverse RL at the intersection of imitation learning and RL.

### Source reconciliation

Slides 1-3 provide the title, concise course review, and algorithm map. The transcript supplies the end-of-course schedule and expands every branch of the map.

### Additional explanation

The atlas is better understood as a map of assumptions than as a list of interchangeable optimizers. Each branch chooses where knowledge comes from—demonstrations, rewards, a model, or interaction—and which constraints matter most.

## 2. Why RL has many algorithms and two meanings of “reinforcement learning”

**Transcript coverage:** lines 224-411

### What the lecturer said - transcript only

A natural question is why RL needs so many algorithms rather than one universal recipe analogous to gradient descent and backpropagation in supervised learning. The answer is that RL methods address several practically different problems. Although those problems share nearly the same mathematical abstraction, their operating constraints can be very different.

The term *reinforcement learning* itself has two meanings. Historically, it arose from psychology as a particular account of adaptation in animals: behavior changes in response to rewards, punishments, and good or bad outcomes. Temporal-difference learning can model such behavioral change.

In computer science, RL also names a problem definition—maximizing reward through sequential decisions. Under that usage, any method that solves the problem may be called an RL method, even if it does not resemble the psychological learning mechanism.

This distinction frequently causes misunderstanding, but it is not the main reason so many RL algorithms exist. The deeper reason is variation in practical constraints that the common formal definition does not fully expose.

### Source reconciliation

Slides 4-5 pose the “why so many algorithms?” question and revisit the two descriptions of RL from Lecture 1.

### Additional explanation

The two meanings distinguish a **mechanism** from an **objective class**. “The agent learns by reinforcement” describes how adaptation happens; “this is an RL problem” describes what is being optimized. The same objective can be solved through very different mechanisms.

## 3. One formalism, very different practical constraints

**Transcript coverage:** lines 412-639

### What the lecturer said - transcript only

The standard RL formalism has a state as input at each time step, an action as output, and trajectories as data. Problems as different as training a dog, making a quadruped run, and managing warehouse inventory can all be cast this way. A common definition makes the problem “small” in the sense that one abstraction covers many settings and can motivate very general algorithms.

That abstraction does not tell the whole practical story. Even seemingly similar applications may have incompatible constraints. A quadruped locomotion system was trained with PPO in simulation, using randomization and system identification, and then transferred directly to the real robot. A coffee-making robot instead used no simulator; it combined human demonstrations with an offline-to-online RL method.

Coffee is difficult to simulate accurately, but humans can demonstrate the task. A human cannot directly demonstrate quadrupedal motor control, while its rigid-body mechanics can be simulated. Thus, two robotic systems that share the same broad RL formulation require different data sources and algorithms.

### Source reconciliation

Slides 6-10 revisit the state-action interface, early application examples, locomotion, kitchen manipulation, and RL for language models. The transcript identifies the different training constraints behind superficially similar robot examples.

### Additional explanation

The formal MDP suppresses operational facts such as whether resets are cheap, whether demonstrations exist, whether a reward can be verified, and whether interaction is safe. Those omitted facts often determine the algorithm more strongly than the state and action notation does.

## 4. Three perspectives on RL and the engineering-tool view

**Transcript coverage:** lines 640-882

### What the lecturer said - transcript only

Language-model training introduces still another constraint regime: it uses much larger-scale and often highly suboptimal datasets. Across all these domains, the same conceptual foundations may apply, but the practical circumstances differ substantially.

The lecturer proposed three perspectives on RL:

1. an engineering tool for optimal control;
2. a method for solving hard problems through real-world interaction;
3. a general, potentially universal perspective on machine learning.

The most pragmatic perspective treats RL as an engineering tool for optimal control. This departs from the picture of an animal receiving rewards and punishments. A simulated quadruped is not principally trained by someone giving it a “treat”; most of the work is in constructing a simulation that represents many locomotion conditions.

Traditional control engineering provides the comparison. To control a rocket, engineers write a mathematical description—often differential equations—of how its state responds to previous controls, then invert that description to find controls that produce a desired state. Modern aerospace achievements, including the Apollo program, depended on optimal control and probabilistic estimation of this sort.

### Source reconciliation

Slides 11-14 introduce the three perspectives, contrast the popular reward-and-punishment picture with the engineering view, and use rocket control as the motivating example.

### Additional explanation

Under this perspective, reward is a specification inside a designed computational system, not necessarily a natural teaching signal. RL automates controller synthesis after engineers have supplied the world in which synthesis occurs.

## 5. Control as inversion of a mathematical model

**Transcript coverage:** lines 883-1086

### What the lecturer said - transcript only

Traditional control begins with a mathematical dynamics model and analytically derives the control law that drives the system toward a desired state. This is an exercise in differential equations and optimal control rather than machine learning, and it has worked for aircraft and spacecraft.

Simulation-to-real robot learning follows the same high-level pattern. Someone first writes a rigid-body dynamics model and implements it as a simulator. The RL algorithm then learns how to invert that model: it discovers controls that make the simulated robot enter desired states.

Previously, an engineer wrote equations and spent substantial effort at a board deriving a controller. Now an engineer writes equations, implements a simulator, and runs RL to derive the controller. The output plays the same role.

The major benefit is that the model can be extremely complicated. A hand-derived controller encourages simplifying the equations until they are analytically manageable. An RL optimizer can instead work with an inscrutable simulator—potentially involving complex finite-element calculations—and thereby make controller synthesis useful across many areas of science and engineering.

### Source reconciliation

Slide 15 summarizes the characterize-simulate-control sequence. The transcript explains precisely what “inverting” the model means and why numerical RL permits richer models than board-derived control laws.

### Additional explanation

Let a simulator define a transition model $p(s_{t+1}\mid s_t,a_t)$. Optimizing a policy in that model approximately solves an inverse question: which sequence of actions makes the forward dynamics generate desirable states? The inverse need not have a closed-form expression.

“Inverting a model” is an analogy for solving a constrained optimization problem. A dynamics model need not be invertible: some target states are unreachable, many controls can lead to the same state, and stochastic transitions prevent prescribing exact outcomes. Traditional control also uses numerical optimization, MPC, and robust/adaptive methods; it is not limited to hand-derived analytic controllers.

## 6. Simulator exploitation and domain randomization

**Transcript coverage:** lines 1087-1347

### What the lecturer said - transcript only

In response to a question about simulator design, the lecturer contrasted traditional optimal control with RL. A control engineer expects to invert the model manually and therefore writes comparatively simple equations. If RL performs the optimization, the simulator can be much more complex and there is little incentive to simplify it.

This creates a form of overfitting in which the engineer overfits the simulator rather than merely the learning algorithm overfitting data. RL is extremely good at exploiting flaws. Early deep-RL humanoids could exploit imperfections in simulated frictional contact—for example, rapidly making and breaking contact to inject nonphysical “phantom” energy and fly forward with high simulated reward.

The simulator must therefore be difficult to exploit. A common technique is randomization. Instead of training in one deterministic simulator, train one policy across an ensemble of simulators, usually produced by varying the parameters of a shared simulator. If a policy controls all these slightly different systems, it may be robust to the discrepancy between simulation and reality.

The analogy was to designing one controller for a distribution of slightly different airplanes rather than one exact airplane. Robust performance across the distribution can encompass the unknown real system.

### Source reconciliation

The spoken question-and-answer material expands slide 16's warning that a simulator is both the power and the weakness of this approach.

### Additional explanation

Domain randomization turns model uncertainty into a training distribution:

$$
\max_\theta\;\mathbb E_{\xi\sim p(\xi),\,\tau\sim p_{\theta,\xi}(\tau)}[R(\tau)],
$$

where $\xi$ indexes masses, friction, delays, or other simulator parameters. This optimizes average performance under the training distribution, not a worst-case robustness guarantee. Even a represented but rare failure mode can be sacrificed to improve the average. Coverage, weighting, simulator fidelity, and optimization quality all matter; omitted failure modes remain exploitable.

## 7. Anything simulatable is controllable: AlphaGo as an example

**Transcript coverage:** lines 1348-1545

### What the lecturer said - transcript only

If RL supplies a general-purpose way to invert a simulator, then—within practical limits—anything that can be simulated can be controlled. Training occurs “inside the matrix.” The workflow is close to conventional optimal control: characterize a system, simulate it, and derive a controller. RL replaces manual derivation with automatic inversion of the mathematical model.

The weakness is that the output depends on the model supplied. The algorithm need not observe real-world data at all, so engineers remain responsible for including every relevant physical phenomenon in the simulator. The lecturer said that most current industrial uses of RL follow this simulation-centered principle.

AlphaGo is a famous example. Its first version used real game data for imitation-learning initialization, while AlphaZero did not. Apart from that initialization, training relied on simulation, including simulated opponents. Board games are especially suitable because their rules are simple and known. Older copies of the agent can serve as opponents, producing a simulator of increasingly strong games.

In that sense, no new physical observation of the outside world is needed. Nevertheless, the resulting system can play Go better than the strongest human players.

### Source reconciliation

Slides 16-17 state the simulation-control principle and show AlphaGo. The transcript distinguishes the first AlphaGo's imitation bootstrap from later self-play training.

### Additional explanation

Self-play solves a special modeling problem: the most important changing component of a competitive environment is the opponent. Training against a population of prior policies makes that component endogenous to learning.

The slogan “anything simulatable is controllable” is not a controllability theorem. Accurate simulation permits evaluating candidate behavior but does not ensure reachability, tractable search, adequate exploration, or reliable optimization. Self-play also need not converge to a strong equilibrium without suitable game/learning conditions.

## 8. LLM sandboxes, verification, and simulation-to-real systems

**Transcript coverage:** lines 1546-1758

### What the lecturer said - transcript only

RL for large language models is implemented in several ways and can use real experience, but a powerful technique fine-tunes an agent on downstream tasks inside a sandbox. For mathematics or coding, the sandbox supplies problems and a way to check outcomes. This is the language counterpart of training a walking robot in simulation: the capability learned is constrained by the problems placed in the sandbox.

Not every desired capability is easy to simulate or verify. Training an agent to calm an irritated customer can be harder than training one to write code, because code can be executed and checked in a sandbox while a genuine customer interaction cannot be reproduced or verified so cleanly. This creates a practical gulf even for very large models: much successful RL training still occurs in simulation.

Simulation-trained policies nevertheless affect the real world. The lecturer cited a prominent Spring Festival demonstration of Unitree robots whose policies were trained in simulation. Whenever a useful sandbox or simulator can be constructed, this simulation-first approach is currently a pragmatic tool of choice.

### Source reconciliation

Slides 18-20 connect RL for LLMs, simulation-to-real robots, and the claim that simulatable systems can be controlled.

### Additional explanation

The bottleneck is often not generating candidate actions but building a trustworthy evaluator. Code tests, game rules, and formal proofs provide unusually crisp reward channels; social outcomes are delayed, ambiguous, and affected by unobserved human state.

Passing tests is evidence for the properties those tests cover, not general program correctness. A proof checker validates a formal statement within its assumptions; it does not establish that the statement captured the intended task. Sandbox realism and verifier quality remain separate from optimization success.

## 9. Real-world RL and Moravec's paradox

**Transcript coverage:** lines 1759-1944

### What the lecturer said - transcript only

The second perspective asks about problems that are difficult to simulate and for which RL is harder to apply, yet alternatives may be even less capable. Such real-world learning imposes a different set of constraints and motivates different algorithms.

A computer defeated world chess champion Garry Kasparov in 1996 using tree search, and an RL system defeated a Go world champion in 2016. In both cases, however, a human on the opposite side of the board physically moved the pieces. The computer could solve an activity regarded as cognitively demanding but could not handle the apparently simple physical interaction.

This illustrates Moravec's paradox. Humans are exceptionally capable in perception and motor control and therefore make those difficult problems look easy. Abstract thought is evolutionarily newer and less naturally mastered, so it feels difficult even when its computational rules are comparatively tractable. A related formulation is that many supposedly hard problems are easy for machines while abilities associated with a young child—recognizing faces, lifting a pencil, walking, or answering questions—pose extraordinarily hard engineering problems.

Our judgment of difficulty is biased by what human brains evolved to do well. Chess, Go, and rocket-control derivations are not native human specializations; face perception and conversation are.

### Source reconciliation

Slides 21-23 use the chess/Go photographs and quotations associated with Moravec's paradox. The transcript makes clear that the paradox is about mismatched notions of difficulty, not simply about robot hardware.

### Additional explanation

Tasks with compact rules can still require enormous search, but computation can scale against that search. Open-ended perception and action instead require the system to infer which rules and exceptions are relevant in the first place.

Historical correction: Deep Blue won a game against Kasparov in 1996, but Kasparov won that match. Deep Blue won the rematch in **1997**. See [IBM's Deep Blue history](https://www.ibm.com/history/deep-blue). The 1996 wording above preserves the lecture account and should not be learned as the match-victory date.

## 10. Easy universes, hard universes, and the limits of written rules

**Transcript coverage:** lines 1945-2133

### What the lecturer said - transcript only

Moravec's paradox can be interpreted as a statement about the physical universe. An **easy universe** has clean, compact rules with few exceptions—for example, chess or the ballistic trajectory of a rocket. Once its rules are understood, the remaining task is largely inversion or optimization. Such a problem can require considerable mathematics without being difficult in the open-world sense.

A **hard universe** contains complicated patterns, exceptions, and diverse phenomena that resist a concise written description. Social interaction and open physical environments are examples. One cannot simply enumerate equations that cover everything that might occur; a learning system must acquire patterns from experience.

The underlying reason some problems are easier to automate is therefore how completely people can characterize them with rules they can write down.

### Source reconciliation

Slides 24-26 develop the remote-ship example and the easy-universe/hard-universe contrast. The transcript supplies the criterion: whether concise known rules characterize what can happen.

### Additional explanation

“Easy” here does not mean small or computationally cheap. It means *closed*: the variables, laws, and success criteria are sufficiently specified. A hard universe is open to unmodeled events, so correctness cannot be reduced to optimization inside a fixed specification.

The easy/hard-universe terminology is the lecturer's informal modeling distinction, not a complexity classification. Compact known rules can still define computationally intractable problems, and physical control can remain difficult even with accurate equations.

## 11. Open-world messiness and autonomous discovery

**Transcript coverage:** lines 2134-2331

### What the lecturer said - transcript only

It is algorithmically straightforward to navigate a container ship around the world if sensing is available, yet such ships still carry crews. The crew is not principally there to solve the nominal navigation problem; people are needed when something breaks in the engine room or another unforeseen physical situation requires a flexible response. Long-horizon planning is not the main difficulty. Dealing with the mess is.

An open-world system must produce reasonable behavior when something occurs for which it was not explicitly designed. The lecturer used survival on an unfamiliar desert island as an illustration. The agent does not know what resources it will find. It must combine prior skills and knowledge with creative problem solving, explore the environment, identify available resources, and assemble a new solution.

The setting provides minimal external supervision, unexpected situations, and the need for autonomous discovery. The system also has to remain robust long enough to learn; real-world interaction proceeds on a clock and failures can end the opportunity to improve.

### Source reconciliation

Slides 24-27 connect the ship example to unexpected events and enumerate the main real-world questions.

### Additional explanation

Nominal control optimizes performance on anticipated trajectories. Open-world competence additionally requires *recovery*: detecting that the current model is inadequate, gathering information safely, and preserving the ability to keep acting.

## 12. Requirements for learning in a hard universe

**Transcript coverage:** lines 2332-2568

### What the lecturer said - transcript only

People handle open-world adaptation well, whereas current AI systems are extremely poor at it. Contemporary systems are strongest when they can lean on training data and reproduce familiar patterns. Large datasets make that ability powerful, but discovering genuinely new solutions remains limited.

In principle, RL can discover new solutions. It does not yet do so well outside simulatable environments, but other paradigms cannot directly optimize novel behavior through consequences at all. This leaves a class of unsolved AI problems for which RL is a plausible foundation.

The lecturer contrasted the two universes:

- In an **easy universe**, the rules are known, simulation is plentiful, high reward is clearly defined, and the main question is how well an objective can be optimized.
- In a **hard universe**, merely succeeding matters more than small differences in optimality. The agent inhabits an open world, learns from its own experience, lacks a faithful simulator, and must generalize and adapt.

The hard setting makes reward specification less obvious. A video-game score is available repeatedly. “Survive on the desert island” is not a useful trial-and-error signal if the first failure prevents a second trial. The agent needs feedback and behavior that preserve future learning opportunities.

### Source reconciliation

Slides 25-27 contrast current AI and human adaptability, then list goal specification, autonomous learning, robustness, generalization, and exploration as real-world requirements.

### Additional explanation

This is a distinction between maximizing expected return in a stationary experiment and maintaining **viability** during learning. In the latter, exploration has an irreversible cost, so safety and resetability are part of the learning problem rather than external conveniences.

These are motivating research judgments rather than impossibility results about other learning paradigms. Supervised models can generalize and planning/search can optimize consequences; RL is one framework for combining experience and sequential utility. A survival reward is mathematically definable even when safe learning from repeated failure is impossible.

## 13. Continual learning, resets, and prior experience

**Transcript coverage:** lines 2569-2751

### What the lecturer said - transcript only

In a board game, losing is followed by an automatic reset. In continual interaction, a failed attempt can leave the world in a new state. If an attempt to chop down a tree fails, the agent must determine how to restore the conditions for another attempt. Reset behavior therefore has to be learned as part of the task.

The system must remain robust as the world changes. It should use past experience not merely to repeat old behavior but to build knowledge about what is possible and synthesize new solutions. Prior experience must also bootstrap exploration intelligently.

The lecturer's upcoming examples used robotics because that was his research area and because embodied agents make the analogy to human learning intuitive. The issues were not robot-specific. Inventory management must learn from real purchasing patterns. A medical assistant facing an unexpected crisis, such as a pandemic, would need to combine prior knowledge with adaptation from new experience. Both domains share the same constraints of real-world learning.

### Source reconciliation

Slide 27 lists the general open-world questions; the transcript broadens them beyond robotics to inventory and medicine.

### Additional explanation

Resetability is a hidden assumption in many benchmark MDPs. Removing it converts episodes into one continuing trajectory and couples task performance to state-distribution management: the agent must deliberately keep itself in states from which useful learning remains possible.

## 14. Reset-free robot learning and offline-to-online adaptation

**Transcript coverage:** lines 2752-2943

### What the lecturer said - transcript only

The first robot example asked whether a hand could learn to pick up and reorient an object without simulation or human intervention. If it dropped the object, it had to recover and try again autonomously.

One solution is multi-task learning with a repertoire of skills. Failure at one skill becomes practice for another. Spilling coffee while learning to use a coffee machine, for example, creates an opportunity to practice cleaning. In the robot demonstration, early behavior mostly moved the object across the table because those skills were easiest. Once the robot learned to pick it up, it could practice reorientation. After a drop, it returned to pickup behavior. The repertoire supported uninterrupted improvement.

The coffee-making system represented a different real-world constraint regime. It did not start from scratch: human demonstrations provided extensive prior experience, and real-world RL trials fine-tuned the strategy. It was therefore an offline-to-online approach rather than autonomous skill discovery from an empty initialization.

### Source reconciliation

Slides 28-29 show reset-free autonomous manipulation and the real-data kitchen robot. The transcript explains the behavioral curriculum visible in the videos.

### Additional explanation

The multi-task repertoire closes a state-recovery loop:

$$
\text{failure state for task A}
\longrightarrow
\text{training state for task B}
\longrightarrow
\text{state where task A can be retried}.
$$

This replaces an external reset operator with learned transition competence.

A recovery repertoire only replaces resets inside states from which some available behavior can recover. An object dropped out of reach or an irreversible failure can still require external intervention; reset-free training is not equivalent to unlimited autonomous recoverability.

## 15. The real world as its own simulator and the universal-learning view

**Transcript coverage:** lines 2944-3231

### What the lecturer said - transcript only

Real-world RL has the advantage that the world simulates itself: there is no simulator-design error because interaction occurs with the actual system. Its constraints, however, are severe. Sample efficiency matters much more; reward specification, resets, and robustness become central; and solving them may be necessary for continual or self-improving systems. A system might be initialized in a sandbox and then continue learning in reality, but this remains at the frontier of current methods.

The third, more philosophical perspective treats RL as a universal way to think about machine learning. RL tries to find a solution better than any behavior in the available experience. The data show what is possible; the algorithm determines what is desirable. In supervised learning, by contrast, the target is to reproduce what appears in the examples.

Modern deep learning's central lesson is that large models, large computation budgets, and large datasets are powerful. Early datasets depended heavily on curated human labels. At scale, that labor is insufficient, so current systems commonly use enormous quantities of low-quality data to acquire knowledge and a smaller quantity of high-quality data for post-training. Language, image, and video generative models follow this pattern.

### Source reconciliation

Slides 30-32 close the real-world perspective and open the universal-learning perspective with the large-model, large-data recipe.

### Additional explanation

The universal-learning claim is not that every current ML pipeline should literally be replaced with online RL. It is that decision quality supplies a more general final criterion than resemblance to data, while prediction, imitation, and modeling can remain components inside the decision learner.

Direct real-world samples avoid simulator mismatch for the situations actually observed. They still leave sampling error, partial observability, nonstationarity, and learned-model/value error. Neither RL nor large datasets guarantee a policy better than every demonstrated behavior; unsupported actions remain uncertain, particularly offline.

## 16. Large low-quality datasets and the limitation of estimating $p(x)$

**Transcript coverage:** lines 3232-3429

### What the lecturer said - transcript only

Large-scale pretraining usually uses unsupervised objectives such as next-token prediction or generative modeling. These methods estimate the data distribution, written as $p(x)$. A language model trained on text estimates patterns in text sequences; an image model estimates patterns in the images people place online.

This creates tension when the desired resource is a large amount of low-quality data. If the learning objective models the data distribution, then undesirable properties of that distribution are learned too. Web text reflects what people happened to type and publish, not an ideal distribution of useful actions.

This tension contributes to the difficulty of aligning language models. A system receives huge quantities of uncurated pretraining data and must later be adapted to particular desirable tasks. Practitioners work hard to identify bad tendencies and suppress them with post-training patches. The process is labor-intensive and fragile because an unexpected prompt may expose a behavior that was never corrected.

### Source reconciliation

Slide 33 asks how to reduce the supervision burden and how low-quality data can become a source of knowledge rather than a behavioral target.

### Additional explanation

Maximum-likelihood pretraining rewards fidelity to observed frequency:

$$
\max_\theta\;\mathbb E_{x\sim p_{\text{data}}}
[\log p_\theta(x)].
$$

That objective is valuable for learning structure, but it does not by itself distinguish common behavior from desirable behavior. A utility or preference signal supplies a separate axis.

## 17. Data as knowledge about possibilities; learning as decision-making

**Transcript coverage:** lines 3430-3621

### What the lecturer said - transcript only

Instead of treating a huge low-quality dataset as examples of what the system should do, one could treat it as knowledge about what may happen. Reinforcement learning could reflect on those possibilities and choose behavior according to desirable outcomes.

The lecturer then asked why machine learning—or even a brain—is needed. Neuroscientist Daniel Wolpert argued that the brain exists to produce adaptable, complex movement, because movement is how an organism affects the world. Although Wolpert studies motor control and therefore has a particular perspective, the lecturer saw a deeper point: computation matters only when its outputs are useful.

Accordingly, the purpose of machine learning may be to produce adaptable, complex decisions. An AI system's internal reasoning is not useful if it produces a bad final decision, and without any decision it does nothing. Even an image classifier deployed in an application makes a consequential decision—for example, detecting a red-light violation or identifying a face in a photograph. Its prediction leads to an action with a good or bad outcome.

All such applications can therefore be viewed as decision-making problems, and RL is the formalism for optimizing decisions for high utility.

### Source reconciliation

Slides 34-35 move from the purpose of brains and machine learning to the proposal that experience should be used for decision-making rather than copied as supervision.

### Additional explanation

Prediction becomes decision-making once an output changes what happens next. The relevant loss is then not merely classification error but the downstream cost of false positives, false negatives, delays, and subsequent actions.

Decision theory also includes one-step supervised decisions and contextual bandits. Viewing a prediction through downstream utility does not by itself require a multi-step RL algorithm; sequential RL becomes necessary when present actions affect later states, information, or opportunities.

## 18. A universal RL recipe and the dialogue-agent example

**Transcript coverage:** lines 3622-3852

### What the lecturer said - transcript only

Under the universal-learning view, RL is an engine that consumes heterogeneous, low-quality experience as information about possibilities and uses it to improve decisions. A notional recipe is:

1. collect a very large and diverse dataset containing behavior from many tasks;
2. run a general RL procedure, perhaps offline because stored data are easier to obtain than interactive environments;
3. learn a model, reusable skills, a goal-conditioned policy, or another representation that uses experience to achieve outcomes rather than merely copy actions;
4. adapt that functional understanding to downstream tasks with relatively little additional interaction.

The lecturer made the proposal concrete with a dialogue tutoring agent. Standard RL for LLMs often resembles a bandit: the model receives one math problem and produces one answer. A tutor instead interacts over multiple turns. When asked to explain behavioral cloning, a good tutor might first ask about the student's background rather than immediately output a large block of information. The question calibrates the lesson to the learner.

Current assistants often produce a one-step information dump because the interaction is framed as a single-step response problem.

### Source reconciliation

Slides 36-37 show the diverse-data recipe and contrast a targeted tutoring question with an immediate long answer.

### Additional explanation

The multi-turn objective values information-gathering actions. An early question may have little immediate reward but improve later teaching decisions by reducing uncertainty about the student's knowledge.

A single response can be a contextual-bandit action at the conversation level while token generation is a multi-step process inside that action. Multi-turn dialogue adds user responses and information gathering between assistant actions. The choice of action granularity determines the horizon.

## 19. Model-based RL for multi-turn dialogue

**Transcript coverage:** lines 3853-4044

### What the lecturer said - transcript only

A dialogue agent can be trained using a model-based RL procedure in which the language model itself acts as a simulator. The model generates synthetic hypothetical conversations. Because it was pretrained on human text, it can produce plausible exchanges, though plausibility does not mean the exchanges are good.

An offline RL method can examine these possible conversations and learn which utterances tend to produce valuable later outcomes—for example, a student who can correctly answer assessment questions after the lesson. The learner reflects on experience and optimizes a reward rather than directly prompting the base model to imitate its own samples.

In this construction, the LLM serves both as a simulated environment and as the policy being optimized. The lecturer attributed the example to Berkeley PhD student Joey Hong and compared the learned agent with a heavily prompt-engineered ChatGPT baseline.

### Source reconciliation

Slide 38 diagrams model-based RL through imagined conversations; slide 39 presents the baseline and learned dialogue examples.

### Additional explanation

The simulator-policy dual role creates model bias: if imagined students respond unrealistically, the tutor can optimize an artifact of the language model. The same simulator-exploitation concern from robot control therefore reappears in dialogue, although the simulated object is human language rather than rigid-body physics.

## 20. Dialogue results and the lecture's synthesis

**Transcript coverage:** lines 4045-4261

### What the lecturer said - transcript only

The prompt-engineered baseline was instructed to ask clarifying questions and assess student experience, yet still produced a relatively long response that a casual learner might ignore. The RL-trained agent's replies were shorter, more targeted, and more interactive. It first asked whether the learner had encountered AI and, after a negative answer, moved to an even more basic question about using a computer or smartphone.

This information-seeking behavior is what RL should encourage: actions need not greedily maximize immediate response quality if they lead to better outcomes over a long interaction.

The lecturer concluded that RL has many methods because its application settings impose different constraints:

- In a **simulator or sandbox**, parallelism, stable optimization, and final performance may matter more than sample efficiency; practical policy-gradient methods are natural tools.
- In the **real world**, continual learning and high sample efficiency become central.
- Under the **universal-learning perspective**, multi-task methods use broad experience to build a functional understanding of the world—treating data as a space of possibilities rather than behavior to imitate.

This was the lecturer's final lecture of the course. He encouraged attendance at the guest lectures and thanked the class for joining him during the semester.

### Source reconciliation

Slides 39-44 contain the dialogue comparison, three-perspective synthesis, learning-as-intelligence reflection, and final advice: choose the right problems, remain optimistic under uncertainty, reconsider the problem statement when necessary, value applications, and think big while starting small. The spoken ending concentrates on the three-perspective synthesis; the final advice is present only on the slides.

### Additional explanation

The lecture's unifying lesson is a constraint-to-method mapping:

| Setting | Scarce resource | Reliable structure | Algorithmic priority |
|---|---|---|---|
| Simulator/sandbox | simulator fidelity and compute | known dynamics or verifier | stable large-scale optimization |
| Real world | safe interaction and resets | direct but expensive experience | sample efficiency, robustness, continual learning |
| Universal learning | high-quality supervision | vast mixed-quality experience | multi-task abstraction and utility-directed adaptation |

The dialogue examples illustrate a mechanism, not proof that shorter or more interactive responses are always better. Evaluation should measure later learning or task success on the intended user distribution; unnecessary questions can also reduce utility. The proposed universal recipe is a research perspective, not a general convergence or performance guarantee.

## Slide-only closing reflections

Slides 41-44 were visually inspected. They supplement the spoken synthesis with a broader claim that learning may be the basis of intelligence; a “cake” analogy in which prediction supplies much of the representational substance; and several sources of learning signal, including self-supervised prediction, imitation and cultural knowledge, and long-horizon value propagation. The last slide advises researchers to choose the right problems, remain optimistic under uncertainty, change a problem statement when it blocks progress, take applications seriously, and combine ambitious aims with small initial steps.

These details are slide-derived and are not attributed to the lecturer's spoken account where the transcript does not state them.

## Consolidated takeaways

- A single RL formalism covers many domains, but hidden practical constraints explain why the field needs many algorithms.
- “Reinforcement learning” can name a reward-driven adaptation mechanism or the broader problem of optimizing sequential decisions.
- Simulation-first RL is a modern numerical form of model inversion: characterize a system, simulate it, and optimize its controller.
- RL reliably exploits simulator defects; randomizing simulator parameters can improve robustness but cannot cover phenomena omitted from the model.
- Games, code, mathematics, and some robot mechanics are favorable because their rules or verifiers create useful sandboxes.
- Real-world learning is dominated by scarce interaction, irreversible failure, reset behavior, robustness, and unexpected events.
- Moravec's paradox separates clean, rule-governed “easy universes” from open, exception-filled “hard universes.”
- Continual systems must manage their own state distribution so that failures become recoverable learning opportunities.
- Large low-quality datasets contain knowledge about possibilities, but density modeling alone also reproduces undesirable properties of those datasets.
- A universal RL perspective treats prediction, imitation, and broad experience as ingredients for making high-utility decisions rather than final behavioral targets.
- Multi-turn dialogue illustrates why long-horizon RL can learn to ask informative questions instead of greedily producing a complete one-shot answer.

## Key equations

1. **A simulator defines forward dynamics**

   $$
   s_{t+1}\sim p(s_{t+1}\mid s_t,a_t).
   $$

   Control searches for actions whose forward consequences produce desirable states.

2. **Expected-return optimization across randomized simulators**

   $$
   \max_\theta\;
   \mathbb E_{\xi\sim p(\xi),\,\tau\sim p_{\theta,\xi}(\tau)}
   [R(\tau)].
   $$

3. **Unsupervised density modeling**

   $$
   \max_\theta\;
   \mathbb E_{x\sim p_{\mathrm{data}}}
   [\log p_\theta(x)].
   $$

   This learns what appears in the dataset, not automatically what has high utility.

4. **Decision-directed learning**

   $$
   \max_\pi\;
   \mathbb E_{\tau\sim p_\pi(\tau)}[R(\tau)].
   $$

   The return distinguishes possible behavior from desirable behavior.

## Glossary

- **Engineering-tool view of RL:** Treating RL as automatic controller synthesis inside a designed simulator.
- **Model inversion:** Finding actions that make a forward dynamics model generate desired states.
- **Simulator exploitation:** A policy obtaining high simulated reward through errors or omissions that do not hold in reality.
- **Domain randomization:** Training across varied simulator parameters to obtain a policy robust to model mismatch.
- **Sandbox:** A controllable environment with rules or verifiers in which an agent can generate experience safely and cheaply.
- **Moravec's paradox:** The observation that abilities effortless for humans can be harder to automate than abstract tasks humans find difficult.
- **Easy universe:** A problem domain characterized by compact, known rules and cheap simulation.
- **Hard universe:** An open domain containing diverse, unexpected phenomena that resist complete manual specification.
- **Reset-free learning:** Learning continually without an external mechanism that restores a standard initial state after failure.
- **Offline-to-online RL:** Initial learning from stored experience followed by improvement through live interaction.
- **Density modeling:** Learning the probability distribution of observed data, such as $p(x)$ in generative pretraining.
- **Functional understanding:** Knowledge organized around what actions lead to outcomes rather than only what observations resemble the dataset.
- **Information-seeking action:** An action whose main benefit is improving later decisions by revealing hidden state.

## Self-check questions

1. Why does one mathematical RL definition fail to determine one universally best algorithm?
2. How do the psychological and computer-science meanings of reinforcement learning differ?
3. Why can a coffee-making robot and a walking quadruped require different methods despite sharing the same MDP abstraction?
4. In what sense does simulation-trained RL invert a mathematical model?
5. How can a policy inject “phantom energy,” and what does that reveal about simulator design?
6. What can domain randomization protect against, and what can it not protect against?
7. Why are code and mathematics easier RL sandboxes than customer interaction?
8. What does the human moving pieces in a chess match illustrate about Moravec's paradox?
9. Distinguish an easy universe from a hard universe without using computational cost as the criterion.
10. Why is a reward of “survive” inadequate when failure is irreversible?
11. How can a repertoire of tasks replace an external reset mechanism?
12. Why is real-world interaction simultaneously a perfect simulator and a difficult learning environment?
13. What is the tension between training on low-quality data and estimating $p(x)$?
14. How does treating data as possibilities differ from treating it as supervision?
15. Why can asking a clarifying question be optimal in a multi-turn tutoring problem even if it delays the answer?
16. Map sample efficiency, parallelism, robustness, and multi-task transfer to the three perspectives on RL.

## Source coverage checklist

- [x] All supplied transcript lines 1-4261 are mapped exactly once in increasing, non-overlapping ranges.
- [x] Course logistics, audience questions, examples, historical context, and the lecturer's final thanks are retained.
- [x] Transcript-derived statements, slide-only closing material, and added explanations are explicitly separated.
- [x] All 44 slide pages were rendered and visually inspected.
- [x] Displayed and inline mathematics uses Markdown-compatible LaTeX delimiters.
- [x] The transcript reaches a natural conclusion and is marked source-complete.

**Coverage result:** All 4,261 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. The transcript reaches the lecturer's natural final thanks.
