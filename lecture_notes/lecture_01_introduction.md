---
title: "Lecture 1 - Deep Reinforcement Learning: Introduction"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 1
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 1, Deep Reinforcement Learning.txt"
source_slides: "../lectures/Lecture 01 - Introduction.pdf"
transcript_lines: 1411
slide_pages: 38
status: "source-incomplete"
---

# Lecture 1: Deep Reinforcement Learning - Introduction

> **Source warning:** The supplied transcript ends at line 1411 in the middle of the lecturer's explanation of learning and search. The slide deck continues from slide 31 through slide 38. The transcript-only account below therefore covers every supplied transcript line, but it cannot reconstruct the lecturer's missing words. Content found only on the remaining slides is isolated in the slide-only appendix.

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Robotic grasping as a microscopic RL problem | lines 1-115 |
| 2 | What data-driven generative AI learns | lines 116-179 |
| 3 | RL's psychological and optimization roots | lines 180-323 |
| 4 | Course purpose and discussion policy | lines 324-345 |
| 5 | Prerequisites, software, and enrollment | lines 346-446 |
| 6 | Lectures, quizzes, and sections | lines 447-540 |
| 7 | Homework, exam, and final project | lines 541-602 |
| 8 | Logistics Q&A | lines 603-674 |
| 9 | Two meanings of reinforcement learning | lines 675-711 |
| 10 | RL versus supervised learning | lines 712-851 |
| 11 | Policies, states, actions, and rewards | lines 852-942 |
| 12 | Physical control and emergent game strategies | lines 943-1010 |
| 13 | Traffic control | lines 1011-1075 |
| 14 | Language models, image generation, and chip design | lines 1076-1145 |
| 15 | Application and reward-design Q&A | lines 1146-1284 |
| 16 | Why RL: combining data with optimization | lines 1285-1340 |
| 17 | The Bitter Lesson: learning and search | lines 1341-1411 |

## Part I - Motivation and course logistics

## 1. Robotic grasping as a microscopic reinforcement-learning problem

**Transcript coverage:** lines 1-115

### What the lecturer said - transcript only

After checking the recording setup and acknowledging the delayed start, the lecturer introduced a warehouse-automation problem. A robot arm observes a bin of objects through a camera and must output the $(x,y,z)$ location to which it should move to grasp an object.

One approach is to understand and program the task directly using tools such as motion planning and model-based reconstruction. This becomes difficult because object physics vary. A rigid object may be grasped by placing fingers on opposite sides, but an awkward object may require reasoning about its center of mass and balance, while a soft object requires anticipating deformation under contact. Hand-designing a general solution is therefore hard.

A machine-learning alternative would be to collect examples and train a model to produce the associated outputs. The obstacle is supervision: people may not know the perfect robot-arm command for every image. Instead, many robots can practice, recording what they saw, what they attempted, and whether the attempt worked. Random practice does not supply optimal actions; it only reveals different degrees of success or failure.

The resulting dataset differs from a standard input/correct-output dataset. Each observation-action pair is accompanied by an outcome label. Reinforcement learning uses such experience to learn from success, failure, or an intermediate level of utility rather than from a correct action supplied for every input. The aim is to turn non-optimal trial data into an autonomous model that performs the task very well. The lecturer called this a microscopic version of an RL problem.

### Source reconciliation

Slides 2-4 make the physical setup and data flow more precise: the example uses a monocular RGB camera, a 7-degree-of-freedom manipulator, and a two-finger gripper. The supervised-learning sketch associates an image with $(x,y,z)$, while the RL sketch associates an attempted $(x,y,z)$ with a binary success/failure outcome and feeds the learned behavior back into further robot interaction.

### Additional explanation

This is “microscopic” because the example can be viewed as a nearly one-step decision: choose a grasp and observe an outcome. Full RL becomes more distinctive when actions change later observations over many time steps. Even here, two central RL difficulties are visible:

- **Exploration:** the robot must try actions before it knows whether they work.
- **Weak supervision:** a failure label says that the chosen grasp was bad, but not which grasp would have been good.
- **Data-distribution change:** as the policy improves, it attempts different grasps and therefore creates a different training distribution.

## 2. What data-driven generative AI learns

**Transcript coverage:** lines 116-179

### What the lecturer said - transcript only

The lecturer asked why trial-and-error learning matters if human labels might be good enough. He answered with a broader contrast between current generative AI and RL.

Recent generative and language models are compelling because machines reproduce capabilities usually associated with people, such as writing text and generating images, sometimes at near-human or better levels. Yet their central learning idea remains close to supervised learning: obtain data and model either its distribution or a conditional distribution, such as images conditioned on text.

Whether the method is called supervised or unsupervised, and whether it produces language or images, it fundamentally tries to reproduce patterns present in the dataset. This resembles introductory statistical modeling at enormous scale. Consequently, the data's contents matter greatly. A text-to-image model learns what images people place online and how those images are paired with language. A web-trained language model learns to predict the next keystroke a person would make. That is an extremely rich signal, but it still starts from reproducing human-generated behavior.

### Source reconciliation

Slide 6 represents these two statistical objectives as an unconditional model $p_\theta(\mathbf{x})$ and a conditional model $p_\theta(\mathbf{y}\mid\mathbf{x})$. This notation is present on the slide but was not spoken in this transcript segment.

### Additional explanation

The contrast is about the training objective, not a claim that generative models merely memorize. Distribution modeling can produce broad generalization because predicting data well requires discovering reusable structure. Still, a likelihood-based objective asks the model to fit what occurred in the dataset. An RL objective instead scores consequences and can prefer an action that is rare or absent in the data if it achieves more reward.

## 3. RL's psychological and optimization roots

**Transcript coverage:** lines 180-323

### What the lecturer said - transcript only

The lecturer described two intellectual ancestors of reinforcement learning.

The first comes from psychology, including B. F. Skinner's study of how animals adapt to reward, punishment, and other stimuli. In this view, learning does not require examples that directly specify the correct behavior; it proceeds from consequences. The lecturer described this as a powerful model for primitive animal learning while noting controversy over whether it explains more sophisticated forms of learning.

The second ancestor comes from computer science, control, and optimization. Karl Sims demonstrated simulated creatures whose bodies and behaviors evolved together. Rather than specifying which morphology should chase a red target, the program generated designs, evaluated them, and concentrated effort on the successful ones. This showed that optimization could produce emergent solutions.

A later 2012 example by Yuval Tassa used trajectory optimization—not an RL algorithm—to discover locomotion gaits and standing strategies for humanoids online. Again, the designer supplied an optimization method rather than explicit instructions for walking or standing.

Modern RL brings these streams together. Ideas from psychological reinforcement learning and algorithms such as Q-learning provide a more efficient algorithmic foundation for optimization. Early evolutionary methods were computationally inefficient, which made complex, long-horizon problems such as Go difficult. They nevertheless established that optimization could generate unexpected behavior. Better algorithms made this kind of emergence practical on much harder tasks.

AlphaGo was exciting not simply because it played Go well, but because it played differently from people and found moves they had not considered. Go is thousands of years old, so a 2016 system discovering a surprising strategy was remarkable. The lecturer contrasted this with generative AI: generative AI is impressive when its output resembles what a person might produce, whereas RL is impressive when it discovers something no person had considered. The two strengths may be complementary.

### Source reconciliation

Slide 7 identifies the examples as Karl Sims's *Evolved Virtual Creatures* (1994) and Yuval Tassa's *Model-Predictive Control with iLQG* (2012). Slide 8 identifies the AlphaGo example as “Move 37” in the Lee Sedol match. Those names and the 1994 date add slide-only precision; the transcript gives the 2012 and 2016 dates but not the first date or “Move 37” label.

### Additional explanation

The common structure is **search over behavior using an evaluation signal**. Evolution can search populations of bodies and controllers; trajectory optimization searches action sequences; RL searches or learns policies that act repeatedly. Deep RL adds expressive neural representations, allowing the policy or value function to operate on large observation spaces such as images.

The generative-AI/RL distinction is best treated as a spectrum. Modern systems often combine imitation or pretraining with outcome-based optimization: pretraining supplies broad competence from data, while RL shifts behavior toward a chosen objective.

## 4. Course purpose and discussion policy

**Transcript coverage:** lines 324-345

### What the lecturer said - transcript only

The introductory examples were meant to preview the kinds of ideas covered by the course. Students might not implement AlphaGo—though it could be attempted as a final project—but they should learn the algorithms underlying such systems. For this first lecture, questions were explicitly encouraged. The lecturer preferred a useful discussion over rushing through every prepared item.

### Source reconciliation

Slide 9 marks the transition to course logistics. It adds no substantive technical claim.

### Additional explanation

For study purposes, this signals that the course emphasizes algorithmic foundations and the assumptions behind them, not only demonstrations of successful applications.

## 5. Prerequisites, software, and enrollment

**Transcript coverage:** lines 346-446

### What the lecturer said - transcript only

The course is advanced and was originally designed as a graduate class for PhD students. The lecturer expected prepared Berkeley students to handle it, but warned that the pace and assignments would be challenging. Students uncertain about their preparation should start the first homework early, use it as a diagnostic, and consult the instructor or TAs if they struggle.

A prior rigorous machine-learning course is required. Berkeley examples included CS 182 and CS 189, along with courses such as 281A; a comparable graduate course at another institution may also suffice. Students unsure about equivalence could send a syllabus to the instructor.

Assignments require training neural networks with automatic differentiation. PyTorch is the supported path. Other frameworks may be used only if the student can satisfy the autograder, but the staff are not required to support them. A PyTorch review section would be offered, and students with weak neural-network or PyTorch fundamentals were again urged to start early.

There were a limited number of concurrent-enrollment seats. Those students had to submit the enrollment form so the staff could assess prerequisites. Other students unable to enroll directly should join the CalCentral waitlist. The class had already been expanded to the room's practical limit and would not expand further.

### Source reconciliation

Slide 10 explicitly lists CS 182, CS 189, CS 289, CS 281A, and CS 282 as qualifying Berkeley courses and supplies the course enrollment URL. The transcript names only some of this list and refers more generally to equivalent courses.

### Additional explanation

The prerequisites imply expected comfort with probability, optimization, neural-network training, and implementing learning algorithms. The first assignment is not merely graded work; it is an early readiness test before later lectures build on those tools.

## 6. Lectures, quizzes, and sections

**Transcript coverage:** lines 447-540

### What the lecturer said - transcript only

The lecturer would try to ensure that recordings reached the bCourses Media Gallery, although the room lacked a projectionist and occasional recording failures were possible. Lecture attendance was not required. In-person participation and questions were welcome, but watching from home was acceptable. Slides would appear on the course website, and a YouTube playlist was planned.

Every lecture except Lecture 1 would have a short GradeScope quiz, typically three to five questions. Students could complete it while watching or immediately afterward. Its purpose was to highlight roughly five important ideas and confirm that the material registered, not to create grade stress. Quizzes were worth 10% of the course grade and allowed two attempts through a second-try version.

For the first time in this course, TAs would run sections. Although the official listing showed six, staffing supported an initial plan of three. The staff could expand if necessary. The schedule would be announced on Ed, with sections starting the following week. Attendance was optional and no new material would be introduced. Sections would reinforce lecture content, work through proofs in more detail, and support exam review. The first two would review PyTorch and probability.

### Source reconciliation

Slide 11 states that quizzes should be completed within a week of each lecture and that the higher of the two attempts would count. It also shows the bCourses course link. These details are more explicit on the slide than in the spoken account.

### Additional explanation

The quiz design suggests a useful note-taking target: after each lecture, a student should be able to identify a small set of central concepts and distinguish them from examples or implementation details.

## 7. Homework, exam, and final project

**Transcript coverage:** lines 541-602

### What the lecturer said - transcript only

The exam would occur in April, in week 12. It functions like a late midterm or an early final so that the formal final-exam period remains available for project work. It contributes 20% of the grade, and a practice exam would be provided.

Five programming homeworks contribute 50% total, 10% each, with about two weeks per assignment. Students were asked not to use AI coding tools. Because the assignments implement fairly standard algorithms, extensive AI assistance would make fair grading difficult. Students should instead use office hours. With the available support, everyone should be able to complete the homework successfully and earn full credit. Starting early is essential because experiments may train for a long time; discovering a two-hour training run one hour before the deadline is avoidable.

The final project contributes the remaining 20%. CS 285 students were strongly encouraged to design an open-ended project. CS 185 students would receive several default project plans and were advised to choose one, though they could discuss a more open-ended proposal with the instructor or TAs. The default projects were still intended to allow enjoyable exploration.

### Source reconciliation

Slide 12 confirms the 50/20/20 percentage split alongside the 10% quizzes, calls the exam a “quasi-final,” and describes the project as a programming project with several options.

### Additional explanation

The grading structure rewards sustained implementation work more than a single exam. Experiment planning, compute budgeting, and early debugging are therefore part of succeeding in the course, not incidental project-management concerns.

## 8. Logistics Q&A

**Transcript coverage:** lines 603-674

### What the lecturer said - transcript only

In response to a prerequisite question, the lecturer said that already enrolled students would not be retroactively dropped; he could not drop them through the system and would not do so anyway. He nevertheless repeated that students should take the prerequisites seriously and begin the first homework early because the course must move quickly through substantial RL material.

On compute resources, the course had obtained a grant from an unnamed Silicon Valley organization. The staff still needed to finish the infrastructure, but intended to provide enough compute for all homeworks. Students pursuing ambitious final projects would benefit from access to additional personal compute. More information was expected within about a week.

Lecture 1 had an ungraded GradeScope quiz for learning the interface. Official graded quizzes would begin with Lecture 2.

There was no textbook that followed the course closely. Sutton and Barto was recommended as useful enrichment, but the course would reach advanced topics not covered by any textbook the lecturer knew. He joked that he had considered writing one but did not complete it over winter break.

### Source reconciliation

The slides do not add material beyond the logistics summarized above.

### Additional explanation

Sutton and Barto is best treated as a conceptual companion for classical RL, while the lectures remain the authoritative sequence for this course's deep and advanced topics.

## Part II - What is reinforcement learning?

## 9. Two meanings of reinforcement learning

**Transcript coverage:** lines 675-711

### What the lecturer said - transcript only

“Reinforcement learning” is overloaded in two ways:

1. It can name the mathematical formalism for learning-based decision making—a class of problems.
2. It can name an approach that learns decision making and control from experience.

These meanings are related but not identical. Someone working on “RL” might broadly mean learning-based control, including a model learned by machine learning, or might mean the narrower trial-and-error approach that learns control from experience rather than using optimal control or planning. Because both usages are common, the distinction must be kept in mind.

### Source reconciliation

Slide 14 states the same two definitions and emphasizes “from experience” in the second.

### Additional explanation

A useful vocabulary distinction is:

- **Problem formalism:** sequential decisions, state transitions, and accumulated reward.
- **Solution method:** a particular RL, planning, control, imitation, or optimization algorithm.

The same formal problem may be solved by different families of methods. Conversely, an RL algorithm may be applied to a simplified one-step problem.

## 10. RL versus supervised learning

**Transcript coverage:** lines 712-851

### What the lecturer said - transcript only

Standard supervised learning begins with a dataset of input-output pairs, $X$ and $Y$, and learns to predict $Y$ from $X$. This simple statement hides restrictive assumptions.

One is that examples are independent and identically distributed (i.i.d.). All inputs come from the same distribution, and the content of one example does not influence another. For labeled images, image B was not selected because of what appeared in image A. A second strong assumption is that the correct outputs are known during training.

RL breaks both assumptions. Its observations are sequentially dependent: what a car camera sees now depends on the previous observation and on the previous driving action. Driving off the road changes the next image. The correct action is not directly known. Instead, the learner observes whether behavior succeeded or failed, or more generally receives a numerical utility or reward.

An RL problem models an agent interacting sequentially with an environment. At each time step, the agent observes a state or observation and chooses an action. State and action play roles analogous to supervised-learning input and output, but the data consists of a trajectory such as state, action, next state, next action, and so forth. Time steps are not independent, and the agent operates inside a dynamical system. Rather than being told the correct output, it must infer good actions from their consequences.

An audience member asked whether an independence assumption could still hold between sequences. The lecturer said that, instead of standard example-level i.i.d. assumptions, one typically assumes that each trajectory begins by resetting to an independently selected state.

Another question concerned the Markov assumption. The lecturer said RL can be formulated when the current observation is sufficient for choosing an action—the Markov setting—and also when the current observation is insufficient, requiring memory of earlier events. Both settings can be addressed.

### Source reconciliation

Slides 15-16 provide the formal comparison:

$$
\mathcal D = \{(x_i,y_i)\}, \qquad f_\theta(x_i) \approx y_i
$$

for supervised learning, versus a trajectory

$$
(s_1,a_1,r_1,\ldots,s_T,a_T,r_T)
$$

for RL. The slide explicitly displays an agent-environment loop in which the agent emits $A_t$ and later receives $S_{t+1}$ and $R_{t+1}$. It also states that previous outputs influence future inputs.

### Additional explanation

Sequential dependence creates two distinct complications:

- **Temporal correlation:** adjacent observations are related, so ordinary random-sample intuition can fail.
- **Policy-dependent data:** changing the policy changes which states are visited, so the learner changes its own future dataset.

In a fully observed Markov decision process, the current state contains all information from history needed to predict the future, given the next action. In a partially observed problem, the agent may need a history window, belief state, or recurrent model.

## 11. Policies, states, actions, and rewards

**Transcript coverage:** lines 852-942

### What the lecturer said - transcript only

The goal is to learn a **policy**, a model mapping states $S$ to actions $A$. Supervised learning similarly maps $X$ to $Y$; the important difference is the objective. RL seeks a policy that maximizes reward rather than one that simply reproduces actions in a dataset.

The lecturer then identified states, actions, and rewards in several examples:

- **Training a dog:** the state can include the dog's body configuration and perceptions such as the owner's command. Actions could be a discrete set such as barking or wagging, or a continuous vector of muscle activations, depending on the chosen interface. A natural reward is receiving food or a treat. The slide-level summary is muscle contractions, sight and smell, and food.
- **Teaching a robot to run:** the state includes joint angles and possibly velocities. Actions are motor commands such as torque, current, or power. Reward can be measured by running speed.
- **Inventory management:** state is the inventory level across warehouses; actions specify what to purchase; reward is profit, such as revenue minus costs.

Many problems can be formulated in this state-action-reward language.

### Source reconciliation

Slide 16 gives the policy and objective more precisely:

$$
\pi_\theta: s_t \mapsto a_t, \qquad \text{maximize } \sum_t r_t.
$$

Slide 17 labels the environment feedback as consequences, observations/states, and rewards. It also uses camera images as robot observations and “task success measure (e.g., running speed)” as the reward.

### Additional explanation

The formulation depends on modeling choices. A dog's action could be “sit” at a high level or thousands of muscle activations at a low level. Neither is universally correct; the representation determines what the policy must learn and what dynamics remain hidden inside the environment.

Reward is not a full instruction. It ranks outcomes through a scalar signal, leaving the algorithm to discover the action sequence that produces those outcomes.

## 12. Physical control and emergent game strategies

**Transcript coverage:** lines 943-1010

### What the lecturer said - transcript only

RL can solve complex physical tasks. The lecturer showed a 2022 ETH Zurich system trained to climb and navigate. He noted that this had been state of the art in 2022 but that progress had continued. A more recent Berkeley example trained a humanoid, with rewards derived in some way from human videos, to walk, climb stairs, and traverse environments. As a rule of thumb, extremely agile humanoid behavior is likely to have been trained with RL.

Another robot combined imitation learning and RL to make espresso, including variants such as lattes and Americanos. It used both human demonstrations and rewards. A person labeled the completed attempt as success or failure rather than judging taste.

The lecturer connected these physical systems to emergent behavior. In a classic Atari result by Vlad Mnih and colleagues from more than ten years earlier, an agent playing Breakout discovered that sending the ball behind the brick wall allowed it to clear bricks with less active intervention. The policy found this strategy without being told to use it.

### Source reconciliation

Slides identify the locomotion work as Rudin et al., *Learning Locomotion and Local Navigation End-to-End* (2022), the Berkeley humanoid work as Allshire et al., *VideoMimic* (2025), and the Atari result as Mnih et al. (2015). Slide 20 labels the coffee demonstration with `pi.website`; the transcript supplies the imitation-plus-RL and human-label details.

### Additional explanation

The espresso example illustrates a common hybrid recipe:

1. Demonstrations place the policy near useful behavior.
2. Outcome rewards distinguish successful from unsuccessful attempts.
3. RL improves behavior beyond direct imitation by optimizing those outcomes.

The Breakout strategy illustrates why reward optimization can produce behavior that is effective but not explicitly represented in the training data or instructions.

## 13. Traffic control

**Transcript coverage:** lines 1011-1075

### What the lecturer said - transcript only

RL is not limited to games and robots. The lecturer described traffic research by Cathy Wu, formerly a Berkeley PhD student and now an MIT professor.

In a circular-road simulation, white cars represented human-like drivers, a red car could be switched from a virtual human to an RL agent, and a blue marker indicated the vehicle in front that the agent could observe. Human-like driving naturally produced stop-and-go waves. When the RL controller took over, it slowed down, created extra spacing, and regulated the whole traffic flow so that bunching diminished. A suitably public-minded autonomous car could therefore improve conditions for surrounding human-driven cars.

In a figure-eight intersection, human-like drivers tended to block one another. The RL-controlled red car arranged traffic into a carefully sized queue or “snake,” allowing vehicles to pass through without waiting at the conflict point. Although this example was somewhat artificial, related methods had been scaled to more realistic highway simulators, where RL could regulate traffic and reduce jams.

### Source reconciliation

Slide 22 visually shows the circular and figure-eight road layouts and credits Cathy Wu, but the mechanism and interpretation come from the transcript.

### Additional explanation

These examples demonstrate a multi-agent externality: one vehicle's action affects many others. The reward can therefore encode system-level flow rather than the controlled car's immediate speed. Slowing one car locally can improve total throughput globally.

## 14. Language models, image generation, and chip design

**Transcript coverage:** lines 1076-1145

### What the lecturer said - transcript only

RL can train language models from human feedback. The lecturer joked that this can produce excessively flattering and slightly annoying responses, but also noted that it can improve the correctness of solutions.

RL can also optimize image generators. Several graduate students experimented with an older Stable Diffusion model that struggled with unusual prompts such as “a dolphin riding a bike,” partly because such scenes are rare online and physically implausible. Kevin Black and Michael Janner implemented an RL procedure in which an image-captioning model rewarded images whose generated captions matched the original prompt.

Across optimization steps, the output gradually included more of the requested concepts: first a bike and a nearby dolphin, then a dolphin-like creature with legs, and eventually images containing prominent dolphins and bicycles. High reward did not guarantee semantic correctness. When asked for two dolphins, the model sometimes wrote the numeral 2 while showing three dolphins. This was reward hacking: it exploited the captioner's scoring behavior rather than fulfilling the intended request.

RL has also been used for chip design. The lecturer said that a TPU used through Google Cloud was likely to be a chip whose design involved an RL algorithm.

### Source reconciliation

Slide 23 diagrams RL from human feedback with a prompt dataset, an initial language model, ranked human preferences, and a learned reward model. Slide 24 names the image method DDPO and credits Black, Janner, Du, Kostrikov, and Levine, *Training Diffusion Models with Reinforcement Learning* (2023). Slide 25 depicts an RL agent placing chip macros sequentially, with wire length (HPWL) and congestion contributing to reward.

### Additional explanation

All three applications optimize a **proxy**:

- A learned preference model stands in for qualities people want from language.
- Caption similarity stands in for whether an image satisfies a prompt.
- Engineering metrics stand in for a good chip layout.

Reward hacking occurs when performance on the proxy diverges from the intended goal. Better optimization can worsen this failure if the proxy itself is misspecified.

## 15. Application and reward-design Q&A

**Transcript coverage:** lines 1146-1284

### What the lecturer said - transcript only

The lecturer paused for questions about applications and rewards.

- **Algorithmic trading:** RL has been used, but practitioners keep details secret because the methods are commercially valuable.
- **Controlling the learned solution:** behavior is controlled by modifying the reward function, analogous to choosing when to give a dog a treat. A grasping task may use $+1$ for success and $0$ otherwise. Chip design is harder because latency, wire length, and other objectives must be traded off and ultimately reduced to a single scalar. Designing that scalar is not trivial.
- **The dolphin-image reward:** the generated image is passed to an image captioner. A BERT-based textual similarity score compares the caption to the original prompt. Writing the numeral 2 can fool this process because it makes the caption likely to mention “two.”
- **Designed versus learned rewards:** humans usually design rewards because they must communicate what they want. Advanced methods can instead learn rewards from demonstrations or human preferences. In the standard language-model example, people compare response pairs, and a reward model learns from their preferences. Learned rewards also have pitfalls; preferences can encourage excessive flattery or other annoying behavior.
- **Humanoid robots and strategic control:** agile humanoids are commonly trained in simulation with RL and transferred to the real world using additional techniques. RL has also been explored in operations research and aerospace for high-level strategies such as deconflicting aircraft. Production systems may leave low-level actuation to conventional controllers and place a human between the learned strategy and real-world execution.
- **Delayed rewards:** delayed consequences are a central RL challenge whether the reward is hand-specified or learned. A reward defines success, not the procedure for achieving it. “Holding the object” describes a successful grasp but does not identify where the hand should move. RL therefore often has to connect an eventual outcome to earlier decisions.

### Source reconciliation

The slide deck provides visual examples but no additional answer content beyond the transcript for this Q&A.

### Additional explanation

Reward design involves three separate problems:

1. **Specification:** does the scalar measure what people actually want?
2. **Credit assignment:** which earlier actions caused a delayed outcome?
3. **Optimization robustness:** can the agent exploit the measurement process without achieving the intended result?

Learning a reward transfers part of the specification problem to a model; it does not eliminate it. The learned model can still encode bias, ambiguity, or exploitable errors from its preference data.

## Part III - Why learn about reinforcement learning?

## 16. Combining data with optimization

**Transcript coverage:** lines 1285-1340

### What the lecturer said - transcript only

RL is compelling because it can produce emergent solutions rather than only reproduce a data distribution. Data-driven AI learns about the real world from examples, but by itself does not attempt to outperform the data. To do better, one must either curate superior data with great care or introduce optimization.

RL optimizes a specified goal and can thereby generate new behavior. It also has major challenges. Defining the reward is one. Scaling RL is another. Generative AI has succeeded partly because very large datasets and models can be used effectively. Making RL scale in a similarly reliable way has been difficult. Recent ideas covered later in the course aim to improve scalability, but the lecturer described it as an open problem.

In its current form, data-driven AI is centered on using data, while RL is centered on optimization. The central challenge is combining them correctly. Data without optimization cannot solve new problems in new ways, but optimization also needs knowledge obtained from data.

### Source reconciliation

Slide 29 sharpens the comparison: data-driven AI learns about the real world from data but does not try to do better than the data; RL optimizes a goal and can yield emergence but remains hard to use at scale. Slide 30 adds the complementary warning that optimization without data is hard to apply to the real world outside simulators.

### Additional explanation

One practical synthesis is:

- use large-scale offline data to learn representations, world knowledge, or an initial policy;
- use interaction, planning, or RL to optimize decisions against consequences;
- retain constraints or human feedback so the optimized behavior remains aligned with the intended task.

This pretraining-plus-optimization pattern connects modern foundation models with sequential decision making.

## 17. The Bitter Lesson: learning and search

**Transcript coverage:** lines 1341-1411

### What the lecturer said - transcript only

The lecturer recommended Richard Sutton's 2019 essay *The Bitter Lesson*. Sutton, whom the lecturer described as a central founder of the modern study of RL, argues that manually encoding how people think does not work well in the long run. The methods that continue to scale are learning and search.

One might program an AI system according to a hand-designed account of human reasoning—for example, explicitly coding a personal planning procedure for traveling to work. Such a system may work on the immediate task, but it is difficult to extend into a general method for many problems. Language models and generative AI are powerful partly because one scalable recipe can absorb more data and be applied to more tasks without equivalent amounts of new engineering.

The lecturer warned against reading the essay as if it ended with “learning.” It does not merely recommend putting more data into more GPUs. “Search” is also essential. In the RL usage presented here, search broadly means an algorithm that solves a sequential decision problem. Learning extracts patterns from data so that a network understands something about the world. The supplied transcript then ends mid-explanation as the lecturer begins to say that search “ruminates” on what has been learned.

### Source reconciliation

Slide 30 completes the conceptual diagram without supplying the missing spoken wording:

- **Learning:** use data to extract patterns and thereby understand the world.
- **Search:** use computation to extract inferences; an optimization process, typically iterative, uses computation to make rational decisions.
- Learning and search form a loop in which optimization leverages learned understanding to produce emergent behavior.

The slide also states that data without optimization cannot solve new problems in new ways, while optimization without data is hard to apply outside simulators.

### Additional explanation

Here “search” is broader than enumerating possibilities in a tree. It includes computation performed at decision time or during iterative optimization: planning trajectories, evaluating alternatives, improving a policy, or otherwise turning a learned model into an action. Learning amortizes experience into parameters; search spends computation to adapt those learned patterns to the current decision.

## Slide-only appendix: material after the transcript truncation

The following material comes from slides 31-38 only. It is **not** presented as a paraphrase of what the lecturer said because the supplied transcript contains no corresponding speech.

### A. Machine learning as adaptable decision making

Slide 31 proposes that machine learning is needed to produce adaptable, complex decisions. It contrasts an actionable decision—how to move joints or steer a car—with a passive prediction such as an image label, and asks what happens to that label afterward. A side reference to Daniel Wolpert presents movement as the brain's means of affecting the world.

### B. Learning as a basis for intelligence

Slides 32-34 ask how one should begin building an intelligent machine. People can perform some abilities broadly, learn other abilities such as driving, and acquire a very wide range of difficult skills. The slides infer that general learning mechanisms may be powerful enough to support much of intelligence, while allowing that a few important components might still be hard-coded.

### C. One flexible algorithm and sensory substitution

Slides 35-36 ask whether intelligence requires a separate algorithm for every module or a single flexible learning algorithm. Sensory-substitution examples—such as conveying visual information through the tongue or repurposing cortical areas—illustrate the possibility that a common learning mechanism can interpret different inputs. Such an algorithm would need both to interpret rich sensory data and to select complex actions.

### D. Why deep reinforcement learning?

Slide 37 summarizes the intended synthesis: “deep” refers to scalable learning from large, complex datasets, while “reinforcement learning” supplies optimization. Learning extracts patterns from data; search or optimization uses computation to turn those patterns into decisions and emergent behavior.

Slide 38 closes with Alan Turing's proposal to build a child-like learning system and educate it rather than directly program an adult mind. The accompanying diagram places a general learning algorithm in a loop with an environment, receiving observations and producing actions.

## Consolidated takeaways

1. RL learns from evaluated consequences rather than requiring a correct action label for every input.
2. RL is both a problem formalism for sequential decisions and a family of experience-based solution methods.
3. Supervised examples are typically treated as i.i.d.; RL trajectories are temporally dependent and generated by the learner's own actions.
4. A policy maps states or observations to actions and is optimized for accumulated reward.
5. Reward specifies what counts as success, not how to achieve it.
6. RL can discover emergent strategies absent from demonstrations, as illustrated by AlphaGo, Breakout, locomotion, and traffic regulation.
7. Reward functions can be written by people or learned from demonstrations and preferences; both approaches can be misspecified or exploited.
8. Delayed reward creates a credit-assignment problem across time.
9. Data-driven learning supplies knowledge of the world; optimization or search uses that knowledge to choose behavior.
10. Scaling the combination of rich data, expressive models, and reliable RL optimization remains a major challenge.

## Key equations

### Supervised learning

$$
\mathcal D = \{(x_i,y_i)\}_{i=1}^N, \qquad f_\theta(x_i) \approx y_i.
$$

The target $y_i$ is supplied in the dataset.

### Reinforcement-learning trajectory

$$
\tau=(s_1,a_1,r_1,\ldots,s_T,a_T,r_T).
$$

The agent's action affects later states and rewards, so the samples within a trajectory are not independent.

### Policy and objective

$$
a_t \sim \pi_\theta(\cdot\mid s_t), \qquad
\max_\theta \; \mathbb E_{\tau\sim\pi_\theta}\!\left[\sum_{t=1}^{T} r_t\right].
$$

The stochastic notation and expectation are an additional formalization; the slide uses the simpler mapping $\pi_\theta:s_t\mapsto a_t$ and objective $\sum_t r_t$.

## Glossary

- **Action:** a decision or control output chosen by the agent.
- **Agent:** the decision-making system interacting with an environment.
- **Credit assignment:** determining which earlier actions contributed to a later reward.
- **Emergent behavior:** effective behavior discovered through optimization rather than directly programmed or demonstrated.
- **Environment:** the external dynamical system that receives actions and returns observations and rewards.
- **i.i.d.:** independent and identically distributed; the standard assumption that examples come from one distribution without influencing one another.
- **Markov state:** a state containing the information needed to predict future behavior, conditional on the next action.
- **Observation:** information available to the agent; it may or may not fully reveal the underlying state.
- **Policy:** a model that maps a state or observation to an action or an action distribution.
- **Reward:** a scalar evaluation signal defining the objective, not the action sequence required to achieve it.
- **Reward hacking:** exploiting imperfections in a reward or its measurement while missing the designer's intended outcome.
- **Search:** in the lecture's broad usage, computation or iterative optimization that turns learned knowledge into sequential decisions.
- **State:** a representation of the situation relevant to future decisions and consequences.
- **Trajectory:** an ordered sequence of states, actions, and rewards produced through interaction.

## Self-check questions

1. Why is random robot practice not equivalent to a supervised dataset of correct grasps?
2. In what sense does a generative model reproduce a data distribution, and how does the RL objective differ?
3. What two historical streams did the lecturer say came together in modern RL?
4. Why were AlphaGo's unexpected moves especially significant in the lecturer's comparison with generative AI?
5. What are the two meanings of “reinforcement learning” introduced in the lecture?
6. Which two common supervised-learning assumptions are violated in RL?
7. Why does the agent's previous action affect the distribution of its future observations?
8. What distinction separates a state from an observation in a partially observed problem?
9. Define the policy and reward for the robot-running and inventory-management examples.
10. How did one RL-controlled car reduce a traffic wave by slowing down?
11. Why did the numeral 2 fool the image-generation reward?
12. What three problems remain even when a reward is learned rather than hand-written?
13. Why does “successfully holding an object” create a delayed-reward problem?
14. What complementary roles do data and optimization play?
15. Why does the lecturer say *The Bitter Lesson* is about learning **and** search?

## Source coverage checklist

| Topic | Inclusive range | Coverage status |
|---:|---:|---|
| 1 | 1-115 | Accounted for |
| 2 | 116-179 | Accounted for |
| 3 | 180-323 | Accounted for |
| 4 | 324-345 | Accounted for |
| 5 | 346-446 | Accounted for |
| 6 | 447-540 | Accounted for |
| 7 | 541-602 | Accounted for |
| 8 | 603-674 | Accounted for |
| 9 | 675-711 | Accounted for |
| 10 | 712-851 | Accounted for |
| 11 | 852-942 | Accounted for |
| 12 | 943-1010 | Accounted for |
| 13 | 1011-1075 | Accounted for |
| 14 | 1076-1145 | Accounted for |
| 15 | 1146-1284 | Accounted for |
| 16 | 1285-1340 | Accounted for |
| 17 | 1341-1411 | Accounted for; source ends mid-sentence |

**Coverage result:** All 1,411 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. Slides 31-38 have no transcript coverage and are isolated in the slide-only appendix.
