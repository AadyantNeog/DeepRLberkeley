---
title: "Lecture 20 - Reinforcement Learning Theory"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 20
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 20, Reinforcement Learning Theory.txt"
source_slides: "../lectures/Lecture 20 - RL Theory.pdf"
transcript_lines: 9118
slide_pages: 28
status: "source-incomplete"
---

# Lecture 20: Reinforcement Learning Theory

> **Source warning:** The supplied transcript ends at line 9118 in the middle of a post-lecture question. The notes cover every supplied line and identify that unfinished exchange, but do not invent its missing conclusion.

## Lecture map

**Standing assumptions.** These bounds concern a finite stationary discounted MDP with $0\leq\gamma<1$, bounded rewards, and independent oracle samples at every state-action pair. $N$ denotes samples **per pair** in the model-estimation analysis, so the total is $N|\mathcal S||\mathcal A|$. Matrix formulas use column value vectors and row-stochastic transition matrices. Course dates and logistics below are historical transcript content.

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Logistics and what an RL guarantee asks | lines 1-1017 |
| 2 | Why theory needs assumptions | lines 1018-1764 |
| 3 | Policy-evaluation identities and effective horizon | lines 1765-2361 |
| 4 | Value iteration as a contraction | lines 2362-3257 |
| 5 | Q&A: theory, physics, and worst-case MDPs | lines 3258-3388 |
| 6 | Oracle sampling and the model-based setup | lines 3389-3849 |
| 7 | Concentration inequalities | lines 3850-4812 |
| 8 | The simulation lemma | lines 4813-5850 |
| 9 | From transition error to value error | lines 5851-6435 |
| 10 | Optimal values and the learned policy | lines 6436-6975 |
| 11 | Fitted Q-iteration: sampling and approximation error | lines 6976-9030 |
| 12 | Closing logistics and truncated Q&A | lines 9031-9118 |

## 1. Logistics and what an RL guarantee asks

**Transcript coverage:** lines 1-1017

### What the lecturer said - transcript only

This was the last lecture introducing new material before the midterm. The following Wednesday and Friday lectures would be reviews, the exam was scheduled for April 14, and the TAs would run three review sections announced on Ed.

The lecture was not intended to turn the course into a full theory course. Its aim was to show what an analysis of an RL algorithm looks like, what questions it can answer, and what its limitations are.

A typical result starts with a learned estimate such as $\widehat Q_k$ after $k$ iterations or $N$ samples and asks for a statement of the form

$$
\Pr\!\left(\lVert \widehat Q_k-Q^*\rVert \le \epsilon\right)\ge 1-\delta
\quad\text{when}\quad
N\ge f(\epsilon,\delta).
$$

The function $f$ describes sample complexity. Polynomial dependence on problem size, accuracy, and confidence is generally regarded as meaningful; exponential dependence is usually a warning that the guarantee may be unusable.

In a finite MDP, a Q-function is a vector with $|\mathcal S||\mathcal A|$ entries. Infinite state spaces require additional mathematical machinery, so the lecture deliberately used the tabular setting.

The notation distinguishes several different objects:

- $\widehat Q_k$ is the Q-function actually learned at iteration $k$, perhaps a table or neural-network output.
- $Q^*$ is the true optimal Q-function in the real MDP.
- $Q^{\pi_k}$ is the true Q-function of the policy produced at iteration $k$.
- $\widehat Q^{\pi_k}$ would be an estimated critic for that particular policy.

These objects need not coincide. In Q-learning, for example, $\widehat Q_k$ is not necessarily an estimate of $Q^{\pi_k}$. A bound on the learned Q-function is also not automatically a bound on the true performance of its induced policy. One may therefore ask either

$$
\lVert \widehat Q_k-Q^*\rVert_\infty\le\epsilon
$$

or the more decision-relevant question

$$
\lVert Q^{\pi_k}-Q^*\rVert_\infty\le\epsilon.
$$

Another style of guarantee is regret over $T$ interactions. The slide gave a representative square-root dependence, roughly

$$
\operatorname{Reg}(T)
\le
O\!\left(\sqrt{T N\log\frac{NT}{\delta}}\right)+\delta T.
$$

The lecture concentrated on Q-function error. Exploration and function approximation can also be analyzed, but they introduce further assumptions and technical complications.

### Source reconciliation

Slides 3-5 visually separate learned quantities (hatted symbols) from their true counterparts and distinguish estimation error, policy-performance error, regret, and exploration. The transcript explains these distinctions in more detail than the slides.

### Additional explanation

A useful way to read any theorem is to identify five items: the object being estimated, the norm used to measure error, the probability of success, the assumptions, and how the required data scales. A small error bound can be misleading if it concerns the wrong object or depends on an unrealistically strong data-collection assumption.

## 2. Why theory needs assumptions

**Transcript coverage:** lines 1018-1764

### What the lecturer said - transcript only

Useful theoretical analysis normally makes assumptions much stronger than the conditions of a deep-RL benchmark such as DQN on Atari. The lecture therefore replaced neural networks and enormous observation spaces with a finite tabular MDP.

The purpose is not to pretend that the simplified world is literally real. The lecturer compared it with neglecting air resistance in physics: the approximation can expose an important dependency even when the exact physical situation is more complicated. Assumptions should be strong enough to permit a conclusion without disconnecting the result entirely from the phenomenon of interest.

Worst-case exploration is particularly difficult. Finding a prize hidden somewhere in the continental United States is qualitatively harder than finding one known to be in the lecture room. If an arbitrary MDP can hide reward behind an extremely unlikely sequence, no practical method can promise to find it efficiently without further structure.

To focus on learning rather than exploration, the analysis assumes **oracle exploration**, also called access to a **generative model**. For every state-action pair $(s,a)$, the learner may request independent samples

$$
s'\sim P(\cdot\mid s,a)
$$

and do so $N$ times. In response to questions, the lecturer emphasized that this is effectively the ability to teleport to any chosen state-action pair and observe a next state. It is an analytical abstraction, not what most real robots can do.

One cannot prove that deep RL always works: the claim is false without assumptions, and the real systems are too complicated for such a general theorem. The available choices are often a worst-case statement that says very little or a sharper result under simplifying assumptions.

Theory is still useful as a rough guide. It can suggest qualitative dependencies—such as how error changes with the discount factor or how many more samples are required to halve an error—even when its numerical guarantee should not be read as a literal prediction of real-world performance.

### Source reconciliation

Slide 4 states the generative-model assumption compactly: sample $s'\sim P(s'\mid s,a)$ for any $(s,a)$, with every pair sampled $N$ times. The transcript adds the teleportation interpretation and the extended discussion of what assumptions do and do not justify.

### Additional explanation

The generative model removes a coverage problem. Because the learner can query every pair, it does not need to prove that its behavior policy will reach rare states. This makes the following result a conditional answer: *if coverage is supplied, how accurately can values be learned from a finite number of transition samples?*

## 3. Policy-evaluation identities and effective horizon

**Transcript coverage:** lines 1765-2361

### What the lecturer said - transcript only

For a fixed policy $\pi$, the Bellman equation is

$$
Q^\pi(s,a)
=r(s,a)+\gamma\,\mathbb E_{s'\sim P(\cdot\mid s,a)}[V^\pi(s')]
=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s').
$$

In vector notation, $Q^\pi$ and $r$ have $|\mathcal S||\mathcal A|$ entries, $V^\pi$ has $|\mathcal S|$ entries, and $P$ maps state values to state-action values. Incorporating the policy into the transition gives the square state-action transition matrix

$$
P^\pi(s',a'\mid s,a)=P(s'\mid s,a)\pi(a'\mid s').
$$

The Bellman equation then becomes a linear system:

$$
Q^\pi=r+\gamma P^\pi Q^\pi,
$$

$$
(I-\gamma P^\pi)Q^\pi=r,
$$

$$
Q^\pi=(I-\gamma P^\pi)^{-1}r.
$$

This is a theoretical relationship, not necessarily a practical prescription. The transition matrix may be unknown, too large to store, or not finite at all. Its importance here is that policy evaluation is linear and lets the proof relate an error in dynamics to an error in value.

If one-step rewards have order one, values and the relevant inverse operator have scale about

$$
\frac{1}{1-\gamma},
$$

because $\sum_{t=0}^{\infty}\gamma^t=1/(1-\gamma)$. The quantity $1/(1-\gamma)$ acts like an effective horizon. In a finite-horizon problem, the corresponding scale is roughly $H$. In response to a question, the lecturer stressed that this is an order-of-magnitude statement rather than a claim that every matrix entry is exactly that value.

### Source reconciliation

Slide 6 supplies the matrix dimensions and the explicit definition of $P^\pi$. The transcript verbally develops the same identities and explains why the inverse is useful despite being impractical to compute in large problems.

### Additional explanation

The inverse is a discounted occupancy operator:

$$
(I-\gamma P^\pi)^{-1}
=\sum_{t=0}^{\infty}(\gamma P^\pi)^t.
$$

Each power propagates a signal another step through the Markov chain. This is why a one-step modeling error can influence many future rewards.

For a row-stochastic matrix, the induced infinity norm of this nonnegative inverse is exactly $1/(1-\gamma)$, since every row sums to that amount. Individual entries need not have that size. With terminal states removed the matrix can be substochastic and the same quantity remains an upper bound.

## 4. Value iteration as a contraction

**Transcript coverage:** lines 2362-3257

### What the lecturer said - transcript only

The Bellman optimality operator for state values is

$$
(TV)(s)=\max_a\left[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')\right].
$$

Unlike fixed-policy evaluation, this operator is nonlinear because of the maximization. The optimal value is its fixed point: $TV^*=V^*$.

The key elementary inequality is

$$
\left|\max_x f(x)-\max_x g(x)\right|
\le \max_x|f(x)-g(x)|.
$$

Applying it to two candidate value functions and observing that a probability-weighted average cannot exceed the largest absolute difference gives

$$
\lVert TV-TU\rVert_\infty
\le \gamma\lVert V-U\rVert_\infty.
$$

Thus $T$ is a $\gamma$-contraction in the infinity norm. Setting $U=V^*$ and using the fixed-point identity yields

$$
\lVert TV-V^*\rVert_\infty
\le \gamma\lVert V-V^*\rVert_\infty.
$$

Repeated backups therefore satisfy

$$
\lVert T^kV-V^*\rVert_\infty
\le \gamma^k\lVert V-V^*\rVert_\infty
\longrightarrow 0.
$$

The infinity norm measures the worst error over all states. The lecturer noted that a sampling algorithm cannot generally guarantee low error in an unvisited state. He used the analogy that extensive driving experience in the United States does not establish performance in the United Kingdom. The clean contraction proof temporarily ignores that sampling and coverage issue.

The class then took a four-minute intermission.

### Source reconciliation

Slides 8-9 show the maximum inequality, the statewise derivation, and the geometric convergence argument. The transcript supplies the interpretation of the infinity norm and the unvisited-state caveat.

### Additional explanation

Contraction does two jobs: it establishes uniqueness of the fixed point and makes old errors decay geometrically. Approximate algorithms inherit the second property only partially—new error is injected at every backup, so the eventual error becomes a discounted sum of per-iteration errors.

## 5. Q&A: theory, physics, and worst-case MDPs

**Transcript coverage:** lines 3258-3388

### What the lecturer said - transcript only

After the break, a question compared RL theory with theoretical physics. The lecturer described RL theory as largely statistical learning theory combined with manipulations that handle sequential dependence. Physics can exploit strong structure supplied by the physical world. An arbitrary MDP is much more abstract and may lack that structure, so worst-case RL results must cover deliberately pathological environments as well as realistic ones.

### Source reconciliation

This audience exchange is carried by the transcript; the deck supplies no separate theorem or notation for it. Its role is to qualify the worst-case assumptions used in the surrounding analysis.

### Additional explanation

This explains why domain assumptions can dramatically improve a theorem. Smooth dynamics, low-dimensional latent state, controllability, known symmetries, or access to demonstrations all exclude some pathological MDPs and may make learning provably easier.

## 6. Oracle sampling and the model-based setup

**Transcript coverage:** lines 3389-3849

### What the lecturer said - transcript only

Under oracle exploration, every $(s,a)$ is sampled $N$ times. The empirical transition model is

$$
\widehat P(s'\mid s,a)
=\frac{\#(s,a,s')}{N}.
$$

Given a policy $\pi$, one can evaluate it in this estimated MDP and obtain $\widehat Q^\pi$. The first question is how large $N$ must be to ensure

$$
\lVert Q^\pi-\widehat Q^\pi\rVert_\infty\le\epsilon
$$

with probability at least $1-\delta$. The analysis would then move to the optimal learned Q-function and the true performance of its policy.

The lecturer chose a model-based algorithm because it is convenient to analyze. Under these very strong sampling assumptions, different reasonable methods often exhibit similar qualitative dependencies. The infinity norm was likewise chosen because it supports a clean worst-case argument; other norms are possible but demand more assumptions and more complicated analysis.

### Source reconciliation

Slide 12 draws the three successive targets: fixed-policy evaluation, optimal-Q estimation under $\widehat P$, and performance of the greedy policy. It also identifies $\widehat\pi$ as the argmax policy corresponding to the learned Q-function.

### Additional explanation

There are two proof stages: first estimate how far $\widehat P$ is from $P$; then show how that model error propagates through Bellman evaluation. The second stage is essential because a small probability error is not itself the quantity the agent ultimately cares about.

## 7. Concentration inequalities

**Transcript coverage:** lines 3850-4812

### What the lecturer said - transcript only

Concentration inequalities translate a finite number of samples into a high-probability estimation error. For independent, identically distributed $X_1,\ldots,X_n\in[b_-,b_+]$ with mean $\mu$ and empirical mean $\overline X_n$, Hoeffding's inequality gives one-sided bounds of the form

$$
\Pr(\overline X_n\ge\mu+\epsilon)
\le
\exp\!\left(-\frac{2n\epsilon^2}{(b_+-b_-)^2}\right),
$$

with an analogous lower-tail bound. Consequently,

$$
\Pr(|\overline X_n-\mu|\ge\epsilon)
\le
2\exp\!\left(-\frac{2n\epsilon^2}{(b_+-b_-)^2}\right).
$$

Solving for the error shows the familiar $1/\sqrt n$ rate. The lecturer used estimating the average age of people in the room as an intuitive example.

Transition estimation requires concentration for an entire discrete probability vector. Let $Z\in\{1,\ldots,d\}$ have distribution $q$, and let $\widehat q$ be the empirical distribution from $N$ samples. A suitable result implies, up to constants and the displayed intermediate term,

$$
\Pr\!\left(
\lVert\widehat q-q\rVert_1
\gtrsim \sqrt d\left(\frac{1}{\sqrt N}+\epsilon\right)
\right)
\le e^{-N\epsilon^2}.
$$

Here $d=|\mathcal S|$ because the categories are possible next states. Substituting a failure probability $\delta$ and simplifying gives the qualitative bound

$$
\lVert\widehat P(\cdot\mid s,a)-P(\cdot\mid s,a)\rVert_1
\le
c_1\sqrt{\frac{|\mathcal S|\log(1/\delta)}{N}}
$$

with probability at least $1-\delta$, for a constant $c_1$. This controls the transition model for one state-action pair; a later union bound handles all pairs simultaneously.

### Source reconciliation

Slides 13-14 display the precise concentration statements used in class. On slide 13, the line labelled “or...” appears to show $n\le \frac{(b_+-b_-)^2}{2\epsilon^2}\log\frac{2}{\delta}$; solving the preceding inequality requires $n\ge$ that quantity. The spoken explanation correctly says that more samples reduce error. This note preserves the logical direction and records the apparent slide typo here rather than silently copying it.

### Additional explanation

The state-space factor arises because the proof estimates a distribution with $|\mathcal S|$ categories. It is not merely a time-horizon penalty. The confidence term is only logarithmic in $1/\delta$, whereas accuracy is costly: halving an error governed by $1/\sqrt N$ requires about four times as many samples.

## 8. The simulation lemma

**Transcript coverage:** lines 4813-5850

### What the lecturer said - transcript only

The true and empirical policy values obey

$$
Q^\pi=(I-\gamma P^\pi)^{-1}r,
\qquad
\widehat Q^\pi=(I-\gamma\widehat P^\pi)^{-1}r.
$$

By inserting the appropriate identities, replacing $r$ using the true Bellman equation, and factoring the difference, the lecturer derived the simulation lemma

$$
Q^\pi-\widehat Q^\pi
=
\gamma(I-\gamma\widehat P^\pi)^{-1}(P-\widehat P)V^\pi.
$$

The expression can be interpreted as policy evaluation in the learned dynamics with a pseudo-reward $(P-\widehat P)V^\pi$. A transition error matters in proportion to the values of the states whose probabilities it changes; misplacing probability mass among equally valuable next states has little effect, whereas moving it between high- and low-value states matters greatly.

A second lemma says that, for any vector $v$,

$$
\left\lVert(I-\gamma P^\pi)^{-1}v\right\rVert_\infty
\le
\frac{\lVert v\rVert_\infty}{1-\gamma}.
$$

The proof sets $w=(I-\gamma P^\pi)^{-1}v$ and uses

$$
\lVert v\rVert_\infty
=\lVert(I-\gamma P^\pi)w\rVert_\infty
\ge (1-\gamma)\lVert w\rVert_\infty,
$$

because a stochastic transition operator has infinity norm at most one. In words, a Q-function generated by a pseudo-reward $v$ can amplify that signal by at most the effective horizon.

### Source reconciliation

Slides 15-17 provide the matrix dimensions, the full algebra of the simulation lemma, and the norm proof. The transcript adds the pseudo-reward interpretation and explains when dynamics error affects value.

### Additional explanation

The lemma is called a simulation lemma because it compares returns in two MDPs that differ in their dynamics. Its factorization cleanly separates the one-step discrepancy $(P-\widehat P)V^\pi$ from the repeated propagation performed by the inverse Bellman operator.

The displayed identity assumes the **same reward function** in both MDPs. If rewards are estimated too, the pseudo-reward becomes
$(r-\widehat r)+\gamma(P-\widehat P)V^\pi$.
Consequently, with $\|r-\widehat r\|_\infty\leq\eta_r$, maximum row-wise transition $L_1$ error $\eta_P$, and $|r|\leq R_{\max}$,

$$
\|Q^\pi-\widehat Q^\pi\|_\infty
\leq\frac{\eta_r}{1-\gamma}
+\frac{\gamma R_{\max}\eta_P}{(1-\gamma)^2}.
$$

Here $L_1$ distance is twice total variation; keep that factor consistent when comparing statements.

## 9. From transition error to value error

**Transcript coverage:** lines 5851-6435

### What the lecturer said - transcript only

Combining the two lemmas yields

$$
\lVert Q^\pi-\widehat Q^\pi\rVert_\infty
\le
\frac{\gamma}{1-\gamma}
\lVert(P-\widehat P)V^\pi\rVert_\infty.
$$

For each state-action pair, the expectation difference is at most the one-norm transition error times the largest absolute value:

$$
\lVert(P-\widehat P)V^\pi\rVert_\infty
\le
\left(
\max_{s,a}
\lVert P(\cdot\mid s,a)-\widehat P(\cdot\mid s,a)\rVert_1
\right)
\lVert V^\pi\rVert_\infty.
$$

If rewards lie in $[0,1]$, then

$$
\lVert V^\pi\rVert_\infty\le\frac{1}{1-\gamma}.
$$

The lecturer noted in Q&A that rewards could instead include negative values; bounded rewards can be shifted or rescaled, and the assumption is made without loss of the qualitative point. Combining everything gives, up to constants and a union-bound adjustment over all $(s,a)$,

$$
\lVert Q^\pi-\widehat Q^\pi\rVert_\infty
\le
\frac{\gamma}{(1-\gamma)^2}
c_2\sqrt{\frac{|\mathcal S|\log(1/\delta)}{N}}.
$$

The error decreases as $1/\sqrt N$ and grows quadratically with the effective horizon. One horizon factor bounds the magnitude of the value being multiplied by model error; another reflects the accumulation of that error through repeated backups.

### Source reconciliation

Slides 18-19 show the combined inequality and explicitly call out both qualitative conclusions: more samples lower error, while each backup accumulates error and produces quadratic horizon dependence.

### Additional explanation

Ignoring logarithms and constants, achieving error $\epsilon$ requires a per-pair sample count with the rough scaling

$$
N=\widetilde O\!\left(
\frac{|\mathcal S|}{(1-\gamma)^4\epsilon^2}
\right).
$$

This inversion makes the cost of long horizons especially visible. It is a consequence of this proof and setup, not a universal lower bound for every RL problem.

The union bound replaces a single-pair failure probability with approximately $\delta/(|\mathcal S||\mathcal A|)$ inside logarithms. Multiplying the displayed per-pair count by $|\mathcal S||\mathcal A|$ gives a total sufficient count of roughly $\widetilde O(|\mathcal S|^2|\mathcal A|/((1-\gamma)^4\epsilon^2))$ for this loose proof with unit reward scale. More refined analyses can improve it; “sufficient” does not mean necessary or optimal.

## 10. Optimal values and the learned policy

**Transcript coverage:** lines 6436-6975

### What the lecturer said - transcript only

If the fixed-policy bound holds uniformly for every policy, then the maximum-over-policies inequality implies

$$
\lVert Q^*-\widehat Q^*\rVert_\infty
=
\left\lVert\sup_\pi Q^\pi-\sup_\pi\widehat Q^\pi\right\rVert_\infty
\le
\sup_\pi\lVert Q^\pi-\widehat Q^\pi\rVert_\infty
\le\epsilon.
$$

The supremum appears because the optimal value selects the best policy. The learned empirical MDP produces an optimal policy $\widehat\pi^*$. To compare its true value with the true optimum, add and subtract its empirical value:

$$
\begin{aligned}
\lVert Q^*-Q^{\widehat\pi^*}\rVert_\infty
&\le
\lVert Q^*-\widehat Q^{\widehat\pi^*}\rVert_\infty
+
\lVert\widehat Q^{\widehat\pi^*}-Q^{\widehat\pi^*}\rVert_\infty\\
&\le 2\epsilon.
\end{aligned}
$$

Thus an $\epsilon$-accurate uniform evaluation guarantee produces a $2\epsilon$ guarantee on the true value of the policy optimized in the empirical MDP.

### Source reconciliation

Slide 20 shows both implications and labels the final two terms as comparing the same policy in the learned and true models. The transcript explains why a supremum over policies is introduced.

### Additional explanation

This is a standard “optimization plus estimation” argument. The learned policy may exploit errors in the empirical model, but uniform accuracy prevents any policy—including that data-dependent one—from looking much better in the model than it truly is.

Uniformity here follows from an event bounding **all transition rows**, together with the value bound valid for every policy. No union bound over an uncountable policy class is needed. A high-probability claim for one prespecified policy alone would not justify selecting a policy after seeing the data.

The same two-error argument directly gives $\|V^*-V^{\widehat\pi^*}\|_\infty\leq2\epsilon$ when each policy's state values are uniformly within $\epsilon$ between models. This is the deployment-relevant comparison, including the learned policy's first action.

## 11. Fitted Q-iteration: sampling and approximation error

**Transcript coverage:** lines 6976-9030

### What the lecturer said - transcript only

The exact Q-function Bellman operator is

$$
(TQ)(s,a)
=r(s,a)+\gamma\,
\mathbb E_{s'\sim P(\cdot\mid s,a)}
\left[\max_{a'}Q(s',a')\right].
$$

Exact Q-iteration performs $Q_{k+1}=TQ_k$. Fitted Q-iteration instead uses data to form an approximate Bellman operator

$$
(\widehat TQ)(s,a)
=\widehat r(s,a)+\gamma\,
\mathbb E_{s'\sim\widehat P(\cdot\mid s,a)}
\left[\max_{a'}Q(s',a')\right]
$$

and fits a representable function to its targets:

$$
\widehat Q_{k+1}
\leftarrow
\arg\min_{\widetilde Q}
\lVert\widetilde Q-\widehat T\widehat Q_k\rVert.
$$

Writing $\widehat P$ and $\widehat r$ does not mean the implementation must explicitly construct a model. Averaging observed targets for the same state-action pair has the same effect. The setup assumes each pair appears in the data. The reward estimate is needed only when rewards are stochastic; deterministic rewards can be treated as known from an observation.

Two errors enter:

1. **Sampling error:** $T\ne\widehat T$ because empirical rewards and transitions differ from their expectations.
2. **Approximation error:** the fitted function need not equal the empirical Bellman target.

For one $(s,a)$, sampling error splits as

$$
\begin{aligned}
|\widehat TQ(s,a)-TQ(s,a)|
&\le |\widehat r(s,a)-r(s,a)|\\
&\quad+
\gamma\left|
\mathbb E_{\widehat P}\max_{a'}Q(s',a')
-
\mathbb E_P\max_{a'}Q(s',a')
\right|.
\end{aligned}
$$

Hoeffding bounds the reward term. The transition term is controlled by $\lVert\widehat P-P\rVert_1\lVert Q\rVert_\infty$. A union bound over pairs gives the displayed qualitative result

$$
\lVert\widehat TQ-TQ\rVert_\infty
\le
c_1 R_{\max}
\sqrt{\frac{\log(4|\mathcal S||\mathcal A|/\delta)}{N}}
+c_2\gamma\lVert Q\rVert_\infty
\sqrt{\frac{|\mathcal S|+\log(4|\mathcal S||\mathcal A|/\delta)}{N}}.
$$

This corrected uniform bound makes the transition-dimension and discount factors explicit. An equivalent looser bound can use $|\mathcal S|\log(4|\mathcal S||\mathcal A|/\delta)$ in the transition numerator. A dimension-free constant cannot hide the missing next-state factor in an $L_1$ transition-estimation argument.

To isolate approximation error, the lecture temporarily made the strong assumption

$$
\lVert\widehat Q_{k+1}-T\widehat Q_k\rVert_\infty
\le\epsilon_k.
$$

The lecturer explicitly called this a “magic” infinity-norm fitter and said the assumption is generally false or unrealistic for neural-network regression, whose objective is normally closer to mean-squared error. Under this assumption and the contraction of $T$,

$$
\lVert\widehat Q_k-Q^*\rVert_\infty
\le
\epsilon_{k-1}
+\gamma\lVert\widehat Q_{k-1}-Q^*\rVert_\infty.
$$

Unrolling the recursion gives

$$
\lVert\widehat Q_k-Q^*\rVert_\infty
\le
\sum_{i=0}^{k-1}\gamma^i\epsilon_{k-i-1}
+\gamma^k\lVert\widehat Q_0-Q^*\rVert_\infty,
$$

so the initialization disappears asymptotically and

$$
\limsup_{k\to\infty}
\lVert\widehat Q_k-Q^*\rVert_\infty
\le
\frac{\sup_k\epsilon_k}{1-\gamma}.
$$

Intuitively, every iteration uses a slightly wrong current network to create new targets and then fits those targets imperfectly. Errors are discounted, but new ones continually enter and compound over the horizon.

Sampling and approximation errors can be combined by adding and subtracting the empirical Bellman target:

$$
\begin{aligned}
\lVert\widehat Q_k-T\widehat Q_{k-1}\rVert_\infty
&\le
\lVert\widehat Q_k-\widehat T\widehat Q_{k-1}\rVert_\infty\\
&\quad+
\lVert\widehat T\widehat Q_{k-1}-T\widehat Q_{k-1}\rVert_\infty.
\end{aligned}
$$

The first term is approximation error and the second is sampling error. Because $\lVert Q\rVert_\infty$ itself scales like $R_{\max}/(1-\gamma)$ and backup errors are accumulated for another horizon, sampling effects can acquire quadratic horizon dependence; approximation error acquires one additional horizon factor.

The final conclusion was deliberately qualified. The analysis so far requires strong worst-case infinity-norm assumptions. More refined results use $p$-norms under a data distribution,

$$
\lVert\widehat Q_k-Q^*\rVert_{p,\mu}
=
\left(
\mathbb E_{(s,a)\sim\mu}
\left[|\widehat Q_k(s,a)-Q^*(s,a)|^p\right]
\right)^{1/p},
$$

but then require assumptions about the sampling distribution and coverage. The lecturer pointed students to the RL Theory textbook by Agarwal, Jiang, Kakade, and Sun at `rltheorybook.github.io` for more advanced results.

### Source reconciliation

Slides 22-28 contain the fitted-Q abstraction, the sampling-error split, the strong approximation assumption, its recursive unrolling, and the final $p$-norm definition. Slide 28 also contains a small floating expression near the title whose placement is visually garbled; the transcript's spoken conclusion is used for the explanation rather than attempting to reconstruct that stray typesetting.

### Additional explanation

The central lesson is an error-propagation template:

$$
\text{current error}
\le
\text{new one-step error}
+\gamma\,\text{previous error}.
$$

This template reappears throughout approximate dynamic programming. What changes between theorems is how the one-step error is measured and what coverage or function-class assumptions make it small.

The quantity $\|\widehat Q_{k+1}-T\widehat Q_k\|_\infty$ measures **total** one-step error unless sampling error has been set to zero; approximation/optimization error alone is measured relative to $\widehat T\widehat Q_k$. Use a supremum over bounded $\epsilon_k$ if their maximum is not attained.

The same-data learned $\widehat Q_k$ is random and data dependent. Applying Hoeffding as if its target were a fixed independent function is not valid without extra work. The uniform transition-row bound above avoids this problem and applies to every bounded $Q$ simultaneously. One still needs bounded iterates (for example, clipping to an appropriate value range); arbitrary neural fitted iterates are not automatically bounded by $R_{\max}/(1-\gamma)$.

Mean-squared error on the dataset does not control infinity-norm error at rarely visited pairs. Results that use data-weighted norms need coverage/concentrability assumptions and suitable function-class approximation properties. See the [RL theory textbook](https://rltheorybook.github.io/) referenced in the lecture.

## 12. Closing logistics and truncated Q&A

**Transcript coverage:** lines 9031-9118

### What the lecturer said - transcript only

The following week would contain review rather than new material. Attendance remained optional. Sections and practice problems would support preparation, and students were encouraged to balance exam study with homework and project work. The midterm was again identified as April 14. After it, the course would return to new material, including more inspiring or guest lectures, while leaving time for projects.

After the formal close, a student began a question about how a constant $C$ could fail to depend on another quantity. The supplied transcript cuts off during the lecturer's acknowledgement—“Mhm”—before the question or answer becomes complete. No conclusion can be recovered from the available source.

### Source reconciliation

The slides do not contain the missing post-lecture question or answer. They therefore cannot be used to complete the truncated exchange; only the scheduling context can be reconciled with the deck's closing material.

### Additional explanation

The unfinished exchange should not be treated as course content. If a fuller recording or transcript becomes available, this final line range is the only part of the lecturer layer that needs reconstruction.

## Consolidated takeaways

- A guarantee is meaningful only after identifying the learned object, performance object, norm, confidence, assumptions, and sample dependence.
- Worst-case exploration is impossible to handle efficiently without structure; this lecture bypassed it with a generative-model oracle.
- Bellman operators contract, so old errors decay, but repeated approximate backups accumulate newly injected errors.
- Concentration produces $1/\sqrt N$ estimation rates, while long horizons amplify one-step errors.
- A uniform fixed-policy value bound extends to optimal values and gives a $2\epsilon$ performance bound for the policy optimized in the learned model.
- The fitted-Q proof exposes sampling error and approximation error, but its infinity-norm fitting assumption is intentionally much stronger than ordinary neural-network regression provides.
- Theory supplies qualitative scaling and diagnostic insight; its simplified guarantees should not be mistaken for universal promises about deep RL systems.

## Key equations

1. **Fixed-policy Bellman equation**

   $$
   Q^\pi=r+\gamma P^\pi Q^\pi,
   \qquad
   Q^\pi=(I-\gamma P^\pi)^{-1}r.
   $$

2. **Bellman contraction**

   $$
   \lVert TV-TU\rVert_\infty
   \le\gamma\lVert V-U\rVert_\infty.
   $$

3. **Simulation lemma**

   $$
   Q^\pi-\widehat Q^\pi
   =\gamma(I-\gamma\widehat P^\pi)^{-1}(P-\widehat P)V^\pi.
   $$

4. **Policy-evaluation sensitivity**

   $$
   \lVert Q^\pi-\widehat Q^\pi\rVert_\infty
   \lesssim
   \frac{\gamma}{(1-\gamma)^2}
   \sqrt{\frac{|\mathcal S|\log(2|\mathcal S||\mathcal A|/\delta)}{N}}.
   $$

5. **Approximate-backup recursion**

   $$
   \lVert\widehat Q_k-Q^*\rVert_\infty
   \le
   \epsilon_{k-1}
   +\gamma\lVert\widehat Q_{k-1}-Q^*\rVert_\infty.
   $$

## Glossary

- **Sample complexity:** Number of samples required to achieve a stated accuracy and confidence.
- **Regret:** Cumulative shortfall relative to a comparator, often an optimal policy.
- **Generative model / oracle exploration:** Ability to request an independent next-state sample for any chosen state-action pair.
- **Bellman operator:** Mapping that performs one dynamic-programming backup.
- **Contraction:** A mapping that reduces distances by a factor strictly below one.
- **Infinity norm:** Largest absolute component error; here, the worst state or state-action error.
- **Effective horizon:** The scale $1/(1-\gamma)$ in an infinite-horizon discounted problem.
- **Concentration inequality:** High-probability relationship between empirical and population quantities.
- **Union bound:** Inequality used to make many high-probability statements hold simultaneously.
- **Simulation lemma:** Identity or bound relating value differences between two transition models.
- **Sampling error:** Error caused by replacing population expectations with finite-data estimates.
- **Approximation error:** Error caused by restricting fitted values to a function class or solving the fit imperfectly.
- **Coverage:** Extent to which the data distribution includes the state-action pairs on which accuracy is required.

## Self-check questions

1. Why is a bound on $\widehat Q_k-Q^*$ not automatically a bound on the true performance of $\pi_k$?
2. What difficulty is removed by assuming a generative model?
3. Derive $Q^\pi=(I-\gamma P^\pi)^{-1}r$ from the Bellman equation.
4. Which inequality lets the maximization in the Bellman operator preserve contraction?
5. Why does the transition-estimation error decrease like $1/\sqrt N$?
6. Interpret each factor in the simulation lemma.
7. Where do the two factors of $1/(1-\gamma)$ in the model-based value bound come from?
8. Why does a uniform fixed-policy bound imply a $2\epsilon$ bound for the learned optimal policy?
9. Separate fitted Q-iteration's one-step error into sampling and approximation terms.
10. Why did the lecturer call the infinity-norm approximation assumption unrealistic?

## Source coverage checklist

- [x] All supplied transcript lines 1-9118 are mapped exactly once in increasing, non-overlapping ranges.
- [x] Administrative remarks, intermission, audience questions, and closing remarks are retained.
- [x] Transcript-derived statements are kept separate from slide reconciliation and added explanation.
- [x] All 28 slide pages were rendered and visually inspected.
- [x] Displayed mathematics uses Markdown-compatible LaTeX delimiters.
- [x] The incomplete final Q&A is explicitly marked rather than reconstructed.

**Coverage result:** All 9,118 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. The incomplete final audience exchange is marked and not reconstructed.
