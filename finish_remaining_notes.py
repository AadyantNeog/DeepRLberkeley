exec(open("review_remaining_notes.py", encoding="utf-8").read().split("coupling =")[0])

replace(21, r"\le1-(1-\epsilon)^t\le\epsilon t.", r"\le\min\{1,(t-1)\epsilon\}.")
replace(21, "2. Explain the no-mistake/at-least-one-mistake mixture for $p_{\\pi_\\theta}(s_t)$.", "2. Derive the first-disagreement coupling bound. Why is the lecture's unconditional mixture not generally valid?")
replace(21, "- DAgger repairs covariate shift by labeling states visited by the learner itself.", "- DAgger addresses covariate shift by labeling learner-visited states, subject to the expert, capacity, and online-learning assumptions.")
replace(22, "- Exact trajectory importance sampling is unbiased but its variance grows catastrophically with horizon.", "- Exact trajectory importance sampling is unbiased under support and fixed-target assumptions, but its variance can grow severely with horizon.")
replace(22, "- Bellman contraction plus function projection does not guarantee nonlinear fitted-value convergence because the contractions use different norms.", "- Bellman contraction does not guarantee fitted-value convergence: convex projection uses a different norm, and nonlinear neural fitting need not even be nonexpansive.")
replace(22, r"\sum_t r(s_t,a_t)+\alpha\mathcal H(\pi(\cdot\mid s_t))", r"\sum_t\left[r(s_t,a_t)+\alpha\mathcal H(\pi(\cdot\mid s_t))\right]")
replace(22, r"y=r+\gamma\max_{a'}Q_{\bar\theta}(s',a').", r"y=r+\gamma m\max_{a'}Q_{\bar\theta}(s',a').")
replace(22, r"""   \widehat A_t^{\mathrm{GAE}}
   =\sum_{l\ge0}(\gamma\lambda)^l
   \left(r_{t+l}+\gamma V_{t+l+1}-V_{t+l}\right).""", r"""   \delta_t=r_t+\gamma m_tV_{\mathrm{ref}}(s_{t+1})-V_{\mathrm{ref}}(s_t),
   \qquad
   \widehat A_t=\delta_t+\gamma\lambda c_t\widehat A_{t+1}.""")
replace(22, "4. **DQN target**", "   Here $m_t$ permits bootstrapping and $c_t$ permits continuation into the next stored transition; see Section 5 for terminal and cutoff handling.\n\n4. **DQN target**")
replace(22, "5. **Variational lower bound**", "   $m=0$ at true termination and $m=1$ otherwise; the regression target is held fixed.\n\n5. **Variational lower bound**")

add(23, 3, r"""**Discrete versus continuous entropy.** For finite discrete variables, $0\le H(X)\le\log|\mathcal X|$, and deterministic $Y=f(X)$ gives $H(Y\mid X)=0$. Continuous variables use *differential* entropy, which can be negative and changes with units: $h(cX)=h(X)+d\log|c|$. A deterministic continuous relation can make the joint distribution singular and mutual information infinite; do not substitute a zero conditional differential entropy into that case. The KL definition of mutual information remains the safer general definition. Entropy measures how probability is distributed, not just whether the support contains many points.""")
add(23, 4, r"""The policy-specific marginal is $d_t^\pi(s)=\Pr_\pi(S_t=s)$. A time-homogeneous environment does not make this distribution time-independent. State-coverage objectives must specify a distribution, such as the normalized finite-time average $H^{-1}\sum_{t=1}^H d_t^\pi$, a discounted occupancy, or a stationary distribution when one exists. Replay usually mixes policies and times and is not exactly the current policy's marginal.

**Precise empowerment.** At a fixed current state $s$, the one-step channel capacity is

$$
\mathcal E(s)=\max_{\nu(a\mid s)}I_\nu(A;S'\mid S=s)
=\max_\nu\left[H_\nu(S'\mid S=s)-H_\nu(S'\mid A,S=s)\right].
$$

An existing policy's conditional mutual information is a value for that input distribution, not necessarily the maximum. Omitting conditioning on the current state can mistake state-action correlations for control authority. Multi-step empowerment similarly uses a specified action sequence or feedback protocol and a later state. Stochastic outcomes help only when the agent can distinguishably influence their distribution.""")
replace(23, "In its one-step form,", "The lecture's shorthand for its one-step information term is")
replace(23, r"""   I(A_t;S_{t+1})
   =\mathcal H(S_{t+1})-\mathcal H(S_{t+1}\mid A_t).""", r"""   \mathcal E(s)=\max_{\nu(a\mid s)}I_\nu(A;S'\mid S=s).""")
replace(23, "- **Empowerment:** Mutual information between an agent's choices and resulting states; a measure of control authority.", "- **Empowerment:** Maximum mutual information between choices and outcomes conditional on a fixed current state, under a specified control horizon/protocol.")
add(23, 5, "Keep the skill fixed over the intended episode or skill segment. Resampling it independently at every primitive step learns a different control interface and can destroy temporal coherence.")
add(23, 6, r"""For skill prior $p(z)$ and the chosen joint distribution of skills and visited states,

$$
I(Z;S)\ge\mathbb E_{z,s}\left[\log q_\phi(z\mid s)-\log p(z)\right].
$$

The gap is $\mathbb E_s D_{\mathrm{KL}}(p(z\mid s)\|q_\phi(z\mid s))$, so a learned discriminator optimizes a **lower bound**, not necessarily the exact mutual information. The complete intrinsic reward is $\log q_\phi(z\mid s)-\log p(z)$. With a fixed uniform prior and fixed-length rollouts the omitted prior term is constant. DIAYN additionally encourages action entropy, typically through SAC, to allow varied behavior within distinguishable skills. See [DIAYN](https://arxiv.org/abs/1802.06070).""")
replace(23, r"\qquad r(s,z)=\log q_\phi(z\mid s).", r"\qquad r(s,z)=\log q_\phi(z\mid s)-\log p(z).")
add(23, 7, r"For $K$ uniformly sampled skills, $I(Z;S)\le\log K$. Tiny disjoint state clusters can already attain this maximum, so distinguishable skills need not cover a large region or be useful downstream. Mutual information is invariant to invertible relabelings of the state; metric distance is additional structure, not something the MI objective automatically rewards.")
replace(23, "The easiest way for the policy to make the goal predictable is to reach it quickly and reliably.", "Reaching a goal is one way to make it predictable from the reached state, but the mutual-information objective alone does not require goal equality or fast arrival; see the counterexample below.")
replace(23, "The fixed reward can be viewed as an extremely restricted discriminator: it assumes the commanded goal is the reached state only when they match. This sacrifices adaptive classification for a stable, stationary objective.", r"""**Predictability is not goal achievement.** With two equally likely commands, suppose “left” always reaches right and “right” always reaches left. The command is perfectly recoverable from the outcome, so $I(G;S)=\log 2$, while goal-success probability is zero. A fixed equality or distance reward grounds what each command means. Speed additionally requires a suitable time-dependent objective, discount, or step cost.""")
add(23, 9, r"""An identity discriminator does not literally give the binary reward: $\log\mathbf1[g=s]$ is zero for a match and $-\infty$ otherwise. A precise finite-$K$ analogy uses

$$
q_\eta(g\mid s)=
\begin{cases}1-\eta,&g=s,\\ \eta/(K-1),&g\ne s,\end{cases}
\quad 0<\eta<(K-1)/K.
$$

Then $\log q_\eta=b+a\mathbf1[g=s]$ with $a>0$. This is an affine success score, equivalent over fixed-length comparisons, not the log of a hard identity. Continuous goals require a normalized density/kernel and a reference measure; equality and tolerance rewards should not be called exact mutual-information objectives without those details.

If the replay goal density is $d(g)$, weighted likelihood fits a target proportional to $d(g)p_\psi(g)^\alpha$. For $\alpha=-1$ this is uniform only when $p_\psi=d$ on the relevant support (and that support has finite reference volume). Freeze the weighting model during the fit and stabilize tiny densities; VAE likelihood estimates, clipping, and model error make the practical update approximate. The [Skew-Fit paper](https://proceedings.mlr.press/v119/pong20a.html) gives convergence under explicit regularity assumptions, not for arbitrary generative models.""")
replace(23, "Reweighting cannot create arbitrary invalid images because training examples still come only from reached states. It changes density within the empirical support, not the support itself.", "Reweighting selects only valid visited examples in the empirical training objective. A fitted neural generator can nevertheless assign probability outside that support and generate invalid or unreachable goals; valid training examples do not guarantee valid generated samples.")
add(23, 10, "In original Go-Explore, the exploration phase can return to an archived state by restoring a simulator snapshot; a later robustification phase trains behavior that tolerates environment randomness. Learned goal-conditioned return mechanisms are possible variants, not a prerequisite of the original method. Frontier novelty alone does not guarantee reachability or useful skills.")

add(24, 2, "The shared-transition formulation assumes tasks differ in reward/context but have compatible state/action semantics and dynamics. General multi-task MDPs can also have task-dependent initial states or transition kernels; retain those dependencies when defining the augmented process.")
add(24, 4, "Observed task context that remains fixed within a rollout can be included in the Markov state. When task identity is hidden, a policy needs inference from experience; this is a partially observed/meta-RL problem rather than an ordinary fully observed context-conditioned MDP.")
add(24, 5, "Relabeling must recompute task-dependent **termination**, as well as reward, and retain time/horizon information where needed. Shared observations do not suffice when tasks change physical dynamics. A logged action remains fixed data; changing the goal does not make that action an on-policy sample for the new goal.")
replace(24, "Changing the second can help, but it implicitly importance-weights learning relative to the first.", "Changing the second can help optimization, but without correction it changes task weighting in the objective. To estimate the original task expectation from $q(\\omega)$, use $p(\\omega)/q(\\omega)$ where support permits. A curriculum may deliberately optimize a different weighting; that is distinct from an unbiased estimate of the evaluation objective.")
replace(24, "Adding or subtracting a constant from every reward does not change the optimal behavior, which makes the success-reward and step-cost conventions closely related.", "Adding a constant preserves policy ordering for fixed-length returns, or continuing discounted returns where it adds the same $c/(1-\\gamma)$. It can change ordering when reward accumulation stops at a policy-dependent terminal time. Success rewards and step costs are therefore not generally equivalent.")
add(24, 7, r"""For a precise hitting-time convention, charge $-1$ for each action taken before the first arrival at $g$, including the action that enters it, then terminate. If $\tau_g$ is the number of those actions and $\mathbb E_\pi\tau_g<\infty$, then $V^\pi(s,g)=-\mathbb E_\pi\tau_g$ at $\gamma=1$. A one-time $+1$ reward on goal entry instead maximizes success probability at $\gamma=1$ and is indifferent to arrival time among certain successes. Unreachable goals can give infinite hitting time; finite-horizon truncation or $\gamma<1$ changes the exact interpretation.""")
replace(24, r"""V(s,g)=
\begin{cases}
-1+\mathbb E[V(s',g)],&s\ne g,\\
0,&s=g.
\end{cases}""", r"""V^\pi(s,g)=
\begin{cases}
-1+\mathbb E_{a\sim\pi(\cdot\mid s,g),\,s'\sim P}[V^\pi(s',g)],&s\ne g,\\
0,&s=g.
\end{cases}""")
replace(24, "The negative value is expected time to goal and behaves like a distance.", "The negative **optimal** value, under the conditions explained below, is minimum expected time to goal and behaves like a directed distance.")
replace(24, r"V(s,g)\ge V(s,w)+V(w,g)", r"V^*(s,g)\ge V^*(s,w)+V^*(w,g)")
add(24, 8, r"""**Two different hindsight biases.** Changing the goal frequencies changes the training objective. Separately, choosing a goal from the realized future can select favorable transition noise: although physical dynamics are goal-independent, the relabeled dataset may satisfy $p_{\mathcal D}(s'\mid s,a,g)\ne P(s'\mid s,a)$. For example, relabeling only successful lottery outcomes as the desired goal hides losing outcomes. Mixing commanded goals back in does not generally eliminate this stochastic selection bias, and merely choosing an off-policy algorithm does not correct it. See [USHER](https://arxiv.org/abs/2207.01115).

**When the triangle inequality holds.** The displayed policy-evaluation equation alone does not imply it. Define $d^*(s,g)=\inf_\pi\mathbb E_\pi\tau_g=-V^*(s,g)$ for a proper shortest-path problem. If a policy can reach waypoint $w$ and then switch to a policy reaching $g$, that feasible concatenation proves $d^*(s,g)\le d^*(s,w)+d^*(w,g)$. Use infinite values for unreachable pairs; finite distances need finite hitting-time assumptions. An arbitrary fixed goal-conditioned policy can be needlessly slow to $g$ and need not satisfy this inequality. Approximate neural values also need not satisfy it.

The optimal backup is $V^*(s,g)=\max_a[-1+\mathbb E_P V^*(s',g)]$ for $s\ne g$, with $V^*(g,g)=0$. This differs from the policy-averaged evaluation backup above.""")
add(24, 9, "A reachability/time-to-goal function is model-like, but generally does not identify the full one-step stochastic transition law. Successor predictions below are specific to a continuation policy; they are not arbitrary-action world simulators.")
add(24, 10, r"""**Normalization convention.** Standard successor counts use $M^\pi(s,i)=\mathbb E_\pi\sum_{k\ge0}\gamma^k\mathbf1[S_{t+k}=i]$, whereas this section uses $\mu^\pi=(1-\gamma)M^\pi$. Accordingly $V^\pi=M^\pi r=\mu^\pi r/(1-\gamma)$. The normalized geometric time has $\Pr(K=k)=(1-\gamma)\gamma^k$ for $k=0,1,\ldots$, so it includes the current state. With terminal states, include a zero-reward absorbing continuation to retain total mass one; dropping post-terminal visits instead gives a sub-probability measure.""")
replace(24, "The original successor representation is the special case where $\\phi(s)$ is a one-hot state vector.", "With one-hot $\\phi(s)$, these unnormalized successor features equal the standard successor counts $M^\\pi$, or $\\mu^\\pi/(1-\\gamma)$ under the preceding section's normalized convention.")
add(24, 11, "The identities require reward to be linear in the chosen features. For transition rewards use matching features $\\phi(s,a,s')$ and expected discounted sums of those same features. If reward approximation has uniform error at most $\\eta$, its fixed-policy discounted value error is at most $\\eta/(1-\\gamma)$. Reusing successor features without reevaluation assumes the same dynamics and continuation policy; changing reward is the part handled by the new weights.")
add(24, 12, "With exact Q-values, shared dynamics, bounded discounted rewards, and exact action maximization, GPI guarantees a policy at least as good as each base policy for the new reward. It evaluates all candidate actions under all base continuation policies; it does not merely choose among the actions each base actor emits. Approximate successor features, fitted reward weights, and approximate maximization weaken this guarantee. Transfer quality depends on both feature coverage and the policy library. See [Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312).")
add(24, 13, r"""The displayed odds assume equal positive/negative class priors and compatible support. With positive prior $\rho$, odds instead equal $\frac{\rho}{1-\rho}\frac{p^\pi(s_f\mid s,a)}{p(s_f)}$. Omit the marginal only when comparing actions for the **same** candidate goal; it cannot generally be dropped when integrating across goals to reconstruct an arbitrary reward value.

Distinguish a future-only successor measure from the current-inclusive one:

$$
\mu_+^\pi(\cdot\mid s,a)
=(1-\gamma)\sum_{k=0}^\infty\gamma^k
\Pr_\pi(S_{t+1+k}\in\cdot\mid s,a),
\qquad
\mu^\pi(\cdot\mid s,a)=(1-\gamma)\delta_s+\gamma\mu_+^\pi(\cdot\mid s,a).
$$

Sampling strictly later positives estimates the first object after normalization. The full measure includes a current-state atom, which in continuous spaces is not an ordinary density; later dynamics can also be singular. The density-ratio derivation requires a common dominating measure/support. Future samples from old replay trajectories follow their behavior continuation, not automatically the current $\pi$; a recursive off-policy method needs a derived correction/backup.""")
replace(24, "Fixed-duration context switching turns hierarchy into ordinary multi-task RL at the low level and a semi-Markov decision process at the high level. Its practical appeal is modularity: each layer can use familiar algorithms.", "Fixed-duration context switching yields a high-level MDP at decision boundaries with discount $\\gamma^K$. Variable option durations give a semi-Markov decision process. Its practical appeal is modularity: each layer can use familiar algorithms.")
add(24, 14, r"""The displayed option backup is a target, not an exact replacement for a stochastic Q estimate. For observed duration $h$, use

$$
y=\sum_{k=0}^{h-1}\gamma^k r_{t+k}
+\gamma^h m\max_{o'\in\mathcal O(s_{t+h})}Q_{\mathrm{target}}(s_{t+h},o'),
\qquad Q(s_t,o_t)\leftarrow Q(s_t,o_t)+\eta[y-Q(s_t,o_t)].
$$

Here $m=0$ at environment termination; ending an option alone does not terminate the environment. Maximize over options available at that state. In the general options formalism, $\beta_o(s)$ is a termination probability, not necessarily a deterministic set. If the low-level policies keep changing, the high-level transition law changes too, so old option replay is not automatically valid. Good options can help, but arbitrary discovered options offer no guaranteed efficiency gain. Retaining primitive actions preserves the original policy possibilities.""")
replace(24, r"V(s,g)=-1+\mathbb E[V(s',g)]\quad(s\ne g),", r"V^*(s,g)=\max_a\{-1+\mathbb E_P[V^*(s',g)\mid s,a]\}\quad(s\ne g),")
replace(24, "- Goal values behave like directed distances and can support waypoint planning and structural auxiliary losses.", "- Optimal proper hitting-time values define directed distances; arbitrary policy values and approximate learned values need not obey the triangle inequality.")

add(25, 5, "“Inverting a model” is an analogy for solving a constrained optimization problem. A dynamics model need not be invertible: some target states are unreachable, many controls can lead to the same state, and stochastic transitions prevent prescribing exact outcomes. Traditional control also uses numerical optimization, MPC, and robust/adaptive methods; it is not limited to hand-derived analytic controllers.")
replace(25, "The guarantee is only as good as the coverage of $p(\\xi)$; omitted failure modes remain exploitable.", "This optimizes average performance under the training distribution, not a worst-case robustness guarantee. Even a represented but rare failure mode can be sacrificed to improve the average. Coverage, weighting, simulator fidelity, and optimization quality all matter; omitted failure modes remain exploitable.")
add(25, 7, "The slogan “anything simulatable is controllable” is not a controllability theorem. Accurate simulation permits evaluating candidate behavior but does not ensure reachability, tractable search, adequate exploration, or reliable optimization. Self-play also need not converge to a strong equilibrium without suitable game/learning conditions.")
add(25, 8, "Passing tests is evidence for the properties those tests cover, not general program correctness. A proof checker validates a formal statement within its assumptions; it does not establish that the statement captured the intended task. Sandbox realism and verifier quality remain separate from optimization success.")
add(25, 9, "Historical correction: Deep Blue won a game against Kasparov in 1996, but Kasparov won that match. Deep Blue won the rematch in **1997**. See [IBM's Deep Blue history](https://www.ibm.com/history/deep-blue). The 1996 wording above preserves the lecture account and should not be learned as the match-victory date.")
add(25, 10, "The easy/hard-universe terminology is the lecturer's informal modeling distinction, not a complexity classification. Compact known rules can still define computationally intractable problems, and physical control can remain difficult even with accurate equations.")
add(25, 12, "These are motivating research judgments rather than impossibility results about other learning paradigms. Supervised models can generalize and planning/search can optimize consequences; RL is one framework for combining experience and sequential utility. A survival reward is mathematically definable even when safe learning from repeated failure is impossible.")
add(25, 14, "A recovery repertoire only replaces resets inside states from which some available behavior can recover. An object dropped out of reach or an irreversible failure can still require external intervention; reset-free training is not equivalent to unlimited autonomous recoverability.")
add(25, 15, "Direct real-world samples avoid simulator mismatch for the situations actually observed. They still leave sampling error, partial observability, nonstationarity, and learned-model/value error. Neither RL nor large datasets guarantee a policy better than every demonstrated behavior; unsupported actions remain uncertain, particularly offline.")
add(25, 17, "Decision theory also includes one-step supervised decisions and contextual bandits. Viewing a prediction through downstream utility does not by itself require a multi-step RL algorithm; sequential RL becomes necessary when present actions affect later states, information, or opportunities.")
add(25, 18, "A single response can be a contextual-bandit action at the conversation level while token generation is a multi-step process inside that action. Multi-turn dialogue adds user responses and information gathering between assistant actions. The choice of action granularity determines the horizon.")
add(25, 20, "The dialogue examples illustrate a mechanism, not proof that shorter or more interactive responses are always better. Evaluation should measure later learning or task success on the intended user distribution; unnecessary questions can also reduce utility. The proposed universal recipe is a research perspective, not a general convergence or performance guarantee.")
replace(25, "2. **Robust optimization across randomized simulators**", "2. **Expected-return optimization across randomized simulators**")

for n, (p, s) in notes.items():
    p.write_text(s, encoding="utf-8", newline="\n")
print("Completed remaining exploration, transfer, and closing-lecture corrections.")
