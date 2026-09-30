from pathlib import Path
import re

root = Path("lecture_notes")
notes = {}
for n in [1, 2, 3, 4, 21, 22, 23, 24, 25]:
    p = next(root.glob(f"lecture_{n:02d}_*.md"))
    notes[n] = [p, p.read_text(encoding="utf-8")]

def replace(n, old, new):
    assert old in notes[n][1], (n, old[:100])
    notes[n][1] = notes[n][1].replace(old, new)

def add(n, section, text):
    s = notes[n][1]
    m = re.search(rf"^## {section}\. .*?(?=^## |\Z)", s, re.M | re.S)
    assert m, (n, section)
    block = m.group().rstrip() + "\n\n" + text.strip() + "\n\n"
    notes[n][1] = s[:m.start()] + block + s[m.end():]

coupling = r"""
**Corrected coupling proof.** The displayed mixture in the lecture is a heuristic, not an identity implied by average expert-distribution error. Write $d_t^*$ for the expert's state distribution at time $t$ and assume $\mathbb E_{s\sim d_t^*}[e(s)]\le\epsilon$ at every time, where $e(s)=1-\pi_\theta(\pi^*(s)\mid s)$. Start expert and learner in the same sampled state and use shared transition randomness while their actions agree. Continue the expert rollout after a disagreement.

The probability that the *first* disagreement is at step $k$ is at most $\mathbb E_{d_k^*}[e(s)]$: the first-disagreement event is a subset of the event that a learner action sampled at the expert's step-$k$ state disagrees. A union bound over the $t-1$ actions before $s_t$ therefore gives

$$
D_{\mathrm{TV}}(d_t^*,d_t^\pi)
\le \Pr(\text{some disagreement before }t)
\le \min\{1,(t-1)\epsilon\}.
$$

Conditioning on no disagreement can change the distribution of expert states: states where the learner is accurate are overrepresented among surviving trajectories. Thus the conditional no-error distribution cannot generally be replaced by the unconditional $d_t^*$. Nor does an average error bound imply independent constant-rate errors or survival probability $(1-\epsilon)^t$. A *uniform conditional* error bound gives survival **at least** $(1-\epsilon)^{t-1}$, but still does not justify that mixture identity.

For $f(s)=\mathbb E_{a\sim\pi}[c(s,a)]\in[0,1]$, the sharper bounded-expectation inequality is $|\mathbb E_p f-\mathbb E_q f|\le D_{\mathrm{TV}}(p,q)$ (the lecture's $2D_{\mathrm{TV}}$ is a valid looser bound). Hence

$$
J_{\mathrm{imit}}(\pi)
\le \sum_{t=1}^H\min\{1,t\epsilon\}
\le \min\left\{H,\frac{\epsilon H(H+1)}2\right\}.
$$

This proves the intended $O(\epsilon H^2)$ upper bound without the invalid mixture. Quadratic growth describes the small-error regime; the count can never exceed $H$. These are population assumptions, not guarantees obtained merely from a low training-set loss.
"""

dagger = r"""
**What DAgger guarantees.** Expert labels alone are insufficient: the policy class must fit them adequately, and the supervised updates must satisfy the online-learning/no-regret assumptions. The analysis controls average learner-distribution loss over iterations and supports a suitable mixture or selected iterate; it does not automatically guarantee the final neural-network checkpoint or convergence of visitation distributions. If the deployed policy's average disagreement on its *own* states is $\epsilon$, its expected disagreement count is $H\epsilon$. Translating disagreement into excess task cost additionally needs a bound on the expert's cost-to-go sensitivity to a wrong action, often related to recoverability. See the [original DAgger analysis](https://proceedings.mlr.press/v15/ross11a.html).
"""

add(1, 1, "A single grasp with no action-dependent future episode is more precisely a **contextual bandit**. Learning a success predictor from logged trials and maximizing it can solve that special case; multi-step RL additionally handles how actions affect later opportunities.")
replace(1, "Even here, two central RL difficulties are visible:", "Even here, three central RL difficulties are visible:")
add(1, 2, "Preferring an unseen action is not evidence that it will work. Offline RL needs data coverage or defensible generalization assumptions; online exploration or a sufficiently accurate simulator can supply evidence about new behavior. Distribution modeling also learns compositional structure, so novelty is not exclusive to RL.")
add(1, 10, "The i.i.d. comparison is a common introductory setup, not a definition of supervised learning: supervised sequence models can learn from dependent data, and offline RL can use a fixed dataset from another policy. Independently reset trajectories are i.i.d. only when the behavior policy, environment, and initial-state distribution are also held fixed. Continuing RL need not have independent resets at all.")
add(1, 11, r"""For precision, an MDP state satisfies

$$
p(s_{t+1},r_t\mid s_{1:t},a_{1:t})
=p(s_{t+1},r_t\mid s_t,a_t).
$$

A camera frame is not automatically such a state: identical images can hide different velocities. History or an inferred belief can supply missing information. Finite-horizon policies may also need the remaining time as input.""")
add(1, 16, "The data-versus-optimization contrast is motivational. Supervised learning itself uses optimization and can generalize beyond examples. RL supplies an outcome-based objective; it does not guarantee novelty, superiority to demonstrations, or recovery of outcomes unsupported by available experience.")
replace(1, r"\tau=(s_1,a_1,r_1,\ldots,s_T,a_T,r_T).", r"\tau=(s_1,a_1,r_1,\ldots,s_T,a_T,r_T,s_{T+1}).")

add(2, 3, r"""In controlled dynamics the precise condition includes the chosen action:

$$
p(s_{t+1},r_t\mid s_{1:t},a_{1:t})
=p(s_{t+1},r_t\mid s_t,a_t).
$$

The slide's state-only independence follows after fixing a memoryless policy and integrating its actions. It need not hold under an arbitrary history-dependent controller unless the controller's memory is included in the state.""")
replace(2, "A coarse state can have a well-defined stochastic transition while still discarding information that would improve control.", "A conditional distribution can be fitted to almost any coarse observation, but its existence does **not** prove the Markov property. For example, current price alone may leave history informative about the next price. A decision-sufficient state must preserve reward information as well as transition information; a constant representation can evolve trivially while discarding everything needed to choose rewarding actions.")
add(2, 7, r"""For fixed identity covariance,

$$
\log\pi_\theta(a\mid o)=-\tfrac12\|a-\mu_\theta(o)\|^2-\tfrac d2\log(2\pi).
$$

Thus maximizing likelihood is equivalent to minimizing squared error. Learned covariance also contributes a log-determinant penalty and must be positive definite; a diagonal model still needs positive variances, usually enforced through a log-scale parameterization. Flow matching and diffusion usually use their own tractable training losses, not this direct Gaussian likelihood.""")
add(2, 9, "A $1/\\sqrt N$ rate typically describes an estimation/generalization term under specified capacity and sampling assumptions, not guaranteed decay of total prediction error to zero. Correlated frames also need not provide $N$ independent examples.")
add(2, 12, "This zero-one loss is appropriate for discrete actions and a deterministic expert. For a continuous stochastic policy, exact equality to one demonstrated real vector usually has probability zero; use an appropriate continuous loss or divergence and a corresponding stability analysis.")
add(2, 14, coupling)
add(2, 16, dagger)
replace(2, "Slides 28-31 show the exact decomposition", "Slides 28-31 display the following heuristic decomposition, which is not generally exact under the stated assumption (see the corrected proof below)")
old = r"""### Visitation-distribution decomposition

$$
p_{\pi_\theta,t}(s)=
(1-\epsilon)^t p_{\mathrm{train}}(s)
+\left[1-(1-\epsilon)^t\right]p_{\mathrm{mistake}}(s).
$$"""
replace(2, old, r"""### Valid coupling bound

$$
D_{\mathrm{TV}}(d_t^*,d_t^\pi)\le\min\{1,(t-1)\epsilon\}.
$$

Here $t=1$ is the shared initial state; the average expert-distribution error is at most $\epsilon$ at each time. An exact unconditional no-error mixture is not required.""")
replace(2, r"D_{\mathrm{TV}}(p_{\mathrm{train}},p_{\pi_\theta,t})\le\epsilon t,", r"D_{\mathrm{TV}}(d_t^*,d_t^\pi)\le\min\{1,(t-1)\epsilon\},")
replace(2, r"\le\sum_{t=1}^{H}(\epsilon+2\epsilon t)", r"\le\min\left\{H,\epsilon H(H+1)/2\right\}")

add(3, 2, r"""For general partial observability, use the information available before acting, $h_t=(o_{1:t},a_{1:t-1})$, and possibly past rewards if they reveal hidden state. Observation-only history is a restricted shorthand. A policy that conditions on only the immediately preceding token is a bigram next-token model; a unigram model ignores previous tokens.""")
add(3, 3, "The memoryless-optimum statement assumes a standard fully observed MDP with suitable existence conditions, such as finite state/action spaces and bounded discounted rewards. In finite-horizon problems the policy may depend on time, or equivalently on an augmented state containing remaining time. In a POMDP, a belief over latent state is a sufficient information state under the model.")
add(3, 5, dagger)
replace(3, "5. DAgger fixes causal-confusion shift in principle, though pathological correlations may require much data.", "5. DAgger can mitigate causal confusion under adequate expert labeling, policy capacity, and online-learning assumptions; it does not universally eliminate it.")
add(3, 6, "Squared-error training puts the predicted mean between modes. Sampling a fitted Gaussian instead of taking its mean does not repair the distribution: it can still put substantial probability on unsafe intermediate actions. Even a correct multimodal one-step model can switch left/right plans across time; history, a persistent latent plan, or action chunks can help maintain coherence.")
add(3, 7, "The $Kd$ count is the number of logits evaluated along a sampled action, not a guarantee of linear parameter count for every possible joint distribution. Representing arbitrary dependencies can still require a very complex network. Teacher forcing computes training predictions in parallel with a causal mask; ordinary ancestral inference samples components sequentially.")
add(3, 11, r"""The regression target is a sample-specific velocity, not the value of the optimal marginal field for every pair:

$$
v^*(x,\tau,o)
=\mathbb E[x_1-x_0\mid x_\tau=x,\tau,o].
$$

This conditional expectation minimizes squared loss. Different training pairs can pass through the same region with different targets, so the network learns their conditional average. Integrating that field follows the marginal probability path under the usual regularity assumptions; it does not recover each training pair's straight line. Finite data, imperfect fitting, and numerical integration introduce approximation error. See [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).""")
replace(3, r"v^*(x_\tau,\tau)=x_1-x_0.", r"u_{\mathrm{target}}=x_1-x_0,\qquad v^*(x,\tau)=\mathbb E[x_1-x_0\mid x_\tau=x,\tau].")
replace(3, "15. Why do contradictory straight-line supervision targets yield curved generated paths?", "15. Why is the optimal marginal velocity a conditional expectation rather than each sampled pair's displacement?")
replace(3, "- **Action chunking:** predicting and executing a block of consecutive actions before observing again.", "- **Action chunking:** jointly predicting a block of actions, then executing all or a prefix before replanning.")
replace(3, "The parameters are updated by gradient descent on this loss.", "The parameters are updated by gradient descent on this loss. Training samples intermediate points directly; it does not need to integrate the ODE through every training example.")
replace(3, "### C. Narrow versus broad data", "In receding-horizon execution, the policy can execute a shorter prefix than the predicted block and then observe again. The inclusive notation $a_{t:t+K}$ contains $K+1$ actions; a length-$K$ chunk is $a_{t:t+K-1}$. Joint prediction can still help when only one action is executed, although it then gives no reduction in policy-query frequency.\n\n### C. Narrow versus broad data")
replace(3, "## Consolidated takeaways", "Hindsight labels say what happened, not that the actions were optimal or reliably cause that endpoint. Goal-conditioned cloning can copy detours and lucky stochastic outcomes; success on unseen commanded goals is not guaranteed. See Lecture 24 for the additional selection bias that can arise in stochastic hindsight replay.\n\n## Consolidated takeaways")

add(4, 1, "Straight-looking paths alone do not make Euler exact: one step is exact when the initial velocity equals the entire required displacement, as for constant velocity along the path. Reflow trains a new flow using teacher-coupled noise/endpoints to straighten transport; a separate direct sampler can be distilled from those pairs. Reflow does not universally make a learned field exact in one step.")
add(4, 2, "Executing one action still allows joint prediction to affect representation learning and action consistency; it only removes the reduction in policy-query frequency. Executing more actions trades faster inference and coherent plans against delayed feedback. These are empirical benefits, not a guarantee that the original $H^2$ bound improves by substituting a smaller horizon.")
add(4, 5, "Achieved-endpoint labels need not come from originally successful attempts: a failed attempt at A may provide data for the endpoint B. They do not establish shortest paths, robust success, or optimal rewards; conditioning on lucky stochastic outcomes can bias the learned behavior.")
add(4, 6, "A complete control objective also specifies the initial-state distribution and a horizon, discount, or average-reward criterion. Rewards may depend on $(s,a,s')$ or be random; $r(s,a)$ often denotes their conditional mean. Model-based methods can use a known model as well as learn one, and RL can incorporate demonstrations rather than requiring their absence.")
add(4, 8, "A stationary Markov policy induces a time-homogeneous chain. A time-dependent policy induces time-dependent transitions unless time is added to the state. A history-dependent POMDP controller requires its memory/history in the combined chain; latent environment state alone does not necessarily make the closed-loop process Markov.")
add(4, 9, r"Use $\tau=(s_1,a_1,\ldots,s_H,a_H,s_{H+1})$ for the displayed product: $H$ decisions produce $H+1$ states. Marginalizing out $s_{H+1}$ gives a trajectory ending in $(s_H,a_H)$.")
replace(4, r"\mathcal T_\theta^{,t-1}", r"\mathcal T_\theta^{t-1}")
replace(4, "The formula exposes two ways $\\theta$ matters: it changes the transition matrix of the combined policy-environment chain and therefore changes every future visitation marginal.", r"The formula exposes two places where $\theta$ enters: the initial state-action distribution $\mu_{1,\theta}(s,a)=p(s_1=s)\pi_\theta(a\mid s)$, and the transition matrix $\mathcal T_\theta$. Thus $\mu_1$ also depends on $\theta$ even when the environment's initial-state distribution does not. Matrix powers assume a stationary policy; time-dependent policies require a product of time-indexed matrices.")
add(4, 12, r"""For a **finite** chain, a stationary probability vector always exists. Irreducibility gives uniqueness; aperiodicity additionally gives convergence of the ordinary marginals from every start. For example, deterministic alternation between two states has stationary distribution $(1/2,1/2)$, but its marginals oscillate. Time-averaged visitation can nevertheless converge. Disconnected closed classes admit multiple stationary distributions and mixtures of them, so long-run behavior can depend on the start. Infinite state spaces need additional recurrence/existence assumptions.

The fixed-point equation must be accompanied by $\bar\mu\ge0$ and $\mathbf1^\top\bar\mu=1$. Discounted return is a different infinite-horizon objective:

$$
J_\gamma(\pi)=\mathbb E_\pi\sum_{t=1}^\infty\gamma^{t-1}r_t,\qquad 0\le\gamma<1.
$$

Bounded rewards make this sum finite without requiring stationarity or ergodicity. Do not use an undiscounted infinite reward sum as the value function for the average-reward objective.""")
add(4, 13, "Smoothness of the expected return still needs differentiable policy probabilities and conditions permitting differentiation under the expectation. A deterministic threshold policy can yield a nonsmooth objective; sampling alone is not a proof of smoothness. Low imitation loss also need not force visitation distributions to converge.")
add(4, 15, r"""Finite-horizon values are really $V_t^\pi(s)$ and $Q_t^\pi(s,a)$ because remaining time changes the return. With terminal value $V_{T+1}^\pi=0$,

$$
Q_t^\pi(s,a)=r(s,a)+\mathbb E[V_{t+1}^\pi(s')\mid s,a],
\qquad V_t^\pi(s)=\mathbb E_{\pi_t}[Q_t^\pi(s,a)].
$$

The time-free notation assumes time is encoded in state or a stationary discounted formulation. Discounted backups multiply the successor value by $\gamma$.""")
replace(4, r"""\pi'(a\mid s)=1
\quad\text{if}\quad
a\in\arg\max_{a'}Q^\pi(s,a'),""", r"""\pi'(a\mid s)=\mathbf1[a=a^*(s)],
\qquad a^*(s)\in\arg\max_{a'}Q^\pi(s,a'),""")
add(4, 16, "Resolve ties by choosing one maximizer or distributing probability mass over them; assigning probability one to every maximizer is invalid. The policy-improvement theorem requires exact evaluation and improvement at every relevant state (and time for finite horizons). It guarantees nondecreasing value, not a strictly better initial return if improved states are never reached. A finite shared-network gradient step with approximate advantages has no automatic monotonic guarantee.")
add(4, 20, "On-policy describes the relationship between the target policy and behavior distribution used by an estimator. Basic REINFORCE uses fresh batches; PPO uses several local surrogate updates per batch. Off-policy does not mean arbitrary data are sufficient: coverage and estimator assumptions still matter. There is no universal sample-efficiency ranking among these algorithm families.")
replace(4, "2. Flow matching generates actions by integrating an observation-conditioned vector field; reflow can distill many integration steps into one.", "2. Flow matching generates actions by integrating an observation-conditioned field; reflow and distillation can reduce sampling cost, with approximation error.")
replace(4, "14. Greedy or advantage-weighted policy updates improve behavior when values are accurate.", "14. Exact statewise policy improvement has a nondecreasing-value guarantee; approximate shared-network updates need additional care.")

add(21, 3, coupling)
replace(21, "A held-out validation loss is an unbiased empirical estimate of this expectation.", "For a fixed policy independent of validation sampling, a held-out validation loss is an unbiased empirical estimate of this expectation; adaptive model selection can invalidate that qualification.")
replace(21, "This yields the mixture", "The lecture sketches the following mixture; it is not generally valid under the stated average-error assumption, as corrected below:")
replace(21, "The bound saturates at one if the exact expression is retained. The linear approximation is most informative while $\\epsilon t$ is small; once it exceeds one, the trivial TV bound is tighter.", r"The valid average-error bound from Section 3 is $\min\{1,(t-1)\epsilon\}$. The exponential survival bound needs the stronger uniform conditional-error assumption; it does not follow from the average-error assumption. Both bounds saturate at one, and the linear bound is useful before that saturation.")
add(21, 5, dagger)
add(21, 7, "A categorical distribution over the entire joint action set is fully expressive for finite actions. Separate independent categoricals for each coordinate are not: they lose dependencies. Autoregression avoids that independence assumption, but its compact output does not guarantee a small network can represent every joint distribution. Across environment time, persistent intent may still require memory or a latent plan.")
add(21, 8, r"The optimal flow-matching field is $v^*(x,\tau,s)=\mathbb E[x_1-x_0\mid x_\tau=x,\tau,s]$, the conditional average of sampled velocity targets. It need not follow any individual straight training pair. Here $\tau$ is internal generation time. Numerical integration and fitting are approximate; reflow can reduce curvature, and direct distillation can reduce evaluation count, but one-step exactness is not automatic. Hindsight goal cloning teaches achieved outcomes, not necessarily shortest or reliably repeatable paths.")
add(21, 10, "The likelihood-ratio derivation assumes differentiable policy probabilities, sufficient support, integrability, and environment/reward factors independent of the policy parameters. In a score-function update, sampled actions and return weights are held fixed when differentiating log probabilities. For discounted start-state return, the time-$t$ term additionally carries the outside factor $\\gamma^{t-1}$; see Lecture 5.")
add(21, 11, "An action-independent baseline preserves the expected score-function gradient, but does not universally reduce variance. With score $g=\\nabla\\log\\pi(a\\mid s)$ and return $G$, the scalar baseline minimizing conditional gradient variance is $\\mathbb E[\\|g\\|^2G\\mid s]/\\mathbb E[\\|g\\|^2\\mid s]$, not generally $V(s)$. A mean computed from the same samples is action-dependent through self-inclusion; for independent samples it scales the expected gradient by $(N-1)/N$. Detach the baseline/advantage in the actor loss; leave-one-out or independent fitting can avoid that particular bias.")

add(22, 1, "The importance identity requires target support to be covered by behavior support and a fixed target policy for the usual unbiased-estimator statement. Variance can grow severely but need not do so in every problem. An exact candidate-policy gradient needs candidate-policy future values as well as the correct visitation distribution. Reusing old-policy advantages with a one-step ratio defines a local surrogate; it is not made exact merely by repairing the state ratio.")
add(22, 2, "Clipping is pessimistic relative to the **unclipped sampled surrogate**, not a certified lower bound on true return. It does not enforce a hard ratio or KL constraint: shared parameters, unsampled actions, and other loss terms can still move the policy beyond the interval.")
replace(22, "The old policy's value is a constant with respect to $\\theta'$, so replacing a new-policy advantage with the old-policy advantage changes the objective by a harmless constant relationship rather than a crude approximation.", "This is an exact performance-difference identity under **new-policy trajectories**; it follows by telescoping old value terms, not by freely substituting old advantages into a new-policy score-function gradient. Differentiating the identity includes how the new policy changes the whole trajectory distribution.")
add(22, 3, r"""For the displayed discounted identity, index time from $t=0$, or use $\gamma^{t-1}$ if time starts at 1. Replacing new state visitation by old visitation yields a surrogate whose gradient **matches at the old policy** with exact advantages and consistent discount weighting; away from that anchor it generally differs.

Uniform per-state TV control supports the coupling bound. Empirical mean KL over old states does not itself imply this uniform condition, so practical PPO does not inherit an unconditional monotonic-improvement theorem. Keep any KL penalty multiplier nonnegative.

Freeze old log probabilities, the reference value estimates used to form targets, and actor advantages for the batch. A standard target is raw GAE plus the reference value; standardize a separate copy for the actor, not the critic target. Finite-batch normalization is a useful heuristic, not an exact unbiased-baseline argument.""")
add(22, 4, "A3C uses asynchronous workers with local rollout updates to shared parameters; synchronous aggregation is A2C. Correlated samples are not automatically biased or invalid, but reduce effective sample size and can destabilize optimization. One-step TD advantages have correct conditional expectation with the true value function; approximate bootstrapping can bias the actor update. Critic regression holds its target fixed.")
add(22, 5, r"""The bias/variance trends are heuristics, not monotonic laws. Deterministic transitions alone do not imply zero rollout variance: a stochastic policy or random initial state can still create it. A finite rollout with $\lambda=1$ yields a bootstrapped return unless it ends at a true terminal.

For implementation, separate the bootstrap mask $m_t$ from trace continuation $c_t$:

$$
\delta_t=r_t+\gamma m_tV_{\mathrm{ref}}(s_{t+1})-V_{\mathrm{ref}}(s_t),
\qquad
\widehat A_t=\delta_t+\gamma\lambda c_t\widehat A_{t+1}.
$$

At a true terminal, $m_t=c_t=0$. At an external rollout cutoff, bootstrap from the final **pre-reset** observation ($m_t=1$) but stop the trace ($c_t=0$). For $K$ remaining transitions, the finite mixture weights are $(1-\lambda)\lambda^{n-1}$ for $n<K$ and $\lambda^{K-1}$ for the final $K$-step estimate; the last weight collects the remaining mass.""")
add(22, 6, "For a diagonal Gaussian, reparameterization is $a=\\mu_\\theta(s)+\\sigma_\\theta(s)\\odot\\epsilon$. Freeze critic parameters during the actor update while preserving the action derivative through the critic. Optimizing replay-state $Q_\\phi(s,\\pi_\\theta(s))$ is a fixed-critic surrogate; it is not automatically the exact gradient of start-state return. Logged one-step transitions remain valid only for the same Markov dynamics/reward; multi-step behavior returns need suitable policy corrections.")
add(22, 7, "Use a deterministic tie rule for argmax. Include a true-terminal mask in every target, and detach the target network evaluation. Double DQN typically reduces maximization bias; correlated estimators do not guarantee its removal. Neither a target network nor Double Q alone guarantees convergence.")
add(22, 9, r"""The ELBO identity is

$$
\log p_\theta(x)=\mathcal L(\theta,q)+D_{\mathrm{KL}}(q(z)\|p_\theta(z\mid x)).
$$

Increasing the lower bound can reduce its gap while the evidence decreases; evidence monotonicity needs the stronger exact-EM conditions. The derivation needs compatible support and finite expectations. A continuous deterministic decoder is a singular conditional distribution and cannot simply be inserted as an ordinary finite log density into the Gaussian VAE formula. In reparameterization, $\sigma$ denotes standard deviation, not variance.""")
add(22, 10, "A latent representation that reconstructs observations is not thereby a sufficient Markov state. Predictive learning encourages sufficiency but does not guarantee it. A deployed encoder/filter must use only information available at decision time; a training posterior that sees future observations cannot be used unchanged for online control.")
replace(22, r"""q(s_{1:T},a_{1:T})
=p(s_1)
\prod_t p(s_{t+1}\mid s_t,a_t)q(a_t\mid s_t).""", r"""q(s_{1:T+1},a_{1:T})
=p(s_1)
\prod_{t=1}^T p(s_{t+1}\mid s_t,a_t)q(a_t\mid s_t).""")
replace(22, r"\sum_t r(s_t,a_t)+\mathcal H(q(a_t\mid s_t))", r"\sum_t\left[r(s_t,a_t)+\mathcal H(q(\cdot\mid s_t))\right]")
replace(22, r"""\pi(a_t\mid s_t)
=\exp(Q_t(s_t,a_t)-V_t(s_t))
=\exp(A_t(s_t,a_t)).""", r"""\pi(a_t\mid s_t)
=p_0(a_t\mid s_t)\exp(Q_t(s_t,a_t)-V_t(s_t)),
\quad V_t(s)=\log\int p_0(a\mid s)e^{Q_t(s,a)}\,da.""")
add(22, 11, r"""**Prior and normalization.** A proper probability model needs an action prior $p_0(a\mid s)$ in $p(\tau)$. The corrected posterior above retains it; equal Q-values imply equal posterior action probabilities only when their prior weights are equal. Keeping the environment factors fixed gives the objective

$$
\mathbb E_q\sum_t\left[r(s_t,a_t)-\log\frac{q(a_t\mid s_t)}{p_0(a_t\mid s_t)}\right].
$$

Uniform $p_0$ on a fixed finite action set yields reward plus entropy up to a fixed per-step constant, explaining the lecture shorthand. There is no proper uniform probability on all of $\mathbb R^d$. The usual unweighted continuous maximum-entropy objective is a separate entropy convention, not a normalized uniform prior on an unbounded space. Shifting rewards to be nonpositive preserves fixed-length preferences but can change preferences when episode lengths vary.""")
add(22, 12, "The displayed partition-function gradient samples the globally reward-reweighted trajectory distribution. Under stochastic dynamics, this posterior changes the distribution of lucky transitions (and potentially initial states); it is not generally the rollout distribution of a dynamics-preserving SAC policy. Maximum causal entropy formulations handle that distinction. GAIL with a state-action discriminator matches the corresponding occupancy distributions, not necessarily full trajectories. A balanced discriminator is $1/2$ on matched support, not necessarily everywhere outside observed support. SAC targets also require terminal masks.")
add(22, 13, "Ensemble disagreement is an imperfect epistemic-uncertainty estimate: members can agree and all be wrong. Sampling one model for a rollout represents a persistent possible world; resampling a model each step represents different uncertainty. Fresh transition noise models aleatoric randomness. Receding-horizon planning restores feedback by replanning after observations; short imagined rollouts limit but do not eliminate model bias.")

for n, (p, s) in notes.items():
    p.write_text(s, encoding="utf-8", newline="\n")
print("Updated introductory notes and midterm reviews.")
