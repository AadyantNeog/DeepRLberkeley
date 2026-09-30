---
title: "Lecture 21 - Midterm Review, Part 1"
course: "CS 185/285 Deep Reinforcement Learning (Spring 2026)"
lecture: 21
source_transcript: "../transcripts/CS 185285 (Spring 2026) Lecture 21, Midterm Review 1.txt"
source_slides: "../lectures/Lecture 21 - Midterm Review Part 1.pdf"
transcript_lines: 4441
slide_pages: 34
status: "complete"
source_qualifier: "unspoken-slides-isolated"
---

# Lecture 21: Midterm Review, Part 1

> **Source note:** The supplied transcript ends normally when the lecturer says the review will continue on Friday. Slides 17-34 of this deck were not reached in the recorded session. They are identified in a slide-only appendix and are not attributed to the lecturer's spoken account.

## Lecture map

| Section | Topic | Transcript coverage |
|---:|---|---:|
| 1 | Review purpose and behavioral cloning | lines 1-244 |
| 2 | Zero-one imitation cost and expected mistakes | lines 245-493 |
| 3 | Generalization assumption and distribution shift | lines 494-1024 |
| 4 | Total-variation bound on state-distribution drift | lines 1025-1558 |
| 5 | Quadratic horizon error and DAgger | lines 1559-2053 |
| 6 | DAgger in practice and model-based fixes | lines 2054-2299 |
| 7 | Memory, multimodality, and autoregressive actions | lines 2300-2842 |
| 8 | Flow matching, reflow, and goal conditioning | lines 2843-3592 |
| 9 | Goal-conditioned BC Q&A and policy-gradient setup | lines 3593-3859 |
| 10 | Direct policy-gradient derivation | lines 3860-4162 |
| 11 | Baselines and closing Q&A | lines 4163-4441 |

## 1. Review purpose and behavioral cloning

**Transcript coverage:** lines 1-244

### What the lecturer said - transcript only

This lecture introduced no new material. It was a retrospective intended both to prepare students for the exam and to place earlier topics in a clearer context now that the class had seen several families of RL methods. Questions and requests for clarification were especially encouraged.

The first reviewed family was imitation learning. Standard behavioral cloning is methodologically simple:

1. Collect demonstration trajectories containing observations and actions.
2. Fit a policy by supervised maximum likelihood.

For demonstrations $\{(o_t^{(i)},a_t^{(i)})\}$, the objective is

$$
\max_\theta
\sum_i\sum_t
\log\pi_\theta(a_t^{(i)}\mid o_t^{(i)}).
$$

The policy does not normally output an action directly. It outputs parameters of an action distribution—for example, logits for a categorical distribution or a mean and covariance for a Gaussian—and an action is sampled from that distribution. Behavioral cloning is both the most basic and, according to the lecturer, the most widely used imitation-learning method.

To analyze whether it is good or bad, one must first define a loss. The lecture used a theoretical zero-one cost,

$$
c(s_t,a_t)=
\begin{cases}
0,&a_t=\pi^*(s_t),\\
1,&\text{otherwise},
\end{cases}
$$

where $\pi^*$ denotes the demonstrator. This cost is an analytical device rather than an extra quantity used by the behavioral-cloning algorithm.

### Source reconciliation

Slides 2-4 show the two-step algorithm, the distribution-parameter output, and the zero-one cost. The transcript adds examples of categorical and Gaussian outputs and distinguishes the analytical loss from the training procedure.

### Additional explanation

Maximum likelihood is equivalent to minimizing cross-entropy or negative log likelihood on the demonstrations. It answers “how well can the model reproduce demonstrated actions on demonstration-like inputs?” The analysis below asks the harder sequential question: “what happens after the learned policy changes the inputs it sees?”

## 2. Zero-one imitation cost and expected mistakes

**Transcript coverage:** lines 245-493

### What the lecturer said - transcript only

For a stochastic demonstrator, the comparison is more nuanced; total-variation divergence is a natural generalization, and it reduces to the zero-one mismatch for a deterministic expert. The exam-level analysis in this lecture used the deterministic, discrete-action version. In response to a question, the lecturer said that exact equality of continuous actions is not a useful zero-one criterion, although the ideas can be extended with a different loss.

At each time $t$, the learned policy induces a state distribution $p_{\pi_\theta}(s_t)$. That distribution depends on all earlier actions, so the expected cost is

$$
\sum_{t=1}^{H}
\mathbb E_{
s_t\sim p_{\pi_\theta}(s_t),
a_t\sim\pi_\theta(\cdot\mid s_t)
}
[c(s_t,a_t)].
$$

With zero-one cost, the expected cost at one time step is the probability of making a mistake at that step and lies between zero and one. Summing through the horizon gives the expected total number of mistakes, which lies between zero and $H$. The central question is therefore how this total scales with horizon.

The lecturer described each time-indexed state distribution as a cloud around a trajectory. A stochastic or unstable policy may make the cloud spread over time; a feedback policy may correct errors. The expected cost integrates over the policy's own cloud, not merely the demonstration states.

### Source reconciliation

Slides 4-5 show the zero-one imitation cost, the policy-induced state distributions, and the expected-mistake objective. The transcript supplies the continuous-action Q&A and the “cloud” interpretation.

### Additional explanation

This evaluation differs from ordinary validation error because deployment is sequential. One wrong action can change the next state, which changes the input to every later prediction.

## 3. Generalization assumption and distribution shift

**Transcript coverage:** lines 494-1024

### What the lecturer said - transcript only

The proof begins by assuming that supervised learning works on the training distribution:

$$
\mathbb E_{s_t\sim p_{\mathrm{train}}(s_t)}
\left[
\Pr_{a_t\sim\pi_\theta(\cdot\mid s_t)}
(a_t\ne\pi^*(s_t))
\right]
\le\epsilon.
$$

This is the population zero-one validation loss on the distribution that produced the demonstrations. The notation $p_{\mathrm{train}}$ means the training *distribution*, not the finite training set. For a fixed policy independent of validation sampling, a held-out validation loss is an unbiased empirical estimate of this expectation; adaptive model selection can invalidate that qualification.

The proof's main difficulty is not converting an action mismatch into a cost; it is relating action-error probability on $p_{\mathrm{train}}$ to the state probabilities generated by the learned policy.

If the learned policy takes the same action as the expert at every preceding step, it evolves under the same transition probabilities and reaches the expert's state distribution. If it makes at least one mistake, the resulting state distribution may be arbitrary in a worst-case analysis. The lecture sketches the following mixture; it is not generally valid under the stated average-error assumption, as corrected below:

$$
p_{\pi_\theta}(s_t)
=(1-\epsilon)^t p_{\mathrm{train}}(s_t)
+\left[1-(1-\epsilon)^t\right]p_{\mathrm{mistake}}(s_t).
$$

The important observation is that the probability of no mistake for $t$ steps is $(1-\epsilon)^t$. The second component collects every history containing at least one mistake; the proof deliberately makes no favorable assumption about where those histories go.

The lecturer acknowledged that writing the mixture as an exact equality is subtle in expectation, but emphasized the no-mistake versus at-least-one-mistake decomposition as the creative step. The remaining proof is mostly algebra.

### Source reconciliation

Slide 5 states the on-training-distribution error assumption and the mixture identity. The transcript clarifies that the probability sits inside the expectation and can be understood as a sum of policy mass over all non-expert actions.

### Additional explanation

This is a coupling argument: imagine executing the expert and learner with shared environment randomness until their actions disagree. Their states remain coupled before the first disagreement; afterward, the analysis treats them as unrelated.

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

## 4. Total-variation bound on state-distribution drift

**Transcript coverage:** lines 1025-1558

### What the lecturer said - transcript only

Total-variation divergence is

$$
D_{\mathrm{TV}}(p,q)
=\frac12\sum_x|p(x)-q(x)|,
$$

and always lies between zero and one. Substituting the mixture for $p_{\pi_\theta}$ into the divergence from $p_{\mathrm{train}}$, canceling the shared no-mistake component, and using the worst-case bound $D_{\mathrm{TV}}\le1$ gives

$$
D_{\mathrm{TV}}
\left(p_{\mathrm{train}}(s_t),p_{\pi_\theta}(s_t)\right)
\le 1-(1-\epsilon)^t.
$$

For $\epsilon\in[0,1]$, Bernoulli's inequality gives

$$
1-(1-\epsilon)^t\le\epsilon t.
$$

The last step deliberately loosens the bound but makes its linear growth in time easy to interpret. What grows is not necessarily the width of the policy's state cloud; it is the discrepancy, or decreasing overlap, between the training and deployment state distributions.

In Q&A, the lecturer explained that total variation makes the algebra simple because it is based on absolute probability differences. KL divergence introduces logarithms. In later PPO analysis, one can first prove a total-variation relationship and then relate it to KL through an inequality such as Pinsker's. Retaining the exponential expression would be valid; the linear upper bound is used for interpretability.

### Source reconciliation

Slides 6-8 show the full cancellation and the conversion to $\epsilon t$. The transcript adds the geometric interpretation of overlap and explains why TV rather than KL is used at this step.

### Additional explanation

The valid average-error bound from Section 3 is $\min\{1,(t-1)\epsilon\}$. The exponential survival bound needs the stronger uniform conditional-error assumption; it does not follow from the average-error assumption. Both bounds saturate at one, and the linear bound is useful before that saturation.

## 5. Quadratic horizon error and DAgger

**Transcript coverage:** lines 1559-2053

### What the lecturer said - transcript only

The analysis is worst case: after one error, it permits the learner to enter an unrelated distribution and never recover, like a tightrope walker who falls and cannot return. Real systems often can recover, which helps explain why behavioral cloning may work better than this bound suggests. Incorporating recovery is possible but substantially more complicated and was not exam material.

To convert state-distribution drift into cost, write the policy-state expectation as a sum over states, add and subtract $p_{\mathrm{train}}(s_t)$, and split the result. The training-distribution term is bounded by $\epsilon$. The distribution-difference term is at most twice TV because $c(s,a)\le1$. Thus the per-step cost is bounded on the order of $\epsilon+2\epsilon t$. Summing over $H$ steps yields

$$
\mathbb E[\text{total mistakes}]
=O(\epsilon H^2).
$$

One factor of $H$ comes from distribution shift growing over time; the other comes from summing costs over $H$ time steps. Keeping $1-(1-\epsilon)^t$ would give a more exact but less interpretable expression. The key classification is that the worst-case error is superlinear in horizon.

The principled repair is DAgger, or Dataset Aggregation. Train on demonstrations, execute the learned policy, ask the expert to label the observations that the learner actually encountered, add those labeled observations to the dataset, retrain, and repeat. The learner's executed actions need not be good; what matters is obtaining correct labels for its own visited states.

### Source reconciliation

Slides 7-9 derive the cost bound and mark $O(\epsilon H^2)$ as undesirable. Slide 9 then gives the four-step DAgger loop.

### Additional explanation

DAgger changes the data distribution rather than merely improving the classifier on the old one. It converts part of the sequential covariate-shift problem into an interactive supervised-learning problem.

**What DAgger guarantees.** Expert labels alone are insufficient: the policy class must fit them adequately, and the supervised updates must satisfy the online-learning/no-regret assumptions. The analysis controls average learner-distribution loss over iterations and supports a suitable mixture or selected iterate; it does not automatically guarantee the final neural-network checkpoint or convergence of visitation distributions. If the deployed policy's average disagreement on its *own* states is $\epsilon$, its expected disagreement count is $H\epsilon$. Translating disagreement into excess task cost additionally needs a bound on the expert's cost-to-go sensitivity to a wrong action, often related to recoverability. See the [original DAgger analysis](https://proceedings.mlr.press/v15/ross11a.html).

## 6. DAgger in practice and model-based fixes

**Transcript coverage:** lines 2054-2299

### What the lecturer said - transcript only

Adding relabeled observations from the deployed policy gradually makes the dataset resemble the learner's actual visitation distribution while retaining expert action labels. The original 2011 analysis by Stéphane Ross and collaborators uses online-learning theory to quantify the required iterations. In practice, even one iteration can help substantially. There are many variants for choosing states to label, but the essential condition is that the dataset include observations the current policy actually sees.

A second, less principled but common response is to make the supervised model so accurate that $\epsilon$ becomes extremely small. Even an $H^2$ multiplier is less damaging when the one-step error is nearly zero. Two important sources of residual error are non-Markovian demonstrations and multimodal actions.

### Source reconciliation

Slide 9 gives the DAgger loop. The transcript adds the practical observation that even one aggregation round may help and separates data-distribution repair from reducing $\epsilon$ with a stronger policy model.

### Additional explanation

These two repairs attack different factors. DAgger reduces deployment/train distribution mismatch; richer models reduce conditional prediction error on whatever distribution is supplied. They can be combined.

## 7. Memory, multimodality, and autoregressive actions

**Transcript coverage:** lines 2300-2842

### What the lecturer said - transcript only

A memoryless policy sees only the present observation, whereas a human demonstrator's choice may depend on prior events even when the task itself could be solved without memory. The direct repair is to provide observation history—by concatenating a short history when feasible or by using a sequence model for longer or larger observations.

Multimodality is primarily problematic for continuous actions. A categorical distribution can represent an arbitrary distribution over a finite action set, but a Gaussian has one mode. If demonstrations sometimes pass an obstacle on the left and sometimes on the right, a unimodal fit may average the two strategies and drive into the obstacle.

Two general expressive solutions reviewed were autoregressive discretization and flow matching. Autoregressive discretization avoids an exponentially large joint grid. For an action $a_t=(a_{t,0},a_{t,1},a_{t,2})$, it factors

$$
p(a_t\mid s_t)
=p(a_{t,0}\mid s_t)
p(a_{t,1}\mid s_t,a_{t,0})
p(a_{t,2}\mid s_t,a_{t,0},a_{t,1}).
$$

Each coordinate is discretized and predicted with a softmax by a sequence model. Every later coordinate conditions on *all* earlier coordinates, so this is the chain rule rather than an independence assumption and can represent any joint distribution up to discretization resolution.

The coordinate order introduces an architectural inductive bias. The lecturer said there is sometimes a plausible task interpretation, but generally it should be regarded as a neural-network design trick whose best ordering is not well understood.

Training uses teacher forcing: feed the ground-truth earlier action coordinates and maximize likelihood. Sampling is autoregressive: sample the first coordinate, feed it back, then sample the next.

### Source reconciliation

Slides 10-11 depict non-Markovian and multimodal failure cases and display the autoregressive factorization. The transcript supplies the discrete-versus-continuous distinction and the training/sampling Q&A.

### Additional explanation

Autoregression trades one exponential classification problem for a sequence of small classification problems. Its cost is sequential action generation and sensitivity to the arbitrary coordinate order.

A categorical distribution over the entire joint action set is fully expressive for finite actions. Separate independent categoricals for each coordinate are not: they lose dependencies. Autoregression avoids that independence assumption, but its compact output does not guarantee a small network can represent every joint distribution. Across environment time, persistent intent may still require memory or a latent plan.

## 8. Flow matching, reflow, and goal conditioning

**Transcript coverage:** lines 2843-3592

### What the lecturer said - transcript only

Flow matching represents a complex action distribution through a learned velocity field rather than directly predicting probabilities or samples. To sample, draw $x_0\sim\mathcal N(0,I)$ and numerically integrate

$$
x_{t+\Delta t}
=x_t+v_\theta(x_t,t)\Delta t
$$

until the noise has been transformed into a sample from the target distribution.

Training avoids backpropagating through the full integration. Draw independent noise $x_0$ and a data point $x_1$, sample an interpolation time $t$, form

$$
x_t=(1-t)x_0+tx_1,
$$

and regress the velocity field toward

$$
u_t=x_1-x_0.
$$

At a heavily noised point, many target data points are possible and velocity targets point in many directions. Near the data distribution, the compatible targets become more localized, so the average field bends toward the appropriate region. This explains why the integrated path need not be the original straight interpolation line.

Backpropagating through all integration steps is technically possible but numerically unattractive: a many-layer velocity network repeated for many solver steps behaves like an extremely deep network. The optional reflow discussion was not exam material. Reflow first runs the learned integration to create dependent $(x_0,x_1)$ pairs and then distills a more direct mapping; it does not require differentiating through the solver. Flow matching can work in very high dimensions, including video generation.

The final imitation method was goal-conditioned behavioral cloning. Treat each trajectory as a successful demonstration for the state it actually reached and train

$$
\pi_\theta(a_t\mid s_t,g),
\qquad g=s_T,
$$

or condition on a relevant function of the final state, such as position rather than the entire physical state. Heterogeneous, non-optimal trajectories can thereby become demonstrations for different achieved goals, and the model may generalize across goals.

### Source reconciliation

Slides 12-13 give the flow-matching sampling/training recipes and the goal-conditioned relabeling objective. The transcript provides the geometric intuition, solver-gradient Q&A, reflow aside, and high-dimensional example.

### Additional explanation

Goal relabeling changes the question attached to a trajectory: instead of asking whether it was optimal for a predetermined target, it asks which target the behavior actually achieved. It is useful only when tasks can be parameterized by such goals.

The optimal flow-matching field is $v^*(x,\tau,s)=\mathbb E[x_1-x_0\mid x_\tau=x,\tau,s]$, the conditional average of sampled velocity targets. It need not follow any individual straight training pair. Here $\tau$ is internal generation time. Numerical integration and fitting are approximate; reflow can reduce curvature, and direct distillation can reduce evaluation count, but one-step exactness is not automatic. Hindsight goal cloning teaches achieved outcomes, not necessarily shortest or reliably repeatable paths.

## 9. Goal-conditioned BC Q&A and policy-gradient setup

**Transcript coverage:** lines 3593-3859

### What the lecturer said - transcript only

The goal can be any suitable function of the terminal state. Goal-conditioned BC is attractive when one wants a simple imitation method instead of RL; its implementation is only slightly more complex than ordinary behavioral cloning. It does not use a reward or an RL update. It relabels each trajectory with a task for which that trajectory may be close to optimal and relies on generalization over the conditioning variable.

The review then moved to policy gradients. Students were advised to understand the basic derivation. The probability of a trajectory is

$$
p_\theta(\tau)
=p(s_1)
\prod_{t=1}^{H}
\pi_\theta(a_t\mid s_t)
p(s_{t+1}\mid s_t,a_t),
$$

so

$$
\log p_\theta(\tau)
=\log p(s_1)
+\sum_{t=1}^{H}\log\pi_\theta(a_t\mid s_t)
+\sum_{t=1}^{H}\log p(s_{t+1}\mid s_t,a_t).
$$

### Source reconciliation

Slides 13-14 bridge goal-conditioned behavioral cloning to the policy-gradient review and display the trajectory factorization. The transcript supplies the goal-definition Q&A and the reminder that the method remains imitation learning without a reward-based update.

### Additional explanation

This factorization is the bridge between a trajectory-level objective and action-level policy gradients. The environment terms appear in the trajectory probability but disappear when differentiating with respect to policy parameters.

## 10. Direct policy-gradient derivation

**Transcript coverage:** lines 3860-4162

### What the lecturer said - transcript only

Starting from

$$
J(\theta)
=\mathbb E_{\tau\sim p_\theta(\tau)}[r(\tau)]
=\sum_\tau p_\theta(\tau)r(\tau),
$$

the key log-derivative identity is

$$
\nabla_\theta p_\theta(\tau)
=p_\theta(\tau)\nabla_\theta\log p_\theta(\tau).
$$

It follows directly from the derivative of a logarithm. Substitution converts the gradient back into an expectation:

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim p_\theta}
\left[
\nabla_\theta\log p_\theta(\tau)r(\tau)
\right].
$$

Only the policy factors in $p_\theta(\tau)$ depend on $\theta$. The initial-state and transition probabilities are unknown in model-free RL but their derivatives are zero, so they are not needed:

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim p_\theta}
\left[
\left(\sum_{t=1}^{H}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right)
\left(\sum_{t=1}^{H}r(s_t,a_t)\right)
\right].
$$

The score gradient is a vector with the same dimension as the parameter vector. The lecturer described the two ingredients to remember as the trajectory factorization and the log-derivative identity.

### Source reconciliation

Slides 14-15 mark the transition to policy gradients and display the direct derivation. Part of the expectation-to-sum step was written on the board because the slide sequence did not show it explicitly.

### Additional explanation

This estimator does not differentiate through the environment. It differentiates the log probability with which the policy produced the sampled actions and weights that score by observed return.

The likelihood-ratio derivation assumes differentiable policy probabilities, sufficient support, integrability, and environment/reward factors independent of the policy parameters. In a score-function update, sampled actions and return weights are held fixed when differentiating log probabilities. For discounted start-state return, the time-$t$ term additionally carries the outside factor $\gamma^{t-1}$; see Lecture 5.

## 11. Baselines and closing Q&A

**Transcript coverage:** lines 4163-4441

### What the lecturer said - transcript only

Raw policy gradient multiplies every sampled trajectory's score by its return. If all returns are huge but differ only slightly—for example, $1{,}000{,}000$ and $1{,}000{,}001$—all weights are nearly the same and the useful preference signal is poor. Subtracting an average makes above-average trajectories positive and below-average trajectories negative.

This does not bias the expected gradient because, for any constant $b$,

$$
\mathbb E_{\tau\sim p_\theta}
[\nabla_\theta\log p_\theta(\tau)b]
=b\nabla_\theta\int p_\theta(\tau)d\tau
=b\nabla_\theta 1
=0.
$$

The finite-sample estimator changes and usually has lower variance even though its expectation is unchanged. Proving the variance reduction was not exam material. Average return is a reasonable but not optimal baseline.

A baseline may be a function of state as long as it does not depend on the sampled action, because the zero-expectation argument then applies to the action distribution. The lecturer said the Friday review would cover this case. Vector-valued baselines—different baselines for different parameter dimensions—can also be used and lead to an optimal-baseline derivation. In response to a final question, he knew of no particular benefit to choosing a scalar baseline merely larger than the mean return.

The lecture stopped here and would continue Friday with actor-critic and related material.

### Source reconciliation

Slide 16 displays the one-line zero-expectation proof. Slides 17-34 were not discussed in this transcript and are separated below.

### Additional explanation

Baselines are control variates. They remove variability correlated with the score without changing the target expectation. A learned state-value function is especially effective because it predicts the portion of return explainable from the state before the current action is selected.

An action-independent baseline preserves the expected score-function gradient, but does not universally reduce variance. With score $g=\nabla\log\pi(a\mid s)$ and return $G$, the scalar baseline minimizing conditional gradient variance is $\mathbb E[\|g\|^2G\mid s]/\mathbb E[\|g\|^2\mid s]$, not generally $V(s)$. A mean computed from the same samples is action-dependent through self-inclusion; for independent samples it scales the expected gradient by $(N-1)/N$. Detach the baseline/advantage in the actor loss; leave-one-out or independent fitting can avoid that particular bias.

## Slide-only appendix: material not reached in this session

The following topics appear on slides 17-34 but were not spoken in the supplied Lecture 21 transcript. They are listed for source completeness only:

- off-policy importance sampling and a first-order approximation;
- clipped importance weights and the PPO clipped surrogate;
- policy-gradient improvement viewed as policy iteration;
- bounding policy-induced state-distribution change with total variation;
- a KL-constrained PPO form;
- online actor-critic, temporal-difference targets, eligibility traces, and $n$-step returns;
- generalized advantage estimation (GAE);
- policy gradient with GAE;
- off-policy reparameterized actor-critic.

These concepts are not folded into “what the lecturer said” because the transcript explicitly stops before them.

## Consolidated takeaways

- Behavioral cloning is supervised maximum likelihood, but its deployment error is sequential because actions change future inputs.
- Under a worst-case no-recovery assumption, state-distribution drift grows with time and total imitation error scales as $O(\epsilon H^2)$.
- DAgger addresses covariate shift by labeling learner-visited states, subject to the expert, capacity, and online-learning assumptions.
- Memory and expressive multimodal action distributions can make the one-step error $\epsilon$ much smaller.
- Autoregressive discretization and flow matching are two general ways to model complex continuous-action distributions.
- Goal relabeling turns heterogeneous trajectories into demonstrations for achieved goals without introducing an RL reward.
- Direct policy gradient follows from the trajectory factorization plus the log-derivative identity; environment dynamics cancel from the derivative.
- A baseline has zero expected score-weighted contribution and can reduce estimator variance.

## Key equations

1. **Behavioral-cloning likelihood**

   $$
   \max_\theta\sum_{i,t}\log\pi_\theta(a_t^{(i)}\mid o_t^{(i)}).
   $$

2. **State-distribution drift**

   $$
   D_{\mathrm{TV}}(p_{\mathrm{train}}(s_t),p_{\pi_\theta}(s_t))
   \le\min\{1,(t-1)\epsilon\}.
   $$

3. **Worst-case behavioral-cloning error**

   $$
   \mathbb E[\text{mistakes}]=O(\epsilon H^2).
   $$

4. **Autoregressive action factorization**

   $$
   p(a_t\mid s_t)=\prod_j p(a_{t,j}\mid s_t,a_{t,<j}).
   $$

5. **Flow-matching training pair**

   $$
   x_t=(1-t)x_0+tx_1,
   \qquad u_t=x_1-x_0.
   $$

6. **Policy gradient**

   $$
   \nabla_\theta J
   =\mathbb E_{\tau\sim p_\theta}
   \left[\nabla_\theta\log p_\theta(\tau)r(\tau)\right].
   $$

## Glossary

- **Behavioral cloning:** Supervised learning of a policy from demonstration observation-action pairs.
- **Zero-one loss:** Cost of zero for matching a reference action and one otherwise.
- **Covariate shift:** Difference between the input distribution used for training and the one encountered at deployment.
- **Total-variation divergence:** Half the one-norm distance between two probability distributions.
- **DAgger:** Interactive dataset aggregation using learner-visited observations and expert labels.
- **Non-Markovian behavior:** Action choice that depends on history not contained in the current observation.
- **Multimodal distribution:** Distribution with multiple distinct high-density behaviors.
- **Autoregressive discretization:** Sequential categorical prediction of continuous-action coordinates.
- **Flow matching:** Learning a velocity field that transports a simple noise distribution to a data distribution.
- **Goal-conditioned policy:** Policy whose action depends on both current state and desired goal.
- **Log-derivative trick:** Identity $\nabla p=p\nabla\log p$ used for score-function gradients.
- **Baseline:** Action-independent quantity subtracted from return to reduce policy-gradient variance.

## Self-check questions

1. Why does validation error on demonstration states not directly predict rollout error?
2. Derive the first-disagreement coupling bound. Why is the lecture's unconditional mixture not generally valid?
3. Where do the two factors of $H$ in $O(\epsilon H^2)$ arise?
4. Which part of DAgger must come from the learner, and which part must come from the expert?
5. Why can a Gaussian policy fail on left-or-right demonstrations?
6. Why does autoregressive factorization not assume independent action coordinates?
7. Describe flow-matching training and sampling separately.
8. Why is goal-conditioned behavioral cloning still imitation learning rather than RL?
9. Derive the policy gradient without differentiating the transition dynamics.
10. Prove that subtracting a constant baseline is unbiased.

## Source coverage checklist

- [x] All supplied transcript lines 1-4441 are mapped exactly once in increasing, non-overlapping ranges.
- [x] Audience questions, optional asides, and the normal stopping point are retained.
- [x] Transcript, slide-only material, and added explanation are explicitly separated.
- [x] All 34 slide pages were rendered and visually inspected.
- [x] Displayed mathematics uses Markdown-compatible LaTeX delimiters.

**Coverage result:** All 4,441 supplied transcript lines are assigned once in ascending order, with no gaps or overlaps. Unspoken slides 17-34 are isolated from the transcript-only account.
