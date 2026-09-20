import numpy as np
import torch

from edu_rl.algorithms.actor_critic import n_step_estimates
from edu_rl.algorithms.ddpg import ddpg_target
from edu_rl.algorithms.dqn import dqn_target
from edu_rl.algorithms.importance_sampling import bandit_importance_sampling_demo
from edu_rl.algorithms.natural_policy_gradient import conjugate_gradient
from edu_rl.algorithms.ppo import ppo_policy_loss


def test_n_step_estimate_bootstraps_after_requested_number_of_rewards() -> None:
    advantages, targets = n_step_estimates(
        rewards=np.array([1.0, 2.0, 3.0]),
        values=np.array([0.5, 0.6, 0.7]),
        next_values=np.array([0.6, 0.7, 0.0]),
        terminated=np.array([0.0, 0.0, 1.0]),
        gamma=0.9,
        n_steps=2,
    )
    # t=0 uses 1 + .9*2 + .9^2*V(s2); t=1 reaches a terminal.
    np.testing.assert_allclose(targets, [3.367, 4.7, 3.0], atol=1e-6)
    np.testing.assert_allclose(advantages, targets - [0.5, 0.6, 0.7])


def test_double_dqn_separates_selection_from_evaluation() -> None:
    rewards = torch.tensor([0.0])
    terminated = torch.tensor([0.0])
    online = torch.tensor([[1.0, 5.0]])
    target = torch.tensor([[10.0, 2.0]])
    standard = dqn_target(
        rewards, terminated, online, target, gamma=1.0, double_dqn=False
    )
    double = dqn_target(
        rewards, terminated, online, target, gamma=1.0, double_dqn=True
    )
    torch.testing.assert_close(standard, torch.tensor([10.0]))
    torch.testing.assert_close(double, torch.tensor([2.0]))


def test_ddpg_target_masks_only_true_termination() -> None:
    actual = ddpg_target(
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([10.0, 10.0]),
        gamma=0.9,
    )
    torch.testing.assert_close(actual, torch.tensor([1.0, 10.0]))


def test_ppo_ratio_is_one_before_policy_changes() -> None:
    old_log_probability = torch.tensor([-0.2, -1.2])
    advantages = torch.tensor([1.0, -0.5])
    loss, approximate_kl, clip_fraction = ppo_policy_loss(
        old_log_probability,
        old_log_probability,
        advantages,
        method="clip",
        clip_ratio=0.2,
        kl_beta=1.0,
    )
    torch.testing.assert_close(loss, -advantages.mean())
    torch.testing.assert_close(approximate_kl, torch.tensor(0.0))
    torch.testing.assert_close(clip_fraction, torch.tensor(0.0))


def test_conjugate_gradient_solves_positive_definite_system() -> None:
    matrix = torch.diag(torch.tensor([2.0, 4.0]))
    vector = torch.tensor([2.0, 8.0])
    solution = conjugate_gradient(lambda value: matrix @ value, vector, iterations=5)
    torch.testing.assert_close(solution, torch.tensor([1.0, 2.0]), atol=1e-5, rtol=1e-5)


def test_importance_sampling_recovers_target_bandit_value() -> None:
    result = bandit_importance_sampling_demo(samples=100_000, seed=7)
    assert abs(result["ordinary_is"] - result["exact"]) < 0.02
    assert abs(result["weighted_is"] - result["exact"]) < 0.02
    assert abs(result["uncorrected_behavior_mean"] - 0.2) < 0.02

