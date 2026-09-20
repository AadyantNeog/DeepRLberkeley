import numpy as np
import torch

from edu_rl.advanced.exploration import RNDModule, run_ucb
from edu_rl.advanced.her import GoalTransition, HERReplayBuffer
from edu_rl.advanced.imitation_skills import diayn_reward
from edu_rl.advanced.model_based import cross_entropy_method
from edu_rl.advanced.offline_rl import advantage_weights, expectile_loss
from edu_rl.advanced.preference_rl import (
    bradley_terry_loss,
    group_relative_advantages,
)
from edu_rl.advanced.sac import soft_bellman_target
from edu_rl.advanced.transfer_hierarchy import semi_mdp_target


def test_sac_soft_target_includes_entropy_and_terminal_mask() -> None:
    target = soft_bellman_target(
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([3.0, 3.0]),
        torch.tensor([-2.0, -2.0]),
        gamma=0.9,
        temperature=torch.tensor(0.2),
    )
    torch.testing.assert_close(target, torch.tensor([4.06, 1.0]))


def test_cem_finds_quadratic_maximum() -> None:
    optimum = np.array([0.25, -0.5], dtype=np.float32)
    solution, value = cross_entropy_method(
        lambda samples: -np.square(samples - optimum).sum(axis=1),
        dimension=2,
        lower=-1.0,
        upper=1.0,
        population_size=500,
        iterations=6,
        seed=2,
    )
    np.testing.assert_allclose(solution, optimum, atol=0.05)
    assert value > -0.01


def test_iql_expectile_and_awac_weights() -> None:
    residual = torch.tensor([2.0, -2.0])
    loss = expectile_loss(residual, 0.7)
    torch.testing.assert_close(loss, torch.tensor(2.0))
    weights = advantage_weights(torch.tensor([0.0, 10.0]), 1.0, 5.0)
    torch.testing.assert_close(weights, torch.tensor([1.0, 5.0]))


def test_ucb_initializes_every_arm_and_concentrates_on_best() -> None:
    result = run_ucb(
        np.array([0.0, 0.2, 1.0]),
        steps=500,
        noise_standard_deviation=0.1,
        seed=3,
    )
    assert np.all(result.counts > 0)
    assert result.counts.argmax() == 2


def test_rnd_target_is_frozen() -> None:
    rnd = RNDModule(4, 8)
    assert all(not parameter.requires_grad for parameter in rnd.target.parameters())
    assert all(parameter.requires_grad for parameter in rnd.predictor.parameters())


def test_her_adds_future_goal_relabels() -> None:
    buffer = HERReplayBuffer(100, n_bits=2, her_k=2, seed=0)
    episode = [
        GoalTransition(
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            0,
            np.array([1.0, 0.0]),
        ),
        GoalTransition(
            np.array([1.0, 0.0]),
            np.array([1.0, 1.0]),
            1,
            np.array([1.0, 1.0]),
        ),
    ]
    buffer.add_episode(episode)
    assert len(buffer) == len(episode) * 3


def test_diayn_uniform_discriminator_has_zero_information_reward() -> None:
    rewards = diayn_reward(torch.zeros((4, 4)), torch.arange(4), 4)
    torch.testing.assert_close(rewards, torch.zeros(4))


def test_group_advantages_are_prompt_local_and_zero_mean() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])
    advantages = group_relative_advantages(rewards)
    torch.testing.assert_close(advantages.mean(dim=1), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(advantages[1], torch.zeros(3))


def test_bradley_terry_prefers_larger_score() -> None:
    good = bradley_terry_loss(torch.tensor([3.0]), torch.tensor([0.0]), torch.tensor([1.0]))
    bad = bradley_terry_loss(torch.tensor([0.0]), torch.tensor([3.0]), torch.tensor([1.0]))
    assert good < bad


def test_semi_mdp_backup_discounts_by_option_duration() -> None:
    target = semi_mdp_target(2.0, 3, 10.0, gamma=0.5, terminal=False)
    assert target == 3.25

