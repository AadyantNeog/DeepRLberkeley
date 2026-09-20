import numpy as np
import torch

from edu_rl.core.buffers import ReplayBuffer
from edu_rl.core.networks import SquashedGaussianActor
from edu_rl.core.returns import discounted_returns, generalized_advantage_estimate


def test_discounted_returns_matches_hand_calculation() -> None:
    actual = discounted_returns([1.0, 2.0, 3.0], gamma=0.5)
    np.testing.assert_allclose(actual, [2.75, 3.5, 3.0])


def test_gae_matches_lecture_example() -> None:
    advantages, targets = generalized_advantage_estimate(
        rewards=np.array([0.0, 0.0, 1.0]),
        values=np.array([0.2, 0.4, 0.6]),
        next_values=np.array([0.4, 0.6, 0.0]),
        terminated=np.array([0.0, 0.0, 1.0]),
        episode_ends=np.array([0.0, 0.0, 1.0]),
        gamma=1.0,
        gae_lambda=0.5,
    )
    np.testing.assert_allclose(advantages, [0.4, 0.4, 0.4], atol=1e-6)
    np.testing.assert_allclose(targets, [0.6, 0.8, 1.0], atol=1e-6)


def test_truncation_stops_trace_but_keeps_bootstrap() -> None:
    advantages, targets = generalized_advantage_estimate(
        rewards=np.array([1.0]),
        values=np.array([0.5]),
        next_values=np.array([0.25]),
        terminated=np.array([0.0]),
        episode_ends=np.array([1.0]),
        gamma=0.9,
        gae_lambda=0.95,
    )
    np.testing.assert_allclose(advantages, [0.725])
    np.testing.assert_allclose(targets, [1.225])


def test_replay_buffer_shapes_and_terminal_flag() -> None:
    replay = ReplayBuffer(4, (3,), (), discrete_actions=True)
    for index in range(4):
        replay.add(
            np.full(3, index, dtype=np.float32),
            index % 2,
            float(index),
            np.full(3, index + 1, dtype=np.float32),
            index == 3,
        )
    batch = replay.sample(3, torch.device("cpu"), np.random.default_rng(0))
    assert batch.observations.shape == (3, 3)
    assert batch.actions.shape == (3,)
    assert batch.terminated.dtype == torch.float32


def test_squashed_gaussian_actions_respect_bounds_and_have_finite_log_prob() -> None:
    actor = SquashedGaussianActor(
        3,
        1,
        (8,),
        np.array([-2.0], dtype=np.float32),
        np.array([2.0], dtype=np.float32),
    )
    observations = torch.zeros((32, 3))
    actions, log_probabilities, raw_actions = actor.sample(observations)
    assert torch.all(actions <= 2.0)
    assert torch.all(actions >= -2.0)
    assert torch.isfinite(log_probabilities).all()
    torch.testing.assert_close(actor.log_prob(observations, raw_actions), log_probabilities)

