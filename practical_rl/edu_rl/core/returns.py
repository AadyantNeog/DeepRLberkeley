"""Return and advantage estimators shared by policy-gradient algorithms.

These functions are intentionally framework-light.  Keeping the recurrences in
one place lets the tests compare them with hand-computed examples.
"""

from __future__ import annotations

import numpy as np


def discounted_returns(rewards: np.ndarray | list[float], gamma: float) -> np.ndarray:
    """Compute reward-to-go ``G_t = r_t + gamma * G_{t+1}``.

    This function expects one complete reward sequence.  If a rollout ends only
    because of a time limit, add the desired bootstrap value to the final reward
    before calling it, or use :func:`generalized_advantage_estimate`.
    """

    rewards_array = np.asarray(rewards, dtype=np.float32)
    returns = np.empty_like(rewards_array)
    running_return = 0.0
    for index in range(len(rewards_array) - 1, -1, -1):
        running_return = float(rewards_array[index]) + gamma * running_return
        returns[index] = running_return
    return returns


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    episode_ends: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and critic targets for a rollout.

    ``terminated`` and ``episode_ends`` deliberately mean different things:

    * A true terminal state has no continuation value, so it masks the value in
      the one-step TD residual.
    * A time-limit truncation still has a continuation value, but the GAE trace
      must stop because the environment will reset before the next stored item.

    Treating every time limit as terminal is a common, quiet source of bias.
    """

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=np.float32)
    episode_ends = np.asarray(episode_ends, dtype=np.float32)

    if not (
        rewards.shape
        == values.shape
        == next_values.shape
        == terminated.shape
        == episode_ends.shape
    ):
        raise ValueError("All GAE inputs must have the same one-dimensional shape")

    deltas = rewards + gamma * (1.0 - terminated) * next_values - values
    advantages = np.zeros_like(deltas)
    running_advantage = 0.0
    for index in range(len(deltas) - 1, -1, -1):
        continue_trace = 1.0 - episode_ends[index]
        running_advantage = (
            float(deltas[index])
            + gamma * gae_lambda * continue_trace * running_advantage
        )
        advantages[index] = running_advantage

    value_targets = advantages + values
    return advantages, value_targets


def normalize(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Standardize a batch without amplifying an almost-constant signal."""

    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean()) / (values.std() + epsilon)

