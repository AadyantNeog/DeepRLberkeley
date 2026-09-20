"""Shared rollout machinery for PPO and natural-policy-gradient methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import gymnasium as gym
import numpy as np
import torch

from edu_rl.core.networks import (
    CategoricalActor,
    SquashedGaussianActor,
    ValueNetwork,
)
from edu_rl.core.utils import flatten_observation


Actor: TypeAlias = CategoricalActor | SquashedGaussianActor


@dataclass
class Rollout:
    observations: np.ndarray
    policy_actions: np.ndarray
    old_log_probabilities: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    terminated: np.ndarray
    episode_ends: np.ndarray
    completed_episode_returns: list[float]
    continuous: bool


def make_actor(
    env: gym.Env,
    observation_size: int,
    hidden_sizes: tuple[int, ...],
) -> tuple[Actor, bool]:
    if isinstance(env.action_space, gym.spaces.Discrete):
        return CategoricalActor(observation_size, env.action_space.n, hidden_sizes), False
    if isinstance(env.action_space, gym.spaces.Box):
        action_size = int(np.prod(env.action_space.shape))
        return (
            SquashedGaussianActor(
                observation_size,
                action_size,
                hidden_sizes,
                np.asarray(env.action_space.low).reshape(-1),
                np.asarray(env.action_space.high).reshape(-1),
            ),
            True,
        )
    raise TypeError("Only Discrete and Box action spaces are supported")


@torch.no_grad()
def collect_rollout(
    env: gym.Env,
    actor: Actor,
    critic: ValueNetwork,
    *,
    rollout_steps: int,
    device: torch.device,
    seed: int,
    continuous: bool,
) -> Rollout:
    """Collect a fixed-size fresh batch and all data needed for policy ratios."""

    observations: list[np.ndarray] = []
    policy_actions: list[int | np.ndarray] = []
    log_probabilities: list[float] = []
    rewards: list[float] = []
    values: list[float] = []
    next_values: list[float] = []
    terminated_flags: list[float] = []
    episode_end_flags: list[float] = []
    completed_returns: list[float] = []

    observation, _ = env.reset(seed=seed)
    running_return = 0.0
    for _ in range(rollout_steps):
        observation_array = flatten_observation(observation)
        observation_tensor = torch.as_tensor(
            observation_array, device=device
        ).unsqueeze(0)
        value = float(critic(observation_tensor).item())
        if continuous:
            assert isinstance(actor, SquashedGaussianActor)
            action_tensor, log_probability, raw_action = actor.sample(
                observation_tensor
            )
            environment_action = (
                action_tensor.squeeze(0).cpu().numpy().reshape(env.action_space.shape)
            )
            stored_action: int | np.ndarray = raw_action.squeeze(0).cpu().numpy()
        else:
            assert isinstance(actor, CategoricalActor)
            action_tensor, log_probability = actor.sample(observation_tensor)
            environment_action = int(action_tensor.item())
            stored_action = environment_action

        next_observation, reward, terminated, truncated, _ = env.step(
            environment_action
        )
        with torch.no_grad():
            next_value = float(
                critic(
                    torch.as_tensor(
                        flatten_observation(next_observation), device=device
                    ).unsqueeze(0)
                ).item()
            )

        observations.append(observation_array)
        policy_actions.append(stored_action)
        log_probabilities.append(float(log_probability.item()))
        rewards.append(float(reward))
        values.append(value)
        next_values.append(next_value)
        terminated_flags.append(float(terminated))
        episode_end_flags.append(float(terminated or truncated))
        running_return += float(reward)
        observation = next_observation

        if terminated or truncated:
            completed_returns.append(running_return)
            running_return = 0.0
            observation, _ = env.reset()

    # The last stored transition is a sampling boundary even if its real episode
    # continues.  We bootstrap its next value, but never propagate a GAE trace
    # into a transition that was not actually collected.
    episode_end_flags[-1] = 1.0
    action_dtype = np.float32 if continuous else np.int64
    return Rollout(
        observations=np.asarray(observations, dtype=np.float32),
        policy_actions=np.asarray(policy_actions, dtype=action_dtype),
        old_log_probabilities=np.asarray(log_probabilities, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        next_values=np.asarray(next_values, dtype=np.float32),
        terminated=np.asarray(terminated_flags, dtype=np.float32),
        episode_ends=np.asarray(episode_end_flags, dtype=np.float32),
        completed_episode_returns=completed_returns,
        continuous=continuous,
    )


def policy_log_prob_and_entropy(
    actor: Actor,
    observations: torch.Tensor,
    policy_actions: torch.Tensor,
    *,
    continuous: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if continuous:
        assert isinstance(actor, SquashedGaussianActor)
        distribution = actor.base_distribution(observations)
        log_probability = actor.log_prob_from_raw(distribution, policy_actions)
        # Entropy is measured before the fixed tanh transform.  It remains a
        # useful exploration diagnostic even though the squashed entropy differs.
        entropy = distribution.entropy()
    else:
        assert isinstance(actor, CategoricalActor)
        distribution = actor.distribution(observations)
        log_probability = distribution.log_prob(policy_actions.long())
        entropy = distribution.entropy()
    return log_probability, entropy


def mean_policy_kl(
    old_actor: Actor,
    new_actor: Actor,
    observations: torch.Tensor,
    *,
    continuous: bool,
) -> torch.Tensor:
    """Exact mean KL for the categorical or pre-squash Gaussian policy."""

    if continuous:
        assert isinstance(old_actor, SquashedGaussianActor)
        assert isinstance(new_actor, SquashedGaussianActor)
        old_distribution = old_actor.base_distribution(observations)
        new_distribution = new_actor.base_distribution(observations)
    else:
        assert isinstance(old_actor, CategoricalActor)
        assert isinstance(new_actor, CategoricalActor)
        old_distribution = old_actor.distribution(observations)
        new_distribution = new_actor.distribution(observations)
    return torch.distributions.kl_divergence(
        old_distribution, new_distribution
    ).mean()

