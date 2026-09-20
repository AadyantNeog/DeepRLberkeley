"""Reproducibility, environment creation, evaluation, and target updates."""

from __future__ import annotations

import random
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch
from torch import nn


def set_global_seeds(seed: int) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch and return a local NumPy generator."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def make_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def observation_size(env: gym.Env) -> int:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError("Neural algorithms expect a Box observation space")
    return int(np.prod(env.observation_space.shape))


def flatten_observation(observation: np.ndarray) -> np.ndarray:
    return np.asarray(observation, dtype=np.float32).reshape(-1)


@torch.no_grad()
def evaluate_policy(
    env_id: str,
    action_function: Callable[[np.ndarray], int | np.ndarray],
    *,
    episodes: int = 5,
    seed: int = 10_000,
) -> float:
    """Return mean undiscounted episodic return without exploration noise."""

    env = make_env(env_id, seed)
    episode_returns: list[float] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            while not done:
                action = action_function(flatten_observation(observation))
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
            episode_returns.append(total_reward)
    finally:
        env.close()
    return float(np.mean(episode_returns))


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak update: target <- (1-tau) target + tau source."""

    for target_parameter, source_parameter in zip(
        target.parameters(), source.parameters(), strict=True
    ):
        target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)

