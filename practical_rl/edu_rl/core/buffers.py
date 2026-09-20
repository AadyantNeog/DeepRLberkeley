"""Replay memory used by DQN and DDPG."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor


class ReplayBuffer:
    """A fixed-size circular replay buffer backed by NumPy arrays.

    Only true termination is stored.  Time-limit truncation causes an episode
    reset, but does not erase the valid bootstrap value of its final state.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        *,
        discrete_actions: bool,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.observations = np.empty(
            (capacity, *observation_shape), dtype=np.float32
        )
        self.next_observations = np.empty_like(self.observations)
        action_dtype = np.int64 if discrete_actions else np.float32
        self.actions = np.empty((capacity, *action_shape), dtype=action_dtype)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: int | np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.terminated[self.position] = float(terminated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        rng: np.random.Generator,
    ) -> ReplayBatch:
        if batch_size > self.size:
            raise ValueError("Cannot sample more transitions than the buffer contains")
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return ReplayBatch(
            observations=torch.as_tensor(
                self.observations[indices], device=device
            ),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_observations=torch.as_tensor(
                self.next_observations[indices], device=device
            ),
            terminated=torch.as_tensor(self.terminated[indices], device=device),
        )

    def __len__(self) -> int:
        return self.size

