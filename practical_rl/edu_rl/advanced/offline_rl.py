"""Offline datasets plus SAC+BC, AWAC, IQL, and CQL.

Training functions in this module accept arrays and never call ``env.step``.
Environment interaction is isolated in the dataset generator and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from edu_rl.core.buffers import ReplayBatch
from edu_rl.core.networks import ContinuousQNetwork, SquashedGaussianActor, ValueNetwork
from edu_rl.core.utils import hard_update, set_global_seeds, soft_update


@dataclass
class OfflineDataset:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray

    def __post_init__(self) -> None:
        length = len(self.observations)
        if not all(
            len(array) == length
            for array in (
                self.actions,
                self.rewards,
                self.next_observations,
                self.terminated,
            )
        ):
            raise ValueError("All offline dataset arrays must have equal length")

    def __len__(self) -> int:
        return len(self.observations)

    def sample(
        self, batch_size: int, device: torch.device, rng: np.random.Generator
    ) -> ReplayBatch:
        indices = rng.integers(len(self), size=batch_size)
        return ReplayBatch(
            *[
                torch.as_tensor(array[indices], dtype=torch.float32, device=device)
                for array in (
                    self.observations,
                    self.actions,
                    self.rewards,
                    self.next_observations,
                    self.terminated,
                )
            ]
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            next_observations=self.next_observations,
            terminated=self.terminated,
        )

    @classmethod
    def load(cls, path: str | Path) -> "OfflineDataset":
        with np.load(path) as data:
            return cls(**{name: data[name] for name in data.files})


def generate_pendulum_dataset(
    *,
    transitions: int = 50_000,
    quality: Literal["random", "mixed", "heuristic"] = "mixed",
    seed: int = 0,
) -> OfflineDataset:
    """Create a reproducible dataset without an external benchmark package."""

    env = gym.make("Pendulum-v1")
    env.action_space.seed(seed)
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    terminals: list[float] = []
    observation, _ = env.reset(seed=seed)
    for _ in range(transitions):
        use_random = quality == "random" or (quality == "mixed" and rng.random() < 0.5)
        if use_random:
            action = env.action_space.sample()
        else:
            cosine, sine, angular_velocity = observation
            angle = np.arctan2(sine, cosine)
            torque = np.clip(-2.0 * angle - 0.5 * angular_velocity, -2.0, 2.0)
            action = np.array([torque], dtype=np.float32)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        observations.append(np.asarray(observation, dtype=np.float32))
        actions.append(np.asarray(action, dtype=np.float32))
        rewards.append(float(reward))
        next_observations.append(np.asarray(next_observation, dtype=np.float32))
        terminals.append(float(terminated))
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset()
    env.close()
    return OfflineDataset(
        np.asarray(observations),
        np.asarray(actions),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(next_observations),
        np.asarray(terminals, dtype=np.float32),
    )


def expectile_loss(residual: torch.Tensor, expectile: float) -> torch.Tensor:
    """Asymmetric squared loss used for IQL's upper-expectile value fit."""

    weight = torch.where(residual > 0, expectile, 1.0 - expectile)
    return (weight * residual.pow(2)).mean()


def advantage_weights(
    advantages: torch.Tensor, inverse_temperature: float, maximum: float
) -> torch.Tensor:
    return torch.exp(inverse_temperature * advantages).clamp(max=maximum)


def conservative_q_penalty(
    q_network: ContinuousQNetwork,
    actor: SquashedGaussianActor,
    observations: torch.Tensor,
    dataset_actions: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    *,
    samples: int = 10,
) -> torch.Tensor:
    """Sampled CQL log-sum-exp penalty minus values of dataset actions."""

    batch_size, observation_size = observations.shape
    repeated_observations = observations[:, None, :].expand(-1, samples, -1)
    flat_observations = repeated_observations.reshape(-1, observation_size)
    random_actions = action_low + torch.rand(
        (batch_size * samples, action_low.numel()), device=observations.device
    ) * (action_high - action_low)
    with torch.no_grad():
        policy_actions, _, _ = actor.sample(flat_observations)
    random_q = q_network(flat_observations, random_actions).view(batch_size, samples)
    policy_q = q_network(flat_observations, policy_actions).view(batch_size, samples)
    candidate_q = torch.cat([random_q, policy_q], dim=1)
    dataset_q = q_network(observations, dataset_actions)
    return (torch.logsumexp(candidate_q, dim=1) - dataset_q).mean()


OfflineAlgorithm = Literal["sac_bc", "awac", "iql", "cql"]


@dataclass
class OfflineConfig:
    algorithm: OfflineAlgorithm = "iql"
    gradient_steps: int = 50_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    expectile: float = 0.7
    advantage_inverse_temperature: float = 3.0
    maximum_weight: float = 100.0
    behavior_coefficient: float = 1.0
    conservative_coefficient: float = 1.0
    entropy_temperature: float = 0.2
    hidden_sizes: tuple[int, ...] = (256, 256)
    seed: int = 0
    device: str = "cpu"


@dataclass
class OfflineResult:
    actor: SquashedGaussianActor
    q1: ContinuousQNetwork
    q2: ContinuousQNetwork
    actor_losses: list[float]
    critic_losses: list[float]
    auxiliary_losses: list[float]


def train_offline(
    dataset: OfflineDataset,
    config: OfflineConfig,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> OfflineResult:
    """Train one offline method using only the supplied fixed dataset."""

    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    observation_size = dataset.observations.shape[1]
    action_size = dataset.actions.shape[1]
    actor = SquashedGaussianActor(
        observation_size,
        action_size,
        config.hidden_sizes,
        action_low,
        action_high,
    ).to(device)
    q1 = ContinuousQNetwork(observation_size, action_size, config.hidden_sizes).to(device)
    q2 = ContinuousQNetwork(observation_size, action_size, config.hidden_sizes).to(device)
    target_q1 = ContinuousQNetwork(
        observation_size, action_size, config.hidden_sizes
    ).to(device)
    target_q2 = ContinuousQNetwork(
        observation_size, action_size, config.hidden_sizes
    ).to(device)
    hard_update(target_q1, q1)
    hard_update(target_q2, q2)
    value = ValueNetwork(observation_size, config.hidden_sizes).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    q_optimizer = torch.optim.Adam(
        [*q1.parameters(), *q2.parameters()], lr=config.learning_rate
    )
    value_optimizer = torch.optim.Adam(value.parameters(), lr=config.learning_rate)
    low_tensor = torch.as_tensor(action_low, dtype=torch.float32, device=device)
    high_tensor = torch.as_tensor(action_high, dtype=torch.float32, device=device)
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    auxiliary_losses: list[float] = []

    for _ in range(config.gradient_steps):
        batch = dataset.sample(config.batch_size, device, rng)
        if config.algorithm == "iql":
            with torch.no_grad():
                target = batch.rewards + config.gamma * (
                    1.0 - batch.terminated
                ) * value(batch.next_observations)
            predicted_q1 = q1(batch.observations, batch.actions)
            predicted_q2 = q2(batch.observations, batch.actions)
            q_loss = nn.functional.mse_loss(
                predicted_q1, target
            ) + nn.functional.mse_loss(predicted_q2, target)
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            with torch.no_grad():
                dataset_q = torch.minimum(
                    target_q1(batch.observations, batch.actions),
                    target_q2(batch.observations, batch.actions),
                )
            residual = dataset_q - value(batch.observations)
            auxiliary_loss = expectile_loss(residual, config.expectile)
            value_optimizer.zero_grad()
            auxiliary_loss.backward()
            value_optimizer.step()
            with torch.no_grad():
                weights = advantage_weights(
                    dataset_q - value(batch.observations),
                    config.advantage_inverse_temperature,
                    config.maximum_weight,
                )
            actor_loss = -(
                weights * actor.log_prob_of_action(batch.observations, batch.actions)
            ).mean()
        else:
            with torch.no_grad():
                next_actions, next_log_probabilities, _ = actor.sample(
                    batch.next_observations
                )
                next_q = torch.minimum(
                    target_q1(batch.next_observations, next_actions),
                    target_q2(batch.next_observations, next_actions),
                )
                if config.algorithm == "cql":
                    next_q = next_q - config.entropy_temperature * next_log_probabilities
                target = batch.rewards + config.gamma * (
                    1.0 - batch.terminated
                ) * next_q
            predicted_q1 = q1(batch.observations, batch.actions)
            predicted_q2 = q2(batch.observations, batch.actions)
            q_loss = nn.functional.mse_loss(
                predicted_q1, target
            ) + nn.functional.mse_loss(predicted_q2, target)
            auxiliary_loss = torch.zeros((), device=device)
            if config.algorithm == "cql":
                penalty1 = conservative_q_penalty(
                    q1,
                    actor,
                    batch.observations,
                    batch.actions,
                    low_tensor,
                    high_tensor,
                )
                penalty2 = conservative_q_penalty(
                    q2,
                    actor,
                    batch.observations,
                    batch.actions,
                    low_tensor,
                    high_tensor,
                )
                auxiliary_loss = penalty1 + penalty2
                q_loss = q_loss + config.conservative_coefficient * auxiliary_loss
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            if config.algorithm == "awac":
                with torch.no_grad():
                    policy_actions, _, _ = actor.sample(batch.observations)
                    dataset_q = torch.minimum(
                        q1(batch.observations, batch.actions),
                        q2(batch.observations, batch.actions),
                    )
                    policy_q = torch.minimum(
                        q1(batch.observations, policy_actions),
                        q2(batch.observations, policy_actions),
                    )
                    weights = advantage_weights(
                        dataset_q - policy_q,
                        config.advantage_inverse_temperature,
                        config.maximum_weight,
                    )
                actor_loss = -(
                    weights * actor.log_prob_of_action(batch.observations, batch.actions)
                ).mean()
            else:
                policy_actions, log_probability, _ = actor.sample(batch.observations)
                policy_q = torch.minimum(
                    q1(batch.observations, policy_actions),
                    q2(batch.observations, policy_actions),
                )
                actor_loss = (
                    config.entropy_temperature * log_probability - policy_q
                ).mean()
                if config.algorithm == "sac_bc":
                    actor_loss -= config.behavior_coefficient * actor.log_prob_of_action(
                        batch.observations, batch.actions
                    ).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        soft_update(target_q1, q1, config.tau)
        soft_update(target_q2, q2, config.tau)
        actor_losses.append(float(actor_loss.detach()))
        critic_losses.append(float(q_loss.detach()))
        auxiliary_losses.append(float(auxiliary_loss.detach()))

    return OfflineResult(
        actor, q1, q2, actor_losses, critic_losses, auxiliary_losses
    )
