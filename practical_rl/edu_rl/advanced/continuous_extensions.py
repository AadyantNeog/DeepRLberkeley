"""TD3 and neural successor features for continuous state/action spaces."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from edu_rl.core.buffers import ReplayBuffer
from edu_rl.core.networks import ContinuousQNetwork, DeterministicActor, mlp
from edu_rl.core.utils import (
    flatten_observation,
    hard_update,
    make_env,
    observation_size,
    set_global_seeds,
    soft_update,
)
from edu_rl.envs import PointMass2D


def td3_target(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    next_q1: torch.Tensor,
    next_q2: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """TD3 backs up the smaller target critic to reduce overestimation."""

    return rewards + gamma * (1.0 - terminated) * torch.minimum(next_q1, next_q2)


@dataclass
class TD3Config:
    env_id: str = "Pendulum-v1"
    total_steps: int = 100_000
    replay_capacity: int = 200_000
    batch_size: int = 128
    warmup_steps: int = 2_000
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    exploration_noise: float = 0.1
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    policy_delay: int = 2
    hidden_sizes: tuple[int, ...] = (256, 256)
    seed: int = 0
    device: str = "cpu"


@dataclass
class TD3Result:
    actor: DeterministicActor
    q1: ContinuousQNetwork
    q2: ContinuousQNetwork
    episode_returns: list[float]
    actor_losses: list[float]
    critic_losses: list[float]


def train_td3(config: TD3Config) -> TD3Result:
    """Twin Delayed DDPG with all three stabilizing mechanisms visible."""

    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    env = make_env(config.env_id, config.seed)
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("TD3 requires a continuous Box action space")
    obs_size = observation_size(env)
    action_size = int(np.prod(env.action_space.shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    action_scale = (action_high - action_low) / 2.0
    low_tensor = torch.as_tensor(action_low, device=device)
    high_tensor = torch.as_tensor(action_high, device=device)
    actor = DeterministicActor(obs_size, action_size, config.hidden_sizes, action_low, action_high).to(device)
    target_actor = DeterministicActor(obs_size, action_size, config.hidden_sizes, action_low, action_high).to(device)
    q1 = ContinuousQNetwork(obs_size, action_size, config.hidden_sizes).to(device)
    q2 = ContinuousQNetwork(obs_size, action_size, config.hidden_sizes).to(device)
    target_q1 = ContinuousQNetwork(obs_size, action_size, config.hidden_sizes).to(device)
    target_q2 = ContinuousQNetwork(obs_size, action_size, config.hidden_sizes).to(device)
    hard_update(target_actor, actor)
    hard_update(target_q1, q1)
    hard_update(target_q2, q2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.Adam([*q1.parameters(), *q2.parameters()], lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_capacity, (obs_size,), (action_size,), discrete_actions=False)
    returns: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    observation, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    updates = 0
    try:
        for step in range(1, config.total_steps + 1):
            flat_observation = flatten_observation(observation)
            if step <= config.warmup_steps:
                action = np.asarray(env.action_space.sample(), dtype=np.float32).reshape(-1)
            else:
                with torch.no_grad():
                    action = actor(torch.as_tensor(flat_observation, device=device)[None]).squeeze(0).cpu().numpy()
                action += rng.normal(0, config.exploration_noise, action_size) * action_scale
                action = np.clip(action, action_low, action_high).astype(np.float32)
            next_observation, reward, terminated, truncated, _ = env.step(
                action.reshape(env.action_space.shape)
            )
            replay.add(flat_observation, action, reward, flatten_observation(next_observation), terminated)
            observation = next_observation
            episode_return += reward
            if step >= config.warmup_steps and len(replay) >= config.batch_size:
                updates += 1
                batch = replay.sample(config.batch_size, device, rng)
                with torch.no_grad():
                    # Target-policy smoothing makes the target less sensitive to
                    # narrow, erroneous Q peaks.
                    noise = torch.randn_like(batch.actions) * config.target_policy_noise * torch.as_tensor(action_scale, device=device)
                    limit = config.target_noise_clip * torch.as_tensor(action_scale, device=device)
                    noise = torch.maximum(torch.minimum(noise, limit), -limit)
                    next_actions = (target_actor(batch.next_observations) + noise).clamp(low_tensor, high_tensor)
                    target = td3_target(
                        batch.rewards, batch.terminated,
                        target_q1(batch.next_observations, next_actions),
                        target_q2(batch.next_observations, next_actions),
                        gamma=config.gamma,
                    )
                q_loss = nn.functional.mse_loss(q1(batch.observations, batch.actions), target)
                q_loss += nn.functional.mse_loss(q2(batch.observations, batch.actions), target)
                critic_optimizer.zero_grad()
                q_loss.backward()
                critic_optimizer.step()
                critic_losses.append(float(q_loss.detach()))
                # Delayed updates let the critics settle before moving the actor
                # and all target networks.
                if updates % config.policy_delay == 0:
                    for parameter in [*q1.parameters(), *q2.parameters()]:
                        parameter.requires_grad_(False)
                    actor_loss = -q1(batch.observations, actor(batch.observations)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    for parameter in [*q1.parameters(), *q2.parameters()]:
                        parameter.requires_grad_(True)
                    soft_update(target_actor, actor, config.tau)
                    soft_update(target_q1, q1, config.tau)
                    soft_update(target_q2, q2, config.tau)
                    actor_losses.append(float(actor_loss.detach()))
            if terminated or truncated:
                returns.append(float(episode_return))
                episode_return = 0.0
                observation, _ = env.reset()
    finally:
        env.close()
    return TD3Result(actor, q1, q2, returns, actor_losses, critic_losses)


class NeuralSuccessorFeatures(nn.Module):
    """Predict discounted future RBF feature occupancy ``psi(s,a)``."""

    def __init__(
        self, observation_size: int, action_size: int, feature_size: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.network = mlp(observation_size + action_size, hidden_sizes, feature_size)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([observations, actions], dim=-1))


def rbf_state_features(
    states: torch.Tensor, centers: torch.Tensor, *, bandwidth: float = 0.4
) -> torch.Tensor:
    """Smooth position features suitable for continuous successor learning."""

    squared_distance = (states[:, None, :2] - centers[None]).pow(2).sum(dim=-1)
    features = torch.exp(-squared_distance / (2.0 * bandwidth**2))
    return features / (features.sum(dim=-1, keepdim=True) + 1e-8)


@dataclass
class ContinuousSFConfig:
    total_steps: int = 50_000
    warmup_steps: int = 500
    batch_size: int = 128
    replay_capacity: int = 100_000
    gamma: float = 0.95
    tau: float = 0.01
    hidden_sizes: tuple[int, ...] = (128, 128)
    seed: int = 0
    device: str = "cpu"


@dataclass
class ContinuousSFResult:
    successor: NeuralSuccessorFeatures
    reward_weights: torch.Tensor
    losses: list[float]
    reward_prediction_losses: list[float]


def train_continuous_successor_features(config: ContinuousSFConfig) -> ContinuousSFResult:
    """Learn neural successor features under a fixed goal-seeking policy."""

    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    env = PointMass2D(max_steps=80, terminate_on_goal=False)
    coordinates = torch.linspace(-0.8, 0.8, 3, device=device)
    centers = torch.cartesian_prod(coordinates, coordinates)
    feature_size = len(centers)
    successor = NeuralSuccessorFeatures(4, 2, feature_size, config.hidden_sizes).to(device)
    target_successor = NeuralSuccessorFeatures(4, 2, feature_size, config.hidden_sizes).to(device)
    hard_update(target_successor, successor)
    reward_weights = nn.Parameter(torch.zeros(feature_size, device=device))
    optimizer = torch.optim.Adam([*successor.parameters(), reward_weights], lr=3e-4)
    replay = ReplayBuffer(config.replay_capacity, (4,), (2,), discrete_actions=False)
    observation, _ = env.reset(seed=config.seed)
    losses: list[float] = []
    reward_losses: list[float] = []
    for step in range(1, config.total_steps + 1):
        # The fixed policy defines which future occupancy psi predicts.
        policy_action = np.clip(8.0 * (env.goal - observation[:2]) - 2.0 * observation[2:], -1, 1)
        action = np.clip(policy_action + rng.normal(0, 0.15, size=2), -1, 1).astype(np.float32)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        replay.add(observation, action, reward, next_observation, terminated)
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset()
        if step >= config.warmup_steps and len(replay) >= config.batch_size:
            batch = replay.sample(config.batch_size, device, rng)
            next_policy_actions = (
                8.0 * (torch.as_tensor(env.goal, device=device) - batch.next_observations[:, :2])
                - 2.0 * batch.next_observations[:, 2:]
            ).clamp(-1, 1)
            features = rbf_state_features(batch.next_observations, centers)
            with torch.no_grad():
                target = features + config.gamma * (1.0 - batch.terminated[:, None]) * target_successor(
                    batch.next_observations, next_policy_actions
                )
            prediction = successor(batch.observations, batch.actions)
            sf_loss = nn.functional.mse_loss(prediction, target)
            reward_prediction = features @ reward_weights
            reward_loss = nn.functional.mse_loss(reward_prediction, batch.rewards)
            loss = sf_loss + reward_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            soft_update(target_successor, successor, config.tau)
            losses.append(float(sf_loss.detach()))
            reward_losses.append(float(reward_loss.detach()))
    env.close()
    return ContinuousSFResult(successor, reward_weights.detach(), losses, reward_losses)
