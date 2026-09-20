"""REINFORCE with full returns, reward-to-go, and an optional value baseline."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from edu_rl.core.networks import CategoricalActor, ValueNetwork
from edu_rl.core.returns import discounted_returns, normalize
from edu_rl.core.utils import flatten_observation, make_env, observation_size, set_global_seeds


@dataclass
class ReinforceConfig:
    env_id: str = "CartPole-v1"
    iterations: int = 100
    episodes_per_batch: int = 10
    gamma: float = 0.99
    learning_rate: float = 3e-3
    baseline_learning_rate: float = 1e-3
    reward_to_go: bool = True
    use_baseline: bool = True
    normalize_advantages: bool = True
    entropy_coefficient: float = 0.0
    hidden_sizes: tuple[int, ...] = (64, 64)
    seed: int = 0
    device: str = "cpu"


@dataclass
class ReinforceResult:
    actor: CategoricalActor
    critic: ValueNetwork | None
    mean_returns: list[float]
    policy_losses: list[float]
    value_losses: list[float]


def train_reinforce(config: ReinforceConfig) -> ReinforceResult:
    """Train a categorical policy using Monte Carlo policy gradients."""

    set_global_seeds(config.seed)
    device = torch.device(config.device)
    env = make_env(config.env_id, config.seed)
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError("This REINFORCE implementation expects discrete actions")

    obs_size = observation_size(env)
    actor = CategoricalActor(
        obs_size, env.action_space.n, config.hidden_sizes
    ).to(device)
    critic = (
        ValueNetwork(obs_size, config.hidden_sizes).to(device)
        if config.use_baseline
        else None
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = (
        torch.optim.Adam(critic.parameters(), lr=config.baseline_learning_rate)
        if critic is not None
        else None
    )

    mean_returns: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []

    try:
        for iteration in range(config.iterations):
            batch_observations: list[np.ndarray] = []
            batch_actions: list[int] = []
            batch_weights: list[float] = []
            episode_returns: list[float] = []

            for episode in range(config.episodes_per_batch):
                observation, _ = env.reset(
                    seed=config.seed + iteration * config.episodes_per_batch + episode
                )
                observations: list[np.ndarray] = []
                actions: list[int] = []
                rewards: list[float] = []
                done = False
                while not done:
                    observation_array = flatten_observation(observation)
                    observation_tensor = torch.as_tensor(
                        observation_array, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        action, _ = actor.sample(observation_tensor)
                    next_observation, reward, terminated, truncated, _ = env.step(
                        int(action.item())
                    )
                    observations.append(observation_array)
                    actions.append(int(action.item()))
                    rewards.append(float(reward))
                    observation = next_observation
                    done = terminated or truncated

                reward_to_go = discounted_returns(rewards, config.gamma)
                weights = (
                    reward_to_go
                    if config.reward_to_go
                    else np.full(len(rewards), reward_to_go[0], dtype=np.float32)
                )
                batch_observations.extend(observations)
                batch_actions.extend(actions)
                batch_weights.extend(weights.tolist())
                episode_returns.append(float(sum(rewards)))

            observations_tensor = torch.as_tensor(
                np.asarray(batch_observations), device=device
            )
            actions_tensor = torch.as_tensor(batch_actions, device=device)
            weights_tensor = torch.as_tensor(
                batch_weights, dtype=torch.float32, device=device
            )

            if critic is not None:
                predicted_values = critic(observations_tensor)
                advantages = weights_tensor - predicted_values.detach()
            else:
                predicted_values = None
                advantages = weights_tensor
            if config.normalize_advantages:
                advantages = torch.as_tensor(
                    normalize(advantages.cpu().numpy()), device=device
                )

            distribution = actor.distribution(observations_tensor)
            log_probabilities = distribution.log_prob(actions_tensor)
            policy_loss = -(
                log_probabilities * advantages
            ).mean() - config.entropy_coefficient * distribution.entropy().mean()
            actor_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()

            if critic is not None and critic_optimizer is not None:
                value_loss = nn.functional.mse_loss(predicted_values, weights_tensor)
                critic_optimizer.zero_grad()
                value_loss.backward()
                critic_optimizer.step()
                value_losses.append(float(value_loss.detach()))

            mean_returns.append(float(np.mean(episode_returns)))
            policy_losses.append(float(policy_loss.detach()))
    finally:
        env.close()

    return ReinforceResult(
        actor=actor,
        critic=critic,
        mean_returns=mean_returns,
        policy_losses=policy_losses,
        value_losses=value_losses,
    )

