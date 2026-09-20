"""On-policy actor-critic with TD(0), n-step, or GAE estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from edu_rl.core.networks import CategoricalActor, ValueNetwork
from edu_rl.core.returns import generalized_advantage_estimate, normalize
from edu_rl.core.utils import flatten_observation, make_env, observation_size, set_global_seeds


Estimator = Literal["td0", "nstep", "gae"]


@dataclass
class ActorCriticConfig:
    env_id: str = "CartPole-v1"
    iterations: int = 150
    episodes_per_batch: int = 8
    estimator: Estimator = "gae"
    n_steps: int = 5
    gae_lambda: float = 0.95
    gamma: float = 0.99
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    entropy_coefficient: float = 0.01
    hidden_sizes: tuple[int, ...] = (64, 64)
    seed: int = 0
    device: str = "cpu"


@dataclass
class ActorCriticResult:
    actor: CategoricalActor
    critic: ValueNetwork
    mean_returns: list[float]
    policy_losses: list[float]
    value_losses: list[float]


def n_step_estimates(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    *,
    gamma: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return n-step advantages and value targets for one stored episode."""

    length = len(rewards)
    targets = np.zeros(length, dtype=np.float32)
    for start in range(length):
        target = 0.0
        discount = 1.0
        last_transition = start
        hit_terminal = False
        for index in range(start, min(start + n_steps, length)):
            target += discount * float(rewards[index])
            last_transition = index
            discount *= gamma
            if terminated[index]:
                hit_terminal = True
                break
        if not hit_terminal:
            target += discount * float(next_values[last_transition])
        targets[start] = target
    return targets - values, targets


def episode_estimates(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    *,
    estimator: Estimator,
    gamma: float,
    n_steps: int,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the estimator while preserving a common actor/critic interface."""

    if estimator == "td0":
        targets = rewards + gamma * (1.0 - terminated) * next_values
        return targets - values, targets
    if estimator == "nstep":
        return n_step_estimates(
            rewards,
            values,
            next_values,
            terminated,
            gamma=gamma,
            n_steps=n_steps,
        )
    if estimator == "gae":
        episode_ends = np.zeros_like(terminated)
        episode_ends[-1] = 1.0
        return generalized_advantage_estimate(
            rewards,
            values,
            next_values,
            terminated,
            episode_ends,
            gamma,
            gae_lambda,
        )
    raise ValueError(f"Unknown estimator: {estimator}")


def train_actor_critic(config: ActorCriticConfig) -> ActorCriticResult:
    set_global_seeds(config.seed)
    device = torch.device(config.device)
    env = make_env(config.env_id, config.seed)
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError("This actor-critic implementation expects discrete actions")

    obs_size = observation_size(env)
    actor = CategoricalActor(
        obs_size, env.action_space.n, config.hidden_sizes
    ).to(device)
    critic = ValueNetwork(obs_size, config.hidden_sizes).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )
    mean_returns: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []

    try:
        for iteration in range(config.iterations):
            batch_observations: list[np.ndarray] = []
            batch_actions: list[int] = []
            batch_advantages: list[float] = []
            batch_targets: list[float] = []
            episode_returns: list[float] = []

            for episode in range(config.episodes_per_batch):
                observation, _ = env.reset(
                    seed=config.seed + iteration * config.episodes_per_batch + episode
                )
                observations: list[np.ndarray] = []
                next_observations: list[np.ndarray] = []
                actions: list[int] = []
                rewards: list[float] = []
                terminal_flags: list[float] = []
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
                    next_observations.append(flatten_observation(next_observation))
                    actions.append(int(action.item()))
                    rewards.append(float(reward))
                    terminal_flags.append(float(terminated))
                    observation = next_observation
                    done = terminated or truncated

                observations_tensor = torch.as_tensor(
                    np.asarray(observations), device=device
                )
                next_observations_tensor = torch.as_tensor(
                    np.asarray(next_observations), device=device
                )
                with torch.no_grad():
                    values = critic(observations_tensor).cpu().numpy()
                    next_values = critic(next_observations_tensor).cpu().numpy()
                advantages, targets = episode_estimates(
                    np.asarray(rewards, dtype=np.float32),
                    values,
                    next_values,
                    np.asarray(terminal_flags, dtype=np.float32),
                    estimator=config.estimator,
                    gamma=config.gamma,
                    n_steps=config.n_steps,
                    gae_lambda=config.gae_lambda,
                )
                batch_observations.extend(observations)
                batch_actions.extend(actions)
                batch_advantages.extend(advantages.tolist())
                batch_targets.extend(targets.tolist())
                episode_returns.append(float(sum(rewards)))

            observations_tensor = torch.as_tensor(
                np.asarray(batch_observations), device=device
            )
            actions_tensor = torch.as_tensor(batch_actions, device=device)
            advantages_tensor = torch.as_tensor(
                normalize(np.asarray(batch_advantages)), device=device
            )
            targets_tensor = torch.as_tensor(batch_targets, device=device)

            distribution = actor.distribution(observations_tensor)
            policy_loss = -(
                distribution.log_prob(actions_tensor) * advantages_tensor
            ).mean() - config.entropy_coefficient * distribution.entropy().mean()
            actor_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()

            value_loss = nn.functional.mse_loss(
                critic(observations_tensor), targets_tensor
            )
            critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()

            mean_returns.append(float(np.mean(episode_returns)))
            policy_losses.append(float(policy_loss.detach()))
            value_losses.append(float(value_loss.detach()))
    finally:
        env.close()

    return ActorCriticResult(
        actor=actor,
        critic=critic,
        mean_returns=mean_returns,
        policy_losses=policy_losses,
        value_losses=value_losses,
    )

