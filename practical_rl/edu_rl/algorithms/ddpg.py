"""Deep Deterministic Policy Gradient for continuous control."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from edu_rl.core.buffers import ReplayBuffer
from edu_rl.core.networks import ContinuousQNetwork, DeterministicActor
from edu_rl.core.utils import (
    flatten_observation,
    hard_update,
    make_env,
    observation_size,
    set_global_seeds,
    soft_update,
)


@dataclass
class DDPGConfig:
    env_id: str = "Pendulum-v1"
    total_steps: int = 100_000
    replay_capacity: int = 200_000
    batch_size: int = 128
    warmup_steps: int = 2_000
    gamma: float = 0.99
    tau: float = 0.005
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    exploration_noise: float = 0.1
    hidden_sizes: tuple[int, ...] = (256, 256)
    seed: int = 0
    device: str = "cpu"


@dataclass
class DDPGResult:
    actor: DeterministicActor
    critic: ContinuousQNetwork
    episode_returns: list[float]
    actor_losses: list[float]
    critic_losses: list[float]


def ddpg_target(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    next_q_values: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    return rewards + gamma * (1.0 - terminated) * next_q_values


def train_ddpg(config: DDPGConfig) -> DDPGResult:
    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    env = make_env(config.env_id, config.seed)
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("DDPG requires a continuous Box action space")

    obs_size = observation_size(env)
    action_size = int(np.prod(env.action_space.shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    action_scale = (action_high - action_low) / 2.0

    actor = DeterministicActor(
        obs_size, action_size, config.hidden_sizes, action_low, action_high
    ).to(device)
    target_actor = DeterministicActor(
        obs_size, action_size, config.hidden_sizes, action_low, action_high
    ).to(device)
    critic = ContinuousQNetwork(
        obs_size, action_size, config.hidden_sizes
    ).to(device)
    target_critic = ContinuousQNetwork(
        obs_size, action_size, config.hidden_sizes
    ).to(device)
    hard_update(target_actor, actor)
    hard_update(target_critic, critic)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )
    replay = ReplayBuffer(
        config.replay_capacity,
        (obs_size,),
        (action_size,),
        discrete_actions=False,
    )

    episode_returns: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    episode_return = 0.0
    observation, _ = env.reset(seed=config.seed)

    try:
        for step in range(1, config.total_steps + 1):
            observation_array = flatten_observation(observation)
            if step <= config.warmup_steps:
                action = np.asarray(env.action_space.sample(), dtype=np.float32).reshape(-1)
            else:
                with torch.no_grad():
                    action = (
                        actor(
                            torch.as_tensor(observation_array, device=device).unsqueeze(0)
                        )
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
                noise = rng.normal(
                    0.0, config.exploration_noise, size=action_size
                ) * action_scale
                action = np.clip(action + noise, action_low, action_high).astype(
                    np.float32
                )

            next_observation, reward, terminated, truncated, _ = env.step(
                action.reshape(env.action_space.shape)
            )
            replay.add(
                observation_array,
                action,
                float(reward),
                flatten_observation(next_observation),
                terminated,
            )
            observation = next_observation
            episode_return += float(reward)

            if step >= config.warmup_steps and len(replay) >= config.batch_size:
                batch = replay.sample(config.batch_size, device, rng)

                # Critic: regress toward a slowly moving target supplied entirely
                # by frozen target networks.
                with torch.no_grad():
                    next_actions = target_actor(batch.next_observations)
                    next_q = target_critic(batch.next_observations, next_actions)
                    target = ddpg_target(
                        batch.rewards, batch.terminated, next_q, config.gamma
                    )
                predicted_q = critic(batch.observations, batch.actions.float())
                critic_loss = nn.functional.mse_loss(predicted_q, target)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
                critic_optimizer.step()

                # Actor: follow the action derivative of the learned critic.
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)
                actor_loss = -critic(
                    batch.observations, actor(batch.observations)
                ).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)

                soft_update(target_actor, actor, config.tau)
                soft_update(target_critic, critic, config.tau)
                actor_losses.append(float(actor_loss.detach()))
                critic_losses.append(float(critic_loss.detach()))

            if terminated or truncated:
                episode_returns.append(episode_return)
                episode_return = 0.0
                observation, _ = env.reset()
    finally:
        env.close()

    return DDPGResult(
        actor=actor,
        critic=critic,
        episode_returns=episode_returns,
        actor_losses=actor_losses,
        critic_losses=critic_losses,
    )

