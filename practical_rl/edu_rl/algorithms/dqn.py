"""Deep Q-learning, target networks, replay, and Double DQN."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from edu_rl.core.buffers import ReplayBuffer
from edu_rl.core.networks import DiscreteQNetwork
from edu_rl.core.utils import (
    flatten_observation,
    hard_update,
    make_env,
    observation_size,
    set_global_seeds,
)


@dataclass
class DQNConfig:
    env_id: str = "CartPole-v1"
    total_steps: int = 100_000
    replay_capacity: int = 100_000
    batch_size: int = 128
    warmup_steps: int = 1_000
    train_frequency: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 500
    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 40_000
    double_dqn: bool = True
    hidden_sizes: tuple[int, ...] = (128, 128)
    seed: int = 0
    device: str = "cpu"


@dataclass
class DQNResult:
    q_network: DiscreteQNetwork
    episode_returns: list[float]
    losses: list[float]
    episode_steps: list[int]


def linear_epsilon(config: DQNConfig, step: int) -> float:
    fraction = min(step / max(config.epsilon_decay_steps, 1), 1.0)
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def dqn_target(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    next_online_q: torch.Tensor,
    next_target_q: torch.Tensor,
    *,
    gamma: float,
    double_dqn: bool,
) -> torch.Tensor:
    """Build a detached one-step DQN or Double-DQN regression target.

    Double DQN selects an action with the online network and evaluates that
    action with the target network.  Standard DQN performs both operations with
    the target network and is therefore more exposed to maximization bias.
    """

    if double_dqn:
        next_actions = next_online_q.argmax(dim=1, keepdim=True)
        next_values = next_target_q.gather(1, next_actions).squeeze(1)
    else:
        next_values = next_target_q.max(dim=1).values
    return rewards + gamma * (1.0 - terminated) * next_values


def train_dqn(config: DQNConfig) -> DQNResult:
    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    env = make_env(config.env_id, config.seed)
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError("DQN requires a discrete action space")

    obs_size = observation_size(env)
    action_count = env.action_space.n
    online_q = DiscreteQNetwork(obs_size, action_count, config.hidden_sizes).to(device)
    target_q = DiscreteQNetwork(obs_size, action_count, config.hidden_sizes).to(device)
    hard_update(target_q, online_q)
    target_q.eval()
    optimizer = torch.optim.Adam(online_q.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(
        config.replay_capacity,
        (obs_size,),
        (),
        discrete_actions=True,
    )

    episode_returns: list[float] = []
    episode_steps: list[int] = []
    losses: list[float] = []
    episode_return = 0.0
    observation, _ = env.reset(seed=config.seed)

    try:
        for step in range(1, config.total_steps + 1):
            observation_array = flatten_observation(observation)
            epsilon = linear_epsilon(config, step)
            if step <= config.warmup_steps or rng.random() < epsilon:
                action = int(env.action_space.sample())
            else:
                with torch.no_grad():
                    q_values = online_q(
                        torch.as_tensor(observation_array, device=device).unsqueeze(0)
                    )
                action = int(q_values.argmax(dim=1).item())

            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_observation_array = flatten_observation(next_observation)
            replay.add(
                observation_array,
                action,
                float(reward),
                next_observation_array,
                terminated,
            )
            episode_return += float(reward)
            observation = next_observation

            if (
                step >= config.warmup_steps
                and len(replay) >= config.batch_size
                and step % config.train_frequency == 0
            ):
                for _ in range(config.gradient_steps):
                    batch = replay.sample(config.batch_size, device, rng)
                    predicted_q = online_q(batch.observations).gather(
                        1, batch.actions.long().unsqueeze(1)
                    ).squeeze(1)
                    with torch.no_grad():
                        next_online_q = online_q(batch.next_observations)
                        next_target_q = target_q(batch.next_observations)
                        target = dqn_target(
                            batch.rewards,
                            batch.terminated,
                            next_online_q,
                            next_target_q,
                            gamma=config.gamma,
                            double_dqn=config.double_dqn,
                        )
                    loss = nn.functional.smooth_l1_loss(predicted_q, target)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(online_q.parameters(), 10.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))

            if step % config.target_update_interval == 0:
                hard_update(target_q, online_q)

            if terminated or truncated:
                episode_returns.append(episode_return)
                episode_steps.append(step)
                episode_return = 0.0
                observation, _ = env.reset()
    finally:
        env.close()

    return DQNResult(
        q_network=online_q,
        episode_returns=episode_returns,
        losses=losses,
        episode_steps=episode_steps,
    )

