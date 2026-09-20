"""Soft Actor-Critic with twin critics and automatic temperature tuning."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from edu_rl.core.buffers import ReplayBatch, ReplayBuffer
from edu_rl.core.networks import ContinuousQNetwork, SquashedGaussianActor
from edu_rl.core.utils import (
    flatten_observation,
    hard_update,
    make_env,
    observation_size,
    set_global_seeds,
    soft_update,
)


@dataclass
class SACConfig:
    env_id: str = "Pendulum-v1"
    total_steps: int = 100_000
    replay_capacity: int = 200_000
    batch_size: int = 128
    warmup_steps: int = 1_000
    gamma: float = 0.99
    tau: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temperature_learning_rate: float = 3e-4
    initial_temperature: float = 0.2
    hidden_sizes: tuple[int, ...] = (256, 256)
    seed: int = 0
    device: str = "cpu"


@dataclass
class SACResult:
    agent: "SACAgent"
    episode_returns: list[float]
    actor_losses: list[float]
    critic_losses: list[float]
    temperatures: list[float]


def soft_bellman_target(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    next_q: torch.Tensor,
    next_log_probability: torch.Tensor,
    *,
    gamma: float,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Entropy-regularized Bellman target used by SAC."""

    soft_next_value = next_q - temperature * next_log_probability
    return rewards + gamma * (1.0 - terminated) * soft_next_value


class SACAgent:
    """Reusable SAC learner shared with MBPO and DIAYN."""

    def __init__(
        self,
        observation_size: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        *,
        hidden_sizes: tuple[int, ...] = (256, 256),
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        temperature_learning_rate: float = 3e-4,
        initial_temperature: float = 0.2,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.action_size = self.action_low.size
        self.actor = SquashedGaussianActor(
            observation_size,
            self.action_size,
            hidden_sizes,
            self.action_low,
            self.action_high,
        ).to(device)
        self.q1 = ContinuousQNetwork(
            observation_size, self.action_size, hidden_sizes
        ).to(device)
        self.q2 = ContinuousQNetwork(
            observation_size, self.action_size, hidden_sizes
        ).to(device)
        self.target_q1 = ContinuousQNetwork(
            observation_size, self.action_size, hidden_sizes
        ).to(device)
        self.target_q2 = ContinuousQNetwork(
            observation_size, self.action_size, hidden_sizes
        ).to(device)
        hard_update(self.target_q1, self.q1)
        hard_update(self.target_q2, self.q2)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.q_optimizer = torch.optim.Adam(
            [*self.q1.parameters(), *self.q2.parameters()], lr=critic_learning_rate
        )
        self.log_temperature = torch.tensor(
            np.log(initial_temperature),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=temperature_learning_rate
        )
        self.target_entropy = -float(self.action_size)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    @torch.no_grad()
    def act(self, observation: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        if deterministic:
            action = self.actor.deterministic(tensor)
        else:
            action, _, _ = self.actor.sample(tensor)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        """Perform one critic, actor, temperature, and target-network update."""

        with torch.no_grad():
            next_actions, next_log_probabilities, _ = self.actor.sample(
                batch.next_observations
            )
            next_q = torch.minimum(
                self.target_q1(batch.next_observations, next_actions),
                self.target_q2(batch.next_observations, next_actions),
            )
            target = soft_bellman_target(
                batch.rewards,
                batch.terminated,
                next_q,
                next_log_probabilities,
                gamma=self.gamma,
                temperature=self.temperature.detach(),
            )

        predicted_q1 = self.q1(batch.observations, batch.actions.float())
        predicted_q2 = self.q2(batch.observations, batch.actions.float())
        critic_loss = nn.functional.mse_loss(
            predicted_q1, target
        ) + nn.functional.mse_loss(predicted_q2, target)
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_([*self.q1.parameters(), *self.q2.parameters()], 10.0)
        self.q_optimizer.step()

        actions, log_probabilities, _ = self.actor.sample(batch.observations)
        for parameter in [*self.q1.parameters(), *self.q2.parameters()]:
            parameter.requires_grad_(False)
        q_for_actions = torch.minimum(
            self.q1(batch.observations, actions), self.q2(batch.observations, actions)
        )
        actor_loss = (self.temperature.detach() * log_probabilities - q_for_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for parameter in [*self.q1.parameters(), *self.q2.parameters()]:
            parameter.requires_grad_(True)

        temperature_loss = -(
            self.log_temperature
            * (log_probabilities.detach() + self.target_entropy)
        ).mean()
        self.temperature_optimizer.zero_grad()
        temperature_loss.backward()
        self.temperature_optimizer.step()

        soft_update(self.target_q1, self.q1, self.tau)
        soft_update(self.target_q2, self.q2, self.tau)
        return {
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "temperature": float(self.temperature.detach()),
            "temperature_loss": float(temperature_loss.detach()),
        }


def train_sac(config: SACConfig, env: gym.Env | None = None) -> SACResult:
    rng = set_global_seeds(config.seed)
    own_env = env is None
    env = make_env(config.env_id, config.seed) if env is None else env
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("SAC requires a continuous Box action space")
    device = torch.device(config.device)
    obs_size = observation_size(env)
    action_low = np.asarray(env.action_space.low).reshape(-1)
    action_high = np.asarray(env.action_space.high).reshape(-1)
    agent = SACAgent(
        obs_size,
        action_low,
        action_high,
        hidden_sizes=config.hidden_sizes,
        gamma=config.gamma,
        tau=config.tau,
        actor_learning_rate=config.actor_learning_rate,
        critic_learning_rate=config.critic_learning_rate,
        temperature_learning_rate=config.temperature_learning_rate,
        initial_temperature=config.initial_temperature,
        device=device,
    )
    replay = ReplayBuffer(
        config.replay_capacity,
        (obs_size,),
        (action_low.size,),
        discrete_actions=False,
    )
    episode_returns: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    temperatures: list[float] = []
    episode_return = 0.0
    observation, _ = env.reset(seed=config.seed)
    try:
        for step in range(1, config.total_steps + 1):
            observation_array = flatten_observation(observation)
            if step <= config.warmup_steps:
                action = np.asarray(env.action_space.sample(), dtype=np.float32).reshape(-1)
            else:
                action = agent.act(observation_array)
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
            episode_return += float(reward)
            observation = next_observation
            if step >= config.warmup_steps and len(replay) >= config.batch_size:
                metrics = agent.update(replay.sample(config.batch_size, device, rng))
                actor_losses.append(metrics["actor_loss"])
                critic_losses.append(metrics["critic_loss"])
                temperatures.append(metrics["temperature"])
            if terminated or truncated:
                episode_returns.append(episode_return)
                episode_return = 0.0
                observation, _ = env.reset()
    finally:
        if own_env:
            env.close()
    return SACResult(agent, episode_returns, actor_losses, critic_losses, temperatures)

