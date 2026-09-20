"""UCB, count bonuses, and Random Network Distillation exploration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from edu_rl.core.buffers import ReplayBuffer
from edu_rl.core.networks import DiscreteQNetwork, mlp
from edu_rl.core.utils import hard_update, set_global_seeds


@dataclass
class BanditResult:
    actions: list[int]
    rewards: list[float]
    cumulative_regret: list[float]
    counts: np.ndarray


def run_ucb(
    true_means: np.ndarray,
    *,
    steps: int = 2_000,
    confidence: float = 2.0,
    noise_standard_deviation: float = 1.0,
    seed: int = 0,
) -> BanditResult:
    """UCB1-style optimism for a Gaussian multi-armed bandit."""

    rng = np.random.default_rng(seed)
    true_means = np.asarray(true_means, dtype=np.float64)
    counts = np.zeros(len(true_means), dtype=np.int64)
    estimates = np.zeros(len(true_means), dtype=np.float64)
    actions: list[int] = []
    rewards: list[float] = []
    regrets: list[float] = []
    cumulative_regret = 0.0
    for time in range(1, steps + 1):
        if time <= len(true_means):
            action = time - 1
        else:
            bonus = confidence * np.sqrt(np.log(time) / counts)
            action = int(np.argmax(estimates + bonus))
        reward = float(rng.normal(true_means[action], noise_standard_deviation))
        counts[action] += 1
        estimates[action] += (reward - estimates[action]) / counts[action]
        cumulative_regret += float(true_means.max() - true_means[action])
        actions.append(action)
        rewards.append(reward)
        regrets.append(cumulative_regret)
    return BanditResult(actions, rewards, regrets, counts)


class SparseChain:
    """Only reaching the far-right state gives extrinsic reward."""

    def __init__(self, length: int = 12, max_steps: int = 40) -> None:
        self.length = length
        self.max_steps = max_steps
        self.state = 0
        self.steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[int, dict]:
        del seed
        self.state = 0
        self.steps = 0
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        self.state = int(np.clip(self.state + (-1 if action == 0 else 1), 0, self.length - 1))
        self.steps += 1
        terminated = self.state == self.length - 1
        truncated = self.steps >= self.max_steps and not terminated
        return self.state, float(terminated), terminated, truncated, {}


def count_bonus_q_learning(
    *,
    episodes: int = 300,
    bonus_scale: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Tabular Q-learning with the intrinsic bonus ``beta/sqrt(N(s))``."""

    env = SparseChain()
    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.length, 2))
    counts = np.zeros(env.length)
    extrinsic_returns: list[float] = []
    for episode in range(episodes):
        state, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action = int(rng.integers(2)) if rng.random() < 0.1 else int(q_values[state].argmax())
            next_state, reward, terminated, truncated, _ = env.step(action)
            counts[next_state] += 1
            intrinsic = bonus_scale / np.sqrt(counts[next_state])
            target = reward + intrinsic + (0.0 if terminated else 0.99 * q_values[next_state].max())
            q_values[state, action] += 0.2 * (target - q_values[state, action])
            state = next_state
            total += reward
            done = terminated or truncated
        extrinsic_returns.append(total)
    return q_values, extrinsic_returns


class RNDModule(nn.Module):
    """Fixed random target and learned predictor used as a novelty signal."""

    def __init__(self, observation_size: int, feature_size: int = 32) -> None:
        super().__init__()
        self.target = mlp(observation_size, (64,), feature_size)
        self.predictor = mlp(observation_size, (64,), feature_size)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def intrinsic_reward(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_features = self.target(observations)
        predicted_features = self.predictor(observations)
        return (predicted_features - target_features).pow(2).mean(dim=1)


@dataclass
class RNDConfig:
    total_steps: int = 10_000
    warmup_steps: int = 200
    batch_size: int = 64
    intrinsic_scale: float = 1.0
    chain_length: int = 12
    hidden_sizes: tuple[int, ...] = (64, 64)
    seed: int = 0
    device: str = "cpu"


@dataclass
class RNDResult:
    q_network: DiscreteQNetwork
    rnd: RNDModule
    extrinsic_returns: list[float]
    intrinsic_rewards: list[float]


def _one_hot(states: np.ndarray | list[int], size: int) -> np.ndarray:
    return np.eye(size, dtype=np.float32)[np.asarray(states, dtype=np.int64)]


def train_rnd(config: RNDConfig) -> RNDResult:
    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    env = SparseChain(length=config.chain_length)
    q_network = DiscreteQNetwork(config.chain_length, 2, config.hidden_sizes).to(device)
    target_q = DiscreteQNetwork(config.chain_length, 2, config.hidden_sizes).to(device)
    hard_update(target_q, q_network)
    rnd = RNDModule(config.chain_length).to(device)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-3)
    predictor_optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-3)
    replay = ReplayBuffer(20_000, (config.chain_length,), (), discrete_actions=True)
    state, _ = env.reset()
    episode_return = 0.0
    returns: list[float] = []
    intrinsic_history: list[float] = []
    running_intrinsic_variance = 1.0

    for step in range(1, config.total_steps + 1):
        observation = _one_hot([state], config.chain_length)[0]
        epsilon = max(0.05, 1.0 - step / max(config.total_steps // 2, 1))
        if rng.random() < epsilon:
            action = int(rng.integers(2))
        else:
            with torch.no_grad():
                action = int(q_network(torch.as_tensor(observation, device=device).unsqueeze(0)).argmax())
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_observation = _one_hot([next_state], config.chain_length)[0]
        replay.add(observation, action, reward, next_observation, terminated)
        episode_return += reward
        state = next_state

        if step >= config.warmup_steps and len(replay) >= config.batch_size:
            batch = replay.sample(config.batch_size, device, rng)
            raw_intrinsic = rnd.intrinsic_reward(batch.next_observations)
            mean_square = float(raw_intrinsic.detach().pow(2).mean())
            running_intrinsic_variance = 0.99 * running_intrinsic_variance + 0.01 * mean_square
            normalized_intrinsic = raw_intrinsic.detach() / np.sqrt(
                running_intrinsic_variance + 1e-8
            )
            with torch.no_grad():
                target = batch.rewards + config.intrinsic_scale * normalized_intrinsic
                target += 0.99 * (1.0 - batch.terminated) * target_q(
                    batch.next_observations
                ).max(dim=1).values
            prediction = q_network(batch.observations).gather(
                1, batch.actions.long().unsqueeze(1)
            ).squeeze(1)
            q_loss = nn.functional.smooth_l1_loss(prediction, target)
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            predictor_loss = rnd.intrinsic_reward(batch.next_observations).mean()
            predictor_optimizer.zero_grad()
            predictor_loss.backward()
            predictor_optimizer.step()
            intrinsic_history.append(float(raw_intrinsic.detach().mean()))
            if step % 200 == 0:
                hard_update(target_q, q_network)

        if terminated or truncated:
            returns.append(episode_return)
            episode_return = 0.0
            state, _ = env.reset()
    return RNDResult(q_network, rnd, returns, intrinsic_history)

