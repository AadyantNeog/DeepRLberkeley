"""Goal-conditioned DQN with future-strategy Hindsight Experience Replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from edu_rl.core.buffers import ReplayBuffer
from edu_rl.core.networks import DiscreteQNetwork
from edu_rl.core.utils import hard_update, set_global_seeds
from edu_rl.envs import BitFlipEnv


@dataclass
class GoalTransition:
    state: np.ndarray
    goal: np.ndarray
    action: int
    next_state: np.ndarray


class HERReplayBuffer:
    """Store original transitions plus future-achieved-goal relabels."""

    def __init__(self, capacity: int, n_bits: int, her_k: int, seed: int) -> None:
        self.replay = ReplayBuffer(
            capacity, (2 * n_bits,), (), discrete_actions=True
        )
        self.n_bits = n_bits
        self.her_k = her_k
        self.rng = np.random.default_rng(seed)

    def add_episode(self, episode: list[GoalTransition]) -> None:
        for index, transition in enumerate(episode):
            reward = BitFlipEnv.reward_for(transition.next_state, transition.goal)
            self.replay.add(
                np.concatenate([transition.state, transition.goal]),
                transition.action,
                reward,
                np.concatenate([transition.next_state, transition.goal]),
                reward == 0.0,
            )
            future_indices = self.rng.integers(index, len(episode), size=self.her_k)
            for future_index in future_indices:
                new_goal = episode[int(future_index)].next_state
                new_reward = BitFlipEnv.reward_for(transition.next_state, new_goal)
                self.replay.add(
                    np.concatenate([transition.state, new_goal]),
                    transition.action,
                    new_reward,
                    np.concatenate([transition.next_state, new_goal]),
                    new_reward == 0.0,
                )

    def __len__(self) -> int:
        return len(self.replay)


@dataclass
class HERConfig:
    episodes: int = 3_000
    n_bits: int = 6
    her_k: int = 4
    batch_size: int = 128
    gamma: float = 0.98
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (128, 128)
    seed: int = 0
    device: str = "cpu"


@dataclass
class HERResult:
    q_network: DiscreteQNetwork
    success_history: list[float]
    losses: list[float]


def train_her(config: HERConfig) -> HERResult:
    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    env = BitFlipEnv(config.n_bits)
    q_network = DiscreteQNetwork(
        2 * config.n_bits, config.n_bits, config.hidden_sizes
    ).to(device)
    target_q = DiscreteQNetwork(
        2 * config.n_bits, config.n_bits, config.hidden_sizes
    ).to(device)
    hard_update(target_q, q_network)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=config.learning_rate)
    replay = HERReplayBuffer(200_000, config.n_bits, config.her_k, config.seed)
    success_history: list[float] = []
    losses: list[float] = []

    for episode_index in range(config.episodes):
        observation, _ = env.reset(seed=config.seed + episode_index)
        state = observation[: config.n_bits].copy()
        goal = observation[config.n_bits :].copy()
        episode: list[GoalTransition] = []
        done = False
        success = 0.0
        epsilon = max(0.05, 1.0 - episode_index / max(config.episodes * 0.6, 1))
        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(config.n_bits))
            else:
                with torch.no_grad():
                    action = int(
                        q_network(
                            torch.as_tensor(observation, device=device).unsqueeze(0)
                        ).argmax()
                    )
            next_observation, _, terminated, truncated, _ = env.step(action)
            next_state = next_observation[: config.n_bits].copy()
            episode.append(GoalTransition(state.copy(), goal.copy(), action, next_state.copy()))
            state = next_state
            observation = next_observation
            done = terminated or truncated
            success = float(terminated)
        replay.add_episode(episode)
        success_history.append(success)

        if len(replay) >= config.batch_size:
            for _ in range(max(1, len(episode))):
                batch = replay.replay.sample(config.batch_size, device, rng)
                with torch.no_grad():
                    target = batch.rewards + config.gamma * (
                        1.0 - batch.terminated
                    ) * target_q(batch.next_observations).max(dim=1).values
                prediction = q_network(batch.observations).gather(
                    1, batch.actions.long().unsqueeze(1)
                ).squeeze(1)
                loss = nn.functional.smooth_l1_loss(prediction, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        if (episode_index + 1) % 50 == 0:
            hard_update(target_q, q_network)
    return HERResult(q_network, success_history, losses)

