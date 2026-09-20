"""Successor features, generalized policy improvement, options, and Option-Critic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from edu_rl.core.networks import mlp
from edu_rl.core.utils import set_global_seeds
from edu_rl.envs import FourRooms


def learn_successor_features(
    env: FourRooms,
    policy: np.ndarray,
    *,
    episodes: int = 5_000,
    gamma: float = 0.95,
    learning_rate: float = 0.2,
    epsilon: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Learn state-action successor features for one fixed policy.

    Features are one-hot next-state occupancies.  Dotting the learned tensor with
    any state-reward vector therefore recovers Q-values for that new reward.
    """

    rng = np.random.default_rng(seed)
    successor = np.zeros((env.n_states, env.n_actions, env.n_states))
    for episode in range(episodes):
        state = int(rng.integers(env.n_states))
        for _ in range(env.max_steps):
            action = (
                int(rng.integers(env.n_actions))
                if rng.random() < epsilon
                else int(policy[state])
            )
            # Successor features describe dynamics and a policy, not a particular
            # task's terminal goal.  Use the raw transition graph so reward-task
            # termination does not leak into the transferable representation.
            next_state = env.transition(state, action)
            features = np.eye(env.n_states)[next_state]
            continuation = successor[next_state, int(policy[next_state])]
            target = features + gamma * continuation
            successor[state, action] += learning_rate * (
                target - successor[state, action]
            )
            state = next_state
    return successor


def generalized_policy_improvement(
    successor_library: np.ndarray, reward_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select the best action offered by any stored policy at each state."""

    # [policies, states, actions, features] dot [features]
    library_q = np.tensordot(successor_library, reward_weights, axes=([3], [0]))
    gpi_q = library_q.max(axis=0)
    return gpi_q.argmax(axis=1), gpi_q


@dataclass(frozen=True)
class Option:
    name: str
    target_state: int
    policy: np.ndarray

    def available(self, state: int) -> bool:
        return state != self.target_state


def doorway_options(env: FourRooms) -> list[Option]:
    options: list[Option] = []
    for row, column in sorted(env.doorways):
        target = env.state_for_cell[(row, column)]
        options.append(
            Option(
                name=f"door-{row}-{column}",
                target_state=target,
                policy=env.shortest_path_policy(target),
            )
        )
    return options


def semi_mdp_target(
    discounted_option_reward: float,
    duration: int,
    next_value: float,
    *,
    gamma: float,
    terminal: bool,
) -> float:
    """Option backup uses ``gamma**duration``, not a one-step discount."""

    return discounted_option_reward + (
        0.0 if terminal else gamma**duration * next_value
    )


def execute_option(
    env: FourRooms, option: Option, *, gamma: float
) -> tuple[int, float, float, int, bool, bool]:
    state = env.state
    discounted_reward = 0.0
    raw_reward = 0.0
    duration = 0
    terminated = truncated = False
    while state != option.target_state and not (terminated or truncated):
        action = int(option.policy[state])
        state, reward, terminated, truncated, _ = env.step(action)
        discounted_reward += gamma**duration * reward
        raw_reward += reward
        duration += 1
    return state, discounted_reward, raw_reward, duration, terminated, truncated


@dataclass
class OptionsResult:
    q_values: np.ndarray
    episode_returns: list[float]
    episode_decisions: list[int]


def train_smdp_options(
    env: FourRooms,
    *,
    episodes: int = 1_000,
    gamma: float = 0.99,
    learning_rate: float = 0.2,
    epsilon: float = 0.1,
    seed: int = 0,
) -> OptionsResult:
    """Q-learning over primitive actions and fixed doorway options."""

    rng = np.random.default_rng(seed)
    options = doorway_options(env)
    choice_count = env.n_actions + len(options)
    q_values = np.zeros((env.n_states, choice_count))
    returns: list[float] = []
    decision_counts: list[int] = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total = 0.0
        decisions = 0
        done = False
        while not done:
            available = list(range(env.n_actions)) + [
                env.n_actions + index
                for index, option in enumerate(options)
                if option.available(state)
            ]
            if rng.random() < epsilon:
                choice = int(rng.choice(available))
            else:
                choice = max(available, key=lambda item: q_values[state, item])
            if choice < env.n_actions:
                next_state, reward, terminated, truncated, _ = env.step(choice)
                discounted_reward = reward
                raw_reward = reward
                duration = 1
            else:
                (
                    next_state,
                    discounted_reward,
                    raw_reward,
                    duration,
                    terminated,
                    truncated,
                ) = execute_option(env, options[choice - env.n_actions], gamma=gamma)
            target = semi_mdp_target(
                discounted_reward,
                duration,
                q_values[next_state].max(),
                gamma=gamma,
                terminal=terminated,
            )
            q_values[state, choice] += learning_rate * (
                target - q_values[state, choice]
            )
            total += raw_reward
            decisions += 1
            state = next_state
            done = terminated or truncated
        returns.append(total)
        decision_counts.append(decisions)
    return OptionsResult(q_values, returns, decision_counts)


class OptionCriticNetwork(nn.Module):
    """Shared representation with option values, intra-option policies, and beta."""

    def __init__(self, state_count: int, action_count: int, option_count: int) -> None:
        super().__init__()
        self.state_count = state_count
        self.action_count = action_count
        self.option_count = option_count
        self.encoder = mlp(state_count, (64,), 64)
        self.option_values = nn.Linear(64, option_count)
        self.intra_option_logits = nn.Linear(64, option_count * action_count)
        self.termination_logits = nn.Linear(64, option_count)

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(observations)
        values = self.option_values(features)
        action_logits = self.intra_option_logits(features).view(
            -1, self.option_count, self.action_count
        )
        terminations = torch.sigmoid(self.termination_logits(features))
        return values, action_logits, terminations


@dataclass
class OptionCriticConfig:
    episodes: int = 2_000
    option_count: int = 4
    gamma: float = 0.99
    learning_rate: float = 3e-4
    epsilon: float = 0.1
    deliberation_cost: float = 0.01
    seed: int = 0
    device: str = "cpu"


@dataclass
class OptionCriticResult:
    network: OptionCriticNetwork
    episode_returns: list[float]
    option_switches: list[int]


def train_option_critic(
    env: FourRooms, config: OptionCriticConfig
) -> OptionCriticResult:
    """One-step Option-Critic with termination-gradient and intra-option updates."""

    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    network = OptionCriticNetwork(
        env.n_states, env.n_actions, config.option_count
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    eye = torch.eye(env.n_states, device=device)
    returns: list[float] = []
    switch_history: list[int] = []

    for episode in range(config.episodes):
        state, _ = env.reset(seed=config.seed + episode)
        with torch.no_grad():
            values, _, _ = network(eye[state].unsqueeze(0))
        option = (
            int(rng.integers(config.option_count))
            if rng.random() < config.epsilon
            else int(values.argmax())
        )
        total = 0.0
        switches = 0
        done = False
        while not done:
            observation = eye[state].unsqueeze(0)
            option_values, action_logits, _ = network(observation)
            action_distribution = torch.distributions.Categorical(
                logits=action_logits[0, option]
            )
            action = action_distribution.sample()
            next_state, reward, terminated, truncated, _ = env.step(int(action))
            next_observation = eye[next_state].unsqueeze(0)
            next_values, _, next_terminations = network(next_observation)
            beta = next_terminations[0, option]
            continuation = (1.0 - beta) * next_values[0, option] + beta * next_values.max()
            target = torch.as_tensor(reward, device=device)
            if not terminated:
                target = target + config.gamma * continuation.detach()
            td_error = target - option_values[0, option]
            critic_loss = 0.5 * td_error.pow(2)
            policy_loss = -action_distribution.log_prob(action) * td_error.detach()
            option_advantage = (
                next_values[0, option] - next_values.max() + config.deliberation_cost
            ).detach()
            termination_loss = beta * option_advantage
            loss = critic_loss + policy_loss + termination_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()

            total += reward
            state = next_state
            done = terminated or truncated
            if not done and rng.random() < float(beta.detach()):
                with torch.no_grad():
                    next_option_values, _, _ = network(next_observation)
                option = (
                    int(rng.integers(config.option_count))
                    if rng.random() < config.epsilon
                    else int(next_option_values.argmax())
                )
                switches += 1
        returns.append(total)
        switch_history.append(switches)
    return OptionCriticResult(network, returns, switch_history)
