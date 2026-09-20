"""Sample-based Monte Carlo, TD(0), and Q-learning."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from edu_rl.core.returns import discounted_returns


Policy = Callable[[int, np.random.Generator], int]


def sample_episode(
    env: object,
    policy: Policy,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[float]]:
    observation, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
    states: list[int] = []
    actions: list[int] = []
    rewards: list[float] = []
    done = False
    while not done:
        action = policy(int(observation), rng)
        states.append(int(observation))
        actions.append(action)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        done = terminated or truncated
    return states, actions, rewards


def monte_carlo_prediction(
    env: object,
    policy: Policy,
    *,
    episodes: int,
    gamma: float = 0.99,
    seed: int = 0,
) -> np.ndarray:
    """First-visit Monte Carlo value prediction using sample averages."""

    rng = np.random.default_rng(seed)
    values = np.zeros(env.n_states, dtype=np.float64)
    visit_counts = np.zeros(env.n_states, dtype=np.int64)
    for _ in range(episodes):
        states, _, rewards = sample_episode(env, policy, rng)
        returns = discounted_returns(rewards, gamma)
        visited: set[int] = set()
        for state, return_value in zip(states, returns, strict=True):
            if state in visited:
                continue
            visited.add(state)
            visit_counts[state] += 1
            values[state] += (return_value - values[state]) / visit_counts[state]
    return values


def td0_prediction(
    env: object,
    policy: Policy,
    *,
    episodes: int,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    seed: int = 0,
) -> np.ndarray:
    """One-step TD prediction: bootstrap after every observed transition."""

    rng = np.random.default_rng(seed)
    values = np.zeros(env.n_states, dtype=np.float64)
    for _ in range(episodes):
        state, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
        done = False
        while not done:
            action = policy(int(state), rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # A truncation resets collection but is not an absorbing MDP state.
            bootstrap = 0.0 if terminated else values[next_state]
            td_error = reward + gamma * bootstrap - values[state]
            values[state] += learning_rate * td_error
            state = next_state
            done = terminated or truncated
    return values


def q_learning(
    env: object,
    *,
    episodes: int,
    learning_rate: float = 0.2,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Off-policy tabular Q-learning with linearly annealed exploration."""

    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    episode_returns: list[float] = []
    for episode in range(episodes):
        fraction = episode / max(episodes - 1, 1)
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        state, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
        done = False
        total_reward = 0.0
        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(env.n_actions))
            else:
                action = int(np.argmax(q_values[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            bootstrap = 0.0 if terminated else float(q_values[next_state].max())
            target = reward + gamma * bootstrap
            q_values[state, action] += learning_rate * (
                target - q_values[state, action]
            )
            state = next_state
            total_reward += float(reward)
            done = terminated or truncated
        episode_returns.append(total_reward)
    return q_values, episode_returns

