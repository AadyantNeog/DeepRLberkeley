"""Exact policy evaluation, policy iteration, and value iteration."""

from __future__ import annotations

import numpy as np


def action_values(
    transitions: np.ndarray,
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Apply one Bellman lookahead for every state-action pair."""

    return np.sum(transitions * (rewards + gamma * values[None, None, :]), axis=2)


def policy_evaluation(
    transitions: np.ndarray,
    rewards: np.ndarray,
    policy: np.ndarray,
    *,
    gamma: float = 0.99,
    tolerance: float = 1e-10,
    max_iterations: int = 100_000,
) -> np.ndarray:
    """Iteratively solve the Bellman expectation equation for ``policy``."""

    n_states, n_actions, _ = transitions.shape
    if policy.shape == (n_states,):
        policy_probabilities = np.eye(n_actions)[policy.astype(int)]
    elif policy.shape == (n_states, n_actions):
        policy_probabilities = policy
    else:
        raise ValueError("policy must contain actions or per-action probabilities")

    values = np.zeros(n_states, dtype=np.float64)
    for _ in range(max_iterations):
        q_values = action_values(transitions, rewards, values, gamma)
        updated_values = np.sum(policy_probabilities * q_values, axis=1)
        if np.max(np.abs(updated_values - values)) < tolerance:
            return updated_values
        values = updated_values
    raise RuntimeError("Policy evaluation did not converge")


def policy_iteration(
    transitions: np.ndarray,
    rewards: np.ndarray,
    *,
    gamma: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Alternate exact evaluation with greedy policy improvement."""

    n_states, _, _ = transitions.shape
    policy = np.zeros(n_states, dtype=np.int64)
    while True:
        values = policy_evaluation(transitions, rewards, policy, gamma=gamma)
        improved_policy = action_values(transitions, rewards, values, gamma).argmax(1)
        if np.array_equal(policy, improved_policy):
            return values, policy
        policy = improved_policy


def value_iteration(
    transitions: np.ndarray,
    rewards: np.ndarray,
    *,
    gamma: float = 0.99,
    tolerance: float = 1e-10,
    max_iterations: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeatedly apply the Bellman optimality backup."""

    values = np.zeros(transitions.shape[0], dtype=np.float64)
    for _ in range(max_iterations):
        q_values = action_values(transitions, rewards, values, gamma)
        updated_values = q_values.max(axis=1)
        if np.max(np.abs(updated_values - values)) < tolerance:
            return updated_values, q_values.argmax(axis=1)
        values = updated_values
    raise RuntimeError("Value iteration did not converge")

