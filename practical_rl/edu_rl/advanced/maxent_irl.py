"""Soft Q-learning, maximum-entropy IRL, and Guided Cost Learning.

The tabular setting makes partition functions and occupancy measures explicit.
That is far more useful for learning the algorithms than hiding these quantities
inside a large neural experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from edu_rl.envs import FourRooms


def stable_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.exp(values - maximum).sum(axis=axis)
    )


def four_rooms_dynamics(env: FourRooms) -> np.ndarray:
    """Return deterministic dynamics as a ``[state, action, next_state]`` tensor."""

    dynamics = np.zeros((env.n_states, env.n_actions, env.n_states), dtype=np.float64)
    for state in range(env.n_states):
        for action in range(env.n_actions):
            dynamics[state, action, env.transition(state, action)] = 1.0
    return dynamics


def soft_value_iteration(
    dynamics: np.ndarray,
    rewards: np.ndarray,
    *,
    gamma: float = 0.99,
    temperature: float = 1.0,
    iterations: int = 500,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entropy-regularized dynamic programming and its Boltzmann policy.

    ``rewards`` may be state rewards ``[S]`` or state-action rewards ``[S,A]``.
    The soft value is ``alpha logsumexp(Q / alpha)``.
    """

    transition = np.asarray(dynamics, dtype=np.float64)
    state_count, action_count, _ = transition.shape
    reward = np.asarray(rewards, dtype=np.float64)
    if reward.shape == (state_count,):
        reward = np.repeat(reward[:, None], action_count, axis=1)
    if reward.shape != (state_count, action_count):
        raise ValueError("rewards must have shape [states] or [states, actions]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = np.zeros(state_count, dtype=np.float64)
    for _ in range(iterations):
        q_values = reward + gamma * np.einsum("san,n->sa", transition, values)
        next_values = temperature * stable_logsumexp(q_values / temperature, axis=1)
        if np.max(np.abs(next_values - values)) < tolerance:
            values = next_values
            break
        values = next_values
    q_values = reward + gamma * np.einsum("san,n->sa", transition, values)
    logits = (q_values - values[:, None]) / temperature
    policy = np.exp(logits)
    policy /= policy.sum(axis=1, keepdims=True)
    return q_values, values, policy


@dataclass
class SoftQResult:
    q_values: np.ndarray
    policy: np.ndarray
    episode_returns: list[float]


def train_soft_q_learning(
    env: FourRooms,
    *,
    episodes: int = 2_000,
    learning_rate: float = 0.2,
    gamma: float = 0.99,
    temperature: float = 0.2,
    seed: int = 0,
) -> SoftQResult:
    """Model-free tabular Soft Q-learning with a Boltzmann behavior policy."""

    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    returns: list[float] = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total = 0.0
        done = False
        while not done:
            logits = q_values[state] / temperature
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            action = int(rng.choice(env.n_actions, p=probabilities))
            next_state, reward, terminated, truncated, _ = env.step(action)
            soft_next = temperature * stable_logsumexp(
                q_values[next_state][None, :] / temperature, axis=1
            )[0]
            target = reward if terminated else reward + gamma * soft_next
            q_values[state, action] += learning_rate * (target - q_values[state, action])
            state = next_state
            total += reward
            done = terminated or truncated
        returns.append(total)
    logits = q_values / temperature
    policy = np.exp(logits - logits.max(axis=1, keepdims=True))
    policy /= policy.sum(axis=1, keepdims=True)
    return SoftQResult(q_values, policy, returns)


def expected_state_counts(
    dynamics: np.ndarray,
    policy: np.ndarray,
    initial_distribution: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Finite-horizon visitation counts under a known policy and dynamics."""

    distribution = np.asarray(initial_distribution, dtype=np.float64).copy()
    counts = np.zeros_like(distribution)
    policy_transition = np.einsum("sa,san->sn", policy, dynamics)
    for _ in range(horizon):
        counts += distribution
        distribution = distribution @ policy_transition
    return counts


def demonstrate_policy(
    env: FourRooms,
    policy: np.ndarray,
    *,
    trajectories: int,
    horizon: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    demonstrations: list[np.ndarray] = []
    for episode in range(trajectories):
        state, _ = env.reset(seed=seed + episode)
        states: list[int] = []
        for _ in range(horizon):
            states.append(state)
            row = policy[state]
            action = int(row) if np.ndim(row) == 0 else int(rng.choice(env.n_actions, p=row))
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                states.append(state)
                break
        demonstrations.append(np.asarray(states, dtype=np.int64))
    return demonstrations


def empirical_feature_counts(
    demonstrations: list[np.ndarray], features: np.ndarray
) -> np.ndarray:
    counts = np.zeros(features.shape[1], dtype=np.float64)
    for trajectory in demonstrations:
        counts += features[trajectory].sum(axis=0)
    return counts / len(demonstrations)


@dataclass
class MaxEntIRLResult:
    reward_weights: np.ndarray
    state_rewards: np.ndarray
    policy: np.ndarray
    feature_errors: list[float]


def maximum_entropy_irl(
    dynamics: np.ndarray,
    features: np.ndarray,
    demonstrations: list[np.ndarray],
    initial_state: int,
    *,
    horizon: int,
    iterations: int = 200,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    temperature: float = 1.0,
) -> MaxEntIRLResult:
    """Maximum-entropy IRL by matching expert and policy feature counts."""

    feature_matrix = np.asarray(features, dtype=np.float64)
    weights = np.zeros(feature_matrix.shape[1], dtype=np.float64)
    expert_counts = empirical_feature_counts(demonstrations, feature_matrix)
    initial = np.zeros(len(feature_matrix), dtype=np.float64)
    initial[initial_state] = 1.0
    errors: list[float] = []
    policy = np.full(dynamics.shape[:2], 1.0 / dynamics.shape[1])
    for _ in range(iterations):
        rewards = feature_matrix @ weights
        _, _, policy = soft_value_iteration(
            dynamics, rewards, gamma=gamma, temperature=temperature
        )
        state_counts = expected_state_counts(dynamics, policy, initial, horizon)
        learner_counts = state_counts @ feature_matrix
        gradient = expert_counts - learner_counts
        weights += learning_rate * gradient
        errors.append(float(np.linalg.norm(gradient)))
    return MaxEntIRLResult(weights, feature_matrix @ weights, policy, errors)


def trajectory_log_probability(
    trajectory: np.ndarray, policy: np.ndarray, dynamics: np.ndarray
) -> float:
    """Log proposal probability, marginalizing actions that reach each next state."""

    result = 0.0
    for state, next_state in zip(trajectory[:-1], trajectory[1:], strict=True):
        probability = np.sum(policy[state] * dynamics[state, :, next_state])
        result += np.log(probability + 1e-12)
    return float(result)


@dataclass
class GCLResult:
    reward_weights: np.ndarray
    policy: np.ndarray
    losses: list[float]


def guided_cost_learning(
    dynamics: np.ndarray,
    features: np.ndarray,
    demonstrations: list[np.ndarray],
    initial_state: int,
    *,
    horizon: int,
    iterations: int = 100,
    sampled_trajectories: int = 32,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> GCLResult:
    """Alternating importance-aware reward learning and soft policy updates.

    The discriminator logit is trajectory reward minus log proposal density.
    Including that density correction is the key Guided Cost Learning idea: as
    the sampler changes, reward learning still estimates an energy model rather
    than merely identifying the latest policy.
    """

    rng = np.random.default_rng(seed)
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    expert_features = torch.stack(
        [feature_tensor[trajectory].sum(dim=0) for trajectory in demonstrations]
    )
    weights = torch.zeros(features.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=learning_rate)
    policy = np.full(dynamics.shape[:2], 1.0 / dynamics.shape[1])
    losses: list[float] = []

    for iteration in range(iterations):
        sampled: list[np.ndarray] = []
        log_proposals: list[float] = []
        for _ in range(sampled_trajectories):
            states = [initial_state]
            state = initial_state
            for _ in range(horizon - 1):
                action = int(rng.choice(dynamics.shape[1], p=policy[state]))
                state = int(rng.choice(dynamics.shape[2], p=dynamics[state, action]))
                states.append(state)
            trajectory = np.asarray(states, dtype=np.int64)
            sampled.append(trajectory)
            log_proposals.append(trajectory_log_probability(trajectory, policy, dynamics))
        sampled_features = torch.stack(
            [feature_tensor[trajectory].sum(dim=0) for trajectory in sampled]
        )
        proposal = torch.as_tensor(log_proposals, dtype=torch.float32)
        expert_proposal = torch.as_tensor(
            [trajectory_log_probability(path, policy, dynamics) for path in demonstrations],
            dtype=torch.float32,
        )
        expert_logits = expert_features @ weights - expert_proposal
        sample_logits = sampled_features @ weights - proposal
        loss = nn.functional.softplus(-expert_logits).mean() + nn.functional.softplus(sample_logits).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        rewards = features @ weights.detach().numpy()
        _, _, policy = soft_value_iteration(dynamics, rewards, temperature=1.0)
    return GCLResult(weights.detach().numpy(), policy, losses)
