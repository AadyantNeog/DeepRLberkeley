"""Importance-sampling estimators used to reason about off-policy data."""

from __future__ import annotations

import numpy as np


def importance_ratios(
    target_log_probabilities: np.ndarray,
    behavior_log_probabilities: np.ndarray,
    *,
    max_log_ratio: float = 20.0,
) -> np.ndarray:
    """Compute stable probability ratios ``pi_target / pi_behavior``."""

    log_ratios = np.asarray(target_log_probabilities) - np.asarray(
        behavior_log_probabilities
    )
    return np.exp(np.clip(log_ratios, -max_log_ratio, max_log_ratio))


def trajectory_weights(
    target_log_probabilities: np.ndarray,
    behavior_log_probabilities: np.ndarray,
) -> np.ndarray:
    """Multiply step ratios within each trajectory by summing log-ratios."""

    target = np.asarray(target_log_probabilities, dtype=np.float64)
    behavior = np.asarray(behavior_log_probabilities, dtype=np.float64)
    if target.shape != behavior.shape or target.ndim != 2:
        raise ValueError("log-probability arrays must have shape [trajectories, time]")
    return np.exp(np.clip((target - behavior).sum(axis=1), -20.0, 20.0))


def ordinary_importance_sampling(returns: np.ndarray, weights: np.ndarray) -> float:
    """Unbiased in the ideal setting, but often dominated by rare huge weights."""

    return float(np.mean(np.asarray(returns) * np.asarray(weights)))


def weighted_importance_sampling(returns: np.ndarray, weights: np.ndarray) -> float:
    """Normalize weights to reduce variance at the cost of finite-sample bias."""

    returns = np.asarray(returns, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        raise ValueError("importance weights must have positive total mass")
    return float(np.dot(returns, weights) / total_weight)


def bandit_importance_sampling_demo(
    *,
    samples: int = 10_000,
    behavior_probability: float = 0.2,
    target_probability: float = 0.8,
    seed: int = 0,
) -> dict[str, float]:
    """Estimate a target Bernoulli policy's value using behavior-policy data.

    Action 1 produces reward 1 and action 0 produces reward 0, so the exact
    target value is simply ``target_probability``.
    """

    rng = np.random.default_rng(seed)
    actions = rng.binomial(1, behavior_probability, size=samples)
    behavior_probabilities = np.where(
        actions == 1, behavior_probability, 1.0 - behavior_probability
    )
    target_probabilities = np.where(
        actions == 1, target_probability, 1.0 - target_probability
    )
    weights = target_probabilities / behavior_probabilities
    rewards = actions.astype(np.float64)
    return {
        "exact": target_probability,
        "uncorrected_behavior_mean": float(rewards.mean()),
        "ordinary_is": ordinary_importance_sampling(rewards, weights),
        "weighted_is": weighted_importance_sampling(rewards, weights),
        "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
    }

