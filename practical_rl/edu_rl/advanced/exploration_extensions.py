"""Density pseudo-counts, Context Tree Switching, and empowerment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from edu_rl.envs import FourRooms, GridWorld


def density_pseudo_count(probability_before: float, probability_after: float) -> float:
    """Convert a density model's learning progress into a generalized count.

    The formula is the count that would produce the same before/after
    probabilities in a literal count table.  A non-learning or decreasing
    density has no finite useful pseudo-count and is clipped to a large value.
    """

    before = float(np.clip(probability_before, 1e-12, 1.0 - 1e-12))
    after = float(np.clip(probability_after, 1e-12, 1.0 - 1e-12))
    if after <= before:
        return 1e12
    return max(0.0, before * (1.0 - after) / (after - before))


class DirichletStateDensity:
    """A simple online categorical density whose pseudo-count is interpretable."""

    def __init__(self, state_count: int, prior: float = 0.5) -> None:
        self.counts = np.full(state_count, prior, dtype=np.float64)

    def probability(self, state: int) -> float:
        return float(self.counts[state] / self.counts.sum())

    def update(self, state: int) -> tuple[float, float, float]:
        before = self.probability(state)
        self.counts[state] += 1.0
        after = self.probability(state)
        return before, after, density_pseudo_count(before, after)


def train_pseudocount_q_learning(
    env: GridWorld,
    *,
    episodes: int = 1_000,
    bonus_scale: float = 0.1,
    learning_rate: float = 0.2,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, list[float], list[float]]:
    """Q-learning with ``beta / sqrt(pseudo_count)`` intrinsic reward."""

    rng = np.random.default_rng(seed)
    density = DirichletStateDensity(env.n_states)
    q_values = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    returns: list[float] = []
    bonuses: list[float] = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total = 0.0
        done = False
        while not done:
            action = int(rng.integers(env.n_actions)) if rng.random() < epsilon else int(q_values[state].argmax())
            next_state, reward, terminated, truncated, _ = env.step(action)
            _, _, pseudo_count = density.update(next_state)
            bonus = bonus_scale / np.sqrt(pseudo_count + 1e-8)
            target = reward + bonus
            if not terminated:
                target += gamma * q_values[next_state].max()
            q_values[state, action] += learning_rate * (target - q_values[state, action])
            bonuses.append(float(bonus))
            total += reward
            state = next_state
            done = terminated or truncated
        returns.append(total)
    return q_values, returns, bonuses


class ContextTreeSwitching:
    """Online binary density that switches among variable context depths.

    Every depth is a Krichevsky–Trofimov (KT) predictor.  Bayesian weights are
    updated after each bit, with a small share probability that lets mass move
    between context lengths when the sequence's structure changes.  This is the
    core switching mechanism used by CTS-based exploration density models.
    """

    def __init__(self, maximum_depth: int = 8, switch_rate: float = 0.01) -> None:
        if maximum_depth < 0:
            raise ValueError("maximum_depth must be nonnegative")
        self.maximum_depth = maximum_depth
        self.switch_rate = switch_rate
        self.weights = np.full(maximum_depth + 1, 1.0 / (maximum_depth + 1))
        self.counts: list[dict[tuple[int, ...], np.ndarray]] = [
            {} for _ in range(maximum_depth + 1)
        ]
        self.history: list[int] = []

    def _context(self, depth: int) -> tuple[int, ...]:
        return tuple(self.history[-depth:]) if depth else ()

    def _expert_probability(self, depth: int, bit: int) -> float:
        counts = self.counts[depth].get(self._context(depth))
        if counts is None:
            counts = np.zeros(2, dtype=np.float64)
        return float((counts[bit] + 0.5) / (counts.sum() + 1.0))

    def probability(self, bit: int) -> float:
        if bit not in (0, 1):
            raise ValueError("CTS accepts binary symbols 0 or 1")
        predictions = np.array(
            [self._expert_probability(depth, bit) for depth in range(self.maximum_depth + 1)]
        )
        return float(self.weights @ predictions)

    def update(self, bit: int) -> tuple[float, float]:
        """Return predictive probability before and after learning the bit."""

        before = self.probability(bit)
        predictions = np.array(
            [self._expert_probability(depth, bit) for depth in range(self.maximum_depth + 1)]
        )
        # Fixed-share Bayesian update permits switches between context experts.
        shared = (1.0 - self.switch_rate) * self.weights + self.switch_rate / len(self.weights)
        posterior = shared * predictions
        self.weights = posterior / posterior.sum()
        for depth in range(self.maximum_depth + 1):
            context = self._context(depth)
            counts = self.counts[depth].setdefault(context, np.zeros(2, dtype=np.float64))
            counts[bit] += 1.0
        # Evaluate the same pre-update context after its count changed.  Only
        # then advance history to the context for the next symbol.
        after = self.probability(bit)
        self.history.append(bit)
        return before, after

    def sequence_log_probability(self, sequence: list[int] | np.ndarray) -> float:
        result = 0.0
        for bit in sequence:
            before, _ = self.update(int(bit))
            result += np.log(before + 1e-12)
        return float(result)


@dataclass
class CTSResult:
    surprisals: list[float]
    pseudo_counts: list[float]


def run_cts(sequence: list[int] | np.ndarray, *, maximum_depth: int = 8) -> CTSResult:
    model = ContextTreeSwitching(maximum_depth=maximum_depth)
    surprisals: list[float] = []
    pseudo_counts: list[float] = []
    for bit in sequence:
        before, after = model.update(int(bit))
        surprisals.append(float(-np.log(before + 1e-12)))
        pseudo_counts.append(density_pseudo_count(before, after))
    return CTSResult(surprisals, pseudo_counts)


def blahut_arimoto(
    channel: np.ndarray,
    *,
    iterations: int = 500,
    tolerance: float = 1e-10,
) -> tuple[float, np.ndarray]:
    """Channel capacity and maximizing input distribution, measured in nats."""

    probabilities = np.asarray(channel, dtype=np.float64)
    if probabilities.ndim != 2 or np.any(probabilities < 0):
        raise ValueError("channel must be a nonnegative [input, output] matrix")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("every channel row must have positive mass")
    probabilities = probabilities / row_sums
    input_distribution = np.full(len(probabilities), 1.0 / len(probabilities))
    for _ in range(iterations):
        output_distribution = input_distribution @ probabilities
        divergences = np.sum(
            np.where(
                probabilities > 0,
                probabilities * np.log((probabilities + 1e-300) / (output_distribution + 1e-300)),
                0.0,
            ),
            axis=1,
        )
        updated = np.exp(divergences - divergences.max())
        updated /= updated.sum()
        if np.max(np.abs(updated - input_distribution)) < tolerance:
            input_distribution = updated
            break
        input_distribution = updated
    output_distribution = input_distribution @ probabilities
    capacity = np.sum(
        input_distribution[:, None]
        * probabilities
        * np.where(
            probabilities > 0,
            np.log((probabilities + 1e-300) / (output_distribution + 1e-300)),
            0.0,
        )
    )
    return float(capacity), input_distribution


def action_sequence_channel(
    env: FourRooms, state: int, *, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Channel from open-loop action sequences to achieved terminal goals."""

    sequences = np.asarray(
        list(product(range(env.n_actions), repeat=horizon)), dtype=np.int64
    )
    channel = np.zeros((len(sequences), env.n_states), dtype=np.float64)
    for index, sequence in enumerate(sequences):
        reached = state
        for action in sequence:
            reached = env.transition(reached, int(action))
        channel[index, reached] = 1.0
    return sequences, channel


def goal_conditioned_empowerment(
    env: FourRooms, state: int, *, horizon: int = 3
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mutual information between action sequences and achieved goals.

    For deterministic Four Rooms, this equals the log of the number of distinct
    reachable goals, but Blahut–Arimoto also works unchanged for noisy dynamics.
    """

    sequences, channel = action_sequence_channel(env, state, horizon=horizon)
    capacity, distribution = blahut_arimoto(channel)
    reached_goals = np.flatnonzero(channel.sum(axis=0))
    return capacity, distribution, reached_goals
