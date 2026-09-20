"""Additional model-based RL algorithms: shooting, MVE, MOPO, and COMBO.

These are small, runnable versions of the central update in each method.  They
share the PointMass task so differences come from the algorithm rather than
from preprocessing or environment wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from edu_rl.advanced.model_based import DynamicsEnsemble
from edu_rl.advanced.offline_rl import OfflineConfig, OfflineDataset, OfflineResult, train_offline
from edu_rl.core.buffers import ReplayBuffer
from edu_rl.core.networks import ValueNetwork
from edu_rl.core.utils import set_global_seeds
from edu_rl.envs import PointMass2D
from edu_rl.envs.point_mass import point_mass_reward, point_mass_step


def random_shooting_action(
    state: np.ndarray,
    goal: np.ndarray,
    *,
    horizon: int = 20,
    population_size: int = 1_024,
    seed: int = 0,
) -> np.ndarray:
    """Plan by evaluating uniformly sampled action sequences (no refitting)."""

    rng = np.random.default_rng(seed)
    sequences = rng.uniform(-1.0, 1.0, size=(population_size, horizon, 2)).astype(np.float32)
    states = np.repeat(np.asarray(state, dtype=np.float32)[None], population_size, axis=0)
    returns = np.zeros(population_size, dtype=np.float32)
    for time in range(horizon):
        states = point_mass_step(states, sequences[:, time])
        returns += point_mass_reward(states, goal)
    return sequences[int(returns.argmax()), 0]


def differentiable_plan(
    state: np.ndarray,
    goal: np.ndarray,
    *,
    horizon: int = 20,
    optimization_steps: int = 100,
    learning_rate: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[np.ndarray, list[float]]:
    """Optimize an action sequence by backpropagating through known dynamics."""

    torch.manual_seed(seed)
    initial = torch.as_tensor(state, dtype=torch.float32, device=device)
    target = torch.as_tensor(goal, dtype=torch.float32, device=device)
    raw_actions = nn.Parameter(torch.zeros(horizon, 2, device=device))
    optimizer = torch.optim.Adam([raw_actions], lr=learning_rate)
    objectives: list[float] = []
    for _ in range(optimization_steps):
        current = initial
        total_reward = torch.tensor(0.0, device=device)
        for action in torch.tanh(raw_actions):
            position, velocity = current[:2], current[2:]
            next_velocity = 0.9 * velocity + 0.1 * action
            next_position = torch.clamp(position + 0.1 * next_velocity, -1.0, 1.0)
            current = torch.cat([next_position, next_velocity])
            total_reward = total_reward - torch.linalg.vector_norm(next_position - target)
        loss = -total_reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        objectives.append(float(total_reward.detach()))
    return torch.tanh(raw_actions[0]).detach().cpu().numpy(), objectives


def model_value_expansion_target(
    rewards: torch.Tensor,
    bootstrap_values: torch.Tensor,
    terminated: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Compute an MVE target from ``H`` model rewards plus a value bootstrap.

    Inputs use shape ``[batch, horizon]`` for rewards/termination and ``[batch]``
    for the value after the final model step.
    """

    if rewards.shape != terminated.shape or rewards.ndim != 2:
        raise ValueError("rewards and terminated must both have shape [batch, horizon]")
    target = torch.zeros(rewards.shape[0], device=rewards.device)
    alive = torch.ones_like(target)
    discount = 1.0
    for time in range(rewards.shape[1]):
        target = target + discount * alive * rewards[:, time]
        alive = alive * (1.0 - terminated[:, time])
        discount *= gamma
    return target + discount * alive * bootstrap_values


@dataclass
class MVEConfig:
    transitions: int = 10_000
    gradient_steps: int = 5_000
    model_epochs: int = 20
    rollout_horizon: int = 3
    batch_size: int = 128
    gamma: float = 0.99
    hidden_sizes: tuple[int, ...] = (128, 128)
    seed: int = 0
    device: str = "cpu"


@dataclass
class MVEResult:
    value_network: ValueNetwork
    value_losses: list[float]
    model_losses: list[float]


def generate_point_mass_dataset(*, transitions: int, seed: int = 0) -> OfflineDataset:
    """Mixed random/goal-directed data for offline model-based examples."""

    env = PointMass2D(max_steps=80)
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    terminals: list[float] = []
    for _ in range(transitions):
        if rng.random() < 0.5:
            action = env.action_space.sample()
        else:
            action = np.clip(8.0 * (env.goal - observation[:2]) - 2.0 * observation[2:], -1, 1)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        observations.append(observation.copy())
        actions.append(np.asarray(action, dtype=np.float32))
        rewards.append(reward)
        next_observations.append(next_observation.copy())
        terminals.append(float(terminated))
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset()
    env.close()
    return OfflineDataset(
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(next_observations, dtype=np.float32),
        np.asarray(terminals, dtype=np.float32),
    )


def _dataset_replay(dataset: OfflineDataset) -> ReplayBuffer:
    replay = ReplayBuffer(
        len(dataset), dataset.observations.shape[1:], dataset.actions.shape[1:], discrete_actions=False
    )
    for index in range(len(dataset)):
        replay.add(
            dataset.observations[index], dataset.actions[index], float(dataset.rewards[index]),
            dataset.next_observations[index], bool(dataset.terminated[index]),
        )
    return replay


def train_model_value_expansion(config: MVEConfig) -> MVEResult:
    """Fit a model ensemble and train V with short model-expanded targets."""

    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    dataset = generate_point_mass_dataset(transitions=config.transitions, seed=config.seed)
    replay = _dataset_replay(dataset)
    ensemble = DynamicsEnsemble(
        4, 2, count=3, hidden_sizes=config.hidden_sizes, device=device
    )
    model_losses = ensemble.fit(
        replay, epochs=config.model_epochs, batch_size=config.batch_size, rng=rng
    )
    value = ValueNetwork(4, config.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(value.parameters(), lr=3e-4)
    losses: list[float] = []
    goal = np.array([0.8, 0.8], dtype=np.float32)
    for _ in range(config.gradient_steps):
        indices = rng.integers(len(dataset), size=config.batch_size)
        real_states = dataset.observations[indices]
        simulated = real_states.copy()
        reward_steps: list[np.ndarray] = []
        for _ in range(config.rollout_horizon):
            actions = np.clip(8.0 * (goal - simulated[:, :2]) - 2.0 * simulated[:, 2:], -1, 1)
            simulated, _ = ensemble.predict(simulated, actions.astype(np.float32), rng)
            simulated[:, :2] = np.clip(simulated[:, :2], -1, 1)
            reward_steps.append(point_mass_reward(simulated, goal))
        state_tensor = torch.as_tensor(real_states, device=device)
        final_tensor = torch.as_tensor(simulated, device=device)
        rewards = torch.as_tensor(np.stack(reward_steps, axis=1), device=device)
        with torch.no_grad():
            bootstrap = value(final_tensor)
            target = model_value_expansion_target(
                rewards,
                bootstrap,
                torch.zeros_like(rewards),
                gamma=config.gamma,
            )
        loss = nn.functional.mse_loss(value(state_tensor), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return MVEResult(value, losses, model_losses)


def mopo_penalized_reward(
    predicted_reward: np.ndarray, model_uncertainty: np.ndarray, penalty_coefficient: float
) -> np.ndarray:
    """MOPO pessimism: lower model reward in uncertain regions."""

    return np.asarray(predicted_reward) - penalty_coefficient * np.asarray(model_uncertainty)


def combo_conservative_penalty(
    model_q_values: torch.Tensor, real_q_values: torch.Tensor, *, temperature: float = 1.0
) -> torch.Tensor:
    """COMBO-style regularizer: push down model OOD values vs real-data values."""

    model_term = temperature * torch.logsumexp(model_q_values / temperature, dim=0) - temperature * np.log(len(model_q_values))
    return model_term - real_q_values.mean()


@dataclass
class ModelBasedOfflineConfig:
    algorithm: str = "mopo"
    dataset_transitions: int = 20_000
    bootstrap_steps: int = 2_000
    gradient_steps: int = 20_000
    model_epochs: int = 20
    model_rollout_batch: int = 2_000
    model_rollout_horizon: int = 3
    batch_size: int = 256
    penalty_coefficient: float = 1.0
    hidden_sizes: tuple[int, ...] = (128, 128)
    seed: int = 0
    device: str = "cpu"


@dataclass
class ModelBasedOfflineResult:
    learner: OfflineResult
    model_losses: list[float]
    mean_uncertainties: list[float]
    synthetic_dataset_size: int


def _append_datasets(first: OfflineDataset, second: OfflineDataset) -> OfflineDataset:
    return OfflineDataset(*[
        np.concatenate([getattr(first, field), getattr(second, field)], axis=0)
        for field in OfflineDataset.__dataclass_fields__
    ])


def train_model_based_offline(config: ModelBasedOfflineConfig) -> ModelBasedOfflineResult:
    """Train an educational MOPO or COMBO pipeline on a fixed dataset.

    Both fit an ensemble, start short rollouts from real states, and then learn
    from real plus model data. MOPO penalizes synthetic reward by disagreement.
    COMBO instead applies CQL to the combined data, implementing conservative
    value learning on model-generated state-actions.
    """

    if config.algorithm not in {"mopo", "combo"}:
        raise ValueError("algorithm must be 'mopo' or 'combo'")
    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    real = generate_point_mass_dataset(
        transitions=config.dataset_transitions, seed=config.seed
    )
    replay = _dataset_replay(real)
    ensemble = DynamicsEnsemble(
        4, 2, count=5, hidden_sizes=config.hidden_sizes, device=device
    )
    model_losses = ensemble.fit(
        replay, epochs=config.model_epochs, batch_size=config.batch_size, rng=rng
    )
    # A short IQL fit supplies an in-distribution rollout policy.
    bootstrap = train_offline(
        real,
        OfflineConfig(
            algorithm="iql", gradient_steps=config.bootstrap_steps,
            batch_size=config.batch_size, hidden_sizes=config.hidden_sizes,
            seed=config.seed, device=config.device,
        ),
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
    )
    states = real.observations[
        rng.integers(len(real), size=min(config.model_rollout_batch, len(real)))
    ].copy()
    synthetic_parts: list[OfflineDataset] = []
    uncertainty_history: list[float] = []
    goal = np.array([0.8, 0.8], dtype=np.float32)
    for _ in range(config.model_rollout_horizon):
        with torch.no_grad():
            state_tensor = torch.as_tensor(states, device=device)
            actions, _, _ = bootstrap.actor.sample(state_tensor)
            action_array = actions.cpu().numpy()
        next_states, uncertainty = ensemble.predict(states, action_array, rng)
        next_states[:, :2] = np.clip(next_states[:, :2], -1, 1)
        rewards = point_mass_reward(next_states, goal)
        if config.algorithm == "mopo":
            rewards = mopo_penalized_reward(
                rewards, uncertainty, config.penalty_coefficient
            )
        synthetic_parts.append(
            OfflineDataset(
                states.copy(), action_array.copy(), rewards.astype(np.float32),
                next_states.copy(), np.zeros(len(states), dtype=np.float32),
            )
        )
        uncertainty_history.append(float(np.mean(uncertainty)))
        states = next_states
    synthetic = synthetic_parts[0]
    for part in synthetic_parts[1:]:
        synthetic = _append_datasets(synthetic, part)
    combined = _append_datasets(real, synthetic)
    learner = train_offline(
        combined,
        OfflineConfig(
            algorithm="sac_bc" if config.algorithm == "mopo" else "cql",
            gradient_steps=config.gradient_steps,
            batch_size=config.batch_size,
            behavior_coefficient=0.1 if config.algorithm == "mopo" else 1.0,
            conservative_coefficient=1.0,
            hidden_sizes=config.hidden_sizes,
            seed=config.seed + 1,
            device=config.device,
        ),
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
    )
    return ModelBasedOfflineResult(
        learner, model_losses, uncertainty_history, len(synthetic)
    )
