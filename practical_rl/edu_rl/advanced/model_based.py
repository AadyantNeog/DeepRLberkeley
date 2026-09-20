"""CEM/MPC planning, Dyna-Q, learned dynamics ensembles, and MBPO-style rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from edu_rl.advanced.sac import SACAgent
from edu_rl.core.buffers import ReplayBatch, ReplayBuffer
from edu_rl.core.networks import mlp
from edu_rl.core.utils import set_global_seeds
from edu_rl.envs import GridWorld, PointMass2D
from edu_rl.envs.point_mass import point_mass_reward, point_mass_step


def cross_entropy_method(
    objective: Callable[[np.ndarray], np.ndarray],
    *,
    dimension: int,
    lower: np.ndarray | float,
    upper: np.ndarray | float,
    population_size: int = 512,
    elite_fraction: float = 0.1,
    iterations: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Maximize a black-box objective by repeatedly refitting elite samples."""

    rng = np.random.default_rng(seed)
    lower_array = np.broadcast_to(lower, (dimension,)).astype(np.float32)
    upper_array = np.broadcast_to(upper, (dimension,)).astype(np.float32)
    mean = (lower_array + upper_array) / 2.0
    standard_deviation = (upper_array - lower_array) / 2.0
    elite_count = max(1, int(population_size * elite_fraction))
    best_sample = mean.copy()
    best_value = -np.inf
    for _ in range(iterations):
        samples = rng.normal(mean, standard_deviation, size=(population_size, dimension))
        samples = np.clip(samples, lower_array, upper_array).astype(np.float32)
        values = np.asarray(objective(samples), dtype=np.float32)
        elite_indices = np.argpartition(values, -elite_count)[-elite_count:]
        elites = samples[elite_indices]
        mean = elites.mean(axis=0)
        standard_deviation = np.maximum(elites.std(axis=0), 1e-3)
        candidate_index = int(np.argmax(values))
        if values[candidate_index] > best_value:
            best_value = float(values[candidate_index])
            best_sample = samples[candidate_index].copy()
    return best_sample, best_value


def plan_point_mass_action(
    state: np.ndarray,
    goal: np.ndarray,
    *,
    horizon: int = 20,
    population_size: int = 256,
    iterations: int = 5,
    dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray] = point_mass_step,
    seed: int = 0,
) -> np.ndarray:
    """Use CEM inside MPC and return only the first planned action."""

    action_size = 2

    def objective(flat_sequences: np.ndarray) -> np.ndarray:
        action_sequences = flat_sequences.reshape(-1, horizon, action_size)
        states = np.repeat(np.asarray(state)[None, :], len(action_sequences), axis=0)
        returns = np.zeros(len(action_sequences), dtype=np.float32)
        for time in range(horizon):
            states = dynamics(states, action_sequences[:, time])
            returns += point_mass_reward(states, goal)
        return returns

    sequence, _ = cross_entropy_method(
        objective,
        dimension=horizon * action_size,
        lower=-1.0,
        upper=1.0,
        population_size=population_size,
        iterations=iterations,
        seed=seed,
    )
    return sequence.reshape(horizon, action_size)[0]


def run_mpc(
    *,
    episodes: int = 5,
    horizon: int = 15,
    seed: int = 0,
) -> list[float]:
    env = PointMass2D(max_steps=60)
    returns: list[float] = []
    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)
        total = 0.0
        done = False
        step = 0
        while not done:
            action = plan_point_mass_action(
                state, info["goal"], horizon=horizon, seed=seed + episode * 100 + step
            )
            state, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
            step += 1
        returns.append(total)
    env.close()
    return returns


def train_dyna_q(
    env: GridWorld,
    *,
    episodes: int = 500,
    planning_steps: int = 10,
    learning_rate: float = 0.2,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Tabular Dyna-Q: each real step is followed by simulated Q updates."""

    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    model: dict[tuple[int, int], tuple[int, float, bool]] = {}
    returns: list[float] = []

    def update(state: int, action: int, next_state: int, reward: float, terminal: bool) -> None:
        bootstrap = 0.0 if terminal else q_values[next_state].max()
        q_values[state, action] += learning_rate * (
            reward + gamma * bootstrap - q_values[state, action]
        )

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total = 0.0
        done = False
        while not done:
            action = (
                int(rng.integers(env.n_actions))
                if rng.random() < epsilon
                else int(q_values[state].argmax())
            )
            next_state, reward, terminated, truncated, _ = env.step(action)
            update(state, action, next_state, reward, terminated)
            model[(state, action)] = (next_state, reward, terminated)
            keys = list(model)
            for _ in range(planning_steps):
                simulated_state, simulated_action = keys[int(rng.integers(len(keys)))]
                simulated_next, simulated_reward, simulated_terminal = model[
                    (simulated_state, simulated_action)
                ]
                update(
                    simulated_state,
                    simulated_action,
                    simulated_next,
                    simulated_reward,
                    simulated_terminal,
                )
            state = next_state
            total += reward
            done = terminated or truncated
        returns.append(total)
    return q_values, returns


class ProbabilisticDynamics(nn.Module):
    """Predict a Gaussian distribution over state deltas."""

    def __init__(self, observation_size: int, action_size: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        self.observation_size = observation_size
        self.network = mlp(
            observation_size + action_size, hidden_sizes, 2 * observation_size
        )

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_variance = self.network(
            torch.cat([observations, actions], dim=-1)
        ).chunk(2, dim=-1)
        return mean, log_variance.clamp(-8.0, 2.0)

    def loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
    ) -> torch.Tensor:
        mean, log_variance = self(observations, actions)
        target_delta = next_observations - observations
        return ((mean - target_delta).pow(2) * (-log_variance).exp() + log_variance).mean()


class DynamicsEnsemble:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        *,
        count: int = 5,
        hidden_sizes: tuple[int, ...] = (128, 128),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.models = [
            ProbabilisticDynamics(observation_size, action_size, hidden_sizes).to(device)
            for _ in range(count)
        ]
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in self.models]

    def fit(self, replay: ReplayBuffer, *, epochs: int, batch_size: int, rng: np.random.Generator) -> list[float]:
        losses: list[float] = []
        for model, optimizer in zip(self.models, self.optimizers, strict=True):
            for _ in range(epochs):
                batch = replay.sample(min(batch_size, len(replay)), self.device, rng)
                loss = model.loss(
                    batch.observations, batch.actions.float(), batch.next_observations
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        return losses

    @torch.no_grad()
    def predict(
        self, observations: np.ndarray, actions: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        observation_tensor = torch.as_tensor(observations, device=self.device)
        action_tensor = torch.as_tensor(actions, device=self.device)
        predictions = []
        for model in self.models:
            mean, _ = model(observation_tensor, action_tensor)
            predictions.append(observation_tensor + mean)
        stacked = torch.stack(predictions)
        disagreement = stacked.std(dim=0).mean(dim=1)
        model_indices = rng.integers(len(self.models), size=len(observations))
        selected = stacked[model_indices, torch.arange(len(observations), device=self.device)]
        return selected.cpu().numpy(), disagreement.cpu().numpy()


@dataclass
class MBPOConfig:
    total_steps: int = 20_000
    warmup_steps: int = 500
    batch_size: int = 128
    model_train_interval: int = 250
    model_epochs: int = 20
    model_rollout_batch: int = 256
    model_rollout_horizon: int = 3
    ensemble_size: int = 5
    hidden_sizes: tuple[int, ...] = (128, 128)
    seed: int = 0
    device: str = "cpu"


@dataclass
class MBPOResult:
    agent: SACAgent
    episode_returns: list[float]
    model_losses: list[float]
    model_disagreements: list[float]


def _mixed_batch(
    real: ReplayBuffer,
    model: ReplayBuffer,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> ReplayBatch:
    model_count = min(batch_size // 2, len(model))
    real_count = batch_size - model_count
    real_batch = real.sample(real_count, device, rng)
    if model_count == 0:
        return real_batch
    model_batch = model.sample(model_count, device, rng)
    return ReplayBatch(*[
        torch.cat([getattr(real_batch, field), getattr(model_batch, field)], dim=0)
        for field in ReplayBatch.__dataclass_fields__
    ])


def train_mbpo(config: MBPOConfig) -> MBPOResult:
    """An educational MBPO loop with real-anchored short ensemble rollouts."""

    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    env = PointMass2D(max_steps=80)
    observation_size = 4
    action_size = 2
    agent = SACAgent(
        observation_size,
        env.action_space.low,
        env.action_space.high,
        hidden_sizes=config.hidden_sizes,
        device=device,
    )
    real_buffer = ReplayBuffer(100_000, (4,), (2,), discrete_actions=False)
    model_buffer = ReplayBuffer(200_000, (4,), (2,), discrete_actions=False)
    ensemble = DynamicsEnsemble(
        4,
        2,
        count=config.ensemble_size,
        hidden_sizes=config.hidden_sizes,
        device=device,
    )
    observation, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    returns: list[float] = []
    model_losses: list[float] = []
    disagreements: list[float] = []

    for step in range(1, config.total_steps + 1):
        action = (
            env.action_space.sample()
            if step <= config.warmup_steps
            else agent.act(observation)
        )
        next_observation, reward, terminated, truncated, _ = env.step(action)
        real_buffer.add(observation, action, reward, next_observation, terminated)
        observation = next_observation
        episode_return += reward
        if terminated or truncated:
            returns.append(episode_return)
            episode_return = 0.0
            observation, _ = env.reset()

        if step >= config.warmup_steps and step % config.model_train_interval == 0:
            model_losses.extend(
                ensemble.fit(
                    real_buffer,
                    epochs=config.model_epochs,
                    batch_size=config.batch_size,
                    rng=rng,
                )
            )
            start_indices = rng.integers(
                len(real_buffer), size=min(config.model_rollout_batch, len(real_buffer))
            )
            simulated_states = real_buffer.observations[start_indices].copy()
            for _ in range(config.model_rollout_horizon):
                with torch.no_grad():
                    state_tensor = torch.as_tensor(simulated_states, device=device)
                    actions, _, _ = agent.actor.sample(state_tensor)
                    simulated_actions = actions.cpu().numpy()
                simulated_next, uncertainty = ensemble.predict(
                    simulated_states, simulated_actions, rng
                )
                simulated_next[:, :2] = np.clip(simulated_next[:, :2], -1.0, 1.0)
                rewards = point_mass_reward(simulated_next, env.goal)
                for index in range(len(simulated_states)):
                    model_buffer.add(
                        simulated_states[index],
                        simulated_actions[index],
                        float(rewards[index]),
                        simulated_next[index],
                        False,
                    )
                disagreements.append(float(np.mean(uncertainty)))
                simulated_states = simulated_next

        if step >= config.warmup_steps and len(real_buffer) >= config.batch_size:
            agent.update(
                _mixed_batch(real_buffer, model_buffer, config.batch_size, device, rng)
            )
    env.close()
    return MBPOResult(agent, returns, model_losses, disagreements)

