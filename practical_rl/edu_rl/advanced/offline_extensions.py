"""BRAC, IDQL, Flow Q-Learning, and diffusion Q-steering.

The generative methods normalize actions to ``[-1, 1]`` internally.  Keeping
that transform explicit avoids a common silent error when using diffusion or
flow policies on environments such as Pendulum whose bounds are ``[-2, 2]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from edu_rl.advanced.generative_models import (
    ConditionalDiffusionPolicy,
    FlowResult,
    integrate_flow,
    train_flow_matching,
)
from edu_rl.advanced.offline_rl import OfflineConfig, OfflineDataset, OfflineResult, train_offline
from edu_rl.core.networks import ContinuousQNetwork, DeterministicActor, SquashedGaussianActor
from edu_rl.core.utils import hard_update, set_global_seeds, soft_update


def normalize_actions(
    actions: np.ndarray, action_low: np.ndarray, action_high: np.ndarray
) -> np.ndarray:
    scale = (np.asarray(action_high) - np.asarray(action_low)) / 2.0
    bias = (np.asarray(action_high) + np.asarray(action_low)) / 2.0
    return np.clip((actions - bias) / scale, -1.0, 1.0).astype(np.float32)


def denormalize_actions(
    actions: torch.Tensor, action_low: torch.Tensor, action_high: torch.Tensor
) -> torch.Tensor:
    return (action_high + action_low) / 2.0 + (action_high - action_low) / 2.0 * actions


@dataclass
class BRACConfig:
    gradient_steps: int = 50_000
    behavior_steps: int = 10_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    behavior_regularization: float = 1.0
    hidden_sizes: tuple[int, ...] = (256, 256)
    seed: int = 0
    device: str = "cpu"


@dataclass
class BRACResult:
    actor: SquashedGaussianActor
    behavior: SquashedGaussianActor
    q1: ContinuousQNetwork
    q2: ContinuousQNetwork
    actor_losses: list[float]
    critic_losses: list[float]
    divergences: list[float]


def train_brac(
    dataset: OfflineDataset,
    config: BRACConfig,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> BRACResult:
    """Behavior-Regularized Actor Critic with a learned behavior policy.

    The policy penalty is the sampled forward KL ``log pi(a|s)-log mu(a|s)``.
    This keeps policy actions near dataset support while the Q term improves on
    the behavior policy.
    """

    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    observation_size = dataset.observations.shape[1]
    action_size = dataset.actions.shape[1]
    actor = SquashedGaussianActor(
        observation_size, action_size, config.hidden_sizes, action_low, action_high
    ).to(device)
    behavior = SquashedGaussianActor(
        observation_size, action_size, config.hidden_sizes, action_low, action_high
    ).to(device)
    q1 = ContinuousQNetwork(observation_size, action_size, config.hidden_sizes).to(device)
    q2 = ContinuousQNetwork(observation_size, action_size, config.hidden_sizes).to(device)
    target_q1 = ContinuousQNetwork(observation_size, action_size, config.hidden_sizes).to(device)
    target_q2 = ContinuousQNetwork(observation_size, action_size, config.hidden_sizes).to(device)
    hard_update(target_q1, q1)
    hard_update(target_q2, q2)
    behavior_optimizer = torch.optim.Adam(behavior.parameters(), lr=config.learning_rate)
    for _ in range(config.behavior_steps):
        batch = dataset.sample(config.batch_size, device, rng)
        loss = -behavior.log_prob_of_action(batch.observations, batch.actions).mean()
        behavior_optimizer.zero_grad()
        loss.backward()
        behavior_optimizer.step()
    for parameter in behavior.parameters():
        parameter.requires_grad_(False)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.Adam([*q1.parameters(), *q2.parameters()], lr=config.learning_rate)
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    divergences: list[float] = []
    for _ in range(config.gradient_steps):
        batch = dataset.sample(config.batch_size, device, rng)
        with torch.no_grad():
            next_actions, _, _ = actor.sample(batch.next_observations)
            target_q = torch.minimum(
                target_q1(batch.next_observations, next_actions),
                target_q2(batch.next_observations, next_actions),
            )
            target = batch.rewards + config.gamma * (1.0 - batch.terminated) * target_q
        q_loss = nn.functional.mse_loss(q1(batch.observations, batch.actions), target)
        q_loss = q_loss + nn.functional.mse_loss(q2(batch.observations, batch.actions), target)
        critic_optimizer.zero_grad()
        q_loss.backward()
        critic_optimizer.step()

        actions, actor_log_probability, raw_actions = actor.sample(batch.observations)
        behavior_log_probability = behavior.log_prob_of_action(batch.observations, actions)
        divergence = (actor_log_probability - behavior_log_probability).mean()
        actor_loss = -torch.minimum(
            q1(batch.observations, actions), q2(batch.observations, actions)
        ).mean() + config.behavior_regularization * divergence
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        soft_update(target_q1, q1, config.tau)
        soft_update(target_q2, q2, config.tau)
        actor_losses.append(float(actor_loss.detach()))
        critic_losses.append(float(q_loss.detach()))
        divergences.append(float(divergence.detach()))
    return BRACResult(actor, behavior, q1, q2, actor_losses, critic_losses, divergences)


@dataclass
class GenerativeOfflineConfig:
    algorithm: str = "idql"
    critic_steps: int = 30_000
    generative_steps: int = 30_000
    actor_steps: int = 10_000
    batch_size: int = 256
    diffusion_steps: int = 20
    candidate_actions: int = 32
    hidden_sizes: tuple[int, ...] = (256, 256)
    seed: int = 0
    device: str = "cpu"


@dataclass
class GenerativeOfflineResult:
    critic: OfflineResult
    diffusion: ConditionalDiffusionPolicy | None
    flow: FlowResult | None
    one_step_actor: DeterministicActor | None
    generative_losses: list[float]
    actor_losses: list[float]
    action_low: torch.Tensor
    action_high: torch.Tensor


def _fit_diffusion(
    observations: np.ndarray,
    normalized_actions: np.ndarray,
    config: GenerativeOfflineConfig,
) -> tuple[ConditionalDiffusionPolicy, list[float]]:
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    observation_tensor = torch.as_tensor(observations, device=device)
    action_tensor = torch.as_tensor(normalized_actions, device=device)
    policy = ConditionalDiffusionPolicy(
        observations.shape[1], normalized_actions.shape[1],
        diffusion_steps=config.diffusion_steps, hidden_sizes=config.hidden_sizes,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    losses: list[float] = []
    for _ in range(config.generative_steps):
        indices = rng.integers(len(observations), size=config.batch_size)
        loss = policy.loss(observation_tensor[indices], action_tensor[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return policy, losses


def train_generative_offline(
    dataset: OfflineDataset,
    config: GenerativeOfflineConfig,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> GenerativeOfflineResult:
    """Train IDQL, FQL, or a diffusion policy intended for Q-steering."""

    if config.algorithm not in {"idql", "fql", "diffusion-steering"}:
        raise ValueError("algorithm must be idql, fql, or diffusion-steering")
    set_global_seeds(config.seed)
    device = torch.device(config.device)
    critic = train_offline(
        dataset,
        OfflineConfig(
            algorithm="iql", gradient_steps=config.critic_steps,
            batch_size=config.batch_size, hidden_sizes=config.hidden_sizes,
            seed=config.seed, device=config.device,
        ),
        action_low=action_low,
        action_high=action_high,
    )
    normalized = normalize_actions(dataset.actions, action_low, action_high)
    low_tensor = torch.as_tensor(action_low, dtype=torch.float32, device=device)
    high_tensor = torch.as_tensor(action_high, dtype=torch.float32, device=device)

    if config.algorithm in {"idql", "diffusion-steering"}:
        diffusion, losses = _fit_diffusion(dataset.observations, normalized, config)
        return GenerativeOfflineResult(
            critic, diffusion, None, None, losses, [], low_tensor, high_tensor
        )

    flow = train_flow_matching(
        normalized,
        conditions=dataset.observations,
        gradient_steps=config.generative_steps,
        batch_size=config.batch_size,
        hidden_sizes=config.hidden_sizes,
        seed=config.seed,
        device=config.device,
    )
    actor = DeterministicActor(
        dataset.observations.shape[1], dataset.actions.shape[1], config.hidden_sizes,
        np.full(dataset.actions.shape[1], -1.0, dtype=np.float32),
        np.full(dataset.actions.shape[1], 1.0, dtype=np.float32),
    ).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    rng = np.random.default_rng(config.seed + 1)
    observations = torch.as_tensor(dataset.observations, device=device)
    actor_losses: list[float] = []
    for _ in range(config.actor_steps):
        indices = rng.integers(len(dataset), size=config.batch_size)
        batch_observations = observations[indices]
        with torch.no_grad():
            base = torch.randn(config.batch_size, dataset.actions.shape[1], device=device)
            flow_actions = integrate_flow(flow.field, base, batch_observations, steps=10).clamp(-1, 1)
        normalized_actor_actions = actor(batch_observations)
        environment_actions = denormalize_actions(
            normalized_actor_actions, low_tensor, high_tensor
        )
        q_value = torch.minimum(
            critic.q1(batch_observations, environment_actions),
            critic.q2(batch_observations, environment_actions),
        )
        # FQL distills a multi-step flow policy into a fast one-step actor and
        # simultaneously improves that actor using the offline critic.
        loss = nn.functional.mse_loss(normalized_actor_actions, flow_actions) - 0.1 * q_value.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        actor_losses.append(float(loss.detach()))
    return GenerativeOfflineResult(
        critic, None, flow, actor, flow.losses, actor_losses, low_tensor, high_tensor
    )


def sample_idql_actions(
    result: GenerativeOfflineResult,
    observations: torch.Tensor,
    *,
    candidate_count: int = 32,
    inverse_temperature: float = 10.0,
) -> torch.Tensor:
    """IDQL extraction: sample behavior actions, then advantage-weight/select."""

    if result.diffusion is None:
        raise ValueError("IDQL action sampling requires a diffusion policy")
    repeated = observations[:, None, :].expand(-1, candidate_count, -1).reshape(
        -1, observations.shape[-1]
    )
    with torch.no_grad():
        normalized = result.diffusion.sample(repeated)
        actions = denormalize_actions(normalized, result.action_low, result.action_high)
        q_values = torch.minimum(
            result.critic.q1(repeated, actions), result.critic.q2(repeated, actions)
        ).view(len(observations), candidate_count)
        probabilities = torch.softmax(
            inverse_temperature * (q_values - q_values.mean(dim=1, keepdim=True)), dim=1
        )
        choices = torch.multinomial(probabilities, 1).squeeze(1)
        reshaped = actions.view(len(observations), candidate_count, -1)
        return reshaped[torch.arange(len(observations), device=observations.device), choices]


def sample_diffusion_steered_actions(
    result: GenerativeOfflineResult,
    observations: torch.Tensor,
    *,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Guide every denoising step with gradients from the learned Q-function."""

    if result.diffusion is None:
        raise ValueError("diffusion steering requires a diffusion policy")

    def normalized_q(states: torch.Tensor, normalized_actions: torch.Tensor) -> torch.Tensor:
        actions = denormalize_actions(normalized_actions, result.action_low, result.action_high)
        return torch.minimum(result.critic.q1(states, actions), result.critic.q2(states, actions))

    normalized = result.diffusion.sample(
        observations, q_function=normalized_q, guidance_scale=guidance_scale
    )
    return denormalize_actions(normalized, result.action_low, result.action_high)
