"""Natural policy gradient and a compact TRPO-style line-search update."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch
from torch import nn

from edu_rl.core.networks import ValueNetwork
from edu_rl.core.on_policy import (
    Actor,
    collect_rollout,
    make_actor,
    mean_policy_kl,
    policy_log_prob_and_entropy,
)
from edu_rl.core.returns import generalized_advantage_estimate, normalize
from edu_rl.core.utils import make_env, observation_size, set_global_seeds


NaturalMethod = Literal["npg", "trpo"]


@dataclass
class NaturalPolicyConfig:
    env_id: str = "CartPole-v1"
    iterations: int = 80
    rollout_steps: int = 2_048
    method: NaturalMethod = "trpo"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_kl: float = 0.01
    conjugate_gradient_steps: int = 10
    damping: float = 0.1
    line_search_steps: int = 10
    line_search_decay: float = 0.5
    critic_epochs: int = 10
    critic_learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (64, 64)
    seed: int = 0
    device: str = "cpu"


@dataclass
class NaturalPolicyResult:
    actor: Actor
    critic: ValueNetwork
    mean_returns: list[float]
    policy_kls: list[float]
    accepted_step_fractions: list[float]
    value_losses: list[float]


def flat_parameters(module: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


@torch.no_grad()
def set_flat_parameters(module: nn.Module, flat_values: torch.Tensor) -> None:
    offset = 0
    for parameter in module.parameters():
        count = parameter.numel()
        parameter.copy_(flat_values[offset : offset + count].view_as(parameter))
        offset += count
    if offset != flat_values.numel():
        raise ValueError("Flat parameter vector has the wrong length")


def flat_gradient(
    output: torch.Tensor,
    module: nn.Module,
    *,
    create_graph: bool = False,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        output, tuple(module.parameters()), create_graph=create_graph
    )
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def conjugate_gradient(
    matrix_vector_product: Callable[[torch.Tensor], torch.Tensor],
    vector: torch.Tensor,
    *,
    iterations: int = 10,
    tolerance: float = 1e-10,
) -> torch.Tensor:
    """Approximately solve ``A x = vector`` without constructing matrix A."""

    solution = torch.zeros_like(vector)
    residual = vector.clone()
    direction = residual.clone()
    residual_dot = torch.dot(residual, residual)
    for _ in range(iterations):
        matrix_direction = matrix_vector_product(direction)
        alpha = residual_dot / (torch.dot(direction, matrix_direction) + 1e-8)
        solution = solution + alpha * direction
        residual = residual - alpha * matrix_direction
        new_residual_dot = torch.dot(residual, residual)
        if new_residual_dot < tolerance:
            break
        beta = new_residual_dot / (residual_dot + 1e-8)
        direction = residual + beta * direction
        residual_dot = new_residual_dot
    return solution


def train_natural_policy(config: NaturalPolicyConfig) -> NaturalPolicyResult:
    set_global_seeds(config.seed)
    device = torch.device(config.device)
    env = make_env(config.env_id, config.seed)
    obs_size = observation_size(env)
    actor, continuous = make_actor(env, obs_size, config.hidden_sizes)
    actor = actor.to(device)
    critic = ValueNetwork(obs_size, config.hidden_sizes).to(device)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )

    mean_returns: list[float] = []
    policy_kls: list[float] = []
    accepted_fractions: list[float] = []
    value_losses: list[float] = []

    try:
        for iteration in range(config.iterations):
            rollout = collect_rollout(
                env,
                actor,
                critic,
                rollout_steps=config.rollout_steps,
                device=device,
                seed=config.seed + iteration,
                continuous=continuous,
            )
            advantages, value_targets = generalized_advantage_estimate(
                rollout.rewards,
                rollout.values,
                rollout.next_values,
                rollout.terminated,
                rollout.episode_ends,
                config.gamma,
                config.gae_lambda,
            )
            observations = torch.as_tensor(rollout.observations, device=device)
            action_dtype = torch.float32 if continuous else torch.long
            actions = torch.as_tensor(
                rollout.policy_actions, dtype=action_dtype, device=device
            )
            old_log_probabilities = torch.as_tensor(
                rollout.old_log_probabilities, device=device
            )
            advantages_tensor = torch.as_tensor(normalize(advantages), device=device)
            targets_tensor = torch.as_tensor(value_targets, device=device)

            old_actor = copy.deepcopy(actor).to(device)
            old_actor.requires_grad_(False)

            def surrogate_objective() -> torch.Tensor:
                new_log_probability, _ = policy_log_prob_and_entropy(
                    actor, observations, actions, continuous=continuous
                )
                ratio = (new_log_probability - old_log_probabilities).exp()
                return (ratio * advantages_tensor).mean()

            gradient = flat_gradient(surrogate_objective(), actor).detach()

            def fisher_vector_product(vector: torch.Tensor) -> torch.Tensor:
                kl = mean_policy_kl(
                    old_actor, actor, observations, continuous=continuous
                )
                first_gradient = flat_gradient(kl, actor, create_graph=True)
                directional_derivative = torch.dot(first_gradient, vector)
                second_gradient = flat_gradient(
                    directional_derivative, actor
                ).detach()
                return second_gradient + config.damping * vector

            natural_direction = conjugate_gradient(
                fisher_vector_product,
                gradient,
                iterations=config.conjugate_gradient_steps,
            )
            fisher_direction = fisher_vector_product(natural_direction)
            quadratic_term = 0.5 * torch.dot(
                natural_direction, fisher_direction
            )
            scale = torch.sqrt(
                torch.as_tensor(config.max_kl, device=device)
                / (quadratic_term + 1e-8)
            )
            full_step = natural_direction * scale
            old_parameters = flat_parameters(actor)
            old_surrogate = float(surrogate_objective().detach())

            accepted_fraction = 1.0
            if config.method == "npg":
                set_flat_parameters(actor, old_parameters + full_step)
            elif config.method == "trpo":
                accepted_fraction = 0.0
                for line_search_index in range(config.line_search_steps):
                    fraction = config.line_search_decay**line_search_index
                    set_flat_parameters(
                        actor, old_parameters + fraction * full_step
                    )
                    new_surrogate = float(surrogate_objective().detach())
                    new_kl = float(
                        mean_policy_kl(
                            old_actor, actor, observations, continuous=continuous
                        ).detach()
                    )
                    if new_surrogate > old_surrogate and new_kl <= config.max_kl:
                        accepted_fraction = fraction
                        break
                if accepted_fraction == 0.0:
                    set_flat_parameters(actor, old_parameters)
            else:
                raise ValueError(f"Unknown natural policy method: {config.method}")

            actual_kl = float(
                mean_policy_kl(
                    old_actor, actor, observations, continuous=continuous
                ).detach()
            )

            last_value_loss = 0.0
            for _ in range(config.critic_epochs):
                value_loss = nn.functional.mse_loss(critic(observations), targets_tensor)
                critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optimizer.step()
                last_value_loss = float(value_loss.detach())

            completed = rollout.completed_episode_returns
            mean_returns.append(float(np.mean(completed)) if completed else float("nan"))
            policy_kls.append(actual_kl)
            accepted_fractions.append(accepted_fraction)
            value_losses.append(last_value_loss)
    finally:
        env.close()

    return NaturalPolicyResult(
        actor=actor,
        critic=critic,
        mean_returns=mean_returns,
        policy_kls=policy_kls,
        accepted_step_fractions=accepted_fractions,
        value_losses=value_losses,
    )

