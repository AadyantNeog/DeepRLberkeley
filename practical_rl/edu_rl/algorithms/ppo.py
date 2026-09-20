"""Clipped PPO and adaptive KL-penalty PPO with GAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from edu_rl.core.networks import ValueNetwork
from edu_rl.core.on_policy import (
    Actor,
    collect_rollout,
    make_actor,
    policy_log_prob_and_entropy,
)
from edu_rl.core.returns import generalized_advantage_estimate, normalize
from edu_rl.core.utils import make_env, observation_size, set_global_seeds


PPOMethod = Literal["clip", "kl"]


@dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    iterations: int = 100
    rollout_steps: int = 2_048
    update_epochs: int = 10
    minibatch_size: int = 64
    method: PPOMethod = "clip"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    initial_kl_beta: float = 1.0
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (64, 64)
    seed: int = 0
    device: str = "cpu"


@dataclass
class PPOResult:
    actor: Actor
    critic: ValueNetwork
    mean_returns: list[float]
    policy_losses: list[float]
    value_losses: list[float]
    approximate_kls: list[float]
    clip_fractions: list[float]
    kl_betas: list[float]


def ppo_policy_loss(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    *,
    method: PPOMethod,
    clip_ratio: float,
    kl_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return policy loss, approximate KL, and clipped-sample fraction."""

    log_ratio = new_log_probabilities - old_log_probabilities
    ratio = log_ratio.exp()
    # Under samples from the old policy, E[(ratio - 1) - log(ratio)] estimates
    # KL(old || new) and is non-negative per sample.  It is more stable than the
    # simpler finite-sample mean of ``old_log_prob - new_log_prob``.
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
    if method == "clip":
        unclipped = ratio * advantages
        clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        loss = -torch.minimum(unclipped, clipped).mean()
    elif method == "kl":
        loss = -(ratio * advantages).mean() + kl_beta * approximate_kl
    else:
        raise ValueError(f"Unknown PPO method: {method}")
    return loss, approximate_kl, clip_fraction


def train_ppo(config: PPOConfig) -> PPOResult:
    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    env = make_env(config.env_id, config.seed)
    obs_size = observation_size(env)
    actor, continuous = make_actor(env, obs_size, config.hidden_sizes)
    actor = actor.to(device)
    critic = ValueNetwork(obs_size, config.hidden_sizes).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )
    kl_beta = config.initial_kl_beta

    mean_returns: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    approximate_kls: list[float] = []
    clip_fractions: list[float] = []
    kl_betas: list[float] = []

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
            advantages = normalize(advantages)

            observations = torch.as_tensor(rollout.observations, device=device)
            action_dtype = torch.float32 if continuous else torch.long
            actions = torch.as_tensor(
                rollout.policy_actions, dtype=action_dtype, device=device
            )
            old_log_probabilities = torch.as_tensor(
                rollout.old_log_probabilities, device=device
            )
            advantages_tensor = torch.as_tensor(advantages, device=device)
            targets_tensor = torch.as_tensor(value_targets, device=device)

            iteration_policy_losses: list[float] = []
            iteration_value_losses: list[float] = []
            iteration_kls: list[float] = []
            iteration_clip_fractions: list[float] = []
            stop_early = False
            for _ in range(config.update_epochs):
                indices = rng.permutation(config.rollout_steps)
                for start in range(0, config.rollout_steps, config.minibatch_size):
                    batch_indices = torch.as_tensor(
                        indices[start : start + config.minibatch_size], device=device
                    )
                    new_log_probabilities, entropy = policy_log_prob_and_entropy(
                        actor,
                        observations[batch_indices],
                        actions[batch_indices],
                        continuous=continuous,
                    )
                    policy_loss, approximate_kl, clip_fraction = ppo_policy_loss(
                        new_log_probabilities,
                        old_log_probabilities[batch_indices],
                        advantages_tensor[batch_indices],
                        method=config.method,
                        clip_ratio=config.clip_ratio,
                        kl_beta=kl_beta,
                    )
                    policy_loss -= config.entropy_coefficient * entropy.mean()
                    actor_optimizer.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
                    actor_optimizer.step()

                    value_loss = nn.functional.mse_loss(
                        critic(observations[batch_indices]),
                        targets_tensor[batch_indices],
                    )
                    critic_optimizer.zero_grad()
                    (config.value_coefficient * value_loss).backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
                    critic_optimizer.step()

                    iteration_policy_losses.append(float(policy_loss.detach()))
                    iteration_value_losses.append(float(value_loss.detach()))
                    iteration_kls.append(float(approximate_kl.detach()))
                    iteration_clip_fractions.append(float(clip_fraction.detach()))

                # The early stop is a safeguard, not part of the clipped formula.
                if (
                    config.method == "clip"
                    and iteration_kls
                    and np.mean(iteration_kls[-max(1, config.rollout_steps // config.minibatch_size) :])
                    > 1.5 * config.target_kl
                ):
                    stop_early = True
                if stop_early:
                    break

            mean_kl = float(np.mean(iteration_kls))
            if config.method == "kl":
                if mean_kl > 1.5 * config.target_kl:
                    kl_beta *= 2.0
                elif mean_kl < config.target_kl / 1.5:
                    kl_beta *= 0.5
                kl_beta = float(np.clip(kl_beta, 1e-4, 1e4))

            completed = rollout.completed_episode_returns
            mean_returns.append(float(np.mean(completed)) if completed else float("nan"))
            policy_losses.append(float(np.mean(iteration_policy_losses)))
            value_losses.append(float(np.mean(iteration_value_losses)))
            approximate_kls.append(mean_kl)
            clip_fractions.append(float(np.mean(iteration_clip_fractions)))
            kl_betas.append(kl_beta)
    finally:
        env.close()

    return PPOResult(
        actor=actor,
        critic=critic,
        mean_returns=mean_returns,
        policy_losses=policy_losses,
        value_losses=value_losses,
        approximate_kls=approximate_kls,
        clip_fractions=clip_fractions,
        kl_betas=kl_betas,
    )
