"""GAIL, DIAYN skill discovery, and a compact state-space Skew-Fit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from edu_rl.advanced.sac import SACAgent
from edu_rl.core.buffers import ReplayBatch, ReplayBuffer
from edu_rl.core.networks import CategoricalActor, ValueNetwork, mlp
from edu_rl.core.on_policy import collect_rollout, policy_log_prob_and_entropy
from edu_rl.core.returns import generalized_advantage_estimate, normalize
from edu_rl.core.utils import make_env, observation_size, set_global_seeds
from edu_rl.algorithms.ppo import ppo_policy_loss
from edu_rl.envs import GoalPointMass2D, PointMass2D


class GAILDiscriminator(nn.Module):
    """Classify expert (1) versus learner (0) state-action pairs."""

    def __init__(self, observation_size: int, action_count: int) -> None:
        super().__init__()
        self.action_count = action_count
        self.network = mlp(observation_size + action_count, (64, 64), 1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        one_hot_actions = nn.functional.one_hot(
            actions.long(), self.action_count
        ).float()
        return self.network(torch.cat([observations, one_hot_actions], dim=1)).squeeze(1)


def cartpole_expert_dataset(
    *, episodes: int = 50, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate demonstrations from a transparent balance heuristic."""

    env = gym.make("CartPole-v1")
    observations: list[np.ndarray] = []
    actions: list[int] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        done = False
        while not done:
            # Pole angle and angular velocity dominate this simple controller.
            action = int(observation[2] + 0.25 * observation[3] > 0.0)
            observations.append(np.asarray(observation, dtype=np.float32))
            actions.append(action)
            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()
    return np.asarray(observations), np.asarray(actions, dtype=np.int64)


@dataclass
class GAILConfig:
    iterations: int = 100
    rollout_steps: int = 1_024
    discriminator_epochs: int = 5
    policy_epochs: int = 5
    expert_episodes: int = 50
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    seed: int = 0
    device: str = "cpu"


@dataclass
class GAILResult:
    actor: CategoricalActor
    discriminator: GAILDiscriminator
    environment_returns: list[float]
    discriminator_losses: list[float]


def train_gail(config: GAILConfig) -> GAILResult:
    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    env = make_env("CartPole-v1", config.seed)
    obs_size = observation_size(env)
    actor = CategoricalActor(obs_size, 2, (64, 64)).to(device)
    critic = ValueNetwork(obs_size, (64, 64)).to(device)
    discriminator = GAILDiscriminator(obs_size, 2).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=3e-4)
    expert_observations, expert_actions = cartpole_expert_dataset(
        episodes=config.expert_episodes, seed=config.seed
    )
    environment_returns: list[float] = []
    discriminator_losses: list[float] = []

    for iteration in range(config.iterations):
        rollout = collect_rollout(
            env,
            actor,
            critic,
            rollout_steps=config.rollout_steps,
            device=device,
            seed=config.seed + iteration,
            continuous=False,
        )
        learner_observations = torch.as_tensor(rollout.observations, device=device)
        learner_actions = torch.as_tensor(rollout.policy_actions, device=device)
        for _ in range(config.discriminator_epochs):
            expert_indices = rng.integers(len(expert_observations), size=config.rollout_steps)
            expert_obs = torch.as_tensor(
                expert_observations[expert_indices], device=device
            )
            expert_act = torch.as_tensor(expert_actions[expert_indices], device=device)
            expert_logits = discriminator(expert_obs, expert_act)
            learner_logits = discriminator(learner_observations, learner_actions)
            loss = nn.functional.binary_cross_entropy_with_logits(
                expert_logits, torch.ones_like(expert_logits)
            ) + nn.functional.binary_cross_entropy_with_logits(
                learner_logits, torch.zeros_like(learner_logits)
            )
            discriminator_optimizer.zero_grad()
            loss.backward()
            discriminator_optimizer.step()
        discriminator_losses.append(float(loss.detach()))

        with torch.no_grad():
            # -log(1-D) is softplus(logit); it is dense and non-saturating.
            imitation_rewards = nn.functional.softplus(
                discriminator(learner_observations, learner_actions)
            ).cpu().numpy()
        advantages, targets = generalized_advantage_estimate(
            imitation_rewards,
            rollout.values,
            rollout.next_values,
            rollout.terminated,
            rollout.episode_ends,
            config.gamma,
            config.gae_lambda,
        )
        advantages_tensor = torch.as_tensor(normalize(advantages), device=device)
        targets_tensor = torch.as_tensor(targets, device=device)
        old_log_probabilities = torch.as_tensor(
            rollout.old_log_probabilities, device=device
        )
        for _ in range(config.policy_epochs):
            new_log_probabilities, entropy = policy_log_prob_and_entropy(
                actor,
                learner_observations,
                learner_actions,
                continuous=False,
            )
            policy_loss, _, _ = ppo_policy_loss(
                new_log_probabilities,
                old_log_probabilities,
                advantages_tensor,
                method="clip",
                clip_ratio=config.clip_ratio,
                kl_beta=0.0,
            )
            policy_loss -= 0.01 * entropy.mean()
            actor_optimizer.zero_grad()
            policy_loss.backward()
            actor_optimizer.step()
            value_loss = nn.functional.mse_loss(critic(learner_observations), targets_tensor)
            critic_optimizer.zero_grad()
            value_loss.backward()
            critic_optimizer.step()
        completed = rollout.completed_episode_returns
        environment_returns.append(float(np.mean(completed)) if completed else float("nan"))
    env.close()
    return GAILResult(actor, discriminator, environment_returns, discriminator_losses)


class SkillDiscriminator(nn.Module):
    def __init__(self, state_size: int, skill_count: int) -> None:
        super().__init__()
        self.network = mlp(state_size, (64, 64), skill_count)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


def diayn_reward(logits: torch.Tensor, skills: torch.Tensor, skill_count: int) -> torch.Tensor:
    """Mutual-information reward ``log q(z|s) - log p(z)`` for uniform skills."""

    return nn.functional.log_softmax(logits, dim=1).gather(
        1, skills.long().unsqueeze(1)
    ).squeeze(1) + math.log(skill_count)


@dataclass
class DIAYNConfig:
    total_steps: int = 50_000
    warmup_steps: int = 500
    batch_size: int = 128
    skill_count: int = 6
    hidden_sizes: tuple[int, ...] = (128, 128)
    seed: int = 0
    device: str = "cpu"


@dataclass
class DIAYNResult:
    agent: SACAgent
    discriminator: SkillDiscriminator
    intrinsic_rewards: list[float]
    discriminator_losses: list[float]


def train_diayn(config: DIAYNConfig) -> DIAYNResult:
    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    # DIAYN has no task goal; episodes end by time limit rather than an unrelated
    # external success condition.
    env = PointMass2D(max_steps=80, terminate_on_goal=False)
    augmented_size = 4 + config.skill_count
    agent = SACAgent(
        augmented_size,
        env.action_space.low,
        env.action_space.high,
        hidden_sizes=config.hidden_sizes,
        device=device,
    )
    discriminator = SkillDiscriminator(4, config.skill_count).to(device)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=3e-4)
    replay = ReplayBuffer(100_000, (augmented_size,), (2,), discrete_actions=False)
    state, _ = env.reset(seed=config.seed)
    skill = int(rng.integers(config.skill_count))
    skill_vector = np.eye(config.skill_count, dtype=np.float32)[skill]
    observation = np.concatenate([state, skill_vector])
    intrinsic_rewards: list[float] = []
    discriminator_losses: list[float] = []

    for step in range(1, config.total_steps + 1):
        action = env.action_space.sample() if step <= config.warmup_steps else agent.act(observation)
        next_state, _, terminated, truncated, _ = env.step(action)
        next_observation = np.concatenate([next_state, skill_vector])
        replay.add(observation, action, 0.0, next_observation, terminated)
        observation = next_observation
        if terminated or truncated:
            state, _ = env.reset()
            skill = int(rng.integers(config.skill_count))
            skill_vector = np.eye(config.skill_count, dtype=np.float32)[skill]
            observation = np.concatenate([state, skill_vector])

        if step >= config.warmup_steps and len(replay) >= config.batch_size:
            batch = replay.sample(config.batch_size, device, rng)
            skills = batch.next_observations[:, 4:].argmax(dim=1)
            logits = discriminator(batch.next_observations[:, :4])
            discriminator_loss = nn.functional.cross_entropy(logits, skills)
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()
            with torch.no_grad():
                rewards = diayn_reward(
                    discriminator(batch.next_observations[:, :4]),
                    skills,
                    config.skill_count,
                )
            intrinsic_batch = ReplayBatch(
                batch.observations,
                batch.actions,
                rewards,
                batch.next_observations,
                batch.terminated,
            )
            agent.update(intrinsic_batch)
            intrinsic_rewards.append(float(rewards.mean()))
            discriminator_losses.append(float(discriminator_loss.detach()))
    env.close()
    return DIAYNResult(agent, discriminator, intrinsic_rewards, discriminator_losses)


class GoalVAE(nn.Module):
    """Two-dimensional VAE used as Skew-Fit's adaptive goal generator."""

    def __init__(self, latent_size: int = 2) -> None:
        super().__init__()
        self.encoder = mlp(2, (32, 32), 2 * latent_size)
        self.decoder = mlp(latent_size, (32, 32), 2)
        self.latent_size = latent_size

    def loss_per_item(self, goals: torch.Tensor) -> torch.Tensor:
        mean, log_variance = self.encoder(goals).chunk(2, dim=1)
        standard_deviation = torch.exp(0.5 * log_variance.clamp(-8.0, 4.0))
        latent = mean + standard_deviation * torch.randn_like(mean)
        reconstruction = torch.tanh(self.decoder(latent))
        reconstruction_loss = (reconstruction - goals).pow(2).sum(dim=1)
        kl = -0.5 * (1.0 + log_variance - mean.pow(2) - log_variance.exp()).sum(dim=1)
        return reconstruction_loss + 0.1 * kl

    @torch.no_grad()
    def sample(self, count: int, device: torch.device) -> torch.Tensor:
        return torch.tanh(self.decoder(torch.randn(count, self.latent_size, device=device)))


@dataclass
class SkewFitConfig:
    total_steps: int = 50_000
    warmup_steps: int = 500
    batch_size: int = 128
    goal_update_interval: int = 500
    vae_epochs: int = 20
    skew_alpha: float = -1.0
    hidden_sizes: tuple[int, ...] = (128, 128)
    seed: int = 0
    device: str = "cpu"


@dataclass
class SkewFitResult:
    agent: SACAgent
    vae: GoalVAE
    episode_returns: list[float]
    coverage_history: list[float]
    vae_losses: list[float]


def train_skew_fit(config: SkewFitConfig) -> SkewFitResult:
    rng = set_global_seeds(config.seed)
    device = torch.device(config.device)
    env = GoalPointMass2D(max_steps=60)
    agent = SACAgent(
        6,
        env.action_space.low,
        env.action_space.high,
        hidden_sizes=config.hidden_sizes,
        device=device,
    )
    vae = GoalVAE().to(device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    replay = ReplayBuffer(100_000, (6,), (2,), discrete_actions=False)
    achieved_goals: list[np.ndarray] = []
    vae_losses: list[float] = []
    coverage_history: list[float] = []
    returns: list[float] = []

    def sample_goal() -> np.ndarray:
        if len(achieved_goals) < 64:
            return rng.uniform(-0.9, 0.9, size=2).astype(np.float32)
        return vae.sample(1, device).squeeze(0).cpu().numpy().astype(np.float32)

    observation, _ = env.reset(seed=config.seed, options={"goal": sample_goal()})
    episode_return = 0.0
    for step in range(1, config.total_steps + 1):
        action = env.action_space.sample() if step <= config.warmup_steps else agent.act(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        replay.add(observation, action, reward, next_observation, terminated)
        achieved_goals.append(next_observation[:2].copy())
        if len(achieved_goals) > 20_000:
            achieved_goals.pop(0)
        observation = next_observation
        episode_return += reward
        if terminated or truncated:
            returns.append(episode_return)
            episode_return = 0.0
            observation, _ = env.reset(options={"goal": sample_goal()})

        if step >= config.warmup_steps and len(replay) >= config.batch_size:
            agent.update(replay.sample(config.batch_size, device, rng))

        if len(achieved_goals) >= 64 and step % config.goal_update_interval == 0:
            goals = torch.as_tensor(np.asarray(achieved_goals), device=device)
            for _ in range(config.vae_epochs):
                indices = rng.integers(len(goals), size=min(config.batch_size, len(goals)))
                batch = goals[indices]
                with torch.no_grad():
                    density_proxy = -vae.loss_per_item(batch)
                    weights = torch.softmax(config.skew_alpha * density_proxy, dim=0)
                    weights = weights * len(weights)
                losses = vae.loss_per_item(batch)
                vae_loss = (weights * losses).mean()
                vae_optimizer.zero_grad()
                vae_loss.backward()
                vae_optimizer.step()
                vae_losses.append(float(vae_loss.detach()))
            positions = np.asarray(achieved_goals)
            bins = np.clip(((positions + 1.0) * 5).astype(int), 0, 9)
            coverage_history.append(float(len({tuple(item) for item in bins}) / 100.0))
    env.close()
    return SkewFitResult(agent, vae, returns, coverage_history, vae_losses)
