"""Variational inference and generative action models used in modern RL.

The implementations are intentionally compact, but each exposes the mathematical
object discussed in the lectures: coordinate updates, the EM lower bound,
reparameterized ELBOs, conditional flow matching, diffusion denoising, and the
chain rule used by autoregressive policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from edu_rl.core.networks import mlp
from edu_rl.core.utils import set_global_seeds


@dataclass
class CAVIResult:
    mean: np.ndarray
    variance: np.ndarray
    errors: list[float]


def coordinate_ascent_gaussian_vi(
    precision: np.ndarray,
    natural_parameter: np.ndarray,
    *,
    iterations: int = 50,
) -> CAVIResult:
    """Mean-field CAVI for a multivariate Gaussian in information form.

    For ``p(x) proportional exp(-1/2 x^T Lambda x + eta^T x)``, the optimal
    factor ``q_i`` has variance ``1 / Lambda_ii`` and a mean depending on the
    current means of all other factors.  The means converge to the exact
    Gaussian mean, while mean-field necessarily drops posterior correlations.
    """

    precision = np.asarray(precision, dtype=np.float64)
    eta = np.asarray(natural_parameter, dtype=np.float64)
    if precision.shape != (len(eta), len(eta)):
        raise ValueError("precision must be square and match natural_parameter")
    if not np.allclose(precision, precision.T) or np.min(np.linalg.eigvalsh(precision)) <= 0:
        raise ValueError("precision must be symmetric positive definite")
    mean = np.zeros_like(eta)
    target_mean = np.linalg.solve(precision, eta)
    errors: list[float] = []
    for _ in range(iterations):
        for index in range(len(mean)):
            interaction = precision[index] @ mean - precision[index, index] * mean[index]
            mean[index] = (eta[index] - interaction) / precision[index, index]
        errors.append(float(np.linalg.norm(mean - target_mean)))
    return CAVIResult(mean, 1.0 / np.diag(precision), errors)


@dataclass
class GMMResult:
    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    responsibilities: np.ndarray
    log_likelihoods: list[float]


def _logsumexp(array: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(array, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.exp(array - maximum).sum(axis=axis)
    )


def gaussian_mixture_em(
    data: np.ndarray,
    components: int,
    *,
    iterations: int = 50,
    minimum_variance: float = 1e-4,
    seed: int = 0,
) -> GMMResult:
    """Expectation-maximization for a diagonal-covariance Gaussian mixture."""

    values = np.asarray(data, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if components <= 0 or components > len(values):
        raise ValueError("components must be between 1 and the number of examples")
    rng = np.random.default_rng(seed)
    means = values[rng.choice(len(values), components, replace=False)].copy()
    variances = np.repeat(values.var(axis=0, keepdims=True) + minimum_variance, components, axis=0)
    weights = np.full(components, 1.0 / components)
    log_likelihoods: list[float] = []
    responsibilities = np.empty((len(values), components))

    for _ in range(iterations):
        # E-step: posterior probability that each component generated each point.
        differences = values[:, None, :] - means[None, :, :]
        log_density = -0.5 * (
            np.log(2.0 * np.pi * variances)[None, :, :]
            + differences**2 / variances[None, :, :]
        ).sum(axis=-1)
        log_joint = log_density + np.log(weights + 1e-12)
        log_normalizer = _logsumexp(log_joint, axis=1)
        responsibilities = np.exp(log_joint - log_normalizer[:, None])
        log_likelihoods.append(float(log_normalizer.sum()))

        # M-step: weighted maximum-likelihood parameters.
        effective_count = responsibilities.sum(axis=0) + 1e-8
        weights = effective_count / effective_count.sum()
        means = responsibilities.T @ values / effective_count[:, None]
        differences = values[:, None, :] - means[None, :, :]
        variances = (
            responsibilities[:, :, None] * differences**2
        ).sum(axis=0) / effective_count[:, None]
        variances = np.maximum(variances, minimum_variance)
    return GMMResult(weights, means, variances, responsibilities, log_likelihoods)


def reparameterize(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Differentiable Gaussian sample ``mu + sigma * epsilon``."""

    return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)


def gaussian_vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    *,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Negative ELBO split into reconstruction and analytic KL terms."""

    reconstruction_loss = (reconstruction - target).pow(2).mean()
    kl = -0.5 * (1.0 + log_variance - mean.pow(2) - log_variance.exp()).sum(dim=-1).mean()
    return reconstruction_loss + beta * kl, reconstruction_loss, kl


class VAE(nn.Module):
    def __init__(self, input_size: int, latent_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.encoder = mlp(input_size, (hidden_size,), 2 * latent_size)
        self.decoder = mlp(latent_size, (hidden_size,), input_size)

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(inputs).chunk(2, dim=-1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(inputs)
        latent = reparameterize(mean, log_variance.clamp(-10.0, 5.0))
        return self.decoder(latent), mean, log_variance


class ConditionalVAE(nn.Module):
    """CVAE for action generation conditioned on an observation."""

    def __init__(
        self, observation_size: int, action_size: int, latent_size: int, hidden_size: int = 64
    ) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.encoder = mlp(observation_size + action_size, (hidden_size,), 2 * latent_size)
        self.decoder = mlp(observation_size + latent_size, (hidden_size,), action_size)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encoder(torch.cat([observations, actions], dim=-1)).chunk(2, -1)
        latent = reparameterize(mean, log_variance.clamp(-10.0, 5.0))
        reconstruction = self.decoder(torch.cat([observations, latent], dim=-1))
        return reconstruction, mean, log_variance

    @torch.no_grad()
    def sample(self, observations: torch.Tensor) -> torch.Tensor:
        latent = torch.randn(len(observations), self.latent_size, device=observations.device)
        return self.decoder(torch.cat([observations, latent], dim=-1))


class SequentialVAE(nn.Module):
    """A small sequential VAE with one stochastic latent at every time step.

    The encoder sees the whole input prefix through a GRU.  The decoder GRU
    turns the latent sequence back into observations.  This is the essential
    sequential-ELBO mechanism without the engineering of a full world model.
    """

    def __init__(self, input_size: int, latent_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.posterior = nn.Linear(hidden_size, 2 * latent_size)
        self.decoder = nn.GRU(latent_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, _ = self.encoder(sequence)
        mean, log_variance = self.posterior(encoded).chunk(2, dim=-1)
        latent = reparameterize(mean, log_variance.clamp(-10.0, 5.0))
        decoded, _ = self.decoder(latent)
        return self.output(decoded), mean, log_variance


class ConditionalVectorField(nn.Module):
    """Velocity field ``v_theta(x_t, t, condition)`` for rectified flow."""

    def __init__(
        self, data_size: int, condition_size: int = 0, hidden_sizes: tuple[int, ...] = (64, 64)
    ) -> None:
        super().__init__()
        self.condition_size = condition_size
        self.network = mlp(data_size + condition_size + 1, hidden_sizes, data_size)

    def forward(
        self, points: torch.Tensor, times: torch.Tensor, conditions: torch.Tensor | None = None
    ) -> torch.Tensor:
        if times.ndim == 1:
            times = times[:, None]
        pieces = [points, times]
        if self.condition_size:
            if conditions is None:
                raise ValueError("conditions are required for a conditional field")
            pieces.append(conditions)
        return self.network(torch.cat(pieces, dim=-1))


def flow_matching_loss(
    field: ConditionalVectorField,
    source: torch.Tensor,
    target: torch.Tensor,
    times: torch.Tensor,
    conditions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Conditional flow-matching regression on straight probability paths."""

    interpolation = (1.0 - times[:, None]) * source + times[:, None] * target
    desired_velocity = target - source
    return (field(interpolation, times, conditions) - desired_velocity).pow(2).mean()


def integrate_flow(
    field: ConditionalVectorField,
    source: torch.Tensor,
    conditions: torch.Tensor | None = None,
    *,
    steps: int = 20,
) -> torch.Tensor:
    """Euler integration of the learned ODE from base noise to a sample."""

    points = source
    step_size = 1.0 / steps
    for index in range(steps):
        times = torch.full((len(points),), index / steps, device=points.device)
        points = points + step_size * field(points, times, conditions)
    return points


@dataclass
class FlowResult:
    field: ConditionalVectorField
    losses: list[float]


def train_flow_matching(
    targets: np.ndarray,
    *,
    conditions: np.ndarray | None = None,
    gradient_steps: int = 1_000,
    batch_size: int = 128,
    hidden_sizes: tuple[int, ...] = (64, 64),
    seed: int = 0,
    device: str = "cpu",
) -> FlowResult:
    """Train a rectified-flow model from standard Gaussian noise to data."""

    rng = set_global_seeds(seed)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)
    condition_tensor = (
        None if conditions is None else torch.as_tensor(conditions, dtype=torch.float32, device=device)
    )
    field = ConditionalVectorField(
        target_tensor.shape[1], 0 if condition_tensor is None else condition_tensor.shape[1], hidden_sizes
    ).to(device)
    optimizer = torch.optim.Adam(field.parameters(), lr=3e-4)
    losses: list[float] = []
    for _ in range(gradient_steps):
        indices = rng.integers(len(target_tensor), size=batch_size)
        target = target_tensor[indices]
        condition = None if condition_tensor is None else condition_tensor[indices]
        source = torch.randn_like(target)
        times = torch.rand(batch_size, device=device)
        loss = flow_matching_loss(field, source, target, times, condition)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return FlowResult(field, losses)


def train_reflow(
    targets: np.ndarray,
    *,
    first_stage_steps: int = 1_000,
    reflow_steps: int = 1_000,
    batch_size: int = 128,
    hidden_sizes: tuple[int, ...] = (64, 64),
    seed: int = 0,
    device: str = "cpu",
) -> tuple[FlowResult, FlowResult]:
    """Train rectified flow, then straighten its own transport with Reflow."""

    first = train_flow_matching(
        targets,
        gradient_steps=first_stage_steps,
        batch_size=batch_size,
        hidden_sizes=hidden_sizes,
        seed=seed,
        device=device,
    )
    data_size = np.asarray(targets).shape[1]
    source = torch.randn(len(targets), data_size, device=device)
    with torch.no_grad():
        transported = integrate_flow(first.field, source, steps=20)
    second_field = ConditionalVectorField(data_size, hidden_sizes=hidden_sizes).to(device)
    optimizer = torch.optim.Adam(second_field.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed + 1)
    losses: list[float] = []
    for _ in range(reflow_steps):
        indices = rng.integers(len(source), size=batch_size)
        times = torch.rand(batch_size, device=device)
        loss = flow_matching_loss(second_field, source[indices], transported[indices], times)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return first, FlowResult(second_field, losses)


class ConditionalDiffusionPolicy(nn.Module):
    """A DDPM action policy conditioned on observations.

    This is usable both as a behavior model (IDQL) and as a policy whose reverse
    process can be steered by Q-value gradients.
    """

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        *,
        diffusion_steps: int = 20,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.action_size = action_size
        self.diffusion_steps = diffusion_steps
        self.noise_network = mlp(observation_size + action_size + 1, hidden_sizes, action_size)
        betas = torch.linspace(1e-4, 2e-2, diffusion_steps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def predict_noise(
        self, observations: torch.Tensor, noisy_actions: torch.Tensor, time_indices: torch.Tensor
    ) -> torch.Tensor:
        normalized_time = time_indices.float()[:, None] / max(1, self.diffusion_steps - 1)
        return self.noise_network(torch.cat([observations, noisy_actions, normalized_time], dim=-1))

    def loss(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        time_indices = torch.randint(self.diffusion_steps, (len(actions),), device=actions.device)
        noise = torch.randn_like(actions)
        alpha_bar = self.alpha_bars[time_indices, None]
        noisy = alpha_bar.sqrt() * actions + (1.0 - alpha_bar).sqrt() * noise
        return (self.predict_noise(observations, noisy, time_indices) - noise).pow(2).mean()

    def sample(
        self,
        observations: torch.Tensor,
        *,
        q_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        actions = torch.randn(len(observations), self.action_size, device=observations.device)
        for time_index in reversed(range(self.diffusion_steps)):
            indices = torch.full((len(actions),), time_index, device=actions.device, dtype=torch.long)
            with torch.no_grad():
                predicted_noise = self.predict_noise(observations, actions, indices)
                alpha = self.alphas[time_index]
                alpha_bar = self.alpha_bars[time_index]
                mean = (actions - (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
            if q_function is not None and guidance_scale:
                guided = mean.detach().requires_grad_(True)
                q_value = q_function(observations, guided).sum()
                gradient = torch.autograd.grad(q_value, guided)[0]
                mean = mean + guidance_scale * self.betas[time_index] * gradient
            if time_index:
                actions = mean + self.betas[time_index].sqrt() * torch.randn_like(actions)
            else:
                actions = mean
        return actions.clamp(-1.0, 1.0)


class AutoregressiveContinuousPolicy(nn.Module):
    """Factorize a vector action as ``prod_i pi(a_i | s, a_<i)``."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.action_size = action_size
        self.conditionals = nn.ModuleList(
            mlp(observation_size + action_size, hidden_sizes, 2) for _ in range(action_size)
        )

    def _distribution(
        self, observations: torch.Tensor, partial_actions: torch.Tensor, dimension: int
    ) -> Normal:
        parameters = self.conditionals[dimension](torch.cat([observations, partial_actions], -1))
        mean, log_standard_deviation = parameters.unbind(dim=-1)
        return Normal(mean, log_standard_deviation.clamp(-5.0, 2.0).exp())

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = torch.zeros(len(observations), self.action_size, device=observations.device)
        log_probability = torch.zeros(len(observations), device=observations.device)
        for dimension in range(self.action_size):
            distribution = self._distribution(observations, actions, dimension)
            value = distribution.rsample()
            # Avoid in-place writes on a tensor used by autograd in earlier factors.
            mask = torch.zeros_like(actions)
            mask[:, dimension] = value
            actions = actions + mask
            log_probability = log_probability + distribution.log_prob(value)
        return torch.tanh(actions), log_probability

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        raw_actions = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
        partial = torch.zeros_like(raw_actions)
        total = torch.zeros(len(actions), device=actions.device)
        for dimension in range(self.action_size):
            distribution = self._distribution(observations, partial, dimension)
            value = raw_actions[:, dimension]
            total = total + distribution.log_prob(value) - torch.log(1.0 - actions[:, dimension].pow(2) + 1e-6)
            next_partial = partial.clone()
            next_partial[:, dimension] = value
            partial = next_partial
        return total


def binary_sequences(length: int) -> np.ndarray:
    """Small helper used by examples and tests of autoregressive generation."""

    return np.asarray(list(product((-1.0, 1.0), repeat=length)), dtype=np.float32)
