"""Small neural networks and action distributions used by the algorithms."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal


def mlp(
    input_size: int,
    hidden_sizes: Sequence[int],
    output_size: int,
    *,
    output_activation: type[nn.Module] = nn.Identity,
) -> nn.Sequential:
    """Build a tanh MLP; small smooth networks suit the classic-control tasks."""

    sizes = [input_size, *hidden_sizes, output_size]
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        activation = nn.Tanh if index < len(sizes) - 2 else output_activation
        linear = nn.Linear(sizes[index], sizes[index + 1])
        # Orthogonal initialization is predictable and works well for RL MLPs.
        gain = math.sqrt(2.0) if activation is nn.Tanh else 1.0
        nn.init.orthogonal_(linear.weight, gain=gain)
        nn.init.zeros_(linear.bias)
        layers.extend([linear, activation()])
    return nn.Sequential(*layers)


class CategoricalActor(nn.Module):
    """A stochastic policy for a finite action space."""

    def __init__(
        self, observation_size: int, action_count: int, hidden_sizes: Sequence[int]
    ) -> None:
        super().__init__()
        self.logits_network = mlp(observation_size, hidden_sizes, action_count)

    def distribution(self, observations: torch.Tensor) -> Categorical:
        return Categorical(logits=self.logits_network(observations))

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions)

    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        return self.logits_network(observations).argmax(dim=-1)


class SquashedGaussianActor(nn.Module):
    """A tanh-squashed Gaussian policy with actions scaled to environment bounds.

    The log-probability includes the tanh Jacobian correction.  Omitting it makes
    PPO's likelihood ratio mathematically inconsistent with sampled actions.
    """

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: Sequence[int],
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        super().__init__()
        self.mean_network = mlp(observation_size, hidden_sizes, action_size)
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))
        action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high_tensor - action_low_tensor) / 2)
        self.register_buffer("action_bias", (action_high_tensor + action_low_tensor) / 2)

    def base_distribution(self, observations: torch.Tensor) -> Independent:
        mean = self.mean_network(observations)
        log_std = self.log_std.clamp(-5.0, 2.0)
        return Independent(Normal(mean, log_std.exp().expand_as(mean)), 1)

    def _squash(self, raw_actions: torch.Tensor) -> torch.Tensor:
        return self.action_bias + self.action_scale * torch.tanh(raw_actions)

    def log_prob_from_raw(
        self, distribution: Independent, raw_actions: torch.Tensor
    ) -> torch.Tensor:
        # The affine scaling contributes a constant Jacobian term.  It matters
        # for absolute log-probabilities and cancels in same-scale PPO ratios.
        base_log_prob = distribution.log_prob(raw_actions)
        tanh_correction = torch.log(
            self.action_scale * (1.0 - torch.tanh(raw_actions).pow(2)) + 1e-6
        ).sum(dim=-1)
        return base_log_prob - tanh_correction

    def sample(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.base_distribution(observations)
        raw_actions = distribution.rsample()
        actions = self._squash(raw_actions)
        return actions, self.log_prob_from_raw(distribution, raw_actions), raw_actions

    def log_prob(
        self, observations: torch.Tensor, raw_actions: torch.Tensor
    ) -> torch.Tensor:
        distribution = self.base_distribution(observations)
        return self.log_prob_from_raw(distribution, raw_actions)

    def raw_from_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Invert the affine-tanh transform for dataset actions.

        Offline RL observes bounded environment actions rather than the
        pre-squash Gaussian samples stored by an online policy.  Clamping keeps
        ``atanh`` finite for actions recorded exactly at an environment bound.
        """

        normalized = (actions - self.action_bias) / self.action_scale
        normalized = normalized.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.atanh(normalized)

    def log_prob_of_action(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.log_prob(observations, self.raw_from_action(actions))

    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        return self._squash(self.mean_network(observations))


class ValueNetwork(nn.Module):
    def __init__(self, observation_size: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.network = mlp(observation_size, hidden_sizes, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(-1)


class DiscreteQNetwork(nn.Module):
    def __init__(
        self, observation_size: int, action_count: int, hidden_sizes: Sequence[int]
    ) -> None:
        super().__init__()
        self.network = mlp(observation_size, hidden_sizes, action_count)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class DeterministicActor(nn.Module):
    """The learned continuous-action maximizer used by DDPG."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: Sequence[int],
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        super().__init__()
        self.network = mlp(observation_size, hidden_sizes, action_size)
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2)
        self.register_buffer("action_bias", (high + low) / 2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.action_bias + self.action_scale * torch.tanh(
            self.network(observations)
        )


class ContinuousQNetwork(nn.Module):
    def __init__(
        self, observation_size: int, action_size: int, hidden_sizes: Sequence[int]
    ) -> None:
        super().__init__()
        self.network = mlp(observation_size + action_size, hidden_sizes, 1)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat([observations, actions], dim=-1)).squeeze(-1)
