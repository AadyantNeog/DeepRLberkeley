"""A small, fully inspectable GRPO and preference-learning/RLHF pipeline.

The sequence model is intentionally a tiny GRU rather than a downloaded LLM.
That keeps group advantages, policy ratios, reference KL, Bradley–Terry reward
learning, and reward-model exploitation visible in ordinary PyTorch code.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from edu_rl.core.networks import mlp
from edu_rl.core.utils import set_global_seeds


class SequencePolicy(nn.Module):
    def __init__(
        self, prompt_count: int, vocabulary_size: int, hidden_size: int = 64
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.prompt_embedding = nn.Embedding(prompt_count, hidden_size)
        self.token_embedding = nn.Embedding(vocabulary_size + 1, hidden_size)
        self.start_token = vocabulary_size
        self.recurrent = nn.GRUCell(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocabulary_size)

    def generate(
        self, prompts: torch.Tensor, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.prompt_embedding(prompts))
        previous = torch.full_like(prompts, self.start_token)
        tokens: list[torch.Tensor] = []
        log_probabilities = torch.zeros(len(prompts), device=prompts.device)
        for _ in range(length):
            hidden = self.recurrent(self.token_embedding(previous), hidden)
            distribution = Categorical(logits=self.output(hidden))
            token = distribution.sample()
            tokens.append(token)
            log_probabilities += distribution.log_prob(token)
            previous = token
        return torch.stack(tokens, dim=1), log_probabilities

    def log_prob_and_entropy(
        self, prompts: torch.Tensor, sequences: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.prompt_embedding(prompts))
        previous = torch.full_like(prompts, self.start_token)
        log_probability = torch.zeros(len(prompts), device=prompts.device)
        entropy = torch.zeros_like(log_probability)
        for time in range(sequences.shape[1]):
            hidden = self.recurrent(self.token_embedding(previous), hidden)
            distribution = Categorical(logits=self.output(hidden))
            token = sequences[:, time]
            log_probability += distribution.log_prob(token)
            entropy += distribution.entropy()
            previous = token
        return log_probability, entropy

    def conditional_kl(
        self,
        reference: "SequencePolicy",
        prompts: torch.Tensor,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        """Sum exact token KLs along teacher-forced sampled prefixes."""

        hidden = torch.tanh(self.prompt_embedding(prompts))
        reference_hidden = torch.tanh(reference.prompt_embedding(prompts))
        previous = torch.full_like(prompts, self.start_token)
        total_kl = torch.zeros(len(prompts), device=prompts.device)
        for time in range(sequences.shape[1]):
            hidden = self.recurrent(self.token_embedding(previous), hidden)
            with torch.no_grad():
                reference_hidden = reference.recurrent(
                    reference.token_embedding(previous), reference_hidden
                )
            distribution = Categorical(logits=self.output(hidden))
            reference_distribution = Categorical(
                logits=reference.output(reference_hidden)
            )
            total_kl += torch.distributions.kl_divergence(
                distribution, reference_distribution
            )
            previous = sequences[:, time]
        return total_kl


def target_sequences(
    prompt_count: int, vocabulary_size: int, length: int, device: torch.device
) -> torch.Tensor:
    prompts = torch.arange(prompt_count, device=device)[:, None]
    positions = torch.arange(length, device=device)[None, :]
    return (prompts + positions) % vocabulary_size


def true_sequence_reward(
    prompts: torch.Tensor, sequences: torch.Tensor, vocabulary_size: int
) -> torch.Tensor:
    """Hidden oracle: token accuracy plus a small coherent-transition bonus."""

    positions = torch.arange(sequences.shape[1], device=sequences.device)[None, :]
    targets = (prompts[:, None] + positions) % vocabulary_size
    accuracy = (sequences == targets).float().mean(dim=1)
    if sequences.shape[1] > 1:
        transitions = (
            sequences[:, 1:] == (sequences[:, :-1] + 1) % vocabulary_size
        ).float().mean(dim=1)
    else:
        transitions = torch.zeros_like(accuracy)
    return accuracy + 0.2 * transitions


def group_relative_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Normalize each prompt's completion rewards independently."""

    centered = rewards - rewards.mean(dim=1, keepdim=True)
    return centered / (rewards.std(dim=1, keepdim=True, unbiased=False) + 1e-6)


class PreferenceRewardModel(nn.Module):
    def __init__(
        self, prompt_count: int, vocabulary_size: int, hidden_size: int = 64
    ) -> None:
        super().__init__()
        self.prompt_embedding = nn.Embedding(prompt_count, hidden_size)
        self.token_embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.scorer = mlp(2 * hidden_size, (64,), 1)

    def forward(self, prompts: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        prompt_features = self.prompt_embedding(prompts)
        sequence_features = self.token_embedding(sequences).mean(dim=1)
        return self.scorer(
            torch.cat([prompt_features, sequence_features], dim=1)
        ).squeeze(1)


def bradley_terry_loss(
    score_a: torch.Tensor, score_b: torch.Tensor, prefers_a: torch.Tensor
) -> torch.Tensor:
    """Pairwise likelihood ``P(a>b)=sigmoid(r(a)-r(b))``."""

    return nn.functional.binary_cross_entropy_with_logits(
        score_a - score_b, prefers_a.float()
    )


@dataclass
class GRPOConfig:
    iterations: int = 300
    prompt_count: int = 8
    vocabulary_size: int = 6
    sequence_length: int = 6
    group_size: int = 8
    update_epochs: int = 4
    clip_ratio: float = 0.2
    reference_kl_coefficient: float = 0.02
    learning_rate: float = 3e-4
    hidden_size: int = 64
    seed: int = 0
    device: str = "cpu"


@dataclass
class GRPOResult:
    policy: SequencePolicy
    mean_rewards: list[float]
    policy_losses: list[float]
    reference_kls: list[float]


def train_grpo(
    config: GRPOConfig,
    *,
    reward_model: PreferenceRewardModel | None = None,
) -> GRPOResult:
    set_global_seeds(config.seed)
    device = torch.device(config.device)
    policy = SequencePolicy(
        config.prompt_count, config.vocabulary_size, config.hidden_size
    ).to(device)
    reference = copy.deepcopy(policy).to(device)
    reference.requires_grad_(False)
    if reward_model is not None:
        reward_model = reward_model.to(device)
        reward_model.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    mean_rewards: list[float] = []
    policy_losses: list[float] = []
    reference_kls: list[float] = []

    for _ in range(config.iterations):
        prompts = torch.arange(config.prompt_count, device=device).repeat_interleave(
            config.group_size
        )
        with torch.no_grad():
            sequences, old_log_probability = policy.generate(
                prompts, config.sequence_length
            )
            if reward_model is None:
                rewards = true_sequence_reward(
                    prompts, sequences, config.vocabulary_size
                )
            else:
                rewards = reward_model(prompts, sequences)
            grouped_rewards = rewards.view(config.prompt_count, config.group_size)
            advantages = group_relative_advantages(grouped_rewards).reshape(-1)

        last_loss = torch.zeros((), device=device)
        last_reference_kl = torch.zeros((), device=device)
        for _ in range(config.update_epochs):
            new_log_probability, entropy = policy.log_prob_and_entropy(prompts, sequences)
            ratio = (new_log_probability - old_log_probability).exp()
            clipped_ratio = ratio.clamp(
                1.0 - config.clip_ratio, 1.0 + config.clip_ratio
            )
            policy_gain = torch.minimum(ratio * advantages, clipped_ratio * advantages)
            reference_kl = policy.conditional_kl(reference, prompts, sequences)
            loss = -policy_gain.mean() + config.reference_kl_coefficient * reference_kl.mean()
            loss -= 1e-3 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            last_loss = loss.detach()
            last_reference_kl = reference_kl.detach().mean()
        mean_rewards.append(float(rewards.mean()))
        policy_losses.append(float(last_loss))
        reference_kls.append(float(last_reference_kl))
    return GRPOResult(policy, mean_rewards, policy_losses, reference_kls)


@dataclass
class RLHFResult:
    reward_model: PreferenceRewardModel
    grpo: GRPOResult
    reward_model_losses: list[float]
    true_reward_before: float
    true_reward_after: float


def train_mini_rlhf(
    config: GRPOConfig,
    *,
    preference_pairs: int = 5_000,
    reward_model_epochs: int = 200,
) -> RLHFResult:
    """Train a Bradley–Terry reward model, then optimize it with GRPO."""

    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    prompts = torch.as_tensor(
        rng.integers(config.prompt_count, size=preference_pairs), device=device
    )
    sequence_a = torch.as_tensor(
        rng.integers(
            config.vocabulary_size,
            size=(preference_pairs, config.sequence_length),
        ),
        device=device,
    )
    sequence_b = torch.as_tensor(
        rng.integers(
            config.vocabulary_size,
            size=(preference_pairs, config.sequence_length),
        ),
        device=device,
    )
    reward_a = true_sequence_reward(prompts, sequence_a, config.vocabulary_size)
    reward_b = true_sequence_reward(prompts, sequence_b, config.vocabulary_size)
    preferences = (reward_a > reward_b).float()
    reward_model = PreferenceRewardModel(
        config.prompt_count, config.vocabulary_size, config.hidden_size
    ).to(device)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
    losses: list[float] = []
    for _ in range(reward_model_epochs):
        indices = torch.as_tensor(
            rng.integers(preference_pairs, size=min(256, preference_pairs)), device=device
        )
        loss = bradley_terry_loss(
            reward_model(prompts[indices], sequence_a[indices]),
            reward_model(prompts[indices], sequence_b[indices]),
            preferences[indices],
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    # Recreate the exact initialization used by train_grpo for a fair before/after
    # comparison; reward-model training has consumed random numbers by this point.
    torch.manual_seed(config.seed)
    baseline_policy = SequencePolicy(
        config.prompt_count, config.vocabulary_size, config.hidden_size
    ).to(device)
    evaluation_prompts = torch.arange(config.prompt_count, device=device).repeat_interleave(32)
    with torch.no_grad():
        baseline_sequences, _ = baseline_policy.generate(
            evaluation_prompts, config.sequence_length
        )
        before = float(
            true_sequence_reward(
                evaluation_prompts, baseline_sequences, config.vocabulary_size
            ).mean()
        )
    grpo_result = train_grpo(config, reward_model=reward_model)
    with torch.no_grad():
        final_sequences, _ = grpo_result.policy.generate(
            evaluation_prompts, config.sequence_length
        )
        after = float(
            true_sequence_reward(
                evaluation_prompts, final_sequences, config.vocabulary_size
            ).mean()
        )
    return RLHFResult(reward_model, grpo_result, losses, before, after)
