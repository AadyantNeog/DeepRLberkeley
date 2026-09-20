import math

import numpy as np
import torch

from edu_rl.advanced.continuous_extensions import td3_target
from edu_rl.advanced.exploration_extensions import (
    ContextTreeSwitching,
    blahut_arimoto,
    density_pseudo_count,
)
from edu_rl.advanced.generative_models import (
    ConditionalDiffusionPolicy,
    VAE,
    coordinate_ascent_gaussian_vi,
    gaussian_mixture_em,
    gaussian_vae_loss,
)
from edu_rl.advanced.maxent_irl import soft_value_iteration
from edu_rl.advanced.model_based_extensions import (
    combo_conservative_penalty,
    model_value_expansion_target,
    mopo_penalized_reward,
)
from edu_rl.advanced.offline_extensions import normalize_actions


def test_cavi_mean_matches_exact_gaussian_mean() -> None:
    precision = np.array([[2.0, 0.5], [0.5, 1.0]])
    eta = np.array([1.0, -1.0])
    result = coordinate_ascent_gaussian_vi(precision, eta, iterations=30)
    np.testing.assert_allclose(result.mean, np.linalg.solve(precision, eta), atol=1e-8)
    np.testing.assert_allclose(result.variance, 1.0 / np.diag(precision))


def test_em_log_likelihood_is_non_decreasing() -> None:
    rng = np.random.default_rng(1)
    data = np.concatenate([rng.normal(-2, 0.2, (50, 1)), rng.normal(2, 0.2, (50, 1))])
    result = gaussian_mixture_em(data, 2, iterations=10, seed=1)
    assert np.all(np.diff(result.log_likelihoods) >= -1e-7)
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), 1.0)


def test_vae_reparameterized_elbo_has_gradients() -> None:
    model = VAE(3, 2, hidden_size=8)
    inputs = torch.randn(5, 3)
    reconstruction, mean, log_variance = model(inputs)
    loss, reconstruction_loss, kl = gaussian_vae_loss(
        reconstruction, inputs, mean, log_variance
    )
    loss.backward()
    assert reconstruction_loss >= 0 and kl >= 0
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_diffusion_policy_loss_and_sample_shapes() -> None:
    policy = ConditionalDiffusionPolicy(3, 2, diffusion_steps=3, hidden_sizes=(8,))
    observations = torch.randn(4, 3)
    actions = torch.randn(4, 2).clamp(-1, 1)
    assert torch.isfinite(policy.loss(observations, actions))
    samples = policy.sample(observations)
    assert samples.shape == (4, 2)
    assert torch.all(samples.abs() <= 1.0)


def test_soft_value_iteration_prefers_better_action() -> None:
    dynamics = np.ones((1, 2, 1))
    q_values, values, policy = soft_value_iteration(
        dynamics, np.array([[0.0, 1.0]]), gamma=0.0, temperature=0.5
    )
    np.testing.assert_allclose(q_values, [[0.0, 1.0]])
    assert values[0] > 1.0
    assert policy[0, 1] > policy[0, 0]


def test_model_value_expansion_terminal_mask() -> None:
    rewards = torch.tensor([[1.0, 2.0, 100.0], [1.0, 2.0, 3.0]])
    terminated = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    target = model_value_expansion_target(
        rewards, torch.tensor([10.0, 10.0]), terminated, gamma=0.5
    )
    # Row 1 terminates after reward 2. Row 2 also receives the H-step bootstrap:
    # 1 + .5*2 + .25*3 + .125*10 = 4.
    torch.testing.assert_close(target, torch.tensor([2.0, 4.0]))


def test_mopo_and_combo_penalties_are_pessimistic() -> None:
    penalized = mopo_penalized_reward(np.array([2.0]), np.array([0.5]), 2.0)
    np.testing.assert_allclose(penalized, [1.0])
    penalty = combo_conservative_penalty(
        torch.tensor([3.0, 3.0]), torch.tensor([1.0, 1.0])
    )
    torch.testing.assert_close(penalty, torch.tensor(2.0))


def test_action_normalization_respects_environment_bounds() -> None:
    normalized = normalize_actions(
        np.array([[-2.0], [0.0], [2.0]]), np.array([-2.0]), np.array([2.0])
    )
    np.testing.assert_allclose(normalized[:, 0], [-1.0, 0.0, 1.0])


def test_density_pseudocount_recovers_literal_count() -> None:
    # One of two observations has count 1: p=1/2. Observing it again gives p'=2/3.
    assert math.isclose(density_pseudo_count(0.5, 2.0 / 3.0), 1.0)
    cts = ContextTreeSwitching(maximum_depth=2)
    before, after = cts.update(1)
    assert 0 < before <= after < 1


def test_blahut_arimoto_identity_channel_capacity() -> None:
    capacity, distribution = blahut_arimoto(np.eye(2))
    assert math.isclose(capacity, math.log(2.0), rel_tol=1e-7)
    np.testing.assert_allclose(distribution, [0.5, 0.5])


def test_td3_target_uses_smaller_critic_and_terminal_mask() -> None:
    target = td3_target(
        torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0]),
        torch.tensor([4.0, 4.0]), torch.tensor([2.0, 3.0]), gamma=0.5,
    )
    torch.testing.assert_close(target, torch.tensor([2.0, 1.0]))
