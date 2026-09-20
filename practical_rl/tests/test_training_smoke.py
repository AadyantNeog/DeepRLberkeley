import math

import pytest

from edu_rl.algorithms.actor_critic import ActorCriticConfig, train_actor_critic
from edu_rl.algorithms.ddpg import DDPGConfig, train_ddpg
from edu_rl.algorithms.dqn import DQNConfig, train_dqn
from edu_rl.algorithms.natural_policy_gradient import (
    NaturalPolicyConfig,
    train_natural_policy,
)
from edu_rl.algorithms.ppo import PPOConfig, train_ppo
from edu_rl.algorithms.reinforce import ReinforceConfig, train_reinforce


def test_reinforce_short_training_path() -> None:
    result = train_reinforce(
        ReinforceConfig(
            iterations=1,
            episodes_per_batch=1,
            hidden_sizes=(8,),
            seed=11,
        )
    )
    assert len(result.mean_returns) == 1
    assert math.isfinite(result.policy_losses[0])


@pytest.mark.parametrize("estimator", ["td0", "nstep", "gae"])
def test_every_actor_critic_estimator_short_training_path(estimator: str) -> None:
    result = train_actor_critic(
        ActorCriticConfig(
            iterations=1,
            episodes_per_batch=1,
            estimator=estimator,
            hidden_sizes=(8,),
            seed=12,
        )
    )
    assert len(result.mean_returns) == 1
    assert math.isfinite(result.value_losses[0])


@pytest.mark.parametrize("double_dqn", [False, True])
def test_dqn_and_double_dqn_short_training_path(double_dqn: bool) -> None:
    result = train_dqn(
        DQNConfig(
            total_steps=30,
            warmup_steps=8,
            batch_size=8,
            replay_capacity=64,
            target_update_interval=10,
            hidden_sizes=(8,),
            double_dqn=double_dqn,
            seed=13,
        )
    )
    assert result.losses
    assert all(math.isfinite(loss) for loss in result.losses)


def test_ddpg_short_training_path() -> None:
    result = train_ddpg(
        DDPGConfig(
            total_steps=24,
            warmup_steps=8,
            batch_size=8,
            replay_capacity=64,
            hidden_sizes=(8,),
            seed=14,
        )
    )
    assert result.critic_losses
    assert all(math.isfinite(loss) for loss in result.critic_losses)


@pytest.mark.parametrize("method", ["clip", "kl"])
def test_both_ppo_objectives_short_training_path(method: str) -> None:
    result = train_ppo(
        PPOConfig(
            iterations=1,
            rollout_steps=32,
            update_epochs=1,
            minibatch_size=16,
            method=method,
            hidden_sizes=(8,),
            seed=15,
        )
    )
    assert len(result.approximate_kls) == 1
    assert math.isfinite(result.approximate_kls[0])


@pytest.mark.parametrize("method", ["npg", "trpo"])
def test_natural_gradient_and_trpo_short_training_path(method: str) -> None:
    result = train_natural_policy(
        NaturalPolicyConfig(
            iterations=1,
            rollout_steps=32,
            method=method,
            conjugate_gradient_steps=3,
            line_search_steps=3,
            critic_epochs=1,
            hidden_sizes=(8,),
            seed=16,
        )
    )
    assert len(result.policy_kls) == 1
    assert math.isfinite(result.policy_kls[0])

