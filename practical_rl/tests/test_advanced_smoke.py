import math

import numpy as np
import pytest

from edu_rl.advanced.exploration import RNDConfig, train_rnd
from edu_rl.advanced.her import HERConfig, train_her
from edu_rl.advanced.imitation_skills import (
    DIAYNConfig,
    GAILConfig,
    SkewFitConfig,
    train_diayn,
    train_gail,
    train_skew_fit,
)
from edu_rl.advanced.model_based import MBPOConfig, train_dyna_q, train_mbpo
from edu_rl.advanced.offline_rl import OfflineConfig, OfflineDataset, train_offline
from edu_rl.advanced.preference_rl import GRPOConfig, train_grpo, train_mini_rlhf
from edu_rl.advanced.sac import SACConfig, train_sac
from edu_rl.advanced.transfer_hierarchy import (
    OptionCriticConfig,
    generalized_policy_improvement,
    learn_successor_features,
    train_option_critic,
    train_smdp_options,
)
from edu_rl.envs import FourRooms, GridWorld, PointMass2D


def test_sac_short_training_path() -> None:
    result = train_sac(
        SACConfig(
            total_steps=20,
            warmup_steps=4,
            batch_size=4,
            replay_capacity=64,
            hidden_sizes=(8,),
            seed=21,
        ),
        env=PointMass2D(max_steps=10),
    )
    assert result.critic_losses
    assert all(math.isfinite(value) for value in result.critic_losses)


def test_dyna_and_mbpo_short_paths() -> None:
    q_values, returns = train_dyna_q(
        GridWorld(slip=0.0, max_steps=20), episodes=3, planning_steps=2, seed=22
    )
    assert q_values.shape == (16, 4)
    assert len(returns) == 3
    result = train_mbpo(
        MBPOConfig(
            total_steps=20,
            warmup_steps=8,
            batch_size=4,
            model_train_interval=8,
            model_epochs=1,
            model_rollout_batch=4,
            model_rollout_horizon=1,
            ensemble_size=2,
            hidden_sizes=(8,),
            seed=22,
        )
    )
    assert result.model_losses


def small_offline_dataset() -> OfflineDataset:
    rng = np.random.default_rng(23)
    observations = rng.normal(size=(64, 3)).astype(np.float32)
    actions = rng.uniform(-2, 2, size=(64, 1)).astype(np.float32)
    return OfflineDataset(
        observations,
        actions,
        rng.normal(size=64).astype(np.float32),
        (observations + rng.normal(0, 0.1, size=(64, 3))).astype(np.float32),
        np.zeros(64, dtype=np.float32),
    )


@pytest.mark.parametrize("algorithm", ["sac_bc", "awac", "iql", "cql"])
def test_every_offline_algorithm_short_path(algorithm: str) -> None:
    result = train_offline(
        small_offline_dataset(),
        OfflineConfig(
            algorithm=algorithm,
            gradient_steps=2,
            batch_size=8,
            hidden_sizes=(8,),
            seed=23,
        ),
        action_low=np.array([-2.0], dtype=np.float32),
        action_high=np.array([2.0], dtype=np.float32),
    )
    assert len(result.critic_losses) == 2
    assert all(math.isfinite(value) for value in result.critic_losses)


def test_rnd_and_her_short_paths() -> None:
    rnd = train_rnd(
        RNDConfig(
            total_steps=20,
            warmup_steps=4,
            batch_size=4,
            chain_length=5,
            hidden_sizes=(8,),
            seed=24,
        )
    )
    assert rnd.intrinsic_rewards
    her = train_her(
        HERConfig(
            episodes=6,
            n_bits=3,
            her_k=2,
            batch_size=8,
            hidden_sizes=(8,),
            seed=24,
        )
    )
    assert her.losses


def test_gail_diayn_and_skew_fit_short_paths() -> None:
    gail = train_gail(
        GAILConfig(
            iterations=1,
            rollout_steps=32,
            discriminator_epochs=1,
            policy_epochs=1,
            expert_episodes=2,
            seed=25,
        )
    )
    assert len(gail.discriminator_losses) == 1
    diayn = train_diayn(
        DIAYNConfig(
            total_steps=20,
            warmup_steps=4,
            batch_size=4,
            skill_count=3,
            hidden_sizes=(8,),
            seed=25,
        )
    )
    assert diayn.intrinsic_rewards
    skew = train_skew_fit(
        SkewFitConfig(
            total_steps=70,
            warmup_steps=4,
            batch_size=4,
            goal_update_interval=64,
            vae_epochs=1,
            hidden_sizes=(8,),
            seed=25,
        )
    )
    assert skew.vae_losses


def test_grpo_and_mini_rlhf_short_paths() -> None:
    config = GRPOConfig(
        iterations=2,
        prompt_count=2,
        vocabulary_size=3,
        sequence_length=3,
        group_size=3,
        update_epochs=1,
        hidden_size=8,
        seed=26,
    )
    result = train_grpo(config)
    assert len(result.mean_rewards) == 2
    rlhf = train_mini_rlhf(config, preference_pairs=32, reward_model_epochs=2)
    assert len(rlhf.reward_model_losses) == 2


def test_transfer_and_hierarchy_short_paths() -> None:
    env = FourRooms(max_steps=20, seed=27)
    policy_a = env.shortest_path_policy(env.goal_state)
    successor_a = learn_successor_features(env, policy_a, episodes=3, seed=27)
    env.set_goal(env.state_for_cell[(0, 6)])
    policy_b = env.shortest_path_policy(env.goal_state)
    successor_b = learn_successor_features(env, policy_b, episodes=3, seed=28)
    reward_weights = np.zeros(env.n_states)
    reward_weights[env.goal_state] = 1.0
    policy, q_values = generalized_policy_improvement(
        np.stack([successor_a, successor_b]), reward_weights
    )
    assert policy.shape == (env.n_states,)
    assert q_values.shape == (env.n_states, env.n_actions)
    options = train_smdp_options(env, episodes=2, seed=27)
    assert len(options.episode_returns) == 2
    option_critic = train_option_critic(
        env,
        OptionCriticConfig(
            episodes=2,
            option_count=2,
            seed=27,
        ),
    )
    assert len(option_critic.episode_returns) == 2

