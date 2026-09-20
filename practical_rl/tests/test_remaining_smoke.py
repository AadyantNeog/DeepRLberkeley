import math

import numpy as np
import torch

from edu_rl.advanced.continuous_extensions import (
    ContinuousSFConfig,
    TD3Config,
    train_continuous_successor_features,
    train_td3,
)
from edu_rl.advanced.exploration_extensions import goal_conditioned_empowerment
from edu_rl.advanced.generative_models import train_flow_matching, train_reflow
from edu_rl.advanced.maxent_irl import (
    four_rooms_dynamics,
    guided_cost_learning,
    maximum_entropy_irl,
)
from edu_rl.advanced.model_based_extensions import (
    MVEConfig,
    ModelBasedOfflineConfig,
    generate_point_mass_dataset,
    train_model_based_offline,
    train_model_value_expansion,
)
from edu_rl.advanced.offline_extensions import (
    BRACConfig,
    GenerativeOfflineConfig,
    sample_diffusion_steered_actions,
    sample_idql_actions,
    train_brac,
    train_generative_offline,
)
from edu_rl.advanced.offline_rl import OfflineDataset
from edu_rl.envs import FourRooms


def small_dataset(seed: int = 30) -> OfflineDataset:
    rng = np.random.default_rng(seed)
    observations = rng.normal(size=(32, 3)).astype(np.float32)
    actions = rng.uniform(-2, 2, size=(32, 1)).astype(np.float32)
    return OfflineDataset(
        observations,
        actions,
        rng.normal(size=32).astype(np.float32),
        (observations + rng.normal(0, 0.1, size=(32, 3))).astype(np.float32),
        np.zeros(32, dtype=np.float32),
    )


def test_flow_and_reflow_short_paths() -> None:
    rng = np.random.default_rng(30)
    data = rng.normal(size=(32, 2)).astype(np.float32)
    flow = train_flow_matching(
        data, gradient_steps=2, batch_size=8, hidden_sizes=(8,), seed=30
    )
    assert len(flow.losses) == 2
    first, second = train_reflow(
        data, first_stage_steps=2, reflow_steps=2, batch_size=8,
        hidden_sizes=(8,), seed=30,
    )
    assert first.losses and second.losses


def test_maxent_irl_and_gcl_short_paths() -> None:
    env = FourRooms(max_steps=20, seed=31)
    dynamics = four_rooms_dynamics(env)
    demonstrations = [np.array([env.start_state] * 4, dtype=np.int64)]
    features = np.eye(env.n_states)
    irl = maximum_entropy_irl(
        dynamics, features, demonstrations, env.start_state,
        horizon=4, iterations=2,
    )
    assert len(irl.feature_errors) == 2
    gcl = guided_cost_learning(
        dynamics, features, demonstrations, env.start_state,
        horizon=4, iterations=2, sampled_trajectories=4, seed=31,
    )
    assert len(gcl.losses) == 2
    capacity, _, goals = goal_conditioned_empowerment(env, env.start_state, horizon=2)
    assert capacity >= 0 and len(goals) > 0


def test_mve_mopo_and_combo_short_paths() -> None:
    mve = train_model_value_expansion(
        MVEConfig(
            transitions=32, gradient_steps=2, model_epochs=1,
            rollout_horizon=1, batch_size=4, hidden_sizes=(8,), seed=32,
        )
    )
    assert mve.value_losses
    for algorithm in ("mopo", "combo"):
        result = train_model_based_offline(
            ModelBasedOfflineConfig(
                algorithm=algorithm, dataset_transitions=32, bootstrap_steps=1,
                gradient_steps=1, model_epochs=1, model_rollout_batch=4,
                model_rollout_horizon=1, batch_size=4, hidden_sizes=(8,), seed=32,
            )
        )
        assert result.learner.critic_losses
        assert result.synthetic_dataset_size == 4


def test_brac_and_generative_offline_short_paths() -> None:
    dataset = small_dataset()
    low = np.array([-2.0], dtype=np.float32)
    high = np.array([2.0], dtype=np.float32)
    brac = train_brac(
        dataset,
        BRACConfig(
            behavior_steps=1, gradient_steps=1, batch_size=4,
            hidden_sizes=(8,), seed=33,
        ),
        action_low=low,
        action_high=high,
    )
    assert brac.critic_losses
    for algorithm in ("idql", "fql", "diffusion-steering"):
        result = train_generative_offline(
            dataset,
            GenerativeOfflineConfig(
                algorithm=algorithm, critic_steps=1, generative_steps=1,
                actor_steps=1, batch_size=4, diffusion_steps=2,
                hidden_sizes=(8,), seed=33,
            ),
            action_low=low,
            action_high=high,
        )
        observations = torch.as_tensor(dataset.observations[:2])
        if algorithm == "idql":
            actions = sample_idql_actions(result, observations, candidate_count=2)
        elif algorithm == "diffusion-steering":
            actions = sample_diffusion_steered_actions(result, observations)
        else:
            actions = result.one_step_actor(observations)
        assert actions.shape == (2, 1)
        assert torch.isfinite(actions).all()


def test_td3_and_continuous_successor_features_short_paths() -> None:
    td3 = train_td3(
        TD3Config(
            total_steps=12, warmup_steps=4, batch_size=4, replay_capacity=32,
            hidden_sizes=(8,), seed=34,
        )
    )
    assert td3.critic_losses
    assert all(math.isfinite(value) for value in td3.critic_losses)
    successor = train_continuous_successor_features(
        ContinuousSFConfig(
            total_steps=12, warmup_steps=4, batch_size=4, replay_capacity=32,
            hidden_sizes=(8,), seed=34,
        )
    )
    assert successor.losses
