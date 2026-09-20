import numpy as np

from edu_rl.algorithms.dynamic_programming import (
    action_values,
    policy_iteration,
    value_iteration,
)
from edu_rl.algorithms.tabular_learning import q_learning
from edu_rl.envs import GridWorld, TinyMDP


def test_tiny_mdp_exact_solution() -> None:
    env = TinyMDP()
    values, policy = value_iteration(env.P, env.R, gamma=0.99)
    np.testing.assert_allclose(values, [0.99, 1.0, 0.0], atol=1e-8)
    np.testing.assert_array_equal(policy, [0, 0, 0])


def test_policy_and_value_iteration_agree_on_gridworld() -> None:
    env = GridWorld(slip=0.1)
    values_vi, policy_vi = value_iteration(env.P, env.R)
    values_pi, policy_pi = policy_iteration(env.P, env.R)
    np.testing.assert_allclose(values_pi, values_vi, atol=1e-7)
    # Different argmax tie-breaking can produce different, equally optimal
    # policies.  Verify the value of each selected action instead.
    q_values = action_values(env.P, env.R, values_vi, gamma=0.99)
    np.testing.assert_allclose(q_values[np.arange(env.n_states), policy_vi], values_vi)
    np.testing.assert_allclose(q_values[np.arange(env.n_states), policy_pi], values_vi)
    np.testing.assert_allclose(env.P.sum(axis=2), 1.0)


def test_tabular_q_learning_finds_tiny_mdp_optimal_start_action() -> None:
    env = TinyMDP()
    q_values, episode_returns = q_learning(
        env,
        episodes=1_000,
        learning_rate=0.2,
        epsilon_end=0.1,
        seed=4,
    )
    assert int(q_values[0].argmax()) == 0
    assert int(q_values[1].argmax()) == 0
    assert len(episode_returns) == 1_000
