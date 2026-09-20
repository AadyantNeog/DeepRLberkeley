from edu_rl.algorithms.dynamic_programming import value_iteration
from edu_rl.algorithms.imitation import behavioral_cloning, dagger
from edu_rl.envs import GridWorld


def test_behavioral_cloning_and_dagger_run_and_fit_labels() -> None:
    env = GridWorld(slip=0.0, max_steps=30)
    _, expert = value_iteration(env.P, env.R)
    cloning = behavioral_cloning(
        env, expert, expert_episodes=3, epochs=30, seed=1
    )
    assert cloning.losses[-1] < cloning.losses[0]

    aggregated = dagger(
        env,
        expert,
        initial_expert_episodes=2,
        iterations=2,
        learner_episodes_per_iteration=2,
        epochs_per_iteration=10,
        seed=1,
    )
    assert aggregated.dataset_sizes[1] > aggregated.dataset_sizes[0]
    assert len(aggregated.evaluation_returns) == 2

