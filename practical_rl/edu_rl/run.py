"""Command-line experiments for every algorithm in the learning package.

Run ``python -m edu_rl.run --help`` from the ``practical_rl`` directory.  The
``--quick`` flag is a wiring check; omit it for a meaningful learning run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from edu_rl.algorithms.actor_critic import ActorCriticConfig, train_actor_critic
from edu_rl.algorithms.ddpg import DDPGConfig, train_ddpg
from edu_rl.algorithms.dqn import DQNConfig, train_dqn
from edu_rl.algorithms.dynamic_programming import policy_iteration, value_iteration
from edu_rl.algorithms.imitation import behavioral_cloning, dagger
from edu_rl.algorithms.importance_sampling import bandit_importance_sampling_demo
from edu_rl.algorithms.natural_policy_gradient import (
    NaturalPolicyConfig,
    train_natural_policy,
)
from edu_rl.algorithms.ppo import PPOConfig, train_ppo
from edu_rl.algorithms.reinforce import ReinforceConfig, train_reinforce
from edu_rl.algorithms.tabular_learning import (
    monte_carlo_prediction,
    q_learning,
    td0_prediction,
)
from edu_rl.core.logging import ExperimentLogger
from edu_rl.envs import GridWorld


def save_run(
    name: str,
    metrics: dict[str, list[float]],
    *,
    config: Any,
    model: torch.nn.Module | None = None,
) -> Path:
    """Save human-readable CSV, TensorBoard events, a plot, and a checkpoint."""

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory = Path("runs") / f"{name}-{stamp}"
    with ExperimentLogger(directory, config) as logger:
        length = max((len(values) for values in metrics.values()), default=0)
        for step in range(length):
            row = {
                key: float(values[step])
                for key, values in metrics.items()
                if step < len(values)
            }
            logger.log(step, **row)

    if metrics:
        figure, axes = plt.subplots(
            len(metrics), 1, figsize=(8, max(3, 2.5 * len(metrics))), squeeze=False
        )
        for axis, (label, values) in zip(axes[:, 0], metrics.items(), strict=True):
            axis.plot(values)
            axis.set_title(label.replace("_", " ").title())
            axis.set_xlabel("Update")
            axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(directory / "learning_curves.png", dpi=140)
        plt.close(figure)
    if model is not None:
        torch.save(model.state_dict(), directory / "model.pt")
    print(f"Saved {name} results to {directory.resolve()}")
    return directory


def run_tabular(args: argparse.Namespace) -> None:
    env = GridWorld(seed=args.seed)
    values_vi, policy_vi = value_iteration(env.P, env.R)
    values_pi, policy_pi = policy_iteration(env.P, env.R)

    def expert(state: int, _rng: np.random.Generator) -> int:
        return int(policy_vi[state])

    episodes = 300 if args.quick else 10_000
    mc_values = monte_carlo_prediction(env, expert, episodes=episodes, seed=args.seed)
    td_values = td0_prediction(env, expert, episodes=episodes, seed=args.seed)
    q_values, returns = q_learning(env, episodes=episodes, seed=args.seed)
    print("Optimal GridWorld policy:\n" + env.format_policy(policy_vi))
    print(f"Policy/value iteration max value difference: {np.max(np.abs(values_vi-values_pi)):.3g}")
    if not np.array_equal(policy_vi, policy_pi):
        print("The displayed optimal policies differ only in argmax tie-breaking.")
    print(f"MC start value: {mc_values[0]:.3f}; TD start value: {td_values[0]:.3f}")
    print("Learned Q policy:\n" + env.format_policy(q_values.argmax(axis=1)))
    save_run("tabular", {"episode_return": returns}, config=vars(args))


def run_imitation(args: argparse.Namespace) -> None:
    env = GridWorld(seed=args.seed)
    _, expert = value_iteration(env.P, env.R)
    epochs = 20 if args.quick else 200
    bc = behavioral_cloning(
        env, expert, expert_episodes=3 if args.quick else 20, epochs=epochs, seed=args.seed
    )
    dg = dagger(
        env,
        expert,
        iterations=2 if args.quick else 8,
        learner_episodes_per_iteration=2 if args.quick else 10,
        epochs_per_iteration=10 if args.quick else 50,
        seed=args.seed,
    )
    save_run(
        "behavioral-cloning",
        {"loss": bc.losses, "evaluation_return": bc.evaluation_returns},
        config=vars(args),
        model=bc.policy,
    )
    save_run(
        "dagger",
        {
            "loss": dg.losses,
            "evaluation_return": dg.evaluation_returns,
            "dataset_size": [float(value) for value in dg.dataset_sizes],
        },
        config=vars(args),
        model=dg.policy,
    )


def run_reinforce(args: argparse.Namespace) -> None:
    variants = (
        [args.variant]
        if args.variant in {"full", "reward-to-go", "baseline"}
        else ["full", "reward-to-go", "baseline"]
    )
    for variant in variants:
        config = ReinforceConfig(
            env_id=args.env or "CartPole-v1",
            iterations=2 if args.quick else 100,
            episodes_per_batch=2 if args.quick else 10,
            reward_to_go=variant != "full",
            use_baseline=variant == "baseline",
            hidden_sizes=(16,) if args.quick else (64, 64),
            seed=args.seed,
            device=args.device,
        )
        result = train_reinforce(config)
        save_run(
            f"reinforce-{variant}",
            {
                "mean_return": result.mean_returns,
                "policy_loss": result.policy_losses,
                "value_loss": result.value_losses,
            },
            config=asdict(config),
            model=result.actor,
        )


def run_actor_critic(args: argparse.Namespace) -> None:
    estimators = (
        [args.variant]
        if args.variant in {"td0", "nstep", "gae"}
        else ["td0", "nstep", "gae"]
    )
    for estimator in estimators:
        config = ActorCriticConfig(
            env_id=args.env or "CartPole-v1",
            iterations=2 if args.quick else 150,
            episodes_per_batch=2 if args.quick else 8,
            estimator=estimator,
            hidden_sizes=(16,) if args.quick else (64, 64),
            seed=args.seed,
            device=args.device,
        )
        result = train_actor_critic(config)
        save_run(
            f"actor-critic-{estimator}",
            {
                "mean_return": result.mean_returns,
                "policy_loss": result.policy_losses,
                "value_loss": result.value_losses,
            },
            config=asdict(config),
            model=result.actor,
        )


def run_dqn(args: argparse.Namespace) -> None:
    variants = (
        [args.variant]
        if args.variant in {"standard", "double"}
        else ["standard", "double"]
    )
    for variant in variants:
        config = DQNConfig(
            env_id=args.env or "CartPole-v1",
            total_steps=200 if args.quick else 100_000,
            warmup_steps=20 if args.quick else 1_000,
            batch_size=16 if args.quick else 128,
            replay_capacity=500 if args.quick else 100_000,
            target_update_interval=25 if args.quick else 500,
            double_dqn=variant == "double",
            hidden_sizes=(32,) if args.quick else (128, 128),
            seed=args.seed,
            device=args.device,
        )
        result = train_dqn(config)
        save_run(
            f"dqn-{variant}",
            {"episode_return": result.episode_returns, "loss": result.losses},
            config=asdict(config),
            model=result.q_network,
        )


def run_ddpg(args: argparse.Namespace) -> None:
    config = DDPGConfig(
        env_id=args.env or "Pendulum-v1",
        total_steps=250 if args.quick else 100_000,
        warmup_steps=20 if args.quick else 2_000,
        batch_size=16 if args.quick else 128,
        replay_capacity=500 if args.quick else 200_000,
        hidden_sizes=(32, 32) if args.quick else (256, 256),
        seed=args.seed,
        device=args.device,
    )
    result = train_ddpg(config)
    save_run(
        "ddpg",
        {
            "episode_return": result.episode_returns,
            "actor_loss": result.actor_losses,
            "critic_loss": result.critic_losses,
        },
        config=asdict(config),
        model=result.actor,
    )


def run_ppo(args: argparse.Namespace) -> None:
    methods = [args.variant] if args.variant in {"clip", "kl"} else ["clip", "kl"]
    for method in methods:
        config = PPOConfig(
            env_id=args.env or "CartPole-v1",
            iterations=2 if args.quick else 100,
            rollout_steps=128 if args.quick else 2_048,
            update_epochs=2 if args.quick else 10,
            minibatch_size=32 if args.quick else 64,
            method=method,
            hidden_sizes=(32,) if args.quick else (64, 64),
            seed=args.seed,
            device=args.device,
        )
        result = train_ppo(config)
        save_run(
            f"ppo-{method}",
            {
                "mean_return": result.mean_returns,
                "policy_loss": result.policy_losses,
                "value_loss": result.value_losses,
                "approximate_kl": result.approximate_kls,
                "clip_fraction": result.clip_fractions,
            },
            config=asdict(config),
            model=result.actor,
        )


def run_natural(args: argparse.Namespace) -> None:
    methods = [args.variant] if args.variant in {"npg", "trpo"} else ["npg", "trpo"]
    for method in methods:
        config = NaturalPolicyConfig(
            env_id=args.env or "CartPole-v1",
            iterations=2 if args.quick else 80,
            rollout_steps=128 if args.quick else 2_048,
            method=method,
            conjugate_gradient_steps=3 if args.quick else 10,
            critic_epochs=2 if args.quick else 10,
            hidden_sizes=(16,) if args.quick else (64, 64),
            seed=args.seed,
            device=args.device,
        )
        result = train_natural_policy(config)
        save_run(
            method,
            {
                "mean_return": result.mean_returns,
                "policy_kl": result.policy_kls,
                "accepted_step_fraction": result.accepted_step_fractions,
                "value_loss": result.value_losses,
            },
            config=asdict(config),
            model=result.actor,
        )


def run_importance_sampling(args: argparse.Namespace) -> None:
    result = bandit_importance_sampling_demo(
        samples=1_000 if args.quick else 100_000, seed=args.seed
    )
    for name, value in result.items():
        print(f"{name}: {value:.6f}")
    save_run(
        "importance-sampling",
        {name: [value] for name, value in result.items()},
        config=vars(args),
    )


RUNNERS = {
    "tabular": run_tabular,
    "imitation": run_imitation,
    "reinforce": run_reinforce,
    "actor-critic": run_actor_critic,
    "dqn": run_dqn,
    "ddpg": run_ddpg,
    "ppo": run_ppo,
    "natural": run_natural,
    "importance-sampling": run_importance_sampling,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "algorithm", choices=[*RUNNERS, "all"], help="experiment family to run"
    )
    parser.add_argument(
        "--variant",
        default="all",
        help="algorithm-specific variant, or 'all' for every variant",
    )
    parser.add_argument("--quick", action="store_true", help="short wiring check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--env",
        default=None,
        help="override the default Gymnasium environment, for example Pendulum-v1",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.algorithm == "all":
        for runner in RUNNERS.values():
            runner(args)
    else:
        RUNNERS[args.algorithm](args)


if __name__ == "__main__":
    main()
