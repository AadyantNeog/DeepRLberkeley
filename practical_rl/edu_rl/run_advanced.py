"""Run the post-lecture-10 algorithm implementations.

Examples:
    python -m edu_rl.run_advanced sac --preset learn
    python -m edu_rl.run_advanced offline --variant iql --preset learn
    python -m edu_rl.run_advanced all --quick
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np

from edu_rl.advanced.exploration import (
    RNDConfig,
    count_bonus_q_learning,
    run_ucb,
    train_rnd,
)
from edu_rl.advanced.her import HERConfig, train_her
from edu_rl.advanced.imitation_skills import (
    DIAYNConfig,
    GAILConfig,
    SkewFitConfig,
    train_diayn,
    train_gail,
    train_skew_fit,
)
from edu_rl.advanced.model_based import (
    MBPOConfig,
    run_mpc,
    train_dyna_q,
    train_mbpo,
)
from edu_rl.advanced.offline_rl import (
    OfflineConfig,
    generate_pendulum_dataset,
    train_offline,
)
from edu_rl.advanced.preference_rl import GRPOConfig, train_grpo, train_mini_rlhf
from edu_rl.advanced.sac import SACConfig, train_sac
from edu_rl.advanced.transfer_hierarchy import (
    OptionCriticConfig,
    generalized_policy_improvement,
    learn_successor_features,
    train_option_critic,
    train_smdp_options,
)
from edu_rl.envs import FourRooms, GridWorld
from edu_rl.run import save_run


def amount(args: argparse.Namespace, quick: int, learn: int, full: int) -> int:
    preset = "quick" if args.quick else args.preset
    return {"quick": quick, "learn": learn, "full": full}[preset]


def wanted(args: argparse.Namespace, name: str) -> bool:
    return args.variant in ("all", name)


def run_sac_family(args: argparse.Namespace) -> None:
    config = SACConfig(
        total_steps=amount(args, 40, 30_000, 100_000),
        warmup_steps=amount(args, 8, 500, 1_000),
        batch_size=amount(args, 8, 128, 256),
        replay_capacity=amount(args, 128, 100_000, 200_000),
        hidden_sizes=(16,) if args.quick else (256, 256),
        seed=args.seed,
        device=args.device,
    )
    result = train_sac(config)
    save_run(
        "advanced-sac",
        {
            "episode_return": result.episode_returns,
            "actor_loss": result.actor_losses,
            "critic_loss": result.critic_losses,
            "temperature": result.temperatures,
        },
        config=asdict(config),
        model=result.agent.actor,
    )


def run_model_based(args: argparse.Namespace) -> None:
    if wanted(args, "mpc"):
        returns = run_mpc(
            episodes=amount(args, 1, 5, 20),
            horizon=amount(args, 5, 15, 25),
            seed=args.seed,
        )
        save_run("advanced-cem-mpc", {"episode_return": returns}, config=vars(args))
    if wanted(args, "dyna"):
        q_values, returns = train_dyna_q(
            GridWorld(slip=0.05),
            episodes=amount(args, 10, 1_000, 5_000),
            planning_steps=amount(args, 2, 10, 50),
            seed=args.seed,
        )
        save_run(
            "advanced-dyna-q",
            {"episode_return": returns, "maximum_q": [float(q_values.max())]},
            config=vars(args),
        )
    if wanted(args, "mbpo"):
        config = MBPOConfig(
            total_steps=amount(args, 30, 20_000, 100_000),
            warmup_steps=amount(args, 8, 500, 2_000),
            batch_size=amount(args, 4, 128, 256),
            model_train_interval=amount(args, 8, 250, 250),
            model_epochs=amount(args, 1, 20, 50),
            model_rollout_batch=amount(args, 4, 256, 1_000),
            model_rollout_horizon=amount(args, 1, 3, 5),
            ensemble_size=amount(args, 2, 5, 7),
            hidden_sizes=(8,) if args.quick else (128, 128),
            seed=args.seed,
            device=args.device,
        )
        result = train_mbpo(config)
        save_run(
            "advanced-mbpo",
            {
                "episode_return": result.episode_returns,
                "model_loss": result.model_losses,
                "model_disagreement": result.model_disagreements,
            },
            config=asdict(config),
            model=result.agent.actor,
        )


def run_offline(args: argparse.Namespace) -> None:
    dataset = generate_pendulum_dataset(
        transitions=amount(args, 256, 50_000, 250_000),
        quality="mixed",
        seed=args.seed,
    )
    for algorithm in ("sac_bc", "awac", "iql", "cql"):
        if not wanted(args, algorithm):
            continue
        config = OfflineConfig(
            algorithm=algorithm,
            gradient_steps=amount(args, 3, 30_000, 100_000),
            batch_size=amount(args, 16, 256, 512),
            hidden_sizes=(16,) if args.quick else (256, 256),
            seed=args.seed,
            device=args.device,
        )
        result = train_offline(
            dataset,
            config,
            action_low=np.array([-2.0], dtype=np.float32),
            action_high=np.array([2.0], dtype=np.float32),
        )
        save_run(
            f"advanced-offline-{algorithm}",
            {
                "actor_loss": result.actor_losses,
                "critic_loss": result.critic_losses,
                "auxiliary_loss": result.auxiliary_losses,
            },
            config=asdict(config),
            model=result.actor,
        )


def run_exploration(args: argparse.Namespace) -> None:
    if wanted(args, "ucb"):
        result = run_ucb(
            np.array([0.0, 0.2, 0.5, 1.0]),
            steps=amount(args, 100, 2_000, 20_000),
            seed=args.seed,
        )
        save_run(
            "advanced-ucb",
            {"reward": result.rewards, "cumulative_regret": result.cumulative_regret},
            config=vars(args),
        )
    if wanted(args, "count"):
        _, returns = count_bonus_q_learning(
            episodes=amount(args, 10, 500, 2_000), seed=args.seed
        )
        save_run(
            "advanced-count-bonus", {"episode_return": returns}, config=vars(args)
        )
    if wanted(args, "rnd"):
        config = RNDConfig(
            total_steps=amount(args, 30, 10_000, 50_000),
            warmup_steps=amount(args, 4, 200, 1_000),
            batch_size=amount(args, 4, 64, 128),
            hidden_sizes=(8,) if args.quick else (64, 64),
            seed=args.seed,
            device=args.device,
        )
        result = train_rnd(config)
        save_run(
            "advanced-rnd",
            {
                "extrinsic_return": result.extrinsic_returns,
                "intrinsic_reward": result.intrinsic_rewards,
            },
            config=asdict(config),
            model=result.q_network,
        )


def run_her_family(args: argparse.Namespace) -> None:
    config = HERConfig(
        episodes=amount(args, 10, 3_000, 10_000),
        n_bits=amount(args, 3, 6, 8),
        her_k=4,
        batch_size=amount(args, 8, 128, 256),
        hidden_sizes=(8,) if args.quick else (128, 128),
        seed=args.seed,
        device=args.device,
    )
    result = train_her(config)
    save_run(
        "advanced-her",
        {"success": result.success_history, "loss": result.losses},
        config=asdict(config),
        model=result.q_network,
    )


def run_skills(args: argparse.Namespace) -> None:
    if wanted(args, "gail"):
        config = GAILConfig(
            iterations=amount(args, 1, 100, 300),
            rollout_steps=amount(args, 32, 1_024, 4_096),
            discriminator_epochs=amount(args, 1, 5, 10),
            policy_epochs=amount(args, 1, 5, 10),
            expert_episodes=amount(args, 2, 50, 200),
            seed=args.seed,
            device=args.device,
        )
        result = train_gail(config)
        save_run(
            "advanced-gail",
            {
                "environment_return": result.environment_returns,
                "discriminator_loss": result.discriminator_losses,
            },
            config=asdict(config),
            model=result.actor,
        )
    if wanted(args, "diayn"):
        config = DIAYNConfig(
            total_steps=amount(args, 30, 50_000, 200_000),
            warmup_steps=amount(args, 4, 500, 2_000),
            batch_size=amount(args, 4, 128, 256),
            skill_count=amount(args, 3, 6, 10),
            hidden_sizes=(8,) if args.quick else (128, 128),
            seed=args.seed,
            device=args.device,
        )
        result = train_diayn(config)
        save_run(
            "advanced-diayn",
            {
                "intrinsic_reward": result.intrinsic_rewards,
                "discriminator_loss": result.discriminator_losses,
            },
            config=asdict(config),
            model=result.agent.actor,
        )
    if wanted(args, "skew-fit"):
        config = SkewFitConfig(
            total_steps=amount(args, 70, 50_000, 200_000),
            warmup_steps=amount(args, 4, 500, 2_000),
            batch_size=amount(args, 4, 128, 256),
            goal_update_interval=amount(args, 64, 500, 1_000),
            vae_epochs=amount(args, 1, 20, 50),
            hidden_sizes=(8,) if args.quick else (128, 128),
            seed=args.seed,
            device=args.device,
        )
        result = train_skew_fit(config)
        save_run(
            "advanced-skew-fit",
            {
                "episode_return": result.episode_returns,
                "coverage": result.coverage_history,
                "vae_loss": result.vae_losses,
            },
            config=asdict(config),
            model=result.agent.actor,
        )


def run_preference(args: argparse.Namespace) -> None:
    config = GRPOConfig(
        iterations=amount(args, 2, 300, 1_000),
        prompt_count=amount(args, 2, 8, 16),
        vocabulary_size=amount(args, 3, 6, 12),
        sequence_length=amount(args, 3, 6, 12),
        group_size=amount(args, 3, 8, 16),
        update_epochs=amount(args, 1, 4, 8),
        hidden_size=amount(args, 8, 64, 128),
        seed=args.seed,
        device=args.device,
    )
    if wanted(args, "grpo"):
        result = train_grpo(config)
        save_run(
            "advanced-grpo",
            {
                "mean_reward": result.mean_rewards,
                "policy_loss": result.policy_losses,
                "reference_kl": result.reference_kls,
            },
            config=asdict(config),
            model=result.policy,
        )
    if wanted(args, "rlhf"):
        result = train_mini_rlhf(
            config,
            preference_pairs=amount(args, 32, 5_000, 20_000),
            reward_model_epochs=amount(args, 2, 200, 1_000),
        )
        save_run(
            "advanced-mini-rlhf",
            {
                "reward_model_loss": result.reward_model_losses,
                "policy_reward": result.grpo.mean_rewards,
                "true_reward": [result.true_reward_before, result.true_reward_after],
            },
            config=asdict(config),
            model=result.grpo.policy,
        )


def run_transfer(args: argparse.Namespace) -> None:
    env = FourRooms(max_steps=100, seed=args.seed)
    if wanted(args, "successor"):
        goals = [env.state_for_cell[(0, 6)], env.state_for_cell[(6, 0)]]
        library = []
        for index, goal in enumerate(goals):
            env.set_goal(goal)
            policy = env.shortest_path_policy(goal)
            library.append(
                learn_successor_features(
                    env,
                    policy,
                    episodes=amount(args, 10, 5_000, 20_000),
                    seed=args.seed + index,
                )
            )
        new_goal = env.state_for_cell[(6, 6)]
        reward_weights = np.zeros(env.n_states)
        reward_weights[new_goal] = 1.0
        _, q_values = generalized_policy_improvement(
            np.stack(library), reward_weights
        )
        save_run(
            "advanced-successor-gpi",
            {"maximum_gpi_q": [float(q_values.max())]},
            config=vars(args),
        )
    env.set_goal(env.state_for_cell[(6, 6)])
    if wanted(args, "options"):
        result = train_smdp_options(
            env, episodes=amount(args, 5, 1_000, 5_000), seed=args.seed
        )
        save_run(
            "advanced-smdp-options",
            {
                "episode_return": result.episode_returns,
                "high_level_decisions": [float(value) for value in result.episode_decisions],
            },
            config=vars(args),
        )
    if wanted(args, "option-critic"):
        config = OptionCriticConfig(
            episodes=amount(args, 3, 2_000, 10_000),
            seed=args.seed,
            device=args.device,
        )
        result = train_option_critic(env, config)
        save_run(
            "advanced-option-critic",
            {
                "episode_return": result.episode_returns,
                "option_switches": [float(value) for value in result.option_switches],
            },
            config=asdict(config),
            model=result.network,
        )


RUNNERS = {
    "sac": run_sac_family,
    "model-based": run_model_based,
    "offline": run_offline,
    "exploration": run_exploration,
    "her": run_her_family,
    "skills": run_skills,
    "preference": run_preference,
    "transfer": run_transfer,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=[*RUNNERS, "all"])
    parser.add_argument("--variant", default="all")
    parser.add_argument("--preset", choices=("quick", "learn", "full"), default="learn")
    parser.add_argument("--quick", action="store_true", help="alias for --preset quick")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.family == "all":
        for runner in RUNNERS.values():
            runner(args)
    else:
        RUNNERS[args.family](args)


if __name__ == "__main__":
    main()
