"""Run the remaining lecture algorithms added after the first two code phases.

Examples:
    python -m edu_rl.run_remaining generative --variant flow --preset learn
    python -m edu_rl.run_remaining offline --variant idql --quick
    python -m edu_rl.run_remaining all --quick
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import torch

from edu_rl.advanced.continuous_extensions import (
    ContinuousSFConfig,
    TD3Config,
    train_continuous_successor_features,
    train_td3,
)
from edu_rl.advanced.exploration_extensions import (
    goal_conditioned_empowerment,
    run_cts,
    train_pseudocount_q_learning,
)
from edu_rl.advanced.generative_models import (
    AutoregressiveContinuousPolicy,
    ConditionalDiffusionPolicy,
    ConditionalVAE,
    SequentialVAE,
    VAE,
    coordinate_ascent_gaussian_vi,
    gaussian_mixture_em,
    gaussian_vae_loss,
    train_flow_matching,
    train_reflow,
)
from edu_rl.advanced.maxent_irl import (
    demonstrate_policy,
    four_rooms_dynamics,
    guided_cost_learning,
    maximum_entropy_irl,
    train_soft_q_learning,
)
from edu_rl.advanced.model_based_extensions import (
    MVEConfig,
    ModelBasedOfflineConfig,
    differentiable_plan,
    generate_point_mass_dataset,
    random_shooting_action,
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
from edu_rl.advanced.offline_rl import generate_pendulum_dataset
from edu_rl.envs import FourRooms, GridWorld
from edu_rl.run import save_run


def amount(args: argparse.Namespace, quick: int, learn: int, full: int) -> int:
    preset = "quick" if args.quick else args.preset
    return {"quick": quick, "learn": learn, "full": full}[preset]


def wanted(args: argparse.Namespace, name: str) -> bool:
    return args.variant in ("all", name)


def _train_vae_model(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    steps: int,
    conditions: torch.Tensor | None = None,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses: list[float] = []
    for _ in range(steps):
        if conditions is None:
            reconstruction, mean, log_variance = model(inputs)
        else:
            reconstruction, mean, log_variance = model(conditions, inputs)
        loss, _, _ = gaussian_vae_loss(
            reconstruction, inputs, mean, log_variance, beta=0.1
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def run_generative(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    steps = amount(args, 3, 2_000, 10_000)
    examples = amount(args, 64, 2_000, 10_000)
    mixture = np.concatenate(
        [rng.normal(-1.5, 0.3, (examples // 2, 2)), rng.normal(1.5, 0.4, (examples - examples // 2, 2))]
    ).astype(np.float32)
    if wanted(args, "cavi"):
        result = coordinate_ascent_gaussian_vi(
            np.array([[2.0, 0.6], [0.6, 1.5]]), np.array([1.0, -0.5]),
            iterations=amount(args, 5, 50, 200),
        )
        save_run("remaining-cavi", {"mean_error": result.errors}, config=vars(args))
    if wanted(args, "em"):
        result = gaussian_mixture_em(
            mixture, 2, iterations=amount(args, 3, 50, 200), seed=args.seed
        )
        save_run(
            "remaining-gmm-em", {"log_likelihood": result.log_likelihoods}, config=vars(args)
        )
    data = torch.as_tensor(mixture, device=device)
    if wanted(args, "vae"):
        model = VAE(2, 2, hidden_size=amount(args, 8, 64, 128)).to(device)
        losses = _train_vae_model(model, data, steps=steps)
        save_run("remaining-vae", {"negative_elbo": losses}, config=vars(args), model=model)
    observations = torch.randn(examples, 2, device=device)
    actions = torch.tanh(
        torch.stack([observations[:, 0] + observations[:, 1], observations[:, 0] - observations[:, 1]], dim=-1)
    )
    if wanted(args, "cvae"):
        model = ConditionalVAE(2, 2, 2, hidden_size=amount(args, 8, 64, 128)).to(device)
        losses = _train_vae_model(model, actions, steps=steps, conditions=observations)
        save_run("remaining-cvae", {"negative_elbo": losses}, config=vars(args), model=model)
    if wanted(args, "sequential-vae"):
        time = torch.linspace(0, 2 * torch.pi, amount(args, 4, 20, 50), device=device)
        phases = torch.rand(examples, 1, device=device) * 2 * torch.pi
        sequence = torch.sin(time[None, :] + phases)[:, :, None]
        model = SequentialVAE(1, 2, hidden_size=amount(args, 8, 64, 128)).to(device)
        losses = _train_vae_model(model, sequence, steps=steps)
        save_run(
            "remaining-sequential-vae", {"negative_elbo": losses}, config=vars(args), model=model
        )
    if wanted(args, "flow"):
        result = train_flow_matching(
            mixture, gradient_steps=steps, batch_size=min(64, examples),
            hidden_sizes=(amount(args, 8, 64, 128),), seed=args.seed, device=args.device,
        )
        save_run("remaining-flow-matching", {"loss": result.losses}, config=vars(args), model=result.field)
    if wanted(args, "reflow"):
        first, second = train_reflow(
            mixture, first_stage_steps=steps, reflow_steps=steps,
            batch_size=min(64, examples), hidden_sizes=(amount(args, 8, 64, 128),),
            seed=args.seed, device=args.device,
        )
        save_run(
            "remaining-reflow", {"first_stage_loss": first.losses, "reflow_loss": second.losses},
            config=vars(args), model=second.field,
        )
    if wanted(args, "diffusion"):
        policy = ConditionalDiffusionPolicy(
            2, 2, diffusion_steps=amount(args, 3, 20, 50),
            hidden_sizes=(amount(args, 8, 64, 128),),
        ).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        losses: list[float] = []
        for _ in range(steps):
            loss = policy.loss(observations, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        save_run("remaining-diffusion-policy", {"denoising_loss": losses}, config=vars(args), model=policy)
    if wanted(args, "autoregressive"):
        policy = AutoregressiveContinuousPolicy(
            2, 2, (amount(args, 8, 64, 128),)
        ).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        losses: list[float] = []
        for _ in range(steps):
            loss = -policy.log_prob(observations, actions.clamp(-0.999, 0.999)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        save_run("remaining-autoregressive-policy", {"negative_log_likelihood": losses}, config=vars(args), model=policy)


def _expert_problem(args: argparse.Namespace) -> tuple[FourRooms, np.ndarray, list[np.ndarray]]:
    env = FourRooms(max_steps=60, seed=args.seed)
    dynamics = four_rooms_dynamics(env)
    # Treat the demonstrated goal as absorbing in the finite-horizon IRL model.
    dynamics[env.goal_state] = 0.0
    dynamics[env.goal_state, :, env.goal_state] = 1.0
    deterministic = env.shortest_path_policy(env.goal_state)
    demonstrations = demonstrate_policy(
        env, deterministic, trajectories=amount(args, 3, 20, 100),
        horizon=amount(args, 12, 40, 80), seed=args.seed,
    )
    horizon = amount(args, 12, 40, 80)
    demonstrations = [
        np.pad(path[:horizon], (0, max(0, horizon - len(path))), constant_values=path[-1])
        for path in demonstrations
    ]
    return env, dynamics, demonstrations


def run_maxent(args: argparse.Namespace) -> None:
    env, dynamics, demonstrations = _expert_problem(args)
    if wanted(args, "soft-q"):
        result = train_soft_q_learning(
            env, episodes=amount(args, 5, 2_000, 10_000), seed=args.seed
        )
        save_run("remaining-soft-q", {"episode_return": result.episode_returns}, config=vars(args))
    features = np.eye(env.n_states, dtype=np.float64)
    if wanted(args, "maxent-irl"):
        result = maximum_entropy_irl(
            dynamics, features, demonstrations, env.start_state,
            horizon=len(demonstrations[0]), iterations=amount(args, 3, 200, 1_000),
            learning_rate=0.05,
        )
        save_run("remaining-maxent-irl", {"feature_error": result.feature_errors}, config=vars(args))
    if wanted(args, "gcl"):
        result = guided_cost_learning(
            dynamics, features, demonstrations, env.start_state,
            horizon=len(demonstrations[0]), iterations=amount(args, 2, 100, 500),
            sampled_trajectories=amount(args, 4, 64, 256), seed=args.seed,
        )
        save_run("remaining-guided-cost-learning", {"reward_loss": result.losses}, config=vars(args))


def run_model_based(args: argparse.Namespace) -> None:
    state = np.zeros(4, dtype=np.float32)
    goal = np.array([0.8, 0.8], dtype=np.float32)
    if wanted(args, "shooting"):
        action = random_shooting_action(
            state, goal, horizon=amount(args, 3, 15, 30),
            population_size=amount(args, 32, 1_024, 4_096), seed=args.seed,
        )
        save_run("remaining-random-shooting", {"action_norm": [float(np.linalg.norm(action))]}, config=vars(args))
    if wanted(args, "differentiable-planning"):
        _, objectives = differentiable_plan(
            state, goal, horizon=amount(args, 3, 20, 40),
            optimization_steps=amount(args, 3, 200, 1_000), seed=args.seed, device=args.device,
        )
        save_run("remaining-differentiable-planning", {"predicted_return": objectives}, config=vars(args))
    if wanted(args, "mve"):
        config = MVEConfig(
            transitions=amount(args, 64, 10_000, 50_000),
            gradient_steps=amount(args, 2, 5_000, 20_000),
            model_epochs=amount(args, 1, 20, 50), rollout_horizon=amount(args, 1, 3, 5),
            batch_size=amount(args, 8, 128, 256), hidden_sizes=(amount(args, 8, 128, 256),),
            seed=args.seed, device=args.device,
        )
        result = train_model_value_expansion(config)
        save_run("remaining-mve", {"value_loss": result.value_losses, "model_loss": result.model_losses}, config=asdict(config), model=result.value_network)
    for algorithm in ("mopo", "combo"):
        if not wanted(args, algorithm):
            continue
        config = ModelBasedOfflineConfig(
            algorithm=algorithm, dataset_transitions=amount(args, 64, 20_000, 100_000),
            bootstrap_steps=amount(args, 2, 10_000, 50_000),
            gradient_steps=amount(args, 2, 30_000, 100_000), model_epochs=amount(args, 1, 20, 50),
            model_rollout_batch=amount(args, 8, 2_000, 10_000), model_rollout_horizon=amount(args, 1, 3, 5),
            batch_size=amount(args, 8, 256, 512), hidden_sizes=(amount(args, 8, 128, 256),),
            seed=args.seed, device=args.device,
        )
        result = train_model_based_offline(config)
        save_run(
            f"remaining-{algorithm}",
            {"critic_loss": result.learner.critic_losses, "model_loss": result.model_losses, "model_uncertainty": result.mean_uncertainties},
            config=asdict(config), model=result.learner.actor,
        )


def run_offline(args: argparse.Namespace) -> None:
    dataset = generate_pendulum_dataset(
        transitions=amount(args, 64, 50_000, 250_000), quality="mixed", seed=args.seed
    )
    low = np.array([-2.0], dtype=np.float32)
    high = np.array([2.0], dtype=np.float32)
    if wanted(args, "brac"):
        config = BRACConfig(
            behavior_steps=amount(args, 2, 10_000, 50_000), gradient_steps=amount(args, 2, 30_000, 100_000),
            batch_size=amount(args, 8, 256, 512), hidden_sizes=(amount(args, 8, 256, 512),),
            seed=args.seed, device=args.device,
        )
        result = train_brac(dataset, config, action_low=low, action_high=high)
        save_run("remaining-brac", {"actor_loss": result.actor_losses, "critic_loss": result.critic_losses, "behavior_kl": result.divergences}, config=asdict(config), model=result.actor)
    for algorithm in ("idql", "fql", "diffusion-steering"):
        if not wanted(args, algorithm):
            continue
        config = GenerativeOfflineConfig(
            algorithm=algorithm, critic_steps=amount(args, 2, 30_000, 100_000),
            generative_steps=amount(args, 2, 30_000, 100_000), actor_steps=amount(args, 2, 10_000, 50_000),
            batch_size=amount(args, 8, 256, 512), diffusion_steps=amount(args, 3, 20, 50),
            hidden_sizes=(amount(args, 8, 256, 512),), seed=args.seed, device=args.device,
        )
        result = train_generative_offline(dataset, config, action_low=low, action_high=high)
        observation = torch.as_tensor(dataset.observations[:2], device=args.device)
        if algorithm == "idql":
            sampled = sample_idql_actions(result, observation, candidate_count=amount(args, 2, 32, 128))
        elif algorithm == "diffusion-steering":
            sampled = sample_diffusion_steered_actions(result, observation)
        else:
            with torch.no_grad():
                sampled = result.one_step_actor(observation)  # type: ignore[operator]
        save_run(
            f"remaining-{algorithm}",
            {"generative_loss": result.generative_losses, "actor_loss": result.actor_losses, "sample_action_norm": [float(sampled.detach().norm(dim=1).mean())]},
            config=asdict(config),
            model=result.diffusion or result.one_step_actor or result.flow.field,  # type: ignore[union-attr]
        )


def run_exploration(args: argparse.Namespace) -> None:
    if wanted(args, "pseudo-count"):
        _, returns, bonuses = train_pseudocount_q_learning(
            GridWorld(slip=0.05), episodes=amount(args, 5, 1_000, 5_000), seed=args.seed
        )
        save_run("remaining-density-pseudocount", {"episode_return": returns, "intrinsic_bonus": bonuses}, config=vars(args))
    if wanted(args, "cts"):
        sequence = np.tile([0, 0, 1, 1], amount(args, 4, 100, 1_000))
        result = run_cts(sequence, maximum_depth=amount(args, 2, 8, 12))
        save_run("remaining-cts", {"surprisal": result.surprisals, "pseudo_count": result.pseudo_counts}, config=vars(args))
    if wanted(args, "empowerment"):
        env = FourRooms(seed=args.seed)
        capacities = [
            goal_conditioned_empowerment(env, state, horizon=amount(args, 2, 3, 4))[0]
            for state in range(env.n_states)
        ]
        save_run("remaining-empowerment", {"state_capacity": capacities}, config=vars(args))


def run_continuous(args: argparse.Namespace) -> None:
    if wanted(args, "td3"):
        config = TD3Config(
            total_steps=amount(args, 20, 100_000, 300_000), warmup_steps=amount(args, 4, 2_000, 10_000),
            batch_size=amount(args, 4, 128, 256), replay_capacity=amount(args, 64, 200_000, 1_000_000),
            hidden_sizes=(amount(args, 8, 256, 512),), seed=args.seed, device=args.device,
        )
        result = train_td3(config)
        save_run("remaining-td3", {"episode_return": result.episode_returns, "actor_loss": result.actor_losses, "critic_loss": result.critic_losses}, config=asdict(config), model=result.actor)
    if wanted(args, "continuous-sf"):
        config = ContinuousSFConfig(
            total_steps=amount(args, 20, 50_000, 200_000), warmup_steps=amount(args, 4, 500, 2_000),
            batch_size=amount(args, 4, 128, 256), replay_capacity=amount(args, 64, 100_000, 500_000),
            hidden_sizes=(amount(args, 8, 128, 256),), seed=args.seed, device=args.device,
        )
        result = train_continuous_successor_features(config)
        save_run("remaining-continuous-successor-features", {"successor_loss": result.losses, "reward_prediction_loss": result.reward_prediction_losses}, config=asdict(config), model=result.successor)


RUNNERS = {
    "generative": run_generative,
    "maxent": run_maxent,
    "model-based": run_model_based,
    "offline": run_offline,
    "exploration": run_exploration,
    "continuous": run_continuous,
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
