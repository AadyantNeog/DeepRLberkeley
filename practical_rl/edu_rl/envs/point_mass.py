"""Small continuous-control environments for model-based and skill learning."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def point_mass_step(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    dt: float = 0.1,
    damping: float = 0.9,
) -> np.ndarray:
    """Vectorized deterministic dynamics for state ``[x, y, vx, vy]``."""

    states = np.asarray(states, dtype=np.float32)
    actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
    positions = states[..., :2]
    velocities = states[..., 2:]
    next_velocities = damping * velocities + dt * actions
    next_positions = np.clip(positions + dt * next_velocities, -1.0, 1.0)
    hit_wall = np.abs(next_positions) >= 1.0
    next_velocities = np.where(hit_wall, 0.0, next_velocities)
    return np.concatenate([next_positions, next_velocities], axis=-1).astype(
        np.float32
    )


def point_mass_reward(states: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Dense goal reward used by planning and learning algorithms."""

    positions = np.asarray(states)[..., :2]
    return -np.linalg.norm(positions - np.asarray(goal), axis=-1)


class PointMass2D(gym.Env):
    """A bounded 2-D point mass with acceleration actions and configurable goal."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        goal: tuple[float, float] = (0.8, 0.8),
        max_steps: int = 100,
        random_goal: bool = False,
        terminate_on_goal: bool = True,
    ) -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -2.0, -2.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
        )
        self.goal = np.asarray(goal, dtype=np.float32)
        self.random_goal = random_goal
        self.terminate_on_goal = terminate_on_goal
        self.max_steps = max_steps
        self.state = np.zeros(4, dtype=np.float32)
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if options and "goal" in options:
            self.goal = np.asarray(options["goal"], dtype=np.float32)
        elif self.random_goal:
            self.goal = self.np_random.uniform(-0.9, 0.9, size=2).astype(np.float32)
        self.state = np.concatenate(
            [self.np_random.uniform(-0.1, 0.1, size=2), np.zeros(2)]
        ).astype(np.float32)
        self.steps = 0
        return self.state.copy(), {"goal": self.goal.copy()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.state = point_mass_step(self.state, action)
        self.steps += 1
        distance = float(np.linalg.norm(self.state[:2] - self.goal))
        terminated = self.terminate_on_goal and distance < 0.06
        truncated = self.steps >= self.max_steps and not terminated
        reward = -distance + (1.0 if terminated else 0.0)
        return (
            self.state.copy(),
            reward,
            terminated,
            truncated,
            {"goal": self.goal.copy(), "distance": distance},
        )


class GoalPointMass2D(PointMass2D):
    """PointMass observation augmented with the current desired goal."""

    def __init__(self, *, max_steps: int = 60) -> None:
        super().__init__(max_steps=max_steps, random_goal=True)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -2.0, -2.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float32),
        )

    def _goal_observation(self) -> np.ndarray:
        return np.concatenate([self.state, self.goal]).astype(np.float32)

    def reset(self, **kwargs: object) -> tuple[np.ndarray, dict]:
        _, info = super().reset(**kwargs)
        return self._goal_observation(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        _, reward, terminated, truncated, info = super().step(action)
        return self._goal_observation(), reward, terminated, truncated, info
