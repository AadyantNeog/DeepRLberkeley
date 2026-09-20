"""A sparse goal-conditioned BitFlip task used to demonstrate HER."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class BitFlipEnv(gym.Env):
    """Flip one bit per action until the current vector matches the goal."""

    def __init__(self, n_bits: int = 6, max_steps: int | None = None) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.max_steps = max_steps or n_bits * 2
        self.action_space = gym.spaces.Discrete(n_bits)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(2 * n_bits,), dtype=np.float32
        )
        self.state = np.zeros(n_bits, dtype=np.float32)
        self.goal = np.ones(n_bits, dtype=np.float32)
        self.steps = 0

    def observation(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.concatenate([state, goal]).astype(np.float32)

    @staticmethod
    def reward_for(state: np.ndarray, goal: np.ndarray) -> float:
        return 0.0 if np.array_equal(state, goal) else -1.0

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.state = self.np_random.integers(0, 2, size=self.n_bits).astype(np.float32)
        if options and "goal" in options:
            self.goal = np.asarray(options["goal"], dtype=np.float32)
        else:
            self.goal = self.np_random.integers(0, 2, size=self.n_bits).astype(
                np.float32
            )
            while np.array_equal(self.goal, self.state):
                self.goal = self.np_random.integers(0, 2, size=self.n_bits).astype(
                    np.float32
                )
        self.steps = 0
        return self.observation(self.state, self.goal), {
            "achieved_goal": self.state.copy(),
            "desired_goal": self.goal.copy(),
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.state[int(action)] = 1.0 - self.state[int(action)]
        self.steps += 1
        reward = self.reward_for(self.state, self.goal)
        terminated = reward == 0.0
        truncated = self.steps >= self.max_steps and not terminated
        return self.observation(self.state, self.goal), reward, terminated, truncated, {
            "achieved_goal": self.state.copy(),
            "desired_goal": self.goal.copy(),
        }
