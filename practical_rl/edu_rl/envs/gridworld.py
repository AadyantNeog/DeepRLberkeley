"""A small stochastic GridWorld with an exact transition model."""

from __future__ import annotations

import numpy as np


class GridWorld:
    """A square grid with a goal, a pit, walls, and optional action slip.

    Actions are 0=up, 1=right, 2=down, 3=left.  With probability ``slip`` the
    agent executes either perpendicular action instead.  Goal and pit states are
    absorbing; entering them yields +1 and -1 respectively.
    """

    # ASCII keeps policy diagrams readable in Windows terminals whose default
    # code page cannot encode Unicode arrows.
    action_symbols = ("U", "R", "D", "L")
    action_deltas = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(
        self,
        size: int = 4,
        *,
        slip: float = 0.1,
        step_reward: float = -0.01,
        max_steps: int = 100,
        seed: int = 0,
    ) -> None:
        if size < 3:
            raise ValueError("GridWorld size must be at least 3")
        if not 0.0 <= slip <= 1.0:
            raise ValueError("slip must lie in [0, 1]")
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.start_state = 0
        self.goal_state = self.n_states - 1
        self.pit_state = size + 1
        self.terminal_states = frozenset({self.goal_state, self.pit_state})
        self.slip = slip
        self.step_reward = step_reward
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.P, self.R = self._build_model()
        self.state = self.start_state
        self.steps = 0

    def _move(self, state: int, action: int) -> int:
        if state in self.terminal_states:
            return state
        row, column = divmod(state, self.size)
        delta_row, delta_column = self.action_deltas[action]
        new_row = int(np.clip(row + delta_row, 0, self.size - 1))
        new_column = int(np.clip(column + delta_column, 0, self.size - 1))
        return new_row * self.size + new_column

    def _build_model(self) -> tuple[np.ndarray, np.ndarray]:
        transitions = np.zeros((self.n_states, self.n_actions, self.n_states))
        rewards = np.zeros_like(transitions)
        for state in range(self.n_states):
            for intended_action in range(self.n_actions):
                if state in self.terminal_states:
                    transitions[state, intended_action, state] = 1.0
                    continue
                left = (intended_action - 1) % self.n_actions
                right = (intended_action + 1) % self.n_actions
                outcomes = (
                    (intended_action, 1.0 - self.slip),
                    (left, self.slip / 2.0),
                    (right, self.slip / 2.0),
                )
                for actual_action, probability in outcomes:
                    next_state = self._move(state, actual_action)
                    transitions[state, intended_action, next_state] += probability
                    if next_state == self.goal_state:
                        reward = 1.0
                    elif next_state == self.pit_state:
                        reward = -1.0
                    else:
                        reward = self.step_reward
                    # R stores expected reward conditional on this transition.
                    rewards[state, intended_action, next_state] = reward
        return transitions, rewards

    def reset(self, *, seed: int | None = None) -> tuple[int, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.start_state
        self.steps = 0
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions})")
        probabilities = self.P[self.state, action]
        next_state = int(self.rng.choice(self.n_states, p=probabilities))
        reward = float(self.R[self.state, action, next_state])
        self.state = next_state
        self.steps += 1
        terminated = next_state in self.terminal_states
        truncated = self.steps >= self.max_steps and not terminated
        return next_state, reward, terminated, truncated, {}

    def format_policy(self, policy: np.ndarray) -> str:
        """Return a compact arrow diagram for a deterministic policy."""

        rows: list[str] = []
        for row in range(self.size):
            cells: list[str] = []
            for column in range(self.size):
                state = row * self.size + column
                if state == self.goal_state:
                    cells.append("G")
                elif state == self.pit_state:
                    cells.append("X")
                else:
                    cells.append(self.action_symbols[int(policy[state])])
            rows.append(" ".join(cells))
        return "\n".join(rows)
