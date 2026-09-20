"""Deterministic Four Rooms navigation for transfer and hierarchical RL."""

from __future__ import annotations

from collections import deque

import numpy as np


class FourRooms:
    """A compact 7x7 Four Rooms grid with four doorways in a cross wall."""

    action_deltas = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(self, *, max_steps: int = 100, seed: int = 0) -> None:
        self.size = 7
        # Cross walls; four gaps are the doorways.
        walls = {(3, column) for column in range(self.size)} | {
            (row, 3) for row in range(self.size)
        }
        self.doorways = {(3, 1), (3, 5), (1, 3), (5, 3)}
        walls -= self.doorways
        self.cells = [
            (row, column)
            for row in range(self.size)
            for column in range(self.size)
            if (row, column) not in walls
        ]
        self.state_for_cell = {cell: index for index, cell in enumerate(self.cells)}
        self.n_states = len(self.cells)
        self.n_actions = 4
        self.start_state = self.state_for_cell[(0, 0)]
        self.goal_state = self.state_for_cell[(6, 6)]
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.state = self.start_state
        self.steps = 0

    def transition(self, state: int, action: int) -> int:
        row, column = self.cells[state]
        dr, dc = self.action_deltas[action]
        candidate = (row + dr, column + dc)
        return self.state_for_cell.get(candidate, state)

    def set_goal(self, state: int) -> None:
        self.goal_state = int(state)

    def reset(
        self, *, seed: int | None = None, start_state: int | None = None
    ) -> tuple[int, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.start_state if start_state is None else int(start_state)
        self.steps = 0
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        self.state = self.transition(self.state, int(action))
        self.steps += 1
        terminated = self.state == self.goal_state
        truncated = self.steps >= self.max_steps and not terminated
        reward = 1.0 if terminated else -0.01
        return self.state, reward, terminated, truncated, {}

    def shortest_path_policy(self, target_state: int) -> np.ndarray:
        """Compute a deterministic shortest-path policy with reverse BFS."""

        distances = np.full(self.n_states, np.inf)
        distances[target_state] = 0.0
        queue: deque[int] = deque([target_state])
        predecessors: list[list[int]] = [[] for _ in range(self.n_states)]
        for state in range(self.n_states):
            for action in range(self.n_actions):
                next_state = self.transition(state, action)
                predecessors[next_state].append(state)
        while queue:
            state = queue.popleft()
            for predecessor in predecessors[state]:
                if np.isinf(distances[predecessor]):
                    distances[predecessor] = distances[state] + 1
                    queue.append(predecessor)
        policy = np.zeros(self.n_states, dtype=np.int64)
        for state in range(self.n_states):
            policy[state] = min(
                range(self.n_actions),
                key=lambda action: distances[self.transition(state, action)],
            )
        return policy

