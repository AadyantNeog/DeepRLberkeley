"""A three-state MDP small enough to solve by hand.

State 0 is the start.  Action 0 moves to state 1 for no reward; action 1 ends
immediately with reward 0.4.  From state 1, action 0 reaches the terminal state
with reward 1.0 and action 1 reaches it with reward -1.0.  State 2 is absorbing.
"""

from __future__ import annotations

import numpy as np


class TinyMDP:
    n_states = 3
    n_actions = 2
    start_state = 0
    terminal_states = frozenset({2})

    def __init__(self) -> None:
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros_like(self.P)
        self.P[0, 0, 1] = 1.0
        self.P[0, 1, 2] = 1.0
        self.R[0, 1, 2] = 0.4
        self.P[1, :, 2] = 1.0
        self.R[1, 0, 2] = 1.0
        self.R[1, 1, 2] = -1.0
        self.P[2, :, 2] = 1.0
        self.state = self.start_state

    def reset(self, *, seed: int | None = None) -> tuple[int, dict]:
        del seed
        self.state = self.start_state
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        probabilities = self.P[self.state, action]
        next_state = int(np.argmax(probabilities))
        reward = float(self.R[self.state, action, next_state])
        self.state = next_state
        return next_state, reward, next_state in self.terminal_states, False, {}

