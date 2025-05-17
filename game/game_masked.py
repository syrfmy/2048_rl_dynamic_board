import numpy as np
import torch
from numba import njit
from typing import Callable, Optional
from typing_extensions import TypedDict
 

class VecStepResult(TypedDict):
    state: np.ndarray
    valid_actions: np.ndarray
    step: np.ndarray
    merged: np.ndarray
    reward: np.ndarray
    score: np.ndarray
    terminated: np.ndarray
    invalid: np.ndarray
    total_steps: np.ndarray
    prev_state: np.ndarray
    prev_valid_actions: np.ndarray

class VecGameMasked:
    

    """
    Vectorized 2048 environment with static 5x5 board and dynamic active area via masking.
    Masked (inactive) cells act as walls: tiles cannot move, merge, or spawn there.
    """
    BOARD_SHAPE = (5, 5)
    BOARD_SIZE = 25
    ACTIONS = 4  # up, down, left, right

    def __init__(self, size: int, reward_fn: Optional[Callable] = None, *, two_prob: float = 0.8, debug: bool = False, seed: int | None = None):
        """
        Game Loop Overview (with main function calls):
        - Initialization: Each environment is a 5x5 board, optionally with a mask to set active/inactive (wall) cells.
            - Performed in __init__ and reset() methods.
        - At each step (step()):
            1. The agent selects an action (up, down, left, right) for each environment.
            2. The environment moves and merges tiles according to 2048 rules, but only in active cells.
                - Uses _move(), which calls _push_line() for each row/column.
            3. The merged mask is updated to track which tiles merged.
                - _move() returns merged board; step() stores it in self._merged.
            4. The reward is calculated (by default: sum of merged tile values + potential-based reward for top-left cell).
                - step() calls self._reward_fn (default is _default_reward_fn).
            5. If a move is valid, a new tile spawns in a random empty active cell.
                - step() calls _spawn_tile().
            6. The mask prevents movement, merging, or spawning in inactive cells.
                - Mask is set via set_active_mask(), and respected by _move()/_push_line().
            7. The game ends for an environment when no valid actions remain.
                - step() updates self._terminated and self._valid_actions using _compute_valid_actions().
        - The environment supports batch (vectorized) operation for efficient RL training.
        """

        self._size = size
        self.size = size  # for compatibility
        self.two_prob = two_prob
        # Use default reward function if none provided
        self._reward_fn = reward_fn if reward_fn is not None else self._default_reward_fn
        self.debug = debug
        self._boards = np.zeros((size, 5, 5), dtype=np.uint8)
        self._terminated = np.ones(size, dtype=bool)  # match VecGame: start as terminated
        self._score = np.zeros(size, dtype=np.float32)
        self._reward = np.zeros(size, dtype=np.float32)
        self._merged = np.zeros((size, 5, 5), dtype=np.uint8)
        self._valid_actions = np.zeros((size, 4), dtype=bool)
        self._invalid = np.zeros(size, dtype=bool)
        self._step = np.zeros(size, dtype=np.int32)
        self._id = np.arange(size, dtype=np.int32)
        self._prev_state = np.zeros((size, 5, 5), dtype=np.uint8)
        self._prev_valid_actions = np.zeros((size, 4), dtype=bool)
        self._active_mask = np.ones((5, 5), dtype=bool)  # All cells active by default
        self._game_count = 0
        self._generators = [np.random.RandomState((seed if seed is not None else 0) + i) for i in range(size)]
        self._total_steps = np.zeros(size, dtype=np.int64)
        self.reset()


    def _default_reward_fn(self, board, prev_board, merged):
        """
        Default reward: sum of 2**tile for all merged tiles in the move,
        plus a potential-based reward shaping on the top-left cell (like reward_fn_improved).
        merged: (5,5) array, 1 where a merge occurred.
        """
        reward = float(np.sum((2 ** board) * (merged == 1)))
        extra = np.float32(0)
        factor = 64
        # subtract previous top-left value
        prev_val = int(prev_board[0, 0])
        if prev_val != 0:
            extra -= factor * (1 << prev_val)
        # add new top-left value
        val = int(board[0, 0])
        if val != 0:
            extra += factor * (1 << val)
        return reward + extra


    def set_active_mask(self, mask: np.ndarray):
        assert mask.shape == (5, 5)
        self._active_mask = mask.astype(bool)

    def reset(self, seed: Optional[int] = None):
        # Optionally use seed for np.random
        # if seed is not None:  for the reset it will not reset the generator
        #     self._generator.seed(seed)
        self._boards.fill(0)
        self._terminated.fill(True)
        self._score.fill(0)
        self._reward.fill(0)
        self._merged.fill(0)
        self._valid_actions.fill(False)
        self._invalid.fill(False)
        self._step.fill(0)
        self._id = np.arange(self._size, dtype=np.int32)
        self._prev_state.fill(0)
        self._prev_valid_actions.fill(False)
        self._game_count = 0
        self._total_steps.fill(0)
        # Mark all as terminated, so prepare() will re-init

    def prepare(self) -> tuple[np.ndarray]:
        # Prepare terminated games for a new episode
        indices = np.flatnonzero(self._terminated)
        for idx in indices:
            self._boards[idx].fill(0)
            self._score[idx] = 0
            self._reward[idx] = 0
            self._merged[idx].fill(0)
            self._valid_actions[idx].fill(False)
            self._invalid[idx] = False
            self._step[idx] = 0
            self._id[idx] = self._game_count
            self._game_count += 1
            self._prev_state[idx].fill(0)
            self._prev_valid_actions[idx].fill(False)
            # Inline: spawn two tiles for each reset environment
            self._spawn_tile(idx)
            self._spawn_tile(idx)
            self._compute_valid_actions(idx)
            self._terminated[idx] = False
            self._total_steps[idx] = 0 # reset total steps because we are starting a new episode
        return (indices,)

    def observations(self) -> tuple[np.ndarray, np.ndarray]:
        # Return boards as (N, 25) instead of (N, 5, 5)

        return self._boards.reshape(self._size, 25), self._valid_actions

    def summary(self) -> list:
        maxcell = np.max(self._boards, axis=(1, 2))
        values, counts = np.unique(maxcell, return_counts=True)
        total = counts.sum()
        entries = [
            (2 ** int(maximum), int(count), float(count) / total)
            for maximum, count in zip(values, counts)
        ]
        entries.sort(key=lambda s: s[0], reverse=True)
        return entries



    def _spawn_tile(self, idx: int):
        board = self._boards[idx]
        # Only spawn in empty, active cells
        empties = np.argwhere((board == 0) & self._active_mask)
        if len(empties) == 0:
            if self.debug:
                print(f"DEBUG: No empty active cells available to spawn for board {idx}")
            return
        rng = self._generators[idx]
        y, x = empties[rng.randint(len(empties))]
        assert self._active_mask[y, x], f"Trying to spawn on inactive cell ({y},{x})!"
        board[y, x] = 1 if rng.random() < self.two_prob else 2


    def _compute_valid_actions(self, idx: int):
        board = self._boards[idx]
        mask = self._active_mask
        valid = compute_valid_actions_njit(board, mask)
        if self.debug:
            print(f"DEBUG: Valid actions for board {idx}: {valid}")
        self._valid_actions[idx] = valid

    # _line_can_move is now handled by njit_fn.line_can_move_njit

    def step(self, actions: np.ndarray) -> VecStepResult:
        """
        Perform a step for each environment in the batch given actions. Matches VecGame API.
        Returns a dict with all relevant fields (state, reward, etc).
        Adds debug prints and improved invalid action handling.
        """
        np.copyto(self._prev_state, self._boards)
        np.copyto(self._prev_valid_actions, self._valid_actions)
        for i, action in enumerate(actions):
            # Skip terminated environments
            if self._terminated[i]:
                if self.debug:
                    print(f"DEBUG: Env {i} is terminated; skipping action {action}.")
                self._invalid[i] = True
                continue

            # Skip invalid actions ()
            if not self._valid_actions[i, action]:
                if self.debug:
                    print(f"DEBUG: Invalid action {action} in env {i}. Valid actions: {self._valid_actions[i]}")
                self._invalid[i] = True
                continue


            moved, reward, merged = self._move(i, action)
            self._merged[i] = merged  # Always update merged board for this env
            # Use custom reward function if provided
            if self._reward_fn is not None:
                reward = self._reward_fn(self._boards[i], self._prev_state[i], self._merged[i])
            self._score[i] += reward
            self._reward[i] = reward
            self._step[i] += 1
            self._total_steps[i] += 1

            if moved:
                self._spawn_tile(i)
            self._compute_valid_actions(i)
            self._terminated[i] = not self._valid_actions[i].any()
            self._invalid[i] = not (moved and self._valid_actions[i, action])

            

        if self.debug:
            print(f"Previous Board,  New Board, Merged Board")
            for i in range(self._size):
                for j in range(5):
                    print(f"{self._prev_state[i][j]}  |  {self._boards[i][j]}  |  {self._merged[i][j]}")
                print("\n")

        return VecStepResult(
            state=self._boards.reshape(self._size, 25),
            valid_actions=self._valid_actions,
            merged=self._merged,
            step=self._step,
            reward=self._reward,
            score=self._score,
            terminated=self._terminated,
            invalid=self._invalid,
            total_steps=self._total_steps,
            prev_state=self._prev_state.reshape(self._size, 25),
            prev_valid_actions=self._prev_valid_actions,
        )

    def _move(self, idx: int, action: int):
        board = self._boards[idx]
        mask = self._active_mask
        # Use the njit-accelerated move function
        moved, reward, merged_board = move_njit(board, mask, action)
        return moved, reward, merged_board


    # _push_line is now handled by njit_fn.push_line_njit


    # TESTING ONLY: Set the board state for a specific environment index.
    def set_board(self, idx: int, board: np.ndarray):
        """
        Set the board of environment idx to the given 5x5 array. For testing only.
        """
        assert board.shape == (5, 5)
        self._boards[idx] = board.astype(np.uint8)
    


# --- Numba-accelerated functions ---
@njit(cache=True)
def push_line_njit(line, mask, reverse=False):
    indices = np.arange(5)[mask]
    vals = line[mask]
    original_line = line.copy()
    if reverse:
        vals = vals[::-1]
        indices = indices[::-1]
    compacted = []
    for v in vals:
        if v != 0:
            compacted.append(v)
    merged = np.zeros(5, dtype=np.uint8)
    out = np.zeros(5, dtype=line.dtype)
    reward = 0
    i = 0
    w = 0
    while i < len(compacted):
        if i + 1 < len(compacted) and compacted[i] == compacted[i + 1]:
            out[indices[w]] = compacted[i] + 1
            reward += 2 ** (compacted[i] + 1)
            merged[indices[w]] = 1
            w += 1
            i += 2
        else:
            out[indices[w]] = compacted[i]
            w += 1
            i += 1
    for j in range(w, len(indices)):
        out[indices[j]] = 0
    for j in range(5):
        if not mask[j]:
            out[j] = original_line[j]
    return out, reward, merged

@njit(cache=True)
def line_can_move_njit(line, mask, reverse=False):
    out, _, _ = push_line_njit(line, mask, reverse)
    for i in range(5):
        if mask[i] and line[i] != out[i]:
            return True
    return False

@njit(cache=True)
def move_njit(board, mask, action):
    moved = False
    reward = 0
    merged_board = np.zeros((5, 5), dtype=np.uint8)
    if action == 0:
        for x in range(5):
            col = board[:, x]
            msk = mask[:, x]
            new_col, r, m = push_line_njit(col, msk, False)
            if not np.all(col == new_col):
                board[:, x] = new_col
                moved = True
                reward += r
            merged_board[:, x] = m
    elif action == 1:
        for x in range(5):
            col = board[:, x]
            msk = mask[:, x]
            new_col, r, m = push_line_njit(col, msk, True)
            if not np.all(col == new_col):
                board[:, x] = new_col
                moved = True
                reward += r
            merged_board[:, x] = m
    elif action == 2:
        for y in range(5):
            row = board[y, :]
            msk = mask[y, :]
            new_row, r, m = push_line_njit(row, msk, False)
            if not np.all(row == new_row):
                board[y, :] = new_row
                moved = True
                reward += r
            merged_board[y, :] = m
    elif action == 3:
        for y in range(5):
            row = board[y, :]
            msk = mask[y, :]
            new_row, r, m = push_line_njit(row, msk, True)
            if not np.all(row == new_row):
                board[y, :] = new_row
                moved = True
                reward += r
            merged_board[y, :] = m
    return moved, reward, merged_board

@njit(cache=True)
def compute_valid_actions_njit(board, mask):
    valid = np.zeros(4, dtype=np.bool_)
    for direction in range(4):
        if direction == 0:
            for x in range(5):
                col = board[:, x]
                msk = mask[:, x]
                if line_can_move_njit(col, msk, False):
                    valid[0] = True
                    break
        elif direction == 1:
            for x in range(5):
                col = board[:, x]
                msk = mask[:, x]
                if line_can_move_njit(col, msk, True):
                    valid[1] = True
                    break
        elif direction == 2:
            for y in range(5):
                row = board[y, :]
                msk = mask[y, :]
                if line_can_move_njit(row, msk, False):
                    valid[2] = True
                    break
        elif direction == 3:
            for y in range(5):
                row = board[y, :]
                msk = mask[y, :]
                if line_can_move_njit(row, msk, True):
                    valid[3] = True
                    break
    return valid