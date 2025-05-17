import numpy as np
from numba import njit

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

# You can now call these from your VecGameMasked class by passing the appropriate arrays.
