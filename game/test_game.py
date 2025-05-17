import numpy as np
import torch
from game_masked import VecGameMasked


ACTION_NAMES = ["up", "down", "left", "right"]


def print_board(board):
    for row in board.reshape(5, 5):
        print(' '.join(f"{2**cell if cell else 0:4d}" for cell in row))
    print()

def test_seeded_game():
    print("Test: Seeded Game Reproducibility")
    env1 = VecGameMasked(size=2, seed=42)
    env2 = VecGameMasked(size=2, seed=42)
    env1.reset(); env2.reset()
    env1.prepare(); env2.prepare()

    for i in range(5):
        env1.step(np.array([i % 4]))
        env2.step(np.array([i % 4]))

        # first board comparison
        b1 = env1._boards[0]
        b2 = env2._boards[0]
        assert np.array_equal(b1, b2), f"Mismatch at step {i}"

        # second board comparison
        b1 = env1._boards[1]
        b2 = env2._boards[1]
        assert np.array_equal(b1, b2), f"Mismatch at step {i}"
    print("Seeded reproducibility test passed.\n")

def test_actions_left():
    print("Test: Action Left")
    env = VecGameMasked(size=1, debug=True)
    board = np.array([
        [1, 0, 1, 0, 0],
        [2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    env.reset()
    env.prepare()
    env.set_board(0, board)
    state, _ = env.observations()
    print("Before:")
    print_board(state[0])
    env.step(np.array([2]))  # left
    state, _ = env.observations()
    print("After left:")
    print_board(state[0])

def test_actions_right():
    print("Test: Action Right")
    env = VecGameMasked(size=1, debug=True)
    board = np.array([
        [1, 0, 1, 0, 0],
        [2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    env.reset()
    env.prepare()
    env.set_board(0, board)
    state, _ = env.observations()
    print("Before:")
    print_board(state[0])
    env.step(np.array([3]))  # right
    state, _ = env.observations()
    print("After right:")
    print_board(state[0])

def test_actions_up():
    print("Test: Action Up")
    env = VecGameMasked(size=1, debug=True)
    board = np.array([
        [1, 0, 1, 0, 0],
        [2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    env.reset()
    env.prepare()
    env.set_board(0, board)
    state, _ = env.observations()
    print("Before:")
    print_board(state[0])
    env.step(np.array([0]))  # up
    state, _ = env.observations()
    print("After up:")
    print_board(state[0])

def test_actions_down():
    print("Test: Action Down")
    env = VecGameMasked(size=1, debug=True)
    board = np.array([
        [1, 0, 1, 0, 0],
        [2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    env.reset()
    env.prepare()
    env.set_board(0, board)
    state, _ = env.observations()
    print("Before:")
    print_board(state[0])
    env.step(np.array([1]))  # down
    state, _ = env.observations()
    print("After down:")
    print_board(state[0])

def test_masked():
    print("Test: Custom Mask & Tile Spawn")
    env = VecGameMasked(size=1, debug=True)
    mask = np.ones((5, 5), dtype=bool)
    mask[:, 4] = False
    mask[4, :] = False
    env.set_active_mask(mask)
    env.reset()
    env.prepare()
    print("Active mask:")
    print(mask.astype(int))

    # Fill all active cells except one, all inactive cells are 0
    board = np.zeros((5, 5), dtype=np.uint8)
    # Set all active cells to 1 except (0, 0)
    for y in range(5):
        for x in range(5):
            if mask[y, x] and not (y == 0 and x == 0):
                board[y, x] = 1
    env.set_board(0, board)
    print("Board before spawn attempt:")
    print_board(env._boards[0])
    # Try to spawn a tile many times
    for i in range(100):
        # Reset the board to the same state before each spawn
        env.set_board(0, board.copy())
        env._spawn_tile(0)
        # Check that no masked cell has a tile
        for y in range(5):
            for x in range(5):
                if not mask[y, x]:
                    assert env._boards[0, y, x] == 0, f"Iteration {i}: Tile spawned in masked cell ({y},{x})!"
    print("No tile spawned in masked (inactive) cells in 100 spawn attempts. Test passed.\n")

def test_valid_actions():
    print("Test: Valid Actions")
    env = VecGameMasked(size=1, debug=True)
    mask_4x4 = np.array([
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=bool)
    print("Active mask:")
    print(mask_4x4)
    env.set_active_mask(mask_4x4)
    # Example 1: Only left is valid
    board = np.array([
        [2, 1, 2, 1, 0],
        [1, 2, 1, 2, 0],
        [2, 1, 2, 1, 0],
        [1, 2, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    env.prepare()
    env.set_board(0, board)
    env._compute_valid_actions(0)

    state, valid = env.observations()
    
    print("Board:")
    print_board(state[0])
    print("Valid actions:", valid[0])
    # You may need to adjust this expected value based on your logic
    expected = torch.tensor([False, False, False, False])  # [up, down, left, right]
    assert (valid[0] == expected).all(), f"Expected {expected}, got {valid[0]}"


    print("Valid actions test passed.\n")

def run_all_tests():
    test_seeded_game()
    test_actions_left()
    test_actions_right()
    test_actions_up()
    test_actions_down()
    test_masked()
    test_valid_actions()

if __name__ == "__main__":
    run_all_tests()
