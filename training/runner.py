from typing import Any, Callable, Self, Sequence
import numpy as np
import torch
from pathlib import Path
from numba import njit
from game.game_masked import VecGameMasked, VecStepResult
from agent.actor_critic import Policy
from tracking.stats import TensorStats
from event.emitter import EventEmitter


@njit
def _update_count(
    counter: np.ndarray,
    state: np.ndarray,
    terminated: np.ndarray,
):
    assert state.ndim == 2 and state.shape[-1] in (16, 25), "Bad state shape (expected 16 or 25)"
    assert terminated.ndim == 1, "Bad terminated shape"
    assert state.shape[0] == terminated.shape[0], "bad length"

    size = terminated.shape[0]
    for idx in range(size):
        if not terminated[idx]:
            continue

        maxcell = state[idx, :].max()
        counter[maxcell] += 1

@njit
def _update_terminated_stats(
    terminated_stats: np.ndarray,
    state: np.ndarray,
    terminated: np.ndarray,
):
    assert state.ndim == 2 and state.shape[-1] in (16, 25), "Bad state shape (expected 16 or 25)"
    assert terminated.ndim == 1, "Bad terminated shape"
    assert state.shape[0] == terminated.shape[0], "bad length"

    

class TerminatedGameMasked:
    """
    Despite the name, this class records the stats of terminated games.

    Note that VecGame reset a game when it is terminated.
    The output stats is shifted towards games with smaller steps.
    Use game_id to track a fixed number of games instead.
    """

    terminated_count: int

    def __init__(self):
        self.counts = np.zeros((20,), dtype=np.int32)
        self.terminated_count = 0
        self.total_steps = np.zeros((20,), dtype=np.int64)
        self.terminated_games = []

    def reset(self):
        self.counts.fill(0)
        self.terminated_count = 0
        self.total_steps.fill(0)
        self.terminated_games.clear()

    def on_stepped(
        self,
        game: VecGameMasked,  # VecGameMasked
        result: VecStepResult,
        epoch=None,  # Optionally pass epoch
    ):
        _update_count(self.counts, result["state"], result["terminated"])
        self.terminated_count += np.sum(result["terminated"])
        # Save info about each terminated game
        terminated_indices = np.where(result["terminated"])[0]

        for idx in terminated_indices:
            # Try to get game_id from game, fallback to idx
            max_cell = 2**int(np.max(result["state"][idx]))

            # sum of 2**(tile value) for all tiles in the board
            tile_exponents = result["state"][idx]
            tile_values = np.power(2, tile_exponents, dtype=np.int64)
            # for masked tiles, set value to 0
            tile_values[~game._active_mask.flatten()] = 0
            total_tile = tile_values.sum()

            # Try to get total_steps from game or result
            total_steps = int(result["total_steps"][idx])
            # print("Terminated game: ", idx, " epoch: ", epoch, " max cell: ", max_cell, " total steps: ", total_steps)
            self.terminated_games.append((epoch, max_cell, total_steps, total_tile))

    def summary(self) -> list[tuple[int, int, ...]]:
        total = self.counts.sum()
        entries = []
        for power in range(16, 0, -1):
            count = self.counts[power].item()
            if count == 0:
                continue

            maxcell = 2**power
            entries.append((maxcell, count, count / total))

        return entries

    def saved_terminated_games(self, epoch: int, save_dir: Path):
        if epoch == 0:
            with open(save_dir / "terminated_games.csv", "w") as f:
                f.write("epoch,max_cell,total_steps,total_tile\n")
        # === Save CSV ===
        with open(save_dir / "terminated_games.csv", "a") as f:
            for _, max_cell, total_steps, total_tile in self.terminated_games:
                f.write(f"{epoch},{max_cell},{total_steps},{total_tile}\n")

        # Reset terminated games
        self.terminated_games.clear()
            
    @classmethod
    def combine(cls, seq: Sequence[Self]) -> Self:
        counts = np.sum([s.counts for s in seq], axis=0)
        terminated_count = sum([s.terminated_count for s in seq])
        result = cls()
        result.counts = counts
        result.terminated_count = terminated_count

        return result


class VecRunnerMasked:
    """
    Run multiple 5x5 masked games together (VecGame5x5Masked).
    Memorize the last N steps.
    """
    EVENT_PREPARED: str = "prepared"
    EVENT_STEPPED: str = "stepped"

    def __init__(self, env: VecGameMasked, capacity: int, *, sample_device: torch.device | str | None = None, tensor_stats: TensorStats | None = None):
        self.env = env
        self.sample_device = sample_device
        self._emitter = EventEmitter()
        self._vec_size = self.env._size
        self._capacity = capacity
        self._game_count = self._vec_size
        self._games: dict[int, Any] = {}
        self._game_id = np.arange(self._vec_size, dtype=np.int64)
        self._step_ids = np.zeros((self._vec_size,), dtype=np.int64)
        self._shape = (capacity, self._vec_size)
        self._terminated_stats = TerminatedGameMasked()
        self._tensor_stats = tensor_stats
        self._last_obs = None
        self._last_valid_actions = None
        self._last_result = None

    def add_callback(self, event: str, fn: Callable[..., Any]):
        assert event in {self.EVENT_STEPPED, self.EVENT_PREPARED}
        self._emitter.add_listener(event, fn)

    def step_once(
        self,
        policy: Policy,
        epoch: int,
    ):
        # prepare
        (new_indices,) = self.env.prepare()

        
        state, valid_actions = self.env.observations()

        state = torch.from_numpy(state)
        state = state.to(self.sample_device, torch.long)

        valid_actions = torch.from_numpy(valid_actions)
        valid_actions = valid_actions.to(self.sample_device, torch.bool)

        with torch.no_grad():
            sample_actions, sample_log_probs = policy.sample_actions(
                state,
                valid_actions,
            )
            del state, valid_actions

        result = self.env.step(sample_actions.cpu().numpy())
        self._emitter.emit(
            self.EVENT_STEPPED,
            (self.env, result, sample_actions, sample_log_probs, epoch),
        )
        self._terminated_stats.on_stepped(self.env, result, epoch)

    def step_many(self, policy: Policy, count: int, epoch: int):
        # === Reset terminated stats ===
        self._terminated_stats.reset()
        for _ in range(count):
            self.step_once(policy, epoch)
        
        

        

