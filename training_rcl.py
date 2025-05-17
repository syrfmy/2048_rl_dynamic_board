import numpy as np
from pathlib import Path
import logging
from typing import Any
import os.path
import sys
import torch
from pprint import pformat
import math
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from training.trainer import BaseTrainer
from game.game_masked import VecGameMasked
from training.runner import VecRunnerMasked, VecStepResult
from agent.actor_critic import NewCNNActorCriticPolicy
from agent.tensor_util import new_tensors
from training.replay import REPLAY_SPEC, ADV_SPEC
from tracking.stats import TensorStats
from agent.gae import compute_gae

MASK_4X4 = np.array([[True,  True,  True,  True,  False],
                     [True,  True,  True,  True,  False],
                     [True,  True,  True,  True,  False],
                     [True,  True,  True,  True,  False],
                     [False, False, False, False, False]], dtype=bool)

MASK_5X4 = np.array([[True,  True,  True,  True,  False],
                     [True,  True,  True,  True,  False],
                     [True,  True,  True,  True,  False],
                     [True,  True,  True,  True,  False],
                     [True,  True,  True,  True,  False]], dtype=bool)

MASK_5X5 = np.array([[True,  True,  True,  True,  True],
                     [True,  True,  True,  True,  True],
                     [True,  True,  True,  True,  True],
                     [True,  True,  True,  True,  True],
                     [True,  True,  True,  True,  True]], dtype=bool)



class ChangingMaskTrainer(BaseTrainer):
    """
    Trainer that can change the active mask mid-training. When changing the board mask, all games are terminated and metrics saved before switching mask and continuing.
    """
    def __init__(self, arguments: dict[str, Any], *, save_dir: Path, logger: logging.Logger | None = None):
        super().__init__(arguments, save_dir=save_dir, logger=logger)
        self._game_count = 2048
        self._epoches = 30
        self._stages_masks=[MASK_5X5, MASK_5X4 ,MASK_4X4]
        self._step_count = 16
        self._use_count = 2
        self._eval_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_hyperparams()

        self._game = VecGameMasked(self._game_count)

        # Start with 3x3 mask
        self._game.set_active_mask(self._stages_masks[0])
        self._tensor_stats = TensorStats()
        self._runner = VecRunnerMasked(self._game, self._step_count, sample_device=self._eval_device, tensor_stats=self._tensor_stats)
        self._policy = NewCNNActorCriticPolicy(share_encoder=True).to(self._eval_device)
        self._policy = self._policy.eval()
        self._buffers = new_tensors(REPLAY_SPEC | ADV_SPEC, (self._use_count, self._step_count, self._game_count), device=self._eval_device)
        self._buffer_step = 0
        self._save_dir = save_dir
        self.init_callbacks()
        self._pending_mask = None

    def _init_hyperparams(self):
        batch_size = 1024
        lr_factor = 1 / 2**4
        self._params_default = {
            "lr_factor": lr_factor,
            "gamma": 0.997,
            "lambda": 0.9,
            "ppo_epsilon": 0.1,
            "actor_lr": 4.0e-4 * lr_factor,
            "critic_lr": 1.0e-3 * lr_factor,
            "actor_batch_size": batch_size,
            "critic_batch_size": batch_size * 2,
            "entropy_coef": 0.00025,
            "entropy_period": 50,
            "critic_coef": 1e-5 / 2**10,
        }

    def copy(self, name: str, src: np.ndarray, ui: int, si: int, dtype: torch.dtype | None = None):
        src_tensor = torch.from_numpy(src).to(dtype=dtype)
        dst = self._buffers[name]
        dst[ui, si, ...].copy_(src_tensor)
    
    def on_stepped(self, game: VecGameMasked, result: VecStepResult, actions: torch.Tensor, action_log_probs: torch.Tensor, epoch: int):
        ui = epoch % self._use_count
        si = self._buffer_step
        self._buffer_step += 1
        self.copy("state", result["prev_state"], ui, si, torch.int8)
        self.copy("valid_actions", result["prev_valid_actions"], ui, si, torch.bool)
        self.copy("next_state", result["state"], ui, si, torch.int8)
        self.copy("next_valid_actions", result["valid_actions"], ui, si, torch.bool)
        self.copy("reward", result["reward"], ui, si, torch.float32)
        self.copy("terminated", result["terminated"], ui, si, torch.bool)
        self.copy("step", result["step"], ui, si, torch.int32)
        self.copy("total_steps", result["total_steps"], ui, si, torch.int32)
        self._buffers["action"][ui, si, ...].copy_(actions.detach())
        self._buffers["action_log_prob"][ui, si, ...].copy_(action_log_probs.detach())
    
    def _learning_params(self, epoch: int) -> dict[str, Any]:
        params = self._params_default.copy()
        params["epoch"] = epoch
        params["epoches"] = self._epoches
        params["actor_lr"] *= 32 / math.sqrt(1024 + epoch)
        params["critic_lr"] *= 32 / math.sqrt(1024 + epoch)
        params["loss_coef"] = 1
        return params

    def init_callbacks(self):
        self._runner.add_callback(VecRunnerMasked.EVENT_STEPPED, self.on_stepped)

    def print_summary(self, summary: list[tuple[int, int, float]], label: str):
        desc_entries = [
            f"({maxcell}, {count}, {int(count_per * 100)}%)"
            for maxcell, count, count_per in summary[:6]
        ]
        self.print(label, ", ".join(desc_entries))
    
    def print_tensor_stats(self):
        for name, stats in self._tensor_stats.table.items():
            self.print(f"{name:8s}", stats)
            stats.reset()

    def change_board(self, new_mask: np.ndarray):    
        # Terminate all games The training loop will reset them
        self._game._terminated[:] = True
        self._game.set_active_mask(new_mask)

    def run(self):
        self.print("masked 2048 (4x4 active area, mask can change mid-training)")
        self.print("arguments", pformat(self._arguments))
        self.print("params", pformat(self._params_default))
        self.print("model", self._policy)
        self.print(
            "extra",
            pformat({
                "use_count": self._use_count,
                "game_count": self._game_count,
                "step_count": self._step_count,
            }),
        )
        for epoch in range(self._epoches):
            self.print(f"Epoch {epoch}")
            self._buffer_step = 0

            self._runner.step_many(self._policy, self._step_count, epoch)
            compute_gae(
                self._policy,
                self._buffers,
                gamma=self._params_default["gamma"],
                lambda_=self._params_default["lambda"],
                tensor_stats=self._tensor_stats,
            )
            data = {k: torch.flatten(self._buffers[k], 0, 2) for k in self._buffers}
            self._policy.learn(
                self._learning_params(epoch),
                data,
                tensor_stats=self._tensor_stats,
                device=self._eval_device,
            )
            
            # Change mask for the third third of the training
            if epoch == self._epoches // 3:
                self.print(f"Changing mask at epoch {epoch}")
                self.change_board(self._stages_masks[1])
            if epoch == self._epoches // 3 * 2:
                self.print(f"Changing mask at epoch {epoch}")
                self.change_board(self._stages_masks[2])
                
            


            self._tensor_stats.save_csv(epoch, self._save_dir)
            self._runner._terminated_stats.saved_terminated_games(epoch, self._save_dir)
            self.print_tensor_stats()
            self.print_summary(self._runner.env.summary(), "running")
            self.print_summary(self._runner._terminated_stats.summary(), "terminated")

            
        self.print("Debug run complete.")

if __name__ == "__main__":

    now = datetime.now()
    fmt = "%Y%m%d_%H%M%S"
    save_dir = Path("runs", f"training_cl_{now.strftime(fmt)}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("training_cl")
    logger.setLevel(logging.DEBUG)
    stream = logging.FileHandler(str(save_dir / "output.log"), encoding="utf-8")
    logger.addHandler(stream)
    p = ChangingMaskTrainer.parser()
    ns = p.parse_args()
    trainer = ChangingMaskTrainer(
        vars(ns),
        save_dir=save_dir,
        logger=logger,
    )
    trainer.run()

    # Save the actor critic model and encoder
    torch.save(trainer._policy.state_dict(), save_dir / "actor_critic.pt")




