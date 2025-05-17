from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from torch import nn

# Moved from ml2048.policy
class Policy(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def sample_actions(
        self,
        state: torch.LongTensor | torch.ByteTensor,
        valid_actions: torch.BoolTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.FloatTensor]:
        raise NotImplementedError

    @abstractmethod
    def action_logits(
        self,
        state: torch.LongTensor | torch.ByteTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    @abstractmethod
    def eval_value(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    @abstractmethod
    def learn(
        self,
        params: dict[str, Any],
        data: dict[str, torch.Tensor],
        *,
        tensor_stats: Any,
        device: Any = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError
