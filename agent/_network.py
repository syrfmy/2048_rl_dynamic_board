import math

import torch
from torch import nn
from torch.nn import functional as F

NUM_CELLS = 25  # 5x5 board
NUM_CLASSES = 17  # -1, 0, 2, 4, 8, ..., up to 32768. -1 is for inactive cells
NUM_ACTIONS = 4


class NewCNNEncoder(nn.Module):

    def __init__(
        self,
        out_features: int,
        multiplier: int = 16,
        mask: torch.BoolTensor | None = None,
    ) -> None:
        super().__init__()

        assert out_features >= 1 and out_features % 25 == 0, out_features
        assert multiplier >= 1, multiplier

        out_channels = out_features // 25 # 25 is the number of cells in a board (5x5)

        self.out_features = out_features
        self._out_channels = out_channels

        self.rows = 5
        self.cols = 5

        self._depthwise_full = nn.Conv1d(
            NUM_CLASSES,
            NUM_CLASSES * NUM_CELLS,
            NUM_CELLS,  # kernel size matches input length for 5x5 board
            groups=NUM_CLASSES,
        )
        self._pointwise_full = nn.Conv1d(
            self._depthwise_full.out_channels,
            out_channels * 5, # 5 so that we get 5 channels for each row and column
            1,
        )

        self._depthwise_hori = nn.Conv2d(
            NUM_CLASSES,
            NUM_CLASSES * multiplier,
            (1, self.cols),
            groups=NUM_CLASSES,
        )
        self._pointwise_hori = nn.Conv2d(
            self._depthwise_hori.out_channels,
            out_channels,
            1,
        )

        self._depthwise_vert = nn.Conv2d(
            NUM_CLASSES,
            NUM_CLASSES * multiplier,
            (self.rows, 1),
            groups=NUM_CLASSES,
        )
        self._pointwise_vert = nn.Conv2d(
            self._depthwise_vert.out_channels,
            out_channels,
            1,
        )

        self._conv_out = nn.Conv1d(
            out_channels,
            out_features,
            15,
        )

        self._mask = mask

        self.reset_parameters()

    def reset_parameters(self):
        sqrt2 = math.sqrt(2)

        # nn.init.orthogonal_(self._depthwise_full.weight, sqrt2)
        nn.init.zeros_(self._depthwise_full.bias)

        # nn.init.orthogonal_(self._depthwise_hori.weight, sqrt2)
        nn.init.zeros_(self._depthwise_hori.bias)

        # nn.init.orthogonal_(self._depthwise_vert.weight, sqrt2)
        nn.init.zeros_(self._depthwise_vert.bias)

        nn.init.zeros_(self._conv_out.bias)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        assert x.dtype == torch.long
        assert x.ndim == 2 and x.shape[1] == 25, x.shape

        # (N, 25) -> (N, 25, NUM_CLASSES)
        # Mask logic, set inactive cells to -1 BEFORE one-hot encoding
        if self._mask is not None:
            # self._mask: (5, 5) or (rows, cols)
            # x: (N, 25)
            mask_flat = self._mask.flatten()  # (25,)
            x = x.clone()
            x[:, ~mask_flat] = NUM_CLASSES-1  # Set inactive cells to -1
        
        x = F.one_hot(x, NUM_CLASSES)
        x = x.float()

        # -> (N, NUM_CLASSES, 25)
        x = torch.permute(x, (0, 2, 1))

        # -> (N, NUM_CLASSES * m, 1)
        x_full = self._depthwise_full(x)
        x_full = F.leaky_relu(x_full)
        
        # -> (N, out * 4, 1)
        x_full = self._pointwise_full(x_full)
        x_full = F.leaky_relu(x_full)

        # -> (N, NUM_CLASSES, 5, 5)
        board = torch.reshape(x, (-1, NUM_CLASSES, 5, 5))

        # -> (N, NUM_CLASSES * m, 5, 1)
        x_hori = self._depthwise_hori(board)
        x_hori = F.leaky_relu(x_hori)

        # -> (N, out, 5, 1)
        x_hori = self._pointwise_hori(x_hori)
        x_hori = F.leaky_relu(x_hori)

        # -> (N, NUM_CLASSES * m, 1, 5)
        x_vert = self._depthwise_vert(board)
        x_vert = F.leaky_relu(x_vert)

        # -> (N, out, 1, 5)
        x_vert = self._pointwise_vert(x_vert)
        x_vert = F.leaky_relu(x_vert)

        # first

        first_arr = torch.reshape(x_full, (-1, self._out_channels, 5))
        # second
        second_arr = torch.flatten(x_hori, 2)
        # third
        third_arr = torch.flatten(x_vert, 2)

        x = torch.cat(
            (
                first_arr,
                second_arr,
                third_arr,
            ),
            dim=2,
        )

        x = self._conv_out(x)
        x = F.leaky_relu(x)

        x = torch.flatten(x, 1)
        return x.to(torch.float)

    def update_mask(self, mask: torch.BoolTensor):
        assert mask.shape == (self.rows, self.cols), mask.shape
        self._mask = mask

class CNNActorNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_hidden: int,
        num_hidden2: int,
    ) -> None:
        super().__init__()

        self._fc1 = nn.Linear(in_features, num_hidden)
        self._fc2 = nn.Linear(self._fc1.out_features, num_hidden2)

        # logits output
        self._out = nn.Linear(self._fc2.out_features, NUM_ACTIONS)

        self.reset_parameters()

    def reset_parameters(self):
        sqrt2 = math.sqrt(2)

        nn.init.orthogonal_(self._fc1.weight, sqrt2)
        nn.init.zeros_(self._fc1.bias)

        nn.init.orthogonal_(self._fc2.weight, sqrt2)
        nn.init.zeros_(self._fc2.bias)

        nn.init.orthogonal_(self._out.weight, 0.01)
        nn.init.zeros_(self._out.bias)

    def forward(
        self,
        x: torch.FloatTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        # -> (N, num_hidden)
        x = self._fc1(x)
        x = F.relu(x)

        # -> (N, num_hidden2)
        x = self._fc2(x)
        x = F.relu(x)

        # -> (N, 4)
        logits = self._out(x)

        # translation such that logits <= 0
        # note that logit_max is a constant to the graph (detached)
        logit_max, _ = torch.max(logits.detach(), dim=-1, keepdim=True)
        logits = logits - logit_max

        return logits


class CNNCriticNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_hidden: int,
        num_hidden2: int,
    ) -> None:
        super().__init__()

        self._fc1 = nn.Linear(in_features, num_hidden)

        self._fc2 = nn.Linear(self._fc1.out_features, num_hidden2)

        # value output
        self._out = nn.Linear(self._fc2.out_features, 1)

        self.reset_parameters()

    def reset_parameters(self):
        sqrt2 = math.sqrt(2)

        nn.init.orthogonal_(self._fc1.weight, sqrt2)
        nn.init.zeros_(self._fc1.bias)

        nn.init.orthogonal_(self._fc2.weight, sqrt2)
        nn.init.zeros_(self._fc2.bias)

        nn.init.orthogonal_(self._out.weight, 1)
        nn.init.zeros_(self._out.bias)

    def forward(
        self,
        x: torch.FloatTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        # -> (N, num_hidden)
        x = self._fc1(x)
        x = F.relu(x)

        # -> (N, num_hidden2)
        x = self._fc2(x)
        x = F.relu(x)

        # -> (N, 1)
        x = self._out(x)
        x = torch.squeeze(x, dim=-1)

        return x
