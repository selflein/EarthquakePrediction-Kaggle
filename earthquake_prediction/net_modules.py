import torch
from torch import nn


class AdaptiveMaxAbsolutePooling(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveMaxAbsolutePooling, self).__init__()
        self.pool = nn.AdaptiveMaxPool1d(
            output_size=output_size,
            return_indices=True
        )

    def forward(self, inp):
        _, idxs = self.pool(inp.abs())
        return torch.gather(inp, 2, idxs.long())


class MaxAbsolutePooling(nn.Module):
    def __init__(self, pool_size, stride):
        super(MaxAbsolutePooling, self).__init__()
        self.pool = nn.MaxPool1d(
            kernel_size=pool_size,
            stride=stride,
            return_indices=True
        )

    def forward(self, inp):
        _, idxs = self.pool(inp.abs())
        return torch.gather(inp, 2, idxs.long())
