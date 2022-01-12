import torch
from torch import nn
from typing import Optional, List

from torch import Tensor


class MemModel(nn.Module):
    def __init__(self, mem_channels: int, mem_sizes: List[int]):
        super().__init__()
        self.mem_sizes = mem_sizes
        self.mem_channels = mem_channels

    def init_memories(self, batch=None, dtype=None, device=None) -> List[Tensor]:
        if batch is None:
            return [
                torch.zeros(self.mem_channels, l, dtype=dtype, device=device)
                for l in self.mem_sizes
            ]
        return [
            torch.zeros(batch, self.mem_channels, l,
                        dtype=dtype, device=device)
            for l in self.mem_sizes
        ]


class DilationBase(MemModel):
    def __init__(self, mem_channels, n_blocks, n_layers, radix, descending):
        dilations = [radix ** i for i in range(n_layers)]
        if descending:
            dilations = dilations[::-1]
        dilations = dilations * n_blocks
        mem_sizes = [d * (radix - 1) for d in dilations]
        super().__init__(mem_channels, mem_sizes)
        self.receptive_field = sum(self.mem_sizes) + 1
        self.dilations = dilations
