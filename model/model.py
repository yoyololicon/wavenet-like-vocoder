import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Optional

from .module import FFTNetLayer, WaveNetLayer
from .base import DilationBase


class WaveNet(DilationBase):
    def __init__(self,
                 n_blocks=4,
                 n_layers=10,
                 classes=256,
                 radix=2,
                 descending=False,
                 aux_channels=80,
                 dilation_channels=256,
                 residual_channels=256,
                 skip_channels=256):
        super().__init__(residual_channels, n_blocks, n_layers, radix, descending)

        self.layers = nn.ModuleList(WaveNetLayer(d,
                                                 dilation_channels,
                                                 residual_channels,
                                                 skip_channels,
                                                 radix) for d in self.dilations[:-1])
        self.layers.append(WaveNetLayer(self.dilations[-1],
                                        dilation_channels,
                                        residual_channels,
                                        skip_channels,
                                        radix,
                                        last_layer=True))

        self.end = nn.Sequential(nn.ReLU(inplace=True),
                                 nn.Conv1d(skip_channels,
                                           skip_channels, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv1d(skip_channels, classes, 1))

        self.conditioner = nn.Conv1d(
            aux_channels, dilation_channels * 2 * len(self.layers), 1, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor, zeropad: bool = True, memories: List[Optional[torch.Tensor]] = None):
        cond = self.conditioner(y).chunk(len(self.layers), 1)
        cum_skip = None
        if zeropad is True:
            memories = self.init_memories(x.size(0), device=x.device)
        elif memories is None:
            memories = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            if memories[i] is not None:
                x = torch.cat([memories[i], x], dim=2)
                if not zeropad:
                    memories[i] = x[..., -memories[i].size(-1):].detach()
            tmp, skip = layer(x, cond[i])
            if tmp is not None:
                x = tmp
            if cum_skip is None:
                cum_skip = skip
            else:
                cum_skip = cum_skip[..., -skip.size(2):] + skip
        return self.end(cum_skip)


class FFTNet(DilationBase):
    def __init__(self,
                 n_blocks=1,
                 n_layers=11,
                 classes=256,
                 radix=2,
                 descending=True,
                 aux_channels=26,
                 fft_channels=128):
        super().__init__(fft_channels, n_blocks, n_layers, radix, descending)
        self.layers = nn.ModuleList(FFTNetLayer(d,
                                                fft_channels,
                                                radix) for d in self.dilations)

        self.end = nn.Conv1d(fft_channels, classes, 1)

        self.conditioner = nn.Conv1d(
            aux_channels, fft_channels * len(self.layers), 1, bias=False)

    def forward(self, x, y, zeropad: bool = True, memories: List[Optional[torch.Tensor]] = None):
        cond = self.conditioner(y).chunk(len(self.layers), 1)
        if zeropad is True:
            memories = self.init_memories(x.size(0), device=x.device)
        elif memories is None:
            memories = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            if memories[i] is not None:
                x = torch.cat([memories[i], x], dim=2)
                if not zeropad:
                    memories[i] = x[..., -memories[i].size(-1):].detach()
            x = layer(x, cond[i])

        return self.end(x)
