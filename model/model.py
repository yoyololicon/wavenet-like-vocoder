import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from typing import List, Optional

from .module import FFTNetLayer, WaveNetLayer, Queue


class _BaseModel(nn.Module):
    def __init__(self, n_blocks, n_layers, classes, radix, descending):
        super().__init__()
        dilations = [radix ** i for i in range(n_layers)]
        if descending:
            dilations = dilations[::-1]
        self.dilations = dilations * n_blocks
        self.mem_sizes = [d * (radix - 1) for d in self.dilations]
        self.cls = classes
        self.r_field = sum(self.mem_sizes) + 1
        self.rdx = radix


class WaveNet(_BaseModel):
    def __init__(self,
                 hop_length: int,
                 n_blocks=4,
                 n_layers=10,
                 classes=256,
                 radix=2,
                 descending=False,
                 aux_channels=80,
                 dilation_channels=256,
                 residual_channels=256,
                 skip_channels=256):
        super().__init__(n_blocks, n_layers, classes, radix, descending)
        self.res_chs = residual_channels
        self.hop_length = hop_length

        self.emb = nn.Sequential(nn.Embedding(classes, residual_channels, padding_idx=classes // 2 - 1),
                                 nn.Tanh())

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

    def forward(self, x: torch.Tensor, y: torch.Tensor, zeropad: bool, memory: List[Optional[torch.Tensor]] = None):
        x = self.emb(x).transpose(1, 2)
        cond = F.interpolate(self.conditioner(
            y), scale_factor=self.hop_length, mode='linear')[..., :x.size(-1)].chunk(len(self.layers), 1)

        cum_skip = None
        if memory is None:
            memory = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            if memory[i] is not None:
                x = torch.cat([memory[i], x], dim=2)
                memory[i] = x[..., -memory[i].size(-1):].detach()
            tmp, skip = layer(x, cond[i], zeropad)
            if tmp is not None:
                x = tmp
            if cum_skip is None:
                cum_skip = skip
            else:
                cum_skip = cum_skip[..., -skip.size(2):] + skip
        return self.end(cum_skip)


class FFTNet(_BaseModel):
    def __init__(self,
                 n_blocks=1,
                 n_layers=11,
                 classes=256,
                 radix=2,
                 descending=True,
                 aux_channels=26,
                 fft_channels=128,
                 bias=False):
        super().__init__(n_blocks, n_layers, classes, radix, descending)
        self.fft_chs = fft_channels
        self.aux_chs = aux_channels
        self.bias = bias
        self.emb = nn.Embedding(classes, fft_channels,
                                padding_idx=classes // 2 - 1)
        self.layers = nn.ModuleList(FFTNetLayer(d,
                                                fft_channels,
                                                radix,
                                                bias) for d in self.dilations)

        self.end = nn.Conv1d(fft_channels, classes, 1, bias=bias)

        self.conditioner = nn.Conv1d(
            aux_channels, fft_channels * len(self.layers), 1, bias=bias)

    def forward(self, x, y, zeropad: bool):
        x = self.emb(x).transpose(1, 2)
        cond = self.conditioner(y).chunk(len(self.layers), 1)
        for i, layer in enumerate(self.layers):
            x = layer(x, cond[i], zeropad)
        return self.end(x)


class FastFFTNet(FFTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buf_list = nn.ModuleList(
            Queue(self.fft_chs, d, self.rdx) for d in self.dilations)

    @torch.no_grad()
    def forward(self, y, c=1.):
        total_len = y.size(1)
        max_d = max(self.dilations)
        outputs = torch.empty((total_len + 1,), dtype=torch.long,
                              device=y.device).fill_(self.cls // 2 - 1)
        y = F.pad(y, (max_d, 0)).unsqueeze(0)

        for pos in tqdm(range(total_len)):
            x = self.emb(outputs[pos]).view(1, -1, 1)
            for layer, buf in zip(self.layers, self.buf_list):
                x = layer(buf(x), y[:, :, :pos + max_d + 1], False)
            logits = self.end(x).view(-1).mul_(c)
            probs = F.softmax(logits, 0)
            outputs[pos + 1] = torch.distributions.Categorical(probs).sample()
        return outputs[1:]
