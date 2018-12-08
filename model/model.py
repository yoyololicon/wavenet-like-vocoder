import torch
from torch import nn
from torch.nn import functional as F

from model.module import FFTNetLayer, WaveNetLayer, Queue
from base.base_model import BaseModel


class _BaseModel(BaseModel):
    def __init__(self, n_blocks, n_layers, classes, radix, descending):
        super().__init__()
        dilations = radix ** torch.arange(n_layers)
        dilations = dilations.tolist()
        if descending:
            dilations = dilations[::-1]
        self.dilations = dilations * n_blocks
        self.cls = classes
        self.r_field = sum(self.dilations) + 1
        self.rdx = radix


class WaveNet(_BaseModel):
    def __init__(self,
                 n_blocks=4,
                 n_layers=10,
                 classes=256,
                 radix=2,
                 descending=False,
                 aux_channels=80,
                 dilation_channels=256,
                 residual_channels=256,
                 skip_channels=256,
                 bias=False):
        super().__init__(n_blocks, n_layers, classes, radix, descending)
        self.res_chs = residual_channels
        self.dil_chs = dilation_channels
        self.skp_chs = skip_channels
        self.aux_chs = aux_channels
        self.bias = bias

        self.emb = nn.Sequential(nn.Embedding(classes, residual_channels, padding_idx=classes // 2 - 1),
                                 nn.Tanh())

        self.layers = nn.ModuleList(WaveNetLayer(d,
                                                 dilation_channels,
                                                 residual_channels,
                                                 skip_channels,
                                                 aux_channels,
                                                 radix,
                                                 bias) for d in self.dilations[:-1])
        self.layers.append(WaveNetLayer(self.dilations[-1],
                                        dilation_channels,
                                        residual_channels,
                                        skip_channels,
                                        aux_channels,
                                        radix,
                                        bias,
                                        last_layer=True))

        self.end = nn.Sequential(nn.ReLU(inplace=True),
                                 nn.Conv1d(skip_channels, skip_channels, 1, bias=bias),
                                 nn.ReLU(inplace=True),
                                 nn.Conv1d(skip_channels, classes, 1, bias=bias))

    def forward(self, x, y):
        x = self.emb(x).transpose(1, 2)
        cum_skip = 0
        for layer in self.layers:
            x, skip = layer(x, y, True)
            cum_skip = cum_skip + skip
        return self.end(cum_skip)[..., None]


class FastWaveNet(WaveNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buf_list = nn.ModuleList(Queue(self.res_chs, d, self.rdx) for d in self.dilations)

    @torch.no_grad()
    def forward(self, x, y, c=1.):
        # x should be one sample long
        x = self.emb(x).view(1, -1, 1)
        cum_skip = None
        for i, (layer, buf) in enumerate(zip(self.layers, self.buf_list)):
            x, skip = layer(buf(x), y, False)
            if i:
                cum_skip += skip
            else:
                cum_skip = skip
        logits = self.end(cum_skip).view(-1).mul_(c)
        probs = F.softmax(logits, 0)
        return torch.distributions.Categorical(probs).sample()


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
        self.emb = nn.Embedding(classes, fft_channels, padding_idx=classes // 2 - 1)
        self.layers = nn.ModuleList(FFTNetLayer(d,
                                                fft_channels,
                                                aux_channels,
                                                radix,
                                                bias) for d in self.dilations)

        self.end = nn.Conv1d(fft_channels, classes, 1, bias=bias)

    def forward(self, x, y):
        x = self.emb(x).transpose(1, 2)
        for layer in self.layers:
            x = layer(x, y, True)
        return self.end(x)[..., None]


class FastFFTNet(FFTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buf_list = nn.ModuleList(Queue(self.fft_chs, d, self.rdx) for d in self.dilations)

    @torch.no_grad()
    def forward(self, x, y, c=1.):
        # x should be one sample long
        x = self.emb(x).view(1, -1, 1)
        for layer, buf in zip(self.layers, self.buf_list):
            x = layer(buf(x), y, 0)
        logits = self.end(x).view(-1).mul_(c)
        probs = F.softmax(logits, 0)
        return torch.distributions.Categorical(probs).sample()
