import torch
from torch import nn
import torch.nn.functional as F


class WaveNetLayer(nn.Module):
    def __init__(self,
                 dilation,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 aux_channels,
                 radix,
                 bias,
                 last_layer=False):
        super().__init__()
        self.d_size = dilation_channels
        self.WV = nn.Conv1d(residual_channels + aux_channels, dilation_channels * 2, kernel_size=radix,
                            dilation=dilation, bias=bias)

        self.chs_split = [skip_channels]
        if last_layer:
            self.W_o = nn.Conv1d(dilation_channels, skip_channels, 1, bias=bias)
        else:
            self.W_o = nn.Conv1d(dilation_channels, residual_channels + skip_channels, 1, bias=bias)
            self.chs_split.insert(0, residual_channels)

        self.pad = nn.ConstantPad1d((dilation * (radix - 1), 0), 0.)

    def forward(self, x, y, zeropad):
        xy = torch.cat((x, y[:, :, -x.size(2):]), 1)
        if zeropad:
            xy = self.pad(xy)
        zw, zf = self.WV(xy).split(self.d_size, 1)
        z = zw.tanh() * zf.sigmoid()
        *z, skip = self.W_o(z).split(self.chs_split, 1)
        return z[0] + x[:, :, -z[0].size(2):] if len(z) else None, skip


class FFTNetLayer(nn.Module):
    def __init__(self,
                 dilation,
                 channels,
                 aux_channels,
                 radix,
                 bias):
        super().__init__()
        self.WV = nn.Sequential(
            nn.Conv1d(channels + aux_channels, channels, kernel_size=radix, dilation=dilation,
                      bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1, bias=bias))
        self.pad = nn.ConstantPad1d((dilation * (radix - 1), 0), 0.)

    def forward(self, x, y, zeropad):
        # type: (Tensor, Tensor, bool) -> Tensor
        xy = torch.cat((x, y[:, :, -x.size(2):]), 1)
        if zeropad:
            xy = self.pad(xy)
        z = self.WV(xy)
        return F.relu(z + x[:, :, -z.size(2):], True)  # add residual connection for better converge


class Queue(nn.Module):
    def __init__(self, channels, dilation, radix):
        super().__init__()
        self.buf = nn.Parameter(torch.zeros(1, channels, dilation * (radix - 1) + 1), requires_grad=False)

    def forward(self, sample):
        data = self.buf.data
        torch.cat((data[:, :, 1:], sample), 2, out=self.buf.data)
        return self.buf


if __name__ == '__main__':
    x = torch.rand(128)
    w = torch.randn(128, 128)

    from tqdm import tqdm

    for i in tqdm(range(44100)):
        for j in range(11):
            x = w @ x + x
            # torch.cat((w[:, 1:], w[:, -1:]), 1, out=w)
            # x.max()
            # x = F.softmax(x, 0)
        # torch.distributions.Categorical(x).sample()
