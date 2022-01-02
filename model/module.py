from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


class WaveNetLayer(nn.Module):
    def __init__(self,
                 dilation,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 radix,
                 last_layer=False):
        super().__init__()
        self.dilation = dilation
        self.d_size = dilation_channels
        self.last_layer = last_layer
        self.WV = nn.Conv1d(residual_channels, dilation_channels * 2, kernel_size=radix,
                            dilation=dilation, bias=False)

        if last_layer:
            self.W_o = nn.Conv1d(
                dilation_channels, skip_channels, 1)
        else:
            self.W_o = nn.Conv1d(
                dilation_channels, residual_channels + skip_channels, 1)

        self.pad_size = dilation * (radix - 1)
        self.pad = nn.ConstantPad1d((self.pad_size, 0), 0.)

    def forward(self, x: torch.Tensor, y: torch.Tensor, zeropad: bool = False):
        if zeropad:
            x = self.pad(x)
        xy = self.WV(x)
        xy += y[:, :, -xy.size(2):]
        zw, zf = xy.chunk(2, 1)
        z = zw.tanh() * zf.sigmoid()
        z = self.W_o(z)

        if self.last_layer:
            return None, z
        else:
            return z[:, :x.size(1)] + x[..., -z.size(2):], z[:, x.size(1):]


class FFTNetLayer(nn.Module):
    def __init__(self,
                 dilation,
                 channels,
                 radix,
                 bias):
        super().__init__()
        self.WV = nn.Conv1d(channels, channels,
                            kernel_size=radix, dilation=dilation, bias=bias)
        self.W_o = nn.Conv1d(channels, channels, 1, bias=bias)
        self.pad_size = dilation * (radix - 1)
        self.pad = nn.ConstantPad1d((self.pad_size, 0), 0.)

    def forward(self, x: torch.Tensor, y: torch.Tensor, zeropad: bool, memory: Optional[torch.Tensor] = None):
        new_memory: Optional[torch.Tensor] = None
        if memory is not None:
            x = torch.cat([memory, x], dim=2)
            new_memory = x[..., -self.pad_size:]
        elif zeropad:
            x = self.pad(x)
        z = self.WV(x)
        z += y[:, :, -z.size(2):]
        z.relu_()
        z = self.W_o(z)
        # add residual connection for better converge
        return F.relu(z + x[:, :, -z.size(2):], True), new_memory


class Queue(nn.Module):
    def __init__(self, channels, dilation, radix):
        super().__init__()
        self.register_buffer('buf', torch.zeros(
            1, channels, dilation * (radix - 1) + 1))

    def forward(self, sample):
        torch.cat((self.buf[:, :, 1:], sample), 2, out=self.buf)
        return self.buf


if __name__ == '__main__':
    model = WaveNetLayer(4, 32, 32, 32, 2, True, True)
    # model = FFTNetLayer(4, 32, 2, True)
    print(model)
    model = torch.jit.script(model)
    print(model.code)

    x = torch.randn(1, 32, 100)
    y = torch.randn(1, 64, 100)
    y, mem = model(x, y, True)
    print(y.size())
