import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .model import WaveNet


class FastWaveNet(nn.Module):
    def __init__(self, wavenet: WaveNet):
        super().__init__()
        self.dilations = wavenet.dilations
        self.mem_sizes = wavenet.mem_sizes
        self.r_field = wavenet.r_field
        self.hop_length = wavenet.hop_length
        self.res_chs = wavenet.res_chs

        self.register_buffer('emb', wavenet.emb[0].weight.detach().tanh())

        self.register_buffer(
            'condition_W', wavenet.conditioner.weight.squeeze().detach())

        WV_past_weight = []
        WV_present_weight = []
        W_o_weight = []
        W_o_bias = []

        for i, layer in enumerate(wavenet.layers):
            WV_weight = layer.WV.weight.detach()
            WV_past_weight.append(WV_weight[..., :-1])
            WV_present_weight.append(WV_weight[..., -1])
            W_o_weight.append(layer.W_o.weight.squeeze().detach())
            W_o_bias.append(layer.W_o.bias.data.detach())

        self.register_buffer(
            'WV_past_weight', torch.stack(WV_past_weight, dim=0))
        self.register_buffer('WV_present_weight',
                             torch.stack(WV_present_weight, dim=0))
        self.register_buffer('W_o_weight', torch.stack(W_o_weight[:-1], dim=0))
        self.register_buffer('W_o_bias', torch.stack(W_o_bias[:-1], dim=0))
        self.register_buffer('W_o_weight_last', W_o_weight[-1])
        self.register_buffer('W_o_bias_last', W_o_bias[-1])

        self.register_buffer(
            'end1_w', wavenet.end[1].weight.squeeze().detach())
        self.register_buffer('end1_b', wavenet.end[1].bias.detach())
        self.register_buffer(
            'end2_w', wavenet.end[3].weight.squeeze().detach())
        self.register_buffer('end2_b', wavenet.end[3].bias.detach())

    def forward(self, y: torch.Tensor, c: float = 1.):
        y = F.interpolate(y.unsqueeze(
            0), scale_factor=float(self.hop_length)).squeeze(0)

        num_cls = self.emb.size(0)
        outputs = [num_cls // 2 - 1]

        buffers = []
        for size in self.mem_sizes:
            buffers.append(torch.zeros(self.res_chs, size, device=y.device))
        buf_indexes = [0] * len(self.mem_sizes)

        samples = torch.rand(y.size(1), device=y.device)
        for i in range(y.size(1)):
            x = self.emb[outputs[i]]
            conds = torch.chunk(self.condition_W @
                                y[:, i], len(buffers), dim=0)
            skip = x.new_zeros(self.W_o_weight_last.size(0))
            for j, mem in enumerate(buffers):
                h = conds[j]
                idx = buf_indexes[j]
                for k in range(self.WV_past_weight.size(3)):
                    h += self.WV_past_weight[j, :, :, k] @ mem[:, idx]
                    idx = (idx + self.dilations[j]) % mem.size(1)
                h += self.WV_present_weight[j] @ x
                mem[:, buf_indexes[j]] = x
                buf_indexes[j] = (buf_indexes[j] + 1) % mem.size(1)

                # torch.cat([mem[:, 1:], x.unsqueeze(1)], dim=1, out=mem)

                zw, zf = h.chunk(2, 0)
                z = zw.tanh_().mul_(zf.sigmoid_())
                if j < len(buffers) - 1:
                    z = self.W_o_weight[j] @ z
                    z.add_(self.W_o_bias[j])
                    x = x + z[:x.size(0)]
                    skip += z[x.size(0):]
                else:
                    z = self.W_o_weight_last @ z
                    z.add_(self.W_o_bias_last)
                    skip += z
            skip.relu_()
            skip = self.end1_w @ skip
            skip.add_(self.end1_b)
            skip.relu_()
            skip = self.end2_w @ skip
            skip.add_(self.end2_b)
            skip.mul_(c)
            probs = skip.softmax(0)
            cum_probs = probs.cumsum(0)
            new_x = torch.nonzero(cum_probs > samples[i])[0]
            outputs.append(new_x.item())

        return torch.tensor(outputs[1:], device=y.device)
