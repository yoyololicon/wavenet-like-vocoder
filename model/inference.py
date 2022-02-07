import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .model import WaveNet, FFTNet
from .mem_transformer import Waveformer


class FastWaveformer(nn.Module):
    def __init__(self, waveformer: Waveformer, emb: torch.Tensor):
        super().__init__()
        self._model = waveformer
        self.n_head = waveformer.n_head
        self.d_model = waveformer.d_model

        pos = waveformer.pos_emb(waveformer.pos)

        self.register_buffer('emb', emb.detach().tanh())

        self.register_buffer(
            'condition_W', waveformer.condition.weight.squeeze().detach())

        self.attn_norms = nn.ModuleList()
        self.FFs = nn.ModuleList()

        self.kv_nets = nn.ModuleList()
        self.q_nets = nn.ModuleList()
        self.o_nets = nn.ModuleList()
        head_r = []

        for layer in waveformer.layers:
            self.attn_norms.append(layer.attn_norm)
            self.FFs.append(layer.FF)
            self.kv_nets.append(layer.Attn.kv_net)
            self.q_nets.append(layer.Attn.q_net)
            self.o_nets.append(layer.Attn.o_net)
            head_r.append(layer.Attn.r_net(pos).view(
                pos.size(0), self.n_head, -1).detach())
            self.scale = layer.Attn.scale

        self.register_buffer('head_r', torch.stack(head_r, dim=0))
        self.register_buffer(
            'last_weight', waveformer.linear_proj.weight.detach())
        self.register_buffer('last_bias', waveformer.linear_proj.bias.detach())

    def forward(self, y: torch.Tensor, c: float = 1.):
        num_cls = self.last_weight.shape[0]

        buffers = [None] * self.head_r.size(0)

        y = self.condition_W @ y 

        outputs = y.new_empty(y.size(1), dtype=torch.int64)
        samples = torch.rand(y.size(1), device=y.device)
        x = self.emb[num_cls // 2 - 1]
        for i in tqdm(range(y.size(1))):
            cond = y[:, i]
            for j in range(len(buffers)):
                mem = buffers[j]
                x = cond + x
                tmp = self.attn_norms[j](x[None, None, :])
                head_kv = self.kv_nets[j](tmp).squeeze()
                head_q = self.q_nets[j](tmp).view(self.n_head, -1)

                if mem is not None: 
                    kv = torch.cat([mem, head_kv.unsqueeze(0)], dim=0) 
                else:
                    kv = head_kv.unsqueeze(0)
                if kv.size(0) == self.head_r.size(1):
                    mem = kv[1:]
                else:
                    mem = kv
                buffers[j] = mem

                k, v = kv.chunk(2, dim=1)
                k = k.reshape(k.size(0), self.n_head, -1) + self.head_r[j, -k.size(0):]
                v = v.reshape(v.size(0), self.n_head, -1)
                attn_scores = k.transpose(0, 1) @ head_q.unsqueeze(-1) * self.scale
                attn_weights = F.softmax(attn_scores, dim=1)
                out = attn_weights.transpose(1, 2) @ v.transpose(0, 1)
                out = self.o_nets[j](out.view(1, -1)).squeeze()
                x = out + x
                x = x + self.FFs[j](x[None, None, :]).squeeze()
        
            logits = torch.addmv(self.last_bias, self.last_weight, x)
            logits.mul_(c)
            cum_probs = logits.softmax(0).cumsum_(0)
            new_x = torch.nonzero(cum_probs > samples[i])[0].squeeze()
            outputs[i] = new_x
            x = self.emb[new_x]

        return outputs


class FastWaveNet(nn.Module):
    def __init__(self, wavenet: WaveNet, emb: torch.Tensor):
        super().__init__()
        self.dilations = wavenet.dilations
        self.mem_sizes = wavenet.mem_sizes
        self._model = wavenet

        self.register_buffer('emb', emb.detach().tanh())

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
        num_cls = self.end2_w.shape[0]

        buffers = self._model.init_memories(device=y.device)
        buf_indexes = [0] * len(self.mem_sizes)

        outputs = y.new_empty(y.size(1), dtype=torch.int64)
        samples = torch.rand(y.size(1), device=y.device)
        x = self.emb[num_cls // 2 - 1]
        for i in tqdm(range(y.size(1))):
            conds = torch.chunk(self.condition_W @
                                y[:, i], len(buffers), dim=0)
            skip = y.new_zeros(self.W_o_weight_last.size(0))
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
                    z = torch.addmv(self.W_o_bias[j], self.W_o_weight[j], z)
                    x = x + z[:x.size(0)]
                    skip += z[x.size(0):]
                else:
                    z = torch.addmv(self.W_o_bias_last,
                                    self.W_o_weight_last, z)
                    skip += z
            skip.relu_()
            skip = torch.addmv(self.end1_b, self.end1_w, skip)
            skip.relu_()
            logits = torch.addmv(self.end2_b, self.end2_w, skip)
            logits.mul_(c)
            cum_probs = logits.softmax(0).cumsum_(0)
            new_x = torch.nonzero(cum_probs > samples[i])[0].squeeze()
            outputs[i] = new_x
            x = self.emb[new_x]

        return outputs


class FastFFTNet(nn.Module):
    def __init__(self, fftnet: FFTNet, c: float = 1.):
        super().__init__()
        self.dilations = fftnet.dilations
        self.mem_sizes = fftnet.mem_sizes
        self.r_field = fftnet.r_field
        self.hop_length = fftnet.hop_length
        self.c = c

        self.register_buffer('emb', fftnet.emb[0].weight.detach().tanh())

        self.register_buffer(
            'condition_W', fftnet.conditioner.weight.squeeze().detach())

        WV_past_weight = []
        WV_present_weight = []
        W_o_weight = []
        W_o_bias = []

        for i, layer in enumerate(fftnet.layers):
            WV_weight = layer.WV.weight.detach()
            WV_past_weight.append(WV_weight[..., :-1])
            WV_present_weight.append(WV_weight[..., -1])
            W_o_weight.append(layer.W_o.weight.squeeze().detach())
            W_o_bias.append(layer.W_o.bias.data.detach())

        self.register_buffer(
            'WV_past_weight', torch.stack(WV_past_weight, dim=0))
        self.register_buffer('WV_present_weight',
                             torch.stack(WV_present_weight, dim=0))
        self.register_buffer('W_o_weight', torch.stack(W_o_weight, dim=0))
        self.register_buffer('W_o_bias', torch.stack(W_o_bias, dim=0))

        self.register_buffer(
            'end_w', fftnet.end.weight.squeeze().detach())
        self.register_buffer('end_b', fftnet.end.bias.detach())

    def forward(self, y: torch.Tensor):
        y = F.interpolate(y.unsqueeze(
            0), scale_factor=float(self.hop_length)).squeeze(0)

        num_cls = self.emb.size(0)
        fft_size = self.emb.size(1)

        buffers = []
        for size in self.mem_sizes:
            buffers.append(torch.zeros(fft_size, size, device=y.device))
        buf_indexes = [0] * len(self.mem_sizes)

        outputs = y.new_empty(y.size(1), dtype=torch.int64)
        samples = torch.rand(y.size(1), device=y.device)
        x = self.emb[num_cls // 2 - 1]
        for i in tqdm(range(y.size(1))):
            conds = torch.chunk(self.condition_W @
                                y[:, i], len(buffers), dim=0)
            for j, mem in enumerate(buffers):
                h = conds[j]
                idx = buf_indexes[j]
                for k in range(self.WV_past_weight.size(3)):
                    h += self.WV_past_weight[j, :, :, k] @ mem[:, idx]
                    idx = (idx + self.dilations[j]) % mem.size(1)
                h += self.WV_present_weight[j] @ x
                mem[:, buf_indexes[j]] = x
                buf_indexes[j] = (buf_indexes[j] + 1) % mem.size(1)

                h.relu_()
                z = torch.addmv(self.W_o_bias[j], self.W_o_weight[j], h)
                x = F.relu(z + x, inplace=True)

            logits = torch.addmv(self.end_b, self.end_w, x)
            logits.mul_(self.c)
            cum_probs = logits.softmax(0).cumsum_(0)
            new_x = torch.nonzero(cum_probs > samples[i])[0].squeeze()
            outputs[i] = new_x
            x = self.emb[new_x]

        return outputs
