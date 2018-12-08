import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from model.model import FFTNet

class Queue(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.size = dilation
        self.buf = torch.zeros(self.size, channels)
        self.idx = 0

    def forward(self, sample):
        data = self.buf[self.idx].clone()
        self.buf[self.idx] = sample
        self.idx += 1
        self.idx %= self.size
        return data

    def cpu(self):
        self.buf = self.buf.cpu()

    def cuda(self, device=None):
        self.buf = self.buf.cuda(device)

    def init(self):
        self.buf.zero_()


class FastFFTNet2(FFTNet):
    '''

    only for radix = 2
    '''

    def __init__(self, fftnet_state, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.load_state_dict(fftnet_state)
        self.cpu()

        self.condition_conv = nn.ModuleList(
            nn.Conv1d(self.aux_chs, self.fft_chs, self.rdx, dilation=d, bias=self.bias) for d in self.dilations)
        self.W1 = []
        self.W2 = []
        self.W2_b = []
        for i, layer in enumerate(self.layers):
            self.condition_conv[i].weight.data.copy_(layer.WV[0].weight.data[:, -self.aux_chs:])
            self.W1.append(
                layer.WV[0].weight.data[:, :self.fft_chs].transpose(1, 2).contiguous().view(self.fft_chs,
                                                                                            self.fft_chs * 2))
            self.W2.append(layer.WV[2].weight.data.squeeze())
            if self.bias:
                self.condition_conv[i].bias.data.copy_(layer.WV[0].bias.data)
                self.W2_b.append(layer.WV[2].bias.data.clone())

        self.end_w = self.end.weight.data.squeeze()
        if self.bias:
            self.end_b = self.end.bias.data.clone()
        self.buf_list = nn.ModuleList(Queue(self.fft_chs, d) for d in self.dilations)
        delattr(self, "layers")
        delattr(self, "end")

    def forward(self, x, y):
        raise NotImplementedError

    @torch.no_grad()
    def inference(self, y, c=1.):
        print("pre-compute middle output of feature...")
        y = y[None, ...].cuda()
        self.condition_conv.cuda()
        hidden_y = []
        for d, layer in zip(self.dilations, self.condition_conv):
            hidden_y.append(layer(F.pad(y, (d, 0))).cpu())

        hidden_y = torch.cat(hidden_y, 0)
        print("pre-computation done.")

        outputs = torch.LongTensor(y.size(2) + 1).fill_(self.cls // 2 - 1)
        for pos in tqdm(range(hidden_y.size(2))):
            x = self.emb(outputs[pos]).view(-1)
            for i, d in enumerate(self.dilations):
                prev = self.buf_list[i](x)
                x = torch.cat((prev, x), 0)
                x = F.relu(self.W1[i] @ x + hidden_y[i, :, pos])
                x = self.W2[i] @ x
                if self.bias:
                    x += self.W2_b[i]
                x = F.relu(x)
            x = self.end_w @ x
            if self.bias:
                x += self.end_b
            x *= c
            probs = F.softmax(x, 0)
            outputs[pos + 1] = torch.distributions.Categorical(probs).sample()

        return outputs
