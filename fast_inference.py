import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import model.model as module_arch

import argparse
import numpy as np
from utils.util import class2float, np_mulw_inv
from librosa.output import write_wav
from scipy.interpolate import interp1d


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


def _dot_add(w, x, b):
    y = w @ x
    if b is None:
        pass
    else:
        y += b
    return y


class FastFFTNet(module_arch.FFTNet):
    '''

    only for radix = 2
    '''

    def __init__(self, fftnet_state, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.load_state_dict(fftnet_state)
        self.cpu()

        self.condition_conv = nn.ModuleList(
            nn.Conv1d(self.aux_chs, self.fft_chs, 2, dilation=d, bias=self.bias) for d in self.dilations)
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
            else:
                self.W2_b.append(None)

        self.end_w = self.end.weight.data.squeeze()
        if self.bias:
            self.end_b = self.end.bias.data.clone()
        else:
            self.end_b = None
        self.buf_list = nn.ModuleList(Queue(self.fft_chs, d) for d in self.dilations)

        delattr(self, "layers")
        delattr(self, "end")

    @torch.no_grad()
    def forward(self, y, c=1.):
        print("pre-compute middle output of feature...")
        y = y[None, ...].cuda()
        self.condition_conv.cuda()
        hidden_y = []
        for d, layer in zip(self.dilations, self.condition_conv):
            hidden_y.append(layer(F.pad(y, (d, 0))).squeeze(0).cpu())

        # hidden_y = torch.cat(hidden_y, 0)
        print("pre-computation done.")

        outputs = torch.LongTensor(y.size(2) + 1).fill_(self.cls // 2 - 1)
        for pos in tqdm(range(y.size(2))):
            x = self.emb(outputs[pos]).view(-1)
            for w1, w2, w2b, buf, hid_y in zip(self.W1, self.W2, self.W2_b, self.buf_list, hidden_y):
                prev = buf(x)
                concat = torch.cat((prev, x), 0)
                z = F.relu(_dot_add(w1, concat, hid_y[:, pos]), inplace=True)
                z = _dot_add(w2, z, w2b)
                x += z
                x = F.relu(x, inplace=True)
            x = _dot_add(self.end_w, x, self.end_b)
            x *= c
            probs = F.softmax(x, 0)
            outputs[pos + 1] = torch.distributions.Categorical(probs).sample()

        return outputs


def main(config, resume, npzfile, outfile, c):
    # build model architecture
    model = getattr(module_arch, config['arch']['type'])(**config['arch']['args'])
    q_channels = config['arch']['args']['classes']

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict, False)
    if config['n_gpu'] > 1:
        model = model.module

    # prepare model for testing
    model = model.cpu()
    model.eval()

    infer_model = globals()['Fast' + config['arch']['type']](model.state_dict(), **config['arch']['args'])
    infer_model.summary()

    data = np.load(npzfile)
    sr = data['sr'][0]
    hop_size = data['hop_size'][0]
    feature = data['feature']
    x = np.arange(feature.shape[1]) * hop_size
    f = interp1d(x, feature, axis=1)
    feature = f(np.arange(x[-1]))

    feature = torch.Tensor(feature)
    outputs = infer_model(feature, c)

    c2f = class2float(q_channels)
    inv_fn = np_mulw_inv(q_channels)

    outputs = inv_fn(c2f(outputs.cpu().numpy()))
    write_wav(outfile, outputs, sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npzfile', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('-c', type=float, default=1.)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']

    main(config, args.resume, args.npzfile, args.outfile, args.c)
