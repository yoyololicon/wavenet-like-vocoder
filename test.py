import os
import argparse
import torch
#from librosa.output import write_wav
import soundfile as sf
import numpy as np
from scipy.interpolate import interp1d
import model.model as module_arch
from utils import class2float, np_mulw_inv


def main(config, resume, npzfile, outfile, c, cuda):
    # build model architecture
    model = getattr(module_arch, 'Fast' + config['arch']['type'])(**config['arch']['args'])
    q_channels = config['arch']['args']['classes']
    model.summary()

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict, False)
    if config['n_gpu'] > 1:
        model = model.module

    # prepare model for testing
    device = torch.device('cuda' if cuda else 'cpu')
    model = model.to(device)
    model.eval()

    data = np.load(npzfile)
    sr = data['sr'][0]
    hop_size = data['hop_size'][0]
    feature = data['feature']
    x = np.arange(feature.shape[1]) * hop_size
    f = interp1d(x, feature, axis=1)
    feature = f(np.arange(x[-1]))

    feature = torch.Tensor(feature).to(device)
    outputs = model(feature, c)

    c2f = class2float(q_channels)
    inv_fn = np_mulw_inv(q_channels)

    outputs = inv_fn(c2f(outputs.cpu().numpy()))
    sf.write(outfile, outputs, sr, subtype='PCM_16')
    #write_wav(outfile, outputs, sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npzfile', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('-c', type=float, default=1.)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume, args.npzfile, args.outfile, args.c, args.cuda)
