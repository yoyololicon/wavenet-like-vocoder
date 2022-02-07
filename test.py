import os
import argparse
import torch
from torch.cuda import amp
import torchaudio

from utils import remove_weight_norms
from time import time
import math

from model import LightModel, WaveNet, FastWaveNet, FFTNet, FastFFTNet, Waveformer, FastWaveformer


def main(ckpt, infile, outfile, half):
    lit_model = LightModel.load_from_checkpoint(ckpt, map_location='cpu')
    model = lit_model.model
    conditioner = lit_model.conditioner
    upsampler = lit_model.upsampler
    decoder = lit_model.dec
    emb = lit_model.emb

    infer_model = {
        WaveNet: FastWaveNet,
        FFTNet: FastFFTNet,
        Waveformer: FastWaveformer
    }[type(model)](model, emb[0].weight.detach())

    device = torch.device('cpu')
    infer_model = infer_model.to(device)
    conditioner = conditioner.to(device)
    upsampler = upsampler.to(device)
    decoder = decoder.to(device)
    infer_model.eval()
    # infer_model = torch.jit.script(infer_model)
    # print(infer_model.code)

    y, sr = torchaudio.load(infile)
    y = y.mean(0, keepdim=True).to(device)
    # y = y[..., :sr * 2]

    cond = upsampler(conditioner(y)).squeeze(0)

    # torch.onnx.export(infer_model, cond[:, :2], 'infer.onnx', verbose=True, opset_version=11,
    #                   operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

    # exit()

    if half:
        infer_model = infer_model.half()
        cond = cond.half()
        y = y.half()

    with torch.no_grad():
        start = time()
        x = infer_model(cond)
        x = decoder(x)
        cost = time() - start

    print("Time cost: {:.4f}, Speed: {:.4f} kHz".format(
        cost, x.numel() / cost / 1000))
    print(x.max().item(), x.min().item())

    torchaudio.save(outfile, x.unsqueeze(0).cpu(), sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inferencer')
    parser.add_argument('ckpt', type=str)
    parser.add_argument('infile', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    main(args.ckpt, args.infile, args.outfile, args.half)
