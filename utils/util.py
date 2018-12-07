import os
import numpy as np
from torch import nn

def np_mulaw(qc):
    mu = qc - 1

    def mulaw(x):
        return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

    return mulaw


def np_mulw_inv(qc):
    mu = qc - 1

    def mulaw(x):
        return np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu

    return mulaw


def float2class(qc):
    mu = qc - 1

    def convert(x):
        return np.rint((x + 1) / 2 * mu).astype(int)

    return convert


def class2float(qc):
    mu = qc - 1

    def convert(x):
        return x / mu * 2 - 1

    return convert

def remove_weight_norms(m):
    if hasattr(m, 'weight_g'):
        nn.utils.remove_weight_norm(m)


def add_weight_norms(m):
    if hasattr(m, 'weight'):
        nn.utils.weight_norm(m)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
