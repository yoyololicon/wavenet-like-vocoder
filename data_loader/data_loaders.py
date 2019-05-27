from base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from scipy.interpolate import interp1d
import random
from utils.util import np_mulaw, float2class


class _WAVDataset(Dataset):

    def __init__(self,
                 data_dir,
                 size,
                 segment,
                 quantization_channels,
                 injected_noise=True):
        self.segment = segment
        self.data_path = os.path.expanduser(data_dir)
        self.size = size
        self.q_channels = quantization_channels
        self.noise = injected_noise
        self.mulaw = np_mulaw(quantization_channels)
        self.f2c = float2class(quantization_channels)

        data = np.load(os.path.join(data_dir, 'data.npz'))
        self.idx_name = []
        self.waves = dict()
        for name, x in data.items():
            if name == 'sr':
                self.sr = x[0]
            else:
                self.idx_name.append(name)
                self.waves[name] = x

        data = np.load(os.path.join(data_dir, 'feature.npz'))
        self.features = dict()
        max_len = 0
        for name, h in data.items():
            if name == 'hop_size':
                self.hop_size = h[0]
            else:
                assert name in self.idx_name
                self.features[name] = h
                max_len = max(max_len, h.shape[1])
        self.hop_idx = np.arange(max_len) * self.hop_size

        assert hasattr(self, 'sr') and hasattr(self, 'hop_size')

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        name = random.choice(self.idx_name)
        wav = self.waves[name]
        condition = self.features[name]
        pos = np.random.randint(0, len(wav) - self.segment - 1)

        t = wav[pos + 1:pos + 1 + self.segment]
        t = self.f2c(self.mulaw(t))

        x = wav[pos:pos + self.segment]
        if self.noise:
            x += np.random.randn(self.segment) / self.q_channels
            x = np.clip(x, -1, 1)
        x = self.f2c(self.mulaw(x))

        # fix possible outlier
        x = np.clip(x, 0, self.q_channels - 1)
        t = np.clip(t, 0, self.q_channels - 1)

        f = interp1d(self.hop_idx[:condition.shape[1]], condition, copy=False, axis=1)
        h = f(np.arange(pos + 1, pos + 1 + self.segment))

        return x, h.astype(np.float32), t.reshape(-1, 1)


class RandomWaveFileLoader(DataLoader):

    def __init__(self, steps, data_dir, batch_size, num_workers, **kwargs):
        self.data_dir = data_dir
        self.dataset = _WAVDataset(data_dir, batch_size * steps, **kwargs)
        super().__init__(self.dataset, batch_size, num_workers=num_workers)


if __name__ == '__main__':
    mulaw = np_mulaw(256)
    f2c = float2class(256)

    import numpy as np

    x = np.random.randn(10000)
    print(f2c(mulaw(x)).max(), f2c(mulaw(x)).min())
