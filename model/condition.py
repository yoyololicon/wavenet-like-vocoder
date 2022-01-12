from torch import nn, Tensor
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


class MelSpec(nn.Module):
    def __init__(self, sr, n_fft, hop_length, **kwargs) -> None:
        super().__init__()

        self.mel = nn.Sequential(
            nn.ReflectionPad1d((n_fft // 2 - hop_length // 2,
                                n_fft // 2 + hop_length // 2)),
            MelSpectrogram(sample_rate=sr, n_fft=n_fft,
                           hop_length=hop_length, center=False, normalized=True, **kwargs)
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.threshold(self.mel(x).log10_(), -6, -6, True).add_(6).div_(8)
