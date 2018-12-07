import numpy as np
from librosa import load
from librosa.feature import melspectrogram

eps = np.finfo(np.float32).eps


def get_logmel(filename, sr, n_fft, hop_length, **kwargs):
    """

    :param filename:
    :param sr:
    :param n_fft:
    :param hop_length:
    :param kwargs: arguments to mel filters
    :return: mel-spectrogram
    """
    y, _ = load(filename, sr)
    y /= np.abs(y).max()
    logmel = np.log(melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, **kwargs) + eps)
    logmel = np.pad(logmel, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    return filename, y, logmel
