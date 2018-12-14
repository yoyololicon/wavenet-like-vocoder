# WaveNet-like vocoder models

Basic implementations of WaveNet and modified FFTNet in PyTorch. The project structre is brought from [pytorch-template].

## Requirements
* NumPy
* SciPy
* PyTorch >= 0.4.1
* tqdm
* librosa

## Quick Start

The code in this repo by default will train a WaveNet (or FFTNet) using 80-dimension mel-spectrogram with linear interpolation.

### Preprocess

Use `preprocess.py` to convert your wave files into mel-spectrograms.

```
python preprocess.py wave/files/folder -c config.json --out data
```

The preprocessed data will be stored in `./data`.
You can change the configurations of "feature" in the `.json` file.

### Train

```
python train.py -c config.json
```

### Test

Use `preprocess.py` to convert a single wave file into mel-spectrogram feature.

```
python preprocess.py example.wav -c config.json --out test
```

The result is stored in `test.npz`.

Then use the latest checkpoint file in the `./saved` folder to decoded `test.npz` back to waveform.
The generating process will run on gpu if you add `--cuda`.

```
python test.py test.npz outfile.wav -r saved/your-model-name/XXXX_XXXXXX/checkpoint-stepXXXXX.pth --cuda
```

That's it. Other instructions and advanced usage can be found in [pytorch-template], I didn't change too much of the whole structure. 

## Customization
I add a new folder `feature` which is different from [pytorch-template].
To use other feature like mfcc instead of mel-spectrogram, you can add your own function in `./feature/features.py` with similar arguments style of `get_logmel()`.

Other customization method can be found in [pytorch-template].


[pytorch-template]: https://github.com/victoresque/pytorch-template


## Fast inference

In `test.py` I implement fast-wavenet generation process in a very naive way. Use `fast_inference.py` you can get a huge speed up (CPU only).
The speed is around 1500 samples/s on FFTNet and 300 samples/s on WaveNet.