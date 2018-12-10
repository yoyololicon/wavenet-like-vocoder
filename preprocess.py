import numpy as np
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import os
from tqdm import tqdm
from functools import partial
import feature.features as feature_fn


def main(config, root, data_dir, n_workers):
    # setup data_loader instances
    fn = partial(getattr(feature_fn, config['feature']['type']), **config['feature']['args'])
    sr = config['feature']['args']['sr']
    hop_size = config['feature']['args']['hop_length']

    data_dict = {'sr': np.array([sr])}
    feature_dict = {'hop_size': np.array([hop_size])}
    with ProcessPoolExecutor(n_workers) as executor:
        futures = [executor.submit(fn, os.path.join(root, f)) for f in os.listdir(root) if f.endswith('.wav')]
        for future in tqdm(futures):
            name, y, h = future.result()
            data_dict[name] = y
            feature_dict[name] = h
    os.makedirs(data_dir, exist_ok=True)
    np.savez(os.path.join(data_dir, 'data'), **data_dict)
    np.savez(os.path.join(data_dir, 'feature'), **feature_dict)


def single_file(config, infile, outfile, dur):
    sr = config['feature']['args']['sr']
    hop_size = config['feature']['args']['hop_length']

    *_, feature = getattr(feature_fn, config['feature']['type'])(infile, **config['feature']['args'])
    if dur:
        maxlen = int(sr * dur / hop_size) + 1
        feature = feature[:, :maxlen]
    feature_dict = {'hop_size': np.array([hop_size]), 'sr': np.array([sr])}
    feature_dict['feature'] = feature

    np.savez(outfile, **feature_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessor')
    parser.add_argument('root', type=str, help='can be a single wave file or a directory')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-n', '--n_worker', default=2, type=int, help='number of processes')
    parser.add_argument('--out', type=str, default='data')
    parser.add_argument('-d', '--duration', type=float)
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if os.path.isdir(args.root):
        main(config, args.root, args.out, args.n_worker)
    else:
        single_file(config, args.root, args.out, args.duration)
