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

    data_dict = dict()
    feature_dict = dict()
    with ProcessPoolExecutor(n_workers) as executor:
        futures = [executor.submit(fn, os.path.join(root, f)) for f in os.listdir(root) if f.endswith('.wav')]
        for future in tqdm(futures):
            name, y, h = future.result()
            data_dict[name] = y
            feature_dict[name] = h

    np.savez(os.path.join(data_dir, 'data'), **data_dict)
    np.savez(os.path.join(data_dir, 'feature'), **feature_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessor')
    parser.add_argument('root', type=str)
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-n', '--n_worker', default=1, type=int, help='number of processes')
    parser.add_argument('--out_dir', type=str, default='data/')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    main(config, args.root, args.out_dir, args.n_worker)
