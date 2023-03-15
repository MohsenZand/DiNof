# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import os
from absl import flags, app
from fid_score import compute_statistics_of_generator, save_statistics
from inception import InceptionV3
from itertools import chain
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
import sys
sys.path.append(str(ROOT))
from datasets import get_loaders
from ml_collections.config_flags import config_flags


args = flags.FLAGS
config_flags.DEFINE_config_file('config', str(ROOT / 'configs' / 'celebahq.py'), 
                                    'Training configuration.', lock_config=True)
flags.DEFINE_integer('batch_size', 64, 'batch size per GPU')



def main(argv):
    config = args.config
    device = 'cuda'
    dims = 2048
    # for binary datasets including MNIST and OMNIGLOT, we don't apply binarization for FID computation
    train_queue, valid_queue, _ = get_loaders(config.data.dataset, config)
    print('len train queue', len(train_queue), 'len val queue', len(valid_queue), 'batch size', args.batch_size)
    if config.data.dataset in {'celeba_256', 'omniglot'}:
        train_queue = chain(train_queue, valid_queue)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=config.eval.fid_dir).to(device)
    m, s = compute_statistics_of_generator(train_queue, model, args.batch_size, dims, device, config.eval.num_samples)
    file_path = os.path.join(config.eval.fid_dir, config.data.dataset + '.npz')
    print('saving fid stats at %s' % file_path)
    save_statistics(file_path, m, s)


if __name__ == '__main__':
    app.run(main)