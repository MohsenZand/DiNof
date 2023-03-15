import os
from absl import flags, app
import logging
from colorlog import ColoredFormatter
import tensorflow as tf
from ml_collections.config_flags import config_flags
from torch.utils import tensorboard
from pathlib import Path

import runner

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  

import sys
sys.path.append(str(ROOT))

args = flags.FLAGS

VE_VP_SUBVP = 've'
CONFIG_file = 'cifar10_ncsnpp.py' 
'''
config files and their corresponding SDEs
ve:
    cifar10_ncsnpp.py
    cifar10_ncsnpp_continuous.py
    cifar10_ncsnpp_deep_continuous.py
    celebahq_256_ncsnpp_continuous.py
vp:
    cifar10_ddpmpp.py
    cifar10_ddpmpp_continuous.py
subvp:
    cifar10_ddpmpp_continuous.py
'''
EXP_name = CONFIG_file.split('.')[0]

config_flags.DEFINE_config_file('config', 
        str(ROOT / 'configs' / VE_VP_SUBVP / CONFIG_file), 
        'Training configuration.', lock_config=True)
flags.DEFINE_enum('mode', 'train', ['train', 'eval'], 'Running mode: train or eval')

flags.DEFINE_string('workdir', PATH + EXP_name, 'Work directory.')
flags.DEFINE_string('fid_dir', str(ROOT / 'fids'), 'Path to directory containing fid_stats npz files')
flags.DEFINE_string('tfrecords_path', '', 
                    'Path to directory containing tfrecords for CelebAHQ dataset')

flags.DEFINE_bool('resume_training', False, '')
flags.DEFINE_integer('resume_ckpt', 0, 'checkpoint number if resume_training')
flags.DEFINE_integer('ckpt', 0, 'checkpoint number for evaluation')

flags.mark_flags_as_required(['workdir', 'config', 'mode'])


def main(argv):
    tf.io.gfile.makedirs(args.workdir)
    tb_dir = os.path.join(args.workdir, 'tensorboard')
    tf.io.gfile.makedirs(tb_dir)
    checkpoint_dir = os.path.join(args.workdir, 'checkpoints')
    tf.io.gfile.makedirs(checkpoint_dir)
    # Create directory to eval_folder
    eval_dir = os.path.join(args.workdir, 'eval_results')
    tf.io.gfile.makedirs(eval_dir)
    image_dir = os.path.join(eval_dir, 'images')
    tf.io.gfile.makedirs(image_dir)

    args.config.eval.fid_dir = args.fid_dir
    if args.config.data.dataset == 'CelebAHQ':
        args.config.data.tfrecords_path = args.tfrecords_path

    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)
    formatter = ColoredFormatter('  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s')
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    logger.propagate = False

    if args.mode == 'train':
        writer = tensorboard.SummaryWriter(tb_dir)
        runner.train(args.config, writer, checkpoint_dir, image_dir, logger)
    elif args.mode == 'eval':
        runner.evaluate(args.config, checkpoint_dir, eval_dir, logger, args.ckpt)
    else:
        raise ValueError(f'Mode {args.mode} not recognized.')




if __name__ == '__main__':
    app.run(main)
