import torch
import tensorflow as tf
import os
import numpy as np 
from torchvision.utils import make_grid, save_image



def restore_checkpoint(ckpt_dir, state, device, logger):
    if not tf.io.gfile.exists(ckpt_dir):
        tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
        logger.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'][0].load_state_dict(loaded_state['model_0'], strict=False)
        state['model'][1].load_state_dict(loaded_state['model_1'], strict=False)
        state['ema'][0].load_state_dict(loaded_state['ema_0'])
        state['ema'][1].load_state_dict(loaded_state['ema_1'])
        state['step'] = loaded_state['step']
        logger.info(f"Restored checkpoint: {ckpt_dir}")
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model_0': state['model'][0].state_dict(),
        'model_1': state['model'][1].state_dict(),
        'ema_0': state['ema'][0].state_dict(),
        'ema_1': state['ema'][1].state_dict(),
        'step': state['step']
        }
    torch.save(saved_state, ckpt_dir)


def saveImage(config, samples, save_dir, img_id):
    def stretch_image(X, ch, imsize):
        return X.reshape(len(X), -1, ch, imsize, imsize).permute(0, 2, 1, 4, 3).reshape(len(X), ch, -1, imsize).permute(0, 1, 3, 2)

    if samples.shape[1] != 3:
        gen_samples_to_save = stretch_image(torch.from_numpy(samples/255).permute(0,3,1,2)[:config.eval.batch_size], config.data.channels, config.data.image_size)
    else:
        gen_samples_to_save = stretch_image(samples, config.data.channels, config.data.image_size)
    
    nrow = int(np.sqrt(gen_samples_to_save.shape[0]))
    image_grid = make_grid(gen_samples_to_save, nrow, padding=2)

    with tf.io.gfile.GFile(os.path.join(save_dir, f"img_{img_id}.png"), "wb") as fout:
        save_image(image_grid, fout, format='png')
