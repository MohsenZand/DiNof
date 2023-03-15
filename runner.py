import gc
import io
import os

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
from absl import flags
import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from models import normflows as nf
from models.ema import ExponentialMovingAverage
from losses import get_optimizer, optimization_manager, get_step_fn
from utils import save_checkpoint, restore_checkpoint
from datasets import get_dataset, get_data_scaler, get_data_inverse_scaler
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import get_sampling_fn
from likelihood import get_likelihood_fn
from evaluation import get_inception_model, run_inception_distributed, load_dataset_stats
from utils import saveImage


args = flags.FLAGS

def init_models(config, logger):
    model_name = config.model.name
    if model_name == 'ncsnpp':
        from models import ncsnpp
        score_model = ncsnpp.NCSNpp(config)
    elif model_name == 'ddpm':
        from models import ddpm
        score_model = ddpm.DDPM(config)
    else:
        raise NotImplementedError(f"model {model_name} is unknown!")

    score_model = score_model.to(config.device)
    
    logger.info('SCORE MODEL: {}'.format(model_name.upper()))

    # Set up flows, distributions and merge operations
    L, K = config.flow.num_levels, config.flow.flow_depth 
    q0, merges, flows = [],[], []
    for i in range(L):
        flows_ = []
        for j in range(K):
            flows_ += [nf.flows.GlowBlock(config.flow.channels * 2 ** (L + 1 - i), config.flow.hidden_channels, 
            split_mode=config.flow.split_mode, scale=config.flow.scale)]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        latent_shape = (config.flow.input_shape[0] * 2 ** (L - i), config.flow.input_shape[1] // 2 ** (L - i), config.flow.input_shape[2] // 2 ** (L - i))
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (config.flow.input_shape[0] * 2 ** (L - i), config.flow.input_shape[1] // 2 ** (L - i), config.flow.input_shape[2] // 2 ** (L - i))
        else:
            latent_shape = (config.flow.input_shape[0] * 2 ** (L + 1), config.flow.input_shape[1] // 2 ** L, config.flow.input_shape[2] // 2 ** L)
        if config.flow.class_cond:
            q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, config.flow.num_classes)]
        else:                
            q0 += [nf.distributions.GlowBase(latent_shape)]

    # Construct flow model
    flow = nf.MultiscaleFlow(q0, flows, merges, class_cond=config.flow.class_cond)
    flow = flow.to(config.device)

    logger.info('FLOW MODEL: GLOW with L={}, K={}'.format(config.flow.num_levels, config.flow.flow_depth))

    ema_0 = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    ema_1 = ExponentialMovingAverage(flow.parameters(), decay=config.model.ema_rate)

    return score_model, flow, ema_0, ema_1


def train(config, writer, checkpoint_dir, image_dir, logger):
    """Traininig models.

    Args:
      config: Configuration to use.
      writer: Tensorboard writer 
      checkpoint_dir: Working directory for checkpoints.
      image_dir: The subfolder for storing images.
    """
    # Initialize model
    score_model, flow, ema_0, ema_1 = init_models(config, logger)
    
    parameters = list(score_model.parameters()) + list(flow.parameters())
    optimizer = get_optimizer(config, parameters)
    state = dict(optimizer=optimizer, model=[score_model, flow], ema=[ema_0, ema_1], step=0)
    if args.resume_training:
        load_chkpt = args.resume_ckpt
        state = restore_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{load_chkpt}.pth'), state, config.device, logger=logger)
    
    initial_step = int(state['step'])

    # Build data iterators
    with tf.device('/cpu:0'):
        train_ds, eval_ds, _ = get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)  
    eval_iter = iter(eval_ds)  
    
    # Create data normalizer and its inverse
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)
    
    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    # Build one-step training and evaluation functions
    optimize_fn = optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    
    train_step_fn = get_step_fn(sde, train=True, optimize_fn=optimize_fn, reduce_mean=reduce_mean, 
                        continuous=continuous, likelihood_weighting=likelihood_weighting,
                        start=config.flow.start_step)
    eval_step_fn = get_step_fn(sde, train=False, optimize_fn=optimize_fn, reduce_mean=reduce_mean, 
                        continuous=continuous, likelihood_weighting=likelihood_weighting, 
                        start=config.flow.start_step)

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.eval.batch_size, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, start=config.flow.start_step)

    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logger.info("Starting training loop at step %d." % (initial_step,))
    pbar = tqdm(range(initial_step, num_train_steps + 1))
    for step in pbar:
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
        batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
        batch = batch.permute(0, 3, 1, 2)
        batch = scaler(batch)

        # Execute one training step
        losses = train_step_fn(state, batch)
        
        if step % config.training.log_freq == 0:
            logger.info("step: %d, training_loss: %.5e" % (step, (losses[0]+losses[1]).item())) 
            writer.add_scalar("Score model training loss", losses[0], step)
            writer.add_scalar("Flow training loss", losses[1], step)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
            eval_batch = eval_batch.permute(0, 3, 1, 2)
            eval_batch = scaler(eval_batch)
            eval_losses = eval_step_fn(state, eval_batch)
            logger.info("step: %d, eval_loss: %.5e" % (step, (eval_losses[0]+eval_losses[1]).item())) 
            writer.add_scalar("Score model eval loss", eval_losses[0].item(), step)
            writer.add_scalar("Flow eval loss", eval_losses[1].item(), step)
        
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate and save samples
            if config.training.snapshot_sampling:
                ema_0.store(score_model.parameters())
                ema_0.copy_to(score_model.parameters())
                ema_1.store(flow.parameters())
                ema_1.copy_to(flow.parameters())
                
                sample, _ = sampling_fn([score_model, flow])

                ema_0.restore(score_model.parameters())
                ema_1.restore(flow.parameters())
                this_sample_dir = os.path.join(image_dir, "iter_{}".format(step))
                tf.io.gfile.makedirs(this_sample_dir)

                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(sample, nrow, padding=2)
                sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)

                with tf.io.gfile.GFile(os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                    save_image(image_grid, fout)


def evaluate(config, checkpoint_dir, eval_dir, logger, ckpt_s):
    """Evaluate trained models.

    Args:
      config: Configuration to use.
      checkpoint_dir: Working directory for checkpoints.
      eval_dir: The subfolder for storing evaluation results.
      ckpt_s = The starting checkpoint number for evaluation
    """
    
    # Initialize model
    score_model, flow, ema_0, ema_1 = init_models(config, logger)
    parameters = list(score_model.parameters()) + list(flow.parameters())
    optimizer = get_optimizer(config, parameters)
    state = dict(optimizer=optimizer, model=[score_model, flow], ema=[ema_0, ema_1], step=0)
    
    # Build data pipeline
    with tf.device('/cpu:0'):
        _, eval_ds, _ = get_dataset(config, uniform_dequantization=config.data.uniform_dequantization, evaluation=True)
    
    # Create data normalizer and its inverse
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Create the one-step evaluation function when loss computation is enabled
    if config.eval.enable_loss:
        optimize_fn = optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting

        reduce_mean = config.training.reduce_mean
        
        eval_step = get_step_fn(sde, train=False, optimize_fn=optimize_fn, reduce_mean=reduce_mean,  
                        continuous=continuous, likelihood_weighting=likelihood_weighting, 
                        start=config.flow.start_step)

    if config.eval.enable_bpd:
        # Build the likelihood computation function when likelihood is enabled
        likelihood_fn = get_likelihood_fn(sde, inverse_scaler)

        # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
        with tf.device('/cpu:0'):
            train_ds_bpd, eval_ds_bpd, _ = get_dataset(config, uniform_dequantization=True, evaluation=True)
        if config.eval.bpd_dataset.lower() == 'train':
            ds_bpd = train_ds_bpd
            bpd_num_repeats = 1
        elif config.eval.bpd_dataset.lower() == 'test':
            # Go over the dataset 5 times when computing likelihood on the test dataset
            ds_bpd = eval_ds_bpd
            bpd_num_repeats = 5
        else:
            raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (config.eval.batch_size,
                          config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, start=config.flow.start_step)

    if config.data.dataset != 'CelebAHQ':
        # Use inceptionV3 for images with resolution higher than 256.
        inceptionv3 = config.data.image_size >= 256
        inception_model = get_inception_model(inceptionv3=inceptionv3)
        
    for ckpt in range(ckpt_s, 30):
        try:
            ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
            state = restore_checkpoint(ckpt_path, state, device=config.device, logger=logger)
            if state['step'] == 0:
                exit()

            ema_0.copy_to(score_model.parameters())
            ema_1.copy_to(flow.parameters())

            # Compute the loss function on the full evaluation dataset if loss computation is enabled
            if config.eval.enable_loss:
                all_losses_1 = []
                all_losses_2 = []
                eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
                for i, batch in tqdm(enumerate(eval_iter), total=len(eval_ds)):
                    eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
                    eval_batch = eval_batch.permute(0, 3, 1, 2)
                    eval_batch = scaler(eval_batch)
                    eval_loss = eval_step(state, eval_batch)
                    all_losses_1.append(eval_loss[0].item())
                    all_losses_2.append(eval_loss[1].item())
                    if (i + 1) % 1000 == 0:
                        logger.info("Finished %dth step loss evaluation" % (i + 1))

                # Save loss values to disk or Google Cloud Storage
                all_losses_1 = np.asarray(all_losses_1)
                all_losses_2 = np.asarray(all_losses_2)
                with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss_1.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, all_losses=all_losses_1, mean_loss=all_losses_1.mean())
                    fout.write(io_buffer.getvalue())
                with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss_2.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, all_losses=all_losses_2, mean_loss=all_losses_2.mean())
                    fout.write(io_buffer.getvalue())

            # Compute log-likelihoods (bits/dim) if enabled
            if config.eval.enable_bpd:
                bpds = []
                for repeat in range(bpd_num_repeats):
                    bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
                    for batch_id in range(len(ds_bpd)):
                        batch = next(bpd_iter)
                        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
                        eval_batch = eval_batch.permute(0, 3, 1, 2)
                        eval_batch = scaler(eval_batch)
                        bpd = likelihood_fn([score_model, flow], eval_batch)[0]
                        bpd = bpd.detach().cpu().numpy().reshape(-1)
                        bpds.extend(bpd)
                        logger.info("ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
                        bpd_round_id = batch_id + len(ds_bpd) * repeat
                        # Save bits/dim to disk or Google Cloud Storage
                        with tf.io.gfile.GFile(os.path.join(eval_dir, 
                                                f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                                "wb") as fout:
                            io_buffer = io.BytesIO()
                            np.savez_compressed(io_buffer, bpd)
                            fout.write(io_buffer.getvalue())

            # Generate samples and compute IS/FID/KID when enabled
            if config.eval.enable_sampling:
                this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
                tf.io.gfile.makedirs(this_sample_dir)

                gen_samples = []
                num_sampling_rounds = (config.eval.num_samples - 1) // config.eval.batch_size + 1

                # If interuppted and restarted, 
                # load all statistics that have been previously computed and saved
                num_saved = 0
                smpls = tf.io.gfile.glob(os.path.join(this_sample_dir, "samples_*.npz"))
                for smpl_file in smpls:
                    with tf.io.gfile.GFile(smpl_file, "rb") as fin:
                        smpl = np.load(fin)
                        gen_samples.append(torch.from_numpy(smpl['samples']).to(config.device).permute(0, 3, 1, 2) / 255.)
                num_saved = len(gen_samples)

                for r in range(num_saved, num_sampling_rounds):
                    logger.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

                    final_samples, n = sampling_fn([score_model, flow])
                    gen_samples.append(final_samples)

                    samples = np.clip(final_samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
                    samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
                    # Write samples to disk or Google Cloud Storage
                    with tf.io.gfile.GFile(os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, samples=samples)
                        fout.write(io_buffer.getvalue())

                    if config.data.dataset != 'CelebAHQ':
                        # Force garbage collection before calling TensorFlow code for Inception network
                        gc.collect()
                        latents = run_inception_distributed(samples, inception_model, inceptionv3=inceptionv3)
                        # Force garbage collection again before returning to JAX code
                        gc.collect()
                        # Save latent represents of the Inception network to disk or Google Cloud Storage
                        with tf.io.gfile.GFile(os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
                            io_buffer = io.BytesIO()
                            np.savez_compressed(io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
                            fout.write(io_buffer.getvalue())

                gen_samples = torch.stack(gen_samples, dim=0)
                if config.data.dataset == 'CelebAHQ':
                    from fids.fid_score import evaluate_fid
                    fid = evaluate_fid(config, gen_samples, logger)
                    inception_score, kid = 0.0, 0.0
                else:
                    # Compute inception scores, FIDs and KIDs.
                    # Load all statistics that have been previously computed and saved for each host
                    all_logits = []
                    all_pools = []
                    stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
                    for stat_file in stats:
                        with tf.io.gfile.GFile(stat_file, "rb") as fin:
                            stat = np.load(fin)
                            if not inceptionv3:
                                all_logits.append(stat["logits"])
                            all_pools.append(stat["pool_3"])

                    if not inceptionv3:
                        all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
                    all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

                    # Load pre-computed dataset statistics.
                    data_stats = load_dataset_stats(config)
                    data_pools = data_stats["pool_3"]

                    # Compute FID/KID/IS on all samples together.
                    if not inceptionv3:
                        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
                    else:
                        inception_score = -1

                    fid = tfgan.eval.frechet_classifier_distance_from_activations(data_pools, all_pools)
                    # Hack to get tfgan KID work for eager execution.
                    tf_data_pools = tf.convert_to_tensor(data_pools)
                    tf_all_pools = tf.convert_to_tensor(all_pools)
                    kid = tfgan.eval.kernel_classifier_distance_from_activations(tf_data_pools, tf_all_pools).numpy()
                    del tf_data_pools, tf_all_pools

                logger.info("ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (ckpt, inception_score, fid, kid))

                with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),"wb") as f:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
                    f.write(io_buffer.getvalue())

                gen_samples = gen_samples.reshape(-1, gen_samples.shape[2], gen_samples.shape[3], gen_samples.shape[4])
                saveImage(config, samples=gen_samples[:64], save_dir=this_sample_dir, img_id=0)
        except:
            continue
    
    
        