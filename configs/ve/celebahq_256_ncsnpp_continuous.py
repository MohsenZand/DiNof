"""Training NCSN++ on CelebAHQ with VE SDE."""

from configs.default_celeba import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = 'vesde'
    training.continuous = True

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'

    # data
    data = config.data
    data.dataset = 'CelebAHQ'
    data.image_size = 256
    data.tfrecords_path = ''

    # model
    model = config.model
    model.name = 'ncsnpp'
    model.sigma_max = 348
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'output_skip'
    model.progressive_input = 'input_skip'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conv_size = 3

    # flow
    flow = config.flow
    flow.num_levels = 3
    flow.flow_depth = 16
    flow.channels = data.num_channels
    flow.hidden_channels = 256
    flow.class_cond = False
    flow.num_classes = None
    flow.input_shape = [flow.channels, data.image_size, data.image_size]
    flow.start_step = 0.6

    return config
