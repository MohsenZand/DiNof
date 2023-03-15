"""Training NCSN++ on CIFAR-10 with SMLD."""

from configs.default_cifar10 import get_default_configs


def get_config():
    config = get_default_configs()
    # data
    data = config.data
    data.dataset = 'CIFAR10'
    data.num_channels = 3
    data.image_size = 32

    # training
    training = config.training
    training.sde = 'vesde'
    training.continuous = False

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'

    # model
    model = config.model
    model.name = 'ncsnpp'
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'residual'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.0
    model.embedding_type = 'positional'
    model.conv_size = 3

    # flow
    flow = config.flow
    flow.num_levels = 3
    flow.flow_depth = 16
    flow.channels = data.num_channels
    flow.hidden_channels = 256
    flow.class_cond = False
    flow.num_classes = None
    flow.start_step = 0.6

    return config
