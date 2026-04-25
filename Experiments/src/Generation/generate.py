import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'Utils'))
sys.path.insert(1, UTILS_DIR)

import Diffusion
import Unet
import cfg
import loader
import TinyModels as TM

# Parse arguments (Ngen, device, DATASET, N, n_base, save_every)
parser = argparse.ArgumentParser("Generation of samples from trained diffusion models.")

parser.add_argument("-n", "--num", help="Number of training data", type=int, required=True)
parser.add_argument("-i", "--index", help="Index for the dataset (0 or 1)", type=int, default=0)
parser.add_argument("-s", "--img_size", help="Size of the images used to train", type=int, required=True)
parser.add_argument("-LR", "--learning_rate", help="Learning rate for optimization", type=float, required=True)
parser.add_argument("-O", "--optim", help="Optimisation type (SGD_Momentum or Adam)", type=str, required=True)
parser.add_argument("-W", "--nbase", help="Number of base filters", type=str, required=True)
parser.add_argument("-t", "--time", help="Diffusion timestep", type=int, default=-1)
parser.add_argument(
    "-B",
    "--batch_size",
    type=int,
    help="Batch size used to train the model",
    required=True,
)
parser.add_argument(
    "-D",
    "--dataset",
    type=str,
    help="Dataset used to train the model (CelebA or Sprites).",
    required=True,
)
parser.add_argument(
    "-Ns",
    "--Nsamples",
    type=int,
    help="Number of samples to generate (should be multiple of 100).",
    required=True,
)
parser.add_argument(
    "--model_type",
    type=str,
    help="Model backbone for Sprites (unet or gmm).",
    default="unet",
)
parser.add_argument(
    "--seed",
    type=int,
    help="Seed used to sample the sprites subset.",
    default=1,
)
parser.add_argument(
    "--tag",
    type=str,
    help="Optional tag appended to the sprites experiment folder.",
    default="",
)
parser.add_argument(
    "--timesteps",
    type=int,
    help="Number of diffusion timesteps used during training (optional).",
    default=None,
)
parser.add_argument(
    "--device",
    type=str,
    help="Device used to load and apply the model.",
    default="cuda:0",
)

args = parser.parse_args()
print(args)

DATASET = args.dataset.strip()
dataset_key = DATASET.lower()
n_base = int(args.nbase)
Nsamples = int(args.Nsamples)
size = int(args.img_size)
index = int(args.index)

if dataset_key == 'celeba':
    config = cfg.load_config('CelebA')
    config.IMG_SHAPE = (1, size, size)
    config.n_images = int(args.num)
    config.BATCH_SIZE = int(args.batch_size)
    config.OPTIM = args.optim
    config.LR = float(args.learning_rate)
    config.DEVICE = args.device
    time_step = int(args.time)
    if time_step == -1:
        mode = 'normal'
    else:
        mode = 'fixed_time'
    config.mode = mode
    config.time_step = time_step
    if args.timesteps is not None:
        config.TIMESTEPS = int(args.timesteps)

    if mode == 'normal':
        type_model = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}/'.format(
            config.DATASET,
            size,
            config.n_images,
            n_base,
            config.OPTIM,
            config.BATCH_SIZE,
            config.LR,
            index,
        )
    else:
        type_model = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}_t{:d}/'.format(
            config.DATASET,
            size,
            config.n_images,
            n_base,
            config.OPTIM,
            config.BATCH_SIZE,
            config.LR,
            index,
            time_step,
        )
elif dataset_key == 'sprites':
    config = Diffusion.TrainingConfig()
    config.DATASET = 'Sprites'
    config.path_save = '../../Saves/'
    config.path_data = '../../Data/'
    config.IMG_SHAPE = (3, size, size)
    config.n_images = int(args.num)
    config.BATCH_SIZE = int(args.batch_size)
    config.OPTIM = args.optim
    config.LR = float(args.learning_rate)
    config.DEVICE = args.device
    time_step = int(args.time)
    if time_step == -1:
        mode = 'normal'
    else:
        mode = 'fixed_time'
    config.mode = mode
    config.time_step = time_step
    if args.timesteps is not None:
        config.TIMESTEPS = int(args.timesteps)

    model_type = args.model_type.lower()
    seed = int(args.seed)
    tag = args.tag.strip()

    if mode == 'normal':
        type_model = '{:s}_{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_seed{:d}/'.format(
            model_type,
            config.DATASET,
            size,
            config.n_images,
            n_base,
            config.OPTIM,
            config.BATCH_SIZE,
            config.LR,
            seed,
        )
    else:
        type_model = '{:s}_{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_seed{:d}_t{:d}/'.format(
            model_type,
            config.DATASET,
            size,
            config.n_images,
            n_base,
            config.OPTIM,
            config.BATCH_SIZE,
            config.LR,
            seed,
            time_step,
        )

    if tag:
        type_model = type_model[:-1] + '_{:s}/'.format(tag)
else:
    raise ValueError('Unknown dataset {:s}'.format(DATASET))

if not Nsamples % 100 == 0:
    raise TypeError('Nsamples should be a multiple of 100.')

# Load diffusion config for these data
df = Diffusion.DiffusionConfig(
    n_steps                 = config.TIMESTEPS,
    img_shape               = config.IMG_SHAPE,
    device                  = config.DEVICE,
)

class FlatImageTimeModel(nn.Module):
    def __init__(self, channels, height, width, d_model):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.flat_dim = channels * height * width
        self.backbone = TM.SimpleTimeModel(d=self.flat_dim, d_model=d_model)

    def forward(self, x, t):
        x_flat = x.view(x.shape[0], -1)
        y_flat = self.backbone(x_flat, t)
        return y_flat.view(x.shape[0], self.channels, self.height, self.width)


if dataset_key == 'sprites':
    if model_type == 'unet':
        model_diffusion = Unet.UNet(
            input_channels=config.IMG_SHAPE[0],
            output_channels=config.IMG_SHAPE[0],
            base_channels=n_base,
            base_channels_multiples=(1, 2, 4),
            apply_attention=(False, True, True),
            dropout_rate=0.1,
        )
    elif model_type == 'gmm':
        model_diffusion = FlatImageTimeModel(
            channels=config.IMG_SHAPE[0],
            height=config.IMG_SHAPE[1],
            width=config.IMG_SHAPE[2],
            d_model=n_base,
        )
    else:
        raise ValueError('Unknown model_type {:s}'.format(model_type))
else:
    model_diffusion = Unet.UNet(
        input_channels=config.IMG_SHAPE[0],
        output_channels=config.IMG_SHAPE[0],
        base_channels=n_base,
        base_channels_multiples=(1, 2, 4),
        apply_attention=(False, True, True),
        dropout_rate=0.1,
    )

model_diffusion.to(config.DEVICE)

print('Generating {:d} samples'.format(Nsamples))

# Generate samples
batch_gen = 100
Ns = Nsamples // batch_gen

# Define the training times to sample models (use existing checkpoints)
path_models = config.path_save + type_model + '/Models/'
checkpoint_files = glob.glob(os.path.join(path_models, 'Model_*'))
training_times = sorted({int(os.path.basename(f).split('_')[-1]) for f in checkpoint_files})
if len(training_times) == 0:
    raise FileNotFoundError('No checkpoints found in {:s}'.format(path_models))

# Loop over training times
for (j, checkpoint_id) in enumerate(training_times):
    print(r'Training time = {:d} ({:d}/{:d})'.format(checkpoint_id, j, len(training_times)))
    
    # Load the model
    model_suffix = '/Model_{:d}'.format(checkpoint_id)
    path_model_diffusion = config.path_save + type_model + '/Models/' + model_suffix
    try:
        model_diffusion = loader.load_model(model_diffusion, path_model_diffusion)
    except Exception as exc:
        raise RuntimeError(
            'Failed to load checkpoint {:s}: {:s}'.format(path_model_diffusion, str(exc))
        )
    
    # Loop for generation at the current checkpoint
    for i in range(0, Ns):
        path_save = config.path_save + type_model + '/Samples/' + '{:d}/'.format(checkpoint_id)
        doesExist = os.path.exists(path_save)
        if not doesExist:
            os.makedirs(path_save)
        
        print('Sample {:d}/{:d}'.format(i, Ns))
        samples_gen, samples_init = Diffusion.sample_diffusion_from_noise_DDIM(model_diffusion,
                                            n_images=batch_gen,
                                            config=config,
                                            df=df,
                                            dim=4,
                                            eta=0.0,            # Deterministic trajectories
                                            ddim_steps=100)     # Number of steps reduced (much faster)
        # Save initial samples
        path = path_save + str(config.TIMESTEPS)
        # Create dir if does not exist
        doesExist = os.path.exists(path)
        if not doesExist:
            os.makedirs(path)
        torch.save(samples_init, path + '/samples_a_{:d}'.format(i))
        
        # Save the generated image
        path = path_save + 'generated'
        # Create dir if does not exist
        doesExist = os.path.exists(path)
        if not doesExist:
            os.makedirs(path)
        torch.save(samples_gen, path + '/samples_a_{:d}'.format(i))

print('Done!')
