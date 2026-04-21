#%%
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(1, '../Utils/')  # In case we run from Experiments/src/Training
import Unet
import Plot
import Diffusion
import loader
import cfg
import sprites_dataset
import TinyModels as TM


class NormalizedDataset(Dataset):
    def __init__(self, base_dataset, mean, std):
        self.base_dataset = base_dataset
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image = self.base_dataset[idx]
        return (image - self.mean) / self.std


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


def compute_channel_stats(dataset, batch_size, num_workers=2):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    channel_sum = None
    channel_squared_sum = None
    n_pixels = 0

    for images in data_loader:
        images = images.float()
        b, c, h, w = images.shape
        images = images.view(b, c, -1)

        if channel_sum is None:
            channel_sum = torch.zeros(c, dtype=images.dtype)
            channel_squared_sum = torch.zeros(c, dtype=images.dtype)

        channel_sum += images.sum(dim=(0, 2))
        channel_squared_sum += (images ** 2).sum(dim=(0, 2))
        n_pixels += b * h * w

    mean = channel_sum / n_pixels
    var = torch.clamp(channel_squared_sum / n_pixels - mean ** 2, min=1e-12)
    std = torch.sqrt(var)
    return mean, std


#%%
parser = argparse.ArgumentParser("Diffusion on sprites dataset with U-Net.")
parser.add_argument("-n", "--num", help="Number of training images", type=int, required=True)
parser.add_argument("-s", "--img_size", help="Image size after resizing", type=int, default=16)
parser.add_argument("-LR", "--learning_rate", help="Learning rate for optimization", type=float, required=True)
parser.add_argument("-O", "--optim", help="Optimisation type (SGD_Momentum or Adam)", type=str, required=True)
parser.add_argument("-W", "--nbase", help="Number of base filters", type=int, required=True)
parser.add_argument("-B", "--batch_size", help="Batch size", type=int, default=512)
parser.add_argument("-t", "--time", help="Diffusion timestep", type=int, default=-1)
parser.add_argument(
    "-M",
    "--model_type",
    help="Model backbone (Unet or GMM)",
    type=str,
    default="Unet",
    choices=["Unet", "GMM", "unet", "gmm"],
)
parser.add_argument("--seed", help="Seed used to sample the training subset", type=int, default=1)
parser.add_argument("--n_steps", help="Number of optimization steps", type=int, default=int(2e6))
parser.add_argument("--timesteps", help="Number of diffusion timesteps", type=int, default=1000)
parser.add_argument("--device", help="Training device (e.g., cuda:0 or cpu)", type=str, default=None)
parser.add_argument(
    "--data_file",
    help="Path to sprites .npy file",
    type=str,
    default="../../Data/sprites_1788_16x16.npy",
)
args = vars(parser.parse_args())
print(args)


# Get arguments
n = args['num']
size = args['img_size']
lr = args['learning_rate']
optim = args['optim']
n_base = args['nbase']
batch_size = args['batch_size']
time_step = args['time']
seed = args['seed']
n_steps = args['n_steps']
timesteps = args['timesteps']
data_file = args['data_file']
device_arg = args['device']
model_type = args['model_type'].lower()

if time_step == -1:
    mode = 'normal'
else:
    mode = 'fixed_time'

if device_arg is None:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
else:
    device = device_arg

if not os.path.exists(data_file):
    raise FileNotFoundError('Could not find sprites data file at {:s}'.format(data_file))

sprites_raw = np.load(data_file, mmap_mode='r')
n_total = sprites_raw.shape[0]
del sprites_raw
if n > n_total:
    raise ValueError('Requested n={:d} images but dataset only has {:d}.'.format(n, n_total))


# Overwrite config with command line arguments
config = Diffusion.TrainingConfig()
config.DATASET = 'Sprites'
config.path_save = '../../Saves/'
config.path_data = '../../Data/'
config.n_images = n
config.BATCH_SIZE = min(batch_size, n)
config.N_STEPS = n_steps
config.OPTIM = optim
config.LR = lr
config.mode = mode
config.time_step = time_step
config.DEVICE = device
config.TIMESTEPS = timesteps
config.CENTER = True
config.STANDARDIZE = False

base_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((size, size)),
    ]
)
base_dataset = sprites_dataset.SpritesDataset(
    transform=base_transform,
    img_file=data_file,
    num_samples=n,
    seed=seed,
)

example = base_dataset[0]
config.IMG_SHAPE = tuple(example.shape)

if config.mode == 'normal':
    suffix = '{:s}_{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_seed{:d}/'.format(
        model_type,
        config.DATASET,
        config.IMG_SHAPE[1],
        config.n_images,
        n_base,
        config.OPTIM,
        config.BATCH_SIZE,
        config.LR,
        seed,
    )
elif config.mode == 'fixed_time':
    suffix = '{:s}_{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_seed{:d}_t{:d}/'.format(
        model_type,
        config.DATASET,
        config.IMG_SHAPE[1],
        config.n_images,
        n_base,
        config.OPTIM,
        config.BATCH_SIZE,
        config.LR,
        seed,
        time_step,
    )
    print('Training at fixed diffusion time: {:d}'.format(config.time_step))


# Create path to images and model save
path_images = config.path_save + suffix + 'Images/'
path_models = config.path_save + suffix + 'Models/'
os.makedirs(path_images, exist_ok=True)
os.makedirs(path_models, exist_ok=True)

os.system('cp run_Sprites.py {:s}'.format(path_models + '_run_Sprites.py'))
os.system('cp ../Utils/sprites_dataset.py {:s}'.format(path_models + '_sprites_dataset.py'))
os.system('cp ../Utils/cfg.py {:s}'.format(path_models + '_cfg.py'))

if config.CENTER:
    mean, std = compute_channel_stats(base_dataset, batch_size=config.BATCH_SIZE)
    if not config.STANDARDIZE:
        std = torch.ones_like(std)
    train_images = NormalizedDataset(base_dataset, mean, std)
else:
    mean = torch.zeros(config.IMG_SHAPE[0])
    std = torch.ones(config.IMG_SHAPE[0])
    train_images = base_dataset

config.mean = mean
config.std = std


if __name__ == '__main__':
    trainloader = torch.utils.data.DataLoader(
        train_images,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )


# In[] Plot one random batch of training images
dataiter = iter(trainloader)
images = next(dataiter)

Plot.imshow(images[0:32].cpu(), config.mean, config.std)
plt.savefig(path_images + 'Training_set.pdf', bbox_inches='tight')


# In[] Model definition
if __name__ == '__main__':
    if model_type == 'unet':
        model = Unet.UNet(
            input_channels=config.IMG_SHAPE[0],
            output_channels=config.IMG_SHAPE[0],
            base_channels=n_base,
            base_channels_multiples=(1, 2, 4),
            apply_attention=(False, True, True),
            dropout_rate=0.1,
        )
    elif model_type == 'gmm':
        model = FlatImageTimeModel(
            channels=config.IMG_SHAPE[0],
            height=config.IMG_SHAPE[1],
            width=config.IMG_SHAPE[2],
            d_model=n_base,
        )
    else:
        raise ValueError('Unknown model_type {:s}'.format(model_type))

    # Resume training from last weights in the folder
    weights_files = glob.glob(os.path.join(path_models, 'Model_*'))
    if weights_files:
        offset = max([int(os.path.basename(f).split('_')[-1]) for f in weights_files])
    else:
        offset = 0

    if offset > 0:
        path_checkpoint = os.path.join(path_models, 'Model_{:d}'.format(offset))
        model = loader.load_model(model, path_checkpoint)

    if model_type == 'unet' and torch.cuda.is_available() and config.DEVICE.startswith('cuda') and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.to(config.DEVICE)


if __name__ == '__main__':
    n_params = sum(p.numel() for p in model.parameters())
    print('{:.2f}M'.format(n_params / 1e6))


# In[] Training and saving
if __name__ == '__main__':
    if config.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    elif config.OPTIM == 'SGD_Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.LR, momentum=0.95)
    else:
        raise ValueError('Unknown optimizer {:s}'.format(config.OPTIM))

    df = Diffusion.DiffusionConfig(
        n_steps=config.TIMESTEPS,
        img_shape=config.IMG_SHAPE,
        device=config.DEVICE,
    )
    loss_fn = nn.MSELoss()

    sweeping = 1.0

    # Saving times for the model during training
    times_save = cfg.get_training_times()
    times_save = times_save[times_save <= config.N_STEPS]

    Diffusion.train(
        model,
        trainloader,
        optimizer,
        config,
        df,
        loss_fn,
        sweeping,
        times_save,
        offset,
        suffix,
        generate=True,
    )