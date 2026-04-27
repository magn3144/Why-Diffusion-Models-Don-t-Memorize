#%%
import argparse
import glob
import math
import os
import sys
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(1, '../Utils/')  # In case we run from Experiments/src/Training
import Unet
import Plot
import Diffusion
import loader
import cfg


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


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


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


def find_image_files(data_root):
    patterns = [
        os.path.join(data_root, '*.jpg'),
        os.path.join(data_root, '*.jpeg'),
        os.path.join(data_root, '*.png'),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(files)


#%%
parser = argparse.ArgumentParser('Diffusion on ISIC-2019 dataset with U-Net.')
parser.add_argument('-n', '--num', help='Number of training images', type=int, required=True)
parser.add_argument('-s', '--img_size', help='Image size after resizing', type=int, default=32)
parser.add_argument('-LR', '--learning_rate', help='Learning rate for optimization', type=float, required=True)
parser.add_argument('-O', '--optim', help='Optimisation type (SGD_Momentum or Adam)', type=str, required=True)
parser.add_argument('-W', '--nbase', help='Number of base filters for U-Net', type=int, required=True)
parser.add_argument('-B', '--batch_size', help='Batch size', type=int, default=512)
parser.add_argument('-t', '--time', help='Diffusion timestep', type=int, default=-1)
parser.add_argument('--seed', help='Seed used to sample the training subset', type=int, default=1)
parser.add_argument('--n_steps', help='Number of optimization steps', type=int, default=int(2e6))
parser.add_argument('--timesteps', help='Number of diffusion timesteps', type=int, default=1000)
parser.add_argument('--device', help='Training device (e.g., cuda:0 or cpu)', type=str, default=None)
parser.add_argument('--momentum', help='Momentum for SGD_Momentum', type=float, default=0.95)
parser.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.0)
parser.add_argument('--nesterov', help='Use Nesterov momentum with SGD_Momentum', action='store_true')
parser.add_argument(
    '--scheduler',
    help='Learning rate scheduler (none or cosine)',
    type=str,
    default='none',
    choices=['none', 'cosine'],
)
parser.add_argument('--warmup_steps', help='Number of warmup steps for cosine scheduler', type=int, default=0)
parser.add_argument(
    '--min_lr_ratio',
    help='Minimum lr ratio for cosine scheduler (final lr = min_lr_ratio * base lr)',
    type=float,
    default=0.05,
)
parser.add_argument('--grad_clip', help='Gradient clipping max norm (0 disables)', type=float, default=0.0)
parser.add_argument('--tag', help='Optional experiment tag appended to save path', type=str, default='')
parser.add_argument(
    '--multi_gpu',
    help='Enable DataParallel across all visible GPUs (disabled by default)',
    action='store_true',
)
parser.add_argument(
    '--data_root',
    help='Path to extracted ISIC image directory',
    type=str,
    default='../../Data/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
)
parser.add_argument(
    '--zip_file',
    help='Path to ISIC zip archive used for optional extraction',
    type=str,
    default='../../Data/isic-2019.zip',
)
parser.add_argument(
    '--no_auto_unzip',
    help='Disable automatic unzip when data_root is missing',
    action='store_true',
)
parser.add_argument('--num_workers', help='Number of dataloader workers', type=int, default=2)
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
device_arg = args['device']
momentum = args['momentum']
weight_decay = args['weight_decay']
nesterov = args['nesterov']
scheduler_name = args['scheduler']
warmup_steps = args['warmup_steps']
min_lr_ratio = args['min_lr_ratio']
grad_clip = args['grad_clip']
tag = args['tag'].strip()
multi_gpu = args['multi_gpu']
data_root = args['data_root']
zip_file = args['zip_file']
auto_unzip = not args['no_auto_unzip']
num_workers = max(0, args['num_workers'])

if time_step == -1:
    mode = 'normal'
else:
    mode = 'fixed_time'

if device_arg is None:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
else:
    device = device_arg

if not os.path.isdir(data_root):
    if auto_unzip and os.path.exists(zip_file):
        print('Extracting ISIC dataset from {:s} ...'.format(zip_file))
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(os.path.dirname(zip_file))
    else:
        raise FileNotFoundError(
            'Could not find extracted data_root at {:s}. '
            'Set --data_root or allow auto unzip with --zip_file.'.format(data_root)
        )

image_paths = find_image_files(data_root)
if len(image_paths) == 0:
    raise RuntimeError('No images found in {:s}'.format(data_root))
if n > len(image_paths):
    raise ValueError('Requested n={:d} images but dataset has only {:d}.'.format(n, len(image_paths)))

rng = np.random.default_rng(seed)
indices = rng.choice(len(image_paths), size=n, replace=False)
indices = np.sort(indices)
selected_paths = [image_paths[i] for i in indices]


# Overwrite config with command line arguments
config = Diffusion.TrainingConfig()
config.DATASET = 'ISIC'
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
config.GRAD_CLIP = grad_clip

base_transform = transforms.Compose(
    [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]
)
base_dataset = ImagePathDataset(selected_paths, transform=base_transform)

example = base_dataset[0]
config.IMG_SHAPE = tuple(example.shape)

if config.mode == 'normal':
    suffix = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_seed{:d}/'.format(
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
    suffix = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_seed{:d}_t{:d}/'.format(
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

if tag:
    suffix = suffix[:-1] + '_{:s}/'.format(tag)


# Create path to images and model save
path_images = config.path_save + suffix + 'Images/'
path_models = config.path_save + suffix + 'Models/'
os.makedirs(path_images, exist_ok=True)
os.makedirs(path_models, exist_ok=True)

os.system('cp run_ISIC.py {:s}'.format(path_models + '_run_ISIC.py'))
os.system('cp ../Utils/cfg.py {:s}'.format(path_models + '_cfg.py'))

if config.CENTER:
    mean, std = compute_channel_stats(base_dataset, batch_size=config.BATCH_SIZE, num_workers=num_workers)
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
    trainloader = DataLoader(
        train_images,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
    )


# In[] Plot one random batch of training images
dataiter = iter(trainloader)
images = next(dataiter)

Plot.imshow(images[0:32].cpu(), config.mean, config.std)
plt.savefig(path_images + 'Training_set.pdf', bbox_inches='tight')


# In[] Model definition
if __name__ == '__main__':
    model = Unet.UNet(
        input_channels=config.IMG_SHAPE[0],
        output_channels=config.IMG_SHAPE[0],
        base_channels=n_base,
        base_channels_multiples=(1, 2, 4),
        apply_attention=(False, True, True),
        dropout_rate=0.1,
    )

    # Resume training from last weights in the folder
    weights_files = glob.glob(os.path.join(path_models, 'Model_*'))
    if weights_files:
        offset = max([int(os.path.basename(f).split('_')[-1]) for f in weights_files])
    else:
        offset = 0

    if offset > 0:
        path_checkpoint = os.path.join(path_models, 'Model_{:d}'.format(offset))
        model = loader.load_model(model, path_checkpoint)

    if (
        multi_gpu
        and torch.cuda.is_available()
        and config.DEVICE.startswith('cuda')
        and torch.cuda.device_count() > 1
    ):
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.to(config.DEVICE)


if __name__ == '__main__':
    n_params = sum(p.numel() for p in model.parameters())
    print('{:.2f}M'.format(n_params / 1e6))


# In[] Training and saving
if __name__ == '__main__':
    if config.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=weight_decay)
    elif config.OPTIM == 'SGD_Momentum':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.LR,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    else:
        raise ValueError('Unknown optimizer {:s}'.format(config.OPTIM))

    scheduler = None
    if scheduler_name == 'cosine':
        warmup = max(0, warmup_steps)
        total_steps = max(1, config.N_STEPS)

        # When resuming (last_epoch >= 0), PyTorch expects initial_lr in each
        # optimizer param group. We only checkpoint model weights, so ensure it exists.
        if offset > 0:
            for param_group in optimizer.param_groups:
                param_group.setdefault('initial_lr', param_group['lr'])

        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup

            progress = (step - warmup) / max(1, total_steps - warmup)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=offset - 1)

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
        scheduler,
        config,
        df,
        loss_fn,
        sweeping,
        times_save,
        offset,
        suffix,
        generate=True,
    )
