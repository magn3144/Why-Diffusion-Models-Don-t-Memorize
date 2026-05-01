import argparse
import glob
import os
import re
import subprocess
import sys

import torch
from torch import nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'Utils'))
sys.path.insert(1, UTILS_DIR)

import Diffusion
import Unet
import cfg
import loader
import TinyModels as TM


CELEBA_RE = re.compile(
    r'^CelebA(?P<img_size>\d+)_(?P<num>\d+)_(?P<nbase>\d+)_(?P<optim>.+?)_(?P<batch_size>\d+)_(?P<lr>\d+\.\d+)_index(?P<index>\d+)(?:_t(?P<time>-?\d+))?$'
)

ISIC_RE = re.compile(
    r'^ISIC(?P<img_size>\d+)_(?P<num>\d+)_(?P<nbase>\d+)_(?P<optim>.+?)_(?P<batch_size>\d+)_(?P<lr>\d+\.\d+)_seed(?P<seed>\d+)(?:_t(?P<time>-?\d+))?$'
)

SPRITES_RE = re.compile(
    r'^(?P<model_type>unet|gmm)_Sprites(?P<img_size>\d+)_(?P<num>\d+)_(?P<nbase>\d+)_(?P<optim>.+?)_(?P<batch_size>\d+)_(?P<lr>\d+\.\d+)_seed(?P<seed>\d+)(?:_t(?P<time>-?\d+))?(?:_(?P<tag>.+))?$'
)

GMM8_RE = re.compile(
    r'^GMM8_(?P<num>\d+)_(?P<nbase>\d+)_(?P<optim>.+?)_(?P<batch_size>\d+)_(?P<lr>\d+\.\d+)_B(?P<large_batch>\d+)(?:_t(?P<time>-?\d+))?$'
)


def parse_experiment_name(name):
    """Parse experiment folder name into argument fields."""
    m = CELEBA_RE.match(name)
    if m:
        d = m.groupdict()
        return {
            'dataset': 'CelebA',
            'num': int(d['num']),
            'index': int(d['index']),
            'img_size': int(d['img_size']),
            'learning_rate': float(d['lr']),
            'optim': d['optim'],
            'nbase': int(d['nbase']),
            'batch_size': int(d['batch_size']),
            'time': int(d['time']) if d['time'] is not None else -1,
            'model_type': None,
            'seed': None,
            'tag': '',
            'large_batch': None,
        }

    m = ISIC_RE.match(name)
    if m:
        d = m.groupdict()
        return {
            'dataset': 'ISIC',
            'num': int(d['num']),
            'index': 0,
            'img_size': int(d['img_size']),
            'learning_rate': float(d['lr']),
            'optim': d['optim'],
            'nbase': int(d['nbase']),
            'batch_size': int(d['batch_size']),
            'time': int(d['time']) if d['time'] is not None else -1,
            'model_type': 'unet',
            'seed': int(d['seed']),
            'tag': '',
            'large_batch': None,
        }

    m = SPRITES_RE.match(name)
    if m:
        d = m.groupdict()
        return {
            'dataset': 'Sprites',
            'num': int(d['num']),
            'index': 0,
            'img_size': int(d['img_size']),
            'learning_rate': float(d['lr']),
            'optim': d['optim'],
            'nbase': int(d['nbase']),
            'batch_size': int(d['batch_size']),
            'time': int(d['time']) if d['time'] is not None else -1,
            'model_type': d['model_type'],
            'seed': int(d['seed']),
            'tag': d['tag'] or '',
            'large_batch': None,
        }

    m = GMM8_RE.match(name)
    if m:
        d = m.groupdict()
        return {
            'dataset': 'GMM8',
            'num': int(d['num']),
            'index': 0,
            'img_size': 16,
            'learning_rate': float(d['lr']),
            'optim': d['optim'],
            'nbase': int(d['nbase']),
            'batch_size': int(d['batch_size']),
            'time': int(d['time']) if d['time'] is not None else -1,
            'model_type': 'gmm8',
            'seed': 1,
            'tag': '',
            'large_batch': int(d['large_batch']),
        }

    return None


def infer_sample_batches(exp_path):
    """Infer number of generated sample batches from Samples/*/generated/samples_a_* files."""
    sample_dirs = sorted(glob.glob(os.path.join(exp_path, 'Samples', '*', 'generated')))
    for generated_dir in sample_dirs:
        files = glob.glob(os.path.join(generated_dir, 'samples_a_*'))
        if not files:
            continue

        indices = []
        for fp in files:
            suffix = os.path.basename(fp).replace('samples_a_', '')
            try:
                indices.append(int(suffix))
            except ValueError:
                continue

        if indices:
            return max(indices) + 1

    return 0


parser = argparse.ArgumentParser('Generation of samples from trained diffusion models.')

parser.add_argument('-n', '--num', help='Number of training data', type=int, default=None)
parser.add_argument('-i', '--index', help='Index for the dataset (0 or 1)', type=int, default=0)
parser.add_argument('-s', '--img_size', help='Size of the images used to train', type=int, default=None)
parser.add_argument('-LR', '--learning_rate', help='Learning rate for optimization', type=float, default=None)
parser.add_argument('-O', '--optim', help='Optimisation type (SGD_Momentum or Adam)', type=str, default=None)
parser.add_argument('-W', '--nbase', help='Number of base filters', type=str, default=None)
parser.add_argument('-t', '--time', help='Diffusion timestep', type=int, default=-1)
parser.add_argument('-B', '--batch_size', type=int, help='Batch size used to train the model', default=None)
parser.add_argument('-D', '--dataset', type=str, help='Dataset used to train the model (CelebA, ISIC, Sprites, or GMM8).', default=None)
parser.add_argument('-Ns', '--Nsamples', type=int, help='Number of samples to generate (should be multiple of 100).', default=100)
parser.add_argument('--model_type', type=str, help='Model backbone for Sprites (unet or gmm).', default='unet')
parser.add_argument('--seed', type=int, help='Seed used to sample the sprites subset or ISIC.', default=1)
parser.add_argument('--tag', type=str, help='Optional tag appended to the sprites experiment folder.', default='')
parser.add_argument('--large_batch', type=int, help='Large batch size parameter for GMM8 experiments.', default=512)
parser.add_argument('--generate_all_missing', action='store_true', help='Generate samples for all experiments missing samples.')
parser.add_argument('--saves_dir', type=str, default='../../Saves', help='Path to Saves directory (used with --generate_all_missing).')
parser.add_argument('--timesteps', type=int, help='Number of diffusion timesteps used during training (optional).', default=None)
parser.add_argument('--device', type=str, help='Device used to load and apply the model.', default='cuda:0')

args = parser.parse_args()


def build_single_command(meta, device, nsamples):
    cmd = [
        sys.executable,
        __file__,
        '-n', str(meta['num']),
        '-s', str(meta['img_size']),
        '-LR', str(meta['learning_rate']),
        '-O', meta['optim'],
        '-W', str(meta['nbase']),
        '-B', str(meta['batch_size']),
        '-D', meta['dataset'],
        '-Ns', str(nsamples),
        '-t', str(meta['time']),
        '--device', device,
    ]

    if meta['dataset'] == 'CelebA':
        cmd += ['-i', str(meta['index'])]
    elif meta['dataset'] == 'ISIC':
        cmd += ['--seed', str(meta['seed'])]
    elif meta['dataset'] == 'Sprites':
        cmd += ['--model_type', meta['model_type'], '--seed', str(meta['seed'])]
        if meta['tag']:
            cmd += ['--tag', meta['tag']]
    elif meta['dataset'] == 'GMM8':
        cmd += ['--model_type', 'gmm8', '--large_batch', str(meta['large_batch'])]

    return cmd


if args.generate_all_missing:
    nsamples = args.Nsamples if args.Nsamples is not None else 500
    exp_dirs = sorted(
        d for d in os.listdir(args.saves_dir)
        if os.path.isdir(os.path.join(args.saves_dir, d))
    )

    if not exp_dirs:
        raise FileNotFoundError('No experiment folders found in {:s}'.format(args.saves_dir))

    generated_count = 0
    skipped_count = 0

    for exp_name in exp_dirs:
        exp_path = os.path.join(args.saves_dir, exp_name)
        models_dir = os.path.join(exp_path, 'Models')
        if not os.path.isdir(models_dir):
            continue

        n_batches = infer_sample_batches(exp_path)
        if n_batches > 0:
            print('Skipping (already has samples):', exp_name)
            skipped_count += 1
            continue

        meta = parse_experiment_name(exp_name)
        if meta is None:
            print('Skipping (unsupported folder name):', exp_name)
            skipped_count += 1
            continue

        print('\n=== Generating samples for {:s} ==='.format(exp_name))
        cmd = build_single_command(meta, args.device, nsamples)

        try:
            print('Running:', ' '.join(cmd))
            subprocess.check_call(cmd)
            generated_count += 1
        except subprocess.CalledProcessError as exc:
            print('Failed to generate samples for {:s}: {:s}'.format(exp_name, str(exc)))
            skipped_count += 1

    print('\nDone. Generated samples for {:d} experiment(s), skipped {:d}.'.format(generated_count, skipped_count))
    sys.exit(0)


print(args)

required = [args.num, args.img_size, args.learning_rate, args.optim, args.nbase, args.batch_size, args.dataset, args.Nsamples]
if any(value is None for value in required):
    raise ValueError('When not using --generate_all_missing, the following arguments are required: -n, -s, -LR, -O, -W, -B, -D, -Ns')

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
    mode = 'normal' if time_step == -1 else 'fixed_time'
    config.mode = mode
    config.time_step = time_step
    if args.timesteps is not None:
        config.TIMESTEPS = int(args.timesteps)

    if mode == 'normal':
        type_model = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}'.format(
            config.DATASET, size, config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, index
        )
    else:
        type_model = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}_t{:d}'.format(
            config.DATASET, size, config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, index, time_step
        )

elif dataset_key == 'isic':
    config = cfg.load_config('CelebA')
    config.IMG_SHAPE = (3, size, size)
    config.n_images = int(args.num)
    config.BATCH_SIZE = int(args.batch_size)
    config.OPTIM = args.optim
    config.LR = float(args.learning_rate)
    config.DEVICE = args.device
    time_step = int(args.time)
    mode = 'normal' if time_step == -1 else 'fixed_time'
    config.mode = mode
    config.time_step = time_step
    config.DATASET = 'ISIC'
    if args.timesteps is not None:
        config.TIMESTEPS = int(args.timesteps)

    seed = int(args.seed)
    if mode == 'normal':
        type_model = 'ISIC{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_seed{:d}'.format(
            size, config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, seed
        )
    else:
        type_model = 'ISIC{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_seed{:d}_t{:d}'.format(
            size, config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, seed, time_step
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
    mode = 'normal' if time_step == -1 else 'fixed_time'
    config.mode = mode
    config.time_step = time_step
    if args.timesteps is not None:
        config.TIMESTEPS = int(args.timesteps)

    model_type = args.model_type.lower()
    seed = int(args.seed)
    tag = args.tag.strip()

    if mode == 'normal':
        type_model = '{:s}_{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_seed{:d}'.format(
            model_type, config.DATASET, size, config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, seed
        )
    else:
        type_model = '{:s}_{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_seed{:d}_t{:d}'.format(
            model_type, config.DATASET, size, config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, seed, time_step
        )

    if tag:
        type_model = type_model + '_{:s}'.format(tag)

elif dataset_key == 'gmm8':
    config = Diffusion.TrainingConfig()
    config.DATASET = 'GMM8'
    config.path_save = '../../Saves/'
    config.path_data = '../../Data/'
    config.IMG_SHAPE = (3, size, size)
    config.n_images = int(args.num)
    config.BATCH_SIZE = int(args.batch_size)
    config.OPTIM = args.optim
    config.LR = float(args.learning_rate)
    config.DEVICE = args.device
    time_step = int(args.time)
    mode = 'normal' if time_step == -1 else 'fixed_time'
    config.mode = mode
    config.time_step = time_step
    if args.timesteps is not None:
        config.TIMESTEPS = int(args.timesteps)

    if mode == 'normal':
        type_model = 'GMM8_{:d}_{:d}_{:s}_{:d}_{:.4f}_B{:d}'.format(
            config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, args.large_batch
        )
    else:
        type_model = 'GMM8_{:d}_{:d}_{:s}_{:d}_{:.4f}_B{:d}_t{:d}'.format(
            config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, args.large_batch, time_step
        )

else:
    raise ValueError('Unknown dataset {:s}'.format(DATASET))

if not Nsamples % 100 == 0:
    raise TypeError('Nsamples should be a multiple of 100.')


df = Diffusion.DiffusionConfig(
    n_steps=config.TIMESTEPS,
    img_shape=config.IMG_SHAPE,
    device=config.DEVICE,
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
elif dataset_key == 'gmm8':
    model_diffusion = FlatImageTimeModel(
        channels=config.IMG_SHAPE[0],
        height=config.IMG_SHAPE[1],
        width=config.IMG_SHAPE[2],
        d_model=n_base,
    )
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

batch_gen = 100
Ns = Nsamples // batch_gen

path_models = os.path.join(config.path_save, type_model, 'Models')
checkpoint_files = glob.glob(os.path.join(path_models, 'Model_*'))
training_times = sorted({int(os.path.basename(f).split('_')[-1]) for f in checkpoint_files})
if len(training_times) == 0:
    raise FileNotFoundError('No checkpoints found in {:s}'.format(path_models))

for j, checkpoint_id in enumerate(training_times):
    print('Training time = {:d} ({:d}/{:d})'.format(checkpoint_id, j, len(training_times)))

    model_suffix = os.path.join(path_models, 'Model_{:d}'.format(checkpoint_id))
    try:
        model_diffusion = loader.load_model(model_diffusion, model_suffix)
    except Exception as exc:
        raise RuntimeError('Failed to load checkpoint {:s}: {:s}'.format(model_suffix, str(exc)))

    for i in range(Ns):
        path_save = os.path.join(config.path_save, type_model, 'Samples', '{:d}'.format(checkpoint_id))
        os.makedirs(path_save, exist_ok=True)

        print('Sample {:d}/{:d}'.format(i, Ns))
        samples_gen, samples_init = Diffusion.sample_diffusion_from_noise_DDIM(
            model_diffusion,
            n_images=batch_gen,
            config=config,
            df=df,
            dim=4,
            eta=0.0,
            ddim_steps=100,
        )

        path = os.path.join(path_save, str(config.TIMESTEPS))
        os.makedirs(path, exist_ok=True)
        torch.save(samples_init, os.path.join(path, 'samples_a_{:d}'.format(i)))

        generated_path = os.path.join(path_save, 'generated')
        os.makedirs(generated_path, exist_ok=True)
        torch.save(samples_gen, os.path.join(generated_path, 'samples_a_{:d}'.format(i)))

print('Done!')
