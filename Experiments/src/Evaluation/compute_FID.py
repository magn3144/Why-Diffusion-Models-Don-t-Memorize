"""Compute FID (Fréchet Inception Distance) for diffusion models."""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import warnings

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add Utils to path
sys.path.insert(1, '../Utils/')  # In case we run from Experiments/Evaluation
import Diffusion as dm
import cfg
import sprites_dataset

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute FID (Fréchet Inception Distance) for diffusion models."
    )

    # Model configuration arguments
    parser.add_argument("-n", "--num", help="Number of training data", type=int, required=True)
    parser.add_argument("-i", "--index", help="Index for CelebA subset", type=int, default=0)
    parser.add_argument("-s", "--img_size", help="Size of the images to use", type=int, required=True)
    parser.add_argument("-LR", "--learning_rate", help="Learning rate for optimization", type=float, required=True)
    parser.add_argument("-O", "--optim", help="Optimisation type (SGD_Momentum or Adam)", type=str, required=True)
    parser.add_argument("-W", "--nbase", help="Number of base filters", type=int, required=True)
    parser.add_argument("-B", "--batch_size", help="Batch size used to train the model", type=int, required=True)
    parser.add_argument("-D", "--dataset", help="Dataset used to train the model", type=str, required=True)
    parser.add_argument("--model_type", help="Backbone for Sprites experiments (unet or gmm)", type=str, default='unet')
    parser.add_argument("--seed", help="Sprites subset seed", type=int, default=1)
    parser.add_argument("--tag", help="Optional Sprites experiment tag", type=str, default='')
    parser.add_argument("-t", "--time", help="Diffusion timestep (-1 for normal mode)", type=int, default=-1)
    parser.add_argument("--experiment_dir", help="Optional exact folder name in Saves (overrides name construction)", type=str, default=None)

    # FID reference stats and analysis parameters
    parser.add_argument("-istat", "--id_stat", help="Index of the reference statistics (1 to 5)", type=int, required=True)
    parser.add_argument("--N1", help="Starting batch index", type=int, default=0)
    parser.add_argument("--N2", help="Ending batch index", type=int, default=100)
    parser.add_argument("--batch_size_samples", help="Size of each sample batch", type=int, default=100)
    parser.add_argument("--device", help="Device to use (cuda:0, cpu)", type=str, default='cuda:0')
    parser.add_argument("--rebuild_stats", help="Recompute FID stats even if they already exist", action='store_true')
    parser.add_argument("--no_auto_stats", help="Do not auto-create missing FID stats", action='store_true')

    return parser.parse_args()


def dataset_to_tensor(data, n_images):
    """Convert dataset/tensor inputs into a dense tensor of images."""
    if isinstance(data, torch.Tensor):
        return data[:n_images]

    loader = DataLoader(data, batch_size=min(len(data), n_images), shuffle=False)
    batch = next(iter(loader))
    return batch[:n_images]


def load_sprites_training_data(config, args):
    """Load and normalize the Sprites training subset exactly as in training."""
    base_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((args.img_size, args.img_size)),
        ]
    )
    data_file = os.path.join(config.path_data, 'sprites_1788_16x16.npy')
    dataset = sprites_dataset.SpritesDataset(
        transform=base_transform,
        img_file=data_file,
        num_samples=args.num,
        seed=args.seed,
    )

    train_images = torch.stack([dataset[i] for i in range(len(dataset))]).float()
    mean = train_images.mean(dim=(0, 2, 3))
    std = torch.ones_like(mean)
    config.mean = mean
    config.std = std
    return (train_images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)


def prepare_config(args):
    """Prepare config object for CelebA or Sprites."""
    dataset_key = args.dataset.lower()
    if dataset_key == 'celeba':
        config = cfg.load_config('CelebA')
        config.IMG_SHAPE = (1, args.img_size, args.img_size)
    elif dataset_key == 'sprites':
        config = dm.TrainingConfig()
        config.DATASET = 'Sprites'
        config.path_save = '../../Saves/'
        config.path_data = '../../Data/'
        config.IMG_SHAPE = (3, args.img_size, args.img_size)
        config.CENTER = True
        config.STANDARDIZE = False
        config.TIMESTEPS = 1000
    else:
        raise ValueError('Unsupported dataset: {:s}'.format(args.dataset))

    config.n_images = args.num
    config.BATCH_SIZE = min(args.batch_size, config.n_images)
    config.OPTIM = args.optim
    config.LR = args.learning_rate
    config.DEVICE = args.device
    return config


def build_experiment_dir(args, config):
    """Build experiment folder name in Saves/."""
    if args.experiment_dir is not None and args.experiment_dir.strip() != '':
        folder = args.experiment_dir.strip().strip('/')
        return folder + '/'

    dataset_key = args.dataset.lower()
    if dataset_key == 'celeba':
        return '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}/'.format(
            config.DATASET,
            args.img_size,
            config.n_images,
            args.nbase,
            config.OPTIM,
            config.BATCH_SIZE,
            config.LR,
            args.index,
        )

    model_type = args.model_type.lower()
    if args.time == -1:
        folder = '{:s}_{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_seed{:d}/'.format(
            model_type,
            config.DATASET,
            args.img_size,
            config.n_images,
            args.nbase,
            config.OPTIM,
            config.BATCH_SIZE,
            config.LR,
            args.seed,
        )
    else:
        folder = '{:s}_{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_seed{:d}_t{:d}/'.format(
            model_type,
            config.DATASET,
            args.img_size,
            config.n_images,
            args.nbase,
            config.OPTIM,
            config.BATCH_SIZE,
            config.LR,
            args.seed,
            args.time,
        )

    tag = args.tag.strip()
    if tag:
        folder = folder[:-1] + '_{:s}/'.format(tag)
    return folder


def get_mean_std_tensors(config, images):
    """Get mean/std as tensors with shape [1, C, 1, 1]."""
    mean = torch.as_tensor(config.mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    std = torch.as_tensor(config.std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    return mean, std


def detransform_images(images, config):
    """Detransform images from normalized to original scale."""
    mean, std = get_mean_std_tensors(config, images)
    return images * std + mean


def save_tensor_images(images, out_dir, start_index=0):
    """Save a tensor batch [B, C, H, W] as PNG files."""
    images = images.detach().cpu().clamp(0.0, 1.0)
    for idx, x in enumerate(images):
        torchvision.utils.save_image(x, os.path.join(out_dir, '{:d}.png'.format(start_index + idx)))


def run_pytorch_fid(input_a, input_b, device):
    """Run pytorch_fid and return stdout."""
    cmd = [
        sys.executable,
        '-m',
        'pytorch_fid',
        input_a,
        input_b,
        '--device',
        device,
    ]
    out = subprocess.check_output(cmd, text=True)
    return out


def run_pytorch_fid_save_stats(image_dir, stats_path, device):
    """Build reference statistics file using pytorch_fid."""
    cmd = [
        sys.executable,
        '-m',
        'pytorch_fid',
        '--save-stats',
        image_dir,
        stats_path,
        '--device',
        device,
    ]
    subprocess.check_call(cmd)


def ensure_reference_stats(path_stats_testset, config, train_images, args):
    """Ensure stats file exists; create it from training data if requested."""
    os.makedirs(os.path.dirname(path_stats_testset), exist_ok=True)

    if args.rebuild_stats and os.path.exists(path_stats_testset):
        os.remove(path_stats_testset)

    if os.path.exists(path_stats_testset):
        return

    if args.no_auto_stats:
        raise FileNotFoundError('Missing FID reference stats: {:s}'.format(path_stats_testset))

    tmp_dir = os.path.join(config.path_save, 'FID_ref', 'tmp_stats_{:d}'.format(args.id_stat))
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        train_orig = detransform_images(train_images, config)
        save_tensor_images(train_orig, tmp_dir, start_index=0)
        run_pytorch_fid_save_stats(tmp_dir, path_stats_testset, args.device)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


def parse_fid_output(output):
    """Parse numeric FID score from pytorch_fid output."""
    for line in output.strip().splitlines()[::-1]:
        if 'FID:' in line:
            try:
                return float(line.split('FID:')[-1].strip())
            except ValueError:
                continue
    raise ValueError('Could not parse FID output: {:s}'.format(output))


def compute_fid_for_checkpoint(tau, type_model, config, path_stats_testset, n1, n2, batch_size_samples, file_fid, device):
    """Compute FID for a specific training checkpoint."""
    file_img_gen = os.path.join(config.path_save, type_model, 'FID', '{:d}'.format(tau))
    os.makedirs(file_img_gen, exist_ok=True)

    fid = -1.0
    try:
        valid_batches = 0
        for i in range(n1, n2):
            file_a = os.path.join(config.path_save, type_model, 'Samples', '{:d}'.format(tau), 'generated', 'samples_a_{:d}'.format(i))
            if not os.path.exists(file_a):
                continue

            images_a = torch.load(file_a)
            t = detransform_images(images_a, config)
            save_tensor_images(t, file_img_gen, start_index=i * batch_size_samples)
            valid_batches += 1

        if valid_batches == 0:
            raise FileNotFoundError('No generated batches found for checkpoint {:d}'.format(tau))

        p = run_pytorch_fid(path_stats_testset, file_img_gen, device)
        fid = parse_fid_output(p)

        with open(file_fid, 'a') as myfile:
            myfile.write('\n{:d}\t{:.3f}'.format(tau, fid))

    except Exception as e:
        print('Error computing FID for checkpoint {:d}: {:s}'.format(tau, str(e)))
        with open(file_fid, 'a') as myfile:
            myfile.write('\n{:d}\t{:.3f}'.format(tau, -1.0))
        print('Skipping...')

    finally:
        if os.path.exists(file_img_gen):
            shutil.rmtree(file_img_gen)

    return fid


def compute_fid_all_checkpoints(training_times, type_model, config, args, train_images):
    """Compute FID for all training checkpoints."""
    path_stats_testset = os.path.join(config.path_save, 'FID_ref', 'stats{:d}.npz'.format(args.id_stat))
    path_file = os.path.join(config.path_save, type_model, 'FID')
    file_fid = os.path.join(path_file, 'FID_{:d}.txt'.format(args.id_stat))
    if os.path.exists(file_fid):
        os.remove(file_fid)
    os.makedirs(path_file, exist_ok=True)

    ensure_reference_stats(path_stats_testset, config, train_images, args)

    print('Computing FID for {:d} checkpoints...'.format(len(training_times)))
    print('Model: {:s}'.format(type_model))
    print('Reference statistics: {:s}'.format(path_stats_testset))
    print('Output file: {:s}'.format(file_fid))

    pbar = tqdm(training_times)
    for tau in pbar:
        fid = compute_fid_for_checkpoint(
            tau=tau,
            type_model=type_model,
            config=config,
            path_stats_testset=path_stats_testset,
            n1=args.N1,
            n2=args.N2,
            batch_size_samples=args.batch_size_samples,
            file_fid=file_fid,
            device=args.device,
        )
        pbar.set_description('FID = {:.3f}'.format(fid))


def main():
    """Main function to compute FID scores."""
    args = parse_arguments()
    print('Arguments:', args)

    config = prepare_config(args)
    type_model = build_experiment_dir(args, config)

    path_models = os.path.join(config.path_save, type_model, 'Models')
    checkpoint_files = glob.glob(os.path.join(path_models, 'Model_*'))
    training_times = sorted({int(os.path.basename(f).split('_')[-1]) for f in checkpoint_files})
    if len(training_times) == 0:
        raise FileNotFoundError('No checkpoints found in {:s}'.format(path_models))

    if args.dataset.lower() == 'celeba':
        train_images, _ = cfg.load_training_data(config, args.index)
        train_images = dataset_to_tensor(train_images, config.n_images).to(config.DEVICE)
    else:
        train_images = load_sprites_training_data(config, args).to(config.DEVICE)

    compute_fid_all_checkpoints(
        training_times=training_times,
        type_model=type_model,
        config=config,
        args=args,
        train_images=train_images,
    )

    print('FID computation completed!')


if __name__ == '__main__':
    main()