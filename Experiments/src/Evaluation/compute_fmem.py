"""
Compute fraction collapsed (memorization metric) for diffusion models.
This script analyzes generated samples to compute the fraction of samples that collapse
to training data using gap ratio analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm
import warnings
import glob
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

# Add Utils to path
sys.path.insert(1, '../Utils/')      # In case we run from Experiments/Evaluation
import Diffusion as dm
import cfg
import sprites_dataset

warnings.filterwarnings("ignore")


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


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


def find_image_files(data_root):
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(data_root, pat)))
    return sorted(files)


def compute_channel_stats(dataset, batch_size, num_workers=2):
    loader_local = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    channel_sum = None
    channel_squared_sum = None
    n_pixels = 0

    for images in loader_local:
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


def bootstrap_mean_se(data, threshold, n_bootstrap=1000, random_state=None):
    """
    Compute bootstrap estimate of the mean and its standard error for values below a threshold.

    Parameters:
    - data: 1D array-like of values.
    - threshold: numeric threshold; only values < threshold are considered.
    - n_bootstrap: number of bootstrap samples.
    - random_state: seed for reproducibility.

    Returns:
    - mean_est: bootstrap estimate of the mean.
    - se_est: bootstrap estimate of the standard error of the mean.
    - lower: lower bound of 95% confidence interval.
    - upper: upper bound of 95% confidence interval.
    """
    # Prepare RNG
    rng = np.random.default_rng(random_state)
    
    # Generate bootstrap samples
    means = np.empty(n_bootstrap)
    n_data = len(data)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n_data, replace=True)
        collapsed = np.where(sample < threshold)[0]
        means[i] = len(collapsed) / len(sample)
    
    # Compute estimates
    mean_est = means.mean()
    se_est = means.std(ddof=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return mean_est, se_est, lower, upper


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute fraction collapsed (memorization metric) for diffusion models."
    )
    
    # Model configuration arguments
    parser.add_argument("-n", "--num", help="Number of training data", type=int, required=True)
    parser.add_argument("-i", "--index", help="Index for the dataset (used for CelebA)", type=int, default=0)
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
    parser.add_argument(
        "--isic_data_root",
        type=str,
        default="../../Data/ISIC_2019_Training_Input/ISIC_2019_Training_Input",
        help="Path to extracted ISIC image directory.",
    )
    parser.add_argument("--num_workers", help="DataLoader workers for ISIC stats", type=int, default=2)
    
    # Analysis parameters
    parser.add_argument("-Ns", "--Nsamples", help="Number of sample batches to analyze", type=int, default=100)
    parser.add_argument("--batch_sample_size", help="Size of each sample batch", type=int, default=100)
    parser.add_argument("--gap_threshold", help="Gap ratio threshold for collapsed samples", type=float, default=1/3)
    parser.add_argument("--device", help="Device to use (cuda:0, cpu)", type=str, default='cuda:0')
    
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


def load_isic_training_data(config, args):
    """Load and normalize the ISIC training subset."""
    image_paths = find_image_files(args.isic_data_root)
    if len(image_paths) == 0:
        raise RuntimeError("No ISIC images found in {:s}".format(args.isic_data_root))
    if args.num > len(image_paths):
        raise ValueError("Requested n={:d} but ISIC has {:d} images.".format(args.num, len(image_paths)))

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(image_paths), size=args.num, replace=False)
    indices = np.sort(indices)
    selected_paths = [image_paths[i] for i in indices]

    base_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )
    base_dataset = ImagePathDataset(selected_paths, transform=base_transform)

    example = base_dataset[0]
    config.IMG_SHAPE = tuple(example.shape)

    mean, std = compute_channel_stats(base_dataset, batch_size=config.BATCH_SIZE, num_workers=args.num_workers)
    std = torch.ones_like(std)
    config.mean = mean
    config.std = std

    train_norm = NormalizedDataset(base_dataset, mean, std)
    train_images = torch.stack([train_norm[i] for i in range(len(train_norm))]).float()
    return train_images


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
    elif dataset_key == 'isic':
        config = dm.TrainingConfig()
        config.DATASET = 'ISIC'
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

    if dataset_key == 'isic':
        if args.time == -1:
            folder = 'ISIC{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_seed{:d}/'.format(
                args.img_size,
                config.n_images,
                args.nbase,
                config.OPTIM,
                config.BATCH_SIZE,
                config.LR,
                args.seed,
            )
        else:
            folder = 'ISIC{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_seed{:d}_t{:d}/'.format(
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

def compute_fraction_mem(training_times, train_images, type_model, config, file_fc,
                             nsamples, sample_size, gap_threshold):
    """Compute fraction collapsed for all training times."""
    N = np.prod(config.IMG_SHAPE)
    X = train_images.reshape(-1, N).float().to(config.DEVICE)
    
    pbar = tqdm(training_times)
    for tau in pbar:
        # Load generated images and compute k-nearest neighbors
        k = min(2, len(train_images))
        if k < 2:
            raise ValueError('Need at least 2 training images to compute gap ratio.')

        distances_tensor_all = torch.zeros(nsamples * sample_size, k)
        valid_count = 0
        
        for i in range(nsamples):
            path_save = config.path_save + type_model + 'Samples/' + '/{:d}/'.format(tau)
            path = path_save + 'generated'
            file_a = path + '/samples_a_{:d}'.format(i)
            
            try:
                images_a = torch.load(file_a)
            except FileNotFoundError:
                print(f"Warning: File not found: {file_a}")
                continue
            
            batch_n = images_a.shape[0]
            i1, i2 = valid_count, valid_count + batch_n
            
            # Compute distances to training set
            s = images_a.reshape(-1, 1, N).to(config.DEVICE)
            dist = torch.norm(s - X, dim=2, p=2)
            knn = dist.topk(k, dim=1, largest=False)
            
            distances_tensor_all[i1:i2, :] = knn[0].cpu()
            valid_count += batch_n

        if valid_count == 0:
            print(f"Warning: no generated samples found for checkpoint {tau}.")
            with open(file_fc, "a") as myfile:
                myfile.write(f"\n{tau:d}\t0.000\t0.00000\t0.00000\t0.00000")
            continue

        distances_tensor_all = distances_tensor_all[:valid_count, :]
        
        # Compute gap ratios
        gap_ratio = distances_tensor_all[:, 0] / distances_tensor_all[:, 1]
        
        # Compute fraction collapsed with bootstrap confidence intervals
        collapsed_samples = np.where(gap_ratio < gap_threshold)[0]
        fraction_mem = len(collapsed_samples) / len(gap_ratio)
        
        if len(collapsed_samples) > 0:
            fraction_mem, std_frac, lower, upper = bootstrap_mean_se(
                gap_ratio.numpy(), gap_threshold
            )
        else:
            std_frac = 0.0
            lower = 0.0
            upper = 0.0

        pbar.set_description(f'Fmem = {fraction_mem*100:.2f}% ± {std_frac*100:.2f}')

        # Write results to file
        with open(file_fc, "a") as myfile:
            myfile.write(f"\n{tau:d}\t{fraction_mem*100:.3f}\t{std_frac*100:.5f}\t"
                        f"{lower*100:.5f}\t{upper*100:.5f}")


def main():
    """Main function to compute fraction collapsed."""
    # Parse arguments
    args = parse_arguments()
    print("Arguments:", args)
    
    # Load configuration
    config = prepare_config(args)
    
    # Model type string for paths
    type_model = build_experiment_dir(args, config)
    
    # Create output directory and file
    path_file = config.path_save + type_model + 'Memorization/'
    file_fc = path_file + 'fraction_memorized.txt'
    if os.path.exists(file_fc):     # Remove existing file
        os.remove(file_fc)
    os.makedirs(path_file, exist_ok=True)
    
    # Define training times to analyze (use existing checkpoints)
    path_models = config.path_save + type_model + 'Models/'
    checkpoint_files = glob.glob(os.path.join(path_models, 'Model_*'))
    training_times = sorted({int(os.path.basename(f).split('_')[-1]) for f in checkpoint_files})
    if len(training_times) == 0:
        raise FileNotFoundError('No checkpoints found in {:s}'.format(path_models))
    
    print(f"Computing memorization fraction for {len(training_times)} checkpoints...")
    print(f"Model: {type_model}")
    print(f"Output file: {file_fc}")
    
    # Load training data
    if args.dataset.lower() == 'celeba':
        train_images, _ = cfg.load_training_data(config, args.index)
        train_images = dataset_to_tensor(train_images, config.n_images).to(config.DEVICE)
    elif args.dataset.lower() == 'sprites':
        train_images = load_sprites_training_data(config, args).to(config.DEVICE)
    else:
        train_images = load_isic_training_data(config, args).to(config.DEVICE)
    
    # Setup diffusion configuration
    df = dm.DiffusionConfig(
        n_steps=config.TIMESTEPS,
        img_shape=config.IMG_SHAPE,
        device=config.DEVICE,
    )
    
    # Compute fraction collapsed for each checkpoint
    compute_fraction_mem(
        training_times=training_times,
        train_images=train_images,
        type_model=type_model,
        config=config,
        file_fc=file_fc,
        nsamples=args.Nsamples,
        sample_size=args.batch_sample_size,
        gap_threshold=args.gap_threshold
    )
    
    print("Memorization fraction computation completed!")


if __name__ == "__main__":
    main()