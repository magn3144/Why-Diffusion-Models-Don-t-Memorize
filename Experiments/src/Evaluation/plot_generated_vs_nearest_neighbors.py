"""Create paper-style generated-vs-nearest-neighbor plots for one experiment.

The script takes one experiment folder and two training checkpoints (tau).
For each tau it shows a column pair:
  - left: generated sample
  - right: nearest training neighbor

The script first tries to load pre-generated samples from
  Saves/<experiment>/Samples/<tau>/generated/samples_a_*
If none are found, it can generate samples on the fly from Models/Model_<tau>.

Outputs are saved under Experiments/Results by default.
"""

import argparse
import glob
import os
import random
import re
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# Add Utils to path (script is in Experiments/src/Evaluation)
sys.path.insert(1, "../Utils/")

import cfg
import Diffusion as dm
import loader
import sprites_dataset
import TinyModels as TM
import Unet


CELEBA_RE = re.compile(
    r"^CelebA(?P<img_size>\d+)_(?P<num>\d+)_(?P<nbase>\d+)_(?P<optim>.+?)_(?P<batch_size>\d+)_(?P<lr>\d+\.\d+)_index(?P<index>\d+)(?:_t(?P<time>-?\d+))?(?:_(?P<tag>.+))?$"
)

SPRITES_RE = re.compile(
    r"^(?P<model_type>unet|gmm)_Sprites(?P<img_size>\d+)_(?P<num>\d+)_(?P<nbase>\d+)_(?P<optim>.+?)_(?P<batch_size>\d+)_(?P<lr>\d+\.\d+)_seed(?P<seed>\d+)(?:_t(?P<time>-?\d+))?(?:_(?P<tag>.+))?$"
)

ISIC_RE = re.compile(
    r"^ISIC(?P<img_size>\d+)_(?P<num>\d+)_(?P<nbase>\d+)_(?P<optim>.+?)_(?P<batch_size>\d+)_(?P<lr>\d+\.\d+)_seed(?P<seed>\d+)(?:_t(?P<time>-?\d+))?(?:_(?P<tag>.+))?$"
)


@dataclass
class ExperimentMeta:
    dataset: str
    experiment_name: str
    experiment_path: str
    img_size: int
    n: int
    nbase: int
    optim: str
    batch_size: int
    lr: float
    index: int = 0
    seed: int = 1
    model_type: str = "unet"


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create generated-vs-nearest-neighbor plots for one experiment folder."
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment folder name inside Saves/ (or absolute path).",
    )
    parser.add_argument("--taus", type=int, nargs=2, required=True, metavar=("TAU1", "TAU2"))

    parser.add_argument("--rows", type=int, default=5, help="Number of sample pairs shown per tau.")
    parser.add_argument(
        "--selection",
        type=str,
        default="closest",
        choices=["closest", "random"],
        help="How to pick displayed generated samples from the available sample pool.",
    )

    parser.add_argument(
        "--sample_pool_size",
        type=int,
        default=200,
        help="Number of generated samples to gather per tau (load and/or generate).",
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=100,
        help="Batch size used when generating samples on the fly.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="DDIM steps for on-the-fly generation.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Diffusion timesteps used during training.",
    )
    parser.add_argument(
        "--no_generate_fallback",
        action="store_true",
        help="If set, fail when generated samples are missing instead of generating from checkpoints.",
    )
    parser.add_argument(
        "--tau_policy",
        type=str,
        default="exact",
        choices=["exact", "nearest"],
        help=(
            "How to handle requested taus that are not saved checkpoints: "
            "exact=error, nearest=use closest available checkpoint."
        ),
    )

    parser.add_argument("--fid_id", type=int, default=1, help="Which FID file to read: FID/FID_<fid_id>.txt.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--isic_data_root",
        type=str,
        default="../../Data/ISIC_2019_Training_Input/ISIC_2019_Training_Input",
        help="Path to extracted ISIC image directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../Results",
        help="Directory where figures are saved.",
    )
    return parser.parse_args()


def resolve_experiment_path(exp_arg):
    if os.path.isabs(exp_arg):
        path = exp_arg
    else:
        path = os.path.join("../../Saves", exp_arg)
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError("Experiment folder not found: {:s}".format(path))
    return path


def parse_experiment_meta(exp_arg):
    exp_path = resolve_experiment_path(exp_arg)
    name = os.path.basename(os.path.normpath(exp_path))

    m = CELEBA_RE.match(name)
    if m:
        d = m.groupdict()
        return ExperimentMeta(
            dataset="CelebA",
            experiment_name=name,
            experiment_path=exp_path,
            img_size=int(d["img_size"]),
            n=int(d["num"]),
            nbase=int(d["nbase"]),
            optim=d["optim"],
            batch_size=int(d["batch_size"]),
            lr=float(d["lr"]),
            index=int(d["index"]),
            model_type="unet",
        )

    m = SPRITES_RE.match(name)
    if m:
        d = m.groupdict()
        return ExperimentMeta(
            dataset="Sprites",
            experiment_name=name,
            experiment_path=exp_path,
            img_size=int(d["img_size"]),
            n=int(d["num"]),
            nbase=int(d["nbase"]),
            optim=d["optim"],
            batch_size=int(d["batch_size"]),
            lr=float(d["lr"]),
            seed=int(d["seed"]),
            model_type=d["model_type"],
        )

    m = ISIC_RE.match(name)
    if m:
        d = m.groupdict()
        return ExperimentMeta(
            dataset="ISIC",
            experiment_name=name,
            experiment_path=exp_path,
            img_size=int(d["img_size"]),
            n=int(d["num"]),
            nbase=int(d["nbase"]),
            optim=d["optim"],
            batch_size=int(d["batch_size"]),
            lr=float(d["lr"]),
            seed=int(d["seed"]),
            model_type="unet",
        )

    raise ValueError(
        "Could not parse experiment folder name: {:s}. "
        "Expected CelebA*, (unet|gmm)_Sprites*, or ISIC*.".format(name)
    )


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


def dataset_to_tensor(data, n_images):
    if isinstance(data, torch.Tensor):
        return data[:n_images]

    load = DataLoader(data, batch_size=min(len(data), n_images), shuffle=False)
    batch = next(iter(load))
    return batch[:n_images]


def load_training_data(meta, args):
    if meta.dataset == "CelebA":
        config = cfg.load_config("CelebA")
        config.IMG_SHAPE = (1, meta.img_size, meta.img_size)
        config.n_images = meta.n
        config.BATCH_SIZE = min(meta.batch_size, meta.n)
        config.OPTIM = meta.optim
        config.LR = meta.lr
        config.DEVICE = args.device
        config.TIMESTEPS = args.timesteps

        train_images, _ = cfg.load_training_data(config, meta.index)
        train_images = dataset_to_tensor(train_images, meta.n).float()
        return train_images, config

    if meta.dataset == "Sprites":
        config = dm.TrainingConfig()
        config.DATASET = "Sprites"
        config.path_save = "../../Saves/"
        config.path_data = "../../Data/"
        config.IMG_SHAPE = (3, meta.img_size, meta.img_size)
        config.n_images = meta.n
        config.BATCH_SIZE = min(meta.batch_size, meta.n)
        config.OPTIM = meta.optim
        config.LR = meta.lr
        config.DEVICE = args.device
        config.TIMESTEPS = args.timesteps
        config.CENTER = True
        config.STANDARDIZE = False

        base_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((meta.img_size, meta.img_size)),
            ]
        )
        data_file = os.path.join(config.path_data, "sprites_1788_16x16.npy")
        full_dataset = sprites_dataset.SpritesDataset(
            transform=base_transform,
            img_file=data_file,
            num_samples=None,
            seed=meta.seed,
        )

        n_full = len(full_dataset)
        n_train_full = int(0.8 * n_full)
        if meta.n > n_train_full:
            raise ValueError(
                "Requested n={:d} but 80% sprites train split has {:d}.".format(meta.n, n_train_full)
            )

        rng = np.random.RandomState(meta.seed)
        perm = rng.permutation(n_full)
        train_indices = perm[:n_train_full]
        train_subset_indices = train_indices[: meta.n]
        base_train_dataset = Subset(full_dataset, train_subset_indices.tolist())

        mean, std = compute_channel_stats(base_train_dataset, batch_size=config.BATCH_SIZE)
        std = torch.ones_like(std)
        config.mean = mean
        config.std = std

        train_norm = NormalizedDataset(base_train_dataset, mean, std)
        train_images = torch.stack([train_norm[i] for i in range(len(train_norm))]).float()
        return train_images, config

    if meta.dataset == "ISIC":
        config = dm.TrainingConfig()
        config.DATASET = "ISIC"
        config.path_save = "../../Saves/"
        config.path_data = "../../Data/"
        config.n_images = meta.n
        config.BATCH_SIZE = min(meta.batch_size, meta.n)
        config.OPTIM = meta.optim
        config.LR = meta.lr
        config.DEVICE = args.device
        config.TIMESTEPS = args.timesteps
        config.CENTER = True
        config.STANDARDIZE = False

        image_paths = find_image_files(args.isic_data_root)
        if len(image_paths) == 0:
            raise RuntimeError("No ISIC images found in {:s}".format(args.isic_data_root))
        if meta.n > len(image_paths):
            raise ValueError("Requested n={:d} but ISIC has {:d} images.".format(meta.n, len(image_paths)))

        rng = np.random.default_rng(meta.seed)
        indices = rng.choice(len(image_paths), size=meta.n, replace=False)
        indices = np.sort(indices)
        selected_paths = [image_paths[i] for i in indices]

        base_transform = transforms.Compose(
            [
                transforms.Resize((meta.img_size, meta.img_size)),
                transforms.ToTensor(),
            ]
        )
        base_dataset = ImagePathDataset(selected_paths, transform=base_transform)

        example = base_dataset[0]
        config.IMG_SHAPE = tuple(example.shape)

        mean, std = compute_channel_stats(base_dataset, batch_size=config.BATCH_SIZE)
        std = torch.ones_like(std)
        config.mean = mean
        config.std = std

        train_norm = NormalizedDataset(base_dataset, mean, std)
        train_images = torch.stack([train_norm[i] for i in range(len(train_norm))]).float()
        return train_images, config

    raise ValueError("Unsupported dataset: {:s}".format(meta.dataset))


def build_model(meta, config):
    if meta.dataset == "Sprites" and meta.model_type.lower() == "gmm":
        return FlatImageTimeModel(
            channels=config.IMG_SHAPE[0],
            height=config.IMG_SHAPE[1],
            width=config.IMG_SHAPE[2],
            d_model=meta.nbase,
        )

    return Unet.UNet(
        input_channels=config.IMG_SHAPE[0],
        output_channels=config.IMG_SHAPE[0],
        base_channels=meta.nbase,
        base_channels_multiples=(1, 2, 4),
        apply_attention=(False, True, True),
        dropout_rate=0.1,
    )


def load_generated_pool(exp_path, tau, pool_size):
    generated_dir = os.path.join(exp_path, "Samples", str(tau), "generated")
    if not os.path.isdir(generated_dir):
        return None

    files = sorted(glob.glob(os.path.join(generated_dir, "samples_a_*")))
    if not files:
        return None

    chunks = []
    loaded = 0
    for fp in files:
        x = torch.load(fp, map_location="cpu").float()
        chunks.append(x)
        loaded += x.shape[0]
        if loaded >= pool_size:
            break

    if not chunks:
        return None

    out = torch.cat(chunks, dim=0)
    return out[:pool_size]


def generate_pool_from_checkpoint(meta, config, exp_path, tau, pool_size, batch_size, ddim_steps):
    model = build_model(meta, config)
    model = model.to(config.DEVICE)

    ckpt = os.path.join(exp_path, "Models", "Model_{:d}".format(tau))
    if not os.path.exists(ckpt):
        raise FileNotFoundError("Checkpoint not found: {:s}".format(ckpt))

    model = loader.load_model(model, ckpt, verbose=False)
    model.eval()

    df = dm.DiffusionConfig(
        n_steps=config.TIMESTEPS,
        img_shape=config.IMG_SHAPE,
        device=config.DEVICE,
    )

    all_batches = []
    generated = 0
    while generated < pool_size:
        cur_bs = min(batch_size, pool_size - generated)
        samples_gen, _ = dm.sample_diffusion_from_noise_DDIM(
            model,
            n_images=cur_bs,
            config=config,
            df=df,
            dim=4,
            eta=0.0,
            ddim_steps=ddim_steps,
        )
        all_batches.append(samples_gen.detach().cpu())
        generated += cur_bs

    return torch.cat(all_batches, dim=0)[:pool_size]


def list_checkpoint_steps(exp_path):
    model_dir = os.path.join(exp_path, "Models")
    files = glob.glob(os.path.join(model_dir, "Model_*"))
    steps = []
    for fp in files:
        stem = os.path.basename(fp).replace("Model_", "")
        try:
            steps.append(int(stem))
        except ValueError:
            continue
    return sorted(set(steps))


def resolve_tau(meta, requested_tau, tau_policy):
    available = list_checkpoint_steps(meta.experiment_path)
    if not available:
        raise FileNotFoundError(
            "No checkpoints found in {:s}".format(os.path.join(meta.experiment_path, "Models"))
        )

    if requested_tau in available:
        return requested_tau, requested_tau

    if tau_policy == "nearest":
        used_tau = min(available, key=lambda x: abs(x - requested_tau))
        print(
            "Requested tau={:d} not found for {:s}; using nearest checkpoint tau={:d}.".format(
                requested_tau, meta.dataset, used_tau
            )
        )
        return requested_tau, used_tau

    if len(available) > 12:
        head = ", ".join(str(x) for x in available[:6])
        tail = ", ".join(str(x) for x in available[-6:])
        avail_msg = "{:s}, ..., {:s}".format(head, tail)
    else:
        avail_msg = ", ".join(str(x) for x in available)

    raise FileNotFoundError(
        (
            "Requested tau={:d} not found for dataset {:s} in experiment {:s}. "
            "Available checkpoints include: {:s}. "
            "Use --tau_policy nearest to auto-select the closest checkpoint."
        ).format(requested_tau, meta.dataset, meta.experiment_name, avail_msg)
    )


def get_generated_pool(meta, config, tau, args):
    pool = load_generated_pool(meta.experiment_path, tau, args.sample_pool_size)
    if pool is not None:
        return pool

    if args.no_generate_fallback:
        raise FileNotFoundError(
            "No generated samples found for tau={:d} in {:s}".format(tau, meta.experiment_path)
        )

    print(
        "No generated samples for dataset={:s}, tau={:d}; generating from checkpoint.".format(
            meta.dataset, tau
        )
    )
    return generate_pool_from_checkpoint(
        meta=meta,
        config=config,
        exp_path=meta.experiment_path,
        tau=tau,
        pool_size=args.sample_pool_size,
        batch_size=args.generation_batch_size,
        ddim_steps=args.ddim_steps,
    )


def get_mean_std_tensors(config, device="cpu", dtype=torch.float32):
    mean = torch.as_tensor(config.mean, dtype=dtype, device=device).view(1, -1, 1, 1)
    std = torch.as_tensor(config.std, dtype=dtype, device=device).view(1, -1, 1, 1)
    return mean, std


def detransform_images(images, config):
    mean, std = get_mean_std_tensors(config, device=images.device, dtype=images.dtype)
    return images * std + mean


def nearest_neighbor_indices(train_images, gen_images, chunk_size=64):
    train_flat = train_images.reshape(train_images.shape[0], -1)
    gen_flat = gen_images.reshape(gen_images.shape[0], -1)

    nn_indices = []
    nn_distances = []

    for i in range(0, gen_flat.shape[0], chunk_size):
        j = min(i + chunk_size, gen_flat.shape[0])
        g = gen_flat[i:j]
        d = torch.cdist(g, train_flat, p=2)
        min_d, min_i = torch.min(d, dim=1)
        nn_indices.append(min_i.cpu())
        nn_distances.append(min_d.cpu())

    return torch.cat(nn_indices, dim=0), torch.cat(nn_distances, dim=0)


def select_display_indices(nn_distances, rows, mode):
    n = nn_distances.shape[0]
    if rows > n:
        rows = n

    if mode == "closest":
        idx = torch.argsort(nn_distances)[:rows]
        return idx.tolist()

    all_idx = list(range(n))
    random.shuffle(all_idx)
    return all_idx[:rows]


def read_metric_for_tau(file_path, tau, value_col=1):
    if not os.path.exists(file_path):
        return None

    values = None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) <= value_col:
                continue
            try:
                t = int(parts[0])
            except ValueError:
                continue
            if t == tau:
                try:
                    values = float(parts[value_col])
                except ValueError:
                    values = None
                break
    return values


def plot_one_dataset(meta, taus, train_images, config, args):
    rows = args.rows
    n_tau = len(taus)

    # Build a grid with very tight spacing inside each tau pair and wider spacing between tau blocks.
    n_grid_cols = 3 * n_tau - 1  # [gen, nn, spacer] x (n_tau-1) + [gen, nn]
    width_ratios = []
    for i in range(n_tau):
        width_ratios.extend([1.0, 1.0])
        if i != n_tau - 1:
            width_ratios.append(0.28)

    fig = plt.figure(figsize=(4.0 * n_tau + 1.0, 2.2 * rows))
    gs = fig.add_gridspec(
        rows,
        n_grid_cols,
        width_ratios=width_ratios,
        wspace=0.02,
        hspace=0.02,
    )

    axes = [[None for _ in range(n_grid_cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(n_grid_cols):
            axes[r][c] = fig.add_subplot(gs[r, c])

    fid_file = os.path.join(meta.experiment_path, "FID", "FID_{:d}.txt".format(args.fid_id))
    fmem_file = os.path.join(meta.experiment_path, "Memorization", "fraction_memorized.txt")

    for tau_idx, tau in enumerate(taus):
        requested_tau, used_tau = resolve_tau(meta, tau, args.tau_policy)

        gen_pool = get_generated_pool(meta, config, used_tau, args)
        gen_pool = gen_pool.float().cpu()

        nn_idx, nn_dist = nearest_neighbor_indices(train_images, gen_pool)
        chosen = select_display_indices(nn_dist, rows, args.selection)

        gen_show = detransform_images(gen_pool[chosen], config).clamp(0.0, 1.0)
        nn_show = detransform_images(train_images[nn_idx[chosen]], config).clamp(0.0, 1.0)

        fid_val = read_metric_for_tau(fid_file, used_tau, value_col=1)
        fmem_val = read_metric_for_tau(fmem_file, used_tau, value_col=1)

        if fid_val is None:
            fid_str = "NA"
        else:
            fid_str = "{:.1f}".format(fid_val)

        if fmem_val is None:
            fmem_str = "NA"
        else:
            fmem_str = "{:.1f}%".format(fmem_val)

        block_title = "n = {:d}, tau = {:d}\nFID = {:s}, f_mem = {:s}".format(
            meta.n, used_tau, fid_str, fmem_str
        )

        c0 = 3 * tau_idx
        c1 = c0 + 1

        axes[0][c0].set_title(block_title, fontsize=12, pad=8)

        for r in range(rows):
            g = gen_show[r]
            nimg = nn_show[r]

            if g.shape[0] == 1:
                axes[r][c0].imshow(g[0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
                axes[r][c1].imshow(nimg[0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            else:
                axes[r][c0].imshow(np.transpose(g.numpy(), (1, 2, 0)))
                axes[r][c1].imshow(np.transpose(nimg.numpy(), (1, 2, 0)))

            axes[r][c0].axis("off")
            axes[r][c1].axis("off")

            spacer_col = c0 + 2
            if spacer_col < n_grid_cols:
                axes[r][spacer_col].axis("off")

        axes[rows - 1][c0].set_xlabel("Generated", fontsize=12)
        axes[rows - 1][c1].set_xlabel("Nearest\nNeighbor", fontsize=12)

    fig.suptitle("{:s} - Generated Samples and Nearest Neighbors".format(meta.dataset), fontsize=14, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_name = "generated_vs_nn_{:s}.png".format(meta.experiment_name)
    out_path = os.path.join(args.output_dir, out_name)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Saved {:s}".format(out_path))


def ensure_device(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu.")
        return "cpu"
    return device


def main():
    args = parse_args()
    if args.sample_pool_size < args.rows:
        raise ValueError("sample_pool_size must be >= rows.")

    args.device = ensure_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    meta = parse_experiment_meta(args.experiment)
    print("Preparing dataset {:s} from {:s}".format(meta.dataset, meta.experiment_name))
    train_images, config = load_training_data(meta, args)
    train_images = train_images.float().cpu()
    config.DEVICE = args.device

    plot_one_dataset(meta, args.taus, train_images, config, args)


if __name__ == "__main__":
    main()
