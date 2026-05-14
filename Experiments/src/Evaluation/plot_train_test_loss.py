"""Plot train/test loss curves for multiple Sprites experiments.

This script reads test losses from Models/test_loss_checkpoints.csv and
computes train losses (cached to Models/train_loss_checkpoints.csv) if needed.
"""

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# Add Utils to path (script is in Experiments/src/Evaluation)
sys.path.insert(1, "../Utils/")

import Diffusion as dm
import loader
import sprites_dataset
import Unet


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

SPRITES_RE = re.compile(
    r"^(?P<model_type>unet|gmm)_Sprites(?P<img_size>\d+)_(?P<num>\d+)_(?P<nbase>\d+)_(?P<optim>.+?)_(?P<batch_size>\d+)_(?P<lr>\d+\.\d+)_seed(?P<seed>\d+)(?:_t(?P<time>-?\d+))?(?:_(?P<tag>.+))?$"
)


@dataclass
class ExperimentMeta:
    experiment_name: str
    experiment_path: str
    img_size: int
    n: int
    nbase: int
    optim: str
    batch_size: int
    lr: float
    seed: int
    model_type: str


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


def resolve_experiment_path(exp_arg):
    if os.path.isabs(exp_arg):
        path = exp_arg
    else:
        path = os.path.join(EXPERIMENTS_ROOT, "Saves", exp_arg)
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError("Experiment folder not found: {:s}".format(path))
    return path


def parse_experiment_meta(exp_arg):
    exp_path = resolve_experiment_path(exp_arg)
    name = os.path.basename(os.path.normpath(exp_path))

    m = SPRITES_RE.match(name)
    if not m:
        raise ValueError("Could not parse Sprites experiment name: {:s}".format(name))

    d = m.groupdict()
    return ExperimentMeta(
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


def read_loss_csv(file_path, loss_field):
    if not os.path.isfile(file_path):
        return [], [], []

    steps = []
    losses = []
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "step" not in row or loss_field not in row:
                continue
            try:
                step = int(row["step"])
                loss = float(row[loss_field])
            except (ValueError, TypeError):
                continue

            sample_field = None
            if "n_train_samples" in row:
                sample_field = "n_train_samples"
            elif "n_test_samples" in row:
                sample_field = "n_test_samples"

            n_samples = 0
            if sample_field is not None:
                try:
                    n_samples = int(row[sample_field])
                except (ValueError, TypeError):
                    n_samples = 0

            steps.append(step)
            losses.append(loss)
            samples.append(n_samples)

    return steps, losses, samples


def write_loss_csv(file_path, steps, losses, samples, loss_field):
    if loss_field == "train_loss":
        sample_field = "n_train_samples"
    else:
        sample_field = "n_test_samples"

    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", loss_field, sample_field])
        for step, loss, n_samples in zip(steps, losses, samples):
            writer.writerow([step, "{:.8f}".format(loss), n_samples])


def build_sprites_loaders(meta, device, timesteps, n_test_eval):
    config = dm.TrainingConfig()
    config.DATASET = "Sprites"
    config.path_save = os.path.join(EXPERIMENTS_ROOT, "Saves") + os.sep
    config.path_data = os.path.join(EXPERIMENTS_ROOT, "Data")
    config.IMG_SHAPE = (3, meta.img_size, meta.img_size)
    config.n_images = meta.n
    config.BATCH_SIZE = min(meta.batch_size, meta.n)
    config.OPTIM = meta.optim
    config.LR = meta.lr
    config.DEVICE = device
    config.TIMESTEPS = timesteps
    config.CENTER = True
    config.STANDARDIZE = False
    config.mode = "normal"
    config.time_step = -1

    data_file = os.path.join(config.path_data, "sprites_1788_16x16.npy")
    if not os.path.isfile(data_file):
        raise FileNotFoundError("Missing sprites data file: {:s}".format(data_file))

    base_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((meta.img_size, meta.img_size)),
        ]
    )
    full_dataset = sprites_dataset.SpritesDataset(
        transform=base_transform,
        img_file=data_file,
        num_samples=None,
        seed=meta.seed,
    )

    n_full = len(full_dataset)
    n_train_full = int(0.8 * n_full)
    n_test_full = n_full - n_train_full
    if meta.n > n_train_full:
        raise ValueError(
            "Requested n={:d} but 80% sprites train split has {:d}.".format(meta.n, n_train_full)
        )

    rng = np.random.RandomState(meta.seed)
    perm = rng.permutation(n_full)
    train_indices = perm[:n_train_full]
    test_indices = perm[n_train_full:]

    train_subset_indices = train_indices[: meta.n]
    base_train_dataset = Subset(full_dataset, train_subset_indices.tolist())

    test_eval_count = 0
    if n_test_eval is not None:
        test_eval_count = int(max(0, min(n_test_eval, n_test_full)))
    base_test_dataset = None
    if test_eval_count > 0:
        base_test_dataset = Subset(full_dataset, test_indices[:test_eval_count].tolist())

    mean, std = compute_channel_stats(base_train_dataset, batch_size=config.BATCH_SIZE)
    std = torch.ones_like(std)
    config.mean = mean
    config.std = std

    train_norm = NormalizedDataset(base_train_dataset, mean, std)
    test_norm = NormalizedDataset(base_test_dataset, mean, std) if base_test_dataset is not None else None

    train_loader = DataLoader(train_norm, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = None
    if test_norm is not None:
        test_loader = DataLoader(test_norm, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, config


def build_model(meta, config):
    if meta.model_type.lower() != "unet":
        raise ValueError("Only unet model_type is supported for this plot.")

    return Unet.UNet(
        input_channels=config.IMG_SHAPE[0],
        output_channels=config.IMG_SHAPE[0],
        base_channels=meta.nbase,
        base_channels_multiples=(1, 2, 4),
        apply_attention=(False, True, True),
        dropout_rate=0.1,
    )


def ensure_train_losses(exp_dir, meta, steps, device, timesteps, train_loss_path, recompute=False):
    train_steps, train_losses, train_samples = read_loss_csv(train_loss_path, "train_loss")
    train_loss_by_step = {s: l for s, l in zip(train_steps, train_losses)}

    missing_steps = []
    if recompute:
        missing_steps = list(steps)
    else:
        missing_steps = [s for s in steps if s not in train_loss_by_step]

    if missing_steps:
        train_loader, _, config = build_sprites_loaders(meta, device, timesteps, n_test_eval=0)
        n_train_samples = len(train_loader.dataset)

        model = build_model(meta, config)
        model.to(config.DEVICE)
        loss_fn = nn.MSELoss()
        df = dm.DiffusionConfig(
            n_steps=config.TIMESTEPS,
            img_shape=config.IMG_SHAPE,
            device=config.DEVICE,
        )

        for step in missing_steps:
            checkpoint = os.path.join(exp_dir, "Models", "Model_{:d}".format(step))
            if not os.path.isfile(checkpoint):
                print("Missing checkpoint: {:s}".format(checkpoint))
                continue

            loader.load_model(model, checkpoint, verbose=False)
            train_loss = dm.evaluate_loss(train_loader, model, loss_fn, config, df)
            train_loss_by_step[step] = train_loss

        steps_sorted = sorted(train_loss_by_step.keys())
        losses_sorted = [train_loss_by_step[s] for s in steps_sorted]
        samples_sorted = [n_train_samples for _ in steps_sorted]
        write_loss_csv(train_loss_path, steps_sorted, losses_sorted, samples_sorted, "train_loss")

    return train_loss_by_step


def ensure_device(device):
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu.")
        return "cpu"
    return device


def plot_train_test_loss(
    experiments,
    output_path,
    device="auto",
    timesteps=1000,
    recompute_train=False,
    title="Sprites train/test loss",
):
    device = ensure_device(device)

    runs = []
    for exp in experiments:
        meta = parse_experiment_meta(exp)
        exp_dir = meta.experiment_path

        test_loss_path = os.path.join(exp_dir, "Models", "test_loss_checkpoints.csv")
        test_steps, test_losses, test_samples = read_loss_csv(test_loss_path, "test_loss")
        if not test_steps:
            raise FileNotFoundError("Missing or empty test loss file: {:s}".format(test_loss_path))

        train_loss_path = os.path.join(exp_dir, "Models", "train_loss_checkpoints.csv")
        train_loss_by_step = ensure_train_losses(
            exp_dir,
            meta,
            test_steps,
            device,
            timesteps,
            train_loss_path,
            recompute=recompute_train,
        )

        aligned_steps = []
        aligned_train = []
        aligned_test = []
        for step, test_loss in zip(test_steps, test_losses):
            if step not in train_loss_by_step:
                continue
            aligned_steps.append(step)
            aligned_train.append(train_loss_by_step[step])
            aligned_test.append(test_loss)

        if not aligned_steps:
            raise RuntimeError("No overlapping train/test loss steps for {:s}".format(exp))

        aligned_steps = np.asarray(aligned_steps, dtype=float)
        aligned_train = np.asarray(aligned_train, dtype=float)
        aligned_test = np.asarray(aligned_test, dtype=float)

        positive_mask = aligned_steps > 0
        aligned_steps = aligned_steps[positive_mask]
        aligned_train = aligned_train[positive_mask]
        aligned_test = aligned_test[positive_mask]

        if len(aligned_steps) == 0:
            raise RuntimeError("No positive steps to plot for {:s}".format(exp))

        runs.append(
            {
                "label": "n = {:d}".format(meta.n),
                "steps": aligned_steps,
                "train": aligned_train,
                "test": aligned_test,
            }
        )

    x_min = min(float(np.min(run["steps"])) for run in runs)
    x_max = max(float(np.max(run["steps"])) for run in runs)

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    cmap = plt.get_cmap("tab10")

    for idx, run in enumerate(runs):
        color = cmap(idx % 10)
        ax.plot(run["steps"], run["train"], color=color, lw=2.2, label=run["label"])
        ax.plot(run["steps"], run["test"], color=color, lw=2.0, ls="--")

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(r"$\tau$", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    if title:
        ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.25, linestyle="-")
    ax.tick_params(axis="both", labelsize=12)

    legend_n = ax.legend(loc="upper left", frameon=False, fontsize=12)
    ax.add_artist(legend_n)

    style_handles = [
        Line2D([0], [0], color="black", lw=2.2, label=r"$\mathcal{L}_{\mathrm{train}}$"),
        Line2D([0], [0], color="black", lw=2.2, ls="--", label=r"$\mathcal{L}_{\mathrm{test}}$"),
    ]
    ax.legend(
        handles=style_handles,
        loc="lower left",
        frameon=False,
        fontsize=12,
        title="Line style",
        title_fontsize=11,
    )

    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(EXPERIMENTS_ROOT, "Results", "train_test_loss.png")
    elif not os.path.isabs(output_path):
        output_path = os.path.join(EXPERIMENTS_ROOT, output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Saved plot to: {:s}".format(output_path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot train/test loss curves for Sprites experiments."
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=[
            "unet_Sprites16_512_32_Adam_512_0.0001_seed1",
            "unet_Sprites16_1024_32_Adam_512_0.0001_seed1_new",
            "unet_Sprites16_2048_32_Adam_512_0.0001_seed1",
            "unet_Sprites16_4096_32_Adam_512_0.0001_seed1",
        ],
        help="Experiment folder names under Saves/ (or absolute paths).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("Results", "train_test_loss.png"),
        help="Output image path (relative to Experiments/).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda:0, ...).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps.",
    )
    parser.add_argument(
        "--recompute-train",
        action="store_true",
        help="Recompute train losses even if cached.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Sprites train/test loss",
        help="Plot title.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_train_test_loss(
        experiments=args.experiments,
        output_path=args.output,
        device=args.device,
        timesteps=args.timesteps,
        recompute_train=args.recompute_train,
        title=args.title,
    )


if __name__ == "__main__":
    main()
