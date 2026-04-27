"""Plot FID and memorization curves for one or more evaluated experiment folders.

The script expects:
  - FID/FID_*.txt (2 columns: tau, fid)
  - Memorization/fraction_memorized.txt
    (at least 2 columns: tau, f_mem_percent; optional CI columns)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot FID and f_mem versus training time for one or more experiments."
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Path to one experiment folder inside Saves/ (or absolute path).",
    )
    parser.add_argument(
        "--experiment_dirs",
        type=str,
        nargs="+",
        default=None,
        help="One or more experiment folders inside Saves/ (or absolute paths).",
    )
    parser.add_argument(
        "--fid_file",
        type=str,
        default="FID/FID_1.txt",
        help="Path to FID file relative to experiment_dir.",
    )
    parser.add_argument(
        "--fmem_file",
        type=str,
        default="Memorization/fraction_memorized.txt",
        help="Path to memorization file relative to experiment_dir.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=None,
        help="Optional dataset size n used for legend and inset x-axis scaling (tau/n) for one run.",
    )
    parser.add_argument(
        "--dataset_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Optional dataset sizes n for each run in --experiment_dirs (same order).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Legend labels for each run in --experiment_dirs (same order).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional plot title.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Defaults to <experiment_dir>/FID/fid_fmem_plot.png",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(7.0, 6.0),
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI.",
    )
    parser.add_argument(
        "--no_inset",
        action="store_true",
        help="Disable inset showing normalized f_mem versus tau/n.",
    )
    return parser.parse_args()


def resolve_experiment_dir(path):
    if os.path.isabs(path):
        return os.path.normpath(path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    candidate_1 = os.path.normpath(os.path.join(experiments_root, path))
    candidate_2 = os.path.normpath(os.path.join(experiments_root, "Saves", path))

    if os.path.isdir(candidate_1):
        return candidate_1
    if os.path.isdir(candidate_2):
        return candidate_2

    raise FileNotFoundError(
        "Could not resolve experiment_dir. Tried:\n  {:s}\n  {:s}".format(candidate_1, candidate_2)
    )


def read_two_column_file(file_path):
    data = np.loadtxt(file_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError("File must contain at least 2 columns: {:s}".format(file_path))
    return data[:, 0], data[:, 1]


def read_fmem_file(file_path):
    data = np.loadtxt(file_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError("f_mem file must contain at least 2 columns: {:s}".format(file_path))

    tau = data[:, 0]
    fmem = data[:, 1]

    lower = None
    upper = None
    if data.shape[1] >= 5:
        lower = data[:, 3]
        upper = data[:, 4]

    return tau, fmem, lower, upper


def trim_trailing_invalid_rows(tau, *arrays, valid_mask):
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        return [tau, *arrays]

    end_index = valid_indices[-1] + 1
    out = [tau[:end_index]]
    for arr in arrays:
        if arr is None:
            out.append(None)
        else:
            out.append(arr[:end_index])
    return out


def keep_positive_tau(tau, *arrays):
    mask = tau > 0
    out = [tau[mask]]
    for arr in arrays:
        if arr is None:
            out.append(None)
        else:
            out.append(arr[mask])
    return out


def get_experiment_dirs(args):
    exp_dirs = []
    if args.experiment_dirs is not None:
        exp_dirs.extend(args.experiment_dirs)
    if args.experiment_dir is not None:
        exp_dirs.append(args.experiment_dir)
    if len(exp_dirs) == 0:
        raise ValueError("Please provide --experiment_dir or --experiment_dirs.")
    return [resolve_experiment_dir(path) for path in exp_dirs]


def get_dataset_sizes(args, num_experiments):
    if args.dataset_sizes is not None:
        if len(args.dataset_sizes) != num_experiments:
            raise ValueError(
                "--dataset_sizes must have one value per experiment ({:d} given, {:d} required).".format(
                    len(args.dataset_sizes), num_experiments
                )
            )
        return list(args.dataset_sizes)

    if args.dataset_size is not None:
        if num_experiments == 1:
            return [args.dataset_size]
        return [None] * num_experiments

    return [None] * num_experiments


def get_labels(args, exp_dirs, dataset_sizes):
    if args.labels is not None:
        if len(args.labels) != len(exp_dirs):
            raise ValueError(
                "--labels must have one value per experiment ({:d} given, {:d} required).".format(
                    len(args.labels), len(exp_dirs)
                )
            )
        return list(args.labels)

    labels = []
    for exp_dir, dataset_size in zip(exp_dirs, dataset_sizes):
        if dataset_size is not None:
            labels.append("n={:d}".format(dataset_size))
        else:
            labels.append(os.path.basename(exp_dir))
    return labels


def main():
    args = parse_arguments()
    exp_dirs = get_experiment_dirs(args)
    dataset_sizes = get_dataset_sizes(args, len(exp_dirs))
    labels = get_labels(args, exp_dirs, dataset_sizes)

    runs = []
    for exp_dir, dataset_size, label in zip(exp_dirs, dataset_sizes, labels):
        fid_path = os.path.join(exp_dir, args.fid_file)
        fmem_path = os.path.join(exp_dir, args.fmem_file)

        if not os.path.isfile(fid_path):
            raise FileNotFoundError("Missing FID file: {:s}".format(fid_path))
        if not os.path.isfile(fmem_path):
            raise FileNotFoundError("Missing f_mem file: {:s}".format(fmem_path))

        tau_fid, fid = read_two_column_file(fid_path)
        tau_mem, fmem, fmem_lower, fmem_upper = read_fmem_file(fmem_path)

        tau_fid, fid = trim_trailing_invalid_rows(tau_fid, fid, valid_mask=fid > 0)
        mem_valid_mask = fmem > 0
        if fmem_lower is not None:
            mem_valid_mask = np.logical_or(mem_valid_mask, fmem_lower > 0)
        if fmem_upper is not None:
            mem_valid_mask = np.logical_or(mem_valid_mask, fmem_upper > 0)
        tau_mem, fmem, fmem_lower, fmem_upper = trim_trailing_invalid_rows(
            tau_mem, fmem, fmem_lower, fmem_upper, valid_mask=mem_valid_mask
        )

        tau_fid, fid = keep_positive_tau(tau_fid, fid)
        tau_mem, fmem, fmem_lower, fmem_upper = keep_positive_tau(
            tau_mem, fmem, fmem_lower, fmem_upper
        )

        if len(tau_fid) == 0 or len(tau_mem) == 0:
            raise ValueError(
                "No positive tau values found to plot for {:s} (log x-axis requires tau > 0).".format(
                    exp_dir
                )
            )

        runs.append(
            {
                "exp_dir": exp_dir,
                "label": label,
                "dataset_size": dataset_size,
                "tau_fid": tau_fid,
                "fid": fid,
                "tau_mem": tau_mem,
                "fmem": fmem,
                "fmem_lower": fmem_lower,
                "fmem_upper": fmem_upper,
            }
        )

    x_min = min(
        min(float(np.min(run["tau_fid"])), float(np.min(run["tau_mem"]))) for run in runs
    )
    x_max = max(
        max(float(np.max(run["tau_fid"])), float(np.max(run["tau_mem"]))) for run in runs
    )

    fig, ax_fid = plt.subplots(figsize=tuple(args.figsize))
    ax_mem = ax_fid.twinx()

    cmap = plt.get_cmap("tab10")
    for idx, run in enumerate(runs):
        color = cmap(idx % 10)
        ax_fid.plot(run["tau_fid"], run["fid"], color=color, lw=2.2, label=run["label"])
        ax_mem.plot(run["tau_mem"], run["fmem"], color=color, lw=2.0, ls="--")

        if run["fmem_lower"] is not None and run["fmem_upper"] is not None:
            ax_mem.fill_between(
                run["tau_mem"],
                run["fmem_lower"],
                run["fmem_upper"],
                color=color,
                alpha=0.12,
                linewidth=0,
            )

    ax_fid.set_xscale("log")
    ax_fid.set_xlim(x_min, x_max)
    ax_fid.set_xlabel(r"$\tau$", fontsize=16)
    ax_fid.set_ylabel("FID", fontsize=16)
    ax_mem.set_ylabel(r"$f_{\mathrm{mem}}$ [%]", fontsize=16)
    ax_mem.set_ylim(0, 100)

    ax_fid.grid(alpha=0.25, linestyle="-")
    ax_fid.tick_params(axis="both", labelsize=12)
    ax_mem.tick_params(axis="y", labelsize=12)
    ax_fid.legend(loc="lower left", frameon=False, fontsize=12)

    if args.title:
        ax_fid.set_title(args.title, fontsize=14)

    has_any_dataset_size = any(run["dataset_size"] is not None for run in runs)
    if (not args.no_inset) and has_any_dataset_size:
        inset = ax_fid.inset_axes([0.43, 0.61, 0.4, 0.37])
        inset_plotted = False
        for idx, run in enumerate(runs):
            if run["dataset_size"] is None:
                continue

            max_mem = np.max(run["fmem"])
            if max_mem <= 0:
                continue

            color = cmap(idx % 10)
            tau_scaled = run["tau_mem"] / float(run["dataset_size"])
            fmem_norm = run["fmem"] / max_mem
            inset.plot(tau_scaled, fmem_norm, color=color, lw=2.0, ls="--")

            if run["fmem_lower"] is not None and run["fmem_upper"] is not None:
                inset.fill_between(
                    tau_scaled,
                    run["fmem_lower"] / max_mem,
                    run["fmem_upper"] / max_mem,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
            inset_plotted = True

        if inset_plotted:
            inset.set_xscale("log")
            inset.set_ylim(0.0, 1.0)
            inset.set_xlabel(r"$\tau/n$", fontsize=11)
            inset.set_ylabel(r"$f_{\mathrm{mem}}(\tau)/f_{\mathrm{mem}}(\tau_{\max})$", fontsize=11)
            inset.tick_params(axis="both", labelsize=8)
            inset.grid(alpha=0.2)
        else:
            inset.remove()

    fig.tight_layout()

    output_path = args.output
    if output_path is None:
        if len(runs) == 1:
            output_path = os.path.join(runs[0]["exp_dir"], "FID", "fid_fmem_plot.png")
        else:
            output_path = os.path.join("/work3/s204164/Why-Diffusion-Models-Don-t-Memorize/Experiments/Results", "fid_fmem_plot_multi.png")
    elif not os.path.isabs(output_path):
        output_path = os.path.join(runs[0]["exp_dir"], output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("Saved plot to: {:s}".format(output_path))


if __name__ == "__main__":
    main()