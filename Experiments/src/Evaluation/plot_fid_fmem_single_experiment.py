"""Plot FID and memorization curves for one evaluated experiment folder.

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
        description="Plot FID and f_mem versus training time for a single experiment."
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to one experiment folder inside Saves/ (or absolute path).",
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
        help="Optional dataset size n used for legend and inset x-axis scaling (tau/n).",
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


def keep_positive_tau(tau, *arrays):
    mask = tau > 0
    out = [tau[mask]]
    for arr in arrays:
        if arr is None:
            out.append(None)
        else:
            out.append(arr[mask])
    return out


def main():
    args = parse_arguments()
    exp_dir = resolve_experiment_dir(args.experiment_dir)

    fid_path = os.path.join(exp_dir, args.fid_file)
    fmem_path = os.path.join(exp_dir, args.fmem_file)

    if not os.path.isfile(fid_path):
        raise FileNotFoundError("Missing FID file: {:s}".format(fid_path))
    if not os.path.isfile(fmem_path):
        raise FileNotFoundError("Missing f_mem file: {:s}".format(fmem_path))

    tau_fid, fid = read_two_column_file(fid_path)
    tau_mem, fmem, fmem_lower, fmem_upper = read_fmem_file(fmem_path)

    tau_fid, fid = keep_positive_tau(tau_fid, fid)
    tau_mem, fmem, fmem_lower, fmem_upper = keep_positive_tau(
        tau_mem, fmem, fmem_lower, fmem_upper
    )

    if len(tau_fid) == 0 or len(tau_mem) == 0:
        raise ValueError("No positive tau values found to plot (log x-axis requires tau > 0).")

    x_min = min(float(np.min(tau_fid)), float(np.min(tau_mem)))
    x_max = min(float(np.max(tau_fid)), float(np.max(tau_mem)))
    if x_max <= x_min:
        x_max = max(float(np.max(tau_fid)), float(np.max(tau_mem)))

    color = "#009E73"
    label = "n={:d}".format(args.dataset_size) if args.dataset_size is not None else "experiment"

    fig, ax_fid = plt.subplots(figsize=tuple(args.figsize))
    ax_mem = ax_fid.twinx()

    ax_fid.plot(tau_fid, fid, color=color, lw=2.2, label=label)
    ax_mem.plot(tau_mem, fmem, color=color, lw=2.2, ls="--")

    if fmem_lower is not None and fmem_upper is not None:
        ax_mem.fill_between(tau_mem, fmem_lower, fmem_upper, color=color, alpha=0.15, linewidth=0)

    ax_fid.set_xscale("log")
    ax_fid.set_xlim(x_min, x_max)
    ax_fid.set_xlabel(r"$\tau$", fontsize=16)
    ax_fid.set_ylabel("FID", fontsize=16)
    ax_mem.set_ylabel(r"$f_{\mathrm{mem}}$ [%]", fontsize=16)
    ax_mem.set_ylim(0, 100)

    ax_fid.grid(alpha=0.25, linestyle="-")
    ax_fid.tick_params(axis="both", labelsize=12)
    ax_mem.tick_params(axis="y", labelsize=12)
    ax_fid.legend(loc="best", frameon=False, fontsize=12)

    if args.title:
        ax_fid.set_title(args.title, fontsize=14)

    if (not args.no_inset) and args.dataset_size is not None:
        inset = ax_fid.inset_axes([0.43, 0.61, 0.4, 0.37])
        tau_scaled = tau_mem / float(args.dataset_size)
        max_mem = np.max(fmem)
        if max_mem > 0:
            fmem_norm = fmem / max_mem
            inset.plot(tau_scaled, fmem_norm, color=color, lw=2.0, ls="--")
            if fmem_lower is not None and fmem_upper is not None:
                inset.fill_between(
                    tau_scaled,
                    fmem_lower / max_mem,
                    fmem_upper / max_mem,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
            inset.set_xscale("log")
            inset.set_ylim(0.0, 1.0)
            inset.set_xlabel(r"$\tau/n$", fontsize=11)
            inset.set_ylabel(r"$f_{\mathrm{mem}}(\tau)/f_{\mathrm{mem}}(\tau_{\max})$", fontsize=11)
            inset.tick_params(axis="both", labelsize=8)
            inset.grid(alpha=0.2)

    fig.tight_layout()

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(exp_dir, "FID", "fid_fmem_plot.png")
    elif not os.path.isabs(output_path):
        output_path = os.path.join(exp_dir, output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("Saved plot to: {:s}".format(output_path))


if __name__ == "__main__":
    main()