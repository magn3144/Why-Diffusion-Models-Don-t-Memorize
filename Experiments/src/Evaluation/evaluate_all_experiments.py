"""Run compute_fmem.py and compute_FID.py across all experiments in Saves/."""

import argparse
import glob
import os
import re
import subprocess
import sys


CELEBA_RE = re.compile(
    r'^CelebA(?P<img_size>\d+)_(?P<num>\d+)_(?P<nbase>\d+)_(?P<optim>.+?)_(?P<batch_size>\d+)_(?P<lr>\d+\.\d+)_index(?P<index>\d+)(?:_t(?P<time>-?\d+))?$'
)

SPRITES_RE = re.compile(
    r'^(?P<model_type>unet|gmm)_Sprites(?P<img_size>\d+)_(?P<num>\d+)_(?P<nbase>\d+)_(?P<optim>.+?)_(?P<batch_size>\d+)_(?P<lr>\d+\.\d+)_seed(?P<seed>\d+)(?:_t(?P<time>-?\d+))?(?:_(?P<tag>.+))?$'
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


def has_existing_metrics(exp_path, id_stat):
    """Check whether both memorization and FID outputs already exist for an experiment."""
    fmem_file = os.path.join(exp_path, 'Memorization', 'fraction_memorized.txt')
    fid_file = os.path.join(exp_path, 'FID', 'FID_{:d}.txt'.format(id_stat))
    return os.path.isfile(fmem_file) and os.path.isfile(fid_file)


def build_common_cmd(py_exec, script, meta, exp_name, device):
    """Build common CLI args used by both metric scripts."""
    cmd = [
        py_exec,
        script,
        '-D', meta['dataset'],
        '-n', str(meta['num']),
        '-s', str(meta['img_size']),
        '-LR', str(meta['learning_rate']),
        '-O', meta['optim'],
        '-W', str(meta['nbase']),
        '-B', str(meta['batch_size']),
        '-t', str(meta['time']),
        '--experiment_dir', exp_name,
        '--device', device,
    ]

    if meta['dataset'] == 'CelebA':
        cmd += ['-i', str(meta['index'])]
    else:
        cmd += ['--model_type', meta['model_type'], '--seed', str(meta['seed'])]
        if meta['tag']:
            cmd += ['--tag', meta['tag']]

    return cmd


def run_command(cmd):
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(description='Evaluate all experiment folders in Saves/.')
    parser.add_argument('--saves_dir', type=str, default='../../Saves', help='Path to Saves directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for evaluation scripts')
    parser.add_argument('--id_stat', type=int, default=1, help='FID stats file index (stats{id_stat}.npz)')
    parser.add_argument('--batch_size_samples', type=int, default=100, help='Generated batch size per samples_a_i file')
    parser.add_argument('--gap_threshold', type=float, default=1.0 / 3.0, help='Memorization threshold')
    parser.add_argument('--skip_fid', action='store_true', help='Skip FID evaluation')
    parser.add_argument('--skip_fmem', action='store_true', help='Skip memorization evaluation')
    parser.add_argument('--reuse_stats', action='store_true', help='Reuse existing stats{id_stat}.npz for all experiments')
    parser.add_argument('--only', type=str, default='', help='Evaluate only folders containing this substring')
    args = parser.parse_args()

    py_exec = sys.executable
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    script_fmem = os.path.join(eval_dir, 'compute_fmem.py')
    script_fid = os.path.join(eval_dir, 'compute_FID.py')

    exp_dirs = sorted(
        d for d in os.listdir(args.saves_dir)
        if os.path.isdir(os.path.join(args.saves_dir, d))
    )

    if args.only:
        exp_dirs = [d for d in exp_dirs if args.only in d]

    if not exp_dirs:
        raise FileNotFoundError('No experiment folders found in {:s}'.format(args.saves_dir))

    evaluated = 0
    skipped = 0

    for exp_name in exp_dirs:
        exp_path = os.path.join(args.saves_dir, exp_name)
        models_dir = os.path.join(exp_path, 'Models')
        if not os.path.isdir(models_dir):
            continue

        if has_existing_metrics(exp_path, args.id_stat):
            print('Skipping (FID and fmem already exist):', exp_name)
            skipped += 1
            continue

        meta = parse_experiment_name(exp_name)
        if meta is None:
            print('Skipping unsupported folder name:', exp_name)
            skipped += 1
            continue

        n_batches = infer_sample_batches(exp_path)
        if n_batches <= 0:
            print('Skipping (no generated samples):', exp_name)
            skipped += 1
            continue

        print('\n=== Evaluating {:s} ==='.format(exp_name))
        print('Detected generated batches per checkpoint:', n_batches)

        common_fmem = build_common_cmd(py_exec, script_fmem, meta, exp_name, args.device)
        common_fid = build_common_cmd(py_exec, script_fid, meta, exp_name, args.device)

        if not args.skip_fmem:
            cmd_fmem = common_fmem + [
                '-Ns', str(n_batches),
                '--batch_sample_size', str(args.batch_size_samples),
                '--gap_threshold', str(args.gap_threshold),
            ]
            run_command(cmd_fmem)

        if not args.skip_fid:
            cmd_fid = common_fid + [
                '-istat', str(args.id_stat),
                '--N1', '0',
                '--N2', str(n_batches),
                '--batch_size_samples', str(args.batch_size_samples),
            ]
            if not args.reuse_stats:
                cmd_fid += ['--rebuild_stats']
            run_command(cmd_fid)

        evaluated += 1

    print('\nDone. Evaluated {:d} experiment(s), skipped {:d}.'.format(evaluated, skipped))


if __name__ == '__main__':
    main()
