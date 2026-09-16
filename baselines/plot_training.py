"""Training curves from the baselines' logs (DOLCE, SWORD, FBPConvNet).

Reads the `iter N: loss L (H h)` lines of dolce/train.py and sword/train.py and
the `epoch N: train MSE .., val MSE ..` lines of fbpconvnet/train.py; anything
else in the file (tracebacks, SLURM output) is ignored, so a slurm-*.out works
as well as log.txt. Log-scale loss against iterations, with a smoothed curve.

    python -m baselines.plot_training output/baselines/limited0-45_stride10/dolce/loss.csv
    python -m baselines.plot_training output/baselines/limited0-45_stride10/dolce/log.txt
    python -m baselines.plot_training output/baselines/limited0-45_stride10/sword/log_full.txt \
        output/baselines/limited0-45_stride10/sword/log_high.txt --out sword_training.png
"""
import argparse
import os
import re

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ITER = re.compile(r'iter (\d+): loss ([\d.eE+-]+) \(([\d.]+) h\)')
EPOCH = re.compile(r'epoch (\d+): train MSE ([\d.eE+-]+), val MSE ([\d.eE+-]+).*\(([\d.]+) h\)')


def parse(path):
    """(x label, iterations/epochs, hours, {curve name: values})."""
    if path.endswith('.csv'):      # loss.csv written by the trainers' TrainRecord
        rows = np.genfromtxt(path, delimiter=',', names=True)
        rows = np.atleast_1d(rows)
        x_name = rows.dtype.names[0]
        curves = {n.replace('_', ' '): rows[n] for n in rows.dtype.names[2:]}
        return x_name, rows[x_name], rows['hours'], curves
    with open(path, errors='replace') as f:
        text = f.read()
    it = [(int(i), float(l), float(h)) for i, l, h in ITER.findall(text)]
    if it:
        i, l, h = map(np.array, zip(*it))
        return 'iteration', i, h, {'train loss': l}
    ep = [(int(e), float(t), float(v), float(h)) for e, t, v, h in EPOCH.findall(text)]
    if ep:
        e, t, v, h = map(np.array, zip(*ep))
        return 'epoch', e, h, {'train MSE': t, 'val MSE': v}
    raise SystemExit(f'{path}: no training lines found')


def smooth(y, k):
    if k <= 1 or len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode='valid')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logs', nargs='+')
    parser.add_argument('--out', default=None, help='default: <first log>_curve.png')
    parser.add_argument('--smooth', type=int, default=20, help='moving-average window in log lines (0 = off)')
    parser.add_argument('--linear', action='store_true', help='linear loss axis from 0 instead of log scale')
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for path in args.logs:
        xlabel, x, hours, curves = parse(path)
        tag = os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)
        for name, y in curves.items():
            label = f'{tag} {name}' if len(args.logs) > 1 else name
            line, = ax.plot(x, y, alpha=0.3 if args.smooth > 1 else 1.0, lw=0.8, label=label if args.smooth <= 1 else None)
            if args.smooth > 1 and len(y) >= args.smooth:
                ys = smooth(y, args.smooth)
                ax.plot(x[args.smooth - 1:], ys, color=line.get_color(), lw=1.6,
                        label=f'{label} ({args.smooth}-point mean)')
        ax.set_xlabel(xlabel)
        print(f'{path}: {len(x)} points, {xlabel} {x[0]}..{x[-1]}, {hours[-1]:.2f} h; '
              + ', '.join(f'{n} last {v[-1]:.4g} (min {v.min():.4g} at {xlabel} {x[v.argmin()]})'
                          for n, v in curves.items()))
    if args.linear:
        ax.set_ylim(bottom=0)
    else:
        ax.set_yscale('log')
    ax.set_ylabel('loss')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    # hours on a second axis, from the last log (they all run in parallel anyway)
    top = ax.secondary_xaxis('top', functions=(lambda i: np.interp(i, x, hours), lambda h: np.interp(h, hours, x)))
    top.set_xlabel('hours')
    fig.tight_layout()
    out = args.out or os.path.splitext(args.logs[0])[0] + '_curve.png'
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
