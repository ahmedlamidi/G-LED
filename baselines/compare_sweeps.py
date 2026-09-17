"""FBP vs SWORD vs SD-Flow on the view settings all three have: the
intersection of the baselines' view sweep (baselines/sweep.py) and the SD-Flow
model sweep (baselines/sdflow/model_sweep.py), same 10 test slices.

Writes to <out>/: intersection.md / .csv (mean +- std per setting and method),
metrics.png (body SSIM and PSNR per setting, one dot per slice), and, when the
sweep cache with the labels and FBP images is available, grid.png: for each
slice a block of three rows (FBP, SWORD, SD-Flow), one column per setting, the
label last, in the standard overview layout.

    python -m baselines.compare_sweeps
    python -m baselines.compare_sweeps --slices 5          # every other slice in the grid
"""
import argparse
import csv
import json
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from .common.config import DEFAULT_CACHE_ROOT, DEFAULT_OUT_ROOT  # noqa: E402
from .common.metrics import METRICS, to_hu, water_level  # noqa: E402
from .evaluate import WINDOWS, body_box  # noqa: E402

# (baseline sweep name, SD-Flow model name, cache folder tag)
SHARED = [('45°/1', '45/1', 'limited0-45_stride1'), ('90°/1', '90/1', 'limited0-90_stride1'),
          ('45°/10', '45/10', 'limited0-45_stride10'), ('360°/2', '360/2', 'limited0-360_stride2'),
          ('360°/4', '360/4', 'limited0-360_stride4'), ('360°/8', '360/8', 'limited0-360_stride8'),
          ('360°/10', '360/10', 'limited0-360_stride10')]
METHODS = [('FBP', 'tab:gray'), ('SWORD', 'tab:blue'), ('SD-Flow', 'tab:red')]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sword_root', default=os.path.join(DEFAULT_OUT_ROOT, 'sword_sweep'))
    parser.add_argument('--sdflow_root', default=os.path.join(DEFAULT_OUT_ROOT, 'sdflow_sweep'))
    parser.add_argument('--cache_root', default=os.path.join(DEFAULT_CACHE_ROOT, 'sword_sweep'),
                        help="the baselines sweep's per-setting caches (labels, FBP)")
    parser.add_argument('--out', default=os.path.join(DEFAULT_OUT_ROOT, 'compare_sweeps'))
    parser.add_argument('--slices', type=int, default=5, help='slices in the grid (evenly spaced from the 10)')
    parser.add_argument('--window', default='lung', choices=list(WINDOWS))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.sword_root, 'sweep', 'per_slice.csv')) as f:
        sw = list(csv.DictReader(f))
    with open(os.path.join(args.sdflow_root, 'sweep', 'per_slice.csv')) as f:
        sd = list(csv.DictReader(f))
    keys = [k for k, _ in METRICS]
    score = {}      # (setting, method, slice) -> row
    for r in sw:
        score[(r['setting'], r['method'], r['slice_id'])] = r
    for r in sd:
        score[(r['model'], 'SD-Flow', r['slice_id'])] = r
    ids = sorted({r['slice_id'] for r in sd} & {r['slice_id'] for r in sw})
    settings = [(b, s, tag) for b, s, tag in SHARED
                if all((b if m != 'SD-Flow' else s, m, sid) in score for m, _ in METHODS for sid in ids)]
    print(f'{len(ids)} slices, {len(settings)} shared settings: ' + ', '.join(b for b, _, _ in settings))

    def vals(b, s, method, key):
        return np.array([float(score[(b if method != 'SD-Flow' else s, method, sid)][key]) for sid in ids])

    lines = [f'FBP, SWORD and SD-Flow on the settings all three have, {len(ids)} test slices. Baseline arcs start at '
             '0 deg; SD-Flow uses its trained window (45: 285-330, 90: 300-390).', '',
             '| Setting | Method | ' + ' | '.join(t for _, t in METRICS) + ' |', '|---' * (len(METRICS) + 2) + '|']
    with open(os.path.join(args.out, 'intersection.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['setting', 'method'] + [f'{k}_{st}' for k in keys for st in ('mean', 'std')])
        for b, s, _ in settings:
            for method, _ in METHODS:
                cols = {k: vals(b, s, method, k) for k in keys}
                w.writerow([b, method] + [v for k in keys for v in (float(cols[k].mean()), float(cols[k].std()))])
                lines.append(f'| {b} | {method} | ' + ' | '.join(
                    f'{cols[k].mean():.3f} ± {cols[k].std():.3f}' if k.startswith('ssim') else
                    f'{cols[k].mean():.2f} ± {cols[k].std():.2f}' for k in keys) + ' |')
    with open(os.path.join(args.out, 'intersection.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
    x = np.arange(len(settings))
    for ax, key, ylab in [(axes[0], 'ssim_body', 'SSIM inside the body'), (axes[1], 'psnr_body', 'PSNR inside the body (dB)')]:
        for (method, colr), dx in zip(METHODS, (-0.2, 0.0, 0.2)):
            means = []
            for j, (b, s, _) in enumerate(settings):
                v = vals(b, s, method, key)
                ax.scatter(np.full(len(v), x[j] + dx), v, s=10, color=colr, alpha=0.3)
                means.append(v.mean())
            ax.plot(x + dx, means, 'o', color=colr, label=f'{method} (mean)')
        ax.set_xticks(x)
        ax.set_xticklabels([b for b, _, _ in settings])
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        if key == 'ssim_body':
            ax.set_ylim(-0.05, 1)
    axes[0].legend(fontsize=9, loc='upper left')
    fig.suptitle(f'FBP vs SWORD vs SD-Flow, settings shared by all three (arc / row stride), {len(ids)} slices',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'metrics.png'), dpi=130)
    plt.close(fig)

    # the grid needs the labels and FBP images from the sweep cache
    first = os.path.join(args.cache_root, settings[0][2], 'test')
    if not os.path.exists(os.path.join(first, 'label.npy')):
        print(f'no {first}/label.npy: skipping grid.png (run on the cluster, or copy the sweep caches)')
        return
    K = np.load(os.path.join(args.cache_root, settings[0][2], 'K.npy'))
    with open(os.path.join(first, 'slices.json')) as f:
        cache_ids = [s['id'] for s in json.load(f)]
    labels = np.load(os.path.join(first, 'label.npy'))
    fbp = {tag: np.load(os.path.join(args.cache_root, tag, 'test', 'fbp.npy')) for _, _, tag in settings}
    grid_ids = [ids[j] for j in np.linspace(0, len(ids) - 1, min(args.slices, len(ids))).round().astype(int)]
    level, width = WINDOWS[args.window]
    ncol = len(settings) + 1
    nrow = len(grid_ids) * len(METHODS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 1.75 * nrow + 0.8), squeeze=False)
    for gi, sid in enumerate(grid_ids):
        ci = cache_ids.index(sid)
        water = water_level(labels[ci] + K)
        hl = to_hu(labels[ci] + K, water)
        box = body_box(hl)
        for mi, (method, _) in enumerate(METHODS):
            r = gi * len(METHODS) + mi
            for c, (b, s, tag) in enumerate(settings):
                ax = axes[r, c]
                ax.set_xticks([])
                ax.set_yticks([])
                if method == 'FBP':
                    img = fbp[tag][ci]
                elif method == 'SWORD':
                    img = np.load(os.path.join(args.sword_root, tag, 'sword', 'recon', sid + '.npy'))
                else:
                    img = np.load(os.path.join(args.sdflow_root, s.replace('/', '_'), 'recon', sid + '.npy'))
                ax.imshow(to_hu(img + K, water)[box], cmap='gray', vmin=level - width / 2, vmax=level + width / 2)
                m = float(score[(b if method != 'SD-Flow' else s, method, sid)]['ssim_body'])
                ax.set_title(f'{b}\nSSIM {m:.2f}' if r == 0 else f'{m:.2f}', fontsize=8)
            ax = axes[r, -1]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(hl[box], cmap='gray', vmin=level - width / 2, vmax=level + width / 2)
            ax.set_title('Label' if r == 0 else '', fontsize=8)
            axes[r, 0].set_ylabel(f'{sid}\n{method}' if mi == 1 else method, fontsize=8)
    fig.suptitle(f'FBP / SWORD / SD-Flow on the shared settings (arc / row stride), body SSIM per panel; '
                 f'{args.window} window L {level} W {width} HU', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'grid.png'), dpi=100)
    print(f'wrote {os.path.join(args.out, "grid.png")}')


if __name__ == '__main__':
    main()
