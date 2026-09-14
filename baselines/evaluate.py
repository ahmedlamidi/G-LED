"""Score every method on the same test slices against the same label, and draw
comparison figures with one display window for all panels.

Metrics follow validation/compare_ssim.py (CT part). SD-Flow runs (the contour/
folders main_diff_eval_bfs.py writes) are matched to test slices by their
ground-truth sinogram, not by folder index, and are refused if they were run
with different measured rows.

    python -m baselines.evaluate
    python -m baselines.evaluate --sdflow "SD-Flow=best_model_folder/<run>/diffusion_folder/<series>/contour"

Repeat --sdflow for several SD-Flow variants; give one NAME several folders,
comma-separated, when the test patients were evaluated separately.
"""
import argparse
import csv
import glob
import hashlib
import math
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from .common.config import add_common_args, cond_indices, method_dir, setting_dir, setting_tag  # noqa: E402
from .common.data import SplitData  # noqa: E402
from .common.metrics import ct_metrics  # noqa: E402
from .common.runtime import evenly_spaced  # noqa: E402

BASELINES = {'fbp': 'FBP', 'tv': 'TV', 'sart': 'SART', 'fbpconvnet': 'FBP-ConvNet',
             'dolce': 'DOLCE', 'sword': 'SWORD'}


def _fingerprint(sino):
    return hashlib.sha1(np.ascontiguousarray(sino, dtype=np.float32).tobytes()).hexdigest()


def import_sdflow(args, name, dirs, test, cond):
    """Turn an SD-Flow run into recon/<slice id>.npy images like the baselines'."""
    from .common.ct import full_fbp

    key = 'sdflow_' + ''.join(ch if ch.isalnum() else '_' for ch in name).strip('_').lower()
    recon_dir = os.path.join(method_dir(args, key), 'recon')
    os.makedirs(recon_dir, exist_ok=True)
    by_hash, thumbs = {}, []
    for i in range(len(test)):
        sino = test.sinogram(i)
        by_hash[_fingerprint(sino)] = i
        thumbs.append(sino[::16, ::16])
    thumbs = np.stack(thumbs)

    matched = unmatched = 0
    for folder in dirs:
        batch_dirs = sorted(glob.glob(os.path.join(folder, 'batch*')))
        if not batch_dirs:
            print(f'WARNING {name}: no batch folders in {folder}')
        for bdir in batch_dirs:
            gt = np.load(os.path.join(bdir, 'ground_truth_full.npy')).astype(np.float32)
            cond_path = os.path.join(bdir, 'cond_indices.npy')
            if os.path.exists(cond_path):
                run_cond = np.load(cond_path).tolist()
                if run_cond != cond:
                    raise SystemExit(
                        f'{name}: {bdir} was run with {len(run_cond)} measured rows starting {run_cond[:3]}, '
                        f'but the baselines use {len(cond)} rows starting {cond[:3]}. Re-run the SD-Flow '
                        f'evaluation with angle_start={args.angle_start:g}, angle_end={args.angle_end:g}, '
                        f'angle_stride={args.angle_stride}.')
            i = by_hash.get(_fingerprint(gt))
            if i is None:
                # Not bit-identical (e.g. the sinogram was made on another machine): nearest test slice
                t = gt[::16, ::16]
                err = ((thumbs - t) ** 2).mean(axis=(1, 2)) / max(float((t ** 2).mean()), 1e-12)
                i = int(err.argmin()) if err.min() < 1e-4 else None
            if i is None:
                unmatched += 1
                continue
            rec = np.load(os.path.join(bdir, 'recon_micro_0.npy'))
            rec = rec.reshape(rec.shape[-2:])[:gt.shape[0], :gt.shape[1]]
            np.save(os.path.join(recon_dir, test.slices[i]['id'] + '.npy'), full_fbp(rec).astype(np.float32))
            matched += 1
    print(f'{name}: matched {matched} SD-Flow slices to test slices, {unmatched} not in the test set')
    return key


def draw(sid, present, scores, test, index, args, out_dir):
    i = index[sid]
    label = test.image('label', i)
    lo, hi = float(label.min()), float(label.max())
    ref = (label - lo) / (hi - lo)
    vmin, vmax = np.percentile(ref, [0.5, 99.5])
    panels = []
    for key, name in present:
        path = os.path.join(args.out_root, setting_tag(args), key, 'recon', sid + '.npy')
        if os.path.exists(path):
            panels.append((name, (np.load(path) - lo) / (hi - lo), scores[key].get(sid)))
    panels.append(('Label', ref, None))

    ncol = min(4, len(panels))
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.5 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (name, img, m) in zip(axes.flat, panels):
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(name if m is None else f'{name}\n{m["psnr"]:.2f} dB, SSIM {m["ssim"]:.3f}', fontsize=10)
    s = test.slices[i]
    fig.suptitle(f'{s["patient"]}, slice {s["slice"]}: {args.angle_start:g}-{args.angle_end:g} deg, '
                 f'every {args.angle_stride}th view', fontsize=11)
    fig.tight_layout()
    stem = os.path.join(out_dir, f'compare_{sid}')
    fig.savefig(stem + '.png', dpi=200, bbox_inches='tight')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument('--methods', default=','.join(BASELINES))
    parser.add_argument('--sdflow', action='append', default=[], metavar='NAME=DIR[,DIR]',
                        help="SD-Flow contour folder(s) written by main_diff_eval_bfs.py")
    parser.add_argument('--fig_slices', default='', help='comma-separated test slice ids to draw')
    parser.add_argument('--n_fig', type=int, default=3, help='slices to draw when --fig_slices is empty')
    args = parser.parse_args()

    test = SplitData(setting_dir(args), 'test')
    cond = cond_indices(args.angle_start, args.angle_end, args.angle_stride)
    report_dir = os.path.join(args.out_root, setting_tag(args), 'evaluation')
    os.makedirs(report_dir, exist_ok=True)

    methods = [(m, BASELINES.get(m, m)) for m in args.methods.split(',') if m]
    for spec in args.sdflow:
        name, _, paths = spec.partition('=')
        if not paths:
            raise SystemExit(f'--sdflow expects NAME=DIR, got {spec!r}')
        methods.append((import_sdflow(args, name, [p for p in paths.split(',') if p], test, cond), name))

    index = {sid: i for i, sid in enumerate(test.ids)}
    scores = {}
    for key, name in methods:
        files = sorted(glob.glob(os.path.join(args.out_root, setting_tag(args), key, 'recon', '*.npy')))
        files = [f for f in files if os.path.basename(f)[:-4] in index]
        if not files:
            print(f'{name}: no test reconstructions, skipped')
            continue
        scores[key] = {}
        for f in files:
            sid = os.path.basename(f)[:-4]
            scores[key][sid] = ct_metrics(np.load(f), test.image('label', index[sid]))
    present = [(k, n) for k, n in methods if k in scores]
    if not present:
        raise SystemExit('nothing to evaluate')
    common = sorted(set.intersection(*(set(scores[k]) for k, _ in present)), key=index.get)
    if not common:
        raise SystemExit('the methods share no test slices')

    with open(os.path.join(report_dir, 'per_slice.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'slice_id', 'patient', 'slice', 'psnr', 'ssim', 'mse'])
        for key, name in present:
            for sid in sorted(scores[key], key=index.get):
                m, s = scores[key][sid], test.slices[index[sid]]
                writer.writerow([name, sid, s['patient'], s['slice'], m['psnr'], m['ssim'], m['mse']])

    rows = []
    for key, name in present:
        psnr = np.array([scores[key][sid]['psnr'] for sid in common])
        ssim = np.array([scores[key][sid]['ssim'] for sid in common])
        rows.append((name, psnr.mean(), psnr.std(), ssim.mean(), ssim.std()))
    with open(os.path.join(report_dir, 'summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std', 'n_slices'])
        for row in rows:
            writer.writerow(list(row) + [len(common)])
    lines = [f'Test slices: {len(common)} (common to every method), measured rows: '
             f'{args.angle_start:g}-{args.angle_end:g} deg, every {args.angle_stride}th view ({len(cond)} views)',
             '', '| Method | PSNR (dB) | SSIM |', '|---|---|---|']
    lines += [f'| {n} | {pm:.2f} ± {ps:.2f} | {sm:.4f} ± {ss:.4f} |' for n, pm, ps, sm, ss in rows]
    partial = [f'{n}: {len(scores[k])}' for k, n in present if len(scores[k]) != len(common)]
    if partial:
        lines += ['', 'Slices per method before intersecting: ' + ', '.join(partial)]
    with open(os.path.join(report_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    fig_dir = os.path.join(report_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_ids = [s for s in args.fig_slices.split(',') if s] or [common[j] for j in evenly_spaced(len(common), args.n_fig)]
    for sid in fig_ids:
        if sid not in index:
            print(f'unknown slice id {sid}, skipped')
            continue
        draw(sid, present, scores, test, index, args, fig_dir)
    print(f'Report: {report_dir}')


if __name__ == '__main__':
    main()
