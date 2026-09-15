"""Score every method on the same test slices against the same label, and draw
comparison figures with the same HU display windows for all panels.

Metrics follow validation/compare_ssim.py (CT part). SD-Flow runs (the contour/
folders main_diff_eval_bfs.py writes) are matched to test slices by their
ground-truth sinogram, not by folder index. They may use other measured views
than the baselines; summary.md lists each run's views.

    python -m baselines.evaluate
    python -m baselines.evaluate --sdflow "SD-Flow=best_model_folder/<run>/diffusion_folder/<series>/contour"

Repeat --sdflow for several SD-Flow variants; give one NAME several folders,
comma-separated, when the test patients were evaluated separately.
"""
import argparse
import csv
import glob
import hashlib
import json
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from skimage.filters import threshold_otsu  # noqa: E402

from .common.config import SAMPLE_H, add_common_args, cond_indices, method_dir, setting_dir, setting_tag  # noqa: E402
from .common.data import SplitData, load_offset  # noqa: E402
from .common.metrics import ct_metrics  # noqa: E402
from .common.runtime import evenly_spaced  # noqa: E402

BASELINES = {'fbp': 'FBP', 'tv': 'TV', 'sart': 'SART', 'fbpconvnet': 'FBP-ConvNet',
             'dolce': 'DOLCE', 'sword': 'SWORD', 'sdflow': 'SD-Flow'}
# Display windows as (level, width) in HU
WINDOWS = {'lung': (-600, 1500), 'soft': (40, 400), 'bone': (400, 1800)}


def recon_name(split):
    """Folder the samplers write a split's reconstructions to."""
    return 'recon' if split == 'test' else f'recon_{split}'


def _fingerprint(sino):
    return hashlib.sha1(np.ascontiguousarray(sino, dtype=np.float32).tobytes()).hexdigest()


def describe_views(rows):
    """'9 views, 0-40 deg every 5 deg' for a list of sinogram rows."""
    step = 360 / SAMPLE_H
    rows = list(rows)
    if len(rows) > 1 and len(set(np.diff(rows))) == 1:
        return f'{len(rows)} views, {rows[0] * step:g}-{rows[-1] * step:g} deg every {(rows[1] - rows[0]) * step:g} deg'
    return f'{len(rows)} views (rows {rows})'


def import_sdflow(args, name, dirs, test):
    """Turn an SD-Flow run into recon/<slice id>.npy images like the baselines'.
    Returns the method key and the set of measured-row lists the run used."""
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
    views = set()
    for folder in dirs:
        batch_dirs = sorted(glob.glob(os.path.join(folder, 'batch*')))
        if not batch_dirs:
            print(f'WARNING {name}: no batch folders in {folder}')
        for bdir in batch_dirs:
            gt = np.load(os.path.join(bdir, 'ground_truth_full.npy')).astype(np.float32)
            cond_path = os.path.join(bdir, 'cond_indices.npy')
            if os.path.exists(cond_path):
                views.add(tuple(np.load(cond_path).tolist()))
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
    print(f'{name}: matched {matched} SD-Flow slices to test slices, {unmatched} not in the test set; '
          + ('; '.join(describe_views(v) for v in sorted(views)) if views else 'no cond_indices.npy found'))
    return key, views


def water_level(mu):
    """Soft-tissue level of an image proportional to mu with air at 0 (label + K).

    Each sinogram is rescaled on its own, so the images have no fixed HU scale.
    The most common value above the Otsu air/body threshold inside the field of
    view is taken as water (0 HU). Display only; the metrics never use it.
    """
    n = mu.shape[0]
    yy, xx = np.mgrid[:n, :n]
    fov = mu[np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2) < 0.45 * n]
    body = fov[fov > threshold_otsu(fov)]
    hist, edges = np.histogram(body, bins=200)
    k = int(np.convolve(hist, np.ones(5) / 5, mode='same').argmax())
    return 0.5 * (edges[k] + edges[k + 1])


def body_box(hu, margin=12):
    """Rows and columns around the body (above -500 HU) inside the field of view."""
    n = hu.shape[0]
    yy, xx = np.mgrid[:n, :n]
    mask = (hu > -500) & (np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2) < 0.47 * n)
    if not mask.any():
        return slice(None), slice(None)
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return (slice(max(rows[0] - margin, 0), rows[-1] + margin + 1),
            slice(max(cols[0] - margin, 0), cols[-1] + margin + 1))


def draw(sid, present, scores, data, index, offset, args, out_dir):
    """One column per image, one row per display window, all in the HU scale of
    the slice's label so every panel is shown the same way."""
    i = index[sid]
    label = data.image('label', i)
    water = water_level(label + offset)

    def hu(mu):
        return 1000 * (mu / water - 1)

    # RLS fits y = s + 1, so it is already in label + K units
    panels = [('RLS input (DOLCE\'s condition)', hu(data.image('rls', i)), None)]
    for key, name in present:
        path = os.path.join(args.out_root, setting_tag(args), key, recon_name(args.split), sid + '.npy')
        if os.path.exists(path):
            panels.append((name, hu(np.load(path) + offset), scores[key].get(sid)))
    panels.append(('Label', hu(label + offset), None))
    box = body_box(panels[-1][1])
    crop_h, crop_w = panels[-1][1][box].shape

    windows = [w for w in args.windows.split(',') if w]
    ncol, nrow = len(panels), len(windows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.9 * crop_h / crop_w * nrow + 1.0), squeeze=False)
    for r, w in enumerate(windows):
        level, width = WINDOWS[w]
        for c, (name, img, m) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img[box], cmap='gray', vmin=level - width / 2, vmax=level + width / 2)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(name if m is None else f'{name}\n{m["psnr"]:.2f} dB, SSIM {m["ssim"]:.3f}',
                             fontsize=10)
        axes[r, 0].set_ylabel(f'{w}\nL {level}, W {width} HU', fontsize=10)
    s = data.slices[i]
    fig.suptitle(f'{s["patient"]}, slice {s["slice"]}: {args.angle_start:g}-{args.angle_end:g} deg, '
                 f'every {args.angle_stride}th view (HU estimated per slice from the label)', fontsize=11)
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
    parser.add_argument('--split', default='test',
                        help='test, or val to check reconstructions made with the samplers\' --split val')
    parser.add_argument('--fig_slices', default='', help='comma-separated slice ids to draw')
    parser.add_argument('--n_fig', type=int, default=3, help='slices to draw when --fig_slices is empty')
    parser.add_argument('--windows', default='lung,soft',
                        help=f'display windows, one figure row each: {", ".join(WINDOWS)}')
    args = parser.parse_args()
    unknown = [w for w in args.windows.split(',') if w and w not in WINDOWS]
    if unknown:
        raise SystemExit(f'unknown window(s) {unknown}; choose from {list(WINDOWS)}')
    if args.sdflow and args.split != 'test':
        raise SystemExit('--sdflow runs are matched to test slices; use --split test')

    test = SplitData(setting_dir(args), args.split)
    cond = cond_indices(args.angle_start, args.angle_end, args.angle_stride)
    report_dir = os.path.join(args.out_root, setting_tag(args),
                              'evaluation' if args.split == 'test' else f'evaluation_{args.split}')
    os.makedirs(report_dir, exist_ok=True)

    methods = [(m, BASELINES.get(m, m)) for m in args.methods.split(',') if m]
    view_notes = []
    for spec in args.sdflow:
        name, _, paths = spec.partition('=')
        if not paths:
            raise SystemExit(f'--sdflow expects NAME=DIR, got {spec!r}')
        key, views = import_sdflow(args, name, [p for p in paths.split(',') if p], test)
        methods.append((key, name))
        if views and views != {tuple(cond)}:
            view_notes.append(f'{name}: ' + '; '.join(describe_views(v) for v in sorted(views)))

    index = {sid: i for i, sid in enumerate(test.ids)}
    scores = {}
    for key, name in methods:
        files = sorted(glob.glob(os.path.join(args.out_root, setting_tag(args), key, recon_name(args.split), '*.npy')))
        files = [f for f in files if os.path.basename(f)[:-4] in index]
        if not files:
            print(f'{name}: no {args.split} reconstructions, skipped')
            continue
        scores[key] = {}
        views_path = os.path.join(args.out_root, setting_tag(args), key, 'views.json')   # written by sdflow/sample.py
        if os.path.exists(views_path):
            with open(views_path) as f:
                rows = json.load(f)['cond_indices']
            if rows != cond:
                view_notes.append(f'{name}: {describe_views(rows)}')
        for f in files:
            sid = os.path.basename(f)[:-4]
            scores[key][sid] = ct_metrics(np.load(f), test.image('label', index[sid]))
    present = [(k, n) for k, n in methods if k in scores]
    if not present:
        raise SystemExit('nothing to evaluate')
    common = sorted(set.intersection(*(set(scores[k]) for k, _ in present)), key=index.get)
    if not common:
        raise SystemExit(f'the methods share no {args.split} slices')

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
    lines = [f'{args.split.capitalize()} slices: {len(common)} (common to every method), measured rows: '
             f'{args.angle_start:g}-{args.angle_end:g} deg, every {args.angle_stride}th view ({len(cond)} views)',
             '', '| Method | PSNR (dB) | SSIM |', '|---|---|---|']
    lines += [f'| {n} | {pm:.2f} ± {ps:.2f} | {sm:.4f} ± {ss:.4f} |' for n, pm, ps, sm, ss in rows]
    partial = [f'{n}: {len(scores[k])}' for k, n in present if len(scores[k]) != len(common)]
    if partial:
        lines += ['', 'Slices per method before intersecting: ' + ', '.join(partial)]
    if view_notes:
        lines += ['', f'Measured views: baselines {describe_views(cond)}; ' + '; '.join(view_notes)]
    with open(os.path.join(report_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    fig_dir = os.path.join(report_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_ids = [s for s in args.fig_slices.split(',') if s] or [common[j] for j in evenly_spaced(len(common), args.n_fig)]
    offset = load_offset(setting_dir(args))
    for sid in fig_ids:
        if sid not in index:
            print(f'unknown slice id {sid}, skipped')
            continue
        draw(sid, present, scores, test, index, offset, args, fig_dir)
    print(f'Report: {report_dir}')


if __name__ == '__main__':
    main()
