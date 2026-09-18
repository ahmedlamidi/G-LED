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

from .common.config import SAMPLE_H, add_common_args, cond_indices, method_dir, setting_dir, setting_tag  # noqa: E402
from .common.data import SplitData, load_offset  # noqa: E402
from .common.metrics import METRICS, body_mask, ct_scores, to_hu, water_level  # noqa: E402
from .common.runtime import evenly_spaced  # noqa: E402

BASELINES = {'fbp': 'FBP', 'tv': 'TV', 'sart': 'SART', 'fbpconvnet': 'FBP-ConvNet', 'dudotrans': 'DuDoTrans',
             'dolce': 'DOLCE', 'dps': 'DPS', 'sword': 'SWORD', 'sdflow': 'SD-Flow'}
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


def body_box(hu, margin=12):
    """Rows and columns around the body (the metrics' body mask)."""
    mask = body_mask(hu)
    if not mask.any():
        return slice(None), slice(None)
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return (slice(max(rows[0] - margin, 0), rows[-1] + margin + 1),
            slice(max(cols[0] - margin, 0), cols[-1] + margin + 1))


def slice_panels(sid, present, data, index, offset, args):
    """(file name, display name, HU image) for the RLS input, every method that
    has this slice, and the label, all in the HU scale of the slice's label so
    every panel is shown the same way; plus the crop around the body."""
    i = index[sid]
    label = data.image('label', i)
    water = water_level(label + offset)

    def hu(mu):
        return to_hu(mu, water)

    # RLS fits y = s + 1, so it is already in label + K units
    panels = [('rls_input', 'RLS input', hu(data.image('rls', i)))]
    for key, name in present:
        path = os.path.join(args.out_root, setting_tag(args), key, recon_name(args.split), sid + '.npy')
        if os.path.exists(path):
            panels.append((key, name, hu(np.load(path) + offset)))
    panels.append(('label', 'Label', hu(label + offset)))
    return panels, body_box(panels[-1][2])


def draw(sid, panels, box, windows, scores, out_dir):
    """Overview of one slice: one column per image, one row per window, each
    method titled with its SSIM and PSNR inside the body (in HU)."""
    crop_h, crop_w = panels[-1][2][box].shape
    fig, axes = plt.subplots(len(windows), len(panels),
                             figsize=(2.9 * len(panels), 2.9 * crop_h / crop_w * len(windows) + 1.0), squeeze=False)
    for r, w in enumerate(windows):
        level, width = WINDOWS[w]
        for c, (key, name, img) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img[box], cmap='gray', vmin=level - width / 2, vmax=level + width / 2)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                m = scores.get(key, {}).get(sid)
                ax.set_title(name if m is None else f'{name}\nSSIM {m["ssim_body"]:.3f}, {m["psnr_body"]:.2f} dB',
                             fontsize=10)
        axes[r, 0].set_ylabel(f'{w}\nL {level}, W {width} HU', fontsize=10)
    fig.suptitle(f'{sid}: SSIM / PSNR inside the body, in HU (per_slice.csv has every metric)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'compare_{sid}.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)


def grid(rows, columns, image, score, window, title, path):
    """The standard overview: one row per slice, one column per method or
    setting, the label last; every panel cropped to the body and shown in one
    HU window; column titles carry the body SSIM.

    rows: list of (row label, label image in HU, body crop); columns: list of
    (column key, column title); image(key, r) -> HU image or None; score(key, r) -> SSIM or None.
    """
    level, width = WINDOWS[window]
    fig, axes = plt.subplots(len(rows), len(columns) + 1,
                             figsize=(2.0 * (len(columns) + 1), 1.75 * len(rows) + 0.7), squeeze=False)
    for r, (name, hu_label, box) in enumerate(rows):
        for c, (key, ctitle) in enumerate(columns):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            img = image(key, r)
            if img is None:
                ax.set_facecolor('0.9')
                ax.set_title(f'{ctitle}\nnot done' if r == 0 else 'not done', fontsize=8, color='0.5')
                continue
            ax.imshow(img[box], cmap='gray', vmin=level - width / 2, vmax=level + width / 2)
            m = score(key, r)
            ax.set_title((f'{ctitle}\nSSIM {m:.2f}' if r == 0 else f'{m:.2f}') if m is not None else
                         (ctitle if r == 0 else ''), fontsize=8)
        ax = axes[r, -1]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(hu_label[box], cmap='gray', vmin=level - width / 2, vmax=level + width / 2)
        ax.set_title('Label' if r == 0 else '', fontsize=8)
        axes[r, 0].set_ylabel(name, fontsize=8)
    fig.suptitle(f'{title}\nbody SSIM per panel; {window} window L {level} W {width} HU', fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def export_slice(sid, panels, box, windows, out_dir):
    """One PDF per image and window, just the cropped image: <name>_<window>.pdf."""
    os.makedirs(out_dir, exist_ok=True)
    crop_h, crop_w = panels[-1][2][box].shape
    for w in windows:
        level, width = WINDOWS[w]
        for key, _, img in panels:
            fig = plt.figure(figsize=(4.0, 4.0 * crop_h / crop_w))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img[box], cmap='gray', vmin=level - width / 2, vmax=level + width / 2)
            ax.axis('off')
            fig.savefig(os.path.join(out_dir, f'{key}_{w}.pdf'))
            plt.close(fig)
    with open(os.path.join(out_dir, 'windows.txt'), 'w') as f:
        f.write(''.join(f'{w}: level {WINDOWS[w][0]} HU, width {WINDOWS[w][1]} HU '
                        f'(HU estimated per slice from the label)\n' for w in windows))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument('--methods', default=','.join(BASELINES))
    parser.add_argument('--sdflow', action='append', default=[], metavar='NAME=DIR[,DIR]',
                        help="SD-Flow contour folder(s) written by main_diff_eval_bfs.py")
    parser.add_argument('--split', default='test',
                        help='test, or val to check reconstructions made with the samplers\' --split val')
    parser.add_argument('--fig_slices', default='', help='comma-separated slice ids to preview')
    parser.add_argument('--n_fig', type=int, default=0,
                        help='preview this many evenly spaced slices when --fig_slices is empty (0 = every shared slice)')
    parser.add_argument('--export_slice', default='',
                        help='instead of previews, write one PDF per method and window for this slice id')
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
    offset = load_offset(setting_dir(args))
    waters = {}   # HU calibration of each slice, from its label only

    def score(pred, sid):
        label = test.image('label', index[sid])
        if sid not in waters:
            waters[sid] = water_level(label + offset)
        return ct_scores(pred, label, offset, waters[sid])

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
            scores[key][sid] = score(np.load(f), sid)
    present = [(k, n) for k, n in methods if k in scores]
    if not present:
        raise SystemExit('nothing to evaluate')
    common = sorted(set.intersection(*(set(scores[k]) for k, _ in present)), key=index.get)
    if not common:
        raise SystemExit(f'the methods share no {args.split} slices')

    keys = [k for k, _ in METRICS]
    with open(os.path.join(report_dir, 'per_slice.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'slice_id', 'patient', 'slice'] + keys)
        for key, name in present:
            for sid in sorted(scores[key], key=index.get):
                m, s = scores[key][sid], test.slices[index[sid]]
                writer.writerow([name, sid, s['patient'], s['slice']] + [m[k] for k in keys])

    stats = {key: {k: (float(np.mean([scores[key][sid][k] for sid in common])),
                       float(np.std([scores[key][sid][k] for sid in common]))) for k in keys}
             for key, _ in present}
    with open(os.path.join(report_dir, 'summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method'] + [f'{k}_{s}' for k in keys for s in ('mean', 'std')] + ['n_slices'])
        for key, name in present:
            writer.writerow([name] + [v for k in keys for v in stats[key][k]] + [len(common)])

    def cell(k, mean, std):
        return f'{mean:.3f} ± {std:.3f}' if k.startswith('ssim') else f'{mean:.2f} ± {std:.2f}'

    lines = [f'{args.split.capitalize()} slices: {len(common)} (common to every method), measured rows: '
             f'{args.angle_start:g}-{args.angle_end:g} deg, every {args.angle_stride}th view ({len(cond)} views)',
             '', '| Method | ' + ' | '.join(t for _, t in METRICS) + ' |', '|---' * (len(METRICS) + 1) + '|']
    lines += [f'| {name} | ' + ' | '.join(cell(k, *stats[key][k]) for k in keys) + ' |' for key, name in present]
    lines += ['', "body: inside the patient (label above -500 HU); HU: the whole image. Both in HU from each slice's "
              'label (air 0, soft-tissue peak 0 HU), clipped to [-1000, 1000] HU. legacy: '
              'validation/compare_ssim.py (label min/max, whole image), which rates every method high.']
    partial = [f'{n}: {len(scores[k])}' for k, n in present if len(scores[k]) != len(common)]
    if partial:
        lines += ['', 'Slices per method before intersecting: ' + ', '.join(partial)]
    if view_notes:
        lines += ['', f'Measured views: baselines {describe_views(cond)}; ' + '; '.join(view_notes)]
    with open(os.path.join(report_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    windows = [w for w in args.windows.split(',') if w]
    if args.export_slice:
        sid = args.export_slice
        if sid not in index:
            raise SystemExit(f'unknown slice id {sid}')
        panels, box = slice_panels(sid, present, test, index, offset, args)
        out = os.path.join(report_dir, f'slice_{sid}')
        export_slice(sid, panels, box, windows, out)
        print(f'{sid}: {len(panels) * len(windows)} PDFs ({", ".join(k for k, _, _ in panels)}) in {out}')
        return

    fig_dir = os.path.join(report_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_ids = [s for s in args.fig_slices.split(',') if s] or (
        [common[j] for j in evenly_spaced(len(common), args.n_fig)] if args.n_fig else common)
    names = ['RLS input'] + [n for _, n in present] + ['Label']
    with open(os.path.join(fig_dir, 'columns.txt'), 'w') as f:
        f.write('columns, left to right: ' + ', '.join(names) + '\n'
                + 'rows: ' + ', '.join(f'{w} (L {WINDOWS[w][0]}, W {WINDOWS[w][1]} HU)' for w in windows) + '\n')
    for n, sid in enumerate(fig_ids, 1):
        if sid not in index:
            print(f'unknown slice id {sid}, skipped')
            continue
        panels, box = slice_panels(sid, present, test, index, offset, args)
        draw(sid, panels, box, windows, scores, fig_dir)
        if n % 25 == 0 or n == len(fig_ids):
            print(f'previews: {n}/{len(fig_ids)}')

    # the overview grid: --n_fig slices (default 10) x methods, label last
    grid_ids = [s for s in args.fig_slices.split(',') if s and s in index] or \
        [common[j] for j in evenly_spaced(len(common), args.n_fig or 10)]
    rows, hu = [], {}
    for sid in grid_ids:
        i = index[sid]
        label = test.image('label', i)
        hu_label = to_hu(label + offset, waters[sid])
        rows.append((sid, hu_label, body_box(hu_label)))
        for key, _ in present:
            path = os.path.join(args.out_root, setting_tag(args), key, recon_name(args.split), sid + '.npy')
            hu[(key, sid)] = to_hu(np.load(path) + offset, waters[sid]) if os.path.exists(path) else None
    grid(rows, present, lambda key, r: hu[(key, grid_ids[r])],
         lambda key, r: scores[key].get(grid_ids[r], {}).get('ssim_body'), windows[0],
         f'{args.split} slices, every method', os.path.join(fig_dir, 'overview.png'))
    print(f'Report: {report_dir}')


if __name__ == '__main__':
    main()
