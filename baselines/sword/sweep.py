"""SWORD over a sweep of view settings, on the same test slices.

SWORD's two score models never see which views are measured, so the pair
trained once (--sword_dir) serves every setting; only sampling is repeated.
Settings, all arcs starting at 0 deg:

    full resolution   45, 90, 270 deg every row
    full arc          360 deg every 2nd, 4th, 8th, 10th row
    reference         45 deg every 10th row (the baselines' setting)

For each setting a small cache folder is built from --base_setting: K, the
picked test slices' labels, and the FBP of the measured views exactly as the
FBP baseline computes it (no RLS is built, so evaluate.py's figures do not
work on these folders). The checkpoints are linked in, sword/sample.py runs,
and FBP and SWORD are both scored on every slice: SSIM / PSNR inside the body
and over the image in HU, the legacy compare_ssim.py numbers, and for SWORD
the SSIM of the completed sinogram (full and unmeasured rows). The table and
one strip per slice (FBP row, SWORD row, every setting, label) go to
<out_root>/sweep/. Re-running skips slices already sampled.

    python -m baselines.sword.sweep
    python -m baselines.sword.sweep --no_sample      # only rescore and redraw
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from ..common.config import DEFAULT_CACHE_ROOT, DEFAULT_OUT_ROOT, cond_indices, setting_tag  # noqa: E402
from ..common.ct import sparse_fbp  # noqa: E402
from ..common.data import SplitData, load_offset  # noqa: E402
from ..common.metrics import METRICS, ct_scores, sino_scores, to_hu, water_level  # noqa: E402
from ..common.runtime import Logger, evenly_spaced  # noqa: E402
from ..evaluate import WINDOWS, body_box  # noqa: E402

# (angle_start, angle_end, angle_stride)
SETTINGS = [(0, 45, 1), (0, 90, 1), (0, 270, 1),
            (0, 360, 2), (0, 360, 4), (0, 360, 8), (0, 360, 10),
            (0, 45, 10)]
SINO_METRICS = [('ssim_sino', 'SSIM sinogram'), ('ssim_sino_unknown', 'SSIM unmeasured rows')]
CKPT_FILES = ('full_last.pt', 'high_last.pt', 'full_train_done', 'high_train_done')
METHODS = [('fbp', 'FBP'), ('sword', 'SWORD')]


class Setting:
    def __init__(self, start, end, stride):
        self.start, self.end, self.stride = start, end, stride
        self.tag = setting_tag(self)
        self.cond = cond_indices(start, end, stride)

    @property
    def angle_start(self):
        return self.start

    @property
    def angle_end(self):
        return self.end

    @property
    def angle_stride(self):
        return self.stride

    @property
    def label(self):
        return f'{self.end - self.start:g}°/{self.stride}'

    @property
    def sweep(self):
        return 'arc' if self.stride == 1 else 'stride'


def build_cache(setting, base, picks, offset, cache_root, log):
    """<cache_root>/<tag>/{K.npy, test/{slices.json, label.npy, fbp.npy, DONE}} for the picked
    slices; fbp is the FBP baseline's reconstruction (fbp/reconstruct.py), in the label's units."""
    sd = os.path.join(cache_root, setting.tag)
    test = os.path.join(sd, 'test')
    slices = [dict(base.slices[i]) for i in picks]
    manifest = os.path.join(test, 'slices.json')
    if os.path.exists(os.path.join(test, 'DONE')) and os.path.exists(os.path.join(test, 'fbp.npy')):
        with open(manifest) as f:
            if json.load(f) == slices:
                return sd
    os.makedirs(test, exist_ok=True)
    shutil.copyfile(os.path.join(base.dir, '..', 'K.npy'), os.path.join(sd, 'K.npy'))
    np.save(os.path.join(test, 'label.npy'), np.stack([base.image('label', i) for i in picks]))
    np.save(os.path.join(test, 'fbp.npy'),
            np.stack([sparse_fbp(base.sinogram(i) + 1.0, setting.cond) - offset for i in picks]).astype(np.float32))
    with open(manifest, 'w') as f:
        json.dump(slices, f, indent=1)
    open(os.path.join(test, 'DONE'), 'w').close()
    log(f'{setting.label}: cache for {len(picks)} slices in {sd}')
    return sd


def link_checkpoints(sword_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name in CKPT_FILES:
        src, dst = os.path.abspath(os.path.join(sword_dir, name)), os.path.join(out_dir, name)
        if not os.path.exists(src):
            raise SystemExit(f'{src} missing: train SWORD first (baselines/sword/train.sh full and high)')
        if not os.path.exists(dst):
            os.symlink(src, dst)


def reuse_reference(ref_dir, out_dir, ids):
    """The reference setting is usually sampled in full already: copy those slices."""
    n = 0
    for sub in ('recon', 'sino'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        for sid in ids:
            src, dst = os.path.join(ref_dir, sub, sid + '.npy'), os.path.join(out_dir, sub, sid + '.npy')
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
                n += sub == 'recon'
    return n


def strip(sid, i, settings, results, images, data, offset, out_root, window, out_dir):
    """Two rows, FBP and SWORD: the image under every setting, then the label, in one HU window."""
    label = data.image('label', i)
    water = water_level(label + offset)
    hu_label = to_hu(label + offset, water)
    box = body_box(hu_label)
    level, width = WINDOWS[window]
    crop_h, crop_w = hu_label[box].shape
    ncol = len(settings) + 1
    fig, axes = plt.subplots(len(METHODS), ncol, figsize=(2.6 * ncol, 2.6 * crop_h / crop_w * len(METHODS) + 0.9),
                             squeeze=False)
    for r, (key, name) in enumerate(METHODS):
        for c, s in enumerate(settings):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            img = images(s, key, sid)
            if img is None:
                ax.set_title(f'{s.label}\nnot sampled', fontsize=9, color='0.5')
                ax.set_facecolor('0.9')
                continue
            ax.imshow(to_hu(img + offset, water)[box], cmap='gray', vmin=level - width / 2, vmax=level + width / 2)
            ax.set_title(f'{s.label}\nSSIM {results[s.tag][key][sid]["ssim_body"]:.3f}', fontsize=9)
        axes[r, 0].set_ylabel(name, fontsize=10)
        axes[r, -1].imshow(hu_label[box], cmap='gray', vmin=level - width / 2, vmax=level + width / 2)
        axes[r, -1].set_title('Label', fontsize=9)
        axes[r, -1].set_xticks([])
        axes[r, -1].set_yticks([])
    fig.suptitle(f'{sid}: arc/stride, body SSIM ({window} window, L {level} W {width} HU)', fontsize=10)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(os.path.join(out_dir, f'strip_{sid}.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base_setting', default='limited0-45_stride10',
                        help='prepared setting (under --base_cache) whose test split supplies the slices')
    parser.add_argument('--base_cache', default=DEFAULT_CACHE_ROOT)
    parser.add_argument('--base_out', default=DEFAULT_OUT_ROOT, help='where the trained SWORD and its results are')
    parser.add_argument('--sword_dir', default=None, help='default: <base_out>/<base_setting>/sword')
    parser.add_argument('--cache_root', default=os.path.join(DEFAULT_CACHE_ROOT, 'sword_sweep'))
    parser.add_argument('--out_root', default=os.path.join(DEFAULT_OUT_ROOT, 'sword_sweep'))
    parser.add_argument('--n_slices', type=int, default=10, help='evenly spaced test slices')
    parser.add_argument('--window', default='lung', choices=list(WINDOWS))
    parser.add_argument('--no_sample', action='store_true', help='only score and draw what is already sampled')
    parser.add_argument('--sample_args', default='', help='extra arguments for sword/sample.py, quoted')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    sword_dir = args.sword_dir or os.path.join(args.base_out, args.base_setting, 'sword')
    report_dir = os.path.join(args.out_root, 'sweep')
    fig_dir = os.path.join(report_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    log = Logger(os.path.join(report_dir, 'log.txt'))

    base_sd = os.path.join(args.base_cache, args.base_setting)
    base = SplitData(base_sd, 'test')
    picks = evenly_spaced(len(base), args.n_slices)
    ids = [base.slices[i]['id'] for i in picks]
    offset = load_offset(base_sd)
    settings = [Setting(*s) for s in SETTINGS]
    log(f'sweep: {len(settings)} settings x {len(ids)} slices ({ids[0]} .. {ids[-1]}), checkpoints {sword_dir}')

    results = {}
    caches = {}
    for s in settings:
        sd = build_cache(s, base, picks, offset, args.cache_root, log)
        caches[s.tag] = sd
        out_dir = os.path.join(args.out_root, s.tag, 'sword')
        link_checkpoints(sword_dir, out_dir)
        if s.tag == args.base_setting:
            n = reuse_reference(os.path.join(args.base_out, args.base_setting, 'sword'), out_dir, ids)
            if n:
                log(f'{s.label}: reused {n} slices from {args.base_out}')
        if not args.no_sample:
            cmd = [sys.executable, '-m', 'baselines.sword.sample', '--angle_start', str(s.start),
                   '--angle_end', str(s.end), '--angle_stride', str(s.stride), '--cache_root', args.cache_root,
                   '--out_root', args.out_root, '--device', args.device] + args.sample_args.split()
            log(f'{s.label}: {" ".join(cmd[2:])}')
            subprocess.run(cmd, check=True)
        data = SplitData(sd, 'test')
        results[s.tag] = {'fbp': {}, 'sword': {}}
        for i in range(len(data)):
            sid = data.slices[i]['id']
            label = data.image('label', i)
            water = water_level(label + offset)
            results[s.tag]['fbp'][sid] = ct_scores(data.image('fbp', i), label, offset, water)
            recon = os.path.join(out_dir, 'recon', sid + '.npy')
            if os.path.exists(recon):
                m = ct_scores(np.load(recon), label, offset, water)
                m.update(sino_scores(np.load(os.path.join(out_dir, 'sino', sid + '.npy')), data.sinogram(i), s.cond))
                results[s.tag]['sword'][sid] = m
        log(f'{s.label}: scored FBP on {len(results[s.tag]["fbp"])} and SWORD on {len(results[s.tag]["sword"])} slices')

    keys = [k for k, _ in METRICS + SINO_METRICS]
    titles = [t for _, t in METRICS + SINO_METRICS]
    with open(os.path.join(report_dir, 'per_slice.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['setting', 'sweep', 'arc_deg', 'stride', 'n_views', 'method', 'slice_id'] + keys)
        for s in settings:
            for key, name in METHODS:
                for sid in ids:
                    m = results[s.tag][key].get(sid)
                    if m:
                        w.writerow([s.label, s.sweep, s.end - s.start, s.stride, len(s.cond), name, sid]
                                   + [m.get(k, float('nan')) for k in keys])

    def cell(k, v):
        if np.all(np.isnan(v)):
            return '-'
        return f'{np.nanmean(v):.3f} ± {np.nanstd(v):.3f}' if k.startswith('ssim') else f'{np.nanmean(v):.2f} ± {np.nanstd(v):.2f}'

    lines = [f'FBP and SWORD over view settings, {len(ids)} evenly spaced test slices of '
             f'{args.base_setting}\'s test patient',
             '', '| Setting | views | Method | ' + ' | '.join(titles) + ' |', '|---' * (len(titles) + 3) + '|']
    with open(os.path.join(report_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['setting', 'sweep', 'arc_deg', 'stride', 'n_views', 'method', 'n_slices']
                   + [f'{k}_{st}' for k in keys for st in ('mean', 'std')])
        for s in settings:
            for key, name in METHODS:
                r = results[s.tag][key]
                if not r:
                    lines.append(f'| {s.label} | {len(s.cond)} | {name} | ' + ' | '.join('-' for _ in keys) + ' |')
                    continue
                cols = {k: np.array([r[sid].get(k, float('nan')) for sid in r]) for k in keys}
                w.writerow([s.label, s.sweep, s.end - s.start, s.stride, len(s.cond), name, len(r)]
                           + [v for k in keys for v in (float(np.nanmean(cols[k])) if not np.all(np.isnan(cols[k])) else float('nan'),
                                                        float(np.nanstd(cols[k])) if not np.all(np.isnan(cols[k])) else float('nan'))])
                lines.append(f'| {s.label} | {len(s.cond)} | {name} | ' + ' | '.join(cell(k, cols[k]) for k in keys) + ' |')
    lines += ['', 'Setting = arc from 0 deg / row stride (one row = 0.5 deg). FBP = FBP of the measured views '
              '(the FBP baseline). body: inside the patient, HU: whole image, both in HU from the label; legacy: '
              'validation/compare_ssim.py; sinogram: SSIM of SWORD\'s completed sinogram against the true one, '
              'normalized by its min/max, over all rows and over the unmeasured rows only (FBP completes no sinogram).']
    with open(os.path.join(report_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    def images(s, key, sid):
        if key == 'fbp':
            d = SplitData(caches[s.tag], 'test')
            return d.image('fbp', d.ids.index(sid))
        path = os.path.join(args.out_root, s.tag, 'sword', 'recon', sid + '.npy')
        return np.load(path) if os.path.exists(path) else None

    for i, sid in zip(picks, ids):
        strip(sid, i, settings, results, images, base, offset, args.out_root, args.window, fig_dir)
    log(f'sweep: report in {report_dir}')


if __name__ == '__main__':
    main()
