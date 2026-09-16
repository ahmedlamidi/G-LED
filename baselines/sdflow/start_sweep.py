"""Which conditioning window was an SD-Flow model trained with? Slide the
window's start angle around the circle, sample with each, and plot the score.

For every start angle the window covers [start, start + arc) with the config's
row stride (the arc and stride come from the config; only the start moves, and
rows wrap past 360 deg). The model is sampled on a few slices with that window
and scored against the label: SSIM inside the body in HU (the sweep's metric),
the legacy compare_ssim.py SSIM, and the SSIM of the completed sinogram. The
trained window should give the clear peak.

The trained window is also written in <model>/diffusion_folder/logging/args.txt
(angle_start, angle_end, angle_stride) when that file exists; this script prints
it next to the peak so the two can be compared.

Models are given as config JSONs (--config) or as model folders with their
arc and row stride, --model FOLDER:ARC:STRIDE (the saved checkpoint holds only
weights; if the folder has a diffusion_folder/logging/args.txt its values are
used and ARC:STRIDE may be left out).
For a full 360 deg arc only the row phase matters, so those models are swept
over their `stride` possible phases instead of 0..360 deg.

    python -m baselines.sdflow.start_sweep --config configurations/Limited45Sparse10.json
    python -m baselines.sdflow.start_sweep --model best_model_folder/LimitedView90:90:1 \\
        --model best_model_folder/Limited_view_45:45:1 --step 5 --split val --n_slices 4
"""
import argparse
import csv
import json
import os

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from ..common.config import ANGLE_STEP, SAMPLE_H, add_common_args, setting_dir  # noqa: E402
from ..common.ct import full_fbp  # noqa: E402
from ..common.data import SplitData, load_offset  # noqa: E402
from ..common.metrics import ct_scores, sino_scores, water_level  # noqa: E402
from ..common.runtime import Logger, evenly_spaced, pick_device  # noqa: E402
from .sample import build_trainer, find_checkpoint, sdflow_modules  # noqa: E402


def window_rows(start_deg, arc_deg, stride):
    """Measured rows for a window starting at start_deg, wrapping past 360 deg."""
    first = int(round(start_deg / ANGLE_STEP))
    n_rows = int(round(arc_deg / ANGLE_STEP))
    return [(first + r) % SAMPLE_H for r in range(0, n_rows, stride)]


def train_args(model_dir):
    path = os.path.join(model_dir, 'diffusion_folder', 'logging', 'args.txt')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def trained_window(model_dir):
    a = train_args(model_dir)
    if all(k in a for k in ('angle_start', 'angle_end', 'angle_stride')):
        return a['angle_start'], a['angle_end'], a['angle_stride']
    return None


def load_model_specs(configs, models, num_sample_steps):
    """(name, cfg dict as sdflow/sample.py expects) for every --config and --model."""
    specs = []
    for config in configs:
        with open(config) as f:
            cfg = json.load(f)
        specs.append((os.path.splitext(os.path.basename(config))[0], cfg))
    for spec in models:
        folder, *rest = spec.split(':')
        a = train_args(folder)
        if len(rest) == 2:
            start, arc, stride = 0.0, float(rest[0]), int(rest[1])
        elif all(k in a for k in ('angle_start', 'angle_end', 'angle_stride')):
            start, arc, stride = a['angle_start'], a['angle_end'] - a['angle_start'], a['angle_stride']
        else:
            raise SystemExit(f'{folder}: no args.txt with the angles; give the arc and stride as FOLDER:ARC:STRIDE')
        specs.append((os.path.basename(os.path.normpath(folder)),
                      {'bfs_dynamic_folder': folder, 'angle_start': start, 'angle_end': start + arc,
                       'angle_stride': stride, 'sample_H': a.get('sample_H', SAMPLE_H),
                       'detector_count': a.get('detector_count', 816), 'unet_dim': a.get('unet_dim', 32),
                       'num_sample_steps': a.get('num_sample_steps', 20)}))
    for _, cfg in specs:
        cfg.setdefault('num_sample_steps', 20)
        cfg.setdefault('unet_dim', 32)
        if num_sample_steps:
            cfg['num_sample_steps'] = num_sample_steps
    return specs


def draw(curves, args, n_slices):
    """The plot from whatever has been scored so far (called after every start angle)."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for name, (rows_out, known) in curves.items():
        if not rows_out:
            continue
        best = max(rows_out, key=lambda r: r['ssim_body'])
        x = [r['start_deg'] for r in rows_out]
        for ax, key in zip(axes, ('ssim_body', 'ssim_legacy', 'ssim_sino')):
            line, = ax.plot(x, [r[key] for r in rows_out], '-o', ms=3, label=f'{name} (peak {best["start_deg"]:g}°)')
            ax.axvline(best['start_deg'], color=line.get_color(), ls='--', lw=0.8)
            if known:
                ax.axvline(known[0], color=line.get_color(), ls=':', lw=1.2)
    for ax, title in zip(axes, ('SSIM inside the body (HU)', 'legacy SSIM (compare_ssim.py)', 'SSIM of the completed sinogram')):
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('start of the conditioning window (deg); dashed = best start, dotted = args.txt')
    axes[-1].set_xticks(np.arange(0, 361, 30))
    axes[0].legend(fontsize=8)
    fig.suptitle(f'SD-Flow: score vs start angle of the conditioning window ({args.split} split, {n_slices} slices)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'start_sweep.png'), dpi=130)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)   # --angle_* only pick the baselines' cache folder for slices and labels
    parser.add_argument('--config', action='append', default=[], help='SD-Flow config JSON; repeat for several models')
    parser.add_argument('--model', action='append', default=[],
                        help='SD-Flow model folder as FOLDER:ARC_DEG:STRIDE, e.g. best_model_folder/LimitedView90:90:1')
    parser.add_argument('--step', type=float, default=5, help='start angles every STEP degrees')
    parser.add_argument('--split', default='val', help='val (default) or test')
    parser.add_argument('--n_slices', type=int, default=4, help='evenly spaced slices of the split')
    parser.add_argument('--num_sample_steps', type=int, default=20,
                        help='diffusion sampling steps for every model (0 = each model\'s own config / args.txt)')
    parser.add_argument('--out', default=os.path.join('output', 'baselines', 'sdflow_start_sweep'))
    args = parser.parse_args()
    if not args.config and not args.model:
        raise SystemExit('give at least one --config or --model')

    os.makedirs(args.out, exist_ok=True)
    log = Logger(os.path.join(args.out, 'log.txt'))
    device = torch.device(pick_device(args.device))
    sd = setting_dir(args)
    data = SplitData(sd, args.split)
    offset = load_offset(sd)
    picks = evenly_spaced(len(data), args.n_slices)
    modules = sdflow_modules()
    reproject = modules[3]
    labels = {i: data.image('label', i) for i in picks}
    waters = {i: water_level(labels[i] + offset) for i in picks}
    sinos = {i: data.sinogram(i) for i in picks}
    H, W = next(iter(sinos.values())).shape

    curves = {}
    for name, cfg in load_model_specs(args.config, args.model, args.num_sample_steps):
        arc, stride = cfg['angle_end'] - cfg['angle_start'], cfg['angle_stride']
        if arc >= 360:      # full circle: only the row phase can change
            starts = np.arange(stride) * ANGLE_STEP
        else:
            starts = np.arange(0, 360, args.step)
        ckpt = find_checkpoint(cfg['bfs_dynamic_folder'])
        known = trained_window(cfg['bfs_dynamic_folder'])
        log(f'{name}: arc {arc:g} deg, stride {stride}, {cfg["num_sample_steps"]} steps, checkpoint {ckpt}; '
            f'config says start {cfg["angle_start"]:g}; args.txt says {known}')
        csv_path = os.path.join(args.out, f'{name}.csv')
        fields = ['model', 'start_deg', 'arc_deg', 'stride', 'n_views', 'ssim_body', 'ssim_legacy', 'ssim_sino']
        rows_out = []
        if os.path.exists(csv_path):     # resume: keep starts already scored with the same arc and stride
            with open(csv_path) as f:
                rows_out = [{k: (v if k == 'model' else float(v)) for k, v in r.items()} for r in csv.DictReader(f)
                            if float(r['arc_deg']) == arc and int(float(r['stride'])) == stride]
            if rows_out:
                log(f'{name}: {len(rows_out)} start angles already in {csv_path}, continuing')
        done_starts = {r['start_deg'] for r in rows_out}
        curves[name] = (rows_out, known)
        todo = [float(st) for st in starts if float(st) not in done_starts]
        trainer = build_trainer(modules, cfg, device) if todo else None
        if trainer is not None:
            trainer.load(path=ckpt)
        torch.manual_seed(args.seed)

        with torch.no_grad():
            for start in todo:
                cond = window_rows(float(start), arc, stride)
                per = {'ssim_body': [], 'ssim_legacy': [], 'ssim_sino': []}
                for i in picks:
                    s = sinos[i]
                    rep = reproject(s, cond, det_count=W, num_angles=H)
                    full = torch.from_numpy(s).to(device)[None, None, None]
                    masked, mask = torch.zeros_like(full), torch.zeros_like(full)
                    masked[..., cond, :] = full[..., cond, :]
                    mask[..., cond, :] = 1.0
                    cond_images = torch.cat([masked, mask, torch.from_numpy(rep).to(device)[None, None, None]],
                                            dim=2).permute(0, 2, 1, 3, 4)
                    out = trainer.sample(video_frames=1, cond_images=cond_images, batch_size=1)
                    s_hat = out.detach().float().cpu().numpy().reshape(H, W)
                    m = ct_scores(full_fbp(s_hat), labels[i], offset, waters[i])
                    per['ssim_body'].append(m['ssim_body'])
                    per['ssim_legacy'].append(m['ssim_legacy'])
                    per['ssim_sino'].append(sino_scores(s_hat, s, cond)['ssim_sino'])
                row = {'model': name, 'start_deg': float(start), 'arc_deg': arc, 'stride': stride, 'n_views': len(cond)}
                row.update({k: float(np.mean(v)) for k, v in per.items()})
                rows_out.append(row)
                rows_out.sort(key=lambda r: r['start_deg'])
                with open(csv_path, 'w', newline='') as f:       # the CSV and the plot are always current
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(rows_out)
                draw(curves, args, len(picks))
                log(f'{name}: start {start:6.1f} deg -> body SSIM {row["ssim_body"]:.3f}, legacy {row["ssim_legacy"]:.3f}, '
                    f'sino {row["ssim_sino"]:.3f}  ({len(rows_out)}/{len(starts)} starts)')
        best = max(rows_out, key=lambda r: r['ssim_body'])
        log(f'{name}: best start {best["start_deg"]:g} deg (body SSIM {best["ssim_body"]:.3f}); '
            f'window {best["start_deg"]:g}-{best["start_deg"] + arc:g} deg every {stride}th row'
            + (f'; args.txt: {known[0]:g}-{known[1]:g} stride {known[2]}' if known else ''))
        del trainer
        torch.cuda.empty_cache()

    draw(curves, args, len(picks))
    log(f'wrote {os.path.join(args.out, "start_sweep.png")}')


if __name__ == '__main__':
    main()
