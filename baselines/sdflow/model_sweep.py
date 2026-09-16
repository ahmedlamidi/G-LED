"""Every SD-Flow model on the sweep's test slices, each with the conditioning
window it was trained with, scored and drawn like the SWORD view sweep.

Models and windows come from configurations/sdflow_models.json (start, arc,
stride per model; rows wrap past 360 deg). Models whose start is null are
skipped with a note. The slices are the same 10 evenly spaced test slices
baselines/sweep.py uses (from --base_setting's test split), so the SD-Flow
grid lines up with the SWORD / FBP grids.

Outputs in <out_root>/: <model name>/{recon,sino}/, and in sweep/ the table
(summary.md, summary.csv, per_slice.csv: body / HU / legacy image metrics and
the completed sinogram's SSIM) and figures/grid_sdflow.png (rows: slices,
columns: models, label last). Re-running skips slices already sampled.

    python -m baselines.sdflow.model_sweep
    python -m baselines.sdflow.model_sweep --no_sample      # rescore and redraw only
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from ..common.config import ANGLE_STEP, DEFAULT_OUT_ROOT, SAMPLE_H, add_common_args, setting_dir
from ..common.ct import full_fbp
from ..common.data import SplitData, load_offset
from ..common.metrics import METRICS, ct_scores, sino_scores, to_hu, water_level
from ..common.runtime import Logger, evenly_spaced, pick_device
from ..evaluate import body_box, grid
from .sample import build_trainer, find_checkpoint, sdflow_modules

SINO_METRICS = [('ssim_sino', 'SSIM sinogram'), ('ssim_sino_unknown', 'SSIM unmeasured rows')]


def window_rows(start_deg, arc_deg, stride):
    first = int(round(start_deg / ANGLE_STEP))
    return [(first + r) % SAMPLE_H for r in range(0, int(round(arc_deg / ANGLE_STEP)), stride)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)   # --angle_* pick the base setting whose test split supplies the slices
    parser.add_argument('--models', default=os.path.join('configurations', 'sdflow_models.json'))
    parser.add_argument('--sweep_root', default=os.path.join(DEFAULT_OUT_ROOT, 'sdflow_sweep'),
                        help='where the per-model results and the report go')
    parser.add_argument('--n_slices', type=int, default=10, help='evenly spaced test slices')
    parser.add_argument('--num_sample_steps', type=int, default=None, help='default: the models file')
    parser.add_argument('--window', default='lung')
    parser.add_argument('--no_sample', action='store_true', help='only score and draw what is already sampled')
    args = parser.parse_args()

    with open(args.models) as f:
        spec = json.load(f)
    steps = args.num_sample_steps or spec.get('num_sample_steps', 20)
    report_dir = os.path.join(args.sweep_root, 'sweep')
    fig_dir = os.path.join(report_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    log = Logger(os.path.join(report_dir, 'log.txt'))

    sd = setting_dir(args)
    data = SplitData(sd, 'test')
    offset = load_offset(sd)
    picks = evenly_spaced(len(data), args.n_slices)
    ids = [data.slices[i]['id'] for i in picks]
    labels = {sid: data.image('label', i) for sid, i in zip(ids, picks)}
    waters = {sid: water_level(labels[sid] + offset) for sid in ids}
    sinos = {sid: data.sinogram(i) for sid, i in zip(ids, picks)}
    H, W = next(iter(sinos.values())).shape
    log(f'sdflow sweep: {len(spec["models"])} models, {len(ids)} slices ({ids[0]} .. {ids[-1]}), {steps} steps')

    device = torch.device(pick_device(args.device))
    modules = sdflow_modules() if not args.no_sample else None
    results, models = {}, []
    for m in spec['models']:
        name = m['name']
        key = name.replace('/', '_')
        if m['start'] is None:
            log(f'{name}: start unknown, skipped (see the note in {args.models})')
            continue
        models.append((key, name))
        cond = window_rows(m['start'], m['arc'], m['stride'])
        out = os.path.join(args.sweep_root, key)
        recon_dir, sino_dir = os.path.join(out, 'recon'), os.path.join(out, 'sino')
        os.makedirs(recon_dir, exist_ok=True)
        os.makedirs(sino_dir, exist_ok=True)
        with open(os.path.join(out, 'views.json'), 'w') as f:
            json.dump({'model': m, 'cond_indices': cond, 'num_sample_steps': steps}, f, indent=1)
        todo = [sid for sid in ids if not os.path.exists(os.path.join(recon_dir, sid + '.npy'))]
        if todo and not args.no_sample:
            cfg = {'bfs_dynamic_folder': m['folder'], 'sample_H': spec.get('sample_H', SAMPLE_H),
                   'detector_count': spec.get('detector_count', W), 'unet_dim': spec.get('unet_dim', 32),
                   'num_sample_steps': steps}
            ckpt = find_checkpoint(m['folder'])
            log(f'{name}: window {m["start"]:g}-{m["start"] + m["arc"]:g} deg every {m["stride"]}th row '
                f'({len(cond)} views), {len(todo)} slices to sample, checkpoint {ckpt}')
            trainer = build_trainer(modules, cfg, device)
            trainer.load(path=ckpt)
            reproject = modules[3]
            torch.manual_seed(args.seed)
            with torch.no_grad():
                for sid in todo:
                    s = sinos[sid]
                    rep = reproject(s, cond, det_count=W, num_angles=H)
                    full = torch.from_numpy(s).to(device)[None, None, None]
                    masked, mask = torch.zeros_like(full), torch.zeros_like(full)
                    masked[..., cond, :] = full[..., cond, :]
                    mask[..., cond, :] = 1.0
                    cond_images = torch.cat([masked, mask, torch.from_numpy(rep).to(device)[None, None, None]],
                                            dim=2).permute(0, 2, 1, 3, 4)
                    s_hat = trainer.sample(video_frames=1, cond_images=cond_images, batch_size=1)
                    s_hat = s_hat.detach().float().cpu().numpy().reshape(H, W).astype(np.float32)
                    np.save(os.path.join(sino_dir, sid + '.npy'), s_hat)
                    np.save(os.path.join(recon_dir, sid + '.npy'), full_fbp(s_hat).astype(np.float32))
            del trainer
            torch.cuda.empty_cache()
        results[key] = {}
        for sid in ids:
            path = os.path.join(recon_dir, sid + '.npy')
            if not os.path.exists(path):
                continue
            r = ct_scores(np.load(path), labels[sid], offset, waters[sid])
            r.update(sino_scores(np.load(os.path.join(sino_dir, sid + '.npy')), sinos[sid], cond))
            results[key][sid] = r
        log(f'{name}: scored {len(results[key])} slices')

    keys = [k for k, _ in METRICS + SINO_METRICS]
    titles = [t for _, t in METRICS + SINO_METRICS]
    with open(os.path.join(report_dir, 'per_slice.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'slice_id'] + keys)
        for key, name in models:
            for sid in ids:
                if sid in results[key]:
                    w.writerow([name, sid] + [results[key][sid][k] for k in keys])

    def cell(k, v):
        return f'{np.nanmean(v):.3f} ± {np.nanstd(v):.3f}' if k.startswith('ssim') else f'{np.nanmean(v):.2f} ± {np.nanstd(v):.2f}'

    lines = [f'SD-Flow models on {len(ids)} evenly spaced test slices, each with its trained window (arc/stride)',
             '', '| Model | views | ' + ' | '.join(titles) + ' |', '|---' * (len(titles) + 2) + '|']
    with open(os.path.join(report_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'n_views', 'n_slices'] + [f'{k}_{st}' for k in keys for st in ('mean', 'std')])
        for key, name in models:
            r = results[key]
            m = next(x for x in spec['models'] if x['name'] == name)
            n_views = len(window_rows(m['start'], m['arc'], m['stride']))
            if not r:
                lines.append(f'| {name} | {n_views} | ' + ' | '.join('-' for _ in keys) + ' |')
                continue
            cols = {k: np.array([r[sid][k] for sid in r]) for k in keys}
            w.writerow([name, n_views, len(r)] + [v for k in keys for v in (float(np.nanmean(cols[k])), float(np.nanstd(cols[k])))])
            lines.append(f'| {name} | {n_views} | ' + ' | '.join(cell(k, cols[k]) for k in keys) + ' |')
    lines += ['', 'body: inside the patient, HU: whole image, both in HU from the label; legacy: '
              'validation/compare_ssim.py; sinogram: SSIM of the completed sinogram, all rows / unmeasured rows.']
    with open(os.path.join(report_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    rows, hu = [], {}
    for sid in ids:
        hl = to_hu(labels[sid] + offset, waters[sid])
        rows.append((sid, hl, body_box(hl)))
        for key, _ in models:
            path = os.path.join(args.sweep_root, key, 'recon', sid + '.npy')
            hu[(key, sid)] = to_hu(np.load(path) + offset, waters[sid]) if os.path.exists(path) else None
    grid(rows, models, lambda key, r: hu[(key, ids[r])], lambda key, r: results[key].get(ids[r], {}).get('ssim_body'),
         args.window, 'SD-Flow, every model with its trained window (arc/stride)',
         os.path.join(fig_dir, 'grid_sdflow.png'))
    log(f'report in {report_dir}')


if __name__ == '__main__':
    main()
