"""Driver for the reconstruction-only baselines (FBP, SART, TV): pick the one
free parameter on the validation patient by mean SSIM against the label, then
reconstruct every test slice with it."""
import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from .config import add_common_args, cond_indices, method_dir, setting_dir
from .data import SplitData, load_offset, load_stats
from .metrics import ct_metrics
from .runtime import Logger, evenly_spaced


def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    parser.add_argument('--tune_slices', type=int, default=8,
                        help='validation slices used to pick the parameter')
    parser.add_argument('--param', type=float, default=None, help='skip tuning and use this value')
    parser.add_argument('--max_slices', type=int, default=0,
                        help='reconstruct only the first N test slices (0 = all)')
    return parser


def context(args):
    """Measured rows, the FBP(1) offset and the dataset statistics."""
    sd = setting_dir(args)
    return cond_indices(args.angle_start, args.angle_end, args.angle_stride), load_offset(sd), load_stats(sd)


def run(args, method, recon_fn, grid=None, param_name='param'):
    """recon_fn(sinogram, param) -> image in the label's units."""
    sd = setting_dir(args)
    out = method_dir(args, method)
    log = Logger(os.path.join(out, 'log.txt'))

    if args.param is not None:
        best = args.param
    elif grid and len(grid) > 1:
        val = SplitData(sd, 'val')
        picks = evenly_spaced(len(val), args.tune_slices)
        scores = []
        for value in grid:
            ssim = float(np.mean([ct_metrics(recon_fn(val.sinogram(i), value), val.image('label', i))['ssim']
                                  for i in picks]))
            scores.append((value, ssim))
            log(f'{method}: {param_name}={value:g} -> val SSIM {ssim:.4f}')
        best = max(scores, key=lambda s: s[1])[0]
        with open(os.path.join(out, 'tuning.json'), 'w') as f:
            json.dump({'param': param_name, 'val_slices': picks, 'scores': scores, 'best': best}, f, indent=2)
    else:
        best = grid[0] if grid else None
    log(f'{method}: using {param_name}={best}')

    test = SplitData(sd, 'test')
    recon_dir = os.path.join(out, 'recon')
    os.makedirs(recon_dir, exist_ok=True)
    n = min(args.max_slices, len(test)) if args.max_slices else len(test)
    for i in tqdm(range(n), desc=method):
        img = recon_fn(test.sinogram(i), best)
        np.save(os.path.join(recon_dir, test.slices[i]['id'] + '.npy'), img.astype(np.float32))
    log(f'{method}: wrote {n} test reconstructions to {recon_dir}')
