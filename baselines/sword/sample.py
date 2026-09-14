"""SWORD sampling, following the official PCsampling loop.

Starting from the zero-filled measured sinogram, 27.5% of the way into a
2000-level schedule, each level runs a reverse-diffusion predictor and a
Langevin corrector (snr 0.16) with the full-frequency model, then the same with
the high-frequency model on the detail bands, writing the measured rows back
after every update. The corrector's step size is set per slice (the official
loop runs one slice at a time).

Deliberate changes, each checked with an exact oracle score (the sampler must
return the true sinogram):
  * Standard stochastic PC sampling: the start gets noise at the starting level
    and every update adds its noise. The official loop keeps each update's mean
    with no noise; with the adaptive Langevin step that drifts to 30x the noise
    level mid-schedule in the oracle test. --official_mean restores it.
  * The last level skips the corrector: the predictor has already removed the
    noise, and a Langevin step on a noise-free sample is unbounded.
  * The official code also overwrites wavelet coefficients with those of the
    full ground-truth sinogram, which carry information from unmeasured rows.
    Here consistency comes only from the measured sinogram rows.

    python -m baselines.sword.sample
"""
import argparse
import math
import os

import numpy as np
import torch
from tqdm import tqdm

from ..common.config import add_common_args, cond_indices, method_dir, setting_dir
from ..common.ct import full_fbp
from ..common.data import SplitData
from ..common.runtime import Logger, load_checkpoint, pick_device, shard
from .core import TOP, build_model, dwt, iwt, pad, score, sigma_schedule, unpad


def load_model(out, band, device):
    if not os.path.exists(os.path.join(out, f'{band}_train_done')):
        raise SystemExit(f'SWORD {band} model is not trained; run python -m baselines.sword.train --band {band}')
    ckpt = load_checkpoint(os.path.join(out, f'{band}_last.pt'))
    model = build_model(ckpt['cfg']).to(device)
    model.load_state_dict(ckpt['ema'])
    model.eval()
    return model, ckpt['cfg']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument('--split', default='test')
    parser.add_argument('--levels', type=int, default=2000, help='noise levels in the schedule (SWORD: 2000)')
    parser.add_argument('--start_frac', type=float, default=0.275, help='SWORD starts at level 550 of 2000')
    parser.add_argument('--snr', type=float, default=0.16)
    parser.add_argument('--official_mean', action='store_true',
                        help="keep each update's mean with no added noise, as the official loop does")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--no_amp', action='store_true', help='run the networks in float32')
    parser.add_argument('--shard', type=int, default=None, help='default: SLURM array task id')
    parser.add_argument('--num_shards', type=int, default=None, help='default: SLURM array size')
    parser.add_argument('--max_slices', type=int, default=0, help='only the first N slices (0 = all)')
    args = parser.parse_args()

    out = method_dir(args, 'sword')
    log = Logger(os.path.join(out, 'log_sample.txt'))
    device = pick_device(args.device)
    amp = not args.no_amp
    model_full, cfg_full = load_model(out, 'full', device)
    model_high, cfg_high = load_model(out, 'high', device)
    sig_full = sigma_schedule(cfg_full, args.levels)
    sig_high = sigma_schedule(cfg_high, args.levels)

    cond = cond_indices(args.angle_start, args.angle_end, args.angle_stride)
    rows = torch.tensor([TOP + r for r in cond], device=device)

    @torch.no_grad()
    def predictor(model, x, sig, i, gen):
        s_next = sig[i + 1] if i + 1 < len(sig) else 0.0
        g2 = sig[i] ** 2 - s_next ** 2
        x_mean = x + g2 * score(model, x, torch.full((x.shape[0],), sig[i], device=device), amp)
        if args.official_mean or i + 1 == len(sig):
            return x_mean
        return x_mean + math.sqrt(g2) * torch.randn(x.shape, generator=gen, device=device)

    @torch.no_grad()
    def corrector(model, x, sig, i, gen):
        grad = score(model, x, torch.full((x.shape[0],), sig[i], device=device), amp)
        noise = torch.randn(x.shape, generator=gen, device=device)
        ratio = noise.flatten(1).norm(dim=1) / grad.flatten(1).norm(dim=1).clamp_min(1e-12)
        step = (2 * (args.snr * ratio) ** 2).view(-1, 1, 1, 1)
        x_mean = x + step * grad
        return x_mean if args.official_mean else x_mean + (2 * step).sqrt() * noise

    data = SplitData(setting_dir(args), args.split)
    recon_dir = os.path.join(out, 'recon' if args.split == 'test' else f'recon_{args.split}')
    sino_dir = os.path.join(out, 'sino' if args.split == 'test' else f'sino_{args.split}')
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(sino_dir, exist_ok=True)
    n = min(args.max_slices, len(data)) if args.max_slices else len(data)
    mine, shard_id, num_shards = shard(range(n), args.shard, args.num_shards)
    todo = [i for i in mine if not os.path.exists(os.path.join(recon_dir, data.slices[i]['id'] + '.npy'))]
    start = int(args.start_frac * args.levels)
    log(f'sword: shard {shard_id}/{num_shards}, {len(todo)} of {len(mine)} slices to do, '
        f'levels {start}..{args.levels}, sigma_max full {cfg_full["sigma_max"]:.1f} high {cfg_high["sigma_max"]:.1f}')

    for b0 in tqdm(range(0, len(todo), args.batch_size), desc='sword'):
        idx = todo[b0:b0 + args.batch_size]
        sino = torch.from_numpy(np.stack([data.sinogram(i) for i in idx]))[:, None].to(device)
        measured = pad(sino + 1.0)[:, :, rows]
        zero_filled = torch.zeros_like(sino)
        zero_filled[:, :, cond] = sino[:, :, cond] + 1.0
        x = dwt(pad(zero_filled))
        gen = torch.Generator(device=device).manual_seed(args.seed * 1_000_003 + idx[0])
        if not args.official_mean:
            # PC sampling starts from a sample at the starting noise level
            x = x + sig_full[start] * torch.randn(x.shape, generator=gen, device=device)

        def consistent(w):
            p = iwt(w)
            p[:, :, rows] = measured
            return dwt(p)

        for i in range(start, args.levels):
            # The last predictor step removes the remaining noise; a Langevin step on a
            # noise-free sample has an unbounded step size, so the last level skips it.
            last = i + 1 == args.levels
            x = consistent(predictor(model_full, x, sig_full, i, gen))
            if not last:
                x = consistent(corrector(model_full, x, sig_full, i, gen))
            x = consistent(torch.cat([x[:, :1], predictor(model_high, x[:, 1:], sig_high, i, gen)], dim=1))
            if not last:
                x = consistent(torch.cat([x[:, :1], corrector(model_high, x[:, 1:], sig_high, i, gen)], dim=1))

        p = iwt(x)
        p[:, :, rows] = measured
        s_hat = (unpad(p) - 1.0)[:, 0].cpu().numpy()
        for k, i in enumerate(idx):
            sid = data.slices[i]['id']
            np.save(os.path.join(sino_dir, sid + '.npy'), s_hat[k].astype(np.float32))
            np.save(os.path.join(recon_dir, sid + '.npy'), full_fbp(s_hat[k]).astype(np.float32))
    log(f'sword: shard {shard_id} done')


if __name__ == '__main__':
    main()
