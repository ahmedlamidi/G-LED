"""Train one of SWORD's two score models (--band full or --band high).

Denoising score matching for the VE SDE with continuous noise levels, the
per-sample loss summed over pixels, Adam 2e-4 with 5000 warm-up steps, gradient
clipping at 1 and EMA 0.999, as in SWORD's configs. sigma_max is set from the
data (the largest distance between training examples, the rule behind SWORD's
378) unless given. The models never see which views are measured, so one pair
serves any angle setting.

Runs in time-limited segments like dolce/train.py.

    python -m baselines.sword.train --band full
    python -m baselines.sword.train --band high
"""
import argparse
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..common.config import add_common_args, method_dir, setting_dir
from ..common.data import SplitData
from ..common.runtime import (EMA, Logger, TrainClock, autocast, load_checkpoint, parse_ints,
                              pick_device, save_checkpoint, seed_everything)
from .core import build_model, to_bands


class Sinograms(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.from_numpy(self.data.sinogram(i))[None]


@torch.no_grad()
def estimate_sigma_max(data, device, n=64):
    # Always from the full bands: both models must share one noise schedule, as
    # SWORD's two configs share sigma_max = 378, since they update the same bands.
    idx = np.random.default_rng(0).choice(len(data), size=min(n, len(data)), replace=False)
    flat = torch.stack([to_bands(torch.from_numpy(data.sinogram(i))[None, None].to(device), 'full')[0].flatten()
                        for i in idx])
    dist = float(torch.cdist(flat[None], flat[None])[0].max())
    return dist if dist > 0 else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument('--band', choices=['full', 'high'], required=True)
    parser.add_argument('--base', type=int, default=64)
    parser.add_argument('--ch_mult', default='1,1,2,2,4')
    parser.add_argument('--num_res', type=int, default=2)
    parser.add_argument('--attn_levels', default='4', help='levels with attention (4 = 24x26)')
    parser.add_argument('--sigma_min', type=float, default=0.01)
    parser.add_argument('--sigma_max', type=float, default=0, help='0 = estimate from the data')
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=5000)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--ema', type=float, default=0.999)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hours', type=float, default=22, help='total training time over all segments')
    parser.add_argument('--segment_hours', type=float, default=22.5, help='training time per SLURM job')
    parser.add_argument('--max_iters', type=int, default=0, help='stop after this many iterations (0 = no limit)')
    parser.add_argument('--ckpt_minutes', type=float, default=30)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    out = method_dir(args, 'sword')
    log = Logger(os.path.join(out, f'log_{args.band}.txt'))
    done = os.path.join(out, f'{args.band}_train_done')
    if os.path.exists(done):
        log(f'sword {args.band}: training already finished')
        return
    seed_everything(args.seed + int(time.time()) % 10000)   # new data order per segment
    device = pick_device(args.device)
    train = SplitData(setting_dir(args), 'train')

    last_path = os.path.join(out, f'{args.band}_last.pt')
    ckpt = load_checkpoint(last_path)
    if ckpt is not None:
        cfg = ckpt['cfg']
    else:
        sigma_max = args.sigma_max or estimate_sigma_max(train, device)
        cfg = {'band': args.band, 'base': args.base, 'ch_mult': parse_ints(args.ch_mult),
               'num_res': args.num_res, 'attn_levels': parse_ints(args.attn_levels),
               'sigma_min': args.sigma_min, 'sigma_max': sigma_max}

    model = build_model(cfg).to(device)
    ema = EMA(model, args.ema)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    it, hours_before = 0, 0.0
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['opt'])
        it, hours_before = ckpt['iter'], ckpt['hours']
        log(f'resumed at iteration {it} ({hours_before:.2f} h)')

    data = Sinograms(train)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=len(data) > args.batch_size,
                        persistent_workers=args.workers > 0)
    clock = TrainClock(args.hours, args.segment_hours, hours_before)
    log(f'sword {args.band}: train {len(data)} slices, '
        f'{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, cfg {cfg}')
    log_ratio = math.log(cfg['sigma_max'] / cfg['sigma_min'])

    def save():
        save_checkpoint(last_path, {'model': model.state_dict(), 'ema': ema.state_dict(),
                                    'opt': opt.state_dict(), 'iter': it, 'hours': clock.total(), 'cfg': cfg})

    last_save, losses = time.time(), []
    while True:
        for sino in loader:
            w = to_bands(sino.to(device, non_blocking=True), cfg['band'])
            b = w.shape[0]
            t = torch.rand(b, device=device) * (1 - args.eps) + args.eps
            sigma = cfg['sigma_min'] * torch.exp(t * log_ratio)
            z = torch.randn_like(w)
            with autocast(device):
                out_net = model(w + sigma.view(b, 1, 1, 1) * z, torch.log(sigma))
            # score = out / sigma, so sigma * score + z = out + z
            loss = ((out_net.float() + z) ** 2).reshape(b, -1).sum(dim=1).mean()
            for group in opt.param_groups:
                group['lr'] = args.lr * min(1.0, (it + 1) / args.warmup)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            ema.update(model)
            it += 1
            losses.append(loss.item())
            if it % args.log_every == 0:
                log(f'iter {it}: loss {np.mean(losses):.1f} ({clock.total():.2f} h)')
                losses = []

            finished = clock.run_over() or (args.max_iters and it >= args.max_iters)
            if finished or clock.segment_over() or time.time() - last_save > args.ckpt_minutes * 60:
                save()
                last_save = time.time()
            if finished:
                open(done, 'w').close()
                log(f'sword {args.band}: training finished at iteration {it}')
                return
            if clock.segment_over():
                log(f'sword {args.band}: segment over at iteration {it}; the next job resumes')
                return


if __name__ == '__main__':
    main()
