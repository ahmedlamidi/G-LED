"""Train DOLCE's conditional diffusion model.

eps-prediction on the label (one global normalization from the training
patients), conditioned on the RLS image of the measured rows, with the
condition dropped 20% of the time (paper Sec. 4.1). As in the paper: T = 2000
linear schedule, Adam at a fixed 1.5e-4, dropout 0.2, and the latest (EMA)
checkpoint is used with no checkpoint selection. The batch is 8 rather than 256
and the network is scaled to train in about a day.

Runs in time-limited segments: each job stops after --segment_hours and the
next one resumes, until --hours of training in total.

    python -m baselines.dolce.train
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..common.config import add_common_args, cond_indices, method_dir, setting_dir
from ..common.data import Normalizer, SplitData, load_stats
from ..common.runtime import (EMA, Logger, TrainClock, TrainRecord, autocast, load_checkpoint, parse_ints,
                              pick_device, save_checkpoint, seed_everything)
from .model import alphas_cumprod, build_model, condition


class DolceData(Dataset):
    def __init__(self, data, norm, use_cond=True):
        self.data, self.norm, self.use_cond = data, norm, use_cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x0 = np.clip(self.norm.to_unit(self.data.image('label', i)), -1, 1).astype(np.float32)
        # an unconditional prior (p_uncond 1, DPS) never sees the condition, so it needs no RLS array
        c = condition(self.data.image('rls', i)) if self.use_cond else np.zeros_like(x0)
        return torch.from_numpy(x0)[None], torch.from_numpy(c)[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument('--base', type=int, default=64)
    parser.add_argument('--ch_mult', default='1,1,2,2,4,4')
    parser.add_argument('--num_res', type=int, default=2)
    parser.add_argument('--attn_levels', default='4,5', help='levels with attention (4, 5 = 32x32, 16x16)')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--T', type=int, default=2000)
    parser.add_argument('--p_uncond', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ema', type=float, default=0.9995)
    parser.add_argument('--hours', type=float, default=22, help='total training time over all segments')
    parser.add_argument('--segment_hours', type=float, default=22.5, help='training time per SLURM job')
    parser.add_argument('--max_iters', type=int, default=0, help='stop after this many iterations (0 = no limit)')
    parser.add_argument('--plateau_hours', type=float, default=3,
                        help='stop when the mean loss of the last N hours improved on the N hours before by '
                             'less than --plateau_tol (0 = never)')
    parser.add_argument('--plateau_tol', type=float, default=0.05)
    parser.add_argument('--min_hours', type=float, default=6, help='no plateau stop before this')
    parser.add_argument('--ckpt_minutes', type=float, default=30)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--name', default='dolce',
                        help="output folder under the setting; e.g. dps_prior with --p_uncond 1 for DPS's prior")
    args = parser.parse_args()

    out = method_dir(args, args.name)
    log = Logger(os.path.join(out, 'log.txt'))
    done = os.path.join(out, 'train_done')
    if os.path.exists(done):
        log('dolce: training already finished')
        return
    seed_everything(args.seed + int(time.time()) % 10000)   # new data order per segment
    device = pick_device(args.device)
    sd = setting_dir(args)

    last_path = os.path.join(out, 'last.pt')
    ckpt = load_checkpoint(last_path)
    if ckpt is not None:
        cfg, norm = ckpt['cfg'], Normalizer(*ckpt['label_norm'])
    else:
        stats = load_stats(sd)
        cfg = {'base': args.base, 'ch_mult': parse_ints(args.ch_mult), 'num_res': args.num_res,
               'attn_levels': parse_ints(args.attn_levels), 'dropout': args.dropout, 'T': args.T}
        norm = Normalizer(stats['label_lo'], stats['label_hi'])

    model = build_model(cfg).to(device)
    ema = EMA(model, args.ema)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    it, hours_before, history = 0, 0.0, []     # history: (hours, mean loss) per log line, over all segments
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['opt'])
        it, hours_before, history = ckpt['iter'], ckpt['hours'], ckpt.get('history', [])
        log(f'resumed at iteration {it} ({hours_before:.2f} h)')

    acp = torch.tensor(alphas_cumprod(cfg['T']), dtype=torch.float32, device=device)
    data = DolceData(SplitData(sd, 'train'), norm, use_cond=args.p_uncond < 1)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=len(data) > args.batch_size,
                        persistent_workers=args.workers > 0)
    clock = TrainClock(args.hours, args.segment_hours, hours_before)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'train {len(data)} slices, {n_params / 1e6:.1f}M parameters, cfg {cfg}')
    cond = cond_indices(args.angle_start, args.angle_end, args.angle_stride)
    record = TrainRecord(out, {
        'method': args.name,
        'setting': {'angle_start': args.angle_start, 'angle_end': args.angle_end, 'angle_stride': args.angle_stride,
                    'n_views': len(cond), 'rows': f'{cond[0]}..{cond[-1]} step {args.angle_stride}'},
        'data': {'train_slices': len(data), 'setting_dir': sd, 'label_norm': [norm.lo, norm.hi]},
        'model': {'parameters': n_params, 'cfg': {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
                  'checkpoint_used': 'last.pt, EMA weights'},
        'optimizer': {'name': 'Adam', 'lr': args.lr, 'batch_size': args.batch_size, 'ema': args.ema,
                      'grad_clip': 1.0, 'p_uncond': args.p_uncond},
        'loss': {'name': 'eps-prediction MSE (DDPM)',
                 'formula': 'mean over batch and pixels of (eps_theta(x_t, t, c) - eps)^2, x_t = sqrt(a_t) x0 + '
                            'sqrt(1 - a_t) eps, t uniform in [0, T), linear beta schedule; the condition c (RLS '
                            'image of the measured views) is zeroed with probability p_uncond',
                 'logged_every_iterations': args.log_every},
        'stopping': {'hours_cap': args.hours, 'max_iters': args.max_iters or None,
                     'plateau': {'window_hours': args.plateau_hours, 'tolerance': args.plateau_tol,
                                 'min_hours': args.min_hours} if args.plateau_hours else None},
        'seed': args.seed}, columns=['iteration', 'hours', 'loss'])

    def save():
        save_checkpoint(last_path, {'model': model.state_dict(), 'ema': ema.state_dict(),
                                    'opt': opt.state_dict(), 'iter': it, 'hours': clock.total(),
                                    'cfg': cfg, 'label_norm': (norm.lo, norm.hi), 'history': history})

    def plateaued():
        """Mean loss over the last plateau_hours vs the plateau_hours before that."""
        w, t = args.plateau_hours, clock.total()
        if not w or t < max(args.min_hours, 2 * w):
            return False
        last = [l for h, l in history if t - w <= h]
        prev = [l for h, l in history if t - 2 * w <= h < t - w]
        if not last or not prev:
            return False
        gain = 1 - np.mean(last) / np.mean(prev)
        if gain < args.plateau_tol:
            log(f'plateau: mean loss {np.mean(prev):.5f} -> {np.mean(last):.5f} over the last '
                f'{2 * w:g} h ({100 * gain:+.1f}%, below {100 * args.plateau_tol:g}%)')
            return True
        return False

    last_save, losses = time.time(), []
    while True:
        for x0, c in loader:
            x0, c = x0.to(device, non_blocking=True), c.to(device, non_blocking=True)
            b = x0.shape[0]
            t = torch.randint(0, cfg['T'], (b,), device=device)
            noise = torch.randn_like(x0)
            ab = acp[t].view(b, 1, 1, 1)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
            keep = (torch.rand(b, device=device) >= args.p_uncond).float().view(b, 1, 1, 1)
            with autocast(device):
                eps = model(torch.cat([xt, c * keep], dim=1), t)
            loss = F.mse_loss(eps.float(), noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            it += 1
            losses.append(loss.item())
            plateau = False
            if it % args.log_every == 0:
                history.append((clock.total(), float(np.mean(losses))))
                record.loss(iteration=it, hours=clock.total(), loss=float(np.mean(losses)))
                log(f'iter {it}: loss {np.mean(losses):.5f} ({clock.total():.2f} h)')
                losses = []
                plateau = plateaued()

            finished = clock.run_over() or plateau or (args.max_iters and it >= args.max_iters)
            if finished or clock.segment_over() or time.time() - last_save > args.ckpt_minutes * 60:
                save()
                last_save = time.time()
                record.update(clock.segment(), iterations=it, epochs=round(it * args.batch_size / len(data), 2),
                              last_loss=history[-1][1] if history else None)
            if finished:
                open(done, 'w').close()
                reason = 'plateau' if plateau else 'max_iters' if args.max_iters and it >= args.max_iters else 'hours_cap'
                record.finish(reason, clock.segment(), iterations=it, epochs=round(it * args.batch_size / len(data), 2),
                              last_loss=history[-1][1] if history else None)
                log(f'dolce: training finished at iteration {it} ({reason})')
                return
            if clock.segment_over():
                record.update(clock.segment())
                log(f'dolce: segment over at iteration {it}; the next job resumes')
                return


if __name__ == '__main__':
    main()
