"""DOLCE sampling (paper Algorithm 1): conditional DDIM steps (40 steps, eta 1,
as in the official evaluation script), each followed by the data-consistency
prox of eq. (10), argmin_z ||z - x||^2 + gamma ||A z - y||^2, solved with
conjugate gradients in the physical image units (see common/ct.py).

--dc_on sample applies the prox to each new sample, as the paper and the
official code do; --dc_on x0 applies it to the denoised estimate instead.
The variant and prox weight are tuned on validation slices (both variants,
three weights) and saved to tuned.json. A run without tuned.json tunes first;
--tune only tunes.

    python -m baselines.dolce.sample
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from ..common.config import add_common_args, cond_indices, method_dir, setting_dir
from ..common.ct import MeasuredOperator
from ..common.data import Normalizer, SplitData, load_offset
from ..common.metrics import ct_metrics
from ..common.runtime import Logger, autocast, evenly_spaced, load_checkpoint, pick_device, shard
from .model import alphas_cumprod, build_model, condition

TUNE_GRID = [(dc_on, lam) for dc_on in ('sample', 'x0') for lam in (0.1, 1.0, 10.0)]


class Sampler:
    def __init__(self, model, cfg, op, cond, offset, norm, args, device):
        self.model, self.op, self.cond, self.offset, self.norm, self.device = model, op, cond, offset, norm, device
        self.acp = alphas_cumprod(cfg['T'])
        stride = max(cfg['T'] // args.steps, 1)
        self.ts = list(range(0, cfg['T'], stride))[:args.steps]   # guided-diffusion's "ddim40" spacing
        self.eta, self.guidance, self.cg_iters = args.eta, args.guidance, args.cg_iters
        self.dc_on, self.lam = 'sample', 1.0

    def configure(self, dc_on, lam_rel):
        # The prox weight is relative to ||A^T A||, so it does not depend on the geometry's scale
        self.dc_on, self.lam = dc_on, lam_rel * self.op.norm_sq()

    def dc(self, x, ys):
        arr = x[:, 0].float().cpu().numpy()
        out = torch.empty_like(x)
        for b, y in enumerate(ys):
            z = self.norm.from_unit(arr[b]) + self.offset
            z = self.op.prox(z.astype(np.float32), y, self.lam, iters=self.cg_iters)
            out[b, 0] = torch.from_numpy(self.norm.to_unit(z - self.offset).astype(np.float32))
        return out

    @torch.no_grad()
    def eps(self, x, c, t):
        tt = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        with autocast(self.device):
            e = self.model(torch.cat([x, c], dim=1), tt).float()
            if self.guidance != 1.0:
                e_u = self.model(torch.cat([x, torch.zeros_like(c)], dim=1), tt).float()
                e = self.guidance * e + (1 - self.guidance) * e_u
        return e

    def run(self, conds, ys, gen):
        x = torch.randn((conds.shape[0], 1) + tuple(conds.shape[2:]), generator=gen, device=self.device)
        for k in reversed(range(len(self.ts))):
            ab = float(self.acp[self.ts[k]])
            ab_prev = float(self.acp[self.ts[k - 1]]) if k > 0 else 1.0
            eps = self.eps(x, conds, self.ts[k])
            x0 = ((x - (1 - ab) ** 0.5 * eps) / ab ** 0.5).clamp(-1, 1)
            if self.dc_on == 'x0':
                x0 = self.dc(x0, ys)
                eps = (x - ab ** 0.5 * x0) / (1 - ab) ** 0.5
            sigma = self.eta * ((1 - ab_prev) / (1 - ab)) ** 0.5 * (1 - ab / ab_prev) ** 0.5
            x = ab_prev ** 0.5 * x0 + max(1 - ab_prev - sigma ** 2, 0.0) ** 0.5 * eps
            if k > 0:
                x = x + sigma * torch.randn(x.shape, generator=gen, device=self.device)
            if self.dc_on == 'sample':
                x = self.dc(x, ys)
        return x

    def reconstruct(self, data, indices, batch_size, seed, save_dir=None, desc='dolce'):
        """Images in the label's units; saved as <slice id>.npy when save_dir is given."""
        results = {}
        for start in tqdm(range(0, len(indices), batch_size), desc=desc):
            idx = indices[start:start + batch_size]
            conds = torch.from_numpy(np.stack([condition(data.image('rls', i)) for i in idx]))[:, None]
            ys = [data.sinogram(i)[self.cond] + 1.0 for i in idx]
            gen = torch.Generator(device=self.device).manual_seed(seed * 1_000_003 + idx[0])
            x = self.run(conds.to(self.device), ys, gen)
            imgs = self.norm.from_unit(x[:, 0].float().cpu().numpy())
            for k, i in enumerate(idx):
                if save_dir:
                    np.save(os.path.join(save_dir, data.slices[i]['id'] + '.npy'), imgs[k].astype(np.float32))
                else:
                    results[i] = imgs[k]
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument('--split', default='test')
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--guidance', type=float, default=1.0,
                        help='classifier-free guidance weight (paper eq. 8); 1 = conditional only')
    parser.add_argument('--dc_on', choices=['sample', 'x0'], default=None, help='default: tuned.json, else sample')
    parser.add_argument('--dc_lambda', type=float, default=None,
                        help='prox weight 1/gamma relative to ||A^T A||; default: tuned.json, else 1')
    parser.add_argument('--cg_iters', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--tune', action='store_true', help='pick dc_on and dc_lambda on validation slices')
    parser.add_argument('--tune_slices', type=int, default=4)
    parser.add_argument('--shard', type=int, default=None, help='default: SLURM array task id')
    parser.add_argument('--num_shards', type=int, default=None, help='default: SLURM array size')
    parser.add_argument('--max_slices', type=int, default=0, help='only the first N slices (0 = all)')
    args = parser.parse_args()

    out = method_dir(args, 'dolce')
    log = Logger(os.path.join(out, 'log.txt'))
    if not os.path.exists(os.path.join(out, 'train_done')):
        raise SystemExit(f'{out} has no finished training; run python -m baselines.dolce.train')
    device = pick_device(args.device)
    ckpt = load_checkpoint(os.path.join(out, 'last.pt'))
    model = build_model(ckpt['cfg']).to(device)
    model.load_state_dict(ckpt['ema'])
    model.eval()

    sd = setting_dir(args)
    cond = cond_indices(args.angle_start, args.angle_end, args.angle_stride)
    op = MeasuredOperator(cond)
    sampler = Sampler(model, ckpt['cfg'], op, cond, load_offset(sd), Normalizer(*ckpt['label_norm']), args, device)
    tuned_path = os.path.join(out, 'tuned.json')

    untuned = not os.path.exists(tuned_path) and args.dc_on is None and args.dc_lambda is None
    if args.tune or untuned:
        val = SplitData(sd, 'val')
        picks = evenly_spaced(len(val), args.tune_slices)
        scores = []
        for dc_on, lam in TUNE_GRID:
            sampler.configure(dc_on, lam)
            imgs = sampler.reconstruct(val, picks, args.batch_size, args.seed, desc=f'tune {dc_on} {lam:g}')
            ssim = float(np.mean([ct_metrics(imgs[i], val.image('label', i))['ssim'] for i in picks]))
            scores.append({'dc_on': dc_on, 'dc_lambda': lam, 'val_ssim': ssim})
            log(f'dolce tune: dc_on={dc_on} dc_lambda={lam:g} -> val SSIM {ssim:.4f}')
        best = max(scores, key=lambda s: s['val_ssim'])
        with open(tuned_path, 'w') as f:
            json.dump({'best': best, 'scores': scores, 'val_slices': picks}, f, indent=2)
        log(f'dolce tune: best {best}')
        if args.tune:
            op.close()
            return

    tuned = {}
    if os.path.exists(tuned_path):
        with open(tuned_path) as f:
            tuned = json.load(f)['best']
    dc_on = args.dc_on or tuned.get('dc_on', 'sample')
    lam = args.dc_lambda if args.dc_lambda is not None else tuned.get('dc_lambda', 1.0)
    sampler.configure(dc_on, lam)

    data = SplitData(sd, args.split)
    recon_dir = os.path.join(out, 'recon' if args.split == 'test' else f'recon_{args.split}')
    os.makedirs(recon_dir, exist_ok=True)
    n = min(args.max_slices, len(data)) if args.max_slices else len(data)
    mine, shard_id, num_shards = shard(range(n), args.shard, args.num_shards)
    todo = [i for i in mine if not os.path.exists(os.path.join(recon_dir, data.slices[i]['id'] + '.npy'))]
    log(f'dolce: shard {shard_id}/{num_shards}, {len(todo)} of {len(mine)} slices to do, '
        f'dc_on={dc_on}, dc_lambda={lam:g}, {args.steps} steps')
    sampler.reconstruct(data, todo, args.batch_size, args.seed, save_dir=recon_dir)
    log(f'dolce: shard {shard_id} done')
    op.close()


if __name__ == '__main__':
    main()
