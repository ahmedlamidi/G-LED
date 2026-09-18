"""DPS: Diffusion Posterior Sampling (Chung et al., ICLR 2023) for CT.

An unconditional image-domain diffusion prior, and at every reverse step the
gradient of the measurement residual through the denoiser (Algorithm 1):

    x0_hat  = (x_t - sqrt(1 - abar_t) eps(x_t, t)) / sqrt(abar_t)
    x_{t-1} = ancestral DDPM step from x_t and x0_hat
              - zeta * grad_{x_t} || y - A x0_hat ||_2

A is the fan-beam projector onto the measured views (common/ct.py), applied in
the physical units DOLCE's data consistency uses (label + K), y = s + 1 the
measured rows. Nothing is trained per view setting: one prior serves them all.

The prior is a DOLCE checkpoint run with its condition zeroed. DOLCE is trained
with 20% condition dropout, so that branch is an unconditional model of the
label images. The dedicated prior is the same network trained with the condition
always dropped (dps/train_prior.sh -> <default setting>/dps_prior); once its
training has finished it is the default --prior, before that DOLCE's is.

zeta is tuned on validation slices (body SSIM) and saved in tuned.json; a run
without tuned.json tunes first, --tune only tunes. The gradient is divided by
||A|| and by the image normalization's scale, so zeta does not depend on the
number of views or on the units.

    python -m baselines.dps.sample
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from ..common.config import DEFAULT_OUT_ROOT, add_common_args, cond_indices, dps_prior, method_dir, setting_dir
from ..common.ct import MeasuredOperator
from ..common.data import Normalizer, SplitData, load_offset
from ..common.metrics import ct_scores
from ..common.runtime import Logger, autocast, evenly_spaced, load_checkpoint, pick_device
from ..common.torch_ct import astra_project
from ..dolce.model import alphas_cumprod, build_model

ZETA_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)


class DPS:
    def __init__(self, model, cfg, op, cond, offset, norm, steps, device):
        self.model, self.op, self.cond, self.norm, self.device = model, op, cond, norm, device
        self.offset = torch.from_numpy(offset.astype(np.float32)).to(device)
        T = cfg['T']
        self.ts = list(range(0, T, max(T // steps, 1)))[:steps]       # respaced schedule, as guided-diffusion does
        acp = alphas_cumprod(T)[self.ts]
        self.abar = acp
        self.abar_prev = np.concatenate([[1.0], acp[:-1]])
        self.beta = 1 - self.abar / self.abar_prev
        self.half_range = (norm.hi - norm.lo) / 2
        self.grad_scale = 1.0 / (np.sqrt(op.norm_sq()) * self.half_range)
        self.zeta = 1.0

    def physical(self, u):
        """[-1, 1] image -> the units of the projections (label + K)."""
        return (u + 1) * self.half_range + self.norm.lo + self.offset

    def run(self, ys, gen, desc=None):
        y = torch.from_numpy(np.stack(ys)).to(self.device)[:, None]                  # [B, 1, views, det]
        b = y.shape[0]
        x = torch.randn((b, 1) + tuple(self.offset.shape), generator=gen, device=self.device)
        zeros = torch.zeros_like(x)
        steps = reversed(range(len(self.ts)))
        for k in (tqdm(list(steps), desc=desc, leave=False) if desc else steps):
            ab, ab_prev, beta = float(self.abar[k]), float(self.abar_prev[k]), float(self.beta[k])
            x = x.detach().requires_grad_(True)
            tt = torch.full((b,), self.ts[k], device=self.device, dtype=torch.long)
            with autocast(self.device):
                eps = self.model(torch.cat([x, zeros], dim=1), tt).float()            # condition zeroed: the prior
            x0 = (x - (1 - ab) ** 0.5 * eps) / ab ** 0.5
            residual = (y - astra_project(self.physical(x0), self.op)).flatten(1).norm(dim=1)
            grad, = torch.autograd.grad(residual.sum(), x)
            with torch.no_grad():
                x0c = x0.clamp(-1, 1)
                mean = (beta * ab_prev ** 0.5 / (1 - ab)) * x0c + ((1 - ab_prev) * (1 - beta) ** 0.5 / (1 - ab)) * x
                if k > 0:
                    var = beta * (1 - ab_prev) / (1 - ab)
                    mean = mean + var ** 0.5 * torch.randn(x.shape, generator=gen, device=self.device)
                x = mean - self.zeta * self.grad_scale * grad
        return x.detach(), residual.detach()

    def reconstruct(self, data, indices, batch_size, seed, save_dir=None, desc='dps'):
        results = {}
        for start in tqdm(range(0, len(indices), batch_size), desc=desc):
            idx = indices[start:start + batch_size]
            ys = [data.sinogram(i)[self.cond] + 1.0 for i in idx]
            gen = torch.Generator(device=self.device).manual_seed(seed * 1_000_003 + idx[0])
            x, _ = self.run(ys, gen)
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
    parser.add_argument('--prior', default=dps_prior(os.path.join(DEFAULT_OUT_ROOT, 'limited0-45_stride10')),
                        help='DOLCE-style checkpoint whose unconditional branch is the prior; default: the '
                             'finished dps_prior of the default setting, else its dolce')
    parser.add_argument('--steps', type=int, default=1000, help='reverse steps (the paper: 1000)')
    parser.add_argument('--zeta', type=float, default=None, help='step size; default: tuned.json, else 1')
    parser.add_argument('--tune', action='store_true', help='only pick zeta on validation slices')
    parser.add_argument('--tune_slices', type=int, default=4)
    parser.add_argument('--tune_dir', default=None,
                        help='setting folder whose val split is used for tuning (default: this setting\'s; the '
                             'sinograms and labels of a split do not depend on the measured views)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--split', default='test')
    parser.add_argument('--max_slices', type=int, default=0, help='only the first N slices (0 = all)')
    parser.add_argument('--name', default='dps', help='method folder, e.g. dps_dolce for a run with another prior')
    args = parser.parse_args()

    out = method_dir(args, args.name)
    log = Logger(os.path.join(out, 'log.txt'))
    device = pick_device(args.device)
    ckpt = load_checkpoint(args.prior)
    if ckpt is None:
        raise SystemExit(f'no prior at {args.prior}: train DOLCE (or a dedicated prior, see --help) first')
    model = build_model(ckpt['cfg']).to(device)
    model.load_state_dict(ckpt['ema'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sd = setting_dir(args)
    cond = cond_indices(args.angle_start, args.angle_end, args.angle_stride)
    op = MeasuredOperator(cond)
    offset = load_offset(sd)
    dps = DPS(model, ckpt['cfg'], op, cond, offset, Normalizer(*ckpt['label_norm']), args.steps, device)
    tuned_path = os.path.join(out, 'tuned.json')
    log(f'dps: prior {args.prior} (iteration {ckpt.get("iter")}), {len(cond)} views, {len(dps.ts)} steps')

    if os.path.exists(tuned_path) and not args.tune:
        with open(tuned_path) as f:
            before = json.load(f).get('prior')
        if before and os.path.abspath(before) != os.path.abspath(args.prior):
            raise SystemExit(f'{out} holds results of another prior ({before}); pass a different --name '
                             f'or delete that folder')

    if args.tune or (args.zeta is None and not os.path.exists(tuned_path)):
        val = SplitData(args.tune_dir or sd, 'val')
        picks = evenly_spaced(len(val), args.tune_slices)
        scores = []
        for zeta in ZETA_GRID:
            dps.zeta = zeta
            imgs = dps.reconstruct(val, picks, args.batch_size, args.seed, desc=f'tune zeta {zeta:g}')
            ssim = float(np.mean([ct_scores(imgs[i], val.image('label', i), offset)['ssim_body'] for i in picks]))
            scores.append({'zeta': zeta, 'val_ssim_body': ssim})
            log(f'dps tune: zeta={zeta:g} -> val body SSIM {ssim:.4f}')
        best = max(scores, key=lambda s: s['val_ssim_body'])
        with open(tuned_path, 'w') as f:
            json.dump({'best': best, 'scores': scores, 'val_slices': picks, 'steps': len(dps.ts),
                       'prior': args.prior}, f, indent=2)
        log(f'dps tune: best {best}')
        if args.tune:
            op.close()
            return

    if args.zeta is not None:
        dps.zeta = args.zeta
    elif os.path.exists(tuned_path):
        with open(tuned_path) as f:
            dps.zeta = json.load(f)['best']['zeta']

    data = SplitData(sd, args.split)
    recon_dir = os.path.join(out, 'recon' if args.split == 'test' else f'recon_{args.split}')
    os.makedirs(recon_dir, exist_ok=True)
    n = min(args.max_slices, len(data)) if args.max_slices else len(data)
    todo = [i for i in range(n) if not os.path.exists(os.path.join(recon_dir, data.slices[i]['id'] + '.npy'))]
    log(f'dps: {len(todo)} of {n} slices to do, zeta={dps.zeta:g}')
    dps.reconstruct(data, todo, args.batch_size, args.seed, save_dir=recon_dir)
    log('dps: done')
    op.close()


if __name__ == '__main__':
    main()
