"""DuDoTrans reconstruction of the test slices with the best validation checkpoint.
Writes recon/<slice id>.npy (the final image X~, in the label's units) and
sino/<slice id>.npy (the restored sinogram, in SD-Flow's [-1, 1] units).

    python -m baselines.dudotrans.test
"""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from ..common.config import add_common_args, method_dir, setting_dir
from ..common.data import SplitData, load_offset
from ..common.runtime import Logger, autocast, load_checkpoint, pick_device
from .train import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument('--split', default='test')
    parser.add_argument('--max_slices', type=int, default=0, help='only the first N slices (0 = all)')
    args = parser.parse_args()

    device = pick_device(args.device)
    out = method_dir(args, 'dudotrans')
    log = Logger(os.path.join(out, 'log.txt'))
    ckpt = load_checkpoint(os.path.join(out, 'best.pt')) or load_checkpoint(os.path.join(out, 'last.pt'))
    if ckpt is None:
        raise SystemExit(f'no checkpoint in {out}; run python -m baselines.dudotrans.train first')
    meta = ckpt['meta']
    model = build_model(meta, device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    sd = setting_dir(args)
    data = SplitData(sd, args.split)
    offset = load_offset(sd)
    recon_dir = os.path.join(out, 'recon' if args.split == 'test' else f'recon_{args.split}')
    sino_dir = os.path.join(out, 'sino' if args.split == 'test' else f'sino_{args.split}')
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(sino_dir, exist_ok=True)
    n = min(args.max_slices, len(data)) if args.max_slices else len(data)
    with torch.no_grad():
        for i in tqdm(range(n), desc='dudotrans'):
            y = torch.from_numpy(data.sinogram(i) + 1.0).to(device)[None]
            with autocast(device):
                y_res, _, _, x = model(y[:, meta['cond']])
            sid = data.slices[i]['id']
            np.save(os.path.join(recon_dir, sid + '.npy'),
                    (x[0].float().cpu().numpy() * meta['z_scale'] - offset).astype(np.float32))
            np.save(os.path.join(sino_dir, sid + '.npy'), (y_res[0].float().cpu().numpy() - 1.0).astype(np.float32))
    log(f'dudotrans: wrote {n} {args.split} reconstructions (epoch {ckpt.get("epoch")}) to {recon_dir}')


if __name__ == '__main__':
    main()
