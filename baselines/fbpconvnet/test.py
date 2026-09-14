"""Reconstruct the test slices with the best FBPConvNet checkpoint.

    python -m baselines.fbpconvnet.test
"""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from ..common.config import add_common_args, method_dir, setting_dir
from ..common.data import Normalizer, SplitData
from ..common.runtime import Logger, autocast, load_checkpoint, pick_device
from .model import FBPConvNet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_slices', type=int, default=0, help='only the first N test slices (0 = all)')
    args = parser.parse_args()

    device = pick_device(args.device)
    out = method_dir(args, 'fbpconvnet')
    log = Logger(os.path.join(out, 'log.txt'))
    ckpt = load_checkpoint(os.path.join(out, 'best.pt')) or load_checkpoint(os.path.join(out, 'last.pt'))
    if ckpt is None:
        raise SystemExit(f'no checkpoint in {out}; run python -m baselines.fbpconvnet.train first')
    meta = ckpt['meta']
    model = FBPConvNet(base=meta['base']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    in_norm, out_norm = Normalizer(*meta['in_norm']), Normalizer(*meta['out_norm'])

    test = SplitData(setting_dir(args), 'test')
    recon_dir = os.path.join(out, 'recon')
    os.makedirs(recon_dir, exist_ok=True)
    n = min(args.max_slices, len(test)) if args.max_slices else len(test)
    with torch.no_grad():
        for start in tqdm(range(0, n, args.batch_size), desc='fbpconvnet'):
            idx = list(range(start, min(start + args.batch_size, n)))
            x = np.stack([in_norm.to_unit(test.image(meta['input'], i)) for i in idx])
            with autocast(device):
                pred = model(torch.from_numpy(x)[:, None].to(device))
            pred = out_norm.from_unit(pred.float().cpu().numpy()[:, 0])
            for k, i in enumerate(idx):
                np.save(os.path.join(recon_dir, test.slices[i]['id'] + '.npy'), pred[k].astype(np.float32))
    log(f'fbpconvnet: wrote {n} test reconstructions (checkpoint epoch {ckpt["epoch"]}) to {recon_dir}')


if __name__ == '__main__':
    main()
