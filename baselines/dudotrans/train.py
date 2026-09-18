"""Train DuDoTrans for one view setting.

Loss (paper eq. 11, lambda_1 = lambda_2 = 1):
    L = ||Y~ - Y_gt||^2 + lambda_1 ||X~2 - X_gt||^2 + lambda_2 ||X~ - X_gt||^2
with Y_gt the full sinogram, X_gt the label, X~2 the FBP of the restored
sinogram and X~ the final image. Adam (0.9, 0.999) at 1e-4, batch 1 and up to
100 epochs as in the paper; the best validation epoch (image MSE on the
validation patient) is kept, and training stops early when validation has not
improved for --patience epochs or after --hours.

Units: sinograms as y = s + 1 (air is 0), images as (label + K) / z_scale so
they are of order one; FBP is linear, so the consistency layer needs no other
change. Needs only the sinograms and labels of prepare_data.py
(--images label), no RLS or FBP arrays.

    python -m baselines.dudotrans.train
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..common.config import SAMPLE_H, add_common_args, cond_indices, method_dir, setting_dir
from ..common.data import SplitData, load_offset, load_stats
from ..common.runtime import (Logger, TrainClock, TrainRecord, autocast, load_checkpoint, pick_device,
                              save_checkpoint, seed_everything)
from ..common.torch_ct import FanFBP, all_angles
from .model import DuDoTrans


def fill_matrix(cond, n_views=SAMPLE_H):
    """[n_views, len(cond)]: measured rows stay, a row between two measured rows one regular stride
    apart is their linear interpolation, and rows in larger gaps (outside a limited arc) are zero."""
    cond = list(cond)
    order = sorted(range(len(cond)), key=lambda k: cond[k])
    rows = [cond[k] for k in order]
    gaps = [(rows[(j + 1) % len(rows)] - rows[j]) % n_views or n_views for j in range(len(rows))]
    regular = min(gaps)
    fill = torch.zeros(n_views, len(cond))
    for j, r0 in enumerate(rows):
        fill[r0, order[j]] = 1.0
        if gaps[j] == regular and regular > 1:
            nxt = order[(j + 1) % len(rows)]
            for d in range(1, regular):
                fill[(r0 + d) % n_views, order[j]] = 1 - d / regular
                fill[(r0 + d) % n_views, nxt] = d / regular
    return fill


def build_model(meta, device):
    cond = meta['cond']
    angles = all_angles()
    model = DuDoTrans(FanFBP(angles), FanFBP(angles[cond]), fill_matrix(cond), cond, 1.0 / meta['z_scale'],
                      **meta['net'])
    return model.to(device)


class Sinograms(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.from_numpy(self.data.sinogram(i) + 1.0), torch.from_numpy(self.data.image('label', i))


def losses(model, y_full, label, offset, meta):
    y_res, _, x2, x = model(y_full[:, meta['cond']])
    target = (label + offset) / meta['z_scale']
    return F.mse_loss(y_res.float(), y_full), F.mse_loss(x2, target), F.mse_loss(x, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument('--dim', type=int, default=48)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--srt_groups', type=int, default=3, help='m: residual blocks in the SRT (paper: 3)')
    parser.add_argument('--srt_blocks', type=int, default=2, help='Swin blocks per SRT block (one module = regular + shifted)')
    parser.add_argument('--rirm_depth', type=int, default=2, help='sub-modules in the RIRM (paper: 2)')
    parser.add_argument('--rirm_width', type=int, default=4, help='Swin blocks per RIRM sub-module (paper: 4)')
    parser.add_argument('--patch', type=int, default=2, help='stride of the embedding conv (1 = full resolution)')
    parser.add_argument('--no_data_consistency', action='store_true', help='do not re-insert the measured rows')
    parser.add_argument('--grad_checkpoint', action='store_true', help='trade compute for memory')
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15,
                        help='stop when the validation MSE has not improved for this many epochs (0 = never)')
    parser.add_argument('--hours', type=float, default=20, help='stop after this much training time')
    parser.add_argument('--max_batches', type=int, default=0, help='batches per epoch (0 = all); for smoke tests')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device(args.device)
    sd = setting_dir(args)
    out = method_dir(args, 'dudotrans')
    log = Logger(os.path.join(out, 'log.txt'))
    if os.path.exists(os.path.join(out, 'train_done')):
        log('dudotrans: training already finished')
        return

    last_path = os.path.join(out, 'last.pt')
    ckpt = load_checkpoint(last_path)
    cond = cond_indices(args.angle_start, args.angle_end, args.angle_stride)
    meta = ckpt['meta'] if ckpt is not None else {
        'cond': cond, 'z_scale': load_stats(sd)['z_scale'],
        'net': {'dim': args.dim, 'heads': args.heads, 'window': args.window, 'mlp_ratio': args.mlp_ratio,
                'srt_groups': args.srt_groups, 'srt_blocks': args.srt_blocks, 'rirm_depth': args.rirm_depth,
                'rirm_width': args.rirm_width, 'patch': args.patch,
                'data_consistency': not args.no_data_consistency, 'grad_checkpoint': args.grad_checkpoint}}
    model = build_model(meta, device)
    offset = torch.from_numpy(load_offset(sd).astype(np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    train, val = Sinograms(SplitData(sd, 'train')), Sinograms(SplitData(sd, 'val'))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers,
                            persistent_workers=args.workers > 0)

    start_epoch, best, best_epoch, hours_before = 0, float('inf'), -1, 0.0
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_epoch, best, best_epoch, hours_before = ckpt['epoch'] + 1, ckpt['best_val'], ckpt['best_epoch'], ckpt['hours']
        log(f'resumed after epoch {ckpt["epoch"]}')
    clock = TrainClock(args.hours, args.hours, hours_before)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'train {len(train)} slices, val {len(val)} slices, {n_params / 1e6:.2f}M parameters, {len(cond)} views')
    record = TrainRecord(out, {
        'method': 'dudotrans',
        'setting': {'angle_start': args.angle_start, 'angle_end': args.angle_end, 'angle_stride': args.angle_stride,
                    'n_views': len(cond), 'rows': f'{cond[0]}..{cond[-1]} step {args.angle_stride}'},
        'data': {'train_slices': len(train), 'val_slices': len(val), 'setting_dir': sd, 'z_scale': meta['z_scale']},
        'model': {'parameters': n_params, 'net': meta['net'], 'checkpoint_used': 'best.pt (lowest validation image MSE)'},
        'optimizer': {'name': 'Adam', 'betas': [0.9, 0.999], 'lr': args.lr, 'schedule': 'constant',
                      'batch_size': args.batch_size},
        'loss': {'name': 'DuDoTrans eq. 11',
                 'formula': f'MSE(Y~, Y_gt) + {args.lambda1:g} MSE(FBP(Y~), X_gt) + {args.lambda2:g} MSE(X~, X_gt); '
                            'sinograms as s + 1, images as (label + K) / z_scale',
                 'logged_every': 'epoch'},
        'stopping': {'epochs_cap': args.epochs, 'hours_cap': args.hours, 'patience_epochs': args.patience or None},
        'seed': args.seed}, columns=['epoch', 'hours', 'train_loss', 'train_sino', 'train_dc', 'train_img', 'val_img_mse'])
    reason = 'epochs_cap'

    for epoch in range(start_epoch, args.epochs):
        model.train()
        sums, n = np.zeros(3), 0
        for b, (y_full, label) in enumerate(train_loader):
            y_full, label = y_full.to(device, non_blocking=True), label.to(device, non_blocking=True)
            with autocast(device):
                l_sino, l_dc, l_img = losses(model, y_full, label, offset, meta)
            loss = l_sino + args.lambda1 * l_dc + args.lambda2 * l_img
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sums += [l_sino.item(), l_dc.item(), l_img.item()]
            n += 1
            if args.max_batches and n >= args.max_batches:
                break
        model.eval()
        v, nv = 0.0, 0
        with torch.no_grad():
            for b, (y_full, label) in enumerate(val_loader):
                y_full, label = y_full.to(device), label.to(device)
                with autocast(device):
                    v += losses(model, y_full, label, offset, meta)[2].item()
                nv += 1
                if args.max_batches and nv >= args.max_batches:
                    break
        v /= max(nv, 1)
        t_sino, t_dc, t_img = sums / max(n, 1)
        if v < best:
            best, best_epoch = v, epoch
            save_checkpoint(os.path.join(out, 'best.pt'), {'model': model.state_dict(), 'epoch': epoch,
                                                           'val_img_mse': v, 'meta': meta})
        save_checkpoint(last_path, {'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch,
                                    'best_val': best, 'best_epoch': best_epoch, 'hours': clock.total(), 'meta': meta})
        record.loss(epoch=epoch, hours=clock.total(), train_loss=float(t_sino + args.lambda1 * t_dc + args.lambda2 * t_img),
                    train_sino=float(t_sino), train_dc=float(t_dc), train_img=float(t_img), val_img_mse=float(v))
        record.update(clock.segment(), epochs=epoch + 1, best_val_img_mse=best, best_epoch=best_epoch)
        log(f'epoch {epoch}: train sino {t_sino:.5f} dc {t_dc:.5f} img {t_img:.5f}, val img MSE {v:.5f}, '
            f'best {best:.5f} at epoch {best_epoch} ({clock.total():.2f} h)')
        if clock.run_over():
            reason = 'hours_cap'
            log('time budget reached')
            break
        if args.patience and epoch - best_epoch >= args.patience:
            reason = 'patience'
            log(f'no validation improvement for {args.patience} epochs; best.pt is epoch {best_epoch}')
            break
    record.finish(reason, clock.segment(), best_val_img_mse=best, best_epoch=best_epoch)
    open(os.path.join(out, 'train_done'), 'w').close()


if __name__ == '__main__':
    main()
