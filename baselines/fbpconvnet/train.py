"""Train FBPConvNet (Jin et al., IEEE TIP 2017).

Input: the FBP of the measured rows, the same image SD-Flow's conditioning is
built from (--input rls uses DOLCE's RLS image instead, as DOLCE's U-Net
baseline does). Target: the label. Each image type gets one global affine
normalization from the training patients. MSE loss, gradient clipping, and the
checkpoint with the lowest validation MSE is kept.

    python -m baselines.fbpconvnet.train
"""
import argparse
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..common.config import add_common_args, cond_indices, method_dir, setting_dir
from ..common.data import Normalizer, SplitData, load_stats
from ..common.runtime import (Logger, TrainClock, TrainRecord, autocast, load_checkpoint, pick_device,
                              save_checkpoint, seed_everything)
from .model import FBPConvNet


class Pairs(Dataset):
    def __init__(self, data, input_name, in_norm, out_norm):
        self.data, self.input_name, self.in_norm, self.out_norm = data, input_name, in_norm, out_norm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.in_norm.to_unit(self.data.image(self.input_name, i))
        y = self.out_norm.to_unit(self.data.image('label', i))
        return torch.from_numpy(x)[None], torch.from_numpy(y)[None]


@torch.no_grad()
def val_mse(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(device):
            pred = model(x)
        total += F.mse_loss(pred.float(), y, reduction='sum').item()
        count += y.numel()
    model.train()
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument('--input', choices=['fbp_in', 'rls'], default='fbp_in')
    parser.add_argument('--base', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--hours', type=float, default=12, help='stop after this much training time')
    parser.add_argument('--patience', type=int, default=15,
                        help='stop when the validation MSE has not improved for this many epochs (0 = never)')
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device(args.device)
    sd = setting_dir(args)
    out = method_dir(args, 'fbpconvnet')
    log = Logger(os.path.join(out, 'log.txt'))

    stats = load_stats(sd)
    in_norm = Normalizer(stats[f'{args.input}_lo'], stats[f'{args.input}_hi'])
    out_norm = Normalizer(stats['label_lo'], stats['label_hi'])
    train = Pairs(SplitData(sd, 'train'), args.input, in_norm, out_norm)
    val = Pairs(SplitData(sd, 'val'), args.input, in_norm, out_norm)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=len(train) > args.batch_size,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers,
                            persistent_workers=args.workers > 0)

    model = FBPConvNet(base=args.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = max(args.epochs * len(train_loader), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)))
    meta = {'input': args.input, 'base': args.base,
            'in_norm': (in_norm.lo, in_norm.hi), 'out_norm': (out_norm.lo, out_norm.hi)}

    last_path = os.path.join(out, 'last.pt')
    start_epoch, best, best_epoch, hours_before = 0, float('inf'), -1, 0.0
    ckpt = load_checkpoint(last_path)
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        start_epoch, best, hours_before = ckpt['epoch'] + 1, ckpt['best_val'], ckpt['hours']
        best_epoch = ckpt.get('best_epoch', -1)
        log(f'resumed after epoch {ckpt["epoch"]}')
    clock = TrainClock(args.hours, args.hours, hours_before)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'train {len(train)} slices, val {len(val)} slices, {n_params / 1e6:.1f}M parameters, input {args.input}')
    cond = cond_indices(args.angle_start, args.angle_end, args.angle_stride)
    record = TrainRecord(out, {
        'method': 'fbpconvnet',
        'setting': {'angle_start': args.angle_start, 'angle_end': args.angle_end, 'angle_stride': args.angle_stride,
                    'n_views': len(cond), 'rows': f'{cond[0]}..{cond[-1]} step {args.angle_stride}'},
        'data': {'train_slices': len(train), 'val_slices': len(val), 'setting_dir': sd, 'input': args.input,
                 'in_norm': [in_norm.lo, in_norm.hi], 'out_norm': [out_norm.lo, out_norm.hi]},
        'model': {'parameters': n_params, 'base': args.base, 'checkpoint_used': 'best.pt (lowest validation MSE)'},
        'optimizer': {'name': 'Adam', 'lr': args.lr, 'schedule': f'cosine to 0 over {args.epochs} epochs',
                      'batch_size': args.batch_size, 'grad_clip': 1.0},
        'loss': {'name': 'MSE',
                 'formula': f'mean over batch and pixels of (net({args.input}) - label)^2, both globally normalized '
                            'to about [-1, 1]; validation MSE is the same on the validation patient',
                 'logged_every': 'epoch'},
        'stopping': {'epochs_cap': args.epochs, 'hours_cap': args.hours, 'patience_epochs': args.patience or None},
        'seed': args.seed}, columns=['epoch', 'hours', 'train_mse', 'val_mse'])
    reason = 'epochs_cap'

    for epoch in range(start_epoch, args.epochs):
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device):
                pred = model(x)
            loss = F.mse_loss(pred.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()

        v = val_mse(model, val_loader, device)
        if v < best:
            best, best_epoch = v, epoch
            save_checkpoint(os.path.join(out, 'best.pt'),
                            {'model': model.state_dict(), 'epoch': epoch, 'val_mse': v, 'meta': meta})
        save_checkpoint(last_path, {'model': model.state_dict(), 'opt': opt.state_dict(),
                                    'sched': sched.state_dict(), 'epoch': epoch, 'best_val': best,
                                    'best_epoch': best_epoch, 'hours': clock.total(), 'meta': meta})
        train_mse = running / max(len(train_loader), 1)
        record.loss(epoch=epoch, hours=clock.total(), train_mse=train_mse, val_mse=v)
        record.update(clock.segment(), epochs=epoch + 1, best_val_mse=best, best_epoch=best_epoch)
        log(f'epoch {epoch}: train MSE {train_mse:.5f}, '
            f'val MSE {v:.5f}, best {best:.5f} at epoch {best_epoch} ({clock.total():.2f} h)')
        if clock.run_over():
            reason = 'hours_cap'
            log('time budget reached')
            break
        if args.patience and epoch - best_epoch >= args.patience:
            reason = 'patience'
            log(f'no validation improvement for {args.patience} epochs; best.pt is epoch {best_epoch}')
            break
    record.finish(reason, clock.segment(), best_val_mse=best, best_epoch=best_epoch)
    open(os.path.join(out, 'train_done'), 'w').close()


if __name__ == '__main__':
    main()
