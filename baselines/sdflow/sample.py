"""SD-Flow on the baselines' test slices, so evaluate.py scores it like every
other method.

Model, conditioning and sampling are main_diff_eval_bfs.py's and
test_diff.py's: three channels (the sinogram with only the measured rows kept,
the mask of those rows, and the FBP of the measured rows re-projected to every
view, compute_fbp_reprojection), then trainer.sample completes the sinogram.
The measured views come from the SD-Flow config and may differ from the
baselines'; views.json records them and evaluate.py reports them. The completed
sinogram goes to sino/, its FBP (the label's FBP, so the label's units) to recon/.
Resubmitting skips slices already done.

    python -m baselines.sdflow.sample --config configurations/Ablation_45_PHYSICS_ONLY.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from ..common.config import REPO_ROOT, add_common_args, method_dir, setting_dir
from ..common.ct import full_fbp
from ..common.data import SplitData
from ..common.runtime import Logger, pick_device


def sdflow_modules():
    """SD-Flow's model classes and its FBP re-projection, from this repo."""
    for path in (REPO_ROOT, os.path.join(REPO_ROOT, 'data')):
        if path not in sys.path:
            sys.path.insert(0, path)
    from mimagen_pytorch import ElucidatedImagen, ImagenTrainer, Unet3D
    from dicom_preprocess import compute_fbp_reprojection
    return Unet3D, ElucidatedImagen, ImagenTrainer, compute_fbp_reprojection


def build_trainer(modules, cfg, device):
    """The network and sampler exactly as main_diff_eval_bfs.py builds them."""
    Unet3D, ElucidatedImagen, ImagenTrainer, _ = modules
    unet1 = Unet3D(dim=cfg['unet_dim'],
                   cond_images_channels=3,  # masked sinogram + binary mask + FBP re-projection
                   memory_efficient=True,
                   dim_mults=(1, 2, 4, 8)).to(device)
    imagen = ElucidatedImagen(
        unets=(unet1),
        image_sizes=(cfg['sample_H']),
        image_width=(cfg['detector_count']),
        channels=1,
        random_crop_sizes=None,
        num_sample_steps=cfg['num_sample_steps'],
        cond_drop_prob=0.1,
        sigma_min=0.002,
        sigma_max=(80),
        sigma_data=0.5,
        rho=7,
        P_mean=-1.2,
        P_std=1.2,
        S_churn=80,
        S_tmin=0.05,
        S_tmax=50,
        S_noise=1.003,
        condition_on_text=False,
        auto_normalize_img=False).to(device)
    return ImagenTrainer(imagen, device=device, fp16=True)


def find_checkpoint(folder):
    """main_diff_eval_bfs.py loads <folder>/best_model_sofar; training writes it
    to <folder>/diffusion_folder/model_save/."""
    for path in (os.path.join(folder, 'best_model_sofar'),
                 os.path.join(folder, 'diffusion_folder', 'model_save', 'best_model_sofar')):
        if os.path.exists(path):
            return path
    raise SystemExit(f'no best_model_sofar in {folder} or {folder}/diffusion_folder/model_save; '
                     f'pass --checkpoint')


def sdflow_cond(cfg):
    """Measured rows, same arithmetic as main_diff_eval_bfs.py; a window past
    360 deg (angle_end > 360) wraps around the circle."""
    step = 360 / cfg['sample_H']
    rows = range(int(cfg['angle_start'] / step), int(cfg['angle_end'] / step), cfg['angle_stride'])
    return [r % cfg['sample_H'] for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)   # --angle_* here only pick the baselines' test slices
    parser.add_argument('--config', required=True,
                        help="SD-Flow config JSON: model folder (bfs_dynamic_folder), measured views, sampling steps")
    parser.add_argument('--checkpoint', default=None, help='default: best_model_sofar of the config\'s model folder')
    parser.add_argument('--name', default='sdflow', help='method folder, e.g. sdflow_physics for a second variant')
    parser.add_argument('--num_sample_steps', type=int, default=None, help='default: the config (20 if absent)')
    parser.add_argument('--unet_dim', type=int, default=None, help='default: the config (32 if absent)')
    parser.add_argument('--batch_size', type=int, default=None, help='default: the config (1 if absent)')
    parser.add_argument('--split', default='test')
    parser.add_argument('--max_slices', type=int, default=0, help='only the first N slices (0 = all)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    cfg.setdefault('num_sample_steps', 20)
    cfg.setdefault('unet_dim', 32)
    cfg.setdefault('batch_size', 1)
    for key in ('num_sample_steps', 'unet_dim', 'batch_size'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cond = sdflow_cond(cfg)
    ckpt = args.checkpoint or find_checkpoint(cfg['bfs_dynamic_folder'])

    out = method_dir(args, args.name)
    log = Logger(os.path.join(out, 'log_sample.txt'))
    run = {'config': args.config, 'checkpoint': ckpt, 'cond_indices': cond,
           'num_sample_steps': cfg['num_sample_steps'], 'unet_dim': cfg['unet_dim']}
    views_path = os.path.join(out, 'views.json')
    if os.path.exists(views_path):
        with open(views_path) as f:
            before = json.load(f)
        if before != run:
            raise SystemExit(f'{out} holds results of another SD-Flow run ({before}); pass a different '
                             f'--name or delete that folder')
    with open(views_path, 'w') as f:
        json.dump(run, f, indent=1)

    data = SplitData(setting_dir(args), args.split)
    recon_dir = os.path.join(out, 'recon' if args.split == 'test' else f'recon_{args.split}')
    sino_dir = os.path.join(out, 'sino' if args.split == 'test' else f'sino_{args.split}')
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(sino_dir, exist_ok=True)
    n = min(args.max_slices, len(data)) if args.max_slices else len(data)
    todo = [i for i in range(n) if not os.path.exists(os.path.join(recon_dir, data.slices[i]['id'] + '.npy'))]
    log(f'sdflow: {len(todo)} of {n} slices to do, {len(cond)} measured rows {cond[0]}..{cond[-1]} '
        f'step {cond[1] - cond[0] if len(cond) > 1 else 0}, {cfg["num_sample_steps"]} steps, checkpoint {ckpt}')
    if not todo:
        return

    device = torch.device(pick_device(args.device))
    modules = sdflow_modules()
    reproject = modules[3]
    trainer = build_trainer(modules, cfg, device)
    trainer.load(path=ckpt)
    torch.manual_seed(args.seed)

    bs = cfg['batch_size']
    with torch.no_grad():
        for start in range(0, len(todo), bs):
            idx = todo[start:start + bs]
            t0 = time.time()
            sinos = np.stack([data.sinogram(i) for i in idx])               # [B, H, W] in [-1, 1]
            H, W = sinos.shape[1:]
            reproj = np.stack([reproject(s, cond, det_count=W, num_angles=H) for s in sinos])
            # [B, T=1, C=1, H, W], as dicom_dataset batches reach test_diff.py
            full = torch.from_numpy(sinos).to(device)[:, None, None]
            masked, mask = torch.zeros_like(full), torch.zeros_like(full)
            masked[..., cond, :] = full[..., cond, :]
            mask[..., cond, :] = 1.0
            rep = torch.from_numpy(reproj).to(device)[:, None, None]
            cond_images = torch.cat([masked, mask, rep], dim=2).permute(0, 2, 1, 3, 4)   # [B, 3, 1, H, W]
            sampled = trainer.sample(video_frames=1, cond_images=cond_images, batch_size=len(idx))
            sampled = sampled.detach().float().cpu().numpy().reshape(len(idx), H, W)
            for k, i in enumerate(idx):
                sid = data.slices[i]['id']
                np.save(os.path.join(sino_dir, sid + '.npy'), sampled[k].astype(np.float32))
                np.save(os.path.join(recon_dir, sid + '.npy'), full_fbp(sampled[k]).astype(np.float32))
            log(f'sdflow: {start + len(idx)}/{len(todo)} slices ({(time.time() - t0) / len(idx):.1f} s per slice)')
    log('sdflow: done')


if __name__ == '__main__':
    main()
