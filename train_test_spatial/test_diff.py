from tqdm import tqdm
import torch
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F

sys.path.insert(0, './util')
from utils import save_loss

try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None


def _sync(device):
    """Make sure queued GPU work is finished before reading the clock."""
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()


def _ssim_metrics(recon_2d, gt_2d, cond_indices):
    """
    SSIM of a reconstructed sinogram against ground truth, on the full image and
    on the unknown rows only. Both images are normalized by the GT min/max so the
    value correspondence is preserved, matching validation/compare_ssim.py.
    """
    if structural_similarity is None:
        raise RuntimeError(
            "scikit-image is required for SSIM. Install it with: pip install scikit-image")

    gmin, gmax = float(gt_2d.min()), float(gt_2d.max())
    if gmax > gmin:
        recon_n = (recon_2d - gmin) / (gmax - gmin)
        gt_n = (gt_2d - gmin) / (gmax - gmin)
    else:
        recon_n, gt_n = recon_2d, gt_2d

    ssim_full = structural_similarity(gt_n, recon_n, data_range=1.0)

    unknown_rows = sorted(set(range(gt_2d.shape[0])) - set(cond_indices))
    # SSIM's default 7x7 window needs at least 7 rows to slide over
    if len(unknown_rows) >= 7:
        ssim_unknown = structural_similarity(gt_n[unknown_rows, :],
                                             recon_n[unknown_rows, :],
                                             data_range=1.0)
    else:
        ssim_unknown = float('nan')

    return ssim_full, ssim_unknown


def _save_overview(out_path, full_sino_np, masked_sino_np, recon_2d,
                   cond_indices, mse, mse_unknown,
                   ssim_full=None, ssim_unknown=None, suptitle=None):
    """Four-panel comparison: ground truth, masked condition, reconstruction, error."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(full_sino_np, cmap='gray', aspect='auto')
    axes[0].set_title('Ground Truth (Full)')
    axes[0].axis('off')

    axes[1].imshow(masked_sino_np, cmap='gray', aspect='auto')
    axes[1].set_title(f'Masked Condition ({len(cond_indices)} known rows)')
    axes[1].axis('off')

    recon_title = f'Reconstruction (MSE={mse:.4f}'
    if ssim_full is not None:
        recon_title += f', SSIM={ssim_full:.4f}'
    axes[2].imshow(recon_2d, cmap='gray', aspect='auto')
    axes[2].set_title(recon_title + ')')
    axes[2].axis('off')

    err_title = f'Error (unknown MSE={mse_unknown:.4f}'
    if ssim_unknown is not None:
        err_title += f', SSIM={ssim_unknown:.4f}'
    axes[3].imshow(np.abs(recon_2d - full_sino_np), cmap='hot', aspect='auto')
    axes[3].set_title(err_title + ')')
    axes[3].axis('off')

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def test_final_overall(args_final,
                       args_seq,
                       args_diff,
                       trainer,
                       seq_model,
                       data_loader,
                       cond_indices=None):
    return test_final(args_final,
                      args_seq,
                      args_diff,
                      trainer,
                      seq_model,
                      data_loader,
                      None,
                      None,
                      cond_indices=cond_indices)


def test_final(args_final,
               args_seq,
               args_diff,
               trainer,
               model,
               data_loader,
               down_sampler,
               up_sampler,
               save_flag=True,
               cond_indices=None):
    if cond_indices is None:
        raise ValueError("cond_indices must be provided — define it in main_diff_eval_bfs.py and pass it through")

    # Timing-only mode records inference time and SSIM, plus a single preview
    # PNG of the first slice — no per-slice .npy dumps or figures. Use it to
    # sweep batch_size / num_sample_steps without disk traffic skewing the
    # measurement, while still being able to eyeball what the run produced.
    timing_only = bool(getattr(args_final, 'timing_only', False))
    num_sample_steps = getattr(args_final, 'num_sample_steps', 'NA')
    batch_size = getattr(args_final, 'batch_size', 1)
    model_name = os.path.basename(os.path.normpath(args_final.bfs_dynamic_folder))

    # The report name is the sweep point, so runs don't overwrite each other.
    # The preview PNG shares the stem, so the images line up alongside it.
    run_stem = f'timing_bs{batch_size}_steps{num_sample_steps}_{model_name}'
    report_path = os.path.join(args_final.experiment_path, run_stem + '.txt')
    preview_path = os.path.join(args_final.experiment_path, run_stem + '.png')

    contour_dir = os.path.join(args_final.experiment_path, 'contour')
    if not timing_only:
        os.makedirs(contour_dir, exist_ok=True)

    vf = 1  # video frames per chunk
    Nvf = args_final.test_Nt // vf

    sample_times = []   # seconds per trainer.sample() call
    batch_times = []    # seconds of sampling per batch
    ssim_full_all = []
    ssim_unknown_all = []

    with open(report_path, 'w') as f:
        f.write(f'{"Timing-only run" if timing_only else "Evaluation run"}\n')
        f.write(f'model            : {model_name}\n')
        f.write(f'batch_size       : {batch_size}\n')
        f.write(f'num_sample_steps : {num_sample_steps}\n')
        f.write(f'device           : {args_final.device}\n')
        f.write(f'batches          : {len(data_loader)}, '
                f'chunks per batch: {Nvf}, frames per chunk: {vf}\n\n')
        f.write(f'{"batch":<16}{"n":>4}{"inference_s":>14}{"s_per_sample":>14}'
                f'{"ssim_full":>12}{"ssim_unknown":>14}\n')

    eval_start = time.perf_counter()

    with torch.no_grad():
        print('total iterations:', len(data_loader))
        for iteration, batch_data in tqdm(enumerate(data_loader)):
            # Unpack: batch [B, T, C, H, W], fbp_reproj [B, T, C, H, W]
            if isinstance(batch_data, (list, tuple)):
                batch, fbp_reproj = batch_data
                fbp_reproj = fbp_reproj.to(args_final.device).float()
            else:
                batch = batch_data
                fbp_reproj = None

            batch = batch.to(args_final.device).float()
            b_size = batch.shape[0]
            H, W = batch.shape[-2], batch.shape[-1]
            if not timing_only:
                print(f"Original batch shape: {batch.shape}")

            # Build mask-based conditioning (same as train_diff)
            # Channel 0: masked sinogram, Channel 1: binary mask, Channel 2: FBP re-projection
            masked_sino = torch.zeros_like(batch)
            mask = torch.zeros_like(batch)
            masked_sino[..., cond_indices, :] = batch[..., cond_indices, :]
            mask[..., cond_indices, :] = 1.0

            if fbp_reproj is not None:
                batch_cond = torch.cat([masked_sino, mask, fbp_reproj], dim=2)  # [B, T, 3, H, W]
            else:
                batch_cond = torch.cat([masked_sino, mask], dim=2)  # [B, T, 2, H, W]

            # Permute to [B, C, T, H, W]
            batch_cond_perm = batch_cond.permute(0, 2, 1, 3, 4)  # [B, 3, T, H, W]

            # Sample in chunks of vf frames
            recon_micro = []
            batch_sample_time = 0.0
            for j in range(Nvf):
                cond_chunk = batch_cond_perm[:, :, vf*j:vf*(j+1), :, :]

                _sync(args_final.device)
                t_start = time.perf_counter()
                # batch_size must be passed explicitly: the sampler builds its
                # noise tensor from this kwarg, not from cond_images' batch dim
                sampled = trainer.sample(
                    video_frames=vf,
                    cond_images=cond_chunk,
                    batch_size=cond_chunk.shape[0]
                )
                _sync(args_final.device)
                t_sample = time.perf_counter() - t_start

                sample_times.append(t_sample)
                batch_sample_time += t_sample

                save_arr = sampled.detach().cpu().numpy()
                recon_micro.append(save_arr)

            batch_times.append(batch_sample_time)

            # Concatenate reconstruction chunks -> [B, 1, T, H, W]
            recon_all = np.concatenate(recon_micro, axis=2)
            full_sino_batch = batch[:, 0, 0].cpu().numpy()      # [B, H, W]
            masked_sino_batch = masked_sino[:, 0, 0].cpu().numpy()
            mask_batch = mask[:, 0, 0].cpu().numpy()

            all_rows = set(range(H))
            unknown_rows = sorted(all_rows - set(cond_indices))

            batch_ssim_full = []
            batch_ssim_unknown = []

            for i in range(b_size):
                recon_2d = recon_all[i, 0, 0]        # [H, W]
                full_sino_np = full_sino_batch[i]

                ssim_full, ssim_unknown = _ssim_metrics(recon_2d, full_sino_np, cond_indices)
                batch_ssim_full.append(ssim_full)
                batch_ssim_unknown.append(ssim_unknown)
                ssim_full_all.append(ssim_full)
                ssim_unknown_all.append(ssim_unknown)

                # Global sample index so batched runs don't collide
                sample_idx = iteration * batch_size + i
                seq_name = 'batch' + str(sample_idx)

                if timing_only:
                    # One preview of the first slice only, so the sweep stays
                    # cheap but the reconstruction is still inspectable
                    if sample_idx == 0:
                        mse = np.mean((recon_2d - full_sino_np) ** 2)
                        mse_unknown = np.mean(
                            (recon_2d[unknown_rows, :] - full_sino_np[unknown_rows, :]) ** 2)
                        _save_overview(preview_path,
                                       full_sino_np,
                                       masked_sino_batch[i],
                                       recon_2d,
                                       cond_indices,
                                       mse,
                                       mse_unknown,
                                       ssim_full=ssim_full,
                                       ssim_unknown=ssim_unknown,
                                       suptitle=f'{model_name} | slice 0 | '
                                                f'steps={num_sample_steps} | bs={batch_size}')
                        print(f"  preview written to {preview_path}")
                    continue

                batch_dir = os.path.join(contour_dir, seq_name)
                os.makedirs(batch_dir, exist_ok=True)

                for j, chunk in enumerate(recon_micro):
                    np.save(os.path.join(batch_dir, f"recon_micro_{j}.npy"), chunk[i:i+1])
                    np.save(os.path.join(batch_dir, f"recon_micro_{j}gt.npy"),
                            batch_cond_perm[i:i+1, :, vf*j:vf*(j+1), :, :].detach().cpu().numpy())

                np.save(os.path.join(batch_dir, "ground_truth_full.npy"), full_sino_np)
                np.save(os.path.join(batch_dir, "masked_sinogram.npy"), masked_sino_batch[i])
                np.save(os.path.join(batch_dir, "mask.npy"), mask_batch[i])
                np.save(os.path.join(batch_dir, "cond_indices.npy"), np.array(cond_indices))

                mse = np.mean((recon_2d - full_sino_np) ** 2)
                mse_unknown = np.mean((recon_2d[unknown_rows, :] - full_sino_np[unknown_rows, :]) ** 2)
                mse_known = np.mean((recon_2d[cond_indices, :] - full_sino_np[cond_indices, :]) ** 2)

                print(f"  {seq_name}: MSE_full={mse:.6f}, MSE_unknown={mse_unknown:.6f}, "
                      f"MSE_known={mse_known:.6f}, SSIM_full={ssim_full:.4f}, "
                      f"SSIM_unknown={ssim_unknown:.4f}")

                # Visualization
                if save_flag:
                    _save_overview(os.path.join(batch_dir, 'overview.png'),
                                   full_sino_np,
                                   masked_sino_batch[i],
                                   recon_2d,
                                   cond_indices,
                                   mse,
                                   mse_unknown,
                                   ssim_full=ssim_full,
                                   ssim_unknown=ssim_unknown)

            with open(report_path, 'a') as f:
                f.write(f'{"batch" + str(iteration):<16}{b_size:>4}'
                        f'{batch_sample_time:>14.3f}{batch_sample_time/max(b_size,1):>14.3f}'
                        f'{np.mean(batch_ssim_full):>12.4f}{np.mean(batch_ssim_unknown):>14.4f}\n')

    total_wall = time.perf_counter() - eval_start
    total_sample = sum(sample_times)
    frac = 100 * total_sample / total_wall if total_wall > 0 else 0.0
    n_samples = len(ssim_full_all)

    with open(report_path, 'a') as f:
        f.write('\n===== Summary =====\n')
        f.write(f'Batches evaluated       : {len(batch_times)} ({n_samples} sinograms)\n')
        f.write(f'Total wall-clock        : {total_wall:.2f} s ({total_wall/60:.2f} min)\n')
        f.write(f'Total inference (sample): {total_sample:.2f} s ({frac:.1f}% of wall-clock)\n')
        if n_samples:
            f.write(f'Inference per sinogram  : {total_sample/n_samples:.3f} s\n')
        if batch_times:
            f.write(f'Inference per batch     : mean {np.mean(batch_times):.3f} s, '
                    f'min {np.min(batch_times):.3f} s, max {np.max(batch_times):.3f} s\n')
        if len(sample_times) > 1:
            # The first call pays CUDA context init and cuDNN autotuning; the
            # steady-state number is the one to compare across sweep points.
            warm = sample_times[1:]
            f.write(f'First sample() call     : {sample_times[0]:.3f} s (includes warm-up)\n')
            f.write(f'Steady-state sample()   : mean {np.mean(warm):.3f} s, '
                    f'min {np.min(warm):.3f} s, max {np.max(warm):.3f} s '
                    f'over {len(warm)} calls\n')
        elif sample_times:
            f.write(f'Single sample() call    : {sample_times[0]:.3f} s (includes warm-up)\n')
        if n_samples:
            f.write(f'SSIM full               : mean {np.mean(ssim_full_all):.4f}, '
                    f'min {np.min(ssim_full_all):.4f}, max {np.max(ssim_full_all):.4f}\n')
            f.write(f'SSIM unknown rows       : mean {np.nanmean(ssim_unknown_all):.4f}, '
                    f'min {np.nanmin(ssim_unknown_all):.4f}, max {np.nanmax(ssim_unknown_all):.4f}\n')

    if not timing_only:
        print(f"\nDone! Results saved to {contour_dir}")
    else:
        print(f"Preview written to {preview_path}")
    print(f"Report written to {report_path}")

    return report_path
