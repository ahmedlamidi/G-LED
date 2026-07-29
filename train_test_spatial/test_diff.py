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


def _sync(device):
    """Make sure queued GPU work is finished before reading the clock."""
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()


def test_final_overall(args_final,
                       args_seq,
                       args_diff,
                       trainer,
                       seq_model,
                       data_loader,
                       cond_indices=None):
    test_final(args_final,
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

    contour_dir = os.path.join(args_final.experiment_path, 'contour')
    os.makedirs(contour_dir, exist_ok=True)

    vf = 1  # video frames per chunk
    Nvf = args_final.test_Nt // vf

    sample_times = []   # seconds per trainer.sample() call
    batch_times = []    # seconds of sampling per batch
    timing_log = os.path.join(args_final.experiment_path, 'inference_timing.txt')
    with open(timing_log, 'w') as f:
        f.write('Inference timing log\n')
        f.write(f'device: {args_final.device}, batches: {len(data_loader)}, '
                f'chunks per batch: {Nvf}, frames per chunk: {vf}\n\n')
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
            print(f"Original batch shape: {batch.shape}")
            b_size = batch.shape[0]
            assert b_size == 1
            H, W = batch.shape[-2], batch.shape[-1]

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

            # Create output directory
            seq_name = 'batch' + str(iteration)
            batch_dir = os.path.join(contour_dir, seq_name)
            os.makedirs(batch_dir, exist_ok=True)

            # Sample in chunks of vf frames
            recon_micro = []
            batch_sample_time = 0.0
            for j in range(Nvf):
                cond_chunk = batch_cond_perm[:, :, vf*j:vf*(j+1), :, :]
                print(f"cond_chunk shape: {cond_chunk.shape}, video_frames: {vf}")

                _sync(args_final.device)
                t_start = time.perf_counter()
                sampled = trainer.sample(
                    video_frames=vf,
                    cond_images=cond_chunk
                )
                _sync(args_final.device)
                t_sample = time.perf_counter() - t_start

                sample_times.append(t_sample)
                batch_sample_time += t_sample

                save_arr = sampled.detach().cpu().numpy()

                np.save(os.path.join(batch_dir, f"recon_micro_{j}.npy"), save_arr)
                np.save(os.path.join(batch_dir, f"recon_micro_{j}gt.npy"),
                        cond_chunk.detach().cpu().numpy())
                recon_micro.append(save_arr)

            # Save ground truth full sinogram
            full_sino_np = batch[0, 0, 0].cpu().numpy()  # [H, W]
            np.save(os.path.join(batch_dir, "ground_truth_full.npy"), full_sino_np)

            # Save condition info
            masked_sino_np = masked_sino[0, 0, 0].cpu().numpy()
            mask_np = mask[0, 0, 0].cpu().numpy()
            np.save(os.path.join(batch_dir, "masked_sinogram.npy"), masked_sino_np)
            np.save(os.path.join(batch_dir, "mask.npy"), mask_np)
            np.save(os.path.join(batch_dir, "cond_indices.npy"), np.array(cond_indices))

            # Concatenate reconstruction chunks
            recon_all = np.concatenate(recon_micro, axis=2)
            recon_2d = recon_all[0, 0, 0]  # [H, W] - full sinogram reconstruction

            # Compute MSE (full sinogram)
            mse = np.mean((recon_2d - full_sino_np) ** 2)

            # Compute MSE on unknown rows only
            all_rows = set(range(H))
            unknown_rows = sorted(all_rows - set(cond_indices))
            mse_unknown = np.mean((recon_2d[unknown_rows, :] - full_sino_np[unknown_rows, :]) ** 2)
            mse_known = np.mean((recon_2d[cond_indices, :] - full_sino_np[cond_indices, :]) ** 2)

            batch_times.append(batch_sample_time)
            print(f"  {seq_name}: MSE_full={mse:.6f}, MSE_unknown={mse_unknown:.6f}, MSE_known={mse_known:.6f}")

            # Append this batch's inference time to the timing log
            with open(timing_log, 'a') as f:
                f.write(f'{seq_name}: inference {batch_sample_time:.3f} s '
                        f'({batch_sample_time/max(Nvf,1):.3f} s per chunk)\n')

            # Visualization
            if save_flag:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                axes[0].imshow(full_sino_np, cmap='gray', aspect='auto')
                axes[0].set_title('Ground Truth (Full)')
                axes[0].axis('off')

                axes[1].imshow(masked_sino_np, cmap='gray', aspect='auto')
                axes[1].set_title(f'Masked Condition ({len(cond_indices)} known rows)')
                axes[1].axis('off')

                axes[2].imshow(recon_2d, cmap='gray', aspect='auto')
                axes[2].set_title(f'Reconstruction (MSE={mse:.4f})')
                axes[2].axis('off')

                axes[3].imshow(np.abs(recon_2d - full_sino_np), cmap='hot', aspect='auto')
                axes[3].set_title(f'Error (unknown MSE={mse_unknown:.4f})')
                axes[3].axis('off')

                plt.tight_layout()
                fig.savefig(os.path.join(batch_dir, 'overview.png'),
                            bbox_inches='tight', dpi=150)
                plt.close(fig)

    total_wall = time.perf_counter() - eval_start
    total_sample = sum(sample_times)
    frac = 100 * total_sample / total_wall if total_wall > 0 else 0.0

    with open(timing_log, 'a') as f:
        f.write('\n===== Timing summary =====\n')
        f.write(f'Batches evaluated       : {len(batch_times)}\n')
        f.write(f'Total wall-clock        : {total_wall:.2f} s ({total_wall/60:.2f} min)\n')
        f.write(f'Total inference (sample): {total_sample:.2f} s ({frac:.1f}% of wall-clock)\n')
        if batch_times:
            f.write(f'Inference per batch     : mean {np.mean(batch_times):.3f} s, '
                    f'min {np.min(batch_times):.3f} s, max {np.max(batch_times):.3f} s\n')
        if sample_times:
            f.write(f'Inference per sample()  : mean {np.mean(sample_times):.3f} s, '
                    f'min {np.min(sample_times):.3f} s, max {np.max(sample_times):.3f} s '
                    f'over {len(sample_times)} calls\n')

    print(f"\nDone! Results saved to {contour_dir}")
    print(f"Timing written to {timing_log}")
