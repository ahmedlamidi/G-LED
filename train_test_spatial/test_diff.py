from tqdm import tqdm
import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F

sys.path.insert(0, './util')
from utils import save_loss


def test_final_overall(args_final,
                       args_seq,
                       args_diff,
                       trainer,
                       seq_model,
                       data_loader):
    test_final(args_final,
               args_seq,
               args_diff,
               trainer,
               seq_model,
               data_loader,
               None,
               None)


def test_final(args_final,
               args_seq,
               args_diff,
               trainer,
               model,
               data_loader,
               down_sampler,
               up_sampler,
               save_flag=True):
    contour_dir = os.path.join(args_final.experiment_path, 'contour')
    os.makedirs(contour_dir, exist_ok=True)

    vf = 1  # video frames per chunk
    Nvf = args_final.test_Nt // vf

    with torch.no_grad():
        print('total iterations:', len(data_loader))
        for iteration, batch in tqdm(enumerate(data_loader)):
            batch = batch.to(args_final.device).float()
            print(f"Original batch shape: {batch.shape}")
            b_size = batch.shape[0]
            assert b_size == 1
            H, W = batch.shape[-2], batch.shape[-1]

            # Same conditioning split as train_diff
            cutoff = int(H * 0.75)
            cond_indices = [i for i in range(H) if i % 4 == 0 and i < cutoff]

            # Build mask-based conditioning (same as train_diff)
            # Channel 0: masked sinogram, Channel 1: binary mask
            masked_sino = torch.zeros_like(batch)
            mask = torch.zeros_like(batch)
            masked_sino[..., cond_indices, :] = batch[..., cond_indices, :]
            mask[..., cond_indices, :] = 1.0
            batch_cond = torch.cat([masked_sino, mask], dim=2)  # [B, T, 2, H, W]

            # Permute to [B, C, T, H, W]
            batch_cond_perm = batch_cond.permute(0, 2, 1, 3, 4)  # [B, 2, T, H, W]

            # Create output directory
            seq_name = 'batch' + str(iteration)
            batch_dir = os.path.join(contour_dir, seq_name)
            os.makedirs(batch_dir, exist_ok=True)

            # Sample in chunks of vf frames
            recon_micro = []
            for j in range(Nvf):
                cond_chunk = batch_cond_perm[:, :, vf*j:vf*(j+1), :, :]
                print(f"cond_chunk shape: {cond_chunk.shape}, video_frames: {vf}")
                save_arr = trainer.sample(
                    video_frames=vf,
                    cond_images=cond_chunk
                ).detach().cpu().numpy()

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

            print(f"  {seq_name}: MSE_full={mse:.6f}, MSE_unknown={mse_unknown:.6f}, MSE_known={mse_known:.6f}")

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

    print(f"\nDone! Results saved to {contour_dir}")
