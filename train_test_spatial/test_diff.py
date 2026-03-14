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

    loss_func = torch.nn.MSELoss()
    vf = 1  # video frames per chunk
    Nvf = args_final.test_Nt // vf

    all_ssim = []
    all_psnr = []

    with torch.no_grad():
        print('total iterations:', len(data_loader))
        for iteration, batch in tqdm(enumerate(data_loader)):
            batch = batch.to(args_final.device).float()
            print(f"Original batch shape: {batch.shape}")
            b_size = batch.shape[0]
            assert b_size == 1
            num_time = batch.shape[1]
            num_velocity = batch.shape[2] if batch.ndim == 5 else 1
            H, W = batch.shape[-2], batch.shape[-1]

            # Same conditioning split as train_diff
            cutoff = int(H * 0.75)
            batch_cond_indices = [i for i in range(H) if i % 4 == 0 and i < cutoff]
            label_indices = [i for i in range(H) if not (i % 4 == 0 and i < cutoff)]
            original_pred_h = len(label_indices)
            original_cond_h = len(batch_cond_indices)

            batch_cond = batch[..., batch_cond_indices, :]   # condition rows
            batch_target = batch[..., label_indices, :]      # ground truth target rows

            # Pad height to multiple of 16 (same as train_diff)
            def pad_to_16(tensor):
                h = tensor.shape[-2]
                pad_h = (16 - h % 16) % 16
                if pad_h > 0:
                    B, T, C, Hp, Wp = tensor.shape
                    tensor = tensor.reshape(B * T, C, Hp, Wp)
                    tensor = F.pad(tensor, (0, 0, 0, pad_h), mode='replicate')
                    tensor = tensor.reshape(B, T, C, Hp + pad_h, Wp)
                return tensor, pad_h

            batch_cond_padded, cond_pad = pad_to_16(batch_cond)
            batch_target_padded, pred_pad = pad_to_16(batch_target)

            # Permute to [B, C, T, H, W] for diffusion model
            batch_cond_perm = batch_cond_padded.permute(0, 2, 1, 3, 4)

            # Create output directory for this batch
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

                # Save reconstruction chunk
                np.save(os.path.join(batch_dir, f"recon_micro_{j}.npy"), save_arr)
                # Save condition chunk
                np.save(os.path.join(batch_dir, f"recon_micro_{j}gt.npy"),
                        cond_chunk.detach().cpu().numpy())
                recon_micro.append(save_arr)

            # Save ground truth target (label rows, unpadded)
            gt_target_np = batch_target[0, 0, 0].cpu().numpy()  # [H_label, W]
            np.save(os.path.join(batch_dir, "ground_truth_target.npy"), gt_target_np)

            # Save ground truth condition (cond rows, unpadded)
            gt_cond_np = batch_cond[0, 0, 0].cpu().numpy()  # [H_cond, W]
            np.save(os.path.join(batch_dir, "ground_truth_cond.npy"), gt_cond_np)

            # Save full original sinogram for reference
            full_sino_np = batch[0, 0, 0].cpu().numpy()  # [H, W]
            np.save(os.path.join(batch_dir, "ground_truth_full.npy"), full_sino_np)

            # Save index mappings for reconstruction
            np.save(os.path.join(batch_dir, "cond_indices.npy"), np.array(batch_cond_indices))
            np.save(os.path.join(batch_dir, "label_indices.npy"), np.array(label_indices))

            # Concatenate all reconstruction chunks and crop padding
            recon_all = np.concatenate(recon_micro, axis=2)  # along time dim
            # Squeeze to 2D: [B, C, T, H_padded, W] -> [H_padded, W]
            recon_2d = recon_all[0, 0, 0]
            # Crop padding to match original prediction height
            recon_2d = recon_2d[:original_pred_h, :]

            # Compute MSE loss
            mse = np.mean((recon_2d - gt_target_np) ** 2)
            print(f"  {seq_name}: MSE = {mse:.6f}")

            # Quick visualization
            if save_flag:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                norm = matplotlib.colors.Normalize(
                    vmin=full_sino_np.min(), vmax=full_sino_np.max())

                axes[0].imshow(full_sino_np, cmap='gray', aspect='auto')
                axes[0].set_title('Full Sinogram (GT)')
                axes[0].axis('off')

                axes[1].imshow(gt_cond_np, cmap='gray', aspect='auto')
                axes[1].set_title(f'Condition ({len(batch_cond_indices)} rows)')
                axes[1].axis('off')

                axes[2].imshow(gt_target_np, cmap='gray', aspect='auto')
                axes[2].set_title(f'GT Target ({original_pred_h} rows)')
                axes[2].axis('off')

                axes[3].imshow(recon_2d, cmap='gray', aspect='auto')
                axes[3].set_title(f'Reconstruction (MSE={mse:.4f})')
                axes[3].axis('off')

                plt.tight_layout()
                fig.savefig(os.path.join(batch_dir, 'overview.png'),
                            bbox_inches='tight', dpi=150)
                plt.close(fig)

    print(f"\nDone! Results saved to {contour_dir}")
