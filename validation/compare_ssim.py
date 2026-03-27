import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import argparse
import astra


# --------------- ASTRA FBP helper ---------------

def fbp_reconstruct_from_sinogram(sinogram, det_count=816, num_angles=720,
                                   DSO=1000, ODD=600, N_pix=512):
    """
    FBP reconstruction from a sinogram (any normalization).
    Uses the same fan-beam geometry as dicom_preprocess.
    Returns a 2D CT reconstruction [N_pix, N_pix].
    """
    sinogram = np.ascontiguousarray(sinogram, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False).astype(np.float32)

    vol_geom = astra.create_vol_geom(N_pix, N_pix)
    proj_geom = astra.create_proj_geom('fanflat', 1.0, det_count, angles, DSO, ODD)

    sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom)

    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    recon = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)
    return recon

parser = argparse.ArgumentParser()
parser.add_argument("--base_folder",
                    default='output/mar_11_horizontal_PIML_5/diffusion_folder/experiment_final_checkpoint_150/contour',
                    help='Folder containing batch outputs from test_diff')
args = parser.parse_args()

base_folder = args.base_folder

# Output folder for comparison images
comparison_output_folder = os.path.join(os.path.dirname(base_folder), 'comparisons')
os.makedirs(comparison_output_folder, exist_ok=True)

# Get all batch folders and sort them
batch_folders = [f for f in os.listdir(base_folder)
                 if f.startswith('batch') and os.path.isdir(os.path.join(base_folder, f))]
batch_folders.sort(key=lambda x: int(x.replace('batch', '')))

print(f"Found {len(batch_folders)} batch folders")

# Collect metrics for saving
metrics_list = []

for batch_name in batch_folders:
    output_folder = os.path.join(base_folder, batch_name)

    try:
        # Load the generated full-sinogram reconstruction
        img_generated = np.load(os.path.join(output_folder, 'recon_micro_0.npy'))

        # Load ground truth full sinogram (saved by test_diff)
        gt_full = np.load(os.path.join(output_folder, 'ground_truth_full.npy'))  # [H, W]

        # Load condition indices and derive unknown (label) indices
        cond_indices = np.load(os.path.join(output_folder, 'cond_indices.npy'))
        H = gt_full.shape[0]
        label_indices = sorted(set(range(H)) - set(cond_indices))

        # Squeeze generated to 2D
        if img_generated.ndim == 5:
            img_generated = img_generated[0, 0, 0]
        elif img_generated.ndim == 4:
            img_generated = img_generated[0, 0]
        elif img_generated.ndim == 3:
            img_generated = img_generated[0]

        # Crop to match GT dimensions (in case of padding)
        img_generated = img_generated[:gt_full.shape[0], :gt_full.shape[1]]

        print(f"{batch_name}: generated={img_generated.shape}, gt_full={gt_full.shape}, "
              f"cond_rows={len(cond_indices)}, unknown_rows={len(label_indices)}")

        # Normalize all images to [0, 1] using shared min/max from full GT
        # so that value correspondence is preserved across GT and reconstruction
        global_min = gt_full.min()
        global_max = gt_full.max()
        def normalize(img):
            if global_max > global_min:
                return (img - global_min) / (global_max - global_min)
            return img

        img_gen_norm = normalize(img_generated)
        gt_full_norm = normalize(gt_full)

        # Full-sinogram metrics
        ssim_full = ssim(gt_full_norm, img_gen_norm, data_range=1.0)
        psnr_full = psnr(gt_full_norm, img_gen_norm, data_range=1.0)
        mse_full = np.mean((gt_full_norm - img_gen_norm) ** 2)

        # Unknown-rows-only metrics (the rows the model had to infer)
        gt_target_norm = gt_full_norm[label_indices, :]
        img_gen_target_norm = img_gen_norm[label_indices, :]
        ssim_target = ssim(gt_target_norm, img_gen_target_norm, data_range=1.0)
        psnr_target = psnr(gt_target_norm, img_gen_target_norm, data_range=1.0)
        mse_target = np.mean((gt_target_norm - img_gen_target_norm) ** 2)

        # Known-rows MSE (sanity check — should be near-zero)
        gt_known_norm = gt_full_norm[cond_indices, :]
        gen_known_norm = img_gen_norm[cond_indices, :]
        mse_known = np.mean((gt_known_norm - gen_known_norm) ** 2)

        # --- FBP reconstruction: sinogram -> CT image ---
        det_count = gt_full.shape[1]
        num_angles = gt_full.shape[0]

        recon_gt = fbp_reconstruct_from_sinogram(gt_full, det_count=det_count, num_angles=num_angles)
        recon_pred = fbp_reconstruct_from_sinogram(img_generated, det_count=det_count, num_angles=num_angles)

        # Normalize FBP reconstructions for metrics (using GT range)
        fbp_min = recon_gt.min()
        fbp_max = recon_gt.max()
        if fbp_max > fbp_min:
            recon_gt_norm = (recon_gt - fbp_min) / (fbp_max - fbp_min)
            recon_pred_norm = (recon_pred - fbp_min) / (fbp_max - fbp_min)
        else:
            recon_gt_norm = recon_gt.copy()
            recon_pred_norm = recon_pred.copy()

        ssim_ct = ssim(recon_gt_norm, recon_pred_norm, data_range=1.0)
        psnr_ct = psnr(recon_gt_norm, recon_pred_norm, data_range=1.0)
        mse_ct = np.mean((recon_gt_norm - recon_pred_norm) ** 2)

        # Create comparison figure (3 rows x 3 cols)
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # Row 1: Full sinograms
        axes[0, 0].imshow(gt_full_norm, cmap='gray', aspect='auto')
        axes[0, 0].set_title('Ground Truth (Full)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(img_gen_norm, cmap='gray', aspect='auto')
        axes[0, 1].set_title(f'Reconstructed (Full)\nSSIM={ssim_full:.4f}, PSNR={psnr_full:.2f}dB')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(np.abs(gt_full_norm - img_gen_norm), cmap='hot', aspect='auto')
        axes[0, 2].set_title(f'Absolute Error (Full)\nMSE={mse_full:.6f}')
        axes[0, 2].axis('off')

        # Row 2: Unknown (target) rows only
        axes[1, 0].imshow(gt_target_norm, cmap='gray', aspect='auto')
        axes[1, 0].set_title(f'GT Target ({len(label_indices)} unknown rows)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(img_gen_target_norm, cmap='gray', aspect='auto')
        axes[1, 1].set_title(f'Generated Target\nSSIM={ssim_target:.4f}, PSNR={psnr_target:.2f}dB')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(np.abs(gt_target_norm - img_gen_target_norm), cmap='hot', aspect='auto')
        axes[1, 2].set_title(f'Absolute Error (Target)\nMSE={mse_target:.6f}')
        axes[1, 2].axis('off')

        # Row 3: FBP CT reconstructions
        axes[2, 0].imshow(recon_gt_norm, cmap='gray')
        axes[2, 0].set_title('FBP from GT Sinogram')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(recon_pred_norm, cmap='gray')
        axes[2, 1].set_title(f'FBP from Predicted\nSSIM={ssim_ct:.4f}, PSNR={psnr_ct:.2f}dB')
        axes[2, 1].axis('off')

        axes[2, 2].imshow(np.abs(recon_gt_norm - recon_pred_norm), cmap='hot')
        axes[2, 2].set_title(f'FBP Absolute Error\nMSE={mse_ct:.6f}')
        axes[2, 2].axis('off')

        fig.suptitle(f'{batch_name} | Sino SSIM: {ssim_full:.4f} | Target SSIM: {ssim_target:.4f} | '
                     f'CT SSIM: {ssim_ct:.4f} | Known MSE: {mse_known:.6f}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(comparison_output_folder, f'{batch_name}_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Store metrics
        metrics_list.append({
            'batch': batch_name,
            'ssim_full': ssim_full,
            'psnr_full': psnr_full,
            'mse_full': mse_full,
            'ssim_target': ssim_target,
            'psnr_target': psnr_target,
            'mse_target': mse_target,
            'mse_known': mse_known,
            'ssim_ct': ssim_ct,
            'psnr_ct': psnr_ct,
            'mse_ct': mse_ct,
        })

        print(f"  Full SSIM={ssim_full:.4f}, PSNR={psnr_full:.2f}dB | "
              f"Target SSIM={ssim_target:.4f}, PSNR={psnr_target:.2f}dB | "
              f"CT SSIM={ssim_ct:.4f}, PSNR={psnr_ct:.2f}dB | "
              f"Known MSE={mse_known:.6f} -> {output_path}")

    except FileNotFoundError as e:
        print(f"Error processing {batch_name}: File not found - {e}")
    except Exception as e:
        print(f"Error processing {batch_name}: {e}")
        import traceback
        traceback.print_exc()

# Save metrics to file
metrics_path = os.path.join(comparison_output_folder, 'metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(f"{'Batch':<12} {'SSIM(full)':>11} {'PSNR(full)':>11} {'MSE(full)':>11} "
            f"{'SSIM(tgt)':>10} {'PSNR(tgt)':>10} {'MSE(tgt)':>10} {'MSE(known)':>11} "
            f"{'SSIM(CT)':>10} {'PSNR(CT)':>10} {'MSE(CT)':>10}\n")
    f.write("-" * 140 + "\n")
    for m in metrics_list:
        f.write(f"{m['batch']:<12} {m['ssim_full']:>11.4f} {m['psnr_full']:>11.2f} {m['mse_full']:>11.6f} "
                f"{m['ssim_target']:>10.4f} {m['psnr_target']:>10.2f} {m['mse_target']:>10.6f} "
                f"{m['mse_known']:>11.6f} "
                f"{m['ssim_ct']:>10.4f} {m['psnr_ct']:>10.2f} {m['mse_ct']:>10.6f}\n")

    # Summary statistics
    if metrics_list:
        f.write("-" * 140 + "\n")
        for key, label in [('ssim_full', 'SSIM (full)'),
                           ('psnr_full', 'PSNR (full)'),
                           ('mse_full', 'MSE (full)'),
                           ('ssim_target', 'SSIM (target)'),
                           ('psnr_target', 'PSNR (target)'),
                           ('mse_target', 'MSE (target)'),
                           ('mse_known', 'MSE (known)'),
                           ('ssim_ct', 'SSIM (CT)'),
                           ('psnr_ct', 'PSNR (CT)'),
                           ('mse_ct', 'MSE (CT)')]:
            vals = [m[key] for m in metrics_list]
            f.write(f"Mean {label}:\t{np.mean(vals):.4f}\n")
            f.write(f"Std  {label}:\t{np.std(vals):.4f}\n")

        print(f"\nSummary:")
        for key, label in [('ssim_full', 'SSIM (full sino)'),
                           ('ssim_target', 'SSIM (target sino)'),
                           ('ssim_ct', 'SSIM (CT recon)'),
                           ('psnr_full', 'PSNR (full sino)'),
                           ('psnr_ct', 'PSNR (CT recon)'),
                           ('mse_known', 'MSE (known)')]:
            vals = [m[key] for m in metrics_list]
            print(f"  {label}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

print(f"\nMetrics saved to {metrics_path}")
print(f"Comparison images saved to {comparison_output_folder}")
