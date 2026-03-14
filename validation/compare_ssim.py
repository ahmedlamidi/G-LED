import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

# Base folder containing all batch folders (generated outputs from test_diff)
base_folder = 'output/mar_8_horizontal_step_4_75%/diffusion_folder/experiment_final_checkpoint_150/contour'

# Output folder for comparison images
comparison_output_folder = os.path.join(os.path.dirname(base_folder), 'comparisons')
os.makedirs(comparison_output_folder, exist_ok=True)

# Get all batch folders and sort them
batch_folders = [f for f in os.listdir(base_folder) if f.startswith('batch') and os.path.isdir(os.path.join(base_folder, f))]
batch_folders.sort(key=lambda x: int(x.replace('batch', '')))

print(f"Found {len(batch_folders)} batch folders")

# Collect metrics for saving
metrics_list = []

def normalize(img):
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img

for batch_name in batch_folders:
    output_folder = os.path.join(base_folder, batch_name)
    batch_num = batch_name.replace('batch', '')

    try:
        # Load the generated reconstruction
        img_generated = np.load(os.path.join(output_folder, 'recon_micro_0.npy'))
        # Load the condition
        img_cond = np.load(os.path.join(output_folder, 'recon_micro_0gt.npy'))

        # Squeeze to 2D
        if img_generated.ndim == 5:
            img_generated = img_generated[0, 0, 0]
        elif img_generated.ndim == 4:
            img_generated = img_generated[0, 0]
        elif img_generated.ndim == 3:
            img_generated = img_generated[0]

        if img_cond.ndim == 5:
            img_cond = img_cond[0, 0, 0]
        elif img_cond.ndim == 4:
            img_cond = img_cond[0, 0]
        elif img_cond.ndim == 3:
            img_cond = img_cond[0]

        # Load ground truth files saved by test_diff
        gt_target_path = os.path.join(output_folder, 'ground_truth_target.npy')
        gt_cond_path = os.path.join(output_folder, 'ground_truth_cond.npy')
        gt_full_path = os.path.join(output_folder, 'ground_truth_full.npy')
        cond_indices_path = os.path.join(output_folder, 'cond_indices.npy')
        label_indices_path = os.path.join(output_folder, 'label_indices.npy')

        gt_target = np.load(gt_target_path)          # [H_label, W]
        gt_cond   = np.load(gt_cond_path)             # [H_cond, W]
        img_ground_truth = np.load(gt_full_path)      # [H, W]
        cond_indices  = np.load(cond_indices_path)
        label_indices = np.load(label_indices_path)

        # Crop height padding from generated outputs to match GT dimensions
        img_generated = img_generated[:gt_target.shape[0], :]
        img_cond      = img_cond[:gt_cond.shape[0], :]

        print(f"{batch_name}: generated={img_generated.shape}, cond={img_cond.shape}, gt_target={gt_target.shape}")

        img_gen_norm   = normalize(img_generated)
        gt_target_norm = normalize(gt_target)
        gt_cond_norm   = normalize(gt_cond)
        img_cond_norm  = normalize(img_cond)

        # Calculate SSIM between generated and GT target rows
        ssim_value = ssim(gt_target_norm, img_gen_norm, data_range=1.0)

        # Calculate PSNR between generated and GT target rows
        psnr_value = psnr(gt_target_norm, img_gen_norm, data_range=1.0)

        # Calculate MSE
        mse_value = np.mean((gt_target_norm - img_gen_norm) ** 2)

        # Reconstruct full image by placing condition and label rows back
        full_H, full_W = img_ground_truth.shape
        recon_full = np.zeros((full_H, full_W))
        for idx, row in zip(cond_indices, range(len(cond_indices))):
            recon_full[idx, :] = img_cond_norm[row]
        for idx, row in zip(label_indices, range(len(label_indices))):
            recon_full[idx, :] = img_gen_norm[row]
        gt_full = normalize(img_ground_truth)

        # Full-image SSIM
        ssim_full = ssim(gt_full, recon_full, data_range=1.0)
        psnr_full = psnr(gt_full, recon_full, data_range=1.0)

        # Create 2x3 figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Row 1: Full images
        axes[0, 0].imshow(gt_full, cmap='gray', aspect='auto')
        axes[0, 0].set_title('Ground Truth (Full)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(recon_full, cmap='gray', aspect='auto')
        axes[0, 1].set_title(f'Reconstructed (Full)\nSSIM={ssim_full:.4f}, PSNR={psnr_full:.2f}dB')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(np.abs(gt_full - recon_full), cmap='hot', aspect='auto')
        axes[0, 2].set_title('Absolute Error (Full)')
        axes[0, 2].axis('off')

        # Row 2: Target rows only
        axes[1, 0].imshow(gt_target_norm, cmap='gray', aspect='auto')
        axes[1, 0].set_title(f'GT Target ({len(label_indices)} rows)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(img_gen_norm, cmap='gray', aspect='auto')
        axes[1, 1].set_title(f'Generated Target\nSSIM={ssim_value:.4f}, PSNR={psnr_value:.2f}dB')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(np.abs(gt_target_norm - img_gen_norm), cmap='hot', aspect='auto')
        axes[1, 2].set_title(f'Absolute Error\nMSE={mse_value:.6f}')
        axes[1, 2].axis('off')

        fig.suptitle(f'{batch_name} | Target SSIM: {ssim_value:.4f} | Full SSIM: {ssim_full:.4f}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save the comparison image
        output_path = os.path.join(comparison_output_folder, f'{batch_name}_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Store metrics
        metrics_list.append({
            'batch': batch_name,
            'ssim_target': ssim_value,
            'psnr_target': psnr_value,
            'mse_target': mse_value,
            'ssim_full': ssim_full,
            'psnr_full': psnr_full,
        })

        print(f"  Target SSIM={ssim_value:.4f}, PSNR={psnr_value:.2f}dB | "
              f"Full SSIM={ssim_full:.4f}, PSNR={psnr_full:.2f}dB -> {output_path}")

    except FileNotFoundError as e:
        print(f"Error processing {batch_name}: File not found - {e}")
    except Exception as e:
        print(f"Error processing {batch_name}: {e}")
        import traceback
        traceback.print_exc()

# Save metrics to file
metrics_path = os.path.join(comparison_output_folder, 'metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(f"{'Batch':<12} {'SSIM(target)':>13} {'PSNR(target)':>13} "
            f"{'MSE(target)':>13} {'SSIM(full)':>11} {'PSNR(full)':>11}\n")
    f.write("-" * 80 + "\n")
    for m in metrics_list:
        f.write(f"{m['batch']:<12} {m['ssim_target']:>13.4f} {m['psnr_target']:>13.2f} "
                f"{m['mse_target']:>13.6f} {m['ssim_full']:>11.4f} {m['psnr_full']:>11.2f}\n")

    # Summary statistics
    if metrics_list:
        f.write("-" * 80 + "\n")
        for key, label in [('ssim_target', 'SSIM (target)'),
                           ('psnr_target', 'PSNR (target)'),
                           ('mse_target', 'MSE (target)'),
                           ('ssim_full', 'SSIM (full)'),
                           ('psnr_full', 'PSNR (full)')]:
            vals = [m[key] for m in metrics_list]
            f.write(f"Mean {label}:\t{np.mean(vals):.4f}\n")
            f.write(f"Std  {label}:\t{np.std(vals):.4f}\n")

        print(f"\nSummary:")
        for key, label in [('ssim_target', 'SSIM (target)'),
                           ('psnr_target', 'PSNR (target)'),
                           ('ssim_full', 'SSIM (full)')]:
            vals = [m[key] for m in metrics_list]
            print(f"  {label}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

print(f"\nMetrics saved to {metrics_path}")
print(f"Comparison images saved to {comparison_output_folder}")
