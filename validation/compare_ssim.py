import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

# Base folder containing all batch folders (generated outputs)
base_folder = 'output/feb_19_feather_model_720_820/diffusion_folder/experiment_final_checkpoint_200/contour'

# Ground truth folder
ground_truth_folder = 'output/feb_19_feather_model_720_820/ground_truth'

# Output folder for comparison images
comparison_output_folder = 'output/feb_19_feather_model_720_820/diffusion_folder/experiment_final_checkpoint_200/comparison'
os.makedirs(comparison_output_folder, exist_ok=True)

# Get all batch folders and sort them
batch_folders = [f for f in os.listdir(base_folder) if f.startswith('batch') and os.path.isdir(os.path.join(base_folder, f))]
batch_folders.sort(key=lambda x: int(x.replace('batch', '')))

print(f"Found {len(batch_folders)} batch folders")

# Collect metrics for saving
metrics_list = []

for batch_name in batch_folders:
    output_folder = os.path.join(base_folder, batch_name)
    batch_num = batch_name.replace('batch', '')
    
    try:
        # Load the generated reconstruction
        img_generated = np.load(os.path.join(output_folder, 'recon_micro_0.npy'))
        
        # Squeeze to 2D
        if img_generated.ndim == 5:
            img_generated = img_generated[0, 0, 0]
        elif img_generated.ndim == 4:
            img_generated = img_generated[0, 0]
        elif img_generated.ndim == 3:
            img_generated = img_generated[0]
        
        # Load the ground truth
        gt_path = os.path.join(ground_truth_folder, f'batch{batch_num}.npy')
        img_ground_truth = np.load(gt_path)
        
        # Squeeze to 2D
        if img_ground_truth.ndim == 5:
            img_ground_truth = img_ground_truth[0, 0, 0]
        elif img_ground_truth.ndim == 4:
            img_ground_truth = img_ground_truth[0, 0]
        elif img_ground_truth.ndim == 3:
            img_ground_truth = img_ground_truth[0]
        
        # The model padded the target to width 448 using reflect padding.
        # Original target_width = int(original_W * 0.53).
        # Derive original_W from the ground truth full width, then compute target_width.
        gt_W = img_ground_truth.shape[1]
        original_W = gt_W  # ground truth is the original full-width image
        cond_width = int(original_W * 0.50)   # First 50%
        target_width = int(original_W * 0.53)  # e.g. int(816 * 0.53) = 432
        
        # Remove reflect padding from generated: crop right side back to target_width
        img_generated = img_generated[:, :target_width]
        
        # Slice ground truth: last 53% of the full width (mirrors training: batch[..., (W - target_width):])
        gt_right = img_ground_truth[:, (gt_W - target_width):]            # last 53% = columns 384..815
        
        # Condition: first 50% of ground truth (mirrors training: batch[..., :cond_width])
        gt_cond = img_ground_truth[:, :cond_width]
        
        print(f"{batch_name}: generated shape={img_generated.shape}, gt_right shape={gt_right.shape}, gt_cond shape={gt_cond.shape}")
        
        # Normalize to [0, 1]
        def normalize(img):
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                return (img - img_min) / (img_max - img_min)
            return img
        
        img_gen_norm = normalize(img_generated)
        gt_right_norm = normalize(gt_right)
        gt_cond_norm = normalize(gt_cond)
        
        # Calculate SSIM and PSNR (unchanged)
        ssim_value = ssim(gt_right_norm, img_gen_norm, data_range=1.0)
        psnr_value = psnr(gt_right_norm, img_gen_norm, data_range=1.0)
        
        # --- Proper feathering in the overlap region ---
        # cond covers columns 0..cond_width-1 (408px)
        # generated covers columns (gt_W - target_width)..gt_W-1 (384..815 = 432px)
        # overlap = cond_width - (gt_W - target_width) = 408 - 384 = 24px
        overlap_start = gt_W - target_width   # column 384 in original space
        overlap_end   = cond_width            # column 408 in original space
        overlap_px    = overlap_end - overlap_start  # 24px
        
        # Build output canvas the full original width
        combined_generated = np.zeros((img_gen_norm.shape[0], gt_W), dtype=np.float32)
        
        # Left non-overlap region from condition
        combined_generated[:, :overlap_start] = gt_cond_norm[:, :overlap_start]
        
        # Overlap region: sigmoid blend between cond and gen
        if overlap_px > 0:
            x = np.linspace(0, 1, overlap_px)
            alpha = 1.0 / (1.0 + np.exp(-10 * (x - 0.5)))  # 0→1 sigmoid
            alpha = alpha[np.newaxis, :]  # broadcast over rows
            cond_overlap = gt_cond_norm[:, overlap_start:overlap_end]
            gen_overlap  = img_gen_norm[:, :overlap_px]
            combined_generated[:, overlap_start:overlap_end] = (1 - alpha) * cond_overlap + alpha * gen_overlap
        
        # Right non-overlap region from generated
        combined_generated[:, overlap_end:] = img_gen_norm[:, overlap_px:]
        
        # Ground truth: full image normalized, no blending
        gt_full_norm = normalize(img_ground_truth)
        
        # Plot feathered generated vs full ground truth
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(gt_full_norm, cmap='gray')
        axes[0].set_title('Ground Truth (full)')
        axes[0].axis('off')
        
        axes[1].imshow(combined_generated, cmap='gray')
        axes[1].set_title('Condition + Generated (feathered)')
        axes[1].axis('off')
        
        fig.suptitle(f'SSIM: {ssim_value:.4f} | PSNR: {psnr_value:.2f} dB', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(comparison_output_folder, f'{batch_name}_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        metrics_list.append({'batch': batch_name, 'ssim': ssim_value, 'psnr': psnr_value})
        print(f"{batch_name}: SSIM = {ssim_value:.4f}, PSNR = {psnr_value:.2f} dB -> Saved to {output_path}")
        
    except FileNotFoundError as e:
        print(f"Error processing {batch_name}: File not found - {e}")
    except Exception as e:
        print(f"Error processing {batch_name}: {e}")

# Save metrics to file
metrics_path = os.path.join(comparison_output_folder, 'metrics.txt')
with open(metrics_path, 'w') as f:
    f.write("Batch\tSSIM\tPSNR (dB)\n")
    f.write("-" * 40 + "\n")
    for m in metrics_list:
        f.write(f"{m['batch']}\t{m['ssim']:.4f}\t{m['psnr']:.2f}\n")
    
    # Calculate and write summary statistics
    if metrics_list:
        ssim_values = [m['ssim'] for m in metrics_list]
        psnr_values = [m['psnr'] for m in metrics_list]
        f.write("-" * 40 + "\n")
        f.write(f"Mean SSIM:\t{np.mean(ssim_values):.4f}\n")
        f.write(f"Std SSIM:\t{np.std(ssim_values):.4f}\n")
        f.write(f"Mean PSNR:\t{np.mean(psnr_values):.2f} dB\n")
        f.write(f"Std PSNR:\t{np.std(psnr_values):.2f} dB\n")

print(f"\nMetrics saved to {metrics_path}")
print(f"\nDone! Comparison images saved to {comparison_output_folder}")
