import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

# Base folder containing all batch folders (generated outputs)
base_folder = 'output/mar_8_horizontal_step_4/diffusion_folder/experiment_final_checkpoint_150/contour'

# Ground truth folder
ground_truth_folder = 'ground_truth'

# Output folder for comparison images
comparison_output_folder = 'output/mar_8_horizontal_step_4/diffusion_folder/experiment_final_checkpoint_150/comparisons'
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
        # Load the condition (left half)
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
        
        # Load the ground truth to get original dimensions
        gt_path = os.path.join(ground_truth_folder, f'batch{batch_num}.npy')
        img_ground_truth = np.load(gt_path)
        
        # Handle different shapes for ground truth
        if img_ground_truth.ndim == 5:
            img_ground_truth = img_ground_truth[0, 0, 0]
        elif img_ground_truth.ndim == 4:
            img_ground_truth = img_ground_truth[0, 0]
        elif img_ground_truth.ndim == 3:
            img_ground_truth = img_ground_truth[0]
        
        # Every-4th-row split: 0::4 = condition (known), rest = target (to predict)
        H_gt = img_ground_truth.shape[0]
        cond_indices = [i for i in range(H_gt) if i % 4 == 0]
        label_indices = [i for i in range(H_gt) if i % 4 != 0]
        gt_cond   = img_ground_truth[cond_indices, :]
        gt_target = img_ground_truth[label_indices, :]

        # Crop height padding from generated outputs to match GT dimensions
        img_generated = img_generated[:gt_target.shape[0], :]
        img_cond      = img_cond[:gt_cond.shape[0], :]
        
        print(f"{batch_name}: generated={img_generated.shape}, cond={img_cond.shape}, gt_target={gt_target.shape}")
        
        # Normalize both images to [0, 1] for SSIM calculation
        def normalize(img):
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                return (img - img_min) / (img_max - img_min)
            return img
        
        img_gen_norm   = normalize(img_generated)
        gt_target_norm = normalize(gt_target)
        gt_cond_norm   = normalize(gt_cond)
        img_cond_norm  = normalize(img_cond)
        
        # Calculate SSIM between generated odd rows and GT odd rows
        ssim_value = ssim(gt_target_norm, img_gen_norm, data_range=1.0)
        
        # Calculate PSNR between generated odd rows and GT odd rows
        psnr_value = psnr(gt_target_norm, img_gen_norm, data_range=1.0)
        
        # Reconstruct full image by placing condition and label rows back
        full_H, full_W = img_ground_truth.shape
        recon_full = np.zeros((full_H, full_W))
        for idx, row in zip(cond_indices, range(len(cond_indices))):
            recon_full[idx, :] = img_cond_norm[row]
        for idx, row in zip(label_indices, range(len(label_indices))):
            recon_full[idx, :] = img_gen_norm[row]
        gt_full = normalize(img_ground_truth)
        
        # Create 2x2 figure: top row = full images, bottom row = condition / generated separately
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Top-left: Ground truth full image
        axes[0, 0].imshow(gt_full, cmap='gray')
        axes[0, 0].set_title('Ground Truth')
        axes[0, 0].axis('off')

        # Top-right: Reconstructed full image (condition + generated interleaved)
        axes[0, 1].imshow(recon_full, cmap='gray')
        axes[0, 1].set_title('Condition + Generated (1/4 split)')
        axes[0, 1].axis('off')

        # Bottom-left: Condition only (even rows)
        axes[1, 0].imshow(img_cond_norm, cmap='gray')
        axes[1, 0].set_title('Condition (Every 4th Row)')
        axes[1, 0].axis('off')

        # Bottom-right: Generated only (odd rows)
        axes[1, 1].imshow(img_gen_norm, cmap='gray')
        axes[1, 1].set_title('Generated (3/4 Rows)')
        axes[1, 1].axis('off')

        # Set SSIM and PSNR as the main title
        fig.suptitle(f'SSIM: {ssim_value:.4f} | PSNR: {psnr_value:.2f} dB', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the comparison image
        output_path = os.path.join(comparison_output_folder, f'{batch_name}_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Store metrics
        metrics_list.append({
            'batch': batch_name,
            'ssim': ssim_value,
            'psnr': psnr_value
        })
        
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