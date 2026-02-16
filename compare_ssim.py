import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os

# Base folder containing all batch folders (generated outputs)
base_folder = 'output/feb_12_512_model/diffusion_folder/experiment_final/contour'

# Ground truth folder
ground_truth_folder = 'output/feb_12_512_model/ground_truth'

# Output folder for comparison images
comparison_output_folder = 'output/feb_12_512_model/diffusion_folder/comparisons'
os.makedirs(comparison_output_folder, exist_ok=True)

# Get all batch folders and sort them
batch_folders = [f for f in os.listdir(base_folder) if f.startswith('batch') and os.path.isdir(os.path.join(base_folder, f))]
batch_folders.sort(key=lambda x: int(x.replace('batch', '')))

print(f"Found {len(batch_folders)} batch folders")

for batch_name in batch_folders:
    output_folder = os.path.join(base_folder, batch_name)
    batch_num = batch_name.replace('batch', '')
    
    try:
        # Load the generated reconstruction
        img_generated = np.load(os.path.join(output_folder, 'recon_micro_0.npy'))
        # Load the condition (left half)
        img_cond = np.load(os.path.join(output_folder, 'recon_micro_0gt.npy'))
        
        # Take the last 2 dimensions (512, 512) from shape (1, 1, 1, 512, 512)
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
        
        # Load the ground truth
        gt_path = os.path.join(ground_truth_folder, f'batch{batch_num}.npy')
        img_ground_truth = np.load(gt_path)
        
        # Handle different shapes for ground truth
        if img_ground_truth.ndim == 5:
            img_ground_truth = img_ground_truth[0, 0, 0]
        elif img_ground_truth.ndim == 4:
            img_ground_truth = img_ground_truth[0, 0]
        elif img_ground_truth.ndim == 3:
            img_ground_truth = img_ground_truth[0]
        
        # Split ground truth in half - left side is condition, right side is what we compare
        width = img_ground_truth.shape[1]
        gt_left = img_ground_truth[:, :width//2]  # Condition part
        gt_right = img_ground_truth[:, width//2:]  # Part to compare with generated
        
        # Normalize both images to [0, 1] for SSIM calculation
        def normalize(img):
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                return (img - img_min) / (img_max - img_min)
            return img
        
        img_gen_norm = normalize(img_generated)
        gt_right_norm = normalize(gt_right)
        gt_left_norm = normalize(gt_left)
        img_cond_norm = normalize(img_cond)
        
        # Calculate SSIM between generated and right side of ground truth
        ssim_value = ssim(gt_right_norm, img_gen_norm, data_range=1.0)
        
        # Concatenate left side (condition) with generated and with ground truth right
        combined_generated = np.concatenate([img_cond_norm, img_gen_norm], axis=1)
        combined_gt = np.concatenate([gt_left_norm, gt_right_norm], axis=1)
        
        # Create side-by-side comparison figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Ground truth (left + right) on the left
        axes[0].imshow(combined_gt, cmap='gray')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        # Condition + Generated on the right
        axes[1].imshow(combined_generated, cmap='gray')
        axes[1].set_title('Condition + Generated')
        axes[1].axis('off')
        
        # Set SSIM as the main title
        fig.suptitle(f'SSIM: {ssim_value:.4f}', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the comparison image
        output_path = os.path.join(comparison_output_folder, f'{batch_name}_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"{batch_name}: SSIM = {ssim_value:.4f} -> Saved to {output_path}")
        
    except FileNotFoundError as e:
        print(f"Error processing {batch_name}: File not found - {e}")
    except Exception as e:
        print(f"Error processing {batch_name}: {e}")

print(f"\nDone! Comparison images saved to {comparison_output_folder}")
