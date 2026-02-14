import numpy as np
from matplotlib import pyplot as plt
import os

# Load the 2D array from the .npy file
output_folder = 'output/feb_12_512_model/diffusion_folder/experiment_final/contour/batch0'
img_array = np.load(os.path.join(output_folder, 'recon_micro_0.npy'))
img_array_cond = np.load(os.path.join(output_folder, 'recon_micro_0gt.npy'))
# Take the last 2 dimensions (512, 512) from shape (1, 1, 1, 512, 512)
img_array = img_array[0, 0, 0]
img_array_cond = img_array_cond[0,0,0]

img_array = np.concatenate([img_array_cond,img_array], axis=1)

# Save the array as a grayscale image
plt.imshow(img_array, cmap='gray')
output_path = os.path.join(output_folder, 'visualization.png')
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"Image saved to: {output_path}")