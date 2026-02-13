import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import imageio.v3 as iio
import astra
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pdb
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, cast



def _apply_fov_mask_circle(arr_3d: np.ndarray, radius_factor: float = 0.99) -> np.ndarray:
    
    H, W = [512, 512]
    yy, xx = np.ogrid[-H//2:H//2, -W//2:W//2]
    r2 = (min(H, W) * 0.5 * float(radius_factor)) ** 2
    mask = ((xx.astype(np.float32) ** 2 + yy.astype(np.float32) ** 2) <= r2).astype(np.float32)
    return arr_3d * mask[None, ...]

def apply_post(
    recon: np.ndarray,
    mask_circle_enabled=True,
    mask_radius_factor=0.99,
    normalization_mode="percentile",
    percentiles=(2, 98),
    scale_factor=1.0,
    clip_range=None,
    target_dtype=np.float32,
    fixed_range=(0, 1)
) -> np.ndarray:
    if mask_circle_enabled:
        recon = _apply_fov_mask_circle(recon, radius_factor=mask_radius_factor)

    mode = normalization_mode

    if mode == "minmax_per_slice":
        for s in range(recon.shape[0]):
            v = recon[s]
            vmin, vmax = float(np.min(v)), float(np.max(v))
            if vmax > vmin:
                recon[s] = (v - vmin) / (vmax - vmin)
            else:
                recon[s] = 0.0

    elif mode == "percentile":
        p_lo, p_hi = percentiles
        v = recon
        lo = np.percentile(v, p_lo)
        hi = np.percentile(v, p_hi)
        if hi > lo:
            recon = np.clip((v - lo) / (hi - lo), 0.0, 1.0)

    elif mode == "minmax":
        gmin = float(np.min(recon))
        gmax = float(np.max(recon))
        if gmax > gmin:
            recon = (recon - gmin) / (gmax - gmin)
        else:
            recon = np.zeros_like(recon, dtype=np.float32)

    elif mode == "fixed":
        lo, hi = fixed_range
        if hi <= lo:
            raise ValueError(f"[global_fixed] invalid fixed_range: {lo}, {hi}")
        recon = np.clip((recon - lo) / (hi - lo), 0.0, 1.0)

    elif mode in (None, "none"):
        pass

    else:
        raise ValueError(f"Unsupported normalize mode: {mode}")

    if mask_circle_enabled:
        recon = _apply_fov_mask_circle(recon, radius_factor=mask_radius_factor)

    if scale_factor != 1.0:
        recon = recon * scale_factor

    if clip_range is not None:
        lo, hi = float(clip_range[0]), float(clip_range[1])
        recon = np.clip(recon, lo, hi)

    return recon


# Load the 2D array from the .npy file
#img_array  = img_array_cond


def projection(img_array):
    sino_AD = np.asarray(img_array, dtype=np.float32)

    # Sinogram dimensions: (num_angles, num_detectors)
    num_angles = sino_AD.shape[0]
    num_detectors = sino_AD.shape[1]

    DSO = 1000  
    ODD = 600  

    angles_deg = np.arange(0, 360, 0.703125, dtype=np.float32)
    angles = np.deg2rad(angles_deg)  # ASTRA expects radians

    spacing_xyz = (0.664062, 0.664062, 2.5000000984848483)

    # Reconstruction volume size (typically same as detector count for square output)
    H = num_detectors
    W = num_detectors

    dx = spacing_xyz[0]
    dy = spacing_xyz[1]

    # generate params for the second part
    vol_geom = astra.create_vol_geom(H, W,
        -W * dx / 2.0,  W * dx / 2.0,   # x_min, x_max
        -H * dy / 2.0,  H * dy / 2.0    # y_min, y_max
    )
        
    # Detector count must match sinogram width
    det_count = num_detectors
    det_spacing = dx  

   # proj_geom = astra.create_proj_geom('parallel', det_spacing, det_count, angles)
    proj_geom = astra.create_proj_geom('fanflat', det_spacing, det_count, angles, DSO, ODD)

    projector_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)

    sinogram_id = astra.data2d.create('-sino', proj_geom, sino_AD)
    recon_id = astra.data2d.create('-vol', vol_geom)

    cfg: Dict[str, Any] = astra.astra_dict("FBP_CUDA")
    cfg["ProjectionDataId"] = sinogram_id
    cfg["ReconstructionDataId"] = recon_id
    cfg["ProjectorId"] = projector_id

    alg_id = astra.algorithm.create(cfg)
    try:
        astra.algorithm.run(alg_id)
        result = astra.data2d.get(recon_id).astype(np.float32, copy=False)
    except:
        print("Not done")
    finally:
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(recon_id)

    arr = result
    arr_to_write = apply_post(arr)
    arr_to_write_uint8 = (arr_to_write * 255).astype(np.uint8)  # Scale and convert to uint8
    iio.imwrite("output.png", arr_to_write_uint8)
    #iio.imwrite(f"output.png", arr_to_write)

    print(f"Saved output.png")

# Load the 2D array from the .npy file
img_array = np.load('output/two_day_training_low/diffusion_folder/experiment_final/contour/batch19/recon_micro_0.npy')
img_array_cond = np.load('output/two_day_training_low/diffusion_folder/experiment_final/contour/batch19/recon_micro_0gt.npy')
# Take the last 2 dimensions (512, 512) from shape (1, 1, 1, 512, 512)
img_array = img_array[0, 0, 0]
img_array_cond = img_array_cond[0,0,0]

img_array = np.concatenate([img_array_cond,img_array], axis=1)


projection(img_array)



