import numpy as np
import astra
import SimpleITK as sitk
import os
import itertools
def generate_sinogram_cuda(mu_slice, spacing, angles_deg, DSO=1000, ODD=600, det_count=1500):
    H, W = mu_slice.shape
    dx, dy = spacing[0], spacing[1]

    # 1. Setup Geometries
    vol_geom = astra.create_vol_geom(H, W, -W*dx/2, W*dx/2, -H*dy/2, H*dy/2)
    angles = np.deg2rad(angles_deg)
    proj_geom = astra.create_proj_geom('fanflat', dx, det_count, angles, DSO, ODD)

    # 2. Create Data on GPU
    # Note: FP_CUDA (Forward Projection) is more memory efficient than 'line_fanflat' projector objects
    vol_id = astra.data2d.create('-vol', vol_geom, np.ascontiguousarray(mu_slice, dtype=np.float32))
    sino_id = astra.data2d.create('-sino', proj_geom)

    # 3. Forward Project
    cfg = astra.astra_dict('FP_CUDA')
    cfg['VolumeDataId'] = vol_id
    cfg['ProjectionDataId'] = sino_id
    
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    sino = astra.data2d.get(sino_id)

    # 4. Cleanup
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(vol_id)
    astra.data2d.delete(sino_id)
    
    return sino

def main():
    # --- CONFIGURATION AREA ---
    input_path = "data/extra_data/1-133.dcm"
    output_dir = "./sinograms_output"
    os.makedirs(output_dir, exist_ok=True)

    # Define your different angle/range configs here
    # Define your ranges and steps
    ranges = [360, 270, 180, 90]
    steps = [0.25, 0.5, 1.0, 2.0]

    configs = []

# Generate all combinations
    for r, s in itertools.product(ranges, steps):
        configs.append({
            "name": f"range{r}_step{str(s).replace('.', 'p')}",
            "angles": np.arange(0, r, s, dtype=np.float32)
        })
    # --- LOADING ---
    img = sitk.ReadImage(input_path)
    vol_zyx = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing() # (dx, dy, dz)
    
    # Process specific slices (e.g., first 5 slices)
    for slice_idx in range(min(5, len(vol_zyx))):
        ct_slice = vol_zyx[slice_idx]
        
        # Convert HU to Mu (Essential for realistic forward projection)
        # Simplified linear transform; adjust based on your convert_hu_to_mu logic
        mu_slice = (ct_slice + 1024) / 1000.0 
        mu_slice[mu_slice < 0] = 0

        for conf in configs:
            print(f"Processing Slice {slice_idx} | Config: {conf['name']}")
            
            sino = generate_sinogram_cuda(
                mu_slice, 
                spacing, 
                conf['angles'],
                det_count=1500
            )

            # --- SAVE ---
            filename = f"sino_slice{slice_idx}_{conf['name']}.npy"
            np.save(os.path.join(output_dir, filename), sino)

    astra.clear()
    print("Done. All sinograms saved to:", output_dir)

if __name__ == "__main__":
    main()
