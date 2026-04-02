from pathlib import Path
import SimpleITK as sitk
import numpy as np
import imageio.v3 as iio
import astra
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os



def visualize(ct_slice: np.ndarray):
    
    ct_slice = ct_slice.astype(np.float32, copy=False)
    lo = np.percentile(ct_slice, 2)
    hi = np.percentile(ct_slice, 97)
    #lo, hi = [-3024.0, 1000.0]
    gt_clipped = np.clip(ct_slice, lo, hi).astype(np.float32, copy=False)
    gt_clipped = 0.2 * (1.0 + gt_clipped / 1000.0)
    vmin, vmax = float(np.min(gt_clipped)), float(np.max(gt_clipped))
    if vmax > vmin:
        gt_clipped =  (gt_clipped - vmin)  / (vmax - vmin)
        v_u8 = (gt_clipped * 255.0 + 0.5).astype(np.uint8)
        iio.imwrite("ct.png", v_u8)
    # print("done")
    

#Need to work on this to convert to different configs later
# need to do the sampling using astra
def compute_fbp_reprojection(sino_normalized, cond_indices,
                              det_count=816, num_angles=720,
                              DSO=1000, ODD=600):
    """
    Take a normalized sinogram, run FBP on known rows, then forward-project
    back to all angles to get a coarse full-sinogram estimate.

    sino_normalized: [num_angles, det_count] sinogram in [-1, 1]
    cond_indices: list of known row indices
    Returns: [num_angles, det_count] re-projected sinogram in [-1, 1]
    """
    # Extract known rows only
    sparse_sino = sino_normalized[cond_indices, :].astype(np.float32)

    # Full and sparse angle arrays
    angles_full = np.linspace(0, 2 * np.pi, num_angles, endpoint=False).astype(np.float32)
    angles_sparse = angles_full[cond_indices]

    # Reconstruction grid
    N_pix = 512
    vol_geom = astra.create_vol_geom(N_pix, N_pix)

    # --- FBP with sparse angles ---
    proj_geom_sparse = astra.create_proj_geom(
        'fanflat', 1.0, det_count, angles_sparse, DSO, ODD
    )
    rec_id = astra.data2d.create('-vol', vol_geom)
    sino_id = astra.data2d.create('-sino', proj_geom_sparse, sparse_sino)

    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    fbp_image = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)

    # --- Forward-project FBP image to full sinogram (all angles) ---
    proj_geom_full = astra.create_proj_geom(
        'fanflat', 1.0, det_count, angles_full, DSO, ODD
    )
    vol_id = astra.data2d.create('-vol', vol_geom, fbp_image)
    proj_id = astra.create_projector('line_fanflat', proj_geom_full, vol_geom)
    sino_id2, reproj_sino = astra.create_sino(vol_id, proj_id)

    # Normalize to [-1, 1]
    rmin, rmax = reproj_sino.min(), reproj_sino.max()
    if rmax > rmin:
        reproj_sino = 2.0 * (reproj_sino - rmin) / (rmax - rmin) - 1.0

    # Cleanup
    astra.data2d.delete(sino_id2)
    astra.data2d.delete(vol_id)
    astra.projector.delete(proj_id)

    return reproj_sino.astype(np.float32)


def convert_hu_to_mu(ct_slice):
    """Convert HU to linear attenuation coefficient (mu).
    mu_water ~ 0.02 mm^-1 at typical CT energies.
    """
    ct_slice = ct_slice.astype(np.float32, copy=False)
    # HU to mu: mu = mu_water * (1 + HU/1000)
    mu = 0.02 * (1.0 + ct_slice / 1000.0)
    mu = np.clip(mu, 0, None)  # mu cannot be negative
    return mu

	#Need to convert to sinogram and visualize
def convert_sinogram(ct_slice, dx, dy, dz, detector_count, angle_step):
    
    #det count should be width of a pixel
    #dx_mm which is det_spacing is dx_mm
    
    # Convert HU to attenuation coefficients first
    mu_slice = convert_hu_to_mu(ct_slice)
    
    H, W = list(mu_slice.shape)[:2]
    
    DSO = 1000
    ODD = 600  
    angles_deg = np.arange(0, 360, angle_step, dtype=np.float32)
    angles = np.deg2rad(angles_deg)  # ASTRA expects radians
    
    # generate params for the second part
    vol_geom = astra.create_vol_geom( H, W,
        -W * dx / 2.0,  W* dx/ 2.0,   # x_min, x_max
        -H * dy / 2.0,  H * dy / 2.0    # y_min, y_max
    )
        
    # Detector should cover the full object diagonal
    det_count = detector_count
    det_spacing = dx  
    
    proj_geom = astra.create_proj_geom('fanflat', det_spacing, det_count, angles, DSO, ODD)
    
    projector_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)

    slice2d = np.ascontiguousarray(mu_slice, dtype=np.float32)  # (H, W)
    sid = astra.data2d.create('-vol', vol_geom, slice2d)
    sino_id, sino = astra.create_sino(sid, projector_id) 
    
    # Cleanup ASTRA resources
    astra.data2d.delete(sino_id)
    astra.data2d.delete(sid)
    astra.projector.delete(projector_id)
    # print(sino.shape)
    return sino  # Return the sinogram
        #get the volumes and the spacings

def load_series_from(path):
    # 	Extended: If path is a directory, read all series in it
    if path.endswith(".dcm"):
        img = sitk.ReadImage(path)
        vol_zyx = sitk.GetArrayFromImage(img)
        spacing_x, spacing_y, spacing_z = img.GetSpacing()
        return [vol_zyx, (spacing_x, spacing_y, spacing_z)]
    else:
        series_dir = Path(path)
        sitk_reader = sitk.ImageSeriesReader()
        # Get all series UIDs in the directory
        series_IDs = sitk_reader.GetGDCMSeriesIDs(str(series_dir))
        if not series_IDs:
            raise ValueError(f"No DICOM series found in directory: {series_dir}")
        series_dict = []
        for series_uid in series_IDs:
            print(series_uid)
            file_names = sitk_reader.GetGDCMSeriesFileNames(str(series_dir), series_uid)
            sitk_reader.SetFileNames(file_names)
            img = sitk_reader.Execute()
            vol_zyx = sitk.GetArrayFromImage(img)
            spacing_x, spacing_y, spacing_z = img.GetSpacing()
            series_dict.append((vol_zyx, (spacing_x, spacing_y, spacing_z)))
        return series_dict

#correct so far 
#vol_zyx , spacing = load_series_from("data/Dataset")
#sinograms = [convert_sinogram(slice, spacing[0], spacing[1], spacing[2]) for slice in vol_zyx]

class dicom_dataset(Dataset):

    def __init__(self, data_path="data/Dataset",
                 detector_count=816,
                 angle_step=(360/720),
                 cache_dir="data/sino_cache",
                 cond_indices=None):

        # Sanitize data_path for use in directory name
        def sanitize_path(path):
            return path.strip().replace("/", "_").replace("\\", "_")

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.index_map = []  # list of cache file paths
        self.cond_indices = cond_indices

        num_angles = int(360 / angle_step)

        total_series = load_series_from(data_path)


        for s_idx, series in enumerate(total_series):
            vol_zyx, spacing = series
            for ind in range(len(vol_zyx)):
                if data_path == "data/Dataset":
                    cache_filename = f"sino_s{s_idx}_i{ind}_d{detector_count}_a{angle_step:.4f}.npy"
                else:
                    cache_filename = f"sino_{sanitize_path(data_path)}_s{s_idx}_i{ind}_d{detector_count}_a{angle_step:.4f}.npy"
                cache_path = os.path.join(cache_dir, cache_filename)

                # Only compute if not already cached
                if not os.path.exists(cache_path):
                    sino = convert_sinogram(
                        vol_zyx[ind],
                        spacing[0], spacing[1], spacing[2],
                        detector_count, angle_step
                    )
                    sino_min, sino_max = sino.min(), sino.max()
                    if sino_max > sino_min:
                        sino = 2.0 * (sino - sino_min) / (sino_max - sino_min) - 1.0
                    np.save(cache_path, sino.astype(np.float32))
                    print(f"Cached {cache_path}")

                self.index_map.append(cache_path)

        # Cache FBP re-projections if cond_indices provided
        # Encode config and data_path into filename so stale caches aren't reused
        self.fbp_map = []
        if cond_indices is not None:
            n_cond = len(cond_indices)
            max_ang = max(cond_indices)
            step = cond_indices[1] - cond_indices[0] if len(cond_indices) > 1 else 0
            if data_path == "data/Dataset":
                fbp_tag = f"fbp_n{n_cond}_step{step}_max{max_ang}"
            else:
                fbp_tag = f"fbp_{sanitize_path(data_path)}_n{n_cond}_step{step}_max{max_ang}"
            fbp_cache_dir = os.path.join("data", f"{fbp_tag}_cache")
            os.makedirs(fbp_cache_dir, exist_ok=True)
            print(f"Caching FBP re-projections ({fbp_tag}) in {fbp_cache_dir} ...")
            for sino_path in tqdm(self.index_map, desc="FBP re-projection"):
                base_name = os.path.basename(sino_path)
                if data_path == "data/Dataset":
                    fbp_base_name = base_name.replace("sino_", f"{fbp_tag}_")
                else:
                    fbp_base_name = base_name.replace("sino_", f"{fbp_tag}_")
                fbp_path = os.path.join(fbp_cache_dir, fbp_base_name)
                if not os.path.exists(fbp_path):
                    sino = np.load(sino_path)
                    fbp = compute_fbp_reprojection(
                        sino, cond_indices,
                        det_count=detector_count,
                        num_angles=num_angles
                    )
                    np.save(fbp_path, fbp)
                self.fbp_map.append(fbp_path)
            print(f"FBP re-projections ready: {len(self.fbp_map)} cached")

        print(f"Dataset ready: {len(self.index_map)} sinograms")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        # Load single sinogram on demand — fast numpy load
        sino = np.load(self.index_map[index])                          # [H, W]
        sino = torch.from_numpy(sino).float().unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

        if self.fbp_map:
            fbp = np.load(self.fbp_map[index])                          # [H, W]
            fbp = torch.from_numpy(fbp).float().unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
            return sino, fbp
        return sino
    


class Efficient_LIDC_DicomDataset(Dataset):
    def __init__(self, root_dir="data/LIDC-IDRI", Interpolate=False):
        """
        index_map = [
            (patient_id, study_id, series_folder, dicom_path),
            ...
        ]
        """
        self.root_dir = root_dir
        self.index_map = []
        self.Interpolate = Interpolate

        for patient in sorted(os.listdir(root_dir)):
            patient_path = os.path.join(root_dir, patient)
            if not os.path.isdir(patient_path):
                continue

            for study in sorted(os.listdir(patient_path)):
                study_path = os.path.join(patient_path, study)
                if not os.path.isdir(study_path):
                    continue

                # Series folder name is arbitrary
                for series in sorted(os.listdir(study_path)):
                    series_path = os.path.join(study_path, series)
                    if not os.path.isdir(series_path):
                        continue

                    dicom_files = [
                        f for f in os.listdir(series_path)
                        if f.lower().endswith(".dcm")
                    ]

                    for dcm in sorted(dicom_files):
                        self.index_map.append((
                            patient,
                            study,
                            series,
                            os.path.join(series_path, dcm)
                        ))

    def __len__(self):
        return len(self.index_map)

    # def __getitem__(self, index):
    #     patient_id, study_id, series_id, dicom_path = self.index_map[index]


    #     #print(dicom_path)
    #    # exit(0)
    #     # ---- Load single DICOM slice ----
    #     vol_zyx, spacing = load_series_from(dicom_path)

    #     # If load_series_from returns a single slice wrapped as volume

    #     sino = convert_sinogram(vol_zyx[0], spacing[0], spacing[1], spacing[2])
        
    #     # Normalize to [-1, 1]
    #     sino_min, sino_max = sino.min(), sino.max()
    #     if sino_max > sino_min:
    #         sino = 2.0 * (sino - sino_min) / (sino_max - sino_min) - 1.0
                    
    #     # NumPy -> Torch
    #     # sino = torch.from_numpy(sino).float()
        
    #     # sino = sino.unsqueeze(0).unsqueeze(0)
        
        
    #     sino = torch.from_numpy(sino).float().unsqueeze(0).unsqueeze(0)
    #     return sino

        # Resize
        # sino = F.interpolate(   
        #     sino,
        #     size=(1024, 1024),
        #     mode="nearest"   # correct for sinograms
        # )


if __name__ == '__main__':
	# Create ground truth folder if it doesn't exist
	ground_truth_dir = "output/feb_19_720_820_model/ground_truth"
	os.makedirs(ground_truth_dir, exist_ok=True)
	
	# Use dicom_dataset to load data
	dset = dicom_dataset(Interpolate=True, detector_count= 816, angle_step=(360/720), data_path="data/test_data")
	
	print(f"Total samples in dataset: {len(dset)}")
	
	# Save first 20 sinograms as batch0.npy through batch19.npy
	print("Saving first 20 sinograms as batch files...")
	for i in range(min(100, len(dset))):
		# Get sinogram and squeeze to [H, W]
		sino_np = dset.sinograms_torch[i].squeeze().numpy()
		
		# Save as batchX.npy
		filename = f"batch{i}.npy"
		filepath = os.path.join(ground_truth_dir, filename)
		np.save(filepath, sino_np)
		print(f"Saved {filename} with shape {sino_np.shape}")
	
	print(f"Done! Saved 20 batch files to {ground_truth_dir}")




